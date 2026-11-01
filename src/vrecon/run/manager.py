"""Run manager (PLAN.md 6.1, 6.4, 6.5): try candidates on a working copy, adopt only what passes.

    proposal -> constraints (locks, declared scope) -> working copy -> evaluate -> accept rule
             -> commit atomically (revision + 1, snapshot, history)  or  discard (state unchanged)

The LLM never edits the project; it submits proposals. Any candidate (including one produced by
arbitrary code) is diffed against the adopted state, and touching a locked or undeclared field
rejects it.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from vrecon.schema import Project, load_project, save_project
from vrecon.schema.common import Model, utcnow

Metrics = dict[str, float]


class Change(Model):
    target_id: str
    field: str  # dotted path inside the target, e.g. "rest.pose.translation", "intrinsics.fx", "keys"
    value: Any


class Proposal(Model):
    id: str
    hypothesis: str = ""  # e.g. "camera focal length too short"
    changes: list[Change] = Field(default_factory=list)
    source: Literal["llm", "optimizer", "human", "code"] = "llm"
    declared: list[tuple[str, str]] = Field(default_factory=list)  # (target_id, field) scope for "code"


class Outcome(Model):
    proposal_id: str
    status: Literal["accepted", "rejected", "error", "budget_exhausted"]
    reasons: list[str] = Field(default_factory=list)
    metrics_before: Metrics | None = None
    metrics_after: Metrics | None = None
    revision: int
    at: datetime = Field(default_factory=utcnow)


# --- targets and dotted paths ----------------------------------------------------------------------


def _targets(p: Project) -> dict[str, Model]:
    r = p.reconstruction
    return {x.id: x for x in [*r.objects, *r.cameras, *r.events, *r.overrides]}


def _set(obj: Any, dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], value)


def apply_changes(project: Project, changes: list[Change]) -> Project:
    """Working copy with changes applied and fully re-validated (raises on bad values or paths)."""
    work = project.model_copy(deep=True)
    targets = _targets(work)
    for c in changes:
        if c.target_id not in targets:
            raise KeyError(f"unknown target {c.target_id!r}")
        _set(targets[c.target_id], c.field, c.value)
    return Project.model_validate(work.model_dump())


def changed_fields(before: Project, after: Project) -> list[tuple[str, str]]:
    """(target_id, dotted field) pairs that differ, at the leaf level; also added/removed targets."""
    a = {k: v.model_dump(mode="json") for k, v in _targets(before).items()}
    b = {k: v.model_dump(mode="json") for k, v in _targets(after).items()}
    out: list[tuple[str, str]] = [(k, "*") for k in a.keys() ^ b.keys()]

    def walk(tid, x, y, path):
        if isinstance(x, dict) and isinstance(y, dict):
            for key in x.keys() | y.keys():
                walk(tid, x.get(key), y.get(key), f"{path}.{key}" if path else key)
        elif x != y:
            out.append((tid, path or "*"))

    for k in a.keys() & b.keys():
        walk(k, a[k], b[k], "")
    rest_a = before.model_dump(mode="json", exclude={"reconstruction", "revision", "human", "artifacts"})
    rest_b = after.model_dump(mode="json", exclude={"reconstruction", "revision", "human", "artifacts"})
    if rest_a != rest_b:
        out.append(("__project__", "*"))
    return sorted(out)


def _covers(scope: str, field: str) -> bool:
    return scope == "*" or field == scope or field.startswith(scope + ".")


def check_constraints(adopted: Project, candidate: Project, proposal: Proposal) -> list[str]:
    """Violations: locked fields, changes outside the declared scope, project-level edits."""
    scope = proposal.declared or [(c.target_id, c.field) for c in proposal.changes]
    problems = []
    for tid, field in changed_fields(adopted, candidate):
        if tid == "__project__":
            problems.append("changes outside the reconstruction (source, shots, ...) are not allowed")
            continue
        if adopted.is_locked(tid, field):
            problems.append(f"{tid}.{field} is locked")
        if not any(t == tid and _covers(f, field) for t, f in scope):
            problems.append(f"{tid}.{field} changed but was not declared")
    return problems


# --- accept rule (PLAN.md 6.4) -------------------------------------------------------------------


class AcceptRule(Model):
    """Lower is better for every metric. `primary` must improve by at least `min_gain`;
    each `guard` may not get worse by more than its tolerance."""

    primary: str
    min_gain: float = 0.0
    guards: dict[str, float] = Field(default_factory=dict)

    def judge(self, before: Metrics, after: Metrics) -> list[str]:
        reasons = []
        if self.primary not in after or after[self.primary] is None:
            return [f"metric {self.primary} missing"]
        if not after[self.primary] < before[self.primary] - self.min_gain:
            reasons.append(f"{self.primary} did not improve ({before[self.primary]:.6g} -> {after[self.primary]:.6g})")
        for name, tol in self.guards.items():
            if name in before and name in after and after[name] > before[name] + tol:
                reasons.append(f"{name} got worse ({before[name]:.6g} -> {after[name]:.6g}, tolerance {tol})")
        return reasons


# --- manager ---------------------------------------------------------------------------------------


class RunManager:
    """Owns the adopted project file, the budget and the history for one repair run."""

    def __init__(self, project_path: str | Path, run_dir: str | Path, evaluate: Callable[[Project], Metrics],
                 rule: AcceptRule, max_attempts: int = 20):
        self.project_path = Path(project_path)
        self.run_dir = Path(run_dir)
        (self.run_dir / "snapshots").mkdir(parents=True, exist_ok=True)
        self.evaluate = evaluate
        self.rule = rule
        self.max_attempts = max_attempts
        self.history_path = self.run_dir / "history.jsonl"
        self.attempts = sum(1 for _ in self.history_path.open()) if self.history_path.exists() else 0  # resumable
        self.adopted = load_project(self.project_path)
        self.metrics = evaluate(self.adopted)

    def _log(self, outcome: Outcome, proposal: Proposal) -> Outcome:
        rec = {"outcome": outcome.model_dump(mode="json"), "proposal": proposal.model_dump(mode="json")}
        with self.history_path.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        return outcome

    def try_proposal(self, proposal: Proposal, candidate: Project | None = None) -> Outcome:
        """Evaluate a proposal (or a candidate produced by code) and adopt it only if it passes."""
        rev = self.adopted.revision
        if self.attempts >= self.max_attempts:
            return Outcome(proposal_id=proposal.id, status="budget_exhausted", revision=rev,
                           reasons=[f"budget of {self.max_attempts} attempts used"])
        self.attempts += 1
        try:
            work = candidate if candidate is not None else apply_changes(self.adopted, proposal.changes)
            work = Project.model_validate(work.model_dump())
        except Exception as exc:
            return self._log(Outcome(proposal_id=proposal.id, status="error", revision=rev,
                                     reasons=[f"could not build candidate: {exc}"]), proposal)
        problems = check_constraints(self.adopted, work, proposal)
        if problems:
            return self._log(Outcome(proposal_id=proposal.id, status="rejected", revision=rev, reasons=problems),
                             proposal)
        try:
            after = self.evaluate(work)
        except Exception as exc:
            return self._log(Outcome(proposal_id=proposal.id, status="error", revision=rev,
                                     reasons=[f"evaluation failed: {exc}"]), proposal)
        reasons = self.rule.judge(self.metrics, after)
        if reasons:
            return self._log(Outcome(proposal_id=proposal.id, status="rejected", revision=rev, reasons=reasons,
                                     metrics_before=self.metrics, metrics_after=after), proposal)
        # commit: snapshot the old state, then replace the project file atomically
        save_project(self.adopted, self.run_dir / "snapshots" / f"rev_{rev:04d}.json")
        work.revision = rev + 1
        save_project(work, self.project_path)
        before, self.adopted, self.metrics = self.metrics, work, after
        return self._log(Outcome(proposal_id=proposal.id, status="accepted", revision=work.revision,
                                 metrics_before=before, metrics_after=after), proposal)

    def rollback(self, revision: int) -> Project:
        """Make a snapshot the adopted state again (as a new revision, history is kept)."""
        snap = load_project(self.run_dir / "snapshots" / f"rev_{revision:04d}.json")
        snap.revision = self.adopted.revision + 1
        save_project(self.adopted, self.run_dir / "snapshots" / f"rev_{self.adopted.revision:04d}.json")
        save_project(snap, self.project_path)
        self.adopted, self.metrics = snap, self.evaluate(snap)
        return snap
