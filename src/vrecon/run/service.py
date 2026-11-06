"""One place for the repair-loop operations; the CLI (`vrecon run ...`) and the MCP server call these.

A run directory holds config.json (fixed at creation: project, mask source, accept rule, budget),
history.jsonl and snapshots/. Every call reloads state from disk, so any client (human, Claude Code,
Codex, ...) can continue a run.
"""

from __future__ import annotations

import json
from pathlib import Path

from vrecon.metrics import match_objects
from vrecon.optimize import (MaskObservation, Param, combined_loss, gt_stand_in_masks, optimize, render_silhouettes,
                             silhouette_metrics)
from vrecon.schema import load_project

from .manager import AcceptRule, Proposal, RunManager


def _config(run_dir: Path) -> dict:
    return json.loads((run_dir / "config.json").read_text(encoding="utf-8"))


def init_run(project: str, run_dir: str, masks: str, max_attempts: int = 30, min_gain: float = 0.0) -> dict:
    """Create a run. masks: a synthetic clip folder (ground-truth stand-in, until a segmenter is integrated)."""
    rd = Path(run_dir)
    if (rd / "config.json").exists():
        raise FileExistsError(f"{rd} already has a run")
    rd.mkdir(parents=True, exist_ok=True)
    video = Path(masks) / "video.mp4"
    cfg = {"project": str(Path(project).resolve()), "masks": str(Path(masks).resolve()), "mask_source": "gt_stand_in",
           "video": str(video.resolve()) if video.is_file() else None,
           "rule": {"primary": "silhouette_loss", "min_gain": min_gain}, "max_attempts": max_attempts}
    (rd / "config.json").write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    return cfg


def _observation(cfg: dict) -> MaskObservation:
    return gt_stand_in_masks(cfg["masks"])


def auto_pairs(project, obs: MaskObservation, shot_id: str = "shot_001") -> dict[str, str]:
    """Observed label -> reconstruction object: movers by silhouette overlap, set pieces by label."""
    movers = [o.id for o in project.reconstruction.objects if o.kind != "set"]
    t, h, w = obs.masks.shape
    frames = list(range(0, t, 3))
    sil = render_silhouettes(project, shot_id, frames, movers, (w, h))
    sets = {o.id for o in project.reconstruction.objects if o.kind == "set"}
    obs_map = {k: v for k, v in obs.id_map.items() if v not in sets}
    pairs = match_objects(obs.masks[frames], sil, obs_map, {i + 1: m for i, m in enumerate(movers)})
    # set pieces (floor, wall) pair by label when the observation provides them: they pin the camera
    pairs.update({sid: sid for sid in sets if sid in obs.id_map.values()})
    return pairs


def _tracks(cfg: dict):
    if not cfg.get("video"):
        return None
    from vrecon.optimize.tracks import track_points

    return track_points(cfg["video"])  # OpenCV LK on the reference video, about a second


def _manager(run_dir: Path):
    cfg = _config(run_dir)
    obs = _observation(cfg)
    tracks = _tracks(cfg)
    pairs_cache: dict = {}

    def evaluate(p):
        pairs = pairs_cache.setdefault("pairs", auto_pairs(p, obs))
        sets = sorted(o.id for o in p.reconstruction.objects if o.kind == "set")
        set_pairs = {k: v for k, v in pairs.items() if v in sets}
        return {"silhouette_loss": combined_loss(p, obs, pairs, tracks),
                "camera_loss": combined_loss(p, obs, set_pairs, tracks, sets or None, use_silhouette=bool(set_pairs))}

    rule = AcceptRule(**cfg["rule"])
    mgr = RunManager(cfg["project"], run_dir, evaluate, rule, max_attempts=cfg["max_attempts"])
    mgr.tracks = tracks
    return mgr, obs, pairs_cache


def state(run_dir: str) -> dict:
    """Adopted revision, metrics, attempts used, and what can be changed (targets, locks)."""
    mgr, _, pairs = _manager(Path(run_dir))
    p = mgr.adopted
    return {"revision": p.revision, "metrics": mgr.metrics, "attempts": mgr.attempts, "max_attempts": mgr.max_attempts,
            "pairs": pairs.get("pairs"),
            "objects": [{"id": o.id, "kind": o.kind, "size": o.size, "animated": bool(o.keys)}
                        for o in p.reconstruction.objects],
            "cameras": [{"id": c.id, "fx": c.intrinsics.fx, "keys": len(c.keys)} for c in p.reconstruction.cameras],
            "locks": [lk.model_dump(mode="json") for lk in p.human.locks]}


def diagnose(run_dir: str) -> dict:
    """Per-object silhouette loss and mean image offset (where the reconstruction sits vs the observation)."""
    mgr, obs, _ = _manager(Path(run_dir))
    pairs = auto_pairs(mgr.adopted, obs)
    m = silhouette_metrics(mgr.adopted, obs, pairs)
    m["pairs"] = pairs
    if mgr.tracks is not None:
        from vrecon.optimize.tracks import track_metrics

        sets = [o.id for o in mgr.adopted.reconstruction.objects if o.kind == "set"]
        m["tracks_all"] = track_metrics(mgr.adopted, mgr.tracks)
        m["tracks_on_set"] = track_metrics(mgr.adopted, mgr.tracks, ids=sets or None)
    m["hint"] = ("offset_x > 0: reconstruction is right of the observation; offset_y > 0: lower. "
                 "A shared offset/scale across all objects suggests the camera; one object alone suggests that object. "
                 "tracks_on_set high: suspect the camera (focal). Camera proposals are judged on camera evidence only.")
    return m


def run_optimizer(run_dir: str, params: list[str], hypothesis: str, max_evals: int = 120) -> dict:
    """params like 'focal:cam_001', 'translation:obj_01:0', 'size:obj_01:2', 'time:obj_01' (optional ':axis:step')."""
    mgr, obs, _ = _manager(Path(run_dir))
    parsed = []
    for s in params:
        parts = s.split(":")
        kind, target = parts[0], parts[1]
        axis = int(parts[2]) if len(parts) > 2 and kind not in ("focal", "time") else 0
        step = float(parts[3]) if len(parts) > 3 else {"focal": 0.1, "time": 4.0}.get(kind, 0.2)
        parsed.append(Param(kind, target, axis, step))
    pairs = auto_pairs(mgr.adopted, obs)
    pid = f"opt_{mgr.attempts + 1:03d}"
    camera_only = all(p.kind == "focal" for p in parsed)
    if camera_only:  # judge camera hypotheses on camera evidence only: set outlines + tracks on the set
        sets = sorted(o.id for o in mgr.adopted.reconstruction.objects if o.kind == "set")
        set_pairs = {k: v for k, v in pairs.items() if v in sets}
        prop, info = optimize(mgr.adopted, obs, set_pairs, parsed, hypothesis, pid, max_evals=max_evals,
                              tracks=mgr.tracks, track_ids=sets or None, use_silhouette=bool(set_pairs))
        out = mgr.try_proposal(prop, rule=AcceptRule(primary="camera_loss", min_gain=1e-4))
    else:
        prop, info = optimize(mgr.adopted, obs, pairs, parsed, hypothesis, pid, max_evals=max_evals, tracks=mgr.tracks)
        out = mgr.try_proposal(prop)
    return {"outcome": out.model_dump(mode="json"), "search": info}


def auto(run_dir: str, passes: int = 2, max_evals: int = 60) -> dict:
    """Fixed hypothesis order (camera first, then each object's position, size, timing); no LLM."""
    from .policy import auto_repair

    mgr, obs, _ = _manager(Path(run_dir))
    before = dict(mgr.metrics)
    records = auto_repair(mgr, obs, auto_pairs(mgr.adopted, obs), mgr.tracks, passes, max_evals)
    return {"revision": mgr.adopted.revision, "metrics_before": before, "metrics_after": mgr.metrics,
            "accepted": sum(r["status"] == "accepted" for r in records), "tried": len(records), "attempts": records}


def submit(run_dir: str, proposal: dict) -> dict:
    """An LLM-written proposal: {"id", "hypothesis", "changes": [{"target_id", "field", "value"}]}."""
    mgr, _, _ = _manager(Path(run_dir))
    return mgr.try_proposal(Proposal.model_validate(proposal)).model_dump(mode="json")


def history(run_dir: str, last: int = 20) -> list[dict]:
    path = Path(run_dir) / "history.jsonl"
    if not path.exists():
        return []
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()][-last:]
    return [{"proposal": r["proposal"]["id"], "hypothesis": r["proposal"]["hypothesis"],
             "source": r["proposal"]["source"], **{k: r["outcome"][k] for k in ("status", "reasons", "revision")},
             "after": (r["outcome"].get("metrics_after") or {}).get("silhouette_loss")} for r in rows]


def rollback(run_dir: str, revision: int) -> dict:
    mgr, _, _ = _manager(Path(run_dir))
    p = mgr.rollback(revision)
    return {"revision": p.revision, "metrics": mgr.metrics}


__all__ = ["auto", "auto_pairs", "diagnose", "history", "init_run", "rollback", "run_optimizer", "state", "submit"]
