"""Status over time for props (docs/REVIEW_INSTRUCTIONS.md F-05, F-06, WP-3 step 2). Pure Python.

fit_props.py keeps, per frame or per segment, how a prop's pose was found. Here that becomes:
    labels    per frame (status, basis, evidence kind) or None where the prop has no key
    spans     consecutive frames with the same label as StatusSpan (seconds), in time order, non-overlapping
    evidence  one Evidence entry per span (the per-frame notes go to <object>_frames.json, not into the project)

Mapping
    stick (racket) notes per frame: "sharp" -> observed (outline seen); "swept" -> estimated (pose from the blur
        streak); anything that says "hypothesis" or "interpolated" -> hypothesis (rule: carried between seen frames)
    sphere (ball) segment notes: "held" -> observed (mask at the hand); "ballistic"/"line" -> estimated (path fitted
        to masks); notes with hypothesis=True -> hypothesis, evidence kind "physics" when the model says physics,
        else "rule". Where segments overlap, observed beats estimated beats hypothesis (the fill-in notes span the
        whole unseen envelope, the held notes inside it are what was seen).
"""

from __future__ import annotations

from vrecon.schema import Evidence, StatusSpan

RANK = {"observed": 0, "estimated": 1, "hypothesis": 2}


def stick_labels(notes: list[str]) -> list[tuple[str, str, str] | None]:
    out = []
    for n in notes:
        low = (n or "").lower()
        if low == "sharp":
            out.append(("observed", "outline seen (silhouette fit)", "observation"))
        elif low == "swept":
            out.append(("estimated", "pose from the blur streak (swept silhouette)", "observation"))
        elif "hypothesis" in low or "interpolat" in low:
            out.append(("hypothesis", "turned between seen frames (interpolated)", "rule"))
        else:
            out.append(("estimated", n, "observation"))
    return out


def _sphere_label(note: dict) -> tuple[str, str, str]:
    model = str(note.get("model", ""))
    if note.get("hypothesis"):
        kind = "physics" if "physics" in model or "strike" in model else "rule"
        return "hypothesis", model, kind
    if model.startswith("held"):
        return "observed", "held (mask at the hand)", "observation"
    return "estimated", f"{model} path fitted to the masks", "observation"


def sphere_labels(notes: list[dict], t_n: int, has_key: list[bool]) -> list[tuple[str, str, str] | None]:
    out: list[tuple[str, str, str] | None] = [None] * t_n
    for n in notes:
        lab = _sphere_label(n)
        a, b = n["frames"]
        for f in range(max(a, 0), min(b, t_n - 1) + 1):
            if out[f] is None or RANK[lab[0]] < RANK[out[f][0]]:
                out[f] = lab
    return [lab if has_key[f] else None for f, lab in enumerate(out)]


def spans_from_labels(labels: list, fps: float) -> list[tuple[int, int, tuple[str, str, str]]]:
    """[(first frame, last frame, label)] for runs of equal labels."""
    runs, start = [], None
    for f in range(len(labels) + 1):
        cur = labels[f] if f < len(labels) else None
        if start is not None and (cur != labels[start]):
            runs.append((start, f - 1, labels[start]))
            start = None
        if cur is not None and start is None:
            start = f
    return runs


def to_schema(labels: list, fps: float) -> tuple[list[StatusSpan], list[Evidence]]:
    spans, ev = [], []
    for a, b, (status, basis, kind) in spans_from_labels(labels, fps):
        spans.append(StatusSpan(start=a / fps, end=b / fps, status=status, basis=basis))
        ev.append(Evidence(kind=kind, source_frames=[a, b], note=f"frames {a}-{b}: {status}, {basis}"))
    return spans, ev


def overall_status(labels: list) -> str:
    """The object's own status: the most common label among its keyed frames."""
    counts: dict[str, int] = {}
    for lab in labels:
        if lab is not None:
            counts[lab[0]] = counts.get(lab[0], 0) + 1
    return max(counts, key=counts.get) if counts else "estimated"


__all__ = ["overall_status", "sphere_labels", "spans_from_labels", "stick_labels", "to_schema"]
