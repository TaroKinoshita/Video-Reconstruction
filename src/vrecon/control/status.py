"""What each control pixel shows: observed, standard or hypothesis (docs/REVIEW_INSTRUCTIONS.md F-24, WP-3 steps 3-4).

Pure numpy (cv2 for blurs). The status pass is computed from the index pass the export already renders: each
pixel's front-most object, and that object's status at the frame's time (SceneObject.status_at, so spans count).

    EMPTY 0     no geometry
    OBSERVED 1  status observed, estimated or human_confirmed (comes from the video)
    STANDARD 2  status standard (regulation sizes and positions)
    HYPOTHESIS 3

Treatments of hypothesis pixels per control kind (a control video is a hard constraint; for a guessed region the
intent is "something like this is there", not "these exact lines"):
    full   unchanged
    soft   depth: kept, blurred (Gaussian, sigma SOFT_SIGMA px) inside the region; canny: edges removed;
           gray / clay / look / id: flattened to the region's mean colour per frame
    drop   depth: far plane (black); canny: removed; colour kinds: background (black)
"""

from __future__ import annotations

import numpy as np

from vrecon.schema import Project, frame_time

EMPTY, OBSERVED, STANDARD, HYPOTHESIS = 0, 1, 2, 3
NAMES = {EMPTY: "empty", OBSERVED: "observed", STANDARD: "standard", HYPOTHESIS: "hypothesis"}
CODE = {"observed": OBSERVED, "estimated": OBSERVED, "human_confirmed": OBSERVED, "standard": STANDARD,
        "hypothesis": HYPOTHESIS}
MODES = ("full", "soft", "drop")
SOFT_SIGMA = 6.0


def status_pass(index: np.ndarray, id_map: dict[int, str], project: Project, shot_id: str | None = None) -> np.ndarray:
    """uint8 [T, H, W] status codes from an index pass [T, H, W] and its label -> object id map."""
    objs = {o.id: o for o in project.reconstruction.objects}
    out = np.zeros(index.shape, np.uint8)
    for f in range(index.shape[0]):
        t = frame_time(project, f, shot_id)
        lut = np.zeros(int(index[f].max()) + 1 if index[f].size else 1, np.uint8)
        for lab, oid in id_map.items():
            if lab < len(lut) and oid in objs:
                lut[lab] = CODE.get(objs[oid].status_at(t), OBSERVED)
        out[f] = lut[index[f]]
    return out


def shares(status: np.ndarray) -> dict:
    """Share of pixels per status, per frame and over the clip."""
    n = status[0].size
    per = [{NAMES[c]: float((s == c).sum() / n) for c in NAMES} for s in status]
    whole = {NAMES[c]: float((status == c).mean()) for c in NAMES}
    return {"clip": whole, "frames": per}


def apply_mode(kind: str, frames: np.ndarray, status: np.ndarray, mode: str) -> np.ndarray:
    """frames uint8 RGB [T, H, W, 3] of one control kind; returns a treated copy (mode full: the input)."""
    if mode not in MODES:
        raise ValueError(f"unknown hypothesis mode {mode!r}")
    if mode == "full":
        return frames
    import cv2

    out = frames.copy()
    for f in range(len(frames)):
        hyp = status[f] == HYPOTHESIS
        if not hyp.any():
            continue
        if kind == "canny" or mode == "drop":
            out[f][hyp] = 0
        elif kind == "depth":
            blur = cv2.GaussianBlur(frames[f], (0, 0), SOFT_SIGMA)
            out[f][hyp] = blur[hyp]
        else:  # gray, clay, look, id: one flat colour for the whole guessed region
            out[f][hyp] = frames[f][hyp].mean(axis=0).round().astype(np.uint8)
    return out


__all__ = ["CODE", "EMPTY", "HYPOTHESIS", "MODES", "NAMES", "OBSERVED", "STANDARD", "apply_mode", "shares",
           "status_pass"]
