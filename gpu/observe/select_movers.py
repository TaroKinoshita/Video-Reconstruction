"""Pick the movers prompt (word, frame) for SAM 3.1 without ground truth. numpy + cv2 only, so tests can import it.

A candidate run gives obj id -> bool [T, H, W]. A track counts when it is present in most frames, is not
huge (floor/wall leaking in), and is not a near-copy of a track already counted. More counted tracks win
(two touching objects merged into one track count once); ties go to steadier shapes over time.
"""

from __future__ import annotations

import cv2
import numpy as np

SPLIT_PIECE = 0.2  # a piece counts when it holds at least this share of the mask
MIN_PRESENT = 0.5  # fraction of frames with a non-empty mask
MAX_AREA = 0.25  # mean fraction of the image while present
DUPLICATE_IOU = 0.5


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def _steadiness(m: np.ndarray) -> float:
    """Mean IoU between consecutive frames where both are non-empty."""
    vals = [_iou(m[t], m[t + 1]) for t in range(len(m) - 1) if m[t].any() and m[t + 1].any()]
    return float(np.mean(vals)) if vals else 0.0


def _split_fraction(m: np.ndarray) -> float:
    """Fraction of non-empty frames where the mask is two or more sizeable pieces: two objects merged into
    one track come apart as the objects move apart."""
    hits, n = 0, 0
    for f in m:
        if not f.any():
            continue
        n += 1
        k, _, stats, _ = cv2.connectedComponentsWithStats(f.astype(np.uint8), connectivity=8)
        areas = stats[1:, cv2.CC_STAT_AREA]
        hits += int((areas >= SPLIT_PIECE * areas.sum()).sum() >= 2)
    return hits / n if n else 0.0


def score_candidate(found: dict[int, np.ndarray]) -> dict:
    kept: list[int] = []
    for oid, m in sorted(found.items()):
        present = m.reshape(len(m), -1).any(1)
        if present.mean() < MIN_PRESENT:
            continue
        if m[present].mean() > MAX_AREA:
            continue
        if any(_iou(m, found[k]) > DUPLICATE_IOU for k in kept):
            continue
        kept.append(oid)
    steady = float(np.mean([_steadiness(found[k]) for k in kept])) if kept else 0.0
    # each counted track is worth up to 1: less when it is missing from frames or splits into pieces
    quality = sum(found[k].reshape(len(found[k]), -1).any(1).mean() * (1 - _split_fraction(found[k])) for k in kept)
    return {"tracks": len(kept), "quality": round(float(quality), 4), "steadiness": round(steady, 4), "kept": kept, "raw": len(found)}


def best(scores: list[tuple[object, dict]]):
    """scores: (candidate key, score_candidate result). Most tracks, then quality, then steadiness; first wins exact ties."""
    return max(enumerate(scores), key=lambda e: (e[1][1]["tracks"], round(e[1][1]["quality"], 2),
                                                 e[1][1]["steadiness"], -e[0]))[1]
