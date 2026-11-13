"""Scores for full-venue completion (docs/VENUE_PLAN.md phase A). Pure numpy, per evaluation camera.

Objects are compared by category, not by id: a completed venue need not split the stands the way the ground truth
does. Categories come from id prefixes (both the synthetic venue's and the tennis pipeline's names).

    category_iou[c]  IoU of category c's pixels (only categories present in the ground truth view)
    covered          share of ground-truth pixels (anything built) where the candidate has something
    extra            share of the candidate's pixels where the ground truth has nothing
    same_category    on pixels both cover: share with the same category
    depth_rel        on pixels both cover with the same category: median |d - d_gt| / d_gt
In the source view, `observed_kept` is the share of the ground truth's pixels of the observed categories (what
the source shows) still drawn as the same category: completion must not hide or lose what was seen.
Plausibility of unseen parts (several answers can be right) is scored separately, not here.
"""

from __future__ import annotations

import numpy as np

PREFIXES = [("obj_floor", "floor"), ("obj_court", "court"), ("obj_net", "net"), ("obj_stand", "stands"),
            ("obj_barrier", "wall"), ("obj_wall", "wall"), ("obj_umpire", "furniture"), ("obj_bench", "furniture"),
            ("obj_chair", "furniture"), ("obj_player", "person"), ("obj_person", "person"), ("obj_background", "other")]
CATS = ["floor", "court", "net", "stands", "wall", "furniture", "person", "other"]


def category(oid: str) -> str:
    return next((c for p, c in PREFIXES if oid.startswith(p)), "other")


def category_map(index: np.ndarray, id_map: dict[int, str]) -> np.ndarray:
    """Pass index -> category number (0 = nothing, 1 + CATS index)."""
    lut = np.zeros(max([0, *id_map]) + 1, np.int16)
    for k, v in id_map.items():
        lut[k] = 1 + CATS.index(category(v))
    return lut[np.clip(index, 0, len(lut) - 1)]


def score_view(gt_index, gt_map, gt_depth, rec_index, rec_map, rec_depth) -> dict:
    g, r = category_map(gt_index, gt_map), category_map(rec_index, rec_map)
    gd, rd = gt_depth.astype(float), rec_depth.astype(float)
    out = {"category_iou": {}}
    for i, c in enumerate(CATS, 1):
        gm = g == i
        if gm.any():
            u = (gm | (r == i)).sum()
            out["category_iou"][c] = float((gm & (r == i)).sum() / u)
    both = (g > 0) & (r > 0)
    same = both & (g == r)
    out["covered"] = float(both.sum() / max((g > 0).sum(), 1))
    out["extra"] = float(((r > 0) & (g == 0)).sum() / max((r > 0).sum(), 1))
    out["same_category"] = float(same.sum() / max(both.sum(), 1))
    out["depth_rel"] = float(np.median(np.abs(rd[same] - gd[same]) / np.maximum(gd[same], 1e-6))) if same.any() else None
    return out


def observed_kept(gt_index, gt_map, rec_index, rec_map, observed_categories: set[str]) -> float:
    g, r = category_map(gt_index, gt_map), category_map(rec_index, rec_map)
    obs = np.isin(g, [1 + CATS.index(c) for c in observed_categories])
    return float(((g == r) & obs).sum() / max(obs.sum(), 1))


__all__ = ["CATS", "category", "category_map", "observed_kept", "score_view"]
