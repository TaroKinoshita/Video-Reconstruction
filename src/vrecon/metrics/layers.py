"""Scores for 2D layers against a synthetic clip's ground truth (evaluation only).

Layers are matched by id (the stand-in masks carry the ground-truth ids). Rotation and scale are compared after
removing one constant offset / factor per layer: the reconstruction takes its reference frame as unrotated and at
scale 1, which the video cannot tell apart from a sprite drawn rotated or larger.
"""

from __future__ import annotations

import numpy as np

from vrecon.layers2d.model import render, states
from vrecon.schema import Layer2D, Project


def _segments(layer: Layer2D, fps: float) -> list[tuple[int, int, str]]:
    out = []
    for a, b in zip(layer.keys, layer.keys[1:]):
        if any(abs(getattr(a, c) - getattr(b, c)) > 1e-6 for c in ("x", "y", "rotation", "scale", "opacity")):
            out.append((round(a.time * fps), round(b.time * fps), a.easing))
    return out


def score_layers(gt: Project, layers: list[Layer2D], sprites: dict[str, np.ndarray], index: np.ndarray,
                 id_map: dict[int, str], meta: dict, fps: float) -> dict:
    n, h, w = index.shape
    gt_layers = {lay.id: lay for lay in gt.reconstruction.layers}
    rec = {lay.id: lay for lay in layers}
    _, rec_index = render(layers, sprites, n, fps, (w, h))
    rec_lab = {i + 1: lay.id for i, lay in enumerate(layers)}
    per = {}
    for lid, g in gt_layers.items():
        if lid not in rec:
            per[lid] = {"missing": True}
            continue
        sg, sr = states(g, n, fps), states(rec[lid], n, fps)
        vis = sg[:, 4] >= 0.15
        pos = np.linalg.norm(sr[vis, :2] - sg[vis, :2], axis=1)
        drot = sr[vis, 2] - sg[vis, 2]
        rot = np.abs(drot - np.median(drot))
        ratio = sr[vis, 3] / sg[vis, 3]
        sc = np.abs(ratio / np.median(ratio) - 1)
        op = np.abs(sr[:, 4] - sg[:, 4])
        lab_g = next(k for k, v in id_map.items() if v == lid)
        lab_r = next(k for k, v in rec_lab.items() if v == lid)
        ious = []
        for f in range(n):
            a, b = index[f] == lab_g, rec_index[f] == lab_r
            u = (a | b).sum()
            if u:
                ious.append((a & b).sum() / u)
        # timing and easing: each ground-truth segment against the most overlapping reconstructed one
        segs_r = _segments(rec[lid], fps)
        seg_scores = []
        for s0, e0, ez in _segments(g, fps):
            if e0 - s0 <= 2:
                continue
            best = max(segs_r, key=lambda s: min(e0, s[1]) - max(s0, s[0]), default=None)
            if best is None or min(e0, best[1]) - max(s0, best[0]) <= 0:
                seg_scores.append({"gt": [s0, e0, ez], "rec": None})
                continue
            seg_scores.append({"gt": [s0, e0, ez], "rec": list(best), "start_err": best[0] - s0,
                               "end_err": best[1] - e0, "easing_match": best[2] == ez})
        per[lid] = {"iou": float(np.mean(ious)) if ious else None, "pos_err_px": float(pos.mean()),
                    "rot_err_deg": float(rot.mean()), "scale_err": float(sc.mean()), "opacity_err": float(op.mean()),
                    "keys_gt": len(g.keys), "keys_rec": len(rec[lid].keys), "z_rec": rec[lid].z, "z_gt": g.z,
                    "segments": seg_scores}
    ok = [p for p in per.values() if not p.get("missing")]
    segs = [s for p in ok for s in p["segments"] if s["rec"] is not None]
    # order: only pairs that overlap on screen (otherwise the video shows no order)
    cover = {}
    for lid in gt_layers:
        if lid in rec:
            lab_g = next(k for k, v in id_map.items() if v == lid)
            cover[lid] = np.zeros((h, w), bool)
    pairs = []
    for a in cover:
        for b in cover:
            if a < b:
                la = next(k for k, v in id_map.items() if v == a)
                lb = next(k for k, v in id_map.items() if v == b)
                _, both = render([gt_layers[a].model_copy(update={"z": 0}), gt_layers[b].model_copy(update={"z": 1})],
                                 {a: sprites[a], b: sprites[b]}, n, fps, (w, h), frames=list(range(0, n, 4)))
                _, rev = render([gt_layers[a].model_copy(update={"z": 1}), gt_layers[b].model_copy(update={"z": 0})],
                                {a: sprites[a], b: sprites[b]}, n, fps, (w, h), frames=list(range(0, n, 4)))
                if (both != rev).sum() >= 20:  # the order changes the picture
                    pairs.append((a, b))
    order_ok = (all((per[a]["z_rec"] < per[b]["z_rec"]) == (gt_layers[a].z < gt_layers[b].z) for a, b in pairs)
                if pairs else None)
    summary = {
        "layers": len(gt_layers), "missing": len(gt_layers) - len(ok),
        "iou_mean": float(np.mean([p["iou"] for p in ok if p["iou"] is not None])) if ok else None,
        "pos_err_px": float(np.mean([p["pos_err_px"] for p in ok])) if ok else None,
        "rot_err_deg": float(np.mean([p["rot_err_deg"] for p in ok])) if ok else None,
        "scale_err": float(np.mean([p["scale_err"] for p in ok])) if ok else None,
        "opacity_err": float(np.mean([p["opacity_err"] for p in ok])) if ok else None,
        "segments_found": f"{len(segs)}/{sum(len(p['segments']) for p in ok)}",
        "timing_err_frames": float(np.mean([(abs(s["start_err"]) + abs(s["end_err"])) / 2 for s in segs])) if segs else None,
        "easing_match": f"{sum(s['easing_match'] for s in segs)}/{len(segs)}",
        "z_order_ok": order_ok, "z_pairs_checked": len(pairs),
    }
    return {"summary": summary, "layers": per}
