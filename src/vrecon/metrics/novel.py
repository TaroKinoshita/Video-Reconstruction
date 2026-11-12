"""Scores from views the reference video never shows (scripts/make_novel_views.py). Pure numpy.

The reconstruction lives in its own world (a 4D model's units and axes), and its camera path has its own errors
(syn_03: 4RC's dolly is off by up to 2 m after the best single alignment). So the novel camera is carried over
RELATIVE TO EACH FRAME'S SOURCE CAMERA: the novel camera's pose in the ground-truth source camera's frame is applied
to the reconstruction's source camera at that frame, its offset scaled by the ratio of the two cameras' heights
above the floor (both worlds put the floor at z = 0, Z up). Camera-path errors are scored elsewhere (camera
measures); what is left here is the scene: objects' depth and size along the view, unseen sides, extents. Nothing
is fitted to the objects being scored. The reconstruction is rendered from there (evaluate.score_novel) and
compared with the ground-truth render:

    iou            per object, its pixels in the novel view (both sides front-most masks)
    covered        share of the ground truth's object pixels where the reconstruction draws anything
    extra          share of the reconstruction's object pixels where the ground truth shows nothing

The source view can look right while unseen sides are wrong, so these are reported apart from the source-view
scores (docs/HANDOFF.md, completion plan step 5).
"""

from __future__ import annotations

import numpy as np

from vrecon.schema import Camera, Project

from .poses import camera_pose


def carry_relative(novel: Camera, gt_src: Camera, rec_src: Camera) -> Camera:
    """Novel camera keys in the reconstruction's world, frame by frame relative to the source cameras."""
    from vrecon.scene.convert import matrix_to_quat
    from vrecon.schema import Pose

    keys = []
    for k in novel.keys:
        f = k.shot_frame
        rg, cg, _ = camera_pose(gt_src, f)
        rr, cr, _ = camera_pose(rec_src, f)
        rn, cn, _ = camera_pose(novel, f)
        if cg[2] <= 1e-6 or cr[2] <= 1e-6:
            raise ValueError("a camera is not above the floor: cannot take the scale from camera heights")
        s = cr[2] / cg[2]
        rel_r, rel_t = rg.T @ rn, rg.T @ (cn - cg)  # novel in the source camera's frame
        keys.append(k.model_copy(update={"pose": Pose(translation=tuple(map(float, cr + s * rr @ rel_t)),
                                                      rotation=matrix_to_quat((rr @ rel_r).tolist()))}))
    return novel.model_copy(update={"keys": keys})


def novel_scores(ref_index: np.ndarray, ref_map: dict[int, str], rec_index: np.ndarray, rec_map: dict[int, str],
                 pairs: dict[str, str], set_ids: set[str]) -> dict:
    rec_lab = {v: k for k, v in rec_map.items()}
    per = {}
    for lab, a in ref_map.items():
        g = ref_index == lab
        if not g.any():
            continue
        b = pairs.get(a)
        r = rec_index == rec_lab[b] if b in rec_lab else np.zeros_like(g)
        u = (g | r).sum()
        per[a] = {"iou": float((g & r).sum() / u) if u else None, "covered": float(((rec_index > 0) & g).sum() / g.sum()),
                  "set": a in set_ids}
    ref_any, rec_any = ref_index > 0, rec_index > 0
    movers = [v for v in per.values() if not v["set"] and v["iou"] is not None]
    sets = [v for v in per.values() if v["set"] and v["iou"] is not None]
    return {"objects": per,
            "iou_movers": float(np.mean([v["iou"] for v in movers])) if movers else None,
            "iou_set": float(np.mean([v["iou"] for v in sets])) if sets else None,
            "covered": float((rec_any & ref_any).sum() / max(ref_any.sum(), 1)),
            "extra": float((rec_any & ~ref_any).sum() / max(rec_any.sum(), 1))}


__all__ = ["carry_relative", "novel_scores"]
