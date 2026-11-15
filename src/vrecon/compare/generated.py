"""Score a generated video against the 3D scene its controls came from (RQ2, PLAN.md 11.4; docs/REVIEW_INSTRUCTIONS.md
F-25, WP-4). Pure numpy/cv2: heavy observers (segmentation, camera) run elsewhere and are passed in as files.

Inputs
    video          the generated video, uint8 RGB [T, H, W, 3] (same size and length as the controls)
    controls_dir   an export_controls folder: manifest.json, passes/index.npz (+ id_map), passes/status.npz,
                   control_canny.mp4
    masks          optional, the observer contract: {object id or label: bool [T, H, W]} or a SAM-style npz
                   (masks [T, H, W] int labels, id_map {num: label}); labels are paired to scene objects by best
                   IoU with the index pass (as metrics.match_objects does for reconstructions)
    base_video     optional, the generation from the unedited scene, for the unchanged-region measure
    unchanged      object ids that an edit must leave alone (evaluation/phase1a/edits.json)
    scene, camera  optional Projects: the scene the controls came from and one with a camera estimated from the
                   generated video (same shot id); camera agreement uses metrics.camera_errors

Measures (each split by the status pass where it is per pixel: observed / standard / hypothesis)
    edges        F1 of the video's Canny edges against the control's edges, 2 px tolerance (no observer needed)
    objects      per object: mask IoU and centroid error (fraction of the image diagonal) against the index pass;
                 the object's status share; means over objects by dominant status
    onsets       per moving object: onset frame of its 2D centroid track (first frame beyond 5% of its largest
                 displacement, as metrics.onset_frame_relative) in the masks and in the index pass, and the error
    unchanged    mean absolute difference (0-255) and PSNR between base and this video on the unchanged objects'
                 pixels (where the index pass shows them in both)
    camera       rotation / focal / centre errors (when an estimated camera is given)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from vrecon.control.status import HYPOTHESIS, NAMES, OBSERVED, STANDARD

TOL_PX = 2
ONSET_SHARE = 0.05


def load_controls(controls_dir: str | Path) -> dict:
    from vrecon.control.core import read_video_rgb

    d = Path(controls_dir)
    man = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    index = np.load(d / man["passes"]["index"])["index"]
    id_map = {int(k): v for k, v in man["passes"]["id_map"].items()}
    st_path = d / man["passes"].get("status", "passes/status.npz")
    status = np.load(st_path)["status"] if st_path.is_file() else np.where(index > 0, OBSERVED, 0).astype(np.uint8)
    canny = read_video_rgb(d / "control_canny.mp4")[..., 0] > 127 if (d / "control_canny.mp4").is_file() else None
    return {"manifest": man, "index": index, "id_map": id_map, "status": status, "canny": canny}


def masks_from_index(index: np.ndarray, id_map: dict[int, str]) -> dict[str, np.ndarray]:
    """Per-object masks from an index pass (a perfect observer: dry runs, and the reference side)."""
    return {oid: index == lab for lab, oid in id_map.items() if (index == lab).any()}


def load_masks(path: str | Path) -> dict[str, np.ndarray]:
    """A SAM-style npz (run_sam3.py): masks [T, H, W] int labels, id_map {num: label}."""
    with np.load(path) as z:
        m = z["masks"]
        id_map = json.loads(str(z["id_map"]))
    return {label: m == int(num) for num, label in id_map.items()}


def pair_masks(masks: dict[str, np.ndarray], ref: dict[str, np.ndarray], min_iou: float = 0.05) -> dict[str, str]:
    """{scene object id: mask label}, greedy by clip IoU (each side used once)."""
    cand = []
    for a, r in ref.items():
        for b, m in masks.items():
            u = (r | m).sum()
            if u:
                cand.append(((r & m).sum() / u, a, b))
    out, used = {}, set()
    for iou, a, b in sorted(cand, reverse=True):
        if iou >= min_iou and a not in out and b not in used:
            out[a] = b
            used.add(b)
    return out


def _edges(frames: np.ndarray) -> np.ndarray:
    import cv2

    return np.stack([cv2.Canny(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY), 100, 200) > 0 for f in frames])


def edge_f1(video: np.ndarray, control_edges: np.ndarray, status: np.ndarray) -> dict:
    import cv2

    got = _edges(video)
    k = np.ones((2 * TOL_PX + 1, 2 * TOL_PX + 1), np.uint8)
    near_ctrl = np.stack([cv2.dilate(e.astype(np.uint8), k) > 0 for e in control_edges])
    near_got = np.stack([cv2.dilate(e.astype(np.uint8), k) > 0 for e in got])
    out = {}
    for code in (OBSERVED, STANDARD, HYPOTHESIS, None):
        region = np.ones_like(status, bool) if code is None else status == code
        g, c = got & region, control_edges & region
        if c.sum() == 0 and g.sum() == 0:
            out[NAMES.get(code, "all")] = None
            continue
        prec = float((g & near_ctrl).sum() / max(g.sum(), 1))
        rec = float((c & near_got).sum() / max(c.sum(), 1))
        out[NAMES.get(code, "all")] = {"precision": prec, "recall": rec,
                                       "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0}
    return out


def _centroids(mask: np.ndarray) -> np.ndarray:
    out = np.full((len(mask), 2), np.nan)
    for f, m in enumerate(mask):
        ys, xs = np.nonzero(m)
        if len(xs):
            out[f] = (xs.mean(), ys.mean())
    return out


def onset_2d(track: np.ndarray, min_px: float = 3.0) -> int | None:
    ok = np.all(np.isfinite(track), 1)
    if ok.sum() < 2:
        return None
    first = int(np.argmax(ok))
    d = np.linalg.norm(track - track[first], axis=1)
    d[~ok] = 0
    if d.max() < min_px:
        return None
    return max(int(np.argmax(d > ONSET_SHARE * d.max())) - 1, 0)


def score_generated(video: np.ndarray, controls: dict, masks: dict[str, np.ndarray] | None = None,
                    base_video: np.ndarray | None = None, unchanged: list[str] | None = None,
                    scene=None, camera=None, movers: list[str] | None = None) -> dict:
    """movers: objects whose onset is scored (default: with a scene, its objects with two or more keys; without
    one, none). A still object's 2D centroid also moves when something passes in front of it."""
    if movers is None and scene is not None:
        movers = [o.id for o in scene.reconstruction.objects if o.kind != "set" and len(o.keys) >= 2]
    movers = set(movers or [])
    index, id_map, status = controls["index"], controls["id_map"], controls["status"]
    if video.shape[:3] != index.shape:
        raise ValueError(f"video {video.shape[:3]} does not match the controls {index.shape}")
    diag = float(np.hypot(index.shape[1], index.shape[2]))
    rep: dict = {"frames": int(index.shape[0]), "size": [int(index.shape[2]), int(index.shape[1])]}
    if controls.get("canny") is not None:
        rep["edges"] = edge_f1(video, controls["canny"], status)

    ref = masks_from_index(index, id_map)
    if masks is not None:
        pairs = pair_masks(masks, ref)
        objs = {}
        for oid, r in ref.items():
            st = status[r]
            share = {NAMES[c]: float((st == c).mean()) for c in (OBSERVED, STANDARD, HYPOTHESIS)}
            dominant = max(share, key=share.get)
            m = masks.get(pairs.get(oid)) if oid in pairs else None
            if m is None:
                objs[oid] = {"matched": False, "iou": 0.0, "status": dominant, "status_share": share}
                continue
            inter, union = (r & m).sum((1, 2)), (r | m).sum((1, 2))
            seen = union > 0
            ca, cb = _centroids(r), _centroids(m)
            err = np.linalg.norm(ca - cb, axis=1) / diag
            o = {"matched": True, "label": pairs[oid], "iou": float((inter[seen] / union[seen]).mean()),
                 "centroid_error": float(np.nanmean(err)) if np.isfinite(err).any() else None,
                 "status": dominant, "status_share": share}
            on_ref, on_got = (onset_2d(ca), onset_2d(cb)) if oid in movers else (None, None)
            if on_ref is not None or on_got is not None:
                o["onset"] = {"reference": on_ref, "video": on_got,
                              "error_frames": None if on_ref is None or on_got is None else on_got - on_ref}
            objs[oid] = o
        rep["objects"] = objs
        by = {}
        for name in ("observed", "standard", "hypothesis"):
            vals = [o["iou"] for o in objs.values() if o["status"] == name]
            by[name] = {"objects": len(vals), "iou_mean": float(np.mean(vals)) if vals else None}
        rep["objects_by_status"] = by
        movers = [o for o in objs.values() if "onset" in o and o["onset"]["error_frames"] is not None]
        rep["onset_abs_error_mean"] = float(np.mean([abs(o["onset"]["error_frames"]) for o in movers])) \
            if movers else None

    if base_video is not None and unchanged:
        keep = np.zeros(index.shape, bool)
        for lab, oid in id_map.items():
            if oid in unchanged:
                keep |= index == lab
        if keep.any():
            diff = np.abs(video.astype(np.int16) - base_video.astype(np.int16)).mean(-1)[keep]
            mse = float((diff.astype(np.float64) ** 2).mean())
            rep["unchanged"] = {"objects": sorted(unchanged), "pixels": int(keep.sum()), "mad": float(diff.mean()),
                                "psnr": float(10 * np.log10(255 ** 2 / mse)) if mse > 0 else None}
    if scene is not None and camera is not None:
        from vrecon.metrics import camera_errors

        rep["camera"] = camera_errors(scene, camera, scene.shots[0].id)
    return rep


__all__ = ["edge_f1", "load_controls", "load_masks", "masks_from_index", "onset_2d", "pair_masks",
           "score_generated"]
