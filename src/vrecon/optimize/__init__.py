"""Numeric search for the repair loop (PLAN.md 6.1-6.2): the LLM picks what to suspect, this finds values.

Objective: silhouette mismatch between observed object masks and the reconstruction's proxy boxes,
rasterised directly (box corners projected, convex hull filled, far-to-near), without Blender, so a
search can afford hundreds of evaluations.

Observed masks come from an observation provider. Until a segmentation tool is integrated, the only
provider is a ground-truth stand-in for synthetic clips; its provenance says so (PLAN.md 11.8: scores
obtained this way only show that the machinery works).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from vrecon.metrics.poses import camera_pose, object_pose, project_points
from vrecon.run import Change, Proposal
from vrecon.schema import Observation, Pose, Project, Provenance, TransformKey

_CORNERS = np.array([[x, y, z] for x in (-0.5, 0.5) for y in (-0.5, 0.5) for z in (-0.5, 0.5)])


# --- observations ----------------------------------------------------------------------------------


@dataclass
class MaskObservation:
    masks: np.ndarray  # uint16 [T, H, W], 0 = background
    id_map: dict[int, str]  # mask number -> observed object label
    observation: Observation


def gt_stand_in_masks(clip_dir: str | Path, obs_id: str = "obs_masks_gt_stand_in") -> MaskObservation:
    """Object masks from a synthetic clip's ground truth, labelled as a stand-in."""
    d = Path(clip_dir)
    masks = np.load(d / "gt_passes" / "index.npz")["index"]
    id_map = {int(k): v for k, v in json.loads((d / "gt_passes" / "id_map.json").read_text(encoding="utf-8")).items()}
    obs = Observation(id=obs_id, kind="mask", shot_id="shot_001",
                      summary={"labels": sorted(id_map.values()), "frames": int(masks.shape[0])},
                      provenance=Provenance(tool="ground_truth_stand_in",
                                            params={"warning": "evaluation data used as an observation; "
                                                               "for testing the machinery only"}))
    return MaskObservation(masks, id_map, obs)


def load_masks(path: str | Path, obs_id: str = "obs_masks_file") -> MaskObservation:
    """Masks written by an external segmenter (gpu/observe/run_sam3.py), which runs in its own env."""
    with np.load(path) as d:
        masks, labels, meta = d["masks"], json.loads(str(d["id_map"])), json.loads(str(d["meta"]))
    id_map = {int(k): v for k, v in labels.items()}
    obs = Observation(id=obs_id, kind="mask", shot_id="shot_001",
                      summary={"labels": sorted(id_map.values()), "frames": int(masks.shape[0])},
                      provenance=Provenance(tool=meta["tool"], params=meta))
    return MaskObservation(masks, id_map, obs)


# --- silhouette objective ----------------------------------------------------------------------------


# box faces as corner indices into _CORNERS (x, y, z in {-0.5, 0.5}, index = 4*ix + 2*iy + iz)
_FACES = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
_NEAR = 0.05


def _clip_near(poly: np.ndarray) -> np.ndarray:
    """Sutherland-Hodgman clip of a camera-space polygon against z >= _NEAR."""
    out = []
    for i in range(len(poly)):
        a, b = poly[i], poly[(i + 1) % len(poly)]
        ina, inb = a[2] >= _NEAR, b[2] >= _NEAR
        if ina:
            out.append(a)
        if ina != inb:
            t = (_NEAR - a[2]) / (b[2] - a[2])
            out.append(a + t * (b - a))
    return np.array(out)


def render_silhouettes(project: Project, shot_id: str, frames: list[int], ids: list[str],
                       size: tuple[int, int], down: int = 1) -> np.ndarray:
    """Index image per frame (1-based position in `ids`) with a per-pixel depth test.

    Faces are clipped at the camera's near plane, so large set pieces (floor, wall) that extend
    behind the camera are drawn correctly. Depth per pixel comes from each face's plane:
    z = d / (n . K^-1 [u, v, 1]). `down` renders at 1/down resolution (size is the full size).
    """
    w, h = -(-size[0] // down), -(-size[1] // down)
    out = np.zeros((len(frames), h, w), np.uint16)
    cam = next(c for c in project.reconstruction.cameras if c.shot_id == shot_id)
    objs = {o.id: o for o in project.reconstruction.objects}
    for i, f in enumerate(frames):
        r_c, c_c, k = camera_pose(cam, f)
        if down > 1:  # pixel centres of the strided grid [::down] map to full-res u = down * u'
            k = k.copy()
            k[0, 0] /= down
            k[1, 1] /= down
            k[0, 2] /= down
            k[1, 2] /= down
        zbuf = np.full((h, w), np.inf)
        img = out[i]
        for n, oid in enumerate(ids, start=1):
            o = objs[oid]
            r, t, s = object_pose(project, o, shot_id, f)
            world = (_CORNERS * (np.array(o.size or (1, 1, 1)) * s)) @ r.T + t
            cam_pts = (world - c_c) @ r_c  # camera coordinates (OpenCV)
            if (cam_pts[:, 2] < _NEAR).all():
                continue
            for face in _FACES:
                poly = _clip_near(cam_pts[list(face)])
                if len(poly) < 3:
                    continue
                normal = np.cross(poly[1] - poly[0], poly[2] - poly[0])
                if np.linalg.norm(normal) < 1e-12:
                    continue
                d = float(normal @ poly[0])
                uv = poly[:, :2] / poly[:, 2:3] * np.array([k[0, 0], k[1, 1]]) + np.array([k[0, 2], k[1, 2]])
                x0, y0 = np.floor(uv.min(0)).astype(int)
                x1, y1 = np.ceil(uv.max(0)).astype(int)
                x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, w - 1), min(y1, h - 1)
                if x1 < x0 or y1 < y0:
                    continue
                sub = np.zeros((y1 - y0 + 1, x1 - x0 + 1), np.uint8)
                cv2.fillPoly(sub, [np.round(uv - [x0, y0]).astype(np.int32)], 1)
                ys, xs = np.nonzero(sub)
                if len(xs) == 0:
                    continue
                us, vs = xs + x0, ys + y0
                denom = normal[0] * (us - k[0, 2]) / k[0, 0] + normal[1] * (vs - k[1, 2]) / k[1, 1] + normal[2]
                with np.errstate(divide="ignore", invalid="ignore"):
                    z = d / denom
                ok = (z > 0) & (z < zbuf[vs, us])
                zbuf[vs[ok], us[ok]] = z[ok]
                img[vs[ok], us[ok]] = n
    return out


def silhouette_metrics(project: Project, obs: MaskObservation, pairs: dict[str, str], shot_id: str = "shot_001",
                       stride: int = 3, down: int = 2) -> dict:
    """Loss = mean over objects of mean over frames of (1 - IoU); plus per-object diagnostics.

    pairs: observed label -> reconstruction object id.
    """
    t, h0, w0 = obs.masks.shape
    frames = list(range(0, t, stride))
    ids = list(pairs.values())
    sil = render_silhouettes(project, shot_id, frames, ids, (w0, h0), down=down)
    masks = obs.masks[:, ::down, ::down]
    h, w = masks.shape[1:]
    inv = {v: k for k, v in obs.id_map.items()}
    per = {}
    for n, (label, oid) in enumerate(pairs.items(), start=1):
        ious, dx, dy = [], [], []
        for i, f in enumerate(frames):
            a, b = masks[f] == inv[label], sil[i] == n
            union = np.logical_or(a, b).sum()
            if union == 0:
                continue
            ious.append(np.logical_and(a, b).sum() / union)
            if a.any() and b.any():
                ya, xa = np.nonzero(a)
                yb, xb = np.nonzero(b)
                dx.append((xb.mean() - xa.mean()) / w)
                dy.append((yb.mean() - ya.mean()) / h)
        per[oid] = {"loss": float(1 - np.mean(ious)) if ious else 1.0,
                    "offset_x": float(np.mean(dx)) if dx else None,  # + = reconstruction is to the right
                    "offset_y": float(np.mean(dy)) if dy else None}  # + = reconstruction is lower
    return {"silhouette_loss": float(np.mean([p["loss"] for p in per.values()])), "objects": per}


# --- parameters and search -------------------------------------------------------------------------


@dataclass
class Param:
    """kind: 'translation' (offset added to an object's rest and all keys, along axis 0-2),
    'size' (object size along axis), 'focal' (multiplier on all of a camera's focal lengths),
    'time' (shift of all of an animated object's key times, in frames). A 'time' parameter searched with
    others is scanned over TIME_SCAN_FRAMES, refitting the others at each shift."""

    kind: str
    target_id: str
    axis: int = 0
    step: float = 0.1


def apply_params(project: Project, params: list[Param], x: np.ndarray) -> tuple[Project, list[Change]]:
    p = project.model_copy(deep=True)
    objs = {o.id: o for o in p.reconstruction.objects}
    cams = {c.id: c for c in p.reconstruction.cameras}
    touched: dict[tuple[str, str], object] = {}
    for prm, v in zip(params, x):
        if prm.kind == "translation":
            o = objs[prm.target_id]

            def shift(k: TransformKey) -> TransformKey:
                t = list(k.pose.translation)
                t[prm.axis] += float(v)
                return k.model_copy(update={"pose": Pose(translation=tuple(t), rotation=k.pose.rotation)})

            o.rest = shift(o.rest)
            o.keys = [shift(k) for k in o.keys]
            touched[(o.id, "rest")] = o.rest
            touched[(o.id, "keys")] = o.keys
        elif prm.kind == "size":
            o = objs[prm.target_id]
            sz = list(o.size or (1, 1, 1))
            sz[prm.axis] = max(sz[prm.axis] + float(v), 0.01)
            o.size = tuple(sz)
            touched[(o.id, "size")] = o.size
        elif prm.kind == "time":
            o = objs[prm.target_id]
            if not o.keys:
                raise ValueError(f"{o.id} is not animated; 'time' needs keys")
            dt = float(v) * p.source.fps.den / p.source.fps.num
            o.keys = [k.model_copy(update={"time": k.time + dt}) for k in o.keys]
            touched[(o.id, "keys")] = o.keys
        elif prm.kind == "focal":
            c = cams[prm.target_id]
            m = max(1.0 + float(v), 0.05)
            c.intrinsics = c.intrinsics.model_copy(update={"fx": c.intrinsics.fx * m, "fy": c.intrinsics.fy * m})
            c.keys = [k.model_copy(update={"intrinsics": None if k.intrinsics is None else k.intrinsics.model_copy(
                update={"fx": k.intrinsics.fx * m, "fy": k.intrinsics.fy * m})}) for k in c.keys]
            touched[(c.id, "intrinsics")] = c.intrinsics
            touched[(c.id, "keys")] = c.keys
        else:
            raise ValueError(f"unknown parameter kind {prm.kind!r}")
    changes = [Change(target_id=t, field=f, value=val) for (t, f), val in touched.items()]
    return p, changes


def pattern_search(f, x0: np.ndarray, steps: np.ndarray, max_evals: int = 300, min_ratio: float = 0.02):
    """Derivative-free coordinate search with step halving. Returns (x, fx, evals)."""
    x, fx, evals = x0.astype(float).copy(), f(x0), 1
    step = steps.astype(float).copy()
    while evals < max_evals and (step > steps * min_ratio).any():
        improved = False
        for i in range(len(x)):
            for sign in (1, -1):
                y = x.copy()
                y[i] += sign * step[i]
                fy = f(y)
                evals += 1
                if fy < fx:
                    x, fx, improved = y, fy, True
                    break
            if evals >= max_evals:
                break
        if not improved:
            step *= 0.5
    return x, fx, evals


TIME_SCAN_FRAMES = range(-8, 9, 2)


def _scan_time(loss, params: list[Param], steps: np.ndarray, evals_per_shift: int):
    """Try each timing shift and refit the other parameters there.

    An earlier position search absorbs a timing error, so the correct shift alone looks worse than no
    shift: searching timing by itself stays where it is.
    """
    t = next(i for i, p in enumerate(params) if p.kind == "time")
    others = [i for i in range(len(params)) if i != t]
    best, best_loss, total = None, np.inf, 0
    for shift in TIME_SCAN_FRAMES:
        def at_shift(y):
            x = np.zeros(len(params))
            x[t] = shift
            x[others] = y
            return loss(x)

        y, fy, evals = pattern_search(at_shift, np.zeros(len(others)), steps[others], max_evals=evals_per_shift)
        total += evals
        if fy < best_loss:
            best, best_loss = np.zeros(len(params)), fy
            best[t] = shift
            best[others] = y
    return best, best_loss, total


TRACK_WEIGHT = 50.0  # puts track loss (fraction of the diagonal, ~0.001) on the silhouette loss scale (~0.1)


def combined_loss(project: Project, obs: MaskObservation | None, pairs: dict[str, str], tracks=None,
                  track_ids: list[str] | None = None, stride: int = 3, use_silhouette: bool = True) -> float:
    total = 0.0
    if use_silhouette and obs is not None and pairs:
        total += silhouette_metrics(project, obs, pairs, stride=stride)["silhouette_loss"]
    if tracks is not None:
        from .tracks import track_metrics

        total += TRACK_WEIGHT * track_metrics(project, tracks, ids=track_ids)["track_loss"]
    return total


def optimize(project: Project, obs: MaskObservation | None, pairs: dict[str, str], params: list[Param],
             hypothesis: str, proposal_id: str, max_evals: int = 300, stride: int = 3, tracks=None,
             track_ids: list[str] | None = None, use_silhouette: bool = True) -> tuple[Proposal, dict]:
    """Search the parameters and return a proposal for the run manager (it decides about adoption).

    The searched loss is the silhouette loss and/or the point-track loss (see combined_loss).
    """
    def loss(x, stride=stride):
        return combined_loss(apply_params(project, params, x)[0], obs, pairs, tracks, track_ids, stride, use_silhouette)

    x0 = np.zeros(len(params))
    before = loss(x0)
    steps = np.array([p.step for p in params])
    if any(p.kind == "time" for p in params) and len(params) > 1:
        x, after, evals = _scan_time(loss, params, steps, max_evals)
    else:
        x, after, evals = pattern_search(loss, x0, steps, max_evals=max_evals)
    if any(p.kind == "time" for p in params) and stride > 1:
        # timing is invisible between sampled frames: polish on every frame with small steps
        def fine(y):
            return loss(y, stride=1)

        fine_steps = np.array([1.0 if p.kind == "time" else p.step / 8 for p in params])
        x, after, e2 = pattern_search(fine, x, fine_steps, max_evals=30, min_ratio=0.25)
        evals += e2
        before = fine(x0)
    _, changes = apply_params(project, params, x)
    info = {"loss_before": before, "loss_after": after, "evaluations": evals,
            "values": {f"{p.kind}:{p.target_id}:{p.axis}": float(v) for p, v in zip(params, x)}}
    return Proposal(id=proposal_id, hypothesis=hypothesis, changes=changes, source="optimizer"), info
