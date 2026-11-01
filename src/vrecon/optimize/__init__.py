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
    id_map = {int(k): v for k, v in json.loads((d / "gt_passes" / "id_map.json").read_text()).items()}
    obs = Observation(id=obs_id, kind="mask", shot_id="shot_001",
                      summary={"labels": sorted(id_map.values()), "frames": int(masks.shape[0])},
                      provenance=Provenance(tool="ground_truth_stand_in",
                                            params={"warning": "evaluation data used as an observation; "
                                                               "for testing the machinery only"}))
    return MaskObservation(masks, id_map, obs)


# --- silhouette objective ----------------------------------------------------------------------------


def render_silhouettes(project: Project, shot_id: str, frames: list[int], ids: list[str],
                       size: tuple[int, int]) -> np.ndarray:
    """Index image per frame (1-based position in `ids`), painted far to near."""
    w, h = size
    out = np.zeros((len(frames), h, w), np.uint16)
    cam = next(c for c in project.reconstruction.cameras if c.shot_id == shot_id)
    objs = {o.id: o for o in project.reconstruction.objects}
    for i, f in enumerate(frames):
        r_c, c_c, k = camera_pose(cam, f)
        polys = []
        for n, oid in enumerate(ids, start=1):
            o = objs[oid]
            r, t, s = object_pose(project, o, shot_id, f)
            pts = (_CORNERS * (np.array(o.size or (1, 1, 1)) * s)) @ r.T + t
            uv, z = project_points(pts, r_c, c_c, k)
            if (z <= 1e-3).any():
                continue  # a corner behind the camera: skip (not handled by a convex hull)
            polys.append((float(np.linalg.norm(t - c_c)), n, cv2.convexHull(np.round(uv).astype(np.int32))))
        for _, n, hull in sorted(polys, reverse=True):
            cv2.fillConvexPoly(out[i], hull, n)
    return out


def silhouette_metrics(project: Project, obs: MaskObservation, pairs: dict[str, str], shot_id: str = "shot_001",
                       stride: int = 3) -> dict:
    """Loss = mean over objects of mean over frames of (1 - IoU); plus per-object diagnostics.

    pairs: observed label -> reconstruction object id.
    """
    t, h, w = obs.masks.shape
    frames = list(range(0, t, stride))
    ids = list(pairs.values())
    sil = render_silhouettes(project, shot_id, frames, ids, (w, h))
    inv = {v: k for k, v in obs.id_map.items()}
    per = {}
    for n, (label, oid) in enumerate(pairs.items(), start=1):
        ious, dx, dy = [], [], []
        for i, f in enumerate(frames):
            a, b = obs.masks[f] == inv[label], sil[i] == n
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
    'time' (shift of all of an animated object's key times, in frames)."""

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


def optimize(project: Project, obs: MaskObservation, pairs: dict[str, str], params: list[Param],
             hypothesis: str, proposal_id: str, max_evals: int = 300, stride: int = 3) -> tuple[Proposal, dict]:
    """Search the parameters and return a proposal for the run manager (it decides about adoption)."""
    def loss(x):
        return silhouette_metrics(apply_params(project, params, x)[0], obs, pairs, stride=stride)["silhouette_loss"]

    x0 = np.zeros(len(params))
    before = loss(x0)
    x, after, evals = pattern_search(loss, x0, np.array([p.step for p in params]), max_evals=max_evals)
    _, changes = apply_params(project, params, x)
    info = {"loss_before": before, "loss_after": after, "evaluations": evals,
            "values": {f"{p.kind}:{p.target_id}:{p.axis}": float(v) for p, v in zip(params, x)}}
    return Proposal(id=proposal_id, hypothesis=hypothesis, changes=changes, source="optimizer"), info
