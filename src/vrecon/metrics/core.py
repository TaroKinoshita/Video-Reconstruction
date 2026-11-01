"""Scoring a reconstruction against a reference (PLAN.md 11.3). numpy only, no bpy.

Image-space measures (work for any reference that has object masks):
    match_objects       pair reference and reconstructed objects by mask overlap over the whole clip
    mask_iou            per object, per frame
    centroid_error      mask-centroid distance, normalised by the image diagonal

3D measures (need a ground-truth project, i.e. synthetic clips or annotated real ones):
    camera_errors       relative rotation, focal length, and centre error after a similarity alignment
    anchored_reprojection_error
                        pixels of each reference object are pinned to that object's surface, carried
                        through the reconstruction's object pose and camera, and re-projected

Timing (needs annotated onsets):
    onset_frame / timing_errors
"""

from __future__ import annotations

import itertools
import math

import numpy as np

from vrecon.schema import Project

from .poses import backproject, camera_pose, object_pose, project_points


def _diag(shape) -> float:
    return math.hypot(shape[-1], shape[-2])


# --- image space -------------------------------------------------------------------------------


def iou_matrix(ref: np.ndarray, rec: np.ndarray, ref_ids: list[int], rec_ids: list[int]) -> np.ndarray:
    """Clip-level IoU (intersections and unions summed over all frames)."""
    m = np.zeros((len(ref_ids), len(rec_ids)))
    for i, a in enumerate(ref_ids):
        ra = ref == a
        for j, b in enumerate(rec_ids):
            rb = rec == b
            union = np.logical_or(ra, rb).sum()
            m[i, j] = np.logical_and(ra, rb).sum() / union if union else 0.0
    return m


def match_objects(ref: np.ndarray, rec: np.ndarray, ref_map: dict[int, str], rec_map: dict[int, str],
                  min_iou: float = 0.05) -> dict[str, str]:
    """Reference id -> reconstruction id, maximising total clip IoU. Pairs below min_iou are dropped."""
    ref_ids, rec_ids = sorted(ref_map), sorted(rec_map)
    m = iou_matrix(ref, rec, ref_ids, rec_ids)
    n, k = m.shape
    best, best_score = [], -1.0
    if n <= 7 and k <= 7:  # exhaustive
        small, large = (n, k) if n <= k else (k, n)
        for perm in itertools.permutations(range(large), small):
            pairs = list(zip(range(small), perm)) if n <= k else [(p, i) for i, p in enumerate(perm)]
            score = sum(m[i, j] for i, j in pairs)
            if score > best_score:
                best, best_score = pairs, score
    else:  # greedy
        used_i, used_j = set(), set()
        for i, j in sorted(np.ndindex(n, k), key=lambda ij: -m[ij]):
            if i not in used_i and j not in used_j:
                best.append((i, j))
                used_i.add(i)
                used_j.add(j)
    return {ref_map[ref_ids[i]]: rec_map[rec_ids[j]] for i, j in best if m[i, j] >= min_iou}


def _inv(id_map: dict[int, str]) -> dict[str, int]:
    return {v: k for k, v in id_map.items()}


def mask_iou(ref, rec, ref_map, rec_map, pairs: dict[str, str]) -> dict[str, dict]:
    """Per reference object: IoU per frame (None when absent in both) and the mean over frames."""
    ri, ci = _inv(ref_map), _inv(rec_map)
    out = {}
    for a, b in pairs.items():
        per = []
        for t in range(ref.shape[0]):
            ra, rb = ref[t] == ri[a], rec[t] == ci[b]
            union = np.logical_or(ra, rb).sum()
            per.append(None if union == 0 else float(np.logical_and(ra, rb).sum() / union))
        vals = [v for v in per if v is not None]
        out[a] = {"matched": b, "mean": float(np.mean(vals)) if vals else None, "per_frame": per}
    return out


def _centroid(mask: np.ndarray):
    ys, xs = np.nonzero(mask)
    return None if len(xs) == 0 else (float(xs.mean()), float(ys.mean()))


def centroid_error(ref, rec, ref_map, rec_map, pairs: dict[str, str]) -> dict[str, dict]:
    ri, ci = _inv(ref_map), _inv(rec_map)
    diag = _diag(ref.shape)
    out = {}
    for a, b in pairs.items():
        per = []
        for t in range(ref.shape[0]):
            ca, cb = _centroid(ref[t] == ri[a]), _centroid(rec[t] == ci[b])
            per.append(None if ca is None or cb is None else math.dist(ca, cb) / diag)
        vals = [v for v in per if v is not None]
        out[a] = {"matched": b, "mean": float(np.mean(vals)) if vals else None,
                  "max": float(np.max(vals)) if vals else None, "per_frame": per}
    return out


# --- 3D (ground truth needed) -------------------------------------------------------------------


def _rot_angle_deg(r: np.ndarray) -> float:
    return math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(r) - 1) / 2))))


def umeyama(src: np.ndarray, dst: np.ndarray):
    """Similarity (s, R, t) minimising |s R src + t - dst|. Returns None when src is degenerate."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    var = (xs**2).sum() / len(src)
    if var < 1e-10:
        return None
    cov = xd.T @ xs / len(src)
    u, d, vt = np.linalg.svd(cov)
    sgn = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        sgn[2, 2] = -1
    r = u @ sgn @ vt
    s = np.trace(np.diag(d) @ sgn) / var
    return s, r, mu_d - s * r @ mu_s


def camera_errors(gt: Project, rec: Project, shot_id: str) -> dict:
    """Camera accuracy without assuming a shared world frame or scale.

    rotation: angle between the camera's rotation since the first frame, in gt and rec (degrees)
    focal:    |fx_rec - fx_gt| / fx_gt
    center:   camera-centre error after the best similarity alignment, as a fraction of the gt path
              length plus the camera's distance to the scene origin (None for a static camera)
    """
    shot = next(s for s in gt.shots if s.id == shot_id)
    cg = next(c for c in gt.reconstruction.cameras if c.shot_id == shot_id)
    cr = next(c for c in rec.reconstruction.cameras if c.shot_id == shot_id)
    n = shot.end - shot.start
    g = [camera_pose(cg, f) for f in range(n)]
    r = [camera_pose(cr, f) for f in range(n)]
    rot = []
    for (rg, _, _), (rr, _, _) in zip(g, r):
        rel_g = g[0][0].T @ rg
        rel_r = r[0][0].T @ rr
        rot.append(_rot_angle_deg(rel_g.T @ rel_r))
    focal = [abs(kr[0, 0] - kg[0, 0]) / kg[0, 0] for (_, _, kg), (_, _, kr) in zip(g, r)]
    cen_g = np.array([c for _, c, _ in g])
    cen_r = np.array([c for _, c, _ in r])
    sim = umeyama(cen_r, cen_g)
    center = None
    if sim is not None:
        s, rr, tt = sim
        aligned = (s * (rr @ cen_r.T)).T + tt
        scale = np.linalg.norm(np.diff(cen_g, axis=0), axis=1).sum() + np.linalg.norm(cen_g.mean(0))
        center = float(np.linalg.norm(aligned - cen_g, axis=1).mean() / max(scale, 1e-9))
    return {"rotation_deg_mean": float(np.mean(rot)), "rotation_deg_max": float(np.max(rot)),
            "focal_rel_mean": float(np.mean(focal)), "center_rel_mean": center,
            "rotation_deg_per_frame": rot}


def anchored_reprojection_error(gt: Project, rec: Project, shot_id: str, gt_depth: np.ndarray,
                                gt_index: np.ndarray, gt_map: dict[int, str], pairs: dict[str, str],
                                samples: int = 400, seed: int = 0) -> dict[str, dict]:
    """Per reference object: mean/median pixel error (normalised by the diagonal) of surface points.

    A gt pixel on object A is lifted to 3D with gt depth and camera, expressed in A's local frame
    (scaled by A's size), mapped to the matched reconstructed object B (scaled by B's size), placed
    with B's pose and re-projected with the reconstructed camera.
    """
    rng = np.random.default_rng(seed)
    cg = next(c for c in gt.reconstruction.cameras if c.shot_id == shot_id)
    cr = next(c for c in rec.reconstruction.cameras if c.shot_id == shot_id)
    objs_g = {o.id: o for o in gt.reconstruction.objects}
    objs_r = {o.id: o for o in rec.reconstruction.objects}
    gi = _inv(gt_map)
    diag = _diag(gt_index.shape)
    out = {}
    for a, b in pairs.items():
        og, orr = objs_g.get(a), objs_r.get(b)
        if og is None or orr is None:
            continue
        size_g = np.array(og.size or (1, 1, 1), float)
        size_r = np.array(orr.size or (1, 1, 1), float)
        errs = []
        for t in range(gt_index.shape[0]):
            vs, us = np.nonzero(gt_index[t] == gi[a])
            if len(us) == 0:
                continue
            pick = rng.choice(len(us), size=min(samples, len(us)), replace=False)
            us_t, vs_t = us[pick].astype(float), vs[pick].astype(float)
            d = gt_depth[t][vs[pick], us[pick]].astype(float)
            rcg, ccg, kg = camera_pose(cg, t)
            pw = backproject(us_t, vs_t, d, rcg, ccg, kg)
            rg, tg, sg = object_pose(gt, og, shot_id, t)
            local = ((pw - tg) @ rg) / (sg * size_g)  # unit-box coordinates
            rr, tr, sr = object_pose(rec, orr, shot_id, t)
            pw_r = (local * (sr * size_r)) @ rr.T + tr
            rcr, ccr, kr = camera_pose(cr, t)
            uv, z = project_points(pw_r, rcr, ccr, kr)
            ok = z > 1e-6
            if ok.any():
                errs.extend((np.linalg.norm(uv[ok] - np.stack([us_t, vs_t], 1)[ok], axis=1) / diag).tolist())
        out[a] = {"matched": b, "mean": float(np.mean(errs)) if errs else None,
                  "median": float(np.median(errs)) if errs else None, "points": len(errs)}
    return out


# --- timing ---------------------------------------------------------------------------------------


def onset_frame(project: Project, obj_id: str, shot_id: str, eps: float = 0.01) -> int | None:
    """First shot frame where the object has moved more than eps (m) or turned more than ~0.6 deg."""
    obj = next((o for o in project.reconstruction.objects if o.id == obj_id), None)
    shot = next(s for s in project.shots if s.id == shot_id)
    if obj is None:
        return None
    r0, t0, _ = object_pose(project, obj, shot_id, 0)
    for f in range(shot.end - shot.start):
        r, t, _ = object_pose(project, obj, shot_id, f)
        if np.linalg.norm(t - t0) > eps or _rot_angle_deg(r0.T @ r) > 0.6:
            return f
    return None


def timing_errors(ref_onsets: dict[str, int], rec: Project, shot_id: str, pairs: dict[str, str]) -> dict[str, dict]:
    out = {}
    for a, onset in ref_onsets.items():
        b = pairs.get(a)
        got = onset_frame(rec, b, shot_id) if b else None
        out[a] = {"matched": b, "ref_onset": onset, "rec_onset": got,
                  "error_frames": None if got is None else got - onset}
    return out
