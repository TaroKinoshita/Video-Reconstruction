"""First reconstruction from a feed-forward 4D model (4RC) plus object masks. Pure numpy.

Input (`load_4rc`): per sampled frame a camera-to-world pose (OpenCV axes, world = first camera), pinhole
intrinsics and a world point map at the model's resolution. Output: cameras, a floor at z = 0 (Z up), a back
wall when one is visible, and one upright box per masked object. An object whose per-frame boxes move gets a
key per sampled frame (position and yaw from that frame's points); otherwise one rest pose.
Scale stays in the model's units (recorded as "estimated").
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vrecon.scene.convert import matrix_to_quat
from vrecon.schema import Pose
from vrecon.schema.common import Evidence, Intrinsics, Scale
from vrecon.schema.project import Camera, CameraKey, SceneObject, TransformKey


@dataclass
class Model4D:
    frames: np.ndarray  # [N] source frame of each sample
    c2w: np.ndarray  # [N, 4, 4] OpenCV camera-to-world
    K: np.ndarray  # [N, 3, 3] in SOURCE pixels
    pts: np.ndarray  # [N, h, w, 3] world points at model resolution
    conf: np.ndarray  # [N, h, w]
    src_uv: np.ndarray  # [h, w, 2] source pixel (integer-centred) of each model pixel


def model_to_source(h: int, w: int, src_w: int, src_h: int) -> tuple[float, float, float]:
    """4RC/DUSt3R preprocessing: resize the long side to 512, then centre-crop to (w, h).
    Returns (scale, crop_x, crop_y): source_px + 0.5 = (model_px + 0.5 + crop) / scale."""
    s = 512 / max(src_w, src_h)
    rw, rh = round(src_w * s), round(src_h * s)
    return s, (rw - w) // 2, (rh - h) // 2


def load_4rc(path, src_w: int, src_h: int, frames: list[int]) -> Model4D:
    z = np.load(path)
    n = int(z["n_frames"])
    assert n == len(frames), f"{n} predictions for {len(frames)} frames"
    pts = np.stack([z[f"pred_{i}_pts"][0] for i in range(n)])
    h, w = pts.shape[1:3]
    if "model_to_source" in z:  # other models (e.g. MoRe/VGGT: width to 518, height to a multiple of 14)
        sx, sy, ox, oy = (float(v) for v in z["model_to_source"])
    else:
        s, ox, oy = model_to_source(h, w, src_w, src_h)
        sx = sy = s
    K = np.stack([z[f"pred_{i}_intrinsic"] for i in range(n)]).astype(np.float64)
    K[:, 0, 0] /= sx
    K[:, 1, 1] /= sy
    K[:, 0, 2] = (K[:, 0, 2] + 0.5 + ox) / sx - 0.5
    K[:, 1, 2] = (K[:, 1, 2] + 0.5 + oy) / sy - 0.5
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    src_uv = np.stack([(u + 0.5 + ox) / sx - 0.5, (v + 0.5 + oy) / sy - 0.5], -1)
    return Model4D(np.asarray(frames), np.stack([z[f"pred_{i}_extrinsic"] for i in range(n)]).astype(np.float64),
                   K, pts.astype(np.float64), np.stack([z[f"pred_{i}_conf"][0] for i in range(n)]), src_uv)


def labels_at_model_pixels(m: Model4D, masks: np.ndarray) -> np.ndarray:
    """[N, h, w] mask labels (source masks [T, H, W]) sampled at each model pixel."""
    H, W = masks.shape[1:]
    ui = np.clip(np.rint(m.src_uv[..., 0]).astype(int), 0, W - 1)
    vi = np.clip(np.rint(m.src_uv[..., 1]).astype(int), 0, H - 1)
    return masks[m.frames][:, vi, ui]


def fit_plane(p: np.ndarray, iters: int = 300, tol: float | None = None, seed: int = 0):
    """RANSAC plane: (unit normal n, d) with n.p + d = 0, and the inlier mask."""
    rng = np.random.default_rng(seed)
    if len(p) > 50_000:
        p = p[rng.choice(len(p), 50_000, replace=False)]
    if tol is None:
        tol = 0.01 * np.median(np.linalg.norm(p - np.median(p, 0), axis=1))
    best = (None, 0.0, np.zeros(len(p), bool))
    for _ in range(iters):
        a, b, c = p[rng.choice(len(p), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-12:
            continue
        n /= np.linalg.norm(n)
        inl = np.abs(p @ n - a @ n) < tol
        if inl.sum() > best[2].sum():
            best = (n, -a @ n, inl)
    q = p[best[2]]  # refine with least squares on the inliers
    c = q.mean(0)
    n = np.linalg.svd(q - c, full_matrices=False)[2][-1]
    return n, -c @ n, np.abs(p @ n - c @ n) < tol


def _confident(m: Model4D, sel: np.ndarray, keep: float) -> np.ndarray:
    pts, conf = m.pts[sel], m.conf[sel]
    return pts[conf >= np.quantile(conf, 1 - keep)] if len(conf) else pts


def _confident_frame(m: Model4D, i: int, sel: np.ndarray, keep: float) -> np.ndarray:
    pts, conf = m.pts[i][sel], m.conf[i][sel]
    return pts[conf >= np.quantile(conf, 1 - keep)] if len(conf) else pts


def _touches_border(mask: np.ndarray) -> bool:
    return bool(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any())


MIN_DEPTH_RATIO = 0.5  # side facing away from the camera, at least this share of the other horizontal side


def _deepen(xy: np.ndarray, yaw: float, size: tuple, ratio: float = MIN_DEPTH_RATIO):
    """4RC points cover only the faces the camera sees, so the side along the view comes out thin.
    The first camera looks along +Y (floor_frame). Of the box's two horizontal axes, the one closer to +Y
    is extended to at least `ratio` of the other, away from the camera, keeping the front face in place.
    A prior, not a measurement: the hidden depth is unknown."""
    axes = np.array([[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]])
    k = int(np.argmax(np.abs(axes[:, 1])))
    want = ratio * size[1 - k]
    if size[k] >= want:
        return xy, size
    d = axes[k] * np.sign(axes[k, 1])  # pointing away from the camera
    new = list(size)
    new[k] = want
    return xy + d * (want - size[k]) / 2, tuple(new)


STILL_PROGRESS = 0.1  # share of the start-to-end travel below which a key counts as still at an end


def _snap_still_ends(frames: list[int], xy: np.ndarray, yaw: np.ndarray, n_end: int = 3):
    """Per-frame boxes wobble while the object stands still, which reads as motion from the first frame.
    Along the start-to-end travel, keys within STILL_PROGRESS of either end are replaced by one key at the
    end's median pose, placed at the last still frame (start) and the first still frame (end).
    Left unchanged when the path does not go from one place to another (travel small against the spread)."""
    if len(frames) < 2 * n_end + 1:
        return frames, xy, yaw
    a, b = np.median(xy[:n_end], 0), np.median(xy[-n_end:], 0)
    travel = float(np.linalg.norm(b - a))
    spread = float(np.linalg.norm(xy - np.median(xy, 0), axis=1).max())
    if travel < spread:  # e.g. goes and comes back: no single start and end
        return frames, xy, yaw
    s = (xy - a) @ (b - a) / travel**2
    moving = np.nonzero((s > STILL_PROGRESS) & (s < 1 - STILL_PROGRESS))[0]
    if len(moving) == 0:
        return frames, xy, yaw
    i0, i1 = max(moving[0] - 1, 0), min(moving[-1] + 1, len(frames) - 1)
    ya, yb = np.median(yaw[:i0 + 1]), np.median(yaw[i1:])
    f = [frames[i0]] + list(frames[i0 + 1:i1]) + [frames[i1]]
    p = np.vstack([a, xy[i0 + 1:i1], b])
    y = np.r_[ya, yaw[i0 + 1:i1], yb]
    return f, p, y


MIN_FLOOR_POINTS = 500  # draft: fewer floor-mask points than this and the floor plane is not trusted

MOVE_FRACTION = 0.5 # an object moves when its per-frame box centres spread more than this share of its size


def floor_frame(m: Model4D, floor_pts: np.ndarray) -> tuple[np.ndarray, dict]:
    """World alignment A (4x4): floor -> z = 0, +Z towards the cameras, +Y = first camera's view direction
    on the floor, origin = floor point under the first camera's view centre."""
    n, d, inl = fit_plane(floor_pts)
    cam0 = m.c2w[0]
    if cam0[:3, 3] @ n + d < 0:
        n, d = -n, -d
    fwd = cam0[:3, 2] - (cam0[:3, 2] @ n) * n
    y = fwd / np.linalg.norm(fwd)
    x = np.cross(y, n)
    R = np.stack([x, y, n])  # rows: new axes in old coords
    # where the first camera's optical axis meets the floor
    o, dirn = cam0[:3, 3], cam0[:3, 2]
    t = -(o @ n + d) / (dirn @ n) if abs(dirn @ n) > 1e-6 else 0.0
    origin = o + t * dirn
    A = np.eye(4)
    A[:3, :3] = R
    A[:3, 3] = -R @ origin
    return A, {"floor_inliers": float(inl.mean()), "camera_height": float(cam0[:3, 3] @ n + d)}


def upright_box(p: np.ndarray, lo: float = 0.02, hi: float = 0.98):
    """Box standing on z = 0 around points p (aligned world): (centre, yaw, size)."""
    xy = p[:, :2] - np.median(p[:, :2], 0)
    evals, evecs = np.linalg.eigh(np.cov(xy.T))
    major = evecs[:, -1]
    yaw = float(np.arctan2(major[1], major[0]))
    c, s = np.cos(yaw), np.sin(yaw)
    local = p[:, :2] @ np.array([[c, -s], [s, c]])  # coordinates along (major, minor)
    a0, a1 = np.quantile(local, [lo, hi], axis=0)
    top = float(np.quantile(p[:, 2], hi))
    mid = (a0 + a1) / 2
    centre = np.array([c * mid[0] - s * mid[1], s * mid[0] + c * mid[1], top / 2])
    return centre, yaw, (float(a1[0] - a0[0]), float(a1[1] - a0[1]), max(top, 1e-3))


def _yaw_quat(yaw: float):
    return (float(np.cos(yaw / 2)), 0.0, 0.0, float(np.sin(yaw / 2)))


def initialize(m: Model4D, masks: np.ndarray, id_map: dict[int, str], width: int, height: int,
               shot_id: str, shot_start: int = 0, keep: float = 0.7, observation_id: str | None = None,
               fps: float = 24.0):
    """Returns (cameras, objects, scale, report). Labels: obj_floor, optional obj_wall, movers obs_*."""
    labels = labels_at_model_pixels(m, masks)
    by_name = {v: k for k, v in id_map.items()}
    floor_sel = labels == by_name["obj_floor"] if "obj_floor" in by_name else np.zeros(labels.shape, bool)
    source = "floor mask"
    if floor_sel.sum() < MIN_FLOOR_POINTS:
        # the floor prompt found nothing (px_6077402: "floor" on a basketball court): use the largest plane of
        # the background, which is the floor in a floor-level shot. Recorded in the report as a guess.
        movers = [k for k, v in id_map.items() if v.startswith("obs_")]
        floor_sel = ~np.isin(labels, movers)
        source = "background (no floor mask)"
        if floor_sel.sum() < MIN_FLOOR_POINTS:
            raise ValueError(f"no floor: {int(floor_sel.sum())} background points at model pixels")
    floor = _confident(m, floor_sel, keep)
    A, report = floor_frame(m, floor)
    report["floor_source"] = source
    report["frames"] = [int(f) for f in m.frames]

    def aligned(p):
        return p @ A[:3, :3].T + A[:3, 3]

    fx = float(np.mean(np.r_[m.K[:, 0, 0], m.K[:, 1, 1]]))
    report["fx_fy_rel_diff"] = float(np.mean(np.abs(m.K[:, 0, 0] - m.K[:, 1, 1]) / fx))
    intr = Intrinsics(width=width, height=height, fx=fx, fy=fx, cx=float(np.mean(m.K[:, 0, 2])), cy=float(np.mean(m.K[:, 1, 2])))
    keys = []
    for f, c2w in zip(m.frames, m.c2w):
        M = A @ c2w
        keys.append(CameraKey(shot_frame=int(f) - shot_start, pose=Pose(translation=tuple(float(v) for v in M[:3, 3]),
                                                                       rotation=matrix_to_quat(M[:3, :3].tolist()))))
    ev = [Evidence(observation_id=observation_id, shot_id=shot_id, source_frames=[int(f) for f in m.frames])]
    cameras = [Camera(id="cam_001", shot_id=shot_id, intrinsics=intr, keys=keys, evidence=ev)]

    fp = aligned(floor)
    if source != "floor mask":  # background points: keep those on the fitted plane
        fp = fp[np.abs(fp[:, 2]) < 0.02 * abs(report["camera_height"])]
    ext = np.quantile(fp[:, :2], [0.02, 0.98], axis=0)
    span = ext[1] - ext[0]
    objects = [SceneObject(id="obj_floor", kind="set", size=(float(span[0] * 1.5), float(span[1] * 1.5), 0.002),
                           rest=TransformKey(time=0.0, pose=Pose(translation=(float(ext.mean(0)[0]), float(ext.mean(0)[1]), -0.001))),
                           evidence=ev)]
    if "obj_wall" in by_name:
        wp = aligned(_confident(m, labels == by_name["obj_wall"], keep))
        if len(wp) > 100:
            yw = float(np.median(wp[:, 1]))  # wall facing the first camera: a plane y = const
            top = float(np.quantile(wp[:, 2], 0.98))
            objects.append(SceneObject(id="obj_wall", kind="set", size=(float(span[0] * 1.5), 0.002, max(top, 0.01)),
                                       rest=TransformKey(time=0.0, pose=Pose(translation=(float(ext.mean(0)[0]), yw, max(top, 0.01) / 2))),
                                       evidence=ev))
            report["wall_y"] = yw
    report["objects"] = {}
    floor_tol = 0.01 * report["camera_height"]
    for num, name in sorted(id_map.items()):
        if not name.startswith("obs_"):
            continue
        oid = "obj_" + name.removeprefix("obs_").zfill(2)
        p = _confident(m, labels == num, keep)
        if len(p) < 50:
            report["objects"][name] = {"skipped": f"{len(p)} points"}
            continue
        p = aligned(p)
        p = p[p[:, 2] > -floor_tol]  # drop points that fell through the floor
        centre, yaw, size = upright_box(p)
        # per sampled frame: a box from that frame's points only (the object may move)
        per = []
        for i, f in enumerate(m.frames):
            if _touches_border(masks[f] == num):  # cut off by the frame: the visible part's box drifts
                continue
            q = _confident_frame(m, i, labels[i] == num, keep)
            if len(q) < 30:
                continue
            q = aligned(q)
            q = q[q[:, 2] > -floor_tol]
            if len(q) >= 30:
                per.append((int(f), *upright_box(q)))
        moving = False
        if len(per) >= 2:
            centres = np.array([c for _, c, _, _ in per])
            spread = np.linalg.norm(centres[:, :2] - np.median(centres[:, :2], 0), axis=1).max()
            # compare with one frame's box: the box over all frames stretches over the whole path
            frame_size = np.median([max(s[:2]) for *_, s in per])
            moving = bool(spread > MOVE_FRACTION * frame_size)
        if moving:
            size = tuple(float(v) for v in np.median([s for *_, s in per], 0))
            fs = [f for f, *_ in per]
            xy = np.array([c[:2] for _, c, _, _ in per])
            yaws, prev = [], None
            for _, _, y, _ in per:
                if prev is not None:  # box axes are defined up to half a turn: stay close to the last key
                    y = prev + (y - prev + np.pi / 2) % np.pi - np.pi / 2
                prev = y
                yaws.append(y)
            fs, xy, yaws = _snap_still_ends(fs, xy, np.array(yaws))
            thin = size
            moved = [_deepen(c, float(y), thin) for c, y in zip(xy, yaws)]
            xy, size = np.array([c for c, _ in moved]), moved[0][1] if moved else size
            keys = [TransformKey(time=(f - shot_start) / fps,
                                 pose=Pose(translation=(float(c[0]), float(c[1]), size[2] / 2), rotation=_yaw_quat(float(y))))
                    for f, c, y in zip(fs, xy, yaws)]
            obj = SceneObject(id=oid, kind="rigid", size=size, rest=keys[0], keys=keys, evidence=ev)
        else:
            cxy, size = _deepen(np.asarray(centre[:2]), yaw, size)
            obj = SceneObject(id=oid, kind="rigid", size=size, evidence=ev,
                              rest=TransformKey(time=0.0, pose=Pose(translation=(float(cxy[0]), float(cxy[1]), float(centre[2])),
                                                                    rotation=_yaw_quat(yaw))))
        objects.append(obj)
        report["objects"][name] = {"id": oid, "points": int(len(p)), "size": [round(v, 4) for v in size],
                                   "moving": moving, "frames_with_box": len(per)}
    scale = Scale(source="estimated", note="4RC model units (first camera at the origin); not checked against a known size")
    return cameras, objects, scale, report
