"""Point tracks from the reference video (no learned weights) and a track-based loss.

Tracking: OpenCV pyramidal Lucas-Kanade from Shi-Tomasi corners, seeded every `reseed` frames,
kept only when tracking forward then backward returns to within `fb_max` px (a standard check).

Loss: at a track's first frame, its pixel is cast as a ray into the reconstruction's own proxy boxes;
the hit point is pinned to that box in its local coordinates, then carried by the box's pose and
re-projected with the reconstruction's camera at later frames. The error against where the video
says the point went measures camera motion, focal length and object motion together, without masks.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from vrecon.control.core import read_video_rgb
from vrecon.metrics.poses import camera_pose, object_pose
from vrecon.schema import Observation, Project, Provenance


@dataclass
class Tracks:
    xy: np.ndarray  # float32 [N, T, 2], NaN where not tracked
    size: tuple[int, int]  # (width, height)
    observation: Observation


def _frames(video: str | Path) -> list[np.ndarray]:
    out = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in read_video_rgb(video)]
    if not out:
        raise ValueError(f"no frames read from {video}")
    return out


def track_points(video: str | Path, max_corners: int = 300, reseed: int = 12, span: int = 24,
                 fb_max: float = 1.0, obs_id: str = "obs_tracks_klt") -> Tracks:
    frames = _frames(video)
    t, (h, w) = len(frames), frames[0].shape
    lk = {"winSize": (21, 21), "maxLevel": 3,
          "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)}
    tracks = []
    for s in range(0, t - 1, reseed):
        pts = cv2.goodFeaturesToTrack(frames[s], max_corners, 0.01, 8)
        if pts is None:
            continue
        cur = pts.reshape(-1, 2).astype(np.float32)
        xy = np.full((len(cur), t, 2), np.nan, np.float32)
        xy[:, s] = cur
        alive = np.ones(len(cur), bool)
        for f in range(s + 1, min(s + span, t)):
            nxt, st, _ = cv2.calcOpticalFlowPyrLK(frames[f - 1], frames[f], cur, None, **lk)
            back, st2, _ = cv2.calcOpticalFlowPyrLK(frames[f], frames[f - 1], nxt, None, **lk)
            good = (st.ravel() == 1) & (st2.ravel() == 1) & (np.linalg.norm(back - cur, axis=1) < fb_max)
            good &= (nxt[:, 0] >= 0) & (nxt[:, 0] < w) & (nxt[:, 1] >= 0) & (nxt[:, 1] < h)
            alive &= good
            if not alive.any():
                break
            xy[alive, f] = nxt[alive]
            cur = nxt
        keep = (~np.isnan(xy[:, :, 0])).sum(1) >= 4
        tracks.append(xy[keep])
    xy = np.concatenate(tracks) if tracks else np.zeros((0, t, 2), np.float32)
    obs = Observation(
        id=obs_id, kind="point_track", shot_id="shot_001",
        summary={"tracks": int(len(xy)), "frames": t},
        provenance=Provenance(tool="opencv_lk", tool_version=cv2.__version__,
                              params={"max_corners": max_corners, "reseed": reseed, "span": span, "fb_max": fb_max}))
    return Tracks(xy=xy, size=(w, h), observation=obs)


def load_tracks(path: str | Path, obs_id: str = "obs_tracks_file") -> Tracks:
    """Tracks written by an external tracker (gpu/observe/run_tapnextpp.py), which runs in its own env."""
    with np.load(path) as d:
        xy, (w, h), meta = d["xy"].astype(np.float32), d["size"].tolist(), json.loads(str(d["meta"]))
    obs = Observation(id=obs_id, kind="point_track", shot_id="shot_001",
                      summary={"tracks": int(len(xy)), "frames": int(xy.shape[1])},
                      provenance=Provenance(tool=meta["tool"], params=meta))
    return Tracks(xy=xy, size=(w, h), observation=obs)


_NORMALS = np.eye(3)


def _ray_boxes(origin, dirs, boxes):
    """Nearest hit of rays (origin [3], dirs [N,3]) with oriented boxes [(R, t, half)] -> (box index, local pt)."""
    n = len(dirs)
    best_t = np.full(n, np.inf)
    best_i = np.full(n, -1)
    best_p = np.zeros((n, 3))
    for i, (r, t, half) in enumerate(boxes):
        o = (origin - t) @ r  # into box frame
        d = dirs @ r
        with np.errstate(divide="ignore", invalid="ignore"):
            t1 = (-half - o) / d
            t2 = (half - o) / d
        tmin = np.nanmax(np.minimum(t1, t2), axis=1)
        tmax = np.nanmin(np.maximum(t1, t2), axis=1)
        hit = (tmax >= tmax.clip(min=0)) & (tmax >= tmin) & (tmax > 1e-6)
        tt = np.where(tmin > 1e-6, tmin, tmax)
        better = hit & (tt < best_t)
        best_t[better] = tt[better]
        best_i[better] = i
        best_p[better] = o + tt[better, None] * d[better]
    return best_i, best_p


def track_metrics(project: Project, tracks: Tracks, shot_id: str = "shot_001", ids: list[str] | None = None,
                  horizon: int = 12, max_tracks: int = 400) -> dict:
    """Mean re-projection error of tracked points (fraction of the image diagonal).

    ids: which reconstruction objects tracks may pin to (default: all). Restricting to set pieces gives
    a camera-only signal. Tracks that hit nothing are ignored and counted.
    """
    w, h = tracks.size
    diag = float(np.hypot(w, h))
    objs = {o.id: o for o in project.reconstruction.objects}
    ids = ids or list(objs)
    cam = next(c for c in project.reconstruction.cameras if c.shot_id == shot_id)
    xy = tracks.xy[:max_tracks]
    starts = np.argmax(~np.isnan(xy[:, :, 0]), axis=1)
    errs, missed = [], 0
    for s in np.unique(starts):
        sel = np.nonzero(starts == s)[0]
        r_c, c_c, k = camera_pose(cam, int(s))
        uv = xy[sel, s]
        dirs = np.stack([(uv[:, 0] - k[0, 2]) / k[0, 0], (uv[:, 1] - k[1, 2]) / k[1, 1], np.ones(len(sel))], 1) @ r_c.T
        # cast against every object and keep only rays whose nearest hit is one of `ids`: otherwise a point
        # on a moving box could pass through it and be pinned to the wall behind it
        order = ids + [o for o in objs if o not in ids]
        boxes = []
        for oid in order:
            r, t, sc = object_pose(project, objs[oid], shot_id, int(s))
            boxes.append((r, t, np.array(objs[oid].size or (1, 1, 1)) * sc / 2))
        which, local = _ray_boxes(c_c, dirs, boxes)
        missed += int((which < 0).sum())
        which[which >= len(ids)] = -1  # nearest hit is an object outside `ids`: not counted as missed
        for f in range(int(s) + 1, min(int(s) + horizon, xy.shape[1])):
            seen = ~np.isnan(xy[sel, f, 0]) & (which >= 0)
            if not seen.any():
                continue
            r_c, c_c, k = camera_pose(cam, f)
            for bi in np.unique(which[seen]):
                m = seen & (which == bi)
                r, t, sc = object_pose(project, objs[ids[bi]], shot_id, f)
                old_half = boxes[bi][2]
                new_half = np.array(objs[ids[bi]].size or (1, 1, 1)) * sc / 2
                pw = (local[m] / old_half * new_half) @ r.T + t
                pc = (pw - c_c) @ r_c
                ok = pc[:, 2] > 1e-3
                proj = pc[ok, :2] / pc[ok, 2:3] * [k[0, 0], k[1, 1]] + [k[0, 2], k[1, 2]]
                errs.extend((np.linalg.norm(proj - xy[sel[m]][ok, f], axis=1) / diag).tolist())
    return {"track_loss": float(np.mean(errs)) if errs else 1.0, "residuals": len(errs), "unpinned": missed}
