"""Small fast rigid things (a racket, a ball, a tool...): shape and motion apart, tools chosen by what the video
shows, blur used as an observation. Pure numpy/cv2.

Common base
    sharp_frames(video, masks)        frames where the thing is big, sharp and whole (not cut by the border)
    split_at_changes(xy)              segment boundaries where the 2D motion changes abruptly (a hit, a bounce)
    streak(mask)                      centre, long-axis end points, length, width of a mask (a blur streak or a shape)
What the video shows -> how its motion is found
    small, shape unreadable           ball_fit: centre and trajectory per segment (gravity, straight-line or held);
                                      rotation is left unknown ("estimated" only for the position)
    outline readable, held by a hand  held_poses: grip at a hand point, the far end on its pixel ray at the known
                                      length, roll from the short axis; in blurred frames the pose is the one whose
                                      swept silhouette over the shutter matches the observed streak (sweep_fit)
Every result says per frame how it was found ("sharp", "swept", "interpolated", "trajectory", "held") so the
scene can keep what was measured apart from what was assumed.
"""

from __future__ import annotations

import cv2
import numpy as np

G = 9.81


# ---------------------------------------------------------------- common base

def frame_times(video: np.ndarray, fps: float, ratio: float = 0.35, min_share: float = 0.7) -> np.ndarray:
    """Seconds per frame when frames repeat in a regular cadence (frame-rate padding, e.g. 24 -> 30 fps: every
    5th frame a copy of the one before). A frame is a repeat candidate when its difference from the previous one
    is under `ratio` times the median difference around it; repeats count only when at least `min_share` of the
    candidates fall on one phase of one period (4-6 frames): a nearly still shot has small differences everywhere
    but no cadence. Repeats get their predecessor's time; the other frames are spread evenly over the clip."""
    g = video.astype(np.int16)
    t_n = len(g)
    d = np.r_[np.inf, [float(np.abs(g[i] - g[i - 1]).mean()) for i in range(1, t_n)]]
    cand = []
    for i in range(1, t_n):
        lo, hi = max(1, i - 6), min(t_n, i + 7)
        med = float(np.median(d[lo:hi]))
        if d[i] < ratio * med:
            cand.append(i)
    rep = np.zeros(t_n, bool)
    if len(cand) >= 3:
        best = None
        for per in (4, 5, 6):
            phases = np.bincount(np.array(cand) % per, minlength=per)
            ph = int(np.argmax(phases))
            share = phases[ph] / len(cand)
            expected = len(range(ph, t_n, per))
            if share >= min_share and phases[ph] >= 0.5 * expected and (best is None or share > best[2]):
                best = (per, ph, share)
        if best:
            rep[[i for i in cand if i % best[0] == best[1]]] = True
    k = np.cumsum(~rep) - 1
    dur = t_n / fps
    return k * (dur / max(int((~rep).sum()), 1))


def mask_sharpness(gray: np.ndarray, mask: np.ndarray) -> float:
    """Variance of the Laplacian inside the mask (higher = sharper edges)."""
    if mask.sum() < 4:
        return 0.0
    lap = cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F)
    m = cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    return float(lap[m].var())


def sharp_frames(video: np.ndarray, masks: np.ndarray, top: int = 5, border: int = 2) -> list[dict]:
    """Rank frames by area * sharpness, skipping masks that touch the image border (cut off). video uint8 RGB
    [T,H,W,3] (any size; masks [T,h,w] bool are resized to it)."""
    t, h, w = masks.shape
    out = []
    for f in range(t):
        m = masks[f]
        a = int(m.sum())
        if a < 6:
            continue
        ys, xs = np.nonzero(m)
        if xs.min() < border or ys.min() < border or xs.max() >= w - border or ys.max() >= h - border:
            continue
        g = cv2.cvtColor(video[f], cv2.COLOR_RGB2GRAY)
        if g.shape != (h, w):
            g = cv2.resize(g, (w, h), interpolation=cv2.INTER_AREA)
        s = mask_sharpness(g, m)
        sk = streak(m)
        out.append({"frame": f, "area": a, "sharpness": round(s, 1), "elongation": round(sk["length"] / max(sk["width"], 1), 2),
                     "score": a * s})
    out.sort(key=lambda r: -r["score"])
    return out[:top]


def streak(mask: np.ndarray) -> dict:
    """Centre, long axis end points (pixels), length and width (2-sigma extents) of a mask."""
    ys, xs = np.nonzero(mask)
    if len(xs) < 2:
        c = (float(xs.mean()), float(ys.mean())) if len(xs) else (np.nan, np.nan)
        return {"centre": c, "ends": (c, c), "length": 0.0, "width": 0.0, "axis": (1.0, 0.0)}
    p = np.stack([xs, ys], 1).astype(float)
    c = p.mean(0)
    cov = np.cov((p - c).T)
    ev, evec = np.linalg.eigh(cov)
    ax = evec[:, 1]
    proj = (p - c) @ ax
    lo, hi = np.percentile(proj, [1, 99])
    wproj = (p - c) @ evec[:, 0]
    width = float(np.percentile(wproj, 99) - np.percentile(wproj, 1)) + 1
    return {"centre": (float(c[0]), float(c[1])), "ends": (tuple(c + lo * ax), tuple(c + hi * ax)),
            "length": float(hi - lo) + 1, "width": width, "axis": (float(ax[0]), float(ax[1]))}


def split_at_changes(xy: np.ndarray, min_jump: float = 3.0, min_gap: int = 3) -> list[int]:
    """Frames where the 2D velocity changes abruptly (acceleration well above its typical size): a hit or a
    bounce. xy [T,2] with NaN where unseen. Returns sorted boundary frames (the frame the new motion starts)."""
    v = np.diff(xy, axis=0)
    a = np.linalg.norm(np.diff(v, axis=0), axis=1)  # a[i] ~ change between v[i] and v[i+1] -> at frame i+1
    ok = np.isfinite(a)
    if ok.sum() < 3:
        return []
    typ = np.median(a[ok]) + 1e-6
    cand = [i + 1 for i in np.nonzero(ok & (a > max(min_jump, 6 * typ)))[0]]
    out = []
    for f in cand:
        if out and f - out[-1] < min_gap:
            if a[f - 1] > a[out[-1] - 1]:
                out[-1] = f
            continue
        out.append(f)
    return out


# ---------------------------------------------------------------- rays and trajectories

def pixel_ray(uv, r_c2w: np.ndarray, centre: np.ndarray, k: np.ndarray) -> np.ndarray:
    d = np.array([(uv[0] - k[0, 2]) / k[0, 0], (uv[1] - k[1, 2]) / k[1, 1], 1.0])
    d = r_c2w @ d
    return d / np.linalg.norm(d)


def _skew(d):
    return np.array([[0, -d[2], d[1]], [d[2], 0, -d[0]], [-d[1], d[0], 0]])


def fit_trajectory(obs: list[tuple[float, np.ndarray, np.ndarray]], model: str = "ballistic", g: float = G,
                   depths: list[tuple[float, np.ndarray, np.ndarray, float, float]] | None = None,
                   vpriors: list[tuple[float, np.ndarray, float]] | None = None):
    """Least squares p(t) on the rays: obs = [(time s, camera centre, unit ray)]. model: 'ballistic'
    (p0 + v t - g t^2/2 z), 'linear' (p0 + v t) or 'still' (p0). The constraint (p(t) - c) x d = 0 is linear in
    the unknowns. A still or straight path seen by one still camera has no depth from the rays alone (the camera
    centre itself fits them), so `depths` = [(time, centre, unit ray, depth along the ray from the apparent size,
    sigma)] add soft rows d.(p(t) - c) = depth. Returns (params, rms angle in radians between the rays and the
    fitted points, which a fit through the camera cannot game)."""
    t0 = obs[0][0]
    n = {"still": 3, "linear": 6, "ballistic": 6}[model]

    def solve(ray_w):
        rows, rhs = [], []
        for (t, c, d), w in zip(obs, ray_w):
            tt = t - t0
            S = _skew(d)
            A = np.zeros((3, n))
            A[:, :3] = S
            if n == 6:
                A[:, 3:] = S * tt
            b = S @ c
            if model == "ballistic":
                b = b + S @ np.array([0, 0, g * tt * tt / 2])
            rows.append(A * w)
            rhs.append(b * w)
        for t, c, d, z, sig in depths or []:
            tt = t - t0
            w = 1.0 / max(sig, 1e-3) * 0.05  # depth rows count as much as a ray row ~5 cm off at sigma 1 m
            A = np.zeros((1, n))
            A[0, :3] = d * w
            if n == 6:
                A[0, 3:] = d * tt * w
            bz = z + d @ c
            if model == "ballistic":
                bz = bz + d @ np.array([0, 0, g * tt * tt / 2])
            rows.append(A)
            rhs.append(np.array([bz * w]))
        # vpriors [(time, velocity, sigma m/s)]: the velocity at that time is known roughly (a released ball leaves
        # with the hand's velocity); rows count like a ray row ~5 cm off at sigma 1 m/s
        for t, vp, sig in (vpriors or []) if n == 6 else []:
            tt = t - t0
            w = 0.05 / max(sig, 1e-3)
            A = np.zeros((3, n))
            A[:, 3:] = np.eye(3) * w
            bv = np.asarray(vp, float) + (np.array([0, 0, g * tt]) if model == "ballistic" else 0)
            rows.append(A)
            rhs.append(bv * w)
        x, *_ = np.linalg.lstsq(np.concatenate(rows), np.concatenate(rhs), rcond=None)
        return {"t0": t0, "p0": x[:3], "v": x[3:6] if n == 6 else np.zeros(3), "model": model, "g": g}

    # a ray row's residual is a distance, which grows with depth: reweight by 1/depth (an angle) and resolve, so
    # far observations count like near ones and the fit is not pulled toward the camera
    prm = solve([1.0] * len(obs))
    for _ in range(4):
        dep = [max(float((trajectory_at(prm, t) - c) @ d), 0.3) for t, c, d in obs]
        prm = solve([1.0 / z for z in dep])
    res = []
    for t, c, d in obs:
        q = trajectory_at(prm, t) - c
        nq = np.linalg.norm(q)
        res.append(np.pi if nq < 1e-6 or q @ d <= 0 else np.linalg.norm(np.cross(q, d)) / nq)
    return prm, float(np.sqrt(np.mean(np.square(res))))


def trajectory_at(prm: dict, t: float) -> np.ndarray:
    tt = t - prm["t0"]
    p = prm["p0"] + prm["v"] * tt
    if prm["model"] == "ballistic":
        p = p - np.array([0, 0, prm["g"] * tt * tt / 2])
    return p


def ball_fit(centres: np.ndarray, cams: list, fps: float, shutter: float = 0.0, ends=None, min_obs: int = 3,
             splits: list[int] | None = None, floor_z: float | None = 0.0, widths=None, diameter: float | None = None,
             anchors: dict | None = None, models=("still", "ballistic", "linear"), times=None,
             vpriors: dict | None = None):
    """Ball positions per frame from its 2D centres [T,2] (NaN = unseen) and cameras [(R, c, K)] per frame.
    Segments are split where the 2D motion changes (split_at_changes) or at `splits`; each is fitted with
    the model that explains it best (still / ballistic / linear, by rms on the rays, preferring the simpler).
    Blur: with `ends` [T,2,2] (streak end points) and the shutter (frames open), the ends count as the
    positions at t -/+ shutter/2 (direction chosen to agree with the motion). Frames outside every segment's
    seen span stay NaN; frames inside but unseen are 'interpolated' by the fit. Returns (pos [T,3], how [T],
    segments info). With `widths` [T] (apparent diameter in pixels: a streak's width, which blur does not
    stretch) and the real `diameter` (m), each seen frame adds a depth f*diameter/width (sigma from 1 px of width).
    Model choice is by rms angle: 'still' when under 1 px, else ballistic unless linear is clearly better."""
    t_n = len(centres)
    # times (s) per frame: a video with repeated frames (pulldown) gives two frames the same moment
    T = np.arange(t_n) / fps if times is None else np.asarray(times, float)
    seen = np.all(np.isfinite(centres), axis=1)
    bounds = sorted(set(split_at_changes(centres) if splits is None else splits))
    edges = [0] + [b for b in bounds if 0 < b < t_n] + [t_n]
    pos = np.full((t_n, 3), np.nan)
    how = np.array(["unseen"] * t_n, dtype=object)
    info = []
    for a, b in zip(edges[:-1], edges[1:]):
        fr = [f for f in range(a, b) if seen[f]]
        fr = [f for i, f in enumerate(fr) if i == 0 or T[f] > T[fr[i - 1]] + 1e-6]  # a repeated frame adds nothing
        if len(fr) < min_obs:
            continue
        obs = []
        for f in fr:
            R, c, K = cams[f]
            obs.append((T[f], c, pixel_ray(centres[f], R, c, K)))
        use_blur = ends is not None and shutter > 0
        if use_blur:
            for f in fr:
                e = ends[f]
                # a streak only when it is longer than the ball is wide (a still ball's extent is not motion)
                if not np.all(np.isfinite(e)) or np.linalg.norm(e[1] - e[0]) < 3 or (
                        widths is not None and np.isfinite(widths[f]) and np.linalg.norm(e[1] - e[0]) < widths[f] + 2):
                    continue
                R, c, K = cams[f]
                nxt = centres[min(f + 1, t_n - 1)] if seen[min(f + 1, t_n - 1)] else centres[f]
                e0, e1 = (e[0], e[1]) if np.dot(e[1] - e[0], nxt - centres[f]) >= 0 else (e[1], e[0])
                obs.append((T[f] - shutter / 2 / fps, c, pixel_ray(e0, R, c, K)))
                obs.append((T[f] + shutter / 2 / fps, c, pixel_ray(e1, R, c, K)))
            obs.sort(key=lambda o: o[0])
        deps = []
        if widths is not None and diameter:
            for f in fr:
                wpx = widths[f]
                if np.isfinite(wpx) and wpx >= 2:
                    R, c, K = cams[f]
                    z = K[0, 0] * diameter / wpx
                    deps.append((T[f], c, pixel_ray(centres[f], R, c, K), z, z / wpx))
        # gravity fixes a falling path's scale on its own and the size is the weaker cue (a few pixels wide), so
        # the size depths go only to the paths the rays leave without depth (still, straight)
        # anchors {frame: (depth along the ray, sigma)}: the ball known to be at another object then (in the hand
        # at the toss); strong rows that every model uses
        anc = []
        for f in fr:
            if anchors and f in anchors:
                R, c, K = cams[f]
                anc.append((T[f], c, pixel_ray(centres[f], R, c, K), anchors[f][0], anchors[f][1]))
        # a short arc (a few frames) leaves gravity too little time to fix the depth: then the size depths help it too
        short = len(fr) < 8
        vp = [(T[f], vpriors[f][0], vpriors[f][1]) for f in fr if vpriors and f in vpriors]
        fits = {m: fit_trajectory(obs, m, depths=((deps if short else []) + anc if m == "ballistic" else deps + anc) or None,
                                  vpriors=vp or None)
                for m in models}
        px = 1.0 / cams[fr[0]][2][0, 0]  # one pixel as an angle
        moving = [m for m in ("ballistic", "linear") if m in fits]
        best = "still" if "still" in fits and (fits["still"][1] < px or not moving) else min(
            moving, key=lambda m: fits[m][1] * (1.0 if m == "ballistic" else 1.3))
        prm, rms = fits[best]
        lo, hi = fr[0], fr[-1]
        for f in range(lo, hi + 1):
            p = trajectory_at(prm, T[f])
            if floor_z is not None and p[2] < floor_z:
                p = p.copy()
                p[2] = floor_z
            pos[f] = p
            how[f] = "trajectory" if seen[f] else "interpolated"
        info.append({"frames": [lo, hi], "model": best, "rms_px": round(rms / px, 2), "observations": len(obs), "depth_rows": len(deps), "anchors": len(anc),
                     "blur_ends": use_blur, "v": [round(float(x), 2) for x in prm["v"]],
                     "fit": {"t0": float(prm["t0"]), "p0": [float(x) for x in prm["p0"]], "v_exact": [float(x) for x in prm["v"]], "model": best, "g": prm["g"]}})
    return pos, how, info


# ---------------------------------------------------------------- held rigid shape

def ray_sphere(c: np.ndarray, d: np.ndarray, centre: np.ndarray, r: float):
    """Points where the ray c + s d (s > 0) meets the sphere; [] if it misses."""
    oc = c - centre
    b = d @ oc
    disc = b * b - (oc @ oc - r * r)
    if disc < 0:
        return []
    s = [-b - np.sqrt(disc), -b + np.sqrt(disc)]
    return [c + x * d for x in s if x > 0]


def _box_corners(grip, tip, side, length, width, thick):
    ax = (tip - grip) / np.linalg.norm(tip - grip)
    sd = side - (side @ ax) * ax
    sd = sd / (np.linalg.norm(sd) + 1e-9)
    nz = np.cross(ax, sd)
    out = []
    for a in (0.0, length):
        for s in (-width / 2, width / 2):
            for z in (-thick / 2, thick / 2):
                out.append(grip + a * ax + s * sd + z * nz)
    return np.array(out)


def silhouette(corners: np.ndarray, cam, size) -> np.ndarray:
    R, c, K = cam
    pc = (corners - c) @ R
    if np.any(pc[:, 2] <= 1e-3):
        return np.zeros(size[::-1], bool)
    uv = pc[:, :2] / pc[:, 2:3] * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]
    img = np.zeros(size[::-1], np.uint8)
    hull = cv2.convexHull(uv.astype(np.float32))
    cv2.fillConvexPoly(img, np.round(hull).astype(np.int32), 1)
    return img > 0


def iou(a: np.ndarray, b: np.ndarray) -> float:
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 0.0


def _slerp_dir(a, b, s):
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    om = np.arccos(np.clip(a @ b, -1, 1))
    if om < 1e-6:
        return a
    return (np.sin((1 - s) * om) * a + np.sin(s * om) * b) / np.sin(om)


def _rotate(v, axis, a):
    axis = axis / np.linalg.norm(axis)
    return v * np.cos(a) + np.cross(axis, v) * np.sin(a) + axis * (axis @ v) * (1 - np.cos(a))


def _refine_held(mask, grip, dirn, side, cam, length, width, thick, size, steps=(8, 4, 2, 1)):
    """Pattern search on the silhouette IoU over the tip direction (two tilts) and the roll about it."""
    dirn = dirn / np.linalg.norm(dirn)

    def score(d, sd):
        return iou(silhouette(_box_corners(grip, grip + d * length, sd, length, width, thick), cam, size), mask)

    best = score(dirn, side)
    for st in steps:
        a = np.radians(st)
        moved = True
        while moved:
            moved = False
            u = np.cross(dirn, side)
            for kind, sign in ((0, 1), (0, -1), (1, 1), (1, -1), (2, 1), (2, -1)):
                if kind == 0:
                    d, sd = _rotate(dirn, side, sign * a), side
                elif kind == 1:
                    d, sd = _rotate(dirn, u, sign * a), _rotate(side, u, sign * a)
                else:
                    d, sd = dirn, _rotate(side, dirn, sign * a)
                sc = score(d, sd)
                if sc > best + 1e-4:
                    best, dirn, side, moved = sc, d / np.linalg.norm(d), sd, True
                    break
    return dirn, side, best


def held_poses(masks: np.ndarray, grips: np.ndarray, cams: list, length: float, width: float, thick: float,
               shutter: float = 0.5, sharp_min_iou: float = 0.5, n_sub: int = 5, size=None):
    """Poses (grip, tip direction, side direction) per frame of a straight rigid shape held at `grips` [T,3].
    Sharp frames (the mask is no longer than the shape could look): the far end is the mask end farthest from
    the grip's pixel, put on its ray at `length` from the grip (the solution nearer the previous tip); the side
    direction follows the mask's short axis. Blurred frames (the mask is longer/wider than one pose explains):
    between the neighbouring sharp poses, the pose at frame f is slerp(A, B, s); s is chosen per frame so the
    shape swept over the shutter (n_sub sub-times) best matches the observed streak. Returns (tips [T,3],
    sides [T,3], how [T], scores [T])."""
    t_n, h, w = masks.shape
    size = size or (w, h)
    tips = np.full((t_n, 3), np.nan)
    sides = np.full((t_n, 3), np.nan)
    how = np.array(["unseen"] * t_n, dtype=object)
    score = np.zeros(t_n)
    prev_tip = None
    for f in range(t_n):
        m = masks[f]
        if m.sum() < 8 or not np.all(np.isfinite(grips[f])):
            continue
        R, c, K = cams[f]
        sk = streak(m)
        g_uv = ((grips[f] - c) @ R)
        g_uv = g_uv[:2] / g_uv[2] * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]
        e = np.array(sk["ends"])
        far = e[np.argmax(np.linalg.norm(e - g_uv, axis=1))]
        cands = ray_sphere(c, pixel_ray(far, R, c, K), grips[f], length)
        if not cands:
            continue
        tip = min(cands, key=lambda p: np.linalg.norm(p - prev_tip)) if prev_tip is not None else min(cands, key=lambda p: np.linalg.norm(p - c))
        ax2 = np.array(sk["axis"])
        minor = np.array([-ax2[1], ax2[0]])
        mid = (grips[f] + tip) / 2
        mid_uv = ((mid - c) @ R)
        mid_uv = mid_uv[:2] / mid_uv[2] * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]
        side_ray = pixel_ray(mid_uv + minor * 10, R, c, K)
        side = np.cross(np.cross(tip - grips[f], side_ray), tip - grips[f])
        side = side / (np.linalg.norm(side) + 1e-9)
        # the mask's end is only an outline point; refine the pose on the whole silhouette, from each tip
        # candidate (the two ray-sphere solutions differ in depth, which only the outline's shape tells apart)
        best = None
        starts = list(cands)
        if prev_tip is not None:  # also from the previous pose: the outline alone can fit a mirrored pose as well
            pd = prev_tip - grips[max(f - 1, 0)]
            starts.append(grips[f] + pd / np.linalg.norm(pd) * length)
        for t0 in starts:
            sd0 = np.cross(np.cross(t0 - grips[f], side_ray), t0 - grips[f])
            sd0 = sd0 / (np.linalg.norm(sd0) + 1e-9)
            r = _refine_held(m, grips[f], t0 - grips[f], sd0, cams[f], length, width, thick, size)
            # a box held at one end looks alike pointing either way; prefer the pose near the previous one
            turn = 0.0
            if prev_tip is not None:
                pd = (prev_tip - grips[max(f - 1, 0)])
                turn = np.degrees(np.arccos(np.clip(r[0] @ pd / np.linalg.norm(pd), -1, 1)))
            key = r[2] - 0.002 * turn
            if best is None or key > best[3]:
                best = (*r, key)
        dirn, side, sc, _ = best
        tip = grips[f] + dirn * length
        tips[f], sides[f], score[f] = tip, side, sc
        how[f] = "sharp" if sc >= sharp_min_iou else "blurred"
        prev_tip = tip
    # blurred stretches: sweep fit between sharp neighbours
    sharp = [f for f in range(t_n) if how[f] == "sharp"]
    for f in range(t_n):
        if how[f] != "blurred":
            continue
        a = max([s for s in sharp if s < f], default=None)
        b = min([s for s in sharp if s > f], default=None)
        if a is None or b is None:
            continue
        da, db = tips[a] - grips[a], tips[b] - grips[b]
        best = (-1.0, None)
        lin = (f - a) / (b - a)
        for s in np.linspace(0, 1, 41):
            sil = np.zeros(masks[f].shape, bool)
            for u in np.linspace(-shutter / 2, shutter / 2, n_sub):
                su = np.clip(s + u / (b - a), 0, 1)
                dirn = _slerp_dir(da, db, su)
                gp = grips[f]
                tp = gp + dirn * length
                sd = _slerp_dir(sides[a], sides[b], su)
                sil |= silhouette(_box_corners(gp, tp, sd, length, width, thick), cams[f], size)
            sc = iou(sil, masks[f]) - 0.05 * abs(s - lin)
            if sc > best[0]:
                best = (sc, s)
        s = best[1]
        tips[f] = grips[f] + _slerp_dir(da, db, s) * length
        sides[f] = _slerp_dir(sides[a], sides[b], s)
        how[f], score[f] = "swept", best[0]
    return tips, sides, how, score


__all__ = ["frame_times", "ball_fit", "fit_trajectory", "held_poses", "iou", "mask_sharpness", "pixel_ray", "ray_sphere",
           "sharp_frames", "silhouette", "split_at_changes", "streak", "trajectory_at"]
