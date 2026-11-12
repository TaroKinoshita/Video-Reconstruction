"""Estimate 2D layers from a video and per-layer masks (labelled index masks, as the 3D pipeline's masks).

Per layer: reference frame = the frame where its mask is largest; the sprite is cut out there. Points inside the
mask are tracked with Lucas-Kanade from the reference frame both ways; each frame's similarity transform
(position, rotation, uniform scale) is fitted to the tracked points (RANSAC). Opacity comes from the colour
difference to a static background (per-pixel median of uncovered frames), relative to the reference frame.
Then fit_keys turns the per-frame values into keys with easing.

Assumptions (recorded, not measured): the background is static (no camera motion: check with analysis.triage
first), the layer is flat and rigid, the reference frame shows it at full opacity and unrotated.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from vrecon.schema import Layer2D, LayerKey

from .model import CHANNELS, EASINGS, ease

# weights that put the channels on one scale (px): 1 deg ~ 1 px, 1% scale ~ 1 px, 1% opacity ~ 1 px
WEIGHTS = np.array([1.0, 1.0, 1.0, 100.0, 100.0])
MIN_AREA = 30


def static_background(frames: np.ndarray, covered: np.ndarray) -> np.ndarray:
    """Per-pixel median of the frames where the pixel is not covered by any mask."""
    f = frames.astype(np.float32)
    f[covered] = np.nan
    bg = np.nanmedian(f, axis=0)
    hole = np.isnan(bg)
    if hole.any():
        bg[hole] = np.median(frames.astype(np.float32), axis=0)[hole]
    return bg


def _similarity(m: np.ndarray) -> tuple[float, float]:
    """(scale, rotation deg counter-clockwise on screen) of a 2x3 similarity."""
    return math.hypot(m[0, 0], m[0, 1]), math.degrees(math.atan2(m[0, 1], m[0, 0]))


def _features(gray: np.ndarray, mask: np.ndarray, inner: int = 4, min_inner: int = 15):
    """Corners inside the layer, away from its outline when enough are there: an outline (a circle's edge)
    does not move when the layer turns, so points on it vote for 'no rotation'. Thin text keeps its edges."""
    m8 = mask.astype(np.uint8)
    core = cv2.erode(m8, np.ones((2 * inner + 1, 2 * inner + 1), np.uint8))
    pts = cv2.goodFeaturesToTrack(gray, 300, 0.01, 3, mask=core) if core.any() else None
    if pts is None or len(pts) < min_inner:
        pts = cv2.goodFeaturesToTrack(gray, 300, 0.01, 3, mask=cv2.dilate(m8, np.ones((3, 3), np.uint8)))
    return pts


def _refine_turn(g0: np.ndarray, g1: np.ndarray, m0: np.ndarray, step: np.ndarray, spread: float = 0.3,
                 n: int = 13) -> np.ndarray:
    """Point fits come out short on fast turns; try scaled turn angles about the layer centre and keep the one
    whose warped pixels match the next frame best (sum of squared differences on the layer's pixels)."""
    s, r = _similarity(step)
    if abs(r) < 1.0:
        return step
    ys, xs = np.nonzero(m0)
    c0 = np.array([xs.mean(), ys.mean()])
    c1 = step[:, :2] @ c0 + step[:, 2]
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    pts = np.stack([xs, ys], 1).astype(np.float32)
    v0 = g0[ys, xs].astype(np.float32)
    best, best_err = step, np.inf
    for k in np.linspace(1 - spread, 1 + spread, n):
        cand = cv2.getRotationMatrix2D((0.0, 0.0), r * k, s)
        cand[:, 2] = c1 - cand[:, :2] @ c0
        q = cv2.transform(pts[:, None], cand)[:, 0]
        ok = (q[:, 0] >= 0) & (q[:, 0] < g1.shape[1] - 1) & (q[:, 1] >= 0) & (q[:, 1] < g1.shape[0] - 1)
        if ok.mean() < 0.8:
            continue
        v1 = cv2.remap(g1.astype(np.float32), q[ok, 0][None], q[ok, 1][None], cv2.INTER_LINEAR)[0]
        err = float(np.mean((v1 - v0[ok]) ** 2))
        if err < best_err:
            best, best_err = cand, err
    return best


def _moments(mask: np.ndarray):
    ys, xs = np.nonzero(mask)
    return np.array([xs.mean(), ys.mean()]), float(len(xs))


def track_layer(frames: np.ndarray, gray: np.ndarray, mask: np.ndarray, bg: np.ndarray):
    """frames uint8 RGB [T,H,W,3], gray uint8 [T,H,W], mask bool [T,H,W] (this layer's visible pixels).

    Returns (values [T,5] with NaN where not visible, sprite RGBA float [h,w,4], info)."""
    t_n = len(frames)
    area = mask.reshape(t_n, -1).sum(1)
    vis = area >= MIN_AREA
    edge = mask[:, 0].any(1) | mask[:, -1].any(1) | mask[:, :, 0].any(1) | mask[:, :, -1].any(1)
    inner = vis & ~edge  # a layer cut by the frame edge would give a clipped sprite
    ref = int(np.argmax(np.where(inner, area, -1))) if inner.any() else int(np.argmax(area))
    ys, xs = np.nonzero(mask[ref])
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    c0 = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
    sprite = np.dstack([frames[ref, y0:y1, x0:x1].astype(np.float32) / 255.0,
                        mask[ref, y0:y1, x0:x1].astype(np.float32)])
    grow = cv2.dilate(mask[ref].astype(np.uint8), np.ones((3, 3), np.uint8))  # thin text strokes: keep edges
    p_ref = cv2.goodFeaturesToTrack(gray[ref], 300, 0.01, 3, mask=grow)
    p_ref = np.zeros((0, 1, 2), np.float32) if p_ref is None else p_ref.astype(np.float32)
    m_cent, m_area = _moments(mask[ref])

    d_ref = (frames[ref].astype(np.float32) - bg) * mask[ref][..., None]
    vals = np.full((t_n, 5), np.nan)
    mats: dict[int, np.ndarray] = {ref: np.array([[1.0, 0, 0], [0, 1.0, 0]])}
    fallback = 0
    h_, w_ = mask.shape[1:]

    def compose(step: np.ndarray, m: np.ndarray) -> np.ndarray:
        return (np.vstack([step, [0, 0, 1]]) @ np.vstack([m, [0, 0, 1]]))[:2]

    for direction in (1, -1):
        # frame to frame with fresh points each step (points carried over a fast spin drift off their corners),
        # composed into the reference -> frame transform
        t, m_t = ref, mats[ref]
        last = np.array([[1.0, 0, 0], [0, 1.0, 0]])  # previous step: the guess for this one (fast spins)
        while 0 <= t + direction < t_n:
            nt = t + direction
            if not vis[nt]:
                break
            step = None
            pts = _features(gray[t], mask[t])
            if pts is not None and len(pts) >= 6:
                pts = pts.astype(np.float32)
                dil = cv2.dilate(mask[nt].astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
                start = last
                for _ in range(3):  # LK from the guess, refit, repeat from the refit (fast spins converge short)
                    guess = cv2.transform(pts, start).astype(np.float32)
                    nxt, st, _ = cv2.calcOpticalFlowPyrLK(gray[t], gray[nt], pts, guess, winSize=(21, 21),
                                                          maxLevel=3, flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
                    q = np.round(nxt.reshape(-1, 2)).astype(int)
                    good = (st.ravel() == 1) & (q[:, 0] >= 0) & (q[:, 0] < w_) & (q[:, 1] >= 0) & (q[:, 1] < h_)
                    good[good] = dil[q[good, 1], q[good, 0]]
                    if good.sum() < 6:
                        break
                    fit, inl = cv2.estimateAffinePartial2D(pts[good], nxt[good], method=cv2.RANSAC,
                                                           ransacReprojThreshold=1.5)
                    if fit is None or inl.sum() < 6:
                        break
                    done = np.abs(fit - start).max() < 0.01
                    step = start = fit
                    if done:
                        break
            if step is not None:
                step = _refine_turn(gray[t], gray[nt], mask[t], step)
            if step is None:  # no points: position and size from the masks, rotation kept
                fallback += 1
                (c1, a1), (c2, a2) = _moments(mask[t]), _moments(mask[nt])
                step = np.array([[1.0, 0, 0], [0, 1.0, 0]]) * math.sqrt(a2 / a1)
                step[:, 2] = c2 - step[:, :2] @ c1
            last = step
            m_t = compose(step, m_t)
            mats[nt] = m_t
            t = nt
    # opacity: how far the layer's pixels are from the background, relative to the reference frame. A ratio of
    # mean contrasts, not a pixel-by-pixel comparison, so a small turn or shift error does not lower it
    core = {t: cv2.erode(mask[t].astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) for t in mats}
    contrast = {}
    for t in mats:
        sel = core[t] if core[t].sum() >= MIN_AREA else mask[t]
        contrast[t] = float(np.linalg.norm(frames[t][sel].astype(np.float32) - bg[sel], axis=1).mean())
    rel_op = {t: contrast[t] / max(contrast[ref], 1e-6) for t in mats}
    top = np.nanmax(list(rel_op.values())) if rel_op else 1.0
    for t, m in mats.items():
        s, r = _similarity(m)
        c = m[:, :2] @ c0 + m[:, 2]
        vals[t] = [c[0], c[1], r, s, min(rel_op[t] / top, 1.0) if np.isfinite(rel_op[t]) else np.nan]
    # rotation without wrap-around jumps
    seen = ~np.isnan(vals[:, 2])
    vals[seen, 2] = np.degrees(np.unwrap(np.radians(vals[seen, 2])))
    return vals, sprite, {"reference_frame": ref, "points": int(len(p_ref)), "fallback_frames": fallback,
                          "visible_frames": int(vis.sum())}


def fill_hidden(vals: np.ndarray) -> np.ndarray:
    """Frames without a mask: hold the nearest visible values; before the first / after the last visible frame
    the layer is not shown (opacity 0) unless it is visible at the clip's first / last frame."""
    v = vals.copy()
    seen = np.nonzero(~np.isnan(v[:, 0]))[0]
    if len(seen) == 0:
        return v
    for t in range(len(v)):
        if np.isnan(v[t, 0]):
            near = seen[np.argmin(np.abs(seen - t))]
            v[t] = v[near]
            if t < seen[0] or t > seen[-1]:
                v[t, 4] = 0.0
    v[:, 4] = np.nan_to_num(v[:, 4], nan=0.0)
    return v


HIDDEN_OPACITY = 0.15  # below this the masks do not show the layer


def _fit_one(v: np.ndarray, a: int, b: int, hidden: np.ndarray | None = None):
    """Best easing for one segment a..b (frames): (error, name, worst frame, start values, end values).

    Hidden frames (no mask) are not measured: an end of the segment that falls on one gets opacity 0 and the
    other values solved by least squares from the visible frames; inside the segment they only have to stay
    below HIDDEN_OPACITY."""
    hidden = np.zeros(len(v), bool) if hidden is None else hidden
    t = np.arange(a, b + 1)
    p = (t - a) / max(b - a, 1)
    seen = ~hidden[a:b + 1]
    best = (np.inf, "linear", (a + b) // 2, v[a], v[b])
    if seen.sum() < 2:
        return best
    for name in ("linear", "ease_in", "ease_out", "ease_in_out"):
        e = ease(name, p)
        va, vb = v[a].copy(), v[b].copy()
        ha, hb = bool(hidden[a]), bool(hidden[b])
        if ha or hb:
            ys = v[a:b + 1][seen]
            es = e[seen]
            for c in range(4):
                if ha and hb:
                    A = np.stack([1 - es, es], 1)
                    va[c], vb[c] = np.linalg.lstsq(A, ys[:, c], rcond=None)[0]
                elif ha:  # y - vb e = va (1 - e)
                    w_ = 1 - es
                    va[c] = float((w_ * (ys[:, c] - vb[c] * es)).sum() / max((w_ * w_).sum(), 1e-9))
                else:
                    vb[c] = float((es * (ys[:, c] - va[c] * (1 - es))).sum() / max((es * es).sum(), 1e-9))
            if ha:
                va[4] = 0.0
            if hb:
                vb[4] = 0.0
        pred = va + (vb - va) * e[:, None]
        err = (np.abs(pred - v[a:b + 1]) * WEIGHTS).max(1)
        err[~seen] = np.maximum(pred[~seen, 4] - HIDDEN_OPACITY, 0) * WEIGHTS[4]
        if err.max() < best[0]:
            best = (float(err.max()), name, int(a + np.argmax(np.where(seen, err, -1))), va, vb)
    return best


def _split(v, a, b, tol, hidden=None) -> list[tuple]:
    err, name, worst, va, vb = _fit_one(v, a, b, hidden)
    if err <= tol or b - a <= 2 or worst in (a, b):
        return [(a, b, name, va, vb)]
    return _merge(v, _split(v, a, worst, tol, hidden) + _split(v, worst, b, tol, hidden), tol, hidden)


def _merge(v, parts: list[tuple], tol: float, hidden=None) -> list[tuple]:
    """Join neighbours that one easing fits nearly as well (a split at the worst frame can cut one smooth move)."""
    changed = True
    while changed and len(parts) > 1:
        changed = False
        for i in range(len(parts) - 1):
            a, b = parts[i][0], parts[i + 1][1]
            err, name, _, va, vb = _fit_one(v, a, b, hidden)
            if err <= 1.5 * tol:
                parts = parts[:i] + [(a, b, name, va, vb)] + parts[i + 2:]
                changed = True
                break
    return parts


def moving_runs(v: np.ndarray, eps: float = 0.25, min_still: int = 3) -> list[tuple[int, int]]:
    """Runs of frames where something changes; each run [s, e] starts and ends at a still frame."""
    n = len(v)
    d = np.zeros(n)
    w = v * WEIGHTS
    d[1:-1] = np.abs(w[2:] - w[:-2]).max(1) / 2
    d[0], d[-1] = np.abs(w[1] - w[0]).max(), np.abs(w[-1] - w[-2]).max()
    step = np.r_[np.abs(np.diff(w, axis=0)).max(1), 0]  # jumps (pop-in) between t and t+1
    moving = (d > eps) | (step > 5 * eps)
    # still gaps shorter than min_still between moving frames count as motion
    idx = np.nonzero(moving)[0]
    for i, j in zip(idx, idx[1:]):
        if 1 < j - i <= min_still:
            moving[i:j] = True
    runs, t = [], 0
    while t < n:
        if not moving[t]:
            t += 1
            continue
        s = t
        while t < n and moving[t]:
            t += 1
        runs.append((max(s - 1, 0), min(t, n - 1)))
    return runs


def fit_keys(vals: np.ndarray, fps: float, tol: float = 2.0, slack: int = 6,
             hidden: np.ndarray | None = None) -> list[LayerKey]:
    """Per-frame values [T,5] (no NaN; hidden frames filled by fill_hidden) -> keys with easing.
    Still stretches become equal keys. hidden: frames without a mask (see _fit_one)."""
    v = vals
    n = len(v)
    hidden = np.zeros(n, bool) if hidden is None else hidden
    segs: list[tuple] = []
    for s, e in moving_runs(v):
        # an ease-in starts too gently to be seen as motion, and a fade starts while the layer is still hidden:
        # let the segment start / end move into the stills and through hidden frames
        back = 0
        while s - back - 1 >= 0 and hidden[s - back - 1]:
            back += 1
        fwd = 0
        while e + fwd + 1 < n and hidden[e + fwd + 1]:
            fwd += 1
        lo = max(s - slack - back, segs[-1][1] if segs else 0)
        hi = min(e + slack + fwd, n - 1)
        best = None
        # nearest to the detected run first; a farther start / end must fit clearly better (a still stretch
        # fits the flat start of an ease-in almost as well)
        cands = sorted(((s2, e2) for s2 in range(lo, s + 1) for e2 in range(e, hi + 1)),
                       key=lambda c: (s - c[0]) + (c[1] - e))
        for s2, e2 in cands:
            parts = _split(v, s2, e2, tol, hidden)
            cost = sum(_fit_one(v, p[0], p[1], hidden)[0] for p in parts) + tol * len(parts)
            if best is None or cost < best[0] - 0.1 * tol:
                best = (cost, parts)
        segs.extend(best[1])

    def k(t: int, vals_t, easing: str) -> LayerKey:
        x, y, r, sc, op = (float(c) for c in vals_t)
        return LayerKey(time=t / fps, x=x, y=y, rotation=r, scale=sc, opacity=min(max(op, 0.0), 1.0), easing=easing)

    keys: list[LayerKey] = []
    if not segs:
        return [k(0, v[0], "linear")]
    for a, b, name, va, vb in segs:
        if keys and abs(keys[-1].time - a / fps) < 1e-9:
            keys[-1] = keys[-1].model_copy(update={"easing": name})
        else:
            keys.append(k(a, va, name))
        keys.append(k(b, vb, "linear"))
    return keys


def estimate_layers(frames: np.ndarray, index: np.ndarray, id_map: dict[int, str], fps: float, tol: float = 2.0):
    """-> (layers, sprites id -> RGBA, per-frame values id -> [T,5], info id -> dict)."""
    gray = np.stack([cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames])
    bg = static_background(frames, index > 0)
    layers, sprites, series, infos = [], {}, {}, {}
    for lab, lid in sorted(id_map.items()):
        m = index == lab
        if m.reshape(len(m), -1).sum(1).max() < MIN_AREA:
            continue
        vals, spr, info = track_layer(frames, gray, m, bg)
        hidden = np.isnan(vals[:, 0])
        vals = fill_hidden(vals)
        keys = fit_keys(vals, fps, tol, hidden=hidden)
        lay = Layer2D(id=lid, size=(float(spr.shape[1]), float(spr.shape[0])), keys=keys,
                      evidence=[], status="estimated")
        layers.append(lay)
        sprites[lid], series[lid], infos[lid] = spr, vals, info
    order, net = depth_order(layers, sprites, index, id_map, len(frames), fps)
    layers = [lay.model_copy(update={"z": order[lay.id]}) for lay in layers]
    for lid in infos:
        infos[lid]["overlaps"] = {f"{a}|{b}": v for (a, b), v in net.items() if lid in (a, b)}
    return layers, sprites, series, infos


def depth_order(layers, sprites, index, id_map, n, fps) -> dict[str, int]:
    """z from overlaps: where two layers' sprites overlap, the one the masks show there is in front.
    Pairs that never overlap have no observable order. Returns (z per id, net votes per overlapping pair)."""
    from .model import layer_state, layer_matrix

    lab = {v: k for k, v in id_map.items()}
    h, w = index.shape[1:]
    net = {}  # (a, b) -> frames where the masks show a in front of b minus the reverse
    for f in range(0, n, 2):
        cover = {}
        for lay in layers:
            spr = sprites[lay.id]
            m = layer_matrix((spr.shape[1], spr.shape[0]), layer_state(lay, f / fps))
            cover[lay.id] = cv2.warpAffine(spr[..., 3], m, (w, h)) > 0.5
        for i, la in enumerate(layers):
            for lb in layers[i + 1:]:
                both = cover[la.id] & cover[lb.id]
                if both.sum() < 20:
                    continue
                d = int((index[f][both] == lab[la.id]).sum()) - int((index[f][both] == lab[lb.id]).sum())
                net[(la.id, lb.id)] = net.get((la.id, lb.id), 0) + int(np.sign(d))
    front = {lay.id: 0 for lay in layers}
    for (a, b), v in net.items():
        if v:
            win, lose = (a, b) if v > 0 else (b, a)
            front[win] += 1
            front[lose] -= 1
    ranked = sorted(front, key=lambda i: front[i])
    return {lid: z for z, lid in enumerate(ranked)}, net


__all__ = ["EASINGS", "CHANNELS", "estimate_layers", "fill_hidden", "fit_keys", "moving_runs", "static_background",
           "track_layer"]
