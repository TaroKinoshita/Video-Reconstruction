"""Looking at the video closely (docs/VENUE_PLAN.md, "how the LLM sees the video"): coarse first, then zoom in
time and space where it matters, then check with numbers. The shape follows VideoAgent (a few frames first,
then targeted ones) and the 2025-26 video agents with temporal and spatial zoom tools (VideoZoomer, LongVT,
VideoThinker); TemporalBench's finding that more frames alone help little is why the machine proposes where to
look (events) and the numbers confirm what the eye suggests (track).

    overview(run_dir, n)                   n numbered frames over the whole clip + a motion strip per object
    events(run_dir)                        machine-found moments: starts, stops, fast turns, appear/disappear,
                                           occlusion, picture changes; frame ranges to zoom into
    clip(run_dir, start, end, ...)         consecutive frames at the video's own rate, numbered, optionally cropped
                                           to an object or a box (spatial zoom) and enlarged
    compare(run_dir, frames, ...)          reference next to the reconstruction's silhouettes on the same frames,
                                           same crop options
    track(run_dir, obj, start, end)        numbers per frame: observed mask centroid and area, the
                                           reconstruction's silhouette centroid, the offset; observed and
                                           reconstructed start/stop frames
Only the reference video, its masks and the adopted scene are used: no ground truth. Pictures are written under
<run_dir>/look/ and returned as paths.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

_COLOURS = [(255, 80, 80), (80, 200, 255), (255, 200, 40), (120, 255, 120), (220, 120, 255), (255, 150, 200)]


def _ctx(run_dir):
    from .look import _video
    from .service import _config, _manager

    rd = Path(run_dir)
    cfg = _config(rd)
    mgr, obs, _ = _manager(rd)
    video = _video(cfg)
    t = min(obs.masks.shape[0], len(video))
    return rd, cfg, mgr, obs, video, t


def _save(rd: Path, name: str, img: np.ndarray) -> str:
    import cv2

    out = rd / "look" / name
    out.parent.mkdir(exist_ok=True)
    ok, buf = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))  # not imwrite: non-ASCII paths
    if not ok:
        raise RuntimeError("png encode failed")
    buf.tofile(str(out))
    return str(out)


def _label(img, text, colour=(255, 255, 0)):
    import cv2

    cv2.rectangle(img, (0, 0), (8 + 11 * len(text), 22), (0, 0, 0), -1)
    cv2.putText(img, text, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)
    return img


def _grid(cells, cols):
    h, w = cells[0].shape[:2]
    rows = [np.concatenate(cells[i:i + cols] + [np.zeros_like(cells[0])] * (cols - len(cells[i:i + cols])), 1)
            for i in range(0, len(cells), cols)]
    return np.concatenate(rows, 0)


def _ref_frame(video, f, h, w):
    import cv2

    return cv2.resize(video[f], (w, h), interpolation=cv2.INTER_AREA) if video[f].shape[:2] != (h, w) else video[f].copy()


def camera_shift(video: np.ndarray, masks: np.ndarray, movers: list[int], t: int) -> np.ndarray:
    """Cumulative picture shift (x, y) per frame from the background (movers masked out), by phase correlation of
    consecutive frames: a pan or tilt moves still things on the picture (synthetic clips pan while the camera
    stays put; a still object's mask "moved" 90 px). Subtracting it gives motion relative to the background."""
    import cv2

    th, tw = masks.shape[1:]
    shift = np.zeros((t, 2))
    win = cv2.createHanningWindow((tw, th), cv2.CV_32F)
    prev = None
    for f in range(t):
        g = cv2.cvtColor(cv2.resize(video[f], (tw, th)), cv2.COLOR_RGB2GRAY).astype(np.float32)
        mv = np.isin(masks[f], movers)
        g[mv] = g[~mv].mean() if (~mv).any() else g.mean()
        if prev is not None:
            (dx, dy), resp = cv2.phaseCorrelate(prev, g, win)
            shift[f] = shift[f - 1] + (dx, dy) if resp > 0.05 else shift[f - 1]
        prev = g
    return shift


def object_tracks(obs, t: int, shift: np.ndarray | None = None) -> dict:
    """Per observed label: centroid (x, y) and area per frame (NaN when absent); with `shift` (camera_shift) the
    centroid is relative to the background."""
    out = {}
    for k, name in obs.id_map.items():
        c = np.full((t, 2), np.nan)
        a = np.zeros(t)
        for f in range(t):
            v, u = np.nonzero(obs.masks[f] == k)
            if len(u):
                c[f] = (u.mean(), v.mean())
                a[f] = len(u)
        if shift is not None:
            c = c - shift[:t]
        out[name] = {"label": int(k), "centroid": c, "area": a}
    return out


def _speed(c: np.ndarray, smooth: int = 2) -> np.ndarray:
    v = np.full(len(c), np.nan)
    d = np.linalg.norm(np.diff(c, axis=0), axis=1)
    v[1:] = d
    if smooth:
        k = np.ones(2 * smooth + 1) / (2 * smooth + 1)
        ok = np.isfinite(v)
        vv = np.where(ok, v, 0.0)
        num = np.convolve(vv, k, mode="same")
        den = np.convolve(ok.astype(float), k, mode="same")
        v = np.where(den > 0, num / np.maximum(den, 1e-9), np.nan)
    return v


def onsets(c: np.ndarray, min_px: float = 2.0, share: float = 0.05, area: np.ndarray | None = None):
    """(start, stop) of one move: the first frame the centroid is further than max(min_px, share * travel) from
    where it began, and the last frame it is that far from where it ends. Robust to mask-edge jitter (a speed
    threshold read the jitter of a still object as motion, as the onset metric did, docs/DIRECTION_OPTIONS.md)."""
    ok = np.all(np.isfinite(c), axis=1)
    if area is not None and ok.any():
        # frames where the mask lost or gained over a quarter of its usual area (something passed in front, or
        # it left the picture) shift the centroid without the object moving: left out
        med = np.median(area[ok])
        ok &= np.abs(area - med) < 0.25 * med
    if ok.sum() < 3:
        return None, None
    idx = np.nonzero(ok)[0]
    first, last = c[idx[:3]].mean(0), c[idx[-3:]].mean(0)
    travel = float(np.linalg.norm(last - first))
    thr = max(min_px, share * travel)
    d0 = np.linalg.norm(c - first, axis=1)
    d1 = np.linalg.norm(c - last, axis=1)
    if travel < 2 * thr:
        return None, None
    start = int(idx[np.argmax(d0[idx] > thr)])
    stop = int(idx[len(idx) - 1 - np.argmax(d1[idx][::-1] > thr)]) + 1
    return start, min(stop, int(idx[-1]))


def find_events(tracks: dict, frame_diff: np.ndarray, turn_deg: float = 60.0, min_speed: float = 1.5,
                skip=("obj_floor", "obj_wall")) -> list[dict]:
    """Moments worth a closer look, from mask tracks and the picture's frame-to-frame change. Set pieces (floor,
    wall) are skipped: their mask centroids move whenever something covers them."""
    ev = []
    for name, tr in tracks.items():
        if name.startswith(skip):
            continue
        c, a = tr["centroid"], tr["area"]
        present = a > 0
        st, sp_ = onsets(c, area=a)
        if st is not None:
            ev.append({"frame": st, "object": name, "kind": "starts moving", "range": [st - 6, st + 4]})
            ev.append({"frame": sp_, "object": name, "kind": "stops", "range": [sp_ - 4, sp_ + 6]})
        # sharp turns while clearly moving (direction over 3-frame steps)
        sp = _speed(c, smooth=1)
        for f in range(3, len(c) - 3):
            if np.nan_to_num(sp[f]) < min_speed or not np.all(np.isfinite(c[f - 3:f + 4])):
                continue
            d1, d2 = c[f] - c[f - 3], c[f + 3] - c[f]
            if np.linalg.norm(d1) < 3 * min_speed or np.linalg.norm(d2) < 3 * min_speed:
                continue
            ang = np.degrees(np.arccos(np.clip(d1 @ d2 / (np.linalg.norm(d1) * np.linalg.norm(d2)), -1, 1)))
            if ang > turn_deg:
                ev.append({"frame": f, "object": name, "kind": "turns", "range": [f - 4, f + 4]})
        for f in range(1, len(a)):
            if present[f] and not present[f - 1]:
                ev.append({"frame": f, "object": name, "kind": "appears", "range": [f - 3, f + 5]})
            if present[f - 1] and not present[f]:
                ev.append({"frame": f, "object": name, "kind": "disappears", "range": [f - 5, f + 3]})
            if present[f] and present[f - 1] and a[f] < 0.6 * a[f - 1]:
                ev.append({"frame": f, "object": name, "kind": "shrinks (hidden or leaving?)", "range": [f - 4, f + 4]})
    if len(frame_diff) > 3:
        med = np.median(frame_diff[1:])
        for f in np.nonzero(frame_diff > max(6 * med, med + 20))[0]:
            ev.append({"frame": int(f), "object": None, "kind": "picture changes (cut?)", "range": [int(f) - 3, int(f) + 3]})
    ev.sort(key=lambda e: (str(e["object"]), e["kind"], e["frame"]))
    out = []
    for e in ev:
        if out and out[-1]["object"] == e["object"] and out[-1]["kind"] == e["kind"] and e["frame"] - out[-1]["frame"] <= 3:
            continue
        out.append(e)
    out.sort(key=lambda e: e["frame"])
    return out


def overview(run_dir: str, n: int = 16, cols: int = 4, cell_width: int = 320) -> dict:
    """n numbered frames over the whole clip, then a motion strip: each observed object's speed (px/frame) over
    time, so the moments to zoom into stand out. The first look."""
    import cv2

    rd, cfg, mgr, obs, video, t = _ctx(run_dir)
    frames = [int(f) for f in np.linspace(0, t - 1, n).round()]
    h0, w0 = video.shape[1:3]
    ch = int(cell_width * h0 / w0)
    cells = [_label(cv2.resize(video[f], (cell_width, ch), interpolation=cv2.INTER_AREA), f"frame {f}") for f in frames]
    sheet = _grid(cells, cols)
    tracks = object_tracks(obs, t, camera_shift(video, obs.masks, _movers(obs), t))
    W = sheet.shape[1]
    strip = np.full((40 + 22 * len(tracks), W, 3), 25, np.uint8)
    for i, (name, tr) in enumerate(tracks.items()):
        sp = np.nan_to_num(_speed(tr["centroid"]))
        top = 30 + 22 * i
        col = _COLOURS[i % len(_COLOURS)]
        mx = max(sp.max(), 1.0)
        for f in range(t):
            x0, x1 = int(f * W / t), int((f + 1) * W / t)
            hh = int(18 * sp[f] / mx)
            cv2.rectangle(strip, (x0, top + 18 - hh), (max(x1 - 1, x0), top + 18), col, -1)
        cv2.putText(strip, f"{name} (max {sp.max():.1f} px/frame)", (4, top + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)
    for f in range(0, t, max(1, t // 20)):
        x = int(f * W / t)
        cv2.putText(strip, str(f), (x, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)
    img = np.concatenate([sheet, strip], 0)
    return {"path": _save(rd, "overview.png", img), "frames": frames, "frame_count": t,
            "fps": float(mgr.adopted.source.fps.num / mgr.adopted.source.fps.den),
            "how_to_read": "numbered frames over the whole clip; below, each object's speed over time (bars, camera "
                           "pan removed). "
                           "Next: run_view_events for machine-found moments, run_view_clip to step through a range."}


def _movers(obs) -> list[int]:
    return [k for k, v in obs.id_map.items() if not v.startswith(("obj_floor", "obj_wall"))]


def events(run_dir: str) -> dict:
    rd, cfg, mgr, obs, video, t = _ctx(run_dir)
    shift = camera_shift(video, obs.masks, _movers(obs), t)
    tracks = object_tracks(obs, t, shift)
    g = video[:t].astype(np.float32).mean(-1)
    diff = np.r_[0.0, np.abs(np.diff(g, axis=0)).mean((1, 2))]
    ev = find_events(tracks, diff)
    pairs = dict(cfg.get("pairs", {}))
    for e in ev:
        if e["object"] in pairs:
            e["reconstruction_object"] = pairs[e["object"]]
        e["range"] = [max(0, e["range"][0]), min(t - 1, e["range"][1])]
    return {"events": ev, "frame_count": t, "camera_pan_px": [round(float(x), 1) for x in shift[-1]],
            "note": "machine-found from mask centroids and frame differences (thresholds are drafts); look at the "
                    "ranges with run_view_clip, check with run_view_track. The machine can miss things: say so."}


def _crop_box(obs, label_or_box, frames, h, w, margin: float = 0.35):
    if label_or_box is None:
        return 0, 0, w, h
    if isinstance(label_or_box, (list, tuple)):
        x0, y0, x1, y1 = (int(v) for v in label_or_box)
    else:
        k = next((kk for kk, v in obs.id_map.items() if v == label_or_box), None)
        if k is None:
            raise ValueError(f"no observed object {label_or_box!r}; known: {sorted(obs.id_map.values())}")
        us, vs = [], []
        for f in frames:
            v, u = np.nonzero(obs.masks[f] == k)
            if len(u):
                us += [u.min(), u.max()]
                vs += [v.min(), v.max()]
        if not us:
            return 0, 0, w, h
        x0, x1, y0, y1 = min(us), max(us), min(vs), max(vs)
    mx, my = int((x1 - x0) * margin) + 8, int((y1 - y0) * margin) + 8
    return max(0, x0 - mx), max(0, y0 - my), min(w, x1 + mx), min(h, y1 + my)


def clip(run_dir: str, start: int, end: int, step: int = 1, max_frames: int = 16, region=None,
         cell_width: int = 360, cols: int = 4) -> dict:
    """Consecutive frames start..end (every `step`, at most max_frames, at the video's own rate), each numbered.
    region: an observed object's label (crop around where it is over the range) or [x0, y0, x1, y1] in pixels;
    the crop is enlarged to cell_width (spatial zoom)."""
    import cv2

    rd, cfg, mgr, obs, video, t = _ctx(run_dir)
    start, end = max(0, int(start)), min(t - 1, int(end))
    frames = list(range(start, end + 1, max(1, int(step))))[:max_frames]
    th, tw = obs.masks.shape[1:]
    x0, y0, x1, y1 = _crop_box(obs, region, frames, th, tw)
    cells = []
    for f in frames:
        ref = _ref_frame(video, f, th, tw)[y0:y1, x0:x1]
        hh = int(cell_width * ref.shape[0] / max(ref.shape[1], 1))
        cells.append(_label(cv2.resize(ref, (cell_width, hh), interpolation=cv2.INTER_CUBIC), f"frame {f}"))
    img = _grid(cells, cols)
    name = f"clip_{start:05d}_{end:05d}" + (f"_{region}" if isinstance(region, str) else "") + ".png"
    return {"path": _save(rd, name, img), "frames": frames, "crop_px": [x0, y0, x1, y1],
            "zoom": round(cell_width / max(x1 - x0, 1), 2)}


def compare(run_dir: str, frames: list[int], region=None, cell_width: int = 420) -> dict:
    """Reference (left) and the reconstruction's silhouettes over it (right) on the same frames, cropped like
    clip(): where and when the reconstruction differs from the video."""
    import cv2

    from vrecon.optimize import render_silhouettes

    rd, cfg, mgr, obs, video, t = _ctx(run_dir)
    p = mgr.adopted
    th, tw = obs.masks.shape[1:]
    frames = [int(f) for f in frames if 0 <= int(f) < t][:12]
    ids = [o.id for o in p.reconstruction.objects if o.kind != "set"]
    sil = render_silhouettes(p, p.reconstruction.cameras[0].shot_id, frames, ids, (tw, th))
    x0, y0, x1, y1 = _crop_box(obs, region, frames, th, tw)
    rows = []
    for i, f in enumerate(frames):
        ref = _ref_frame(video, f, th, tw)
        over = (ref * 0.55).astype(np.uint8)
        for n_, oid in enumerate(ids, start=1):
            m = sil[i] == n_
            if m.any():
                col = np.array(_COLOURS[(n_ - 1) % len(_COLOURS)])
                over[m] = (over[m] * 0.4 + col * 0.6).astype(np.uint8)
        a, b = ref[y0:y1, x0:x1], over[y0:y1, x0:x1]
        hh = int(cell_width * a.shape[0] / max(a.shape[1], 1))
        a = _label(cv2.resize(a, (cell_width, hh), interpolation=cv2.INTER_CUBIC), f"frame {f} video")
        b = _label(cv2.resize(b, (cell_width, hh), interpolation=cv2.INTER_CUBIC), f"frame {f} scene")
        rows.append(np.concatenate([a, np.full((hh, 4, 3), 255, np.uint8), b], 1))
    img = np.concatenate(rows, 0)
    legend = {oid: list(_COLOURS[i % len(_COLOURS)]) for i, oid in enumerate(ids)}
    return {"path": _save(rd, f"compare_rev{p.revision:03d}_{frames[0]:05d}.png", img), "frames": frames,
            "crop_px": [x0, y0, x1, y1], "legend": legend, "revision": p.revision}


def track(run_dir: str, obj: str, start: int = 0, end: int | None = None) -> dict:
    """Numbers to check what the pictures suggest: per frame the observed mask centroid and area, the
    reconstruction's silhouette centroid for the paired object, their offset; and the frames where each
    starts and stops moving (onsets(): displacement beyond 5% of the travel)."""
    from vrecon.optimize import render_silhouettes

    rd, cfg, mgr, obs, video, t = _ctx(run_dir)
    p = mgr.adopted
    end = t - 1 if end is None else min(t - 1, int(end))
    frames = list(range(max(0, int(start)), end + 1))
    pairs = dict(cfg.get("pairs", {}))
    label = obj if obj in obs.id_map.values() else next((k for k, v in pairs.items() if v == obj), None)
    if label is None:
        raise ValueError(f"{obj!r} is neither an observed label nor a paired object; pairs: {pairs}")
    rec = pairs.get(label)
    shift = camera_shift(video, obs.masks, _movers(obs), t)
    tr = object_tracks(obs, t, shift)[label]
    th, tw = obs.masks.shape[1:]
    rc = np.full((t, 2), np.nan)
    if rec:
        sil = render_silhouettes(p, p.reconstruction.cameras[0].shot_id, frames, [rec], (tw, th))
        for i, f in enumerate(frames):
            v, u = np.nonzero(sil[i] == 1)
            if len(u):
                rc[f] = (u.mean(), v.mean()) - shift[f]

    so, se = onsets(tr["centroid"], area=tr["area"])
    ro, re_ = onsets(rc) if rec else (None, None)
    rows = []
    for f in frames:
        oc, cc = tr["centroid"][f], rc[f]
        off = (cc - oc) if np.all(np.isfinite(oc)) and np.all(np.isfinite(cc)) else [np.nan, np.nan]
        rows.append({"frame": f, "observed_xy": [round(float(x), 1) for x in oc], "area": int(tr["area"][f]),
                     "scene_xy": [round(float(x), 1) for x in cc], "offset_px": [round(float(x), 1) for x in off]})
    finite = [r["offset_px"] for r in rows if np.all(np.isfinite(r["offset_px"]))]
    return {"observed": label, "scene_object": rec, "frames": rows,
            "mean_offset_px": round(float(np.mean(np.linalg.norm(finite, axis=1))), 2) if finite else None,
            "observed_starts": so, "observed_stops": se, "scene_starts": ro, "scene_stops": re_}


__all__ = ["camera_shift", "clip", "compare", "events", "find_events", "object_tracks", "onsets", "overview", "track"]
