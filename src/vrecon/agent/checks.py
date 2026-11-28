"""The four checks of MAVE section 3.3, against the evidence (evidence.py), as issues with stable ids.

video      primitives in front of observed surfaces: phi (eq. 12) = share of the primitive's pixels, over the
           keyframes, where it lies in front of the observed depth by more than tau(d) = 0.1 m + 5 % of d
coverage   static points of the support / vertical / ceiling classes not within reach of any surface of the scene
           (primitives, and the kept system meshes); reported as clusters
human      per person: height outside 1.45-2.05 m, feet floating or sunk, body inside primitives
complete   floating, duplicated or buried primitives, and the camera inside a primitive (the orbit pictures are
           inspected by the agent itself)
"""
from __future__ import annotations

from collections import defaultdict, deque

import numpy as np

from vrecon.agent import prims
from vrecon.agent.evidence import CLASSES

TAU0, TAU1 = 0.1, 0.05
NEAR_WIN = 3  # observed depth taken as the nearest within this window: 4RC smooths thin structures into what is behind
PHI_FLAG, PHI_MIN_PX = 0.15, 20
HEIGHT_OK = (1.45, 2.05)
FOOT_UP, FOOT_DOWN, FOOT_MIN_FRAMES = 0.15, -0.12, 4
INSIDE_JUSTIFIABLE = 0.15  # a body more than this far inside a primitive is a scene error, not a pose


def rays(ev: dict, frame: int, us: np.ndarray, vs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rays through pixel centres (us, vs) at the evidence depth resolution; t along them = camera depth."""
    W, H = ev["size"]
    step = int(ev["step"])
    w, h, fs = W // step, H // step, float(ev["f"]) / step
    x = (us + 0.5 - w / 2) / fs
    y = (vs + 0.5 - h / 2) / fs
    d = np.stack([x, y, np.ones_like(x)], 1) @ ev["cam_R"][frame].T
    return ev["cam_T"][frame], d


def project_box(ev: dict, frame: int, p: dict) -> tuple[int, int, int, int] | None:
    W, H = ev["size"]
    step = int(ev["step"])
    w, h, fs = W // step, H // step, float(ev["f"]) / step
    c = (prims.corners(p) - ev["cam_T"][frame]) @ ev["cam_R"][frame]
    if (c[:, 2] <= 0.05).all():
        return None
    if (c[:, 2] <= 0.05).any():  # crosses the camera plane: whole picture
        return 0, 0, w, h
    u, v = fs * c[:, 0] / c[:, 2] + w / 2, fs * c[:, 1] / c[:, 2] + h / 2
    u0, u1 = int(max(np.floor(u.min()), 0)), int(min(np.ceil(u.max()) + 1, w))
    v0, v1 = int(max(np.floor(v.min()), 0)), int(min(np.ceil(v.max()) + 1, h))
    if u0 >= u1 or v0 >= v1:
        return None
    return u0, v0, u1, v1


def near_depth(ev: dict) -> np.ndarray:
    """Observed depth min-filtered over NEAR_WIN pixels (inf kept where nothing was observed nearby)."""
    if "_near" not in ev:
        import cv2
        k = np.ones((NEAR_WIN, NEAR_WIN), np.uint8)
        d = np.where(np.isfinite(ev["depth"]), ev["depth"], 1e6).astype(np.float32)
        m = np.stack([cv2.erode(x, k) for x in d])
        ev["_near"] = np.where(m >= 1e6, np.inf, m)
    return ev["_near"]


def video_consistency(ev: dict, scene: dict) -> list[dict]:
    issues = []
    near = near_depth(ev)
    for p in scene["primitives"]:
        viol = over = 0
        worst = (0, None)
        for j, f in enumerate(ev["kf"]):
            box = project_box(ev, int(f), p)
            if box is None:
                continue
            u0, v0, u1, v1 = box
            vs, us = np.mgrid[v0:v1, u0:u1]
            us, vs = us.ravel(), vs.ravel()
            obs = near[j, vs, us]
            ok = np.isfinite(obs)
            if not ok.any():
                continue
            o, d = rays(ev, int(f), us[ok].astype(float), vs[ok].astype(float))
            t = prims.intersect(p, o, d)
            hit = np.isfinite(t)
            n_over = int(hit.sum())
            if not n_over:
                continue
            ob = obs[ok][hit]
            n_viol = int((t[hit] < ob - (TAU0 + TAU1 * ob)).sum())
            viol += n_viol
            over += n_over
            if n_viol > worst[0]:
                worst = (n_viol, int(f))
        if over and viol >= PHI_MIN_PX and viol / over > PHI_FLAG:
            issues.append({"id": f"video:{p['name']}", "check": "video", "primitive": p["name"],
                           "phi": round(viol / over, 3), "pixels_in_front": viol, "pixels_overlapping": over,
                           "worst_frame": worst[1],
                           "text": f"{p['name']} lies in front of the observed surface on {viol / over:.0%} of its "
                                   f"{over} pixels over the keyframes (worst frame {worst[1]})."})
    return issues


VOX = 0.4


def vox_keys(k: np.ndarray) -> np.ndarray:
    """One int64 per integer voxel coordinate triple."""
    k = k + (1 << 20)
    return (k[:, 0] << 42) | (k[:, 1] << 21) | k[:, 2]


def mesh_voxels(V: np.ndarray) -> np.ndarray:
    return np.unique(vox_keys(np.floor(np.asarray(V, np.float64) / VOX).astype(np.int64)))


def _voxel_index(X: np.ndarray, size: float) -> dict:
    keys = np.floor(X / size).astype(np.int64)
    idx = defaultdict(list)
    for i, k in enumerate(map(tuple, keys)):
        idx[k].append(i)
    return idx


def covered(ev: dict, scene: dict, X: np.ndarray, tol: np.ndarray, mesh_vox: set | None) -> np.ndarray:
    """True where a point X lies within tol of (or inside) a scene surface."""
    cov = np.zeros(len(X), bool)
    fx = scene.get("fixed", {})
    if fx.get("ground", False):
        cov |= np.abs(X[:, 2] - ev["ground"](X[:, 0], X[:, 1])) < tol
    if mesh_vox is not None:  # sorted voxel keys (vox_keys) of the kept system meshes
        k = np.floor(X / VOX).astype(np.int64)
        near = np.zeros(len(X), bool)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    near |= np.isin(vox_keys(k + [dx, dy, dz]), mesh_vox, assume_unique=False)
        cov |= near
    order = np.argsort(X[:, 0])
    xs = X[order, 0]
    for p in scene["primitives"]:
        lo, hi = prims.aabb(p)
        a, b = np.searchsorted(xs, lo[0] - 1.5), np.searchsorted(xs, hi[0] + 1.5)
        if a >= b:
            continue
        cand = order[a:b]
        cand = cand[~cov[cand]]
        Y = X[cand]
        m = ((Y[:, 1] > lo[1] - 1.5) & (Y[:, 1] < hi[1] + 1.5) & (Y[:, 2] > lo[2] - 1.5) & (Y[:, 2] < hi[2] + 1.5))
        cand = cand[m]
        if len(cand):
            cov[cand] |= prims.sdf(p, X[cand]) < tol[cand]
    return cov


def clusters(X: np.ndarray, size: float = 0.6, min_pts: int = 150) -> list[np.ndarray]:
    """Connected groups of points (26-neighbour voxels), largest first."""
    vox = _voxel_index(X, size)
    seen, out = set(), []
    for k in vox:
        if k in seen:
            continue
        comp, q = [], deque([k])
        seen.add(k)
        while q:
            c = q.popleft()
            comp.extend(vox[c])
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        nb = (c[0] + dx, c[1] + dy, c[2] + dz)
                        if nb in vox and nb not in seen:
                            seen.add(nb)
                            q.append(nb)
        if len(comp) >= min_pts:
            out.append(np.array(comp))
    return sorted(out, key=len, reverse=True)


def describe_cluster(ev: dict, X: np.ndarray, cls: str, idx: np.ndarray) -> dict:
    P = X[idx]
    lo, hi = np.percentile(P, 5, 0), np.percentile(P, 95, 0)
    c = np.median(P, 0)
    dist = float(np.median(np.linalg.norm(P - ev["cam_T"][ev["kf"][0]], axis=1)))
    return {"class": cls, "points": int(len(idx)), "center": [round(float(v), 2) for v in c],
            "extent_min": [round(float(v), 2) for v in lo], "extent_max": [round(float(v), 2) for v in hi],
            "distance_from_first_camera": round(dist, 1),
            "keyframes_seen": sorted({int(ev["kf"][k]) for k in ev["Q_kf"][idx]})[:12]}


NEAR_M = 15.0  # uncovered structure this close to the camera path is in the action area


def coverage(ev: dict, scene: dict, mesh_vox: set | None) -> tuple[list[dict], dict]:
    """Uncovered observed points, grouped as the structures tool groups them (support layers, columns such as
    trunks and posts, wider wall-like groups, masses), so each issue is one thing the agent can build."""
    from vrecon.agent.evidence import extract_structures
    X = ev["Q"].astype(np.float64)
    dist = np.linalg.norm(X - ev["cam_T"][ev["kf"][ev["Q_kf"]]], axis=1)
    tol = 0.2 + 0.02 * dist
    cov = covered(ev, scene, X, tol, mesh_vox)
    summary = {}
    for k, cls in enumerate(CLASSES[:3]):
        sel = ev["Q_cls"] == k
        if sel.any():
            summary[cls] = round(float(cov[sel].mean()), 3)
    un = ~cov & (ev["Q_cls"] <= 2)
    sub = dict(ev, **{k: ev[k][un] for k in ("Q", "Q_cls", "Q_kf", "Q_n", "Q_sem") if k in ev})
    cams = ev["cam_T"]
    issues = []
    for st in extract_structures(sub):
        c = st.get("center") or [(a + b) / 2 for a, b in zip(st["extent_min"], st["extent_max"])]
        near = float(np.min(np.linalg.norm(cams[:, :2] - np.asarray(c[:2]), axis=1)))
        kind = st["kind"]
        if kind == "layer" and st["points"] < 300:
            continue
        # must be built: the ground layer near the action, and near structure a segmenter recognised (trunk,
        # wall, ceiling); geometry nobody recognised (foliage, blur, depth noise) may be explained instead
        must = near < NEAR_M and ((kind == "layer" and st["points"] >= 300)
                                  or (st.get("recognised") in ("wall", "ceiling") and st["points"] >= 80)
                                  or (st.get("recognised") == "trunk" and kind == "column" and st["points"] >= 80))
        iid = f"coverage:{st['class']}:{kind}:({c[0]:.0f},{c[1]:.0f},{c[2]:.0f})"
        info = {k: v for k, v in st.items() if k not in ("height_grid_2_5m",)}
        if kind == "layer":
            info["height_grid_2_5m"] = st["height_grid_2_5m"][:30]
        issues.append({"id": iid, "check": "coverage", **info, "center": [round(float(v), 2) for v in c],
                       "distance_from_camera_path_m": round(near, 1), "justifiable": not must,
                       "text": f"Observed {st['class']} {kind}"
                               + (f" recognised as {st['recognised']}" if st.get("recognised") else "")
                               + f" ({st['points']} points) around "
                               f"{[round(float(v), 1) for v in c]}, extent {st['extent_min']} to {st['extent_max']}, "
                               f"{near:.0f} m from the camera path, is not covered by the scene."
                               + ("" if not must else " It is in the action area: build it (cannot be justified).")})
    return issues, summary


def _segments(flags: np.ndarray, ts: np.ndarray, min_len: int) -> list[tuple[int, int]]:
    out, start = [], None
    for i, f in enumerate(list(flags) + [False]):
        if f and start is None:
            start = i
        if not f and start is not None:
            if i - start >= min_len:
                out.append((int(ts[start]), int(ts[i - 1])))
            start = None
    return out


def support_height(ev: dict, scene: dict, xy: np.ndarray, below: np.ndarray) -> np.ndarray:
    """Highest supporting surface under each xy that is not above below + 0.3 m (ground and primitive tops)."""
    best = np.full(len(xy), -np.inf)
    if scene.get("fixed", {}).get("ground", False):
        best = ev["ground"](xy[:, 0], xy[:, 1]).astype(float)
    for p in scene["primitives"]:
        if p.get("class") in ("foliage",):
            continue
        lo, hi = prims.aabb(p)
        m = (xy[:, 0] > lo[0]) & (xy[:, 0] < hi[0]) & (xy[:, 1] > lo[1]) & (xy[:, 1] < hi[1])
        if not m.any():
            continue
        top = prims.top_at(p, xy[m])
        ok = np.isfinite(top) & (top < below[m] + 0.3)
        b = best[m]
        b[ok] = np.maximum(b[ok], top[ok])
        best[m] = b
    return best


def human(ev: dict, scene: dict, people: dict) -> tuple[list[dict], dict]:
    issues, summary = [], {}
    for label, P in people.items():
        st = scene["people"].get(label, {})
        if st.get("track_removed"):
            continue
        ts, V = P["t"], P["v"]
        H_est = P["height"] * st.get("k", 1.0)
        low_i = np.argsort(V[:, :, 2], axis=1)[:, :30]
        low = np.take_along_axis(V, low_i[:, :, None], 1).mean(1)
        sup = support_height(ev, scene, low[:, :2], low[:, 2])
        gap = low[:, 2] - sup
        summary[label] = {"height_m": round(float(H_est), 2), "k": round(st.get("k", 1.0), 3),
                          "foot_gap_median_m": round(float(np.nanmedian(gap[np.isfinite(gap)])) if np.isfinite(gap).any()
                                                     else float("nan"), 3),
                          "frames": [int(ts[0]), int(ts[-1])], "removed_segments": st.get("removed", [])}
        if not HEIGHT_OK[0] <= H_est <= HEIGHT_OK[1]:
            issues.append({"id": f"human:height:{label}", "check": "human", "person": label, "justifiable": False,
                           "height_m": round(float(H_est), 2),
                           "text": f"{label} comes out {H_est:.2f} m tall (scale k {st.get('k', 1.0):.3f}); "
                                   f"adults are usually {HEIGHT_OK[0]}-{HEIGHT_OK[1]} m."})
        for a, b in _segments(~np.isfinite(gap), ts, FOOT_MIN_FRAMES):
            issues.append({"id": f"human:unsupported:{label}:{a}-{b}", "check": "human", "person": label,
                           "frames": [a, b], "text": f"Nothing under {label}'s feet on frames {a}-{b} "
                                                     f"(no ground or other support there)."})
        for kind, flags in (("floating", gap > FOOT_UP), ("sunk", gap < FOOT_DOWN)):
            for a, b in _segments(flags & np.isfinite(gap), ts, FOOT_MIN_FRAMES):
                sel = (ts >= a) & (ts <= b)
                g = float(np.median(gap[sel]))
                issues.append({"id": f"human:{kind}:{label}:{a}-{b}", "check": "human", "person": label,
                               "frames": [a, b], "median_gap_m": round(g, 3),
                               "text": f"{label}'s feet are {abs(g):.2f} m {'above' if g > 0 else 'below'} the "
                                       f"supporting surface on frames {a}-{b}."})
        # body inside primitives, every 3rd frame
        inside = defaultdict(list)
        for j in range(0, len(ts), 3):
            X = V[j]
            lo, hi = X.min(0), X.max(0)
            for p in scene["primitives"]:
                if p.get("class") == "foliage":
                    continue
                plo, phi = prims.aabb(p)
                if (plo > hi).any() or (phi < lo).any():
                    continue
                share = float((prims.sdf(p, X) < -0.05).mean())
                if share > 0.02:
                    inside[p["name"]].append((int(ts[j]), share))
        for name, hits in inside.items():
            fr = [h[0] for h in hits]
            share = max(h[1] for h in hits)
            issues.append({"id": f"human:inside:{label}:{name}", "check": "human", "person": label, "primitive": name,
                           "justifiable": share <= INSIDE_JUSTIFIABLE,
                           "frames": [min(fr), max(fr)], "max_share": round(max(h[1] for h in hits), 3),
                           "text": f"{label}'s body is inside {name} on {len(fr)} sampled frames between {min(fr)} "
                                   f"and {max(fr)} (up to {max(h[1] for h in hits):.0%} of the body)."})
    return issues, summary


def completeness(ev: dict, scene: dict) -> list[dict]:
    issues = []
    P = scene["primitives"]
    boxes = [prims.aabb(p) for p in P]
    cams = ev["cam_T"][:: max(1, len(ev["cam_T"]) // 60)]
    ground_kept = scene.get("fixed", {}).get("ground", False)
    for i, p in enumerate(P):
        lo, hi = boxes[i]
        if (prims.sdf(p, cams) < 0).any():
            issues.append({"id": f"complete:camera_inside:{p['name']}", "check": "complete", "primitive": p["name"],
                           "text": f"The camera passes inside {p['name']}; it would block the whole picture."})
        c = np.asarray(p["center"], float)
        g = float(ev["ground"](c[:1], c[1:2])[0]) if ground_kept else None
        if g is not None and hi[2] < g - 0.05:
            issues.append({"id": f"complete:buried:{p['name']}", "check": "complete", "primitive": p["name"],
                           "text": f"{p['name']} is entirely below the ground."})
            continue
        if g is not None and lo[2] > g + 0.3:
            touching = any(j != i and (boxes[j][0] <= hi + 0.05).all() and (boxes[j][1] >= lo - 0.05).all()
                           for j in range(len(P)))
            if not touching:
                issues.append({"id": f"complete:floating:{p['name']}", "check": "complete", "primitive": p["name"],
                               "gap_m": round(lo[2] - g, 2),
                               "text": f"{p['name']} floats {lo[2] - g:.2f} m above the ground and touches nothing."})
    by_cls = defaultdict(list)
    for i, p in enumerate(P):
        by_cls[(p.get("class"), p["type"])].append(i)
    for idx in by_cls.values():
        for a_i, i in enumerate(idx):
            for j in idx[a_i + 1:]:
                lo = np.maximum(boxes[i][0], boxes[j][0])
                hi = np.minimum(boxes[i][1], boxes[j][1])
                inter = np.prod(np.maximum(hi - lo, 0))
                if inter <= 0:
                    continue
                vi, vj = np.prod(boxes[i][1] - boxes[i][0]), np.prod(boxes[j][1] - boxes[j][0])
                if inter / min(vi, vj) > 0.6:
                    a, b = P[i]["name"], P[j]["name"]
                    issues.append({"id": f"complete:duplicate:{a}/{b}", "check": "complete", "primitives": [a, b],
                                   "text": f"{a} and {b} overlap by {inter / min(vi, vj):.0%} of the smaller one: "
                                           f"likely the same thing twice."})
    return issues


def verify(ev: dict, scene: dict, people: dict, mesh_vox: set | None) -> dict:
    iv = video_consistency(ev, scene)
    ic, cov = coverage(ev, scene, mesh_vox)
    ih, hum = human(ev, scene, people)
    ik = completeness(ev, scene)
    issues = iv + ic + ih + ik
    just = scene.get("justified", {})
    for i in issues:
        if i["check"] == "complete":
            i["justifiable"] = False
        i.setdefault("justifiable", True)
        i["justified"] = just.get(i["id"]) if i["justifiable"] else None
    return {"issues": issues, "coverage_share": cov, "people": hum,
            "counts": {c: sum(1 for i in issues if i["check"] == c) for c in ("video", "coverage", "human", "complete")},
            "open": sum(1 for i in issues if not i["justified"])}
