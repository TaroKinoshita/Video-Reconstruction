"""The geometric evidence D of one shot (MAVE eq. 6), in the levelled scene world (z up, metres).

Built from the shot's own analysis only (4RC point maps, SAM people masks, the camera of scripts/v11_mave_prep.py,
the ground of scripts/v11_parts.py); nothing from a benchmark's ground truth.

evidence.npz:
  cam_R [N,3,3], cam_T [N,3]   camera per video frame (OpenCV axes as columns, centre), levelled world
  f, size                       focal in pixels at size (W, H) of the full video picture
  kf [K]                        keyframes with observed depth (the 4RC frames)
  depth [K,h,w]                 observed depth (camera z, metres) at size / step, inf where not observed
  Q [M,3], Q_cls [M], Q_kf [M]  static points (Q_n: unit normals, facing the camera) (people and low confidence removed); class 0 support (faces up),
                                1 vertical, 2 ceiling (faces down, above the camera), 3 other; keyframe index
  ground_v [G,3]                the fitted ground height field (regular grid, in the parts' world: up_R turns it)
  up_R [3,3]                    parts world -> evidence world (identity unless the up was estimated, eq. 3)
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

A = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], float)  # 4RC world (y down) -> prep world (z up)
CLASSES = ("support", "vertical", "ceiling", "other")
SEM = ("support", "wall", "ceiling", "trunk")  # semantic labels (Q_sem index)


def umeyama(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """s, R, t with dst ~ s R src + t (least squares)."""
    ms, md = src.mean(0), dst.mean(0)
    a, b = src - ms, dst - md
    U, S, Vt = np.linalg.svd(b.T @ a / len(src))
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ D @ Vt
    s = float((S * np.diag(D)).sum() / (a ** 2).sum(1).mean())
    return s, R, md - s * R @ ms


def levelled_cameras(prep: dict, GR: np.ndarray, Gt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.einsum("ij,tjk->tik", GR, prep["cam_R"]), prep["cam_T"] @ GR.T + Gt


class Ground:
    """Height of the fitted ground at (x, y): bilinear on the regular grid of ground_v."""

    def __init__(self, gv: np.ndarray, R: np.ndarray | None = None):
        """gv: a regular grid in its own frame; R turns that frame into the evidence world (queries come in the
        evidence world)."""
        self.R = None if R is None or np.allclose(R, np.eye(3)) else np.asarray(R, float)
        xs, ys = np.unique(gv[:, 0].round(4)), np.unique(gv[:, 1].round(4))
        self.x0, self.y0 = xs[0], ys[0]
        self.cell = float(np.median(np.diff(xs)))
        self.nx, self.ny = len(xs), len(ys)
        ix = np.round((gv[:, 0] - self.x0) / self.cell).astype(int)
        iy = np.round((gv[:, 1] - self.y0) / self.cell).astype(int)
        self.z = np.full((self.ny, self.nx), np.nan)
        self.z[iy, ix] = gv[:, 2]

    def __call__(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        if self.R is None:
            return self._h(x, y)
        x, y = np.asarray(x, float), np.asarray(y, float)
        z = np.zeros(np.broadcast(x, y).shape)
        Rt = self.R.T
        for _ in range(4):  # the height where the turned surface meets the vertical line at (x, y)
            p = np.stack(np.broadcast_arrays(x, y, z), -1) @ Rt.T
            z = z + (self._h(p[..., 0], p[..., 1]) - p[..., 2]) / Rt[2, 2]
        return z

    def _h(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        fx = np.clip((np.asarray(x) - self.x0) / self.cell, 0, self.nx - 1.001)
        fy = np.clip((np.asarray(y) - self.y0) / self.cell, 0, self.ny - 1.001)
        i, j = fx.astype(int), fy.astype(int)
        u, v = fx - i, fy - j
        z = self.z
        return ((1 - u) * (1 - v) * z[j, i] + u * (1 - v) * z[j, i + 1] + (1 - u) * v * z[j + 1, i]
                + u * v * z[j + 1, i + 1])


def build(prep_path: Path, parts_path: Path, rc_path: Path, rc_frames: list[int], person_masks: np.ndarray | None,
          out: Path, step: int = 4, q_step: int = 3, calibrate: bool = True, up_mode: str = "ground",
          people_path: Path | None = None, semantic: tuple | None = None) -> dict:
    """person_masks: bool [K, H, W] on the 4RC frames (any resolution), or None.
    calibrate: when the scene was built from another 4RC run (4RC is not deterministic: depths of two runs differed
    by 3-13 % on F002), scale each keyframe's depth so that its ground pixels meet the scene's ground.
    up_mode: "ground" keeps the parts' world (z = normal of the fitted ground plane); "multi" estimates up from ground,
    ceiling and wall normals, the cameras and the people's upright direction (MAVE eq. 3, gravity.py) and turns
    the evidence world so that it is +z (stored as up_R; the session and the .blend apply the same turn).
    semantic: (masks [K, H, W] on the 4RC frames, {mask number: class name}) with class names support, wall,
    ceiling, trunk (MAVE labels points with Mask2Former; any segmenter works). Labelled points take their label's
    class (trunk counts as vertical), unlabelled ones keep the class from their normal; Q_sem keeps the label
    (-1 none), and the checks require building only what was recognised.
    people_path also removes points within 0.4 m of any body on that frame (blurred limbs and held props that
    the masks missed would otherwise count as static structure)."""
    prep = dict(np.load(prep_path))
    info = json.loads(str(prep["info"]))
    W, H = info["size"]
    f_img = info["f_img"]
    pz = np.load(parts_path)
    parts = json.loads(str(pz["parts"]))
    GR, Gt = np.array(parts["world_R"]), np.array(parts["world_t"])
    rc = np.load(rc_path)
    K = len(rc_frames)
    c2w = np.stack([rc[f"pred_{i}_extrinsic"].reshape(4, 4) for i in range(K)])
    # this 4RC run against the camera already in the scene: one similarity (4RC world -> prep world)
    s, Rs, ts = umeyama(c2w[:, :3, 3], prep["cam_T"][rc_frames])
    resid = np.linalg.norm((s * c2w[:, :3, 3] @ Rs.T + ts) - prep["cam_T"][rc_frames], axis=1)
    rot_err = [np.degrees(np.arccos(np.clip((np.trace((Rs @ c2w[i, :3, :3]).T @ prep["cam_R"][f]) - 1) / 2, -1, 1)))
               for i, f in enumerate(rc_frames)]
    cam_R, cam_T = levelled_cameras(prep, GR, Gt)
    h, w = H // step, W // step
    fs = f_img / step
    depth = np.full((K, h, w), np.inf, np.float32)
    Qs, Qc, Qk, Qn, Qsem, ratios = [], [], [], [], [], []
    body_vox = None
    if people_path is not None:
        an = np.load(people_path)
        body_vox = {}
        for l in json.loads(str(an["labels"])):
            for t, V in zip(an[f"{l}_t"], an[f"{l}_v"]):
                if int(t) in rc_frames:
                    k3 = np.floor((V.astype(np.float64) @ GR.T + Gt) / 0.4).astype(np.int64) + (1 << 20)
                    keys = (k3[:, 0] << 42) | (k3[:, 1] << 21) | k3[:, 2]
                    body_vox[int(t)] = np.unique(np.concatenate([body_vox.get(int(t), np.zeros(0, np.int64)), keys]))
    ground = Ground(pz["ground_v"])
    up = np.array([0, 0, 1.0])
    for i, f in enumerate(rc_frames):
        # image-aligned: each 4RC pixel's depth in its own camera, at its own picture position (4RC resizes the
        # long side to 512 and centre-crops), back-projected through the scene's camera; so the evidence agrees
        # with the picture even where this run's camera turned a few degrees from the scene's
        P = rc[f"pred_{i}_pts"].reshape(rc[f"pred_{i}_pts"].shape[-3:]).astype(np.float64)
        h4, w4 = P.shape[:2]
        conf = rc[f"pred_{i}_conf"].reshape(h4, w4)
        w2c = np.linalg.inv(c2w[i])
        z = s * (P.reshape(-1, 3) @ w2c[:3, :3].T + w2c[:3, 3])[:, 2]
        s4 = 512 / max(W, H)
        ox, oy = round(W * s4) // 2 - w4 // 2, round(H * s4) // 2 - h4 // 2
        gj, gi = np.meshgrid(np.arange(w4), np.arange(h4))
        u = ((gj.ravel() + ox + 0.5) / s4) / step  # pixel at the evidence resolution
        v = ((gi.ravel() + oy + 0.5) / s4) / step
        ray = np.stack([(u - w / 2) / fs, (v - h / 2) / fs, np.ones_like(u)], 1) @ cam_R[f].T
        X = cam_T[f] + z[:, None] * ray  # levelled world
        keep = (conf.ravel() > np.percentile(conf, 10)) & (z > 0.05)
        ui, vi = np.floor(u).astype(int), np.floor(v).astype(int)
        inside = (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
        person = np.zeros(h4 * w4, bool)
        if person_masks is not None:
            m = person_masks[i].astype(np.uint8)
            m = cv2.dilate(m, np.ones((5, 5), np.uint8))
            pu = np.clip((u * step * m.shape[1] / W).astype(int), 0, m.shape[1] - 1)
            pv = np.clip((v * step * m.shape[0] / H).astype(int), 0, m.shape[0] - 1)
            person = m[pv, pu] > 0
        sel = keep & inside & ~person
        if body_vox is not None and f in body_vox:
            k3 = np.floor(X / 0.4).astype(np.int64)
            near = np.zeros(len(X), bool)
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        kk = k3 + [dx, dy, dz] + (1 << 20)
                        near |= np.isin((kk[:, 0] << 42) | (kk[:, 1] << 21) | kk[:, 2], body_vox[f])
            sel &= ~near
        if calibrate:  # this run's depth against the scene's ground (built from an earlier run): one ratio per frame
            idx = np.where(sel)[0][::7]
            ts_ = np.geomspace(0.3, 60, 300)
            Pm = cam_T[f][None, None] + ts_[:, None, None] * ray[idx][None]
            below = Pm[..., 2] < ground(Pm[..., 0], Pm[..., 1])
            first = np.where(below.any(0), below.argmax(0), -1)
            tg = np.where(first >= 0, ts_[np.maximum(first, 0)], np.inf)
            r = z[idx] / tg
            r = r[np.isfinite(r) & (np.abs(r - 1) < 0.3)]
            ratio = float(np.median(r)) if len(r) > 200 else 1.0
            ratios.append(round(ratio, 4))
            z = z / ratio
            X = cam_T[f] + z[:, None] * ray
        flat = vi[sel] * w + ui[sel]
        d = depth[i].ravel()
        np.minimum.at(d, flat, z[sel].astype(np.float32))
        depth[i] = d.reshape(h, w)
        # normals from the point map grid
        G = X.reshape(h4, w4, 3)
        du = np.zeros_like(G)
        dv = np.zeros_like(G)
        du[:, 1:-1] = G[:, 2:] - G[:, :-2]
        dv[1:-1] = G[2:] - G[:-2]
        n = np.cross(du, dv).reshape(-1, 3)
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
        n *= np.where(((cam_T[f] - X) * n).sum(1, keepdims=True) < 0, -1, 1)  # facing the camera
        cz = n @ up
        cls = np.full(len(X), 3)
        cls[cz > 0.8] = 0
        cls[np.abs(cz) < 0.35] = 1
        cls[(cz < -0.8) & (X[:, 2] > cam_T[f][2])] = 2
        sem = np.full(len(X), -1, np.int8)
        if semantic is not None:
            sm, names = semantic
            m = sm[i]
            pu = np.clip((u * step * m.shape[1] / W).astype(int), 0, m.shape[1] - 1)
            pv = np.clip((v * step * m.shape[0] / H).astype(int), 0, m.shape[0] - 1)
            lab = m[pv, pu]
            for num, name in names.items():
                if name in SEM:
                    sem[lab == int(num)] = SEM.index(name)
            cls = np.where(sem < 0, cls, np.array([0, 1, 2, 1])[np.maximum(sem, 0)])
        grid = np.zeros((h4, w4), bool)
        grid[::q_step, ::q_step] = True
        qsel = sel & (grid.ravel() | (cls == 1) | (cls == 2))  # every vertical / ceiling point, a sample of the rest
        Qs.append(X[qsel])
        Qc.append(cls[qsel])
        Qk.append(np.full(qsel.sum(), i))
        Qn.append(n[qsel].astype(np.float16))
        Qsem.append(sem[qsel])
    ev = {"cam_R": cam_R, "cam_T": cam_T, "f": np.float64(f_img), "size": np.array([W, H]), "step": np.int32(step),
          "kf": np.array(rc_frames), "depth": depth, "Q": np.concatenate(Qs).astype(np.float32),
          "Q_cls": np.concatenate(Qc).astype(np.int8), "Q_kf": np.concatenate(Qk).astype(np.int16),
          "Q_n": np.concatenate(Qn), "Q_sem": np.concatenate(Qsem),
          "ground_v": pz["ground_v"]}
    up_R, up_info = np.eye(3), None
    if up_mode == "multi":
        from vrecon.agent import gravity
        ppl = {}
        if people_path is not None:
            an = np.load(people_path)
            ppl = {l: {"v": an[f"{l}_v"].astype(np.float64) @ GR.T + Gt} for l in json.loads(str(an["labels"]))}
        r = gravity.estimate(*gravity.cues_from(ev, ppl))
        u = r["up"]
        v = np.cross(u, [0, 0, 1.0])
        sn, c = np.linalg.norm(v), float(u[2])
        if sn > 1e-9:
            k = v / sn
            K_ = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
            up_R = np.eye(3) + sn * K_ + (1 - c) * K_ @ K_
        ev["cam_R"] = np.einsum("ij,tjk->tik", up_R, ev["cam_R"])
        ev["cam_T"] = ev["cam_T"] @ up_R.T
        ev["Q"] = (ev["Q"].astype(np.float64) @ up_R.T).astype(np.float32)
        ev["Q_n"] = (ev["Q_n"].astype(np.float64) @ up_R.T).astype(np.float16)
        up_info = {"tilt_deg": round(r["tilt_from_start_deg"], 2), "up_in_parts_world": np.round(u, 4).tolist()}
    ev["up_R"] = up_R
    np.savez_compressed(out, **ev)
    report = {"up": up_info, "keyframes": K, "align_scale": s, "align_resid_m_max": float(resid.max()),
              "align_rot_deg_max": float(max(rot_err)), "ground_ratio_per_keyframe": ratios, "depth_px_observed": float(np.isfinite(depth).mean()),
              "points": int(len(ev["Q"])), "classes": {c: int((ev["Q_cls"] == k).sum()) for k, c in enumerate(CLASSES)}}
    Path(str(out) + ".json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    return report


def _grid_components(xy: np.ndarray, cell: float, min_pts: int, diag: bool) -> list[np.ndarray]:
    """Point groups whose plan-view cells (with >= min_pts points) touch; largest first."""
    keys = np.floor(xy / cell).astype(np.int64)
    cells: dict = {}
    for i, k in enumerate(map(tuple, keys)):
        cells.setdefault(k, []).append(i)
    full = {k for k, v in cells.items() if len(v) >= min_pts}
    nb = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if (dx or dy) and (diag or not (dx and dy))]
    seen, out = set(), []
    for k in full:
        if k in seen:
            continue
        stack, comp = [k], []
        seen.add(k)
        while stack:
            c = stack.pop()
            comp.extend(cells[c])
            for dx, dy in nb:
                n = (c[0] + dx, c[1] + dy)
                if n in full and n not in seen:
                    seen.add(n)
                    stack.append(n)
        out.append(np.array(comp))
    return sorted(out, key=len, reverse=True)


def _recognised(sem: np.ndarray) -> str | None:
    """The segmenter's label for most of these points (None when under half are labelled)."""
    lab = sem[sem >= 0]
    if len(lab) < 0.5 * max(len(sem), 1):
        return None
    return SEM[int(np.bincount(lab).argmax())]


def extract_structures(ev: dict) -> list[dict]:
    """Geometric evidence in the shape the agent builds with (MAVE's support layers L, walls W, ceilings U; here
    without a semantic model): support layers as fitted planes with a coarse height grid, vertical and ceiling
    points as separate plan-view groups (columns such as trunks or posts, or larger masses such as walls or foliage)."""
    X, cls = ev["Q"].astype(np.float64), ev["Q_cls"]
    out = []
    S = X[cls == 0]
    for comp in _grid_components(S[:, :2], 1.0, 5, diag=False)[:6]:
        P = S[comp]
        if len(P) < 300:
            continue
        A_ = np.c_[P[:, 0], P[:, 1], np.ones(len(P))]
        keep = np.ones(len(P), bool)
        for _ in range(3):
            coef = np.linalg.lstsq(A_[keep], P[keep, 2], rcond=None)[0]
            r = P[:, 2] - A_ @ coef
            mad = np.median(np.abs(r[keep])) + 1e-6
            keep = np.abs(r) < 3 * mad + 0.05
        lo, hi = np.percentile(P, 2, 0), np.percentile(P, 98, 0)
        g = {}
        for (x, y, z) in P[keep]:
            g.setdefault((int(np.floor(x / 2.5)), int(np.floor(y / 2.5))), []).append(z)
        grid = sorted([[round((i + 0.5) * 2.5, 2), round((j + 0.5) * 2.5, 2), round(float(np.median(v)), 2)]
                       for (i, j), v in g.items() if len(v) >= 20])
        n = np.array([-coef[0], -coef[1], 1.0])
        out.append({"class": "support", "kind": "layer", "points": int(len(P)),
                    "plane_z_eq": f"z = {coef[0]:.4f} x + {coef[1]:.4f} y + {coef[2]:.3f}",
                    "tilt_deg": round(float(np.degrees(np.arccos(1 / np.linalg.norm(n)))), 1),
                    "residual_m": round(float(np.median(np.abs(r[keep]))), 3),
                    "extent_min": lo.round(2).tolist(), "extent_max": hi.round(2).tolist(),
                    "area_m2": round(len({(int(a // 1), int(b // 1)) for a, b in P[:, :2]}) * 1.0, 1),
                    "height_grid_2_5m": grid[:80]})
    sem_all = ev["Q_sem"] if "Q_sem" in ev else np.full(len(X), -1)
    for lab in ("trunk", "wall"):  # what a segmenter recognised: grouped on its own
        m = sem_all == SEM.index(lab)
        P_all, kf_all = X[m], ev["Q_kf"][m]
        for comp in _grid_components(P_all[:, :2], 0.25, 2, diag=True)[:60]:
            P = P_all[comp]
            if len(P) < 40:
                continue
            lo, hi = np.percentile(P, 3, 0), np.percentile(P, 97, 0)
            w_ = float(max(hi[0] - lo[0], hi[1] - lo[1]))
            out.append({"class": "vertical", "kind": "column" if w_ < 2.0 else "wall-like", "points": int(len(P)),
                        "center": np.median(P, 0).round(2).tolist(), "extent_min": lo.round(2).tolist(),
                        "extent_max": hi.round(2).tolist(), "plan_width_m": round(w_, 2), "recognised": lab,
                        "keyframes_seen": sorted({int(ev["kf"][i]) for i in kf_all[comp]})[:10]})
    rest_m = ~np.isin(sem_all, [SEM.index("trunk"), SEM.index("wall")])
    X, cls = X[rest_m], cls[rest_m]
    ev = dict(ev, Q_kf=ev["Q_kf"][rest_m], Q_sem=sem_all[rest_m])
    for k, name in ((1, "vertical"), (2, "ceiling")):
        V = X[cls == k]
        kfs = ev["Q_kf"][cls == k]
        semk = ev["Q_sem"][cls == k]
        # columns: plan cells whose points stack over >= 1.5 m of height (trunks, posts, poles)
        keys = np.floor(V[:, :2] / 0.25).astype(np.int64)
        zb = np.floor(V[:, 2] / 0.25).astype(np.int64)
        cell_z: dict = {}
        for (a, b), z in zip(map(tuple, keys), zb):
            cell_z.setdefault((a, b), set()).add(int(z))
        tall = np.array([len(cell_z[(a, b)]) >= 6 for a, b in map(tuple, keys)], bool)
        idx_t = np.where(tall)[0]
        for comp in _grid_components(V[idx_t, :2], 0.25, 1, diag=True)[:60]:
            P = V[idx_t[comp]]
            if len(P) < 30:
                continue
            lo, hi = np.percentile(P, 3, 0), np.percentile(P, 97, 0)
            w_ = float(max(hi[0] - lo[0], hi[1] - lo[1]))
            out.append({"class": name, "kind": "column" if w_ < 2.0 else "wall-like", "points": int(len(P)),
                        "center": np.median(P, 0).round(2).tolist(), "extent_min": lo.round(2).tolist(),
                        "extent_max": hi.round(2).tolist(), "plan_width_m": round(w_, 2),
                        "recognised": _recognised(semk[idx_t[comp]]),
                        "keyframes_seen": sorted({int(ev["kf"][i]) for i in kfs[idx_t[comp]]})[:10]})
        rest = np.where(~tall)[0]
        for comp in _grid_components(V[rest, :2], 1.0, 10, diag=True)[:8]:
            P = V[rest[comp]]
            if len(P) < 200:
                continue
            lo, hi = np.percentile(P, 5, 0), np.percentile(P, 95, 0)
            out.append({"class": name, "kind": "mass", "points": int(len(P)),
                        "center": np.median(P, 0).round(2).tolist(), "extent_min": lo.round(2).tolist(),
                        "extent_max": hi.round(2).tolist(),
                        "plan_width_m": round(float(max(hi[0] - lo[0], hi[1] - lo[1])), 2),
                        "keyframes_seen": sorted({int(ev["kf"][i]) for i in kfs[rest[comp]]})[:10]})
    return out


def load(path: Path) -> dict:
    d = dict(np.load(path))
    d["ground"] = Ground(d["ground_v"], d.get("up_R"))
    return d
