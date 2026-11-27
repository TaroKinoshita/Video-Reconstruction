"""Background as separate parts for one shot (self-built base, MAVEBench F002 test): ground, tree trunks, posts and
beams, a far backdrop for what is left, and imagined surroundings outside what the camera saw. Pure numpy/cv2.

  python scripts/v11_parts.py PREP.npz RC.npz RC_FRAMES.json RC_VIDEO.mp4 OUT.npz \
      --labels BG.masks.npz=ground,trunk,beam,foliage TRUNK.masks.npz=obs:trunk PROPS.masks.npz=post:post,obs:stick \
      --people PEOPLE.masks.npz

RC_VIDEO is the 4RC frames as a video (frame i = RC_FRAMES[i]); the mask files are SAM 3.1 runs on that video.
A label spec FILE=a,b,obs:x reads mask labels a and b as themselves and every obs_N track as instances of x.

Steps (every measure comes from the film's own analysis; nothing from the benchmark's ground truth):
  up        plane fitted to the ground points (4RC, refined least squares); the world is turned so it is +z, origin on it.
  ground    height field (cell --cell m) from the ground points, holes filled and extended to --radius m by diffusion;
            colour per cell = median film colour; cells never seen get colours drawn from the seen ones (marked).
  trunks    per mask component per frame: centre (median of its 3D points), radius (pixel width x depth / f / 2);
            observations within max(0.6 m, 2 r) of each other are one trunk; centre pushed back by r (points are the
            front surface); height = highest point seen, at least --min-tree m when it reaches the top of the frame.
  posts     the same for "post"/"beam"; a cluster whose main axis is level becomes a beam (cylinder along that axis).
  backdrop  4RC surface of the keyframes not covered by people, ground, trunks or posts (mostly far foliage), coloured
            from the film. Kept as one mesh: it is the part that is not yet split into objects.
  imagined  trees outside every camera's view (and 3 m from the camera path) up to --radius m: Poisson-disk positions at the density of trunks seen on
            the ground, radii / colours drawn from the seen trunks, a canopy blob on each (colour of the seen foliage).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

A = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], float)


def main() -> int:
    ap = argparse.ArgumentParser()
    for name in ("prep", "rc", "rc_frames", "rc_video", "out"):
        ap.add_argument(name, type=Path)
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--people", type=Path, required=True, help="people masks on the full video (frame = video frame)")
    ap.add_argument("--keyframes", type=int, nargs="+", required=True)
    ap.add_argument("--cell", type=float, default=0.3)
    ap.add_argument("--radius", type=float, default=35.0)
    ap.add_argument("--min-tree", type=float, default=14.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--texel", type=float, default=0.025, help="ground texture resolution, metres")
    ap.add_argument("--ground-masks", type=Path, required=True, help="the mask file holding the 'ground' label")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    prep = np.load(a.prep)
    info = json.loads(str(prep["info"]))
    W, H = info["size"]
    f_img, unit = info["f_img"], info["unit_4rc_to_body"]
    rc = np.load(a.rc)
    rc_frames = json.loads(a.rc_frames.read_text(encoding="utf-8"))
    n_rc = len(rc_frames)
    P0 = rc["pred_0_pts"].reshape(rc["pred_0_pts"].shape[-3:])
    h4, w4 = P0.shape[:2]
    s = 512 / max(W, H)
    rw, rh = round(W * s), round(H * s)
    ox, oy = rw // 2 - w4 // 2, rh // 2 - h4 // 2

    def to_grid(img: np.ndarray, nearest=True) -> np.ndarray:
        r = cv2.resize(img, (rw, rh), interpolation=cv2.INTER_NEAREST if nearest else cv2.INTER_AREA)
        return r[oy:oy + h4, ox:ox + w4]

    # film colours of the 4RC frames, on the 4RC grid
    cap = cv2.VideoCapture(str(a.rc_video))
    cols = []
    while len(cols) < n_rc:
        ok, fr = cap.read()
        if not ok:
            break
        cols.append(to_grid(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB), nearest=False).astype(np.float32) / 255)
    # 3D points per frame in the output world of PREP (about metres, z roughly up)
    pts = [((rc[f"pred_{i}_pts"].reshape(h4, w4, 3) * unit) @ A.T) for i in range(n_rc)]
    conf = [rc[f"pred_{i}_conf"].reshape(h4, w4) for i in range(n_rc)]
    cam_T = prep["cam_T"]
    # labels: name -> list over frames of instance maps (0 = none, k = instance k)
    inst: dict[str, list[np.ndarray]] = {}
    for spec in a.labels:
        path, names = spec.split("=")
        d = np.load(path)
        m, id_map = d["masks"], json.loads(str(d["id_map"]))
        for item in names.split(","):
            src, _, dst = item.partition(":")
            dst = dst or src
            nums = [int(n) for n, lab in id_map.items() if (lab.startswith("obs_") if src == "obs" else lab == src)]
            for i in range(n_rc):
                g = inst.setdefault(dst, [np.zeros((h4, w4), np.int32) for _ in range(n_rc)])[i]
                for n in nums:
                    mm = to_grid((m[i] == n).astype(np.uint8)).astype(bool)
                    if src == "obs":
                        g[mm & (g == 0)] = n + 1000 * (len(names) + 1)
                    else:  # one label for all instances: split into connected pieces
                        k, cc = cv2.connectedComponents(mm.astype(np.uint8))
                        g[(cc > 0) & (g == 0)] = cc[(cc > 0) & (g == 0)] + 5000
    pm = np.load(a.people)["masks"]
    people = [cv2.dilate(to_grid((pm[t] > 0).astype(np.uint8)), np.ones((5, 5), np.uint8)).astype(bool) for t in rc_frames]

    # ---- up: plane through the ground points
    G = np.concatenate([pts[i][(inst["ground"][i] > 0) & ~people[i]][::5] for i in range(n_rc)])
    n = np.array([0.0, 0.0, 1.0])
    p0 = np.median(G, 0)
    for _ in range(4):
        keep = np.abs((G - p0) @ n) < (1.0 if _ == 0 else 0.3)
        mu = G[keep].mean(0)
        C = G[keep] - mu
        n = np.linalg.eigh(C.T @ C)[1][:, 0]
        n *= np.sign(n[2])
        p0 = mu
    x = np.cross([0, 1, 0], n)
    x /= np.linalg.norm(x)
    y = np.cross(n, x)
    Rg = np.stack([x, y, n])  # rows: new axes in old coordinates; new = Rg @ (old - mu)
    tg = -Rg @ mu

    def gw(X: np.ndarray) -> np.ndarray:
        return X @ Rg.T + tg

    pts = [gw(P) for P in pts]
    cam_g = gw(cam_T)

    # ---- ground height field
    gp, gc = [], []
    for i in range(n_rc):
        mm = (inst["ground"][i] > 0) & ~people[i] & (conf[i] > np.percentile(conf[i], 10))
        gp.append(pts[i][mm])
        gc.append(cols[i][mm])
    gp, gc = np.concatenate(gp), np.concatenate(gc)
    centre = np.median(gp[:, :2], 0)
    R = a.radius
    nx = int(2 * R / a.cell) + 1
    xs = centre[0] - R + a.cell * np.arange(nx)
    ys = centre[1] - R + a.cell * np.arange(nx)
    ix = np.clip(((gp[:, 0] - xs[0]) / a.cell).round().astype(int), 0, nx - 1)
    iy = np.clip(((gp[:, 1] - ys[0]) / a.cell).round().astype(int), 0, nx - 1)
    cell = iy * nx + ix
    order = np.argsort(cell)
    cell_s, gz_s, gc_s = cell[order], gp[order, 2], gc[order]
    uniq, start, count = np.unique(cell_s, return_index=True, return_counts=True)
    zgrid = np.full(nx * nx, np.nan)
    cgrid = np.full((nx * nx, 3), np.nan)
    for u, st, ct in zip(uniq, start, count):
        if ct >= 2:
            zgrid[u] = np.median(gz_s[st:st + ct])
            cgrid[u] = np.median(gc_s[st:st + ct], 0)
    seen = ~np.isnan(zgrid)
    z = np.where(seen, zgrid, 0.0).reshape(nx, nx)
    known = seen.reshape(nx, nx)
    for _ in range(400):  # fill holes / extend: Laplace diffusion with seen cells fixed, then light smoothing
        zz = cv2.blur(z, (3, 3))
        z = np.where(known, z, zz)
    z = np.where(known, cv2.GaussianBlur(z, (0, 0), 1.0), z)
    seen_cols = cgrid[seen]
    cg = cgrid.copy()
    cg[~seen] = seen_cols[rng.integers(0, len(seen_cols), (~seen).sum())]
    cg = cv2.GaussianBlur(cg.reshape(nx, nx, 3).astype(np.float32), (0, 0), 0.7).reshape(-1, 3)
    cg[seen] = cgrid[seen]
    gx, gy = np.meshgrid(xs, ys)
    ground_v = np.stack([gx.ravel(), gy.ravel(), z.ravel()], 1)
    idx = np.arange(nx * nx).reshape(nx, nx)
    q = np.stack([idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel(), idx[1:, 1:].ravel(), idx[1:, :-1].ravel()], 1)
    ground_f = np.concatenate([q[:, [0, 1, 2]], q[:, [0, 2, 3]]])
    # ---- ground texture (--texel m) over the seen cells, each texel from the frame that sees it largest
    seen_idx = np.argwhere(known)
    (y0, x0), (y1, x1) = seen_idx.min(0), seen_idx.max(0)
    tx0, ty0 = xs[x0] - 1, ys[y0] - 1
    tw, th = xs[x1] - xs[x0] + 2, ys[y1] - ys[y0] + 2
    TW, TH = int(tw / a.texel), int(th / a.texel)
    txx, tyy = np.meshgrid(tx0 + (np.arange(TW) + 0.5) * a.texel, ty0 + (np.arange(TH) + 0.5) * a.texel)
    jx = np.clip(((txx - xs[0]) / a.cell), 0, nx - 1.001)
    jy = np.clip(((tyy - ys[0]) / a.cell), 0, nx - 1.001)
    tz = cv2.remap(z.astype(np.float32), jx.astype(np.float32), jy.astype(np.float32), cv2.INTER_LINEAR)
    TX = np.stack([txx, tyy, tz], -1).reshape(-1, 3)
    best = np.full(len(TX), np.inf)
    tex = np.zeros((len(TX), 3), np.float32)
    gmask_full = np.load(a.ground_masks)
    gm_full, gm_map = gmask_full["masks"], json.loads(str(gmask_full["id_map"]))
    g_num = [int(k) for k, v in gm_map.items() if v == "ground"]
    capf = cv2.VideoCapture(str(a.rc_video))
    for i in range(n_rc):
        ok_, fr = capf.read()
        if not ok_:
            break
        img = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
        okm = np.isin(gm_full[i], g_num) & ~cv2.dilate((pm[rc_frames[i]] > 0).astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
        Rc = Rg @ prep["cam_R"][rc_frames[i]]
        q = (TX - cam_g[rc_frames[i]]) @ Rc
        zq = q[:, 2]
        u = f_img * q[:, 0] / np.maximum(zq, 1e-6) + W / 2
        v = f_img * q[:, 1] / np.maximum(zq, 1e-6) + H / 2
        ins = (zq > 0.3) & (u >= 0) & (u < W - 1) & (v >= 0) & (v < H - 1)
        ui, vi = u[ins].astype(int), v[ins].astype(int)
        good = okm[vi, ui]
        sel = np.flatnonzero(ins)[good]
        foot = zq[sel] / max(abs(Rc[:, 2] @ np.array([0, 0, 1.0])), 0.2)  # bigger when far or grazing
        better = foot < best[sel]
        sel, foot = sel[better], foot[better]
        best[sel] = foot
        us, vs = u[sel], v[sel]  # bilinear sample
        u0, v0 = us.astype(int), vs.astype(int)
        fu, fv = (us - u0)[:, None], (vs - v0)[:, None]
        tex[sel] = ((1 - fu) * (1 - fv) * img[v0, u0] + fu * (1 - fv) * img[v0, u0 + 1]
                    + (1 - fu) * fv * img[v0 + 1, u0] + fu * fv * img[v0 + 1, u0 + 1])
    tex_seen = np.isfinite(best).reshape(TH, TW)
    tex = tex.reshape(TH, TW, 3)
    # ground not seen gets a tile of seen ground (mirrored so it repeats without seams) instead of flat cell colours;
    # the film texture fades into it over about 0.5 m
    P = int(round(3.0 / a.texel))
    integ = cv2.integral(tex_seen.astype(np.uint8))
    full = (integ[P:, P:] - integ[:-P, P:] - integ[P:, :-P] + integ[:-P, :-P]) == P * P
    lum = cv2.blur(tex.mean(-1), (P, P))[P // 2:P // 2 + full.shape[0], P // 2:P // 2 + full.shape[1]]
    if full.any():
        target = np.median(tex[tex_seen].mean(-1))
        cand = np.argwhere(full)
        y_, x_ = cand[np.argmin(np.abs(lum[full] - target))]
        patch = tex[y_:y_ + P, x_:x_ + P]
    else:
        patch = np.tile(np.median(tex[tex_seen], 0) if tex_seen.any() else [0.3, 0.25, 0.2], (P, P, 1))
    tile = np.concatenate([np.concatenate([patch, patch[:, ::-1]], 1),
                           np.concatenate([patch[::-1], patch[::-1, ::-1]], 1)], 0)
    tex_alpha = cv2.GaussianBlur(cv2.erode(tex_seen.astype(np.float32), np.ones((9, 9), np.uint8)), (0, 0), 0.5 / a.texel / 2)
    seen_area = cv2.dilate(known.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)  # +-1 m around what was seen

    def ground_z(xq, yq):
        jx = np.clip(((np.asarray(xq) - xs[0]) / a.cell).round().astype(int), 0, nx - 1)
        jy = np.clip(((np.asarray(yq) - ys[0]) / a.cell).round().astype(int), 0, nx - 1)
        return z[jy, jx]

    # ---- trunks and posts from mask components
    def observations(label: str) -> list[dict]:
        obs = []
        for i in range(n_rc):
            g = inst.get(label, [None] * n_rc)[i]
            if g is None:
                continue
            for k in np.unique(g[g > 0]):
                mm = (g == k) & ~people[i]
                if mm.sum() < 12:
                    continue
                P = pts[i][mm]
                cz = np.linalg.norm(P - cam_g[rc_frames[i]], axis=1)
                ys_, xs_ = np.nonzero(mm)
                widths = [np.ptp(xs_[ys_ == r]) + 1 for r in np.unique(ys_)]
                rad = float(np.median(widths)) / s * np.median(cz) / f_img / 2
                cam = cam_g[rc_frames[i]]
                uv = np.stack([xs_, ys_], 1).astype(float)
                cu = uv - uv.mean(0)
                ax2 = np.linalg.eigh(cu.T @ cu)[1][:, -1]
                proj = cu @ ax2
                ends = []
                for q in (np.percentile(proj, 3), np.percentile(proj, 97)):  # 3D point at that end, at median distance
                    j = int(np.argmin(np.abs(proj - q)))
                    d = P[j] - cam
                    ends.append(cam + d / np.linalg.norm(d) * np.median(cz))
                minor = cu @ np.linalg.eigh(cu.T @ cu)[1][:, 0]  # thickness across the 2D main axis
                rad_across = float(np.percentile(minor, 95) - np.percentile(minor, 5)) / s * np.median(cz) / f_img / 2
                obs.append({"frame": i, "c": np.median(P, 0), "r": rad, "r_across": rad_across,
                            "top": float(np.percentile(P[:, 2], 98)), "ends": ends,
                            "at_top": bool(ys_.min() <= 1), "colour": np.median(cols[i][mm], 0), "P": P,
                            "cam": cam})
        return obs

    def cluster(obs: list[dict]) -> list[list[dict]]:
        groups: list[list[dict]] = []
        for o in sorted(obs, key=lambda o: -o["r"]):
            for g in groups:
                c = np.median([q["c"][:2] for q in g], 0)
                if np.linalg.norm(o["c"][:2] - c) < max(0.6, 2 * np.median([q["r"] for q in g])):
                    g.append(o)
                    break
            else:
                groups.append([o])
        return groups

    trunks = []
    for g in cluster(observations("trunk")):
        r = float(np.clip(np.median([o["r"] for o in g]), 0.05, 0.8))
        c = np.median([o["c"] for o in g], 0)
        view = c[:2] - np.median([o["cam"][:2] for o in g], 0)
        c[:2] += r * view / (np.linalg.norm(view) + 1e-9)
        base = float(ground_z(c[0], c[1]))
        top = max(o["top"] for o in g)
        h = max(top - base, a.min_tree if any(o["at_top"] for o in g) else 0.0, 1.0)
        trunks.append({"x": float(c[0]), "y": float(c[1]), "base": base, "r": r, "h": float(h),
                       "colour": np.median([o["colour"] for o in g], 0).tolist(), "frames": len(g)})
    posts, beams = [], []
    for g in cluster(observations("post")) + cluster(observations("beam")):
        ends = []
        for o in g:
            e0, e1 = o["ends"]
            ends.append((e0, e1) if e0[0] <= e1[0] else (e1, e0))
        e0 = np.median([e[0] for e in ends], 0)
        e1 = np.median([e[1] for e in ends], 0)
        axis = (e1 - e0) / (np.linalg.norm(e1 - e0) + 1e-9)
        colour = np.median([o["colour"] for o in g], 0).tolist()
        r = float(np.clip(np.median([o["r_across"] for o in g]), 0.03, 0.4))
        if abs(axis[2]) < 0.5:  # level: a beam
            if np.linalg.norm(e1 - e0) > 6:  # longer than any beam in the shot: a bad fit, left out
                continue
            beams.append({"a": e0.tolist(), "b": e1.tolist(), "r": r, "colour": colour, "frames": len(g)})
        else:
            mu = np.median([o["c"] for o in g], 0)
            base = float(ground_z(mu[0], mu[1]))
            top = max(o["top"] for o in g)
            posts.append({"x": float(mu[0]), "y": float(mu[1]), "base": base, "r": r,
                          "h": float(np.clip(top - base, 0.5, 4.0)), "colour": colour, "frames": len(g)})
    # one-frame pieces are often noise: keep trunks seen at least twice unless large
    trunks = [t for t in trunks if t["frames"] >= 2 or t["r"] > 0.15]

    # ---- backdrop: what is left on the keyframes
    # static check: a backdrop point must not be "seen through" from the other 4RC frames. A thing that moved
    # (a swung staff, an arm missed by the masks) sits in front of what the other frames see at its pixel; a static
    # surface is confirmed there (same depth) or hidden (behind). Kept: confirmations >= see-throughs.
    Rc = [Rg @ prep["cam_R"][t] for t in rc_frames]
    zmaps = [((pts[j] - cam_g[rc_frames[j]]) @ Rc[j])[..., 2] for j in range(n_rc)]

    def static_ok(P: np.ndarray, i_self: int) -> np.ndarray:
        conf_n = np.zeros(len(P), int)
        thru_n = np.zeros(len(P), int)
        for j in range(n_rc):
            if j == i_self:
                continue
            Xc = (P - cam_g[rc_frames[j]]) @ Rc[j]
            z = Xc[:, 2]
            zs = np.maximum(z, 1e-6)
            x4 = np.round((f_img * Xc[:, 0] / zs + W / 2) * s - ox).astype(int)
            y4 = np.round((f_img * Xc[:, 1] / zs + H / 2) * s - oy).astype(int)
            ins = (z > 0.3) & (x4 >= 0) & (x4 < w4) & (y4 >= 0) & (y4 < h4)
            zm = np.full(len(P), np.nan)
            zm[ins] = zmaps[j][y4[ins], x4[ins]]
            ins[ins] &= ~people[j][y4[ins], x4[ins]]
            conf_n += ins & (np.abs(zm - z) < 0.06 * z)
            thru_n += ins & (zm > 1.15 * z)
        return conf_n >= thru_n

    bd_v, bd_f, bd_c, off = [], [], [], 0
    dropped_moving = 0
    taken_labels = [k for k in inst if k not in ("foliage",)]
    for kf in a.keyframes:
        i = rc_frames.index(kf)
        taken = people[i].copy()
        for lab in taken_labels:
            taken |= cv2.dilate((inst[lab][i] > 0).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        Pw = pts[i]
        zc = np.linalg.norm(Pw - cam_g[kf], axis=1 if Pw.ndim == 2 else 2)
        keep = ~taken & (conf[i] > np.percentile(conf[i], 10)) & (cols[i].mean(-1) < 0.8)  # very bright = sky
        step = 2
        Pk, zk, kk, ck = Pw[::step, ::step], zc[::step, ::step], keep[::step, ::step], cols[i][::step, ::step]
        hh, ww = kk.shape
        st = static_ok(Pk.reshape(-1, 3), i).reshape(hh, ww)
        dropped_moving += int((kk & ~st).sum())
        kk = kk & st
        ii = np.arange(hh * ww).reshape(hh, ww)
        q1, q2, q3, q4 = ii[:-1, :-1].ravel(), ii[:-1, 1:].ravel(), ii[1:, :-1].ravel(), ii[1:, 1:].ravel()
        tris = np.concatenate([np.stack([q1, q3, q2], 1), np.stack([q2, q3, q4], 1)])
        zt = zk.ravel()[tris]
        Pf = Pk.reshape(-1, 3)
        edge = np.max([np.linalg.norm(Pf[tris[:, j]] - Pf[tris[:, (j + 1) % 3]], axis=1) for j in range(3)], axis=0)
        # a grid step at this distance is about step / s / f_img * distance; slivers are much longer than that
        ok = (kk.ravel()[tris].all(1) & (zt.max(1) / np.maximum(zt.min(1), 1e-9) < 1.08)
              & (edge < 6 * step / s / f_img * zt.min(1)))
        tris = tris[ok]
        used = np.unique(tris)
        remap = -np.ones(hh * ww, np.int64)
        remap[used] = np.arange(len(used))
        bd_v.append(Pk.reshape(-1, 3)[used])
        bd_c.append(ck.reshape(-1, 3)[used])
        bd_f.append(remap[tris] + off)
        off += len(used)

    # ---- imagined trees outside the seen area
    seen_xy = np.array([[t["x"], t["y"]] for t in trunks]) if trunks else np.zeros((0, 2))
    cams = [(cam_g[t], Rg @ prep["cam_R"][t]) for t in rc_frames]  # (centre, camera-to-world rotation, OpenCV axes)
    far = float(np.percentile([np.min(np.linalg.norm(cam_g[rc_frames][:, :2] - [t["x"], t["y"]], axis=1)) for t in trunks], 90)) if trunks else 20.0

    def looked_at(p: np.ndarray) -> bool:
        """True when some 4RC camera had this ground spot (at 1.5 m height) in view within the depth trunks were seen."""
        for c, Rc in cams:
            q = Rc.T @ (np.array([p[0], p[1], ground_z(p[0], p[1]) + 1.5]) - c)
            if 0 < q[2] < far and abs(f_img * q[0] / q[2]) < W / 2 and abs(f_img * q[1] / q[2]) < H / 2:
                return True
        return False

    # trunks per m2 over the ground the cameras looked at (sampled), out to the depth trunks were seen at
    probe = centre + rng.uniform(-R, R, (4000, 2))
    probe = probe[np.linalg.norm(probe - centre, axis=1) < R]
    looked = np.array([looked_at(p) for p in probe])
    looked_m2 = looked.mean() * np.pi * R ** 2
    density = len(trunks) / max(looked_m2, 1.0)
    seen_m2 = known.sum() * a.cell ** 2
    fol = np.concatenate([cols[i][inst["foliage"][i] > 0] for i in range(n_rc)] + [np.zeros((0, 3))])
    leaf_colour = np.median(fol, 0) if len(fol) > 50 else np.array([0.25, 0.35, 0.15])
    min_gap = max(1.5, 0.8 / np.sqrt(density))
    imagined = []
    area = np.pi * R ** 2
    for _ in range(int(density * area * 3)):
        p = centre + rng.uniform(-R, R, 2)
        if np.linalg.norm(p - centre) > R - 1:
            continue
        jx = int(round((p[0] - xs[0]) / a.cell))
        jy = int(round((p[1] - ys[0]) / a.cell))
        if seen_area[jy, jx] or looked_at(p) or np.min(np.linalg.norm(cam_g[:, :2] - p, axis=1)) < 3.0:
            continue  # seen, or in some camera's view, or where the camera itself stood
        others = np.array([[t["x"], t["y"]] for t in imagined]).reshape(-1, 2)
        if len(seen_xy) and np.min(np.linalg.norm(seen_xy - p, axis=1)) < min_gap:
            continue
        if len(others) and np.min(np.linalg.norm(others - p, axis=1)) < min_gap:
            continue
        src = trunks[rng.integers(len(trunks))] if trunks else {"r": 0.2, "colour": [0.3, 0.25, 0.2]}
        imagined.append({"x": float(p[0]), "y": float(p[1]), "base": float(ground_z(p[0], p[1])),
                         "r": float(src["r"] * rng.uniform(0.8, 1.2)), "h": float(a.min_tree * rng.uniform(0.85, 1.25)),
                         "colour": src["colour"]})
        if len(imagined) >= int(density * (area - looked_m2)):
            break
    canopy = []
    for t in trunks + imagined:
        if t["h"] >= a.min_tree * 0.8:
            canopy.append({"c": [t["x"], t["y"], t["base"] + t["h"] * rng.uniform(0.75, 0.9)],
                           "size": [t["h"] * 0.25, t["h"] * 0.25, t["h"] * 0.18],
                           "colour": (leaf_colour * rng.uniform(0.85, 1.15)).tolist()})
    sky = [p for i in range(n_rc) for p in cols[i][:8][(cols[i][:8].mean(-1) > 0.75)]]
    sky_colour = np.median(sky, 0).tolist() if len(sky) > 20 else [0.85, 0.9, 0.95]

    def lin(c):
        c = np.asarray(c, np.float64)
        return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)

    for lst in (trunks, imagined, posts, beams, canopy):
        for t in lst:
            t["colour_lin"] = lin(t["colour"]).tolist()
    parts = {"world_R": Rg.tolist(), "world_t": tg.tolist(), "trunks": trunks, "imagined_trees": imagined,
             "posts": posts, "beams": beams, "canopy": canopy, "sky_colour_lin": lin(sky_colour).tolist(),
             "leaf_colour": leaf_colour.tolist(), "trunk_density_per_m2": density, "seen_ground_m2": seen_m2, "looked_at_m2": looked_m2,
             "radius": R}
    out = {"parts": np.array(json.dumps(parts)), "ground_v": ground_v.astype(np.float32),
           "ground_f": ground_f.astype(np.int32), "ground_col": lin(cg).astype(np.float32),
           "ground_seen": seen.astype(bool), "ground_tex": (np.clip(tex, 0, 1) * 255).astype(np.uint8),
           "ground_tex_seen": tex_seen, "ground_tex_alpha": tex_alpha.astype(np.float32), "ground_tile": (np.clip(tile, 0, 1) * 255).astype(np.uint8), "ground_tile_m": 2 * P * a.texel, "ground_tex_box": np.array([tx0, ty0, tw, th], np.float64), "backdrop_v": np.concatenate(bd_v).astype(np.float32),
           "backdrop_f": np.concatenate(bd_f).astype(np.int32), "backdrop_col": lin(np.concatenate(bd_c)).astype(np.float32)}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, **out)
    print(json.dumps({"up_tilt_deg": round(float(np.degrees(np.arccos(n[2]))), 1),
                      "looked_at_m2": round(looked_m2), "trunks_seen_out_to_m": round(far, 1),
                      "trunks": len(trunks), "posts": len(posts), "beams": len(beams), "imagined_trees": len(imagined),
                      "canopy": len(canopy), "density_per_100m2": round(density * 100, 2),
                      "seen_ground_m2": round(seen_m2), "backdrop_tris": int(sum(len(f) for f in bd_f)), "backdrop_points_dropped_moving": dropped_moving,
                      "camera_height_m": [round(float(cam_g[f][2] - ground_z(cam_g[f][0], cam_g[f][1])), 2) for f in (0, 150, 305)]},
                     indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
