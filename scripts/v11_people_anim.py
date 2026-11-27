"""People as moving bodies with colours from the film: one SAM 3D Body mesh per person per sampled frame (same
topology on every frame), placed in the space of PREP.npz (camera from scripts/v11_mave_prep.py), smoothed over time,
and coloured per vertex from the frames where that part of the body is seen. Pure numpy/cv2.

  python scripts/v11_people_anim.py PREP.npz RC.npz RC_FRAMES.json PEOPLE.masks.npz BODY_DIR VIDEO.mp4 OUT.npz

Scale: the body model gives a size in metres but its depth from one picture is unsure. On the 4RC frames the body's
front surface is set to the 4RC depth inside the person's mask (as in v11_mave_prep.py); between those frames the
factor is interpolated (log, after a 3-point median). Scaling about the camera centre keeps the picture exact.
Time: each sample is scored by how well its silhouette fits the person's mask (cover x inside any person mask);
samples that fit much worse than their neighbours get little weight, and the turn (quaternion) and the shape in the
body's own orientation are smoothed with those weights, so a bad fit is carried by its neighbours and the motion is
smooth instead of following every per-frame fit.
Size: after smoothing, each person gets one body size (median over samples); each sample is scaled about the
camera centre, so the picture is kept and the distance absorbs the change.
Colours: a vertex takes the film's colour on frames where it is nearest the camera at its pixel (vertex z-buffer) and
inside its own person mask; median over frames; vertices never seen take the colour of their seen neighbours.
ASCII paths only (cv2).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v11_mave_prep import read_ply  # noqa: E402


def gauss_smooth(x: np.ndarray, t: np.ndarray, sigma: float, max_gap: float) -> np.ndarray:
    """Smooth samples x[i] at times t[i] (axis 0) with a Gaussian in time; never across a gap above max_gap."""
    seg = np.concatenate([[0], np.cumsum(np.diff(t) > max_gap)])
    out = np.empty_like(x)
    for i in range(len(t)):
        w = np.exp(-0.5 * ((t - t[i]) / sigma) ** 2) * (seg == seg[i])
        w /= w.sum()
        out[i] = np.tensordot(w, x, axes=(0, 0))
    return out


def kabsch(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Rotation R with P @ R.T ~ Q (both centred)."""
    U, _, Vt = np.linalg.svd(P.T @ Q)
    D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    return (U @ D @ Vt).T


def rot_to_quat(R: np.ndarray) -> np.ndarray:
    w = np.sqrt(max(1e-12, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = np.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2 * np.sign(R[2, 1] - R[1, 2] or 1)
    y = np.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2 * np.sign(R[0, 2] - R[2, 0] or 1)
    z = np.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2 * np.sign(R[1, 0] - R[0, 1] or 1)
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def quat_to_rot(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def wsmooth(x: np.ndarray, t: np.ndarray, w: np.ndarray, sigma: float, max_gap: float) -> np.ndarray:
    """Weighted Gaussian in time: samples with low weight (a body that does not fit its mask) borrow from neighbours."""
    seg = np.concatenate([[0], np.cumsum(np.diff(t) > max_gap)])
    out = np.empty_like(x)
    for i in range(len(t)):
        k = np.exp(-0.5 * ((t - t[i]) / sigma) ** 2) * (seg == seg[i]) * w
        k /= k.sum()
        out[i] = np.tensordot(k, x, axes=(0, 0))
    return out


def silhouette_fit(Xc: np.ndarray, F: np.ndarray, own: np.ndarray, anyone: np.ndarray, f_img, W, H) -> float:
    """How well a body (camera space) fits its mask: share of the mask it covers x share of it inside some person
    mask (others may hide it). 1 = perfect."""
    z = Xc[:, 2]
    if (z <= 0.05).any():
        return 0.0
    uv = np.stack([f_img * Xc[:, 0] / z + W / 2, f_img * Xc[:, 1] / z + H / 2], 1)
    sil = np.zeros((H, W), np.uint8)
    cv2.fillPoly(sil, np.round(uv[F]).astype(np.int32), 1)
    sil = sil.astype(bool)
    if not own.any() or not sil.any():
        return 0.0
    cover = (sil & own).sum() / own.sum()
    inside = (sil & anyone).sum() / sil.sum()
    return float(cover * inside)


def constant_size(Xs: np.ndarray, centres: np.ndarray, win: int = 4, seed: int = 0):
    """One body size per person (as MAVE's m_k). Each sample's size is measured on vertex pairs whose distance hardly
    changes with pose (log distance minus the frame's median, lowest spread over time), median over +-win samples;
    each sample is then scaled about its camera centre to the person's median size, so the picture stays exact and
    the size error goes into the distance (MAVE fixes the depth instead; on F002 that moved silhouettes off the film).
    Returns the bodies and the measured sizes."""
    rng = np.random.default_rng(seed)
    n = Xs.shape[1]
    i, k = rng.integers(0, n, 3000), rng.integers(0, n, 3000)
    L = np.log(np.linalg.norm(Xs[:, i] - Xs[:, k], axis=2) + 1e-6)
    long_ = np.exp(L).mean(0) > 0.15
    sd = (L - np.median(L, 1, keepdims=True)).std(0)
    rigid = long_ & (sd < np.percentile(sd[long_], 10))
    size = np.exp(np.median(L[:, rigid] - L[:, rigid].mean(0), 1))
    size_s = np.array([np.median(size[max(0, j - win):j + win + 1]) for j in range(len(size))])
    f = np.median(size_s) / size_s
    o = centres[:, None, :]
    return (o + (Xs - o) * f[:, None, None]).astype(np.float32), size


def fill_colours(col: np.ndarray, seen: np.ndarray, F: np.ndarray, iters: int = 200) -> np.ndarray:
    """Spread colours from seen vertices to unseen ones over mesh edges."""
    n = len(col)
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    e = np.concatenate([e, e[:, ::-1]])
    col, known = col.copy(), seen.copy()
    for _ in range(iters):
        if known.all():
            break
        acc = np.zeros((n, 3))
        cnt = np.zeros(n)
        src = known[e[:, 0]]
        np.add.at(acc, e[src, 1], col[e[src, 0]])
        np.add.at(cnt, e[src, 1], 1)
        new = ~known & (cnt > 0)
        col[new] = acc[new] / cnt[new, None]
        known |= new
    col[~known] = np.median(col[seen], axis=0) if seen.any() else 0.5
    return col


def main() -> int:
    ap = argparse.ArgumentParser()
    for name in ("prep", "rc", "rc_frames", "masks", "body", "video", "out"):
        ap.add_argument(name, type=Path)
    ap.add_argument("--sigma-root", type=float, default=3.0, help="frames; smoothing of the body's position")
    ap.add_argument("--sigma-pose", type=float, default=1.5, help="frames; smoothing of the turn and pose around it")
    ap.add_argument("--constant-size", type=int, default=1, help="1: one body size per person (centre kept)")
    ap.add_argument("--min-fit", type=float, default=0.25, help="samples fitting their mask worse are not used")
    a = ap.parse_args()
    prep = np.load(a.prep)
    info = json.loads(str(prep["info"]))
    W, H = info["size"]
    f_img, unit = info["f_img"], info["unit_4rc_to_body"]
    cam_R, cam_T = prep["cam_R"], prep["cam_T"]
    rc = np.load(a.rc)
    rc_frames = json.loads(a.rc_frames.read_text(encoding="utf-8"))
    md = np.load(a.masks)
    masks, id_map = md["masks"], json.loads(str(md["id_map"]))
    num_of = {lab: int(n) for n, lab in id_map.items()}

    P0 = rc["pred_0_pts"].reshape(rc["pred_0_pts"].shape[-3:])
    h4, w4 = P0.shape[:2]
    s = 512 / max(W, H)
    rw, rh = round(W * s), round(H * s)
    ox, oy = rw // 2 - w4 // 2, rh // 2 - h4 // 2

    def zmap_of(i: int) -> np.ndarray:
        P = rc[f"pred_{i}_pts"].reshape(h4, w4, 3)
        w2c = np.linalg.inv(rc[f"pred_{i}_extrinsic"].reshape(4, 4))
        return (P.reshape(-1, 3) @ w2c[:3, :3].T + w2c[:3, 3])[:, 2].reshape(h4, w4)

    def to_rc_grid(m: np.ndarray) -> np.ndarray:
        r = cv2.resize(m.astype(np.uint8), (rw, rh), interpolation=cv2.INTER_NEAREST)
        return r[oy:oy + h4, ox:ox + w4].astype(bool)

    files: dict[str, list[int]] = {}
    for p in a.body.glob("f*_obs_*_0.ply"):
        m = re.match(r"f(\d+)_(obs_\d+)_0\.ply", p.name)
        files.setdefault(m.group(2), []).append(int(m.group(1)))

    frames_needed = sorted({f for fs in files.values() for f in fs})
    cap = cv2.VideoCapture(str(a.video))
    imgs, i = {}, 0
    while True:
        ok, fr = cap.read()
        if not ok or i > frames_needed[-1]:
            break
        if i in frames_needed:
            imgs[i] = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        i += 1

    out, report = {}, {}
    zmaps = {}
    for label in sorted(files):
        ts = sorted(files[label])
        F = None
        Vc = []
        for t in ts:
            meta = json.loads((a.body / f"f{t:04d}_{label}.json").read_text(encoding="utf-8"))["metadata"]["people"][0]
            V, F = read_ply(a.body / f"f{t:04d}_{label}_0.ply")
            V = V * [1, -1, -1]
            V[:, 2] += (f_img / meta["focal_length"] - 1) * float(np.median(V[:, 2]))
            Vc.append(V)
        Vc = np.array(Vc)  # [S, V, 3] camera space, body units
        # scale (metres per body unit, i.e. 4RC depth x unit / body depth) on the 4RC frames
        cal_t, cal_g = [], []
        for j, t in enumerate(ts):
            if t not in rc_frames:
                continue
            i = rc_frames.index(t)
            if i not in zmaps:
                zmaps[i] = zmap_of(i)
            m4 = cv2.erode(to_rc_grid(masks[t] == num_of[label]).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
            if m4.sum() < 20:
                continue
            cal_t.append(t)
            cal_g.append(float(np.median(zmaps[i][m4])) * unit / float(np.percentile(Vc[j, :, 2], 30)))
        cal_g = np.array(cal_g)
        med = np.array([np.median(cal_g[max(0, k - 1):k + 2]) for k in range(len(cal_g))])
        g = np.exp(np.interp(ts, cal_t, np.log(med)))
        # world positions
        Xw = np.empty_like(Vc)
        for j, t in enumerate(ts):
            Xw[j] = (Vc[j] * g[j]) @ cam_R[t].T + cam_T[t]
        tt = np.array(ts, float)
        # how well each sample fits its own mask: bad fits (e.g. an upside-down body) are carried by their neighbours
        own_n = num_of[label]
        fit = np.array([silhouette_fit((Xw[j] - cam_T[t]) @ cam_R[t], F, masks[t] == own_n, masks[t] > 0, f_img, W, H)
                        for j, t in enumerate(ts)])
        ref_fit = np.array([np.median(fit[max(0, j - 12):j + 13]) for j in range(len(fit))])
        w = np.clip(fit / np.maximum(ref_fit, 1e-3), 0, 1) ** 4
        w[fit < a.min_fit] = 0.0
        w = np.maximum(w, 1e-3)
        root = Xw.mean(1)
        rel = Xw - root[:, None]
        # turn and shape are smoothed apart: averaging a turning body vertex by vertex would shrink it
        ref = rel[int(np.argmax(fit))]
        Rj = np.array([kabsch(ref, r) for r in rel])
        local = np.einsum("svk,skl->svl", rel, Rj)  # rel @ R: back into the reference orientation
        qs = np.array([rot_to_quat(R) for R in Rj])
        for j in range(1, len(qs)):
            if qs[j] @ qs[j - 1] < 0:
                qs[j] = -qs[j]
        root_s = wsmooth(root, tt, w, a.sigma_root, max_gap=12)
        local_s = wsmooth(local, tt, w, a.sigma_pose, max_gap=12)
        q_s = wsmooth(qs, tt, w, a.sigma_pose, max_gap=12)
        Rs = np.array([quat_to_rot(q) for q in q_s])
        rel_s = np.einsum("svl,skl->svk", local_s, Rs)  # local @ R.T
        Xs = (rel_s + root_s[:, None]).astype(np.float32)
        size = np.ones(len(ts))
        if a.constant_size:
            Xs, size = constant_size(Xs, cam_T[np.array(ts)])
        # colours from the film
        nV = Xs.shape[1]
        samples = np.full((len(ts), nV, 3), np.nan, np.float32)
        own = num_of[label]
        for j, t in enumerate(ts):
            Xc = (Xs[j] - cam_T[t]) @ cam_R[t]  # world -> OpenCV camera
            z = Xc[:, 2]
            u = np.round(f_img * Xc[:, 0] / z + W / 2).astype(int)
            v = np.round(f_img * Xc[:, 1] / z + H / 2).astype(int)
            inside = (z > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
            zbuf = np.full((H, W), np.inf)
            np.minimum.at(zbuf, (v[inside], u[inside]), z[inside])
            zb = cv2.erode(zbuf.astype(np.float32), np.ones((3, 3), np.uint8))  # min over 3x3 (erode of inf-filled = min)
            mask = cv2.erode((masks[t] == own).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
            vis = inside.copy()
            vis[inside] = (z[inside] <= zb[v[inside], u[inside]] * 1.02) & mask[v[inside], u[inside]]
            samples[j, vis] = imgs[t][v[vis], u[vis]] / 255.0
        seen = ~np.isnan(samples[..., 0])
        cnt = seen.sum(0)
        col = np.nanmedian(np.where(seen[..., None], samples, np.nan), axis=0)
        col = fill_colours(np.nan_to_num(col), cnt >= 2, F)
        lin = np.where(col <= 0.04045, col / 12.92, ((col + 0.055) / 1.055) ** 2.4)  # sRGB -> linear for Blender
        out[f"{label}_t"] = np.array(ts, np.int32)
        out[f"{label}_v"] = Xs
        out[f"{label}_f"] = F
        out[f"{label}_col"] = lin.astype(np.float32)
        out[f"{label}_fit"] = fit.astype(np.float32)
        out[f"{label}_w"] = w.astype(np.float32)
        height = np.ptp(Xs[:, :, 2], axis=1)
        report[label] = {"samples": len(ts), "first": ts[0], "last": ts[-1], "cal_frames": len(cal_t),
                         "scale_m_per_body_unit": [round(float(x), 2) for x in np.percentile(g, [0, 50, 100])],
                         "seen_share": round(float((cnt >= 2).mean()), 2),
                         "root_jitter_m": round(float(np.median(np.linalg.norm(np.diff(root, axis=0), axis=1))), 3),
                         "root_jitter_smoothed_m": round(float(np.median(np.linalg.norm(np.diff(root_s, axis=0), axis=1))), 3),
                         "z_extent_median_m": round(float(np.median(height)), 2),
                         "fit_median": round(float(np.median(fit)), 2),
                         "size_spread_before": [round(float(x), 3) for x in np.percentile(size, [5, 95])],
                         "bad_fit_frames": [int(t) for t, x in zip(ts, w) if x < 0.1]}
    out["labels"] = np.array(json.dumps(sorted(files)))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, **out)
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
