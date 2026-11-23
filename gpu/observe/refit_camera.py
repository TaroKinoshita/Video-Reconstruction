"""Camera-only candidate (docs/PLAN_v4.md phase 2): re-fit one cut's camera path to the background tracks.
Runs in the fab env (torch).

    <fab env>/python gpu/observe/refit_camera.py SCENE.npz CUT TRACKS.npz OUT.npz [--iters 1500]

TRACKS.npz from scripts/diagnose_camera.py (id, frame, x, y; background only). Only the "fit" tracks (id % 3 != 0)
are used; the "check" tracks stay for scoring (diagnose_camera.py on OUT.npz). The lens stays as it is.
Unknowns: per frame a small rotation and position change of the scene camera, and one 3D point per fit track
(started by triangulating with the scene cameras). Cost: robust reprojection error (pixels, 1280 wide) + smooth
camera motion + a weak pull to the scene camera (keeps the world, its scale and the other cuts where they are).
The body, set and other cuts are not touched. Writes OUT.npz = SCENE with s<CUT>_cam replaced and
info["refit_camera"][CUT] (errors before and after on the fit tracks).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

CV = np.diag([1.0, -1.0, -1.0])
REF_W = 1280.0
W_SMOOTH_R, W_SMOOTH_C, W_PRIOR = 1e4, 1e3, 1e-2
MIN_OBS = 6


def skew_exp(v: torch.Tensor) -> torch.Tensor:
    """axis-angle [N,3] -> rotation [N,3,3] (Rodrigues)."""
    th = v.norm(dim=1, keepdim=True).clamp(min=1e-9)
    k = v / th
    K = torch.zeros(len(v), 3, 3, device=v.device)
    K[:, 0, 1], K[:, 0, 2], K[:, 1, 0] = -k[:, 2], k[:, 1], k[:, 2]
    K[:, 1, 2], K[:, 2, 0], K[:, 2, 1] = -k[:, 0], -k[:, 1], k[:, 0]
    s, c = torch.sin(th)[..., None], torch.cos(th)[..., None]
    return torch.eye(3, device=v.device) + s * K + (1 - c) * K @ K


def triangulate(Rwc, Cw, f, cx, cy, obs):
    """linear least squares over all observations of one track: obs [(t, x, y)]."""
    A = []
    for t, x, y in obs:
        P = np.c_[Rwc[t], -Rwc[t] @ Cw[t]]
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
        M = K @ P
        A += [x * M[2] - M[0], y * M[2] - M[1]]
    _, _, Vt = np.linalg.svd(np.array(A))
    X = Vt[-1]
    return X[:3] / X[3] if abs(X[3]) > 1e-9 else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("scene", type=Path)
    ap.add_argument("cut", type=int)
    ap.add_argument("tracks", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--iters", type=int, default=1500)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    z = dict(np.load(a.scene))
    info = json.loads(str(z["info"]))
    k = a.cut
    cam = z[f"s{k}_cam"].astype(np.float64)
    T = len(cam)
    tz = np.load(a.tracks)
    w, h = int(tz["w"]), int(tz["h"])
    f = info["shots"][k]["lens_35mm"] / 36.0 * w
    cx, cy = w / 2, h / 2
    s = REF_W / w
    Rc2w = cam[:, :3, :3] @ CV  # OpenCV camera axes in the world
    Rwc, Cw = np.transpose(Rc2w, (0, 2, 1)), cam[:, :3, 3]
    rec = tz["tracks"]
    rec = rec[(rec[:, 0].astype(int) % 3 != 0) & (rec[:, 1] < T)]
    by = {}
    for i, t, x, y in rec:
        by.setdefault(int(i), []).append((int(t), float(x), float(y)))
    pts, obs_i, obs_t, obs_xy = [], [], [], []
    for i, o in by.items():
        if len(o) < MIN_OBS:
            continue
        X = triangulate(Rwc, Cw, f, cx, cy, o)
        if X is None:
            continue
        d = [(Rwc[t] @ (X - Cw[t]))[2] for t, _, _ in o]
        if min(d) < 0.3 or max(d) > 500:
            continue
        j = len(pts)
        pts.append(X)
        for t, x, y in o:
            obs_i.append(j)
            obs_t.append(t)
            obs_xy.append((x, y))
    X = torch.tensor(np.array(pts), dtype=torch.float32, device=dev, requires_grad=True)
    oi = torch.tensor(obs_i, device=dev)
    ot = torch.tensor(obs_t, device=dev)
    oxy = torch.tensor(np.array(obs_xy), dtype=torch.float32, device=dev)
    R0 = torch.tensor(Rc2w, dtype=torch.float32, device=dev)
    C0 = torch.tensor(Cw, dtype=torch.float32, device=dev)
    dr = torch.zeros(T, 3, device=dev, requires_grad=True)
    dC = torch.zeros(T, 3, device=dev, requires_grad=True)

    def residual():
        Rn = R0 @ skew_exp(dr + 1e-12)
        Cn = C0 + dC
        pc = torch.einsum("nji,nj->ni", Rn[ot], X[oi] - Cn[ot])  # R^T (X - C): world -> OpenCV camera
        zc = pc[:, 2].clamp(min=0.05)
        uv = torch.stack([cx + f * pc[:, 0] / zc, cy + f * pc[:, 1] / zc], 1)
        return (uv - oxy).norm(dim=1) * s

    with torch.no_grad():
        e0 = residual()
    rep = {"tracks": len(pts), "observations": len(obs_i),
           "fit_px_before": {"median": round(float(e0.median()), 2), "p95": round(float(e0.quantile(0.95)), 2)}}
    print(rep, flush=True)
    opt = torch.optim.Adam([{"params": [X], "lr": 1e-2}, {"params": [dC], "lr": 3e-3}, {"params": [dr], "lr": 3e-4}])
    for it in range(a.iters):
        if it == a.iters * 2 // 3:
            for g in opt.param_groups:
                g["lr"] *= 0.2
        e = residual()
        loss = torch.where(e < 2.0, 0.5 * e ** 2, 2.0 * (e - 1.0)).mean()  # Huber, 2 px
        loss = loss + W_SMOOTH_R * ((dr[2:] - 2 * dr[1:-1] + dr[:-2]) ** 2).sum(1).mean() \
            + W_SMOOTH_C * ((dC[2:] - 2 * dC[1:-1] + dC[:-2]) ** 2).sum(1).mean() \
            + W_PRIOR * ((dC ** 2).sum(1).mean() + (dr ** 2).sum(1).mean())
        opt.zero_grad()
        loss.backward()
        opt.step()
        if it % 300 == 0:
            print(it, round(float(loss), 4), round(float(e.median()), 2), flush=True)
    with torch.no_grad():
        e1 = residual()
        Rn = (R0 @ skew_exp(dr + 1e-12)).cpu().numpy() @ CV
        Cn = (C0 + dC).cpu().numpy()
    rep["fit_px_after"] = {"median": round(float(e1.median()), 2), "p95": round(float(e1.quantile(0.95)), 2)}
    rep["max_move_m"] = round(float(np.linalg.norm(Cn - Cw, axis=1).max()), 3)
    rep["max_turn_deg"] = round(float(np.degrees(dr.detach().norm(dim=1).max().cpu())), 2)
    new = cam.copy()
    new[:, :3, :3], new[:, :3, 3] = Rn, Cn
    z[f"s{k}_cam"] = new.astype(z[f"s{k}_cam"].dtype)
    info.setdefault("refit_camera", {})[str(k)] = rep
    z["info"] = json.dumps(info)
    np.savez_compressed(a.out, **z)
    print(rep, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
