"""Put the cuts of a multi-cut clip in one world (after scripts/multishot_prep.py).

    python scripts/multishot_join.py PREP.npz JOINT_DIR OUT.npz

JOINT_DIR holds one 4RC run over keyframes of several cuts (frames.json = [[cut, frame], ...] in file order,
4rc.npz). 4RC sees all keyframes together, so their cameras share one world. Per joined cut k a similarity maps
the cut's own (metric, Z-up) world into it:
  rotation   the average of R_joint(f) R_cut(f)^T over the cut's keyframes (SVD projection)
  scale      joint units per metre, from the people (4RC depth of the person mask / GVHMR depth, median over
             keyframes); every cut is already in metres, so the same scale serves all
  shift      mean of c_joint(f) - s R c_cut(f)
The joint world is then scaled to metres and turned so the floor of the cut with a fitted floor is z = 0.
Cuts without keyframes (close-ups) stay in their own slot. Residuals per cut say how well the keyframes agree.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from multishot_prep import CAM_CV_TO_BL, CV_TO_Z, angle_deg, m2q, q2m, slerp  # noqa: E402
from vrecon.analysis.from_4d import load_4rc  # noqa: E402


def proj_rot(m: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(m)
    r = u @ vt
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1
        r = u @ vt
    return r


def umeyama_scale(a: np.ndarray, b: np.ndarray, r: np.ndarray) -> float:
    """scale s with b ~ s r a + t (rotation given)."""
    a0, b0 = a - a.mean(0), b - b.mean(0)
    return float((b0 * (a0 @ r.T)).sum() / max((a0 ** 2).sum(), 1e-12))


def rot_to(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """rotation taking unit u onto unit v."""
    c = np.cross(u, v)
    d = float(u @ v)
    if np.linalg.norm(c) < 1e-9:
        return np.eye(3) if d > 0 else np.diag([1.0, -1.0, -1.0])
    k = np.array([[0, -c[2], c[1]], [c[2], 0, -c[0]], [-c[1], c[0], 0]])
    return np.eye(3) + k + k @ k / (1 + d)


def main() -> int:
    prep, jd, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    z = dict(np.load(prep))
    info = json.loads(str(z["info"]))
    keys = json.loads((jd / "frames.json").read_text(encoding="utf-8"))
    W, H = int(z["W"]), int(z["H"])
    m = load_4rc(jd / "4rc.npz", W, H, list(range(len(keys))))
    c2w = np.linalg.inv(m.c2w[0]) @ m.c2w
    JR = CV_TO_Z @ c2w[:, :3, :3] @ CAM_CV_TO_BL  # joint cameras, Blender axes, Z-up
    JC = c2w[:, :3, 3] @ CV_TO_Z.T
    cuts = sorted({k for k, _ in keys})
    rot, idx = {}, {}
    for k in cuts:
        ii = [i for i, (kk, _) in enumerate(keys) if kk == k]
        ff = [f for kk, f in keys if kk == k]
        cam = z[f"s{k}_cam"]
        rot[k] = proj_rot(sum(JR[i] @ cam[f, :3, :3].T for i, f in zip(ii, ff)))
        idx[k] = (ii, ff)
    # joint units per metre from the people: 4RC depth of the person's mask pixels / GVHMR depth, every keyframe
    # (camera centres cannot give it: most cuts barely move, and the joint run may shrink a long path)
    ratios, diag = [], prep.parent
    for i, (k, f) in enumerate(keys):
        gp = diag / f"gvhmr_A{k:02d}" / "gvhmr_skeleton.npz"
        mp = diag / f"A_shot{k:02d}.masks.npz"
        if not gp.exists() or not mp.exists():
            continue
        mk = np.load(mp)["masks"][f]
        h, w = mk.shape
        ui = np.clip(np.rint(m.src_uv[..., 0] * w / W).astype(int), 0, w - 1)
        vi = np.clip(np.rint(m.src_uv[..., 1] * h / H).astype(int), 0, h - 1)
        sel = mk[vi, ui] > 0
        if sel.sum() < 50:
            continue
        w2c = np.linalg.inv(m.c2w[i])
        zc = (m.pts[i][sel] @ w2c[:3, :3].T + w2c[:3, 3])[:, 2]
        ratios.append(float(np.median(zc) / np.median(np.load(gp)["joints_cam"][f][:, 2])))
    s = float(np.median(ratios))
    ref = "people"
    report = {"scale_from": ref, "joint_units_per_m": s, "scale_spread": [float(np.percentile(ratios, 10)), float(np.percentile(ratios, 90))], "cuts": {}}
    # joint world -> metres, then the floor of a cut that has one to z = 0
    to_m = 1.0 / s
    T = {}
    for k in cuts:
        ii, ff = idx[k]
        cc = z[f"s{k}_cam"][ff, :3, 3]
        t = (JC[ii] - s * cc @ rot[k].T).mean(0)
        R, tt = rot[k], t * to_m  # metres: x_world = R x_cut + tt
        T[k] = (R, tt)
        res_pos = np.linalg.norm(JC[ii] * to_m - (cc @ R.T + tt), axis=1)
        res_rot = [angle_deg(JR[i].T @ R @ z[f"s{k}_cam"][f, :3, :3]) for i, f in zip(ii, ff)]
        report["cuts"][k] = {"keyframes": len(ii), "camera_residual_m": float(np.median(res_pos)),
                             "rotation_residual_deg": float(np.median(res_rot))}
    fl = [k for k in cuts if len(z[f"s{k}_floor"])]
    G, g = np.eye(3), np.zeros(3)
    if fl:
        k = max(fl, key=lambda c: len(idx[c][0]))  # the floor seen from the most keyframes
        R, tt = T[k]
        n = R @ z[f"s{k}_floor"][3:6]
        p = R @ z[f"s{k}_floor"][:3] + tt
        G = rot_to(n / np.linalg.norm(n), np.array([0, 0, 1.0]))
        g = -np.array([0, 0, (G @ p)[2]])
        report["floor_from_cut"] = k
    for k in cuts:
        R, tt = T[k]
        R, tt = G @ R, G @ tt + g
        cam = z[f"s{k}_cam"].copy()
        cam[:, :3, :3] = R @ cam[:, :3, :3]
        cam[:, :3, 3] = cam[:, :3, 3] @ R.T + tt
        z[f"s{k}_cam"] = cam
        for key in ("body", "pts"):
            if len(z[f"s{k}_{key}"]):
                z[f"s{k}_{key}"] = z[f"s{k}_{key}"] @ R.T + tt
        if len(z[f"s{k}_body"]) and "floor_from_cut" in report:
            # the person touches the shared floor at some point (as GVHMR assumes): the whole cut moves up or down
            dz = -float(np.percentile(z[f"s{k}_body"][..., 2].min(1), 2))
            for key in ("body", "pts"):
                z[f"s{k}_{key}"] = z[f"s{k}_{key}"] + np.array([0, 0, dz])
            z[f"s{k}_cam"][:, 2, 3] += dz
            report["cuts"][k]["floor_shift_m"] = dz
        ii, ff = idx[k]
        cam = z[f"s{k}_cam"]
        if len(ii) >= 4:
            # the cut's own path and the joint keyframes disagree: bend the path through the keyframes (a rigid
            # correction per keyframe, interpolated between them); the body rides with its camera
            JRw = G @ JR[ii]
            JCw = (JC[ii] * to_m) @ G.T + g + np.array([0, 0, report["cuts"][k].get("floor_shift_m", 0.0)])
            n_t = len(cam)
            dq, dt = [], []
            for j, f in enumerate(ff):
                D = JRw[j] @ cam[f, :3, :3].T
                dq.append(m2q(D[None])[0])
                dt.append(JCw[j] - D @ cam[f, :3, 3])
            dq, dt = np.array(dq), np.array(dt)
            before = float(np.linalg.norm(np.ptp(z[f"s{k}_body"][:, 0, :2], 0))) if len(z[f"s{k}_body"]) else None
            newcam, newbody = cam.copy(), z[f"s{k}_body"].copy()
            for t in range(n_t):
                x = np.clip(t, ff[0], ff[-1])
                a = min(int(np.searchsorted(ff, x, side="right")) - 1, len(ff) - 2)
                u = (x - ff[a]) / max(ff[a + 1] - ff[a], 1)
                D = q2m(slerp(dq[a], dq[a + 1], u)[None])[0]
                d = (1 - u) * dt[a] + u * dt[a + 1]
                newcam[t, :3, :3] = D @ cam[t, :3, :3]
                newcam[t, :3, 3] = D @ cam[t, :3, 3] + d
                if t < len(newbody):
                    newbody[t] = newbody[t] @ D.T + d
            after = float(np.linalg.norm(np.ptp(newbody[:, 0, :2], 0))) if len(newbody) else None
            report["cuts"][k]["pelvis_spread_m"] = {"own_path": before, "bent_path": after}
            if after is not None and before is not None and after < 0.9 * before:  # keep the bend only when the person
                z[f"s{k}_cam"], z[f"s{k}_body"] = newcam, newbody  # stays more in place (a still subject test)
                report["cuts"][k]["path_bent"] = True
        info["shots"][k]["slot"] = 0
        if k != report.get("floor_from_cut", cuts[0]):
            info["shots"][k]["shared_fill_from"] = report.get("floor_from_cut", cuts[0])
    if "floor_from_cut" in report:  # its floor is now z = 0 by construction
        k = report["floor_from_cut"]
        f = z[f"s{k}_floor"].copy()
        f[:3] = [z[f"s{k}_body"][:, 0, 0].mean(), z[f"s{k}_body"][:, 0, 1].mean(), 0.0]
        f[3:6] = [0, 0, 1]
        z[f"s{k}_floor"] = f
    slot = 1
    for k, sh in enumerate(info["shots"]):
        if "slot" not in sh:
            sh["slot"] = slot
            slot += 1
    info["joined"] = report
    z["info"] = json.dumps(info)
    np.savez_compressed(out, **z)
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
