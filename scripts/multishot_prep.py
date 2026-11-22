"""Per-shot camera work, body and background of a multi-cut clip, ready for Blender (numpy only).

    python scripts/multishot_prep.py DIAG_DIR PREFIX OUT.npz

Reads DIAG_DIR/shots_<X>.json, <PREFIX><kk>.mp4, <PREFIX><kk>.masks.npz, rc_<X><kk>/4rc.npz, gvhmr_<X><kk>/.
Per shot k (frames of the source clip [a, b)):
  camera   4RC camera-to-world at its sampled frames, interpolated to every frame (slerp + lerp), in a Z-up world
           whose origin is the shot's first camera. No 4RC: a still camera at the origin.
  scale    4RC has no units. When GVHMR saw a person: metres per 4RC unit = median GVHMR person depth / median
           4RC depth of the person's mask pixels. Else 1 (marked "unitless").
  focal    GVHMR's (the body is placed through it) when there is a body, else 4RC's; as a 35 mm-equivalent width.
  body     GVHMR joints in the camera frame, put in the world through that frame's camera (22 SMPL-X joints).
  points   confident 4RC points outside the person masks of the middle sampled frame, with source colours.
Shots are separate worlds (one film cut = one camera setup); they are not joined here.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vrecon.analysis.from_4d import fit_plane, labels_at_model_pixels, load_4rc  # noqa: E402
from vrecon.control.core import read_video_rgb  # noqa: E402

CV_TO_Z = np.array([[1.0, 0, 0], [0, 0, 1.0], [0, -1.0, 0]])  # OpenCV world (y down, z ahead) -> Z up
CAM_CV_TO_BL = np.diag([1.0, -1.0, -1.0])  # OpenCV camera axes -> Blender camera axes


def sampled(n: int) -> list[int]:
    return list(np.linspace(0, n - 1, 30, dtype=int)) if n > 30 else list(range(n))


def m2q(m):
    """rotation matrices [N,3,3] -> unit quaternions (w,x,y,z)"""
    q = []
    for r in m:
        t = np.trace(r)
        if t > 0:
            s = np.sqrt(t + 1) * 2
            q.append([s / 4, (r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s])
        else:
            i = int(np.argmax(np.diag(r))); j, k = (i + 1) % 3, (i + 2) % 3
            s = np.sqrt(1 + r[i, i] - r[j, j] - r[k, k]) * 2
            v = [0.0, 0.0, 0.0]; v[i] = s / 4; v[j] = (r[j, i] + r[i, j]) / s; v[k] = (r[k, i] + r[i, k]) / s
            q.append([(r[k, j] - r[j, k]) / s] + v)
    q = np.array(q)
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def q2m(q):
    w, x, y, z = q.T
    return np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
                     np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
                     np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], 1)


def slerp(q0, q1, u):
    d = float(q0 @ q1)
    if d < 0:
        q1, d = -q1, -d
    if d > 0.9995:
        q = q0 + u * (q1 - q0)
        return q / np.linalg.norm(q)
    th = np.arccos(d)
    return (np.sin((1 - u) * th) * q0 + np.sin(u * th) * q1) / np.sin(th)


def angle_deg(r):
    return float(np.degrees(np.arccos(np.clip((np.trace(r) - 1) / 2, -1, 1))))


def interp_c2w(frames: list[int], c2w: np.ndarray, n: int) -> np.ndarray:
    if len(frames) == 1:
        return np.repeat(c2w[:1], n, 0)
    t = np.arange(n)
    qs = m2q(c2w[:, :3, :3])
    rq = []
    for x in np.clip(t, frames[0], frames[-1]):
        i = min(int(np.searchsorted(frames, x, side="right")) - 1, len(frames) - 2)
        rq.append(slerp(qs[i], qs[i + 1], (x - frames[i]) / max(frames[i + 1] - frames[i], 1)))
    rot = q2m(np.array(rq))
    pos = np.stack([np.interp(t, frames, c2w[:, i, 3]) for i in range(3)], 1)
    out = np.tile(np.eye(4), (n, 1, 1))
    out[:, :3, :3], out[:, :3, 3] = rot, pos
    return out


def shot(diag: Path, tag: str, k: int, prefix: str) -> dict:
    vid = diag / f"{prefix}{k:02d}.mp4"
    rgb = read_video_rgb(vid)
    n, H, W = rgb.shape[:3]
    mz = np.load(diag / f"{prefix}{k:02d}.masks.npz")
    masks = mz["masks"]
    if masks.shape[1:] != (H, W):  # cut from a smaller copy: nearest-neighbour up
        yi = (np.arange(H) * masks.shape[1] // H)
        xi = (np.arange(W) * masks.shape[2] // W)
        masks = masks[:, yi][:, :, xi]
    gv_path = diag / f"gvhmr_{tag}{k:02d}" / "gvhmr_skeleton.npz"
    gv = np.load(gv_path) if gv_path.exists() else None
    rc_path = diag / f"rc_{tag}{k:02d}" / "4rc.npz"
    info = {"shot": k, "frames": n, "has_4rc": rc_path.exists(), "has_body": gv is not None}
    if rc_path.exists():
        fr = sampled(n)
        m = load_4rc(rc_path, W, H, fr)
        c2w_rc = np.linalg.inv(m.c2w[0]) @ m.c2w  # origin = the shot's first sampled camera
        labels = labels_at_model_pixels(m, masks)
        # 4RC depth of the person in each sampled frame
        scale = 1.0
        if gv is not None:
            zs = []
            for i, f in enumerate(fr):
                sel = labels[i] > 0
                if sel.sum() < 50 or f >= len(gv["joints_cam"]):
                    continue
                p = m.pts[i][sel]
                w2c = np.linalg.inv(m.c2w[i])
                zc = (p @ w2c[:3, :3].T + w2c[:3, 3])[:, 2]
                zs.append((float(np.median(gv["joints_cam"][f][:, 2])), float(np.median(zc))))
            if zs:
                z = np.array(zs)
                scale = float(np.median(z[:, 0] / np.maximum(z[:, 1], 1e-6)))
        info["scale"] = scale
        info["unitless"] = gv is None
        fx = float(np.median(m.K[:, 0, 0]))
        c2w = interp_c2w(fr, c2w_rc, n)
        c2w[:, :3, 3] *= scale
        # background from every sampled frame (the set is assumed still), people left out, merged on a voxel grid
        w0 = np.linalg.inv(m.c2w[0])
        P, C = [], []
        for i, f in enumerate(fr):
            sel = (labels[i] == 0) & (m.conf[i] >= np.quantile(m.conf[i], 0.6))
            P.append((m.pts[i][sel] @ w0[:3, :3].T + w0[:3, 3]) * scale)
            uv = np.rint(m.src_uv[sel]).astype(int)
            C.append(rgb[f][np.clip(uv[:, 1], 0, H - 1), np.clip(uv[:, 0], 0, W - 1)] / 255.0)
        pts, col = np.concatenate(P), np.concatenate(C)
        if gv is not None:  # coat edges and blur the masks missed: nothing within 0.6 m of the body, any frame
            ff = [f for f in fr if f < len(gv["joints_cam"])]
            jc_all = gv["joints_cam"][ff]
            bw = (np.einsum("tij,tkj->tki", c2w[ff, :3, :3], jc_all) + c2w[ff, None, :3, 3]).reshape(-1, 3)
            d = np.full(len(pts), np.inf)
            for j in range(0, len(bw), 1):
                d = np.minimum(d, np.linalg.norm(pts - bw[j], axis=1))
            pts, col = pts[d > 0.6], col[d > 0.6]
        near = np.median(np.linalg.norm(pts - c2w[:, :3, 3].mean(0), axis=1))
        vox = near / 250  # about one source pixel at the typical distance, a little coarser
        key = np.floor(pts / vox).astype(np.int64)
        _, first = np.unique(key, axis=0, return_index=True)
        pts, col = pts[first], col[first]
        keep = np.random.default_rng(0).permutation(len(pts))[:250000]
        pts, col = pts[keep], col[keep]
        info["point_radius"] = float(vox * 1.5)
        info["background_points"] = int(len(pts))
    else:
        c2w = np.tile(np.eye(4), (n, 1, 1))
        fx = 1.2 * W
        pts, col = np.zeros((0, 3)), np.zeros((0, 3))
        info["scale"], info["unitless"], info["point_radius"] = 1.0, True, 0.03
    if gv is not None:
        fx = float(np.median(gv["K"][:, 0, 0]))
        jc = gv["joints_cam"][:n]
        body = np.einsum("tij,tkj->tki", c2w[: len(jc), :3, :3], jc) + c2w[: len(jc), None, :3, 3]
    else:
        body = np.zeros((0, 22, 3))
    # to the Z-up world and Blender camera axes
    cam = np.tile(np.eye(4), (n, 1, 1))
    cam[:, :3, :3] = CV_TO_Z @ c2w[:, :3, :3] @ CAM_CV_TO_BL
    cam[:, :3, 3] = c2w[:, :3, 3] @ CV_TO_Z.T
    info["lens_35mm"] = 36.0 * fx / W
    info["camera_travel_m"] = float(np.linalg.norm(np.diff(cam[:, :3, 3], axis=0), axis=1).sum())
    info["camera_turn_deg"] = angle_deg(cam[0, :3, :3].T @ cam[-1, :3, :3])
    body = body @ CV_TO_Z.T if len(body) else body
    pts = pts @ CV_TO_Z.T
    # fill what the points cannot: the sky / far backdrop (colour of the frame's top rows) and, when a body
    # stands on it, the floor as a plane (it is hidden under the person in every frame)
    info["backdrop_rgb"] = [float(v) for v in np.median(rgb[:, : max(H // 12, 1)].reshape(-1, 3), 0) / 255.0]
    floor = np.zeros(0)
    if len(body) and len(pts) > 1000:
        nrm, d, _ = fit_plane(pts)
        inl = np.abs(pts @ nrm + d) < 0.05
        feet = body[:, [7, 8, 10, 11]].reshape(-1, 3)
        gap = float(np.percentile(np.abs(feet @ nrm + d), 10))
        if inl.mean() > 0.2 and gap < 0.3:
            if nrm @ (body[:, 15].mean(0) - body[:, 0].mean(0)) < 0:
                nrm = -nrm; d = -d
            c = body[:, 0].mean(0)
            c = c - (c @ nrm + d) * nrm
            floor = np.concatenate([c, nrm, np.median(col[inl], 0)])
        info["floor"] = {"inlier_share": float(inl.mean()), "feet_gap_m": gap, "used": bool(len(floor))}
    return {"cam": cam, "body": body, "pts": pts, "col": col, "floor": floor, "info": info, "W": W, "H": H}


def main() -> int:
    diag, prefix, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
    tag = prefix.split("_")[0]
    sh = json.loads((diag / f"shots_{tag}.json").read_text(encoding="utf-8"))
    flat, infos = {}, []
    for k, (a, b) in enumerate(sh["shots"]):
        s = shot(diag, tag, k, prefix)
        s["info"]["source_frames"] = [a, b]
        infos.append(s["info"])
        for key in ("cam", "body", "pts", "col", "floor"):
            flat[f"s{k}_{key}"] = s[key]
        print(json.dumps(s["info"]))
    flat["W"], flat["H"] = s["W"], s["H"]
    np.savez_compressed(out, info=json.dumps({"shots": infos, "transitions": sh["transitions"]}), **flat)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
