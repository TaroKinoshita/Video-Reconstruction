"""Lay out one shot as data for a Blender scene (self-built base, small test on MAVEBench): camera per video frame from
4RC, background surfaces from 4RC point maps of the keyframes, and one SAM 3D Body mesh per person per keyframe placed
in the same space. Pure numpy; src/vrecon/scene/mave_blend.py turns the result into scene.blend inside Blender 4.5.

  python scripts/v11_mave_prep.py RC.npz RC_FRAMES.json PEOPLE.masks.npz BODY_DIR OUT.npz \
      --size 960 408 --keyframes 0 34 68 ... --frames 306

Inputs are only the video's own analysis (4RC, SAM 3.1 masks, SAM 3D Body); nothing from the benchmark's ground truth.
Conventions: 4RC extrinsic = camera-to-world, OpenCV axes (x right, y down, z forward); 4RC resizes the long side to
512 and centre-crops to multiples of 14. SAM 3D Body meshes are in OpenGL camera axes (y up, z back) with the
translation applied, focal length "focal_length" at the image it was given, principal point at the image centre.
Output world: 4RC world rescaled so people come out at their body-model size (about metres), then turned so -y(4RC)
is +z (Blender up, only roughly: 4RC's world follows the first camera, not gravity).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

A = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], float)  # 4RC world (y down, z forward) -> z up


def quat(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> unit quaternion (w, x, y, z)."""
    w = np.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = np.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2
    y = np.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2
    z = np.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2
    x = np.copysign(x, R[2, 1] - R[1, 2])
    y = np.copysign(y, R[0, 2] - R[2, 0])
    z = np.copysign(z, R[1, 0] - R[0, 1])
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def mat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def slerp(q0: np.ndarray, q1: np.ndarray, u: float) -> np.ndarray:
    if np.dot(q0, q1) < 0:
        q1 = -q1
    c = np.clip(np.dot(q0, q1), -1, 1)
    if c > 0.9995:
        return (q0 + u * (q1 - q0)) / np.linalg.norm(q0 + u * (q1 - q0))
    th = np.arccos(c)
    return (np.sin((1 - u) * th) * q0 + np.sin(u * th) * q1) / np.sin(th)


def read_ply(path: Path) -> tuple[np.ndarray, np.ndarray]:
    b = path.read_bytes()
    i = b.index(b"end_header\n") + 11
    head = b[:i].decode().split("\n")
    nv = int(next(l for l in head if l.startswith("element vertex")).split()[-1])
    nf = int(next(l for l in head if l.startswith("element face")).split()[-1])
    v = np.frombuffer(b[i:i + nv * 16], np.dtype([("p", "<f4", 3), ("c", "u1", 4)]))["p"].astype(np.float64)
    f = np.frombuffer(b[i + nv * 16:i + nv * 16 + nf * 13], np.dtype([("n", "u1"), ("i", "<i4", 3)]))
    assert (f["n"] == 3).all()
    return v, f["i"].astype(np.int32)


def grid_mesh(P: np.ndarray, depth: np.ndarray, keep: np.ndarray, step: int, max_ratio: float):
    """Triangulate a point map on its pixel grid; drop triangles touching dropped pixels or spanning a depth jump."""
    P, depth, keep = P[::step, ::step], depth[::step, ::step], keep[::step, ::step]
    h, w = keep.shape
    idx = np.arange(h * w).reshape(h, w)
    a, b, c, d = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel(), idx[1:, :-1].ravel(), idx[1:, 1:].ravel()
    tris = np.concatenate([np.stack([a, c, b], 1), np.stack([b, c, d], 1)])
    k, z = keep.ravel(), depth.ravel()
    zt = z[tris]
    ok = k[tris].all(1) & (zt.max(1) / np.maximum(zt.min(1), 1e-9) < max_ratio)
    tris = tris[ok]
    used = np.unique(tris)
    remap = -np.ones(h * w, np.int64)
    remap[used] = np.arange(len(used))
    return P.reshape(-1, 3)[used], remap[tris].astype(np.int32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("rc", type=Path)
    ap.add_argument("rc_frames", type=Path)
    ap.add_argument("masks", type=Path)
    ap.add_argument("body", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--size", type=int, nargs=2, required=True, help="W H of the video the masks/bodies were made on")
    ap.add_argument("--keyframes", type=int, nargs="+", required=True)
    ap.add_argument("--frames", type=int, required=True)
    ap.add_argument("--step", type=int, default=2, help="point-map subsampling for the background mesh")
    ap.add_argument("--max-ratio", type=float, default=1.08, help="depth jump that cuts the background mesh")
    a = ap.parse_args()
    W, H = a.size
    rc = np.load(a.rc)
    rc_frames = json.loads(a.rc_frames.read_text(encoding="utf-8"))
    md = np.load(a.masks)
    masks, id_map = md["masks"], json.loads(str(md["id_map"]))

    # 4RC image geometry: resize long side to 512, centre-crop
    P0 = rc["pred_0_pts"].reshape(rc["pred_0_pts"].shape[-3:])
    h4, w4 = P0.shape[:2]
    s = 512 / max(W, H)
    rw, rh = round(W * s), round(H * s)
    ox, oy = rw // 2 - w4 // 2, rh // 2 - h4 // 2

    c2w = np.stack([rc[f"pred_{i}_extrinsic"].reshape(4, 4) for i in range(len(rc_frames))])
    # one square-pixel focal fitted to the point maps of all frames (4RC's fx and fy differ)
    num = den = 0.0
    for i in range(len(rc_frames)):
        P = rc[f"pred_{i}_pts"].reshape(h4, w4, 3)
        K = rc[f"pred_{i}_intrinsic"].reshape(3, 3)
        w2c = np.linalg.inv(c2w[i])
        X = P.reshape(-1, 3) @ w2c[:3, :3].T + w2c[:3, 3]
        gu, gv = np.meshgrid(np.arange(w4) - K[0, 2], np.arange(h4) - K[1, 2])
        xn, yn = X[:, 0] / X[:, 2], X[:, 1] / X[:, 2]
        num += (xn * gu.ravel()).sum() + (yn * gv.ravel()).sum()
        den += (xn * xn).sum() + (yn * yn).sum()
    f4 = num / den
    f_img = f4 / s  # at W x H
    fx_mean = float(np.mean([rc[f"pred_{i}_intrinsic"].reshape(3, 3)[0, 0] for i in range(len(rc_frames))]))
    fy_mean = float(np.mean([rc[f"pred_{i}_intrinsic"].reshape(3, 3)[1, 1] for i in range(len(rc_frames))]))

    def to_rc_grid(m: np.ndarray) -> np.ndarray:
        r = cv2.resize(m.astype(np.uint8), (rw, rh), interpolation=cv2.INTER_NEAREST)
        return r[oy:oy + h4, ox:ox + w4].astype(bool)

    # people: body meshes per keyframe, focal changed to ours, depth set from 4RC inside the mask
    people, ks = [], []
    for kf in a.keyframes:
        i = rc_frames.index(kf)
        P = rc[f"pred_{i}_pts"].reshape(h4, w4, 3)
        w2c = np.linalg.inv(c2w[i])
        zmap = (P.reshape(-1, 3) @ w2c[:3, :3].T + w2c[:3, 3])[:, 2].reshape(h4, w4)
        for num_s, label in id_map.items():
            j = a.body / f"f{kf:04d}_{label}.json"
            ply = a.body / f"f{kf:04d}_{label}_0.ply"
            if not j.exists() or not ply.exists():
                continue
            meta = json.loads(j.read_text(encoding="utf-8"))["metadata"]["people"][0]
            V, F = read_ply(ply)
            V = V * [1, -1, -1]  # OpenGL camera -> OpenCV camera
            zr = float(np.median(V[:, 2]))
            V[:, 2] += (f_img / meta["focal_length"] - 1) * zr  # same picture under our focal
            m4 = cv2.erode(to_rc_grid(masks[kf] == int(num_s)).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
            if m4.sum() < 20:
                continue
            d4 = float(np.median(zmap[m4]))
            # front surface of the body (nearest 30% of vertices) against the 4RC surface inside the mask
            zb = float(np.percentile(V[:, 2], 30))
            k = d4 / zb  # 4RC units per body unit for this person on this keyframe
            ks.append(k)
            people.append({"label": label, "kf": kf, "V": V * k, "F": F, "k": k})
    k_med = float(np.median(ks)) if ks else 1.0
    unit = 1 / k_med  # world scale: 4RC units -> about metres

    def world(Xc: np.ndarray, i: int) -> np.ndarray:
        X = Xc @ c2w[i][:3, :3].T + c2w[i][:3, 3]
        return (X * unit) @ A.T

    # background: 4RC point maps of the keyframes minus people (dilated) and low confidence
    bg_v, bg_f, off = [], [], 0
    for kf in a.keyframes:
        i = rc_frames.index(kf)
        P = rc[f"pred_{i}_pts"].reshape(h4, w4, 3)
        conf = rc[f"pred_{i}_conf"].reshape(h4, w4)
        w2c = np.linalg.inv(c2w[i])
        zmap = (P.reshape(-1, 3) @ w2c[:3, :3].T + w2c[:3, 3])[:, 2].reshape(h4, w4)
        person = to_rc_grid(masks[kf] > 0)
        person = cv2.dilate(person.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
        keep = ~person & (conf > np.percentile(conf, 10)) & (zmap > 0)
        v, f = grid_mesh(P, zmap, keep, a.step, a.max_ratio)
        bg_v.append((v * unit) @ A.T)
        bg_f.append(f + off)
        off += len(v)

    # camera per video frame: slerp / linear between the 4RC frames, held at the ends
    times = np.array(rc_frames, float)
    qs = [quat(M[:3, :3]) for M in c2w]
    t_all = np.clip(np.arange(a.frames), times[0], times[-1])
    R = []
    for t in t_all:
        j = min(int(np.searchsorted(times, t, side="right")) - 1, len(times) - 2)
        R.append(mat(slerp(qs[j], qs[j + 1], (t - times[j]) / (times[j + 1] - times[j]))))
    R = np.array(R)
    T = np.stack([np.interp(t_all, times, c2w[:, d, 3]) for d in range(3)], 1)
    cam_R = A @ R  # camera axes (OpenCV) in output world
    cam_T = (T * unit) @ A.T

    out = {"cam_R": cam_R, "cam_T": cam_T, "bg_v": np.concatenate(bg_v).astype(np.float32),
           "bg_f": np.concatenate(bg_f).astype(np.int32)}
    plist = []
    for n, p in enumerate(people):
        i = rc_frames.index(p["kf"])
        out[f"p{n}_v"] = world(p["V"], i).astype(np.float32)
        out[f"p{n}_f"] = p["F"]
        plist.append({"label": p["label"], "kf": p["kf"], "k_rel": p["k"] / k_med})
    info = {"size": [W, H], "f_img": f_img, "f_4rc_fit": f4, "fx_4rc_mean": fx_mean, "fy_4rc_mean": fy_mean,
            "frames": a.frames, "keyframes": a.keyframes, "people": plist, "unit_4rc_to_body": unit}
    out["info"] = np.array(json.dumps(info))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, **out)
    print(json.dumps({k: v for k, v in info.items() if k != "people"}, indent=1))
    print("people instances", len(plist), "k_rel spread", np.round(np.percentile([p["k_rel"] for p in plist], [0, 25, 50, 75, 100]), 2).tolist() if plist else None)
    print("background", len(out["bg_v"]), "verts", len(out["bg_f"]), "tris")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
