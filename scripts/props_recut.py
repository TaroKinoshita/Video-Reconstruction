"""Anchored segmentation (docs/PROPS_PLAN.md, part 1): cut windows of the video around a body part that touches
the prop (e.g. the hand that tosses a ball), so the segmenter looks only where the prop can be; then put its
masks back into full frames. Works for any prop and any joint of a placed GVHMR person; no ground truth.

    python scripts/props_recut.py prepare PROJECT.json VIDEO OUT_DIR --person obj_person_01 --joint left_wrist
        [--reach 1.2] [--up 1.5] [--frames 70-140,195-245 | --auto] [--size 640]
    (GPU env) python gpu/observe/run_sam3.py OUT_DIR/win_XX.mp4 OUT_DIR/win_XX.npz --movers "tennis ball" ...
    python scripts/props_recut.py merge OUT_DIR MERGED.npz

prepare: per window a clip whose frames follow the joint's pixel (smoothed), a box `reach` metres to each side and
below and `reach + up` above (a toss goes up), converted to pixels at the joint's depth, one box size per window;
windows.json records each frame's box. --auto proposes windows where the joint rises above the head joint or moves
fast (a toss, a throw, a swing), padded. merge: masks of every window (labels kept apart per window as wN_label)
pasted back into full frames, same npz format as run_sam3.py (masks, id_map, meta).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from vrecon.analysis.place_person import _rays
from vrecon.control.core import encode_mp4, read_video_rgb
from vrecon.schema import load_project


def joint_track(project_path: Path, person_id: str, joints: list[str]):
    """Pixel position and camera depth (scene units) per frame of each named joint of a placed GVHMR person."""
    from vrecon.analysis.from_gvhmr import load_skeleton
    from vrecon.metrics.poses import camera_pose

    project = load_project(project_path)
    obj = next(o for o in project.reconstruction.objects if o.id == person_id)
    cam = project.reconstruction.cameras[0]
    d = load_skeleton(project_path.parent / obj.asset.path)
    names = list(d["joint_names"])
    s = obj.keys[0].scale[0]
    out = {j: ([], []) for j in joints}
    for f, (r, c, q) in enumerate(_rays(d, cam, 0)):
        pw = c + (q * s) @ r.T
        R, cc, K = camera_pose(cam, f)
        for j in joints:
            pc = (pw[names.index(j)] - cc) @ R
            out[j][0].append(pc[:2] / pc[2] * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]])
            out[j][1].append(pc[2])
    fx = cam.intrinsics.fx
    return {j: (np.array(u), np.array(z)) for j, (u, z) in out.items()}, fx, s


def auto_windows(uv, head_uv, fps, pad=0.5, speed_m=2.5, z=None, fx=None, s=1.0, min_len=8):
    """Frames where the joint is above the head or moves faster than speed_m (m/s, pixel speed at its depth)."""
    above = uv[:, 1] < head_uv[:, 1]
    sp = np.r_[0, np.linalg.norm(np.diff(uv, axis=0), axis=1)] * fps
    if z is not None and fx:
        sp = sp * z / fx / s
    act = above | (sp > speed_m)
    p = int(round(pad * fps))
    grown = np.convolve(act.astype(int), np.ones(2 * p + 1, int), mode="same") > 0
    wins, f = [], 0
    while f < len(grown):
        if grown[f]:
            g = f
            while g < len(grown) and grown[g]:
                g += 1
            if g - f >= min_len:
                wins.append((f, g - 1))
            f = g
        else:
            f += 1
    return wins


def _smooth(x, sigma=2.0):
    k = np.exp(-0.5 * (np.arange(-6, 7) / sigma) ** 2)
    k /= k.sum()
    pad = np.pad(x, ((6, 6), (0, 0)), mode="edge")
    return np.stack([np.convolve(pad[:, i], k, mode="valid") for i in range(x.shape[1])], 1)


def prepare(a) -> int:
    proj = Path(a.project)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tr, fx, s = joint_track(proj, a.person, [a.joint, "head"])
    uv, z = tr[a.joint]
    project = load_project(proj)
    fps = project.source.fps.num / project.source.fps.den
    if a.frames:
        wins = [tuple(int(x) for x in w.split("-")) for w in a.frames.split(",")]
        how = "given"
    else:
        wins = auto_windows(uv, tr["head"][0], fps, z=z, fx=fx, s=s)
        how = "auto: joint above the head or faster than 2.5 m/s, padded 0.5 s"
    video = read_video_rgb(a.video)
    h, w = video.shape[1:3]
    uvs = _smooth(uv)
    rec = {"project": str(proj), "video": a.video, "person": a.person, "joint": a.joint, "reach_m": a.reach,
           "up_m": a.up, "size": a.size, "windows_from": how, "full_size": [w, h], "windows": []}
    for k, (lo, hi) in enumerate(wins):
        px = float(np.median(z[lo:hi + 1]))
        r = a.reach * s * fx / px  # metres -> pixels at the joint's depth
        bw, bh = 2 * r, 2 * r + a.up * s * fx / px
        side = max(bw, bh)
        frames, boxes = [], []
        for f in range(lo, hi + 1):
            cx, cy = uvs[f]
            x0 = cx - side / 2
            y0 = cy + r - side  # the box's bottom is `reach` below the joint, the rest above
            M = np.array([[a.size / side, 0, -x0 * a.size / side], [0, a.size / side, -y0 * a.size / side]])
            frames.append(cv2.warpAffine(video[f], M, (a.size, a.size), flags=cv2.INTER_LINEAR,
                                         borderMode=cv2.BORDER_CONSTANT))
            boxes.append([float(x0), float(y0), float(side)])
        name = f"win_{k:02d}"
        encode_mp4(np.array(frames), out / f"{name}.mp4", fps=fps, crf=12)
        rec["windows"].append({"name": name, "frames": [lo, hi], "boxes": boxes, "box_px": round(side, 1)})
        print(f"{name}: frames {lo}-{hi}, box {side:.0f} px -> {a.size} px")
    (out / "windows.json").write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
    return 0


def merge(a) -> int:
    d = Path(a.dir)
    rec = json.loads((d / "windows.json").read_text(encoding="utf-8"))
    w, h = rec["full_size"]
    t_n = max(win["frames"][1] for win in rec["windows"]) + 1
    project = load_project(Path(rec["project"]))
    t_n = max(t_n, project.source.frame_count)
    masks = np.zeros((t_n, h, w), np.uint16)
    id_map, metas = {}, []
    for k, win in enumerate(rec["windows"]):
        p = d / f"{win['name']}.npz"
        if not p.exists():
            print("missing", p)
            continue
        z = np.load(p, allow_pickle=True)
        local = json.loads(str(z["id_map"]))
        metas.append({"window": win["name"], **json.loads(str(z["meta"]))})
        remap = {}
        for num, lab in local.items():
            n = len(id_map) + 1
            id_map[n] = f"w{k}_{lab}"
            remap[int(num)] = n
        m = z["masks"]
        lo = win["frames"][0]
        for i in range(m.shape[0]):
            x0, y0, side = win["boxes"][i]
            sc = side / m.shape[2]
            M = np.array([[sc, 0, x0], [0, sc, y0]])
            full = cv2.warpAffine(m[i].astype(np.float32), M, (w, h), flags=cv2.INTER_NEAREST,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0).astype(np.uint16)
            for num, n in remap.items():
                masks[lo + i][full == num] = n
    meta = {"tool": "props_recut merge", "windows": rec, "per_window": metas}
    with open(a.out, "wb") as f:
        np.savez_compressed(f, masks=masks, id_map=json.dumps(id_map), meta=json.dumps(meta))
    print(json.dumps({"labels": id_map, "frames_with_mask": {lab: int((masks == n).any((1, 2)).sum())
                                                              for n, lab in id_map.items()}}))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("project")
    p.add_argument("video")
    p.add_argument("out")
    p.add_argument("--person", default="obj_person_01")
    p.add_argument("--joint", default="left_wrist")
    p.add_argument("--reach", type=float, default=1.2, help="metres around the joint")
    p.add_argument("--up", type=float, default=1.5, help="extra metres above (tosses, throws)")
    p.add_argument("--frames", default=None, help="windows lo-hi,lo-hi (default: --auto)")
    p.add_argument("--size", type=int, default=640)
    m = sub.add_parser("merge")
    m.add_argument("dir")
    m.add_argument("out")
    a = ap.parse_args()
    return prepare(a) if a.cmd == "prepare" else merge(a)


if __name__ == "__main__":
    sys.exit(main())
