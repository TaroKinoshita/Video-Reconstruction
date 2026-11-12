"""Plan steps 1-2 on a real clip: replace masked things with simple proxies, then check them from the source view.

    python scripts/proxies_scene.py VIDEO 4RC.npz MASKS.npz SCENE_DIR OUT_DIR [--kinds obj_net=board,obj_chair=box,obj_stands=plane]
        [--check 0,60,120,180,240,300]

SCENE_DIR is a scripts/place_person.py output (project.json with cameras, floor, person; placement.json with the
4RC alignment). Its world is reused, so the proxies land next to the person. Masks (gpu/observe/run_sam3.py):
set pieces by name (obj_*), people obs_*. The person GVHMR already placed is the obs_* whose mask holds most of its
joints; the other people become boxes (from_4d.initialize, moving ones keyed per frame).
The background mesh is rebuilt with every masked thing's box left out (the proxies stand there now).

Step 2: index pass from the source camera at --check frames, without the background mesh; per mask label the IoU
with the object that replaced it. Writes OUT_DIR/project.json, proxies.json (fits and IoU), background.npz,
scene.blend, check_XXXXXX.png (source | masks | proxies, same colour per label).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    for x in ("video", "rc", "masks", "scene", "out"):
        ap.add_argument(x)
    ap.add_argument("--kinds", default="obj_net=board,obj_chair=ground,obj_stands=plane")
    ap.add_argument("--check", default="0,60,120,180,240,300")
    a = ap.parse_args()

    import cv2

    from vrecon.analysis.background import background_mesh
    from vrecon.analysis.from_4d import _confident, _yaw_quat, initialize, labels_at_model_pixels, load_4rc
    from vrecon.analysis.proxies import FITS, ground_box, near_band
    from vrecon.metrics.poses import camera_pose
    from vrecon.control.core import read_video_rgb
    from vrecon.schema import Evidence, Pose, SceneObject, TransformKey, load_project, save_project

    out, scene = Path(a.out).resolve(), Path(a.scene).resolve()
    out.mkdir(parents=True, exist_ok=True)
    video = read_video_rgb(a.video)
    t, h, w = video.shape[:3]
    z = np.load(a.masks)
    masks, id_map = z["masks"], {int(k): v for k, v in json.loads(str(z["id_map"])).items()}
    assert masks.shape[0] == t, (masks.shape, t)
    if masks.shape[1:] != (h, w):
        masks = np.stack([cv2.resize(x, (w, h), interpolation=cv2.INTER_NEAREST) for x in masks])
    place = json.loads((scene / "placement.json").read_text(encoding="utf-8"))
    A = np.array(place["4rc"]["alignment"])
    cam_h = abs(place["4rc"]["camera_height"])
    base = load_project(scene / "project.json")
    with np.load(a.rc) as zz:
        n = int(zz["n_frames"])
    frames = [int(f) for f in np.linspace(0, t - 1, n, dtype=int)]
    m = load_4rc(a.rc, w, h, frames)
    labels = labels_at_model_pixels(m, masks)

    def aligned(p):
        return p @ A[:3, :3].T + A[:3, 3]

    eye = (A @ m.c2w[0])[:3, 3]
    fps = base.source.fps.num / base.source.fps.den
    cam0 = base.reconstruction.cameras[0]

    def largest_part(mk):
        """One prompt can cover several things ("chair" took two chairs and a bag): keep the largest piece."""
        n_, lab, st, _ = cv2.connectedComponentsWithStats(mk.astype(np.uint8), connectivity=8)
        return lab == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA])) if n_ > 1 else mk

    def ground_object(oid, num, name, depth_ratio):
        """Box from the masks' floor contact per sampled frame (proxies.ground_box); one rest pose unless it moves."""
        per = []
        for f in frames:
            r_, c_, k_ = camera_pose(cam0, f)
            g = ground_box(largest_part(masks[f] == num), np.asarray(r_), np.asarray(c_), k_, depth_ratio)
            if g is not None:
                per.append((f, *g))
        if len(per) < 3:
            return None, {"skipped": f"floor contact seen in {len(per)} sampled frames"}
        cs = np.array([c for _, c, _, _ in per])
        size = tuple(float(v) for v in np.median([s for *_, s in per], 0))
        spread = float(np.linalg.norm(cs[:, :2] - np.median(cs[:, :2], 0), axis=1).max())
        moving = spread > 0.5 * max(size[:2]) * 2
        yaw = float(np.median([y for _, _, y, _ in per]))
        if moving:
            keys = [TransformKey(time=f / fps, pose=Pose(translation=(float(c[0]), float(c[1]), size[2] / 2), rotation=_yaw_quat(y)))
                    for f, c, y, _ in per]
            rest = keys[0]
        else:
            mc = np.median(cs, 0)
            keys, rest = [], TransformKey(time=0.0, pose=Pose(translation=(float(mc[0]), float(mc[1]), size[2] / 2), rotation=_yaw_quat(yaw)))
        o = SceneObject(id=oid, name=name, kind="rigid", size=size, status="estimated", rest=rest, keys=keys,
                        evidence=[Evidence(note="floor contact of the SAM mask (proxies.ground_box)")])
        return o, {"size": [round(v, 4) for v in size], "frames": len(per), "moving": bool(moving)}
    rep = {"fits": {}, "people": {}}
    objs = []
    ev = [Evidence(note="SAM 3.1 mask + 4RC points (scripts/proxies_scene.py)")]
    for spec in a.kinds.split(","):
        name, kind = spec.split("=")
        num = next((k for k, v in id_map.items() if v == name), None)
        if num is None:
            continue
        if kind == "ground":
            o, r = ground_object(name, num, name, depth_ratio=1.0)
            rep["fits"][name] = {"kind": kind, **r}
            if o is not None:
                objs.append(o)
            continue
        p = aligned(_confident(m, labels == num, 0.9))
        p = near_band(p[p[:, 2] > -0.01 * cam_h], eye)
        if len(p) < 50:
            rep["fits"][name] = {"skipped": f"{len(p)} points"}
            continue
        c, q, size, r = FITS[kind](p, eye=eye) if kind == "plane" else FITS[kind](p)
        objs.append(SceneObject(id=name, name=name.removeprefix("obj_"), kind="set" if kind == "plane" else "rigid",
                                size=tuple(float(v) for v in size), status="estimated", evidence=ev,
                                rest=TransformKey(time=0.0, pose=Pose(translation=tuple(float(v) for v in c), rotation=q))))
        rep["fits"][name] = {"kind": kind, "size": [round(float(v), 4) for v in size], **r}

    # people: which mask is the GVHMR person
    sk = np.load(scene / "skeleton.npz")
    pc, kg = sk["joints_cam"], sk["K"]
    uv = np.stack([(pc[i, :, :2] / pc[i, :, 2:]) * [kg[i][0, 0], kg[i][1, 1]] + [kg[i][0, 2], kg[i][1, 2]] for i in range(t)])
    people = {k: v for k, v in id_map.items() if v.startswith("obs_")}
    hits = {}
    for k, v in people.items():
        cnt = 0
        for f in range(0, t, 10):
            ui = np.clip(np.rint(uv[f, :, 0]).astype(int), 0, w - 1)
            vi = np.clip(np.rint(uv[f, :, 1]).astype(int), 0, h - 1)
            cnt += int((masks[f, vi, ui] == k).sum())
        hits[v] = cnt
    main_person = max(hits, key=hits.get)
    rep["people"] = {"joint_hits": hits, "gvhmr_is": main_person}
    others = {k: v for k, v in people.items() if v != main_person}
    _, pobjs, _, prep = initialize(m, masks, others, w, h, "shot_001", fps=fps, alignment=A)
    pobjs = [o for o in pobjs if o.id not in ("obj_floor", "obj_wall")]
    for o in pobjs:
        o.id = o.id.replace("obj_", "obj_person_box_")
    rep["people"]["boxes_4rc"] = prep["objects"]
    rep["people"]["ground"] = {}
    for k, v in others.items():  # floor contact first; the 4RC points' box when the feet are never seen
        oid = "obj_person_box_" + v.removeprefix("obs_").zfill(2)
        o, r = ground_object(oid, k, v, depth_ratio=0.6)
        rep["people"]["ground"][v] = r
        if o is not None:
            pobjs = [x for x in pobjs if x.id != oid] + [o]

    def target(lab):
        if lab == main_person:
            return "obj_person_01"
        if lab.startswith("obs_"):
            return "obj_person_box_" + lab.removeprefix("obs_").zfill(2)
        return lab

    # background without the masked things
    exclude = {}
    for i, f in enumerate(frames):
        boxes = []
        for k in id_map:
            ys, xs = np.nonzero(masks[f] == k)
            if len(xs):
                boxes.append((xs.min() - 4, ys.min() - 4, xs.max() + 4, ys.max() + 4))
        exclude[i] = boxes
    verts, faces, colors, rep_bg = background_mesh(m, A, video, exclude, floor_tol=0.03 * cam_h)
    np.savez_compressed(out / "background.npz", verts=verts, faces=faces, colors=colors)
    shutil.copyfile(scene / "skeleton.npz", out / "skeleton.npz")
    rep["background"] = rep_bg

    p = base.model_copy(deep=True)
    keep = [o for o in p.reconstruction.objects if o.id != "obj_background"]
    bg = next(o for o in p.reconstruction.objects if o.id == "obj_background")
    bg = bg.model_copy(update={"size": tuple(float(x) for x in np.ptp(verts, 0)),
                               "evidence": [Evidence(note="4RC point maps, every masked thing's box left out")]})
    p.reconstruction.objects = keep + objs + pobjs + [bg]
    save_project(p, out / "project.json")

    # step 2: source view check
    from vrecon.scene.builder import build_scene, save_blend
    from vrecon.scene.passes import render_passes

    chk = [int(f) for f in a.check.split(",")]
    q = p.model_copy(deep=True)
    q.reconstruction.objects = [o for o in q.reconstruction.objects if o.id != "obj_background"]
    build_scene(q, root=out)
    render_passes(out / "check_passes", shot_frames=chk, shaded=False)
    idx = np.load(out / "check_passes" / "index.npz")["index"]
    rmap = {int(k): v for k, v in json.loads((out / "check_passes" / "id_map.json").read_text(encoding="utf-8")).items()}
    H2, W2 = idx.shape[1:]
    small = [cv2.resize(masks[f], (W2, H2), interpolation=cv2.INTER_NEAREST) for f in chk]

    ious = {}
    for k, lab in id_map.items():
        tid = target(lab)
        rk = [r for r, v in rmap.items() if v == tid]
        per = []
        for i in range(len(chk)):
            g = small[i] == k
            r = np.isin(idx[i], rk)
            u = (g | r).sum()
            if g.sum():
                per.append(float((g & r).sum() / u) if u else 0.0)
        ious[lab] = {"object": tid, "iou_mean": float(np.mean(per)) if per else None, "per_frame": [round(x, 3) for x in per]}
    rep["check"] = {"frames": chk, "iou": ious}

    pal = np.random.default_rng(3).integers(40, 255, (64, 3)).astype(np.uint8)
    pal[0] = 0
    back = {target(v): k for k, v in id_map.items()}
    for i, f in enumerate(chk):
        ri = np.zeros((H2, W2), np.int64)
        for r, v in rmap.items():
            if v in back:
                ri[idx[i] == r] = back[v]
        img = np.concatenate([cv2.resize(video[f], (W2, H2)), pal[small[i] % 64], pal[ri % 64]], 1)
        ok, buf = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        buf.tofile(out / f"check_{f:06d}.png")
    build_scene(p, root=out)
    save_blend(out / "scene.blend")
    (out / "proxies.json").write_text(json.dumps(rep, indent=1, default=float) + "\n", encoding="utf-8")
    print(json.dumps({"fits": rep["fits"], "gvhmr_is": main_person,
                      "iou": {k: v["iou_mean"] for k, v in ious.items()}}, default=float, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
