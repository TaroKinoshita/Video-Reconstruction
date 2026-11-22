"""Objects found by gpu/observe/find_objects_sam3.py, placed in the shared world as proxies.

    python scripts/multishot_objects.py IN.npz DIAG_DIR TAG OUT.npz

Per cut and object track (cuts in the shared world, slot 0, including placed close-ups):
  held    the mask's centre stays near a projected wrist of the cut's body (median distance under the mask's size
          plus HAND_PX): the object is in that hand; each frame it sits on the ray through the mask centre at the
          wrist's depth, its size is the mask box at that depth
  loose   else, when the mask's lowest row is inside the picture: it stands on the floor where the ray through the
          bottom centre meets z = 0 (as people, multishot_people.py); size from the box at that distance
  close-up else, in a placed close-up: on the ray through the mask centre at the body's median depth
  else    not placed (reported): no hand, no floor contact
Writes s<k>_objects [O, T, 7] (x, y, z, width, height, depth, visible), s<k>_object_labels [O] ("held:gun",
"prop:box") and info["objects"].
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from multishot_people import project, ray  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vrecon.scene.mannequin import mesh_silhouette  # noqa: E402
from vrecon.scene.props import prop_mesh  # noqa: E402

SW, SH = 320, 136  # shape check size

HAND_PX = 40
WRISTS = (20, 21)
FRONT = 0.12  # m: a thing seen over the person is at least this far in front of the nearest body joint there
NEAR = 0.15  # m: never closer to the camera than this


def cam_depth(p: np.ndarray, cam: np.ndarray) -> np.ndarray:
    w2c = np.linalg.inv(cam)
    return -(p @ w2c[:3, :3].T + w2c[:3, 3])[..., 2]


def main() -> int:
    src, diag, tag, out = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], Path(sys.argv[4])
    z = dict(np.load(src))
    info = json.loads(str(z["info"]))
    W, H = int(z["W"]), int(z["H"])
    rep = {}
    for k, s in enumerate(info["shots"]):
        T = s["frames"]
        z[f"s{k}_objects"] = np.zeros((0, T, 7))
        z[f"s{k}_object_labels"] = np.array([], dtype="<U32")
        p = diag / f"{tag}_shot{k:02d}.objects.npz"
        if s.get("slot") != 0 or not p.exists():
            continue
        oz = np.load(p)
        masks, id_map = oz["masks"], {int(a): b for a, b in json.loads(str(oz["id_map"])).items()}
        sx, sy = W / masks.shape[2], H / masks.shape[1]
        cam, body = z[f"s{k}_cam"], z[f"s{k}_body"]
        f = s["lens_35mm"] / 36.0 * W
        objs, labels, r, hands = [], [], [], []
        for oid, word in id_map.items():
            on = [t for t in range(min(len(masks), T)) if (masks[t] == oid).any()]
            if not on:
                continue
            geo = {}
            for t in on:
                ys, xs = np.nonzero(masks[t] == oid)
                geo[t] = (xs.mean() * sx, ys.mean() * sy, xs.min() * sx, ys.min() * sy, (xs.max() + 1) * sx,
                          (ys.max() + 1) * sy)
            hand, dist = None, None
            if len(body):
                best = []
                for wi in WRISTS:
                    d = [np.hypot(*(project(body[t][[wi]], cam[t], f, W, H)[0] - geo[t][:2])) for t in on
                         if t < len(body) and cam_depth(body[t][wi], cam[t]) > NEAR]  # a wrist behind the camera says nothing
                    size = np.median([np.hypot(g[4] - g[2], g[5] - g[3]) for g in geo.values()])
                    if d:
                        best.append((float(np.median(d)), wi, size))
                if best:
                    dmin, wi, size = min(best)
                    dist = dmin
                    if dmin < 0.5 * size + HAND_PX:
                        hand = wi
            arr = np.zeros((T, 7))
            placed = fronted = 0
            for t in on:
                cx, cy, x0, y0, x1, y1 = geo[t]
                if hand is not None and t < len(body):
                    wpos = body[t][hand]
                    w2c = np.linalg.inv(cam[t])
                    depth = -(w2c[:3, :3] @ wpos + w2c[:3, 3])[2]
                    o, d = ray(cx, cy, cam[t], f, W, H)
                    fwd = -cam[t][:3, 2]
                    c = o + d * depth / max(d @ fwd, 1e-6)
                elif s.get("points_only_during_cut") and len(body) and t < len(body):
                    # a close-up: the floor is rarely in view, the thing is at the person's distance
                    w2c = np.linalg.inv(cam[t])
                    dj = cam_depth(body[t], cam[t])
                    depth = float(np.median(dj[dj > NEAR])) if (dj > NEAR).any() else 0.0
                    if depth <= 0:
                        continue
                    o, d = ray(cx, cy, cam[t], f, W, H)
                    c = o + d * depth / max(d @ (-cam[t][:3, 2]), 1e-6)
                elif y1 < H - 2:
                    o, d = ray((x0 + x1) / 2, y1, cam[t], f, W, H)
                    if d[2] >= -1e-3:
                        continue
                    g = o + d * (-o[2] / d[2])
                    depth = float((g - o) @ (-cam[t][:3, 2]))
                    if depth > 80:
                        continue
                    c = g.copy()
                else:
                    continue
                if len(body) and t < len(body):
                    # what the video shows over the person is in front of it: never inside or behind the body there
                    dj = cam_depth(body[t], cam[t])
                    uv = project(body[t], cam[t], f, W, H)
                    mx, my = 0.2 * (x1 - x0), 0.2 * (y1 - y0)
                    over = (dj > NEAR) & (uv[:, 0] > x0 - mx) & (uv[:, 0] < x1 + mx) & (uv[:, 1] > y0 - my) & (uv[:, 1] < y1 + my)
                    if over.any() and depth > dj[over].min() - FRONT:
                        depth = max(float(dj[over].min()) - FRONT, NEAR)
                        o, d = ray(cx, cy, cam[t], f, W, H)
                        c = o + d * depth / max(d @ (-cam[t][:3, 2]), 1e-6)
                        fronted += 1
                wm, hm = (x1 - x0) * depth / f, (y1 - y0) * depth / f
                if hand is None and not s.get("points_only_during_cut"):
                    c[2] = hm / 2
                arr[t] = [*c, wm, hm, min(wm, hm), 1.0]
                placed += 1
            r.append({"word": word, "frames_seen": len(on), "frames_placed": placed,
                      "held_by": None if hand is None else ("left" if hand == 20 else "right") + "_wrist",
                      "hand_distance_px": dist, "frames_moved_in_front": fronted})
            if placed:
                # sizes: one size per object (median), the place follows the frames
                vis = arr[:, 6] > 0
                arr[vis, 3:6] = np.median(arr[vis, 3:6], 0)
                # shape: the kind's rough shape only where it matches the mask better than a plain box
                score = {}
                for shape in ("kind", "box"):
                    v0, fc = prop_mesh(word if shape == "kind" else "box", arr[vis][0, 3:6])
                    v0 = np.asarray(v0, float)
                    ious = []
                    for t in np.flatnonzero(vis)[:: max(1, int(vis.sum()) // 8)]:
                        sil = mesh_silhouette(v0 + arr[t, :3], fc, cam[t], s["lens_35mm"], SW, SH)
                        mm = masks[t] == oid
                        mm = mm[np.arange(SH) * mm.shape[0] // SH][:, np.arange(SW) * mm.shape[1] // SW]
                        ious.append((sil & mm).sum() / max((sil | mm).sum(), 1))
                    score[shape] = float(np.mean(ious))
                if score["box"] > score["kind"]:
                    word_shape = "box"
                else:
                    word_shape = word
                r[-1]["shape"] = {"used": word_shape, "iou_kind": score["kind"], "iou_box": score["box"]}
                objs.append(arr)
                labels.append(("held:" if hand is not None else "prop:") + word + ("" if word_shape == word else "|box"))
                hands.append((hand, dist))
        # one thing per hand: the one closest to the wrist; the others were the same thing or a misreading
        keep = [i for i, (hd, dd) in enumerate(hands) if hd is None or dd <= min(e for h2, e in hands if h2 == hd)]
        for i in sorted(set(range(len(objs))) - set(keep)):
            r.append({"dropped": labels[i], "why": "another object is closer to the same hand"})
        objs, labels = [objs[i] for i in keep], [labels[i] for i in keep]
        if objs:
            z[f"s{k}_objects"] = np.array(objs)
            z[f"s{k}_object_labels"] = np.array(labels)
        rep[k] = r
    info["objects"] = rep
    z["info"] = json.dumps(info)
    np.savez_compressed(out, **z)
    print(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
