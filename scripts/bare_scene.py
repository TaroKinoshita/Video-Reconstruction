"""Bare-LLM baseline (CineRecon-style): a scene written by a language model from film keyframes only, turned into
the same scene file the pipeline makes, so the same renderer and the same checks score both.

    python scripts/bare_scene.py SPEC.json CUTS.json OUT.npz

SPEC.json (the model writes it; one world for every cut, metres, Z up, the floor at z = 0):
{"cuts": [{"cut": 0, "lens_mm": 35,
           "camera": [{"frame": 0, "position": [x, y, z], "look_at": [x, y, z]}, ...],      # frame within the cut
           "people": [{"id": "agent", "keys": [{"frame": 0, "position": [x, y], "facing_deg": 90,
                                                 "pose": "standing", "joints": null}]}],
           "objects": [{"kind": "gun", "held_by": "agent", "hand": "right", "size": [w, h, d],
                        "position": [x, y, z]}]}],
 "set": [{"type": "plane"|"box", "category": "ground"|"structure", "center": [x, y, z], "size": [sx, sy, sz],
          "yaw_deg": 0}]}
pose: standing | walking | aiming_right | aiming_both | crouching | leaning_back | lying_back | lying_front;
"joints" (22 SMPL joints, world) replaces the pose when given. facing_deg: 0 = +x, 90 = +y.
Between keys, cameras and people are interpolated linearly; held objects follow the named hand.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

# a generic standing body (pelvis over the origin, facing +y, 1.75 m), SMPL 22 joint order
STAND = np.array([
    [0, 0, .95], [.09, 0, .9], [-.09, 0, .9], [0, 0, 1.05], [.1, 0, .5], [-.1, 0, .5], [0, 0, 1.2], [.1, 0, .08],
    [-.1, 0, .08], [0, 0, 1.32], [.1, .12, .02], [-.1, .12, .02], [0, 0, 1.5], [.07, 0, 1.45], [-.07, 0, 1.45],
    [0, .02, 1.62], [.18, 0, 1.42], [-.18, 0, 1.42], [.2, 0, 1.14], [-.2, 0, 1.14], [.21, 0, .88], [-.21, 0, .88]])
UPPER = [3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]


def _rx(deg):
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _rz(deg):
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def pose_joints(name: str) -> np.ndarray:
    j = STAND.copy()
    if name in ("aiming_right", "aiming_both"):
        for sh, el, wr in ([(17, 19, 21)] + ([(16, 18, 20)] if name == "aiming_both" else [])):
            j[el] = j[sh] + [0, .28, 0]
            j[wr] = j[sh] + [0, .55, 0]
    elif name == "walking":
        j[4] += [0, .15, 0]; j[7] += [0, .25, 0]; j[10] += [0, .25, 0]
        j[5] += [0, -.1, 0]; j[8] += [0, -.2, 0]; j[11] += [0, -.2, 0]
    elif name == "crouching":
        j[[0, 1, 2]] -= [0, 0, .4]; j[UPPER] -= [0, 0, .4]
        j[4] += [0, .3, -.05]; j[5] += [0, .3, -.05]
    elif name == "leaning_back":
        p = j[0].copy()
        j[UPPER] = (j[UPPER] - p) @ _rx(-45).T + p
        j[[4, 5, 7, 8, 10, 11]] += [0, .2, 0]
    elif name in ("lying_back", "lying_front"):
        j = (j - [0, 0, 0]) @ _rx(-90 if name == "lying_back" else 90).T
        j[:, 2] += .12 - j[:, 2].min()
    return j - [0, 0, min(j[:, 2].min(), 0)]


def interp(keys: list, T: int, value) -> np.ndarray:
    fr = np.array([k["frame"] for k in keys], float)
    vals = np.array([value(k) for k in keys], float)
    order = np.argsort(fr)
    fr, vals = fr[order], vals[order]
    out = np.empty((T, *vals.shape[1:]))
    flat = vals.reshape(len(vals), -1)
    for i in range(flat.shape[1]):
        out.reshape(T, -1)[:, i] = np.interp(np.arange(T), fr, flat[:, i])
    return out


def look_at(eye, target) -> np.ndarray:
    fwd = np.asarray(target, float) - eye
    fwd /= max(np.linalg.norm(fwd), 1e-9)
    right = np.cross(fwd, [0, 0, 1.0])
    if np.linalg.norm(right) < 1e-6:
        right = np.array([1.0, 0, 0])
    right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = right, up, -fwd, eye
    return m


def main() -> int:
    spec = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    cuts = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
    out = Path(sys.argv[3])
    W, H = cuts["width"], cuts["height"]
    z, shots = {"W": W, "H": H}, []
    by = {c["cut"]: c for c in spec["cuts"]}
    for c in cuts["cuts"]:
        k, T, a = c["cut"], c["frames"], c["first_frame"]
        sp = by.get(k, {"camera": [{"frame": 0, "position": [0, -5, 1.6], "look_at": [0, 0, 1.2]}]})
        cam = interp(sp["camera"], T, lambda q: [*q["position"], *q["look_at"]])
        z[f"s{k}_cam"] = np.array([look_at(r[:3], r[3:]) for r in cam])
        bodies, held = {}, []
        for p in sp.get("people", []):
            def joints_of(q):
                j = np.array(q["joints"], float) if q.get("joints") else pose_joints(q.get("pose", "standing"))
                if not q.get("joints"):
                    j = j @ _rz(q.get("facing_deg", 90) - 90).T + [*q["position"][:2], 0]
                return j
            bodies[p["id"]] = interp(p["keys"], T, joints_of)
        names = list(bodies)
        z[f"s{k}_body"] = bodies[names[0]] if names else np.zeros((0, 22, 3))
        extra = []
        for n in names[1:]:  # more people: coarse figures as the pipeline's extra people [T, 4] (x, y, height, vis)
            b = bodies[n]
            extra.append(np.concatenate([b[:, 0, :2], (b[:, :, 2].max(1) + .1)[:, None], np.ones((T, 1))], 1))
        z[f"s{k}_extra"] = np.array(extra).reshape(-1, T, 4)
        objs, labels = [], []
        for o in sp.get("objects", []):
            size = np.array(o.get("size", [.3, .3, .3]), float)
            arr = np.zeros((T, 7))
            if o.get("held_by") in bodies:
                wr = 21 if o.get("hand", "right") == "right" else 20
                arr[:, :3] = bodies[o["held_by"]][:, wr]
            else:
                arr[:, :3] = o.get("position", [0, 0, size[1] / 2])
            arr[:, 3:6], arr[:, 6] = size, 1
            objs.append(arr)
            labels.append(("held:" if o.get("held_by") else "prop:") + o.get("kind", "box"))
        z[f"s{k}_objects"] = np.array(objs).reshape(-1, T, 7)
        z[f"s{k}_object_labels"] = np.array(labels, dtype="<U32")
        z[f"s{k}_pts"] = np.zeros((0, 3))
        z[f"s{k}_col"] = np.zeros((0, 3))
        z[f"s{k}_floor"] = np.array([0, 0, 0, 0, 0, 1.0, .4, .4, .4])  # the spec's floor is z = 0, as the pipeline's
        z[f"s{k}_garment"] = np.zeros(2)
        shots.append({"shot": k, "frames": T, "source_frames": [a, a + T], "lens_35mm": float(sp.get("lens_mm", 35)),
                      "slot": 0, "has_body": bool(names), "backdrop_rgb": [.7, .7, .7], "shared_fill_from": 0 if k else None})
    shots[0].pop("shared_fill_from")
    Q, QC, QL, B = [], [], [], []
    for s in spec.get("set", []):
        c, sz = np.array(s["center"], float), np.array(s["size"], float)
        R = _rz(s.get("yaw_deg", 0))
        if s.get("type") == "plane":
            hx, hy = sz[0] / 2, sz[1] / 2
            Q.append(np.array([c + R @ [x, y, 0] for x, y in ((-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy))]))
            QC.append(s.get("category", "ground")); QL.append([.5, .5, .5])
        else:  # a box as its six faces (rotated boxes are not in the pipeline's box format)
            h = sz / 2
            corners = {(i, j, l): c + R @ (np.array([i, j, l]) * h) for i in (-1, 1) for j in (-1, 1) for l in (-1, 1)}
            for face in (((-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1)), ((-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)),
                         ((-1, -1, -1), (1, -1, -1), (1, -1, 1), (-1, -1, 1)), ((-1, 1, -1), (1, 1, -1), (1, 1, 1), (-1, 1, 1)),
                         ((-1, -1, -1), (-1, 1, -1), (-1, 1, 1), (-1, -1, 1)), ((1, -1, -1), (1, 1, -1), (1, 1, 1), (1, -1, 1))):
                Q.append(np.array([corners[f] for f in face])); QC.append(s.get("category", "structure")); QL.append([.5, .5, .5])
    z["layout_quads"] = np.array(Q).reshape(-1, 4, 3)
    z["layout_quad_cat"] = np.array(QC)
    z["layout_quad_col"] = np.array(QL).reshape(-1, 3)
    z["layout_boxes"] = np.zeros((0, 8))
    z["layout_box_col"] = np.zeros((0, 3))
    z["info"] = json.dumps({"shots": shots, "joined": {"floor_from_cut": 0}, "bare_llm": True})
    np.savez_compressed(out, **z)
    print(json.dumps({"cuts": len(shots), "set_quads": len(Q)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
