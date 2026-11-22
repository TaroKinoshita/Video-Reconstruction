"""One gate every per-frame placed thing passes before the scene is built: observations are noisy, the world is not.

    python scripts/multishot_gate.py IN.npz OUT.npz

Every earlier step turns its observation of each frame straight into a place (a mask's centre on a ray, a box at a
depth). A mask that shifts a few pixels puts a far railing 20 m away in the next frame. This step holds every such
track to what things in the world do, the same rules for every video and every kind of thing:

  lifetime   seen in fewer than MIN_FRAMES frames, or in less than MIN_SHARE of its cut: dropped (a flash is a
             misreading, not a thing)
  loose      a thing not in a hand does not move: one place, the median of its frames; when its frames disagree by
             more than LOOSE_SPREAD, or LOOSE_REL of its distance from the camera when farther (median distance to
             that place), the observation is not one thing: dropped
  held       follows the hand, smoothly: a running median over SMOOTH frames; frames farther than HELD_JUMP from it
             are not shown
  people     extra people walk: a running median of the place, steps above WALK_MAX per frame are cut (the frame is
             not shown); a person with most frames cut is dropped
  size       one size per thing (already); clamped to [1/SIZE_FACTOR, SIZE_FACTOR] x the kind's usual size when known
Writes the cleaned tracks back and info["gate"] with what was changed, per cut.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

MIN_FRAMES, MIN_SHARE = 3, 0.15
LOOSE_SPREAD = 0.5  # m, near the camera
LOOSE_REL = 0.08  # farther away: this share of the distance (a few pixels of mask shift are metres there)
SMOOTH = 5
HELD_JUMP = 0.3  # m
WALK_MAX = 0.3  # m per frame (about 7 m/s at 24 fps)
PEOPLE_KEEP = 0.5
SIZE_FACTOR = 3.0
USUAL = {"gun": 0.25, "knife": 0.3, "phone": 0.15, "bottle": 0.3, "cup": 0.12, "book": 0.25, "bag": 0.5,
         "suitcase": 0.7, "chair": 0.9, "table": 1.2, "bench": 1.5, "door": 2.1, "sign": 0.8, "railing": 1.0,
         "car": 4.5, "bicycle": 1.7, "ball": 0.22, "racket": 0.68, "bat": 0.85, "box": 0.5, "crate": 0.6,
         "barrel": 0.9, "lamp": 1.5, "plant": 0.8, "television": 1.0, "laptop": 0.35, "microphone": 0.2}  # m, largest side


def running_median(x: np.ndarray, w: int) -> np.ndarray:
    h = w // 2
    return np.array([np.median(x[max(0, i - h):i + h + 1], axis=0) for i in range(len(x))])


def main() -> int:
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    z = dict(np.load(src))
    info = json.loads(str(z["info"]))
    rep = {}
    for k, s in enumerate(info["shots"]):
        T = s["frames"]
        r = {"objects": [], "people": []}
        if f"s{k}_objects" in z and len(z[f"s{k}_objects"]):
            objs, labels = z[f"s{k}_objects"].copy(), list(z[f"s{k}_object_labels"])
            keep = []
            for i, (tr, lab) in enumerate(zip(objs, labels)):
                vis = tr[:, 6] > 0
                kind, word = str(lab).split(":")[0], str(lab).split(":")[1].split("|")[0]
                n = int(vis.sum())
                if n < MIN_FRAMES or n < MIN_SHARE * T:
                    r["objects"].append({"label": str(lab), "dropped": f"seen in {n} of {T} frames"})
                    continue
                P = tr[vis, :3]
                if kind == "prop":
                    c = np.median(P, 0)
                    spread = float(np.median(np.linalg.norm(P - c, axis=1)))
                    dist = float(np.median(np.linalg.norm(z[f"s{k}_cam"][vis, :3, 3] - c, axis=1)))
                    if spread > max(LOOSE_SPREAD, LOOSE_REL * dist):
                        r["objects"].append({"label": str(lab), "dropped": f"its frames disagree by {spread:.1f} m"})
                        continue
                    tr[vis, :3] = c
                    r["objects"].append({"label": str(lab), "held_still_at": np.round(c, 2).tolist(), "spread_m": round(spread, 2)})
                else:
                    sm = running_median(P, SMOOTH)
                    far = np.linalg.norm(P - sm, axis=1) > HELD_JUMP
                    idx = np.flatnonzero(vis)
                    tr[idx, :3] = sm
                    tr[idx[far], 6] = 0
                    r["objects"].append({"label": str(lab), "smoothed": True, "frames_hidden": int(far.sum())})
                u = USUAL.get(word)
                if u:
                    big = tr[vis, 3:6].max()
                    f = np.clip(big, u / SIZE_FACTOR, u * SIZE_FACTOR) / max(big, 1e-6)
                    if abs(f - 1) > 1e-3:
                        tr[:, 3:6] *= f
                        r["objects"][-1]["size_scaled"] = round(float(f), 2)
                keep.append(i)
            z[f"s{k}_objects"] = objs[keep] if keep else np.zeros((0, T, 7))
            z[f"s{k}_object_labels"] = np.array([labels[i] for i in keep], dtype="<U32")
        if f"s{k}_extra" in z and len(z[f"s{k}_extra"]):
            ex = z[f"s{k}_extra"].copy()
            keep = []
            for i, tr in enumerate(ex):
                vis = tr[:, 3] > 0
                n = int(vis.sum())
                if n < MIN_FRAMES or n < MIN_SHARE * T:
                    r["people"].append({"person": i, "dropped": f"seen in {n} of {T} frames"})
                    continue
                idx = np.flatnonzero(vis)
                xy = running_median(tr[idx, :2], SMOOTH)
                ok = np.ones(len(idx), bool)
                last = xy[0]
                for j in range(1, len(idx)):
                    step = np.linalg.norm(xy[j] - last) / max(idx[j] - idx[j - 1], 1)
                    if step > WALK_MAX:
                        ok[j] = False
                    else:
                        last = xy[j]
                if ok.mean() < PEOPLE_KEEP:
                    r["people"].append({"person": i, "dropped": f"{(~ok).sum()} of {len(ok)} frames teleport"})
                    continue
                tr[idx, :2] = xy
                tr[idx[~ok], 3] = 0
                tr[idx, 2] = np.median(tr[idx, 2])  # one height
                r["people"].append({"person": i, "frames_hidden": int((~ok).sum())})
                keep.append(i)
            z[f"s{k}_extra"] = ex[keep] if keep else np.zeros((0, T, 4))
        if r["objects"] or r["people"]:
            rep[k] = r
    info["gate"] = rep
    z["info"] = json.dumps(info)
    np.savez_compressed(out, **z)
    print(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
