"""Automatic check of a multi-cut scene against what the video shows (no ground truth; works for any clip).

    python scripts/multishot_check.py SCENE.npz SCENE.blend DIAG_DIR TAG OUT.json [--every 2] [--width 384]
                                      [--hide PREFIX ...]

1 picture   the .blend is drawn from every cut's camera as category labels (scripts/multishot_labels.py) and
            compared, per cut, with the observations the pipeline already made:
              person   SAM 3.1 person masks (TAG_shotNN.masks.npz): IoU, recall (mask covered), precision
              object   SAM 3.1 object masks (TAG_shotNN.objects.npz): IoU, recall, precision against held+prop
              empty    share of the picture where nothing is drawn (holes in the set)
2 sense     checks on the scene itself, no picture: joints below the floor, body height, the same person's size in
            different cuts, objects under the floor
3 score     per cut: the mean of its picture IoUs (person, object when the cut has objects), minus the empty share;
            the clip score is the frame-weighted mean. `flags` lists what is clearly wrong, with the cut and numbers.
--hide leaves objects out by name prefix (Body03, Obj02_, Layout_): rerun with it to see whether a part helps.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LABELS = ["none", "person", "ground", "structure", "sky", "held", "prop"]
FLOOR_TOL = 0.15  # m below the floor before a joint counts as under it
HEIGHT = (1.4, 2.1)  # m, a plausible adult
SIZE_SPREAD = 0.15  # the same person's height may differ this much between cuts
MIN_IOU = 0.4  # below: a flag


def resize_nearest(m: np.ndarray, h: int, w: int) -> np.ndarray:
    return m[np.arange(h) * m.shape[0] // h][:, np.arange(w) * m.shape[1] // w]


def iou_parts(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float] | None:
    """a = drawn, b = observed; None when neither has pixels."""
    i, u = (a & b).sum(), (a | b).sum()
    if u == 0:
        return None
    return i / u, (i / b.sum() if b.any() else np.nan), (i / a.sum() if a.any() else np.nan)


def body_height(j: np.ndarray) -> float:
    """SMPL 22 joints -> standing height (bone chain head-pelvis-ankle, plus head top and foot)."""
    up = sum(np.linalg.norm(j[a] - j[b]) for a, b in ((0, 3), (3, 6), (6, 9), (9, 12), (12, 15)))
    leg = np.mean([np.linalg.norm(j[h] - j[k]) + np.linalg.norm(j[k] - j[a]) for h, k, a in ((1, 4, 7), (2, 5, 8))])
    hip = np.mean([np.linalg.norm(j[0] - j[1]), np.linalg.norm(j[0] - j[2])]) * 0.3
    return float(up + leg + hip + 0.18)  # 0.18: head top above joint 15 plus ankle to sole


def sense(z, info) -> tuple[dict, list]:
    rep, flags, heights = {}, [], {}
    joined = "joined" in info
    for k, s in enumerate(info["shots"]):
        r = {}
        body = z[f"s{k}_body"]
        world = joined and s.get("slot") == 0
        if len(body):
            h = float(np.median([body_height(b) for b in body]))
            r["body_height_m"] = h
            heights[k] = h
            if not HEIGHT[0] <= h <= HEIGHT[1] and not s.get("unitless"):
                flags.append({"cut": k, "what": "body height", "value": round(h, 2)})
            if world:
                low = body[:, :, 2].min(1)
                under = float((low < -FLOOR_TOL).mean())
                r["frames_under_floor"] = under
                r["lowest_joint_m_median"] = float(np.median(low))
                if under > 0.2:
                    flags.append({"cut": k, "what": "body under the floor", "share_frames": round(under, 2)})
        if f"s{k}_objects" in z and len(z[f"s{k}_objects"]) and world:
            o = z[f"s{k}_objects"]
            vis = o[..., 6] > 0
            bottom = o[..., 2] - o[..., 4] / 2
            bad = float((bottom[vis] < -FLOOR_TOL).mean()) if vis.any() else 0.0
            r["object_frames_under_floor"] = bad
            if bad > 0.2:
                flags.append({"cut": k, "what": "objects under the floor", "share_frames": round(bad, 2)})
        rep[k] = r
    # the same person in several cuts: close-ups name whose body they hold
    who = {int(k): c["who"] for k, c in info.get("closeups", {}).items() if c.get("placed")}
    groups: dict[int, list] = {}
    for k in heights:
        groups.setdefault(who.get(k, k), []).append(heights[k])
    if len(heights) > 1:
        hs = np.array(list(heights.values()))
        spread = float(hs.max() / hs.min() - 1)
        rep["height_spread_between_cuts"] = spread
        if spread > SIZE_SPREAD:
            flags.append({"what": "people differ in size between cuts", "heights_m": {k: round(v, 2) for k, v in heights.items()}})
    return rep, flags


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", type=Path)
    ap.add_argument("blend", type=Path)
    ap.add_argument("diag", type=Path)
    ap.add_argument("tag")
    ap.add_argument("out", type=Path)
    ap.add_argument("--every", type=int, default=2)
    ap.add_argument("--width", type=int, default=384)
    ap.add_argument("--hide", nargs="*", default=[])
    a = ap.parse_args()
    z = np.load(a.npz)
    info = json.loads(str(z["info"]))
    W, H = int(z["W"]), int(z["H"])
    w = a.width
    h = int(round(H * w / W / 2)) * 2
    shots = info["shots"]
    frames = []
    for s in shots:
        st = s["source_frames"][0] + 1
        frames += list(range(st, st + s["frames"], a.every))
    lab_npz = a.out.with_suffix(".labels.npz")
    rc = subprocess.run([sys.executable, str(ROOT / "scripts/multishot_labels.py"), str(a.blend), str(lab_npz), str(w),
                         str(h), json.dumps(frames), "--hide", *a.hide], capture_output=True, text=True, encoding="utf-8",
                        errors="replace")
    if rc.returncode or not lab_npz.exists():
        print(rc.stdout[-2000:], rc.stderr[-2000:])
        return 1
    L = np.load(lab_npz)
    labels, lf = L["labels"], list(L["frames"])
    P, HELD, PROP, NONE = (LABELS.index(c) for c in ("person", "held", "prop", "none"))
    cuts, flags = {}, []
    tot_w = tot = 0.0
    for k, s in enumerate(shots):
        st = s["source_frames"][0] + 1
        mp = a.diag / f"{a.tag}_shot{k:02d}.masks.npz"
        op = a.diag / f"{a.tag}_shot{k:02d}.objects.npz"
        pm = np.load(mp)["masks"] if mp.exists() else None
        om = np.load(op)["masks"] if op.exists() else None
        acc: dict[str, list] = {"person": [], "object": []}
        empty = []
        for i, f in enumerate(lf):
            t = f - st
            if not 0 <= t < s["frames"]:
                continue
            lab = labels[i]
            empty.append((lab == NONE).mean())
            if pm is not None and t < len(pm):
                r = iou_parts(lab == P, resize_nearest(pm[t] > 0, h, w))
                if r:
                    acc["person"].append(r)
            if om is not None and t < len(om):
                r = iou_parts((lab == HELD) | (lab == PROP), resize_nearest(om[t] > 0, h, w))
                if r:
                    acc["object"].append(r)
        c = {"frames_checked": len(empty), "empty": float(np.mean(empty)) if empty else None}
        parts = []
        for name, rs in acc.items():
            if rs:
                r = np.array(rs, float)
                c[name] = {"iou": float(r[:, 0].mean()), "recall": float(np.nanmean(r[:, 1])) if np.isfinite(r[:, 1]).any() else None,
                           "precision": float(np.nanmean(r[:, 2])) if np.isfinite(r[:, 2]).any() else None}
                parts.append(c[name]["iou"])
                if c[name]["iou"] < MIN_IOU:
                    flags.append({"cut": k, "what": f"{name} does not match the video", **{q: round(v, 2) for q, v in c[name].items() if v is not None}})
        if c["empty"] and c["empty"] > 0.1:
            flags.append({"cut": k, "what": "nothing drawn", "share": round(c["empty"], 2)})
        if parts:
            c["score"] = float(np.mean(parts) - (c["empty"] or 0))
            tot += c["score"] * s["frames"]
            tot_w += s["frames"]
        cuts[k] = c
    sn, sflags = sense(z, info)
    rep = {"score": tot / tot_w if tot_w else None, "hidden": a.hide, "cuts": cuts, "sense": sn, "flags": flags + sflags}
    a.out.write_text(json.dumps(rep, indent=1), encoding="utf-8")
    print(json.dumps({"score": rep["score"], "cuts": {k: round(v.get("score", float("nan")), 3) for k, v in cuts.items()},
                      "flags": rep["flags"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
