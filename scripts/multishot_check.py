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
MAVE-style evidence checks (Chang et al. 2026, "evidence-based verification"), adapted:
  blocked     share of the video's person / object pixels where the render shows the set (ground, structure)
              although without the set the thing is drawn there: the set stands in front of it (depth conflict)
  inside      joints inside a set piece: closer than PEN_TOL to a structure quad, within its extent
  teleport    a placed thing or extra person that jumps more than JUMP_M between two frames it is seen in, or is
              seen in fewer than 3 frames (a flash): noise turned into motion
  floating    a cut in the shared world whose body never comes near the floor (lowest joint > FLOAT_M all the cut)
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
PEN_TOL = 0.08  # m
JUMP_M = 0.6  # m per frame
FLOAT_M = 0.35  # m
MAX_BLOCKED = 0.15
SET_PARTS = ["Layout", "Floor", "Backdrop", "Ground", "Structure", "P_", "LLM_"]  # scene/multishot.py object names of the set


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
            if world and "layout_quads" in z and len(z["layout_quads"]):
                Qs = z["layout_quads"][np.array(z["layout_quad_cat"]) == "structure"]
                ins = 0
                for q in Qs:
                    o, e1, e2 = q[0], q[1] - q[0], q[3] - q[0]
                    n = np.cross(e1, e2)
                    if np.linalg.norm(n) < 1e-9:
                        continue
                    n /= np.linalg.norm(n)
                    d = (body - o) @ n
                    u = (body - o) @ e1 / (e1 @ e1)
                    v = (body - o) @ e2 / (e2 @ e2)
                    hit = (np.abs(d) < PEN_TOL) & (u > 0) & (u < 1) & (v > 0) & (v < 1)
                    ins = max(ins, float(hit.any(1).mean()))
                r["frames_inside_set"] = ins
                if ins > 0.2:
                    flags.append({"cut": k, "what": "body inside a set piece", "share_frames": round(ins, 2)})
            if world and not s.get("points_only_during_cut"):
                lowest = float(body[:, :, 2].min())
                if lowest > FLOAT_M:
                    flags.append({"cut": k, "what": "body floating above the floor", "lowest_m": round(lowest, 2)})
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
        for name, tracks, pos, vcol in (("object", z.get(f"s{k}_objects", np.zeros((0, 0, 7))), slice(0, 3), 6),
                                        ("extra person", z.get(f"s{k}_extra", np.zeros((0, 0, 4))), slice(0, 2), 3)):
            for i, tr in enumerate(tracks):
                vis = np.flatnonzero(tr[:, vcol] > 0)
                if not len(vis):
                    continue
                jump = float(np.linalg.norm(np.diff(tr[vis, pos], axis=0), axis=1).max()) if len(vis) > 1 else 0.0
                if jump > JUMP_M or len(vis) < 3:
                    flags.append({"cut": k, "what": f"{name} teleports or flashes", "index": i,
                                  "max_jump_m": round(jump, 2), "frames_seen": int(len(vis))})
        rep[k] = r
    # the same person in several cuts should have one size: group by person (scripts/multishot_who.py identity,
    # else the close-up's matched body), compare within each person only
    ident = {int(k): w.get("identity") for k, w in info.get("who", {}).items() if w.get("identity") not in (None, "?")}
    held = {int(k): c["who"] for k, c in info.get("closeups", {}).items() if c.get("placed")}
    groups: dict[str, dict] = {}
    for k in heights:
        g = ident.get(k) or f"cut{held.get(k, k)}"
        groups.setdefault(g, {})[k] = heights[k]
    rep["height_spread_by_person"] = {}
    for g, hk in groups.items():
        if len(hk) < 2:
            continue
        hs = np.array(list(hk.values()))
        spread = float(hs.max() / hs.min() - 1)
        rep["height_spread_by_person"][g] = round(spread, 3)
        if spread > SIZE_SPREAD:
            flags.append({"what": f"person {g} differs in size between cuts", "heights_m": {k: round(v, 2) for k, v in hk.items()}})
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
    def render(path: Path, hide: list) -> np.ndarray | None:
        rc = subprocess.run([sys.executable, str(ROOT / "scripts/multishot_labels.py"), str(a.blend), str(path), str(w),
                             str(h), json.dumps(frames), "--hide", *hide], capture_output=True, text=True,
                            encoding="utf-8", errors="replace")
        if rc.returncode or not path.exists():
            print(rc.stdout[-2000:], rc.stderr[-2000:])
            return None
        return np.load(path)

    L = render(a.out.with_suffix(".labels.npz"), a.hide)
    # the same views without the set: what the set hides (a depth conflict needs the thing to be drawn behind it)
    L0 = render(a.out.with_suffix(".labels_noset.npz"), [*a.hide, *SET_PARTS])
    if L is None or L0 is None:
        return 1
    labels, lf, bare = L["labels"], list(L["frames"]), L0["labels"]
    P, HELD, PROP, NONE, G, S = (LABELS.index(c) for c in ("person", "held", "prop", "none", "ground", "structure"))
    cuts, flags = {}, []
    tot_w = tot = 0.0
    for k, s in enumerate(shots):
        st = s["source_frames"][0] + 1
        mp = a.diag / f"{a.tag}_shot{k:02d}.masks.npz"
        op = a.diag / f"{a.tag}_shot{k:02d}.objects.npz"
        pm = np.load(mp)["masks"] if mp.exists() else None
        om = np.load(op)["masks"] if op.exists() else None
        acc: dict[str, list] = {"person": [], "object": []}
        blocked: dict[str, list] = {"person": [], "object": []}
        empty = []
        for i, f in enumerate(lf):
            t = f - st
            if not 0 <= t < s["frames"]:
                continue
            lab = labels[i]
            empty.append((lab == NONE).mean())
            set_px = ((lab == G) | (lab == S)) & (bare[i] != NONE) & (bare[i] != LABELS.index("sky"))
            if pm is not None and t < len(pm):
                obs = resize_nearest(pm[t] > 0, h, w)
                if obs.any():
                    blocked["person"].append((set_px & obs).sum() / obs.sum())
                r = iou_parts(lab == P, obs)
                if r:
                    acc["person"].append(r)
            if om is not None and t < len(om):
                obs = resize_nearest(om[t] > 0, h, w)
                if obs.any():
                    blocked["object"].append((set_px & obs).sum() / obs.sum())
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
        for name, b in blocked.items():
            if b:
                c[f"{name}_blocked"] = float(np.mean(b))
                if c[f"{name}_blocked"] > MAX_BLOCKED:
                    flags.append({"cut": k, "what": f"set in front of the {name} (depth conflict)", "share": round(c[f"{name}_blocked"], 2)})
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
