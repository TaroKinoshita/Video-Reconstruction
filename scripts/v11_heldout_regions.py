"""Compare static structure between a source cut and a render of the same cut by region overlap.

Inputs are SAM 3 label maps (gpu/observe/run_sam3.py with --set floor=floor --set building=building ...).
Per label, IoU between source and render, over pixels that are neither person nor smoke in either image.
A control is built from the source itself: crop by --shift-frac from the top-left and scale back
(a known small camera error), so the numbers can be read against "how much does a small miss cost".

  python scripts/v11_heldout_regions.py SRC.regions.npz REN.regions.npz OUT.json [--labels floor building sky]
"""
from __future__ import annotations
import argparse, json
import cv2
import numpy as np

def load(p):
    d = np.load(p); return d["masks"], {v: int(k) for k, v in json.loads(str(d["id_map"])).items()}

def label_mask(m, ids, name):
    return m == ids[name] if name in ids else np.zeros(m.shape, bool)

def ignore_mask(m, ids):
    out = np.zeros(m.shape, bool)
    for k, v in ids.items():
        if k.startswith("obs_") or k == "smoke": out |= m == v
    return out

def shifted(m, frac):
    T, H, W = m.shape; h, w = round(H * (1 - frac)), round(W * (1 - frac))
    return np.stack([cv2.resize(f[:h, :w], (W, H), interpolation=cv2.INTER_NEAREST) for f in m])

def compare(ms, ids_s, mr, ids_r, labels):
    n = min(len(ms), len(mr)); res = {}
    for lab in labels:
        ious = []
        for i in range(n):
            keep = ~(ignore_mask(ms[i], ids_s) | ignore_mask(mr[i], ids_r))
            a, b = label_mask(ms[i], ids_s, lab) & keep, label_mask(mr[i], ids_r, lab) & keep
            u = (a | b).sum()
            if u: ious.append(float((a & b).sum() / u))
        res[lab] = {"iou": float(np.mean(ious)) if ious else None, "frames": len(ious),
                    "src_share": float(np.mean([label_mask(f, ids_s, lab).mean() for f in ms[:n]])),
                    "ren_share": float(np.mean([label_mask(f, ids_r, lab).mean() for f in mr[:n]]))}
    return res

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("ren"); ap.add_argument("out")
    ap.add_argument("--labels", nargs="+", default=["floor", "building", "sky"])
    ap.add_argument("--shift-frac", type=float, nargs="+", default=[0.04, 0.10])
    a = ap.parse_args()
    ms, ids_s = load(a.src); mr, ids_r = load(a.ren)
    out = {"render": compare(ms, ids_s, mr, ids_r, a.labels),
           "controls": {f"shift_{f}": compare(ms, ids_s, shifted(ms, f), ids_s, a.labels) for f in a.shift_frac}}
    json.dump(out, open(a.out, "w", encoding="utf-8"), indent=1)
    for k, r in [("render", out["render"]), *out["controls"].items()]:
        print(k.ljust(12), {l: (None if v["iou"] is None else round(v["iou"], 3)) for l, v in r.items()})

if __name__ == "__main__":
    main()
