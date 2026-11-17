"""SAM 3.1 masks for the synthetic v2 clips on the ComfyUI pod, in the format the repair loop reads
(masks [T,H,W] labels, id_map {"1": "obj_floor", "2": "obj_wall", "3": "obs_0", ...}).

    python scripts/sam_synthetic_v2.py --server URL [--clips syn2_01 ...] [--out runs/sam_v2]

The same text prompts for every clip, chosen without looking at any clip's ground truth: floor FLOOR, wall
"wall", movers MOVERS (generic primitive words); a mover larger than MAX_MOVER_AREA is a surface and joins the floor when the floor prompt found less. Movers are tracked one by one (up to MAX_OBJECTS); where masks
overlap, a mover wins over the wall and the wall over the floor. Writes <out>/<clip>/masks.npz and a summary.
Ground truth is not read.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MOVERS = "box, cylinder, ball, block"
MAX_OBJECTS = 6
FLOOR = "floor, ground"
MAX_MOVER_AREA = 0.2  # a "mover" covering more than this share of the image on average is a surface, not an object


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", required=True)
    ap.add_argument("--clips", nargs="*")
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "sam_v2")
    ap.add_argument("--threshold", type=float, default=0.4)
    a = ap.parse_args()

    sys.path.insert(0, str(ROOT / "gpu" / "h3_control"))
    from comfy_jobs import Server
    from comfy_sam import masks_for

    from vrecon.control.core import read_video_rgb

    srv = Server(a.server)
    clips = a.clips or sorted(p.name for p in (ROOT / "evaluation" / "synthetic_v2").glob("syn2_*"))
    summary = {}
    for clip in clips:
        video = ROOT / "evaluation" / "synthetic_v2" / clip / "video.mp4"
        n, h, w = read_video_rgb(video).shape[:3]
        out = a.out / clip
        out.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        name = srv.upload(video, name=f"samv2__{clip}.mp4")
        floor = masks_for(srv, name, FLOOR, a.threshold, 4, 24.0, f"vrecon/samv2_{clip}_floor", n, (w, h), out)
        wall = masks_for(srv, name, "wall", a.threshold, 4, 24.0, f"vrecon/samv2_{clip}_wall", n, (w, h), out)
        movers = masks_for(srv, name, MOVERS, a.threshold, MAX_OBJECTS, 24.0, f"vrecon/samv2_{clip}_movers", n,
                           (w, h), out, per_object=True)
        big = [m for m in movers if m.mean() > MAX_MOVER_AREA]
        movers = [m for m in movers if m.mean() <= MAX_MOVER_AREA]
        if big and (not floor or floor[0].mean() < min(b.mean() for b in big)):
            floor = [np.logical_or.reduce([*(floor or []), *big])]  # the floor prompt missed it; the mover prompt took it
        lab = np.zeros((n, h, w), np.uint8)
        id_map = {}
        if floor:
            lab[floor[0]] = 1
            id_map["1"] = "obj_floor"
        if wall:
            lab[wall[0]] = 2
            id_map["2"] = "obj_wall"
        for i, m in enumerate(movers):
            lab[m] = 3 + i
            id_map[str(3 + i)] = f"obs_{i}"
        meta = {"tool": "sam3.1_multiplex (ComfyUI)", "prompts": {"floor": FLOOR, "wall": "wall", "movers": MOVERS},
                "threshold": a.threshold, "movers_found": len(movers), "surfaces_dropped_from_movers": len(big), "seconds": round(time.time() - t0, 1)}
        np.savez_compressed(out / "masks.npz", masks=lab, id_map=json.dumps(id_map), meta=json.dumps(meta))
        summary[clip] = meta
        print(clip, json.dumps(meta), flush=True)
    (a.out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
