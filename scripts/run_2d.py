"""2D layers from a synthetic 2D clip (masks = ground-truth stand-in), then score against the ground truth.

    python scripts/run_2d.py evaluation/synthetic2d/s2d_01 runs/2d/s2d_01 [--refine]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("clip")
    ap.add_argument("run")
    ap.add_argument("--refine", action="store_true")
    a = ap.parse_args()

    from vrecon.control.core import read_video_rgb
    from vrecon.layers2d.estimate import estimate_layers
    from vrecon.metrics.layers import score_layers
    from vrecon.schema import load_project

    clip, run = Path(a.clip), Path(a.run)
    run.mkdir(parents=True, exist_ok=True)
    frames = read_video_rgb(clip / "video.mp4")
    index = np.load(clip / "gt_passes" / "index.npz")["index"]  # ground-truth stand-in for a segmenter
    id_map = {int(k): v for k, v in json.loads((clip / "gt_passes" / "id_map.json").read_text(encoding="utf-8")).items()}
    fps = 24.0
    layers, sprites, series, infos = estimate_layers(frames, index, id_map, fps)
    rec = {"layers": [lay.model_dump() for lay in layers], "info": infos}
    if a.refine:
        from vrecon.layers2d.refine import refine_layers

        layers, rinfo = refine_layers(layers, sprites, index, id_map, fps)
        rec["refine"] = rinfo
        rec["layers_refined"] = [lay.model_dump() for lay in layers]
    gt = load_project(clip / "gt.json")
    meta = json.loads((clip / "meta.json").read_text(encoding="utf-8"))
    rec["score"] = score_layers(gt, layers, sprites, index, id_map, meta, fps)
    (run / "result.json").write_text(json.dumps(rec, indent=1, default=float) + "\n", encoding="utf-8")
    print(json.dumps(rec["score"]["summary"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
