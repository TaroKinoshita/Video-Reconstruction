"""Score generated videos against the scene with masks from SAM 3.1 on the ComfyUI pod (RQ2; scene-independent).

    python scripts/score_generated_sam.py --server URL --controls CONTROLS_DIR --scene PROJECT.json \
        --object obj_person_01="tennis player" [--object ID=PROMPT ...] RUN_DIR [RUN_DIR ...]

For each RUN_DIR/video.mp4: the video is brought to the controls' size, each --object is segmented with its text
prompt (gpu/h3_control/comfy_sam.py), and vrecon.compare.generated scores it against the controls' index pass
(per object: IoU, centroid error, onset error) plus edge agreement by status region. Writes RUN_DIR/score_sam.json
and prints one line per run.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", required=True)
    ap.add_argument("--controls", type=Path, required=True)
    ap.add_argument("--scene", type=Path, required=True)
    ap.add_argument("--object", action="append", required=True, help='ID="text prompt"')
    ap.add_argument("runs", nargs="+", type=Path)
    a = ap.parse_args()

    import cv2

    from vrecon.compare.generated import load_controls, score_generated
    from vrecon.control.core import encode_mp4, read_video_rgb
    from vrecon.schema import load_project

    ctrl = load_controls(a.controls)
    t_n, h, w = ctrl["index"].shape
    scene = load_project(a.scene)
    objs = [o.split("=", 1) for o in a.object]
    for run in a.runs:
        v = read_video_rgb(run / "video.mp4")
        if v.shape[1:3] != (h, w):
            v = np.stack([cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA) for f in v])
        v = v[:t_n]
        small = run / "video_ctrlsize.mp4"
        encode_mp4(v, small, fps=24, crf=14)
        masks = {}
        for oid, prompt in objs:
            out = run / f"sam_{oid}.npz"
            if not out.is_file():
                subprocess.run([sys.executable, str(ROOT / "gpu/h3_control/comfy_sam.py"), "--server", a.server,
                                str(small), str(out), "--prompt", prompt.strip('"'), "--label", oid,
                                "--max-objects", "1"], check=True)
            masks[oid] = np.load(out)["masks"][:t_n] > 0
        rep = score_generated(v, ctrl, masks=masks, scene=scene, movers=[o for o, _ in objs])
        rep["run"] = run.name
        (run / "score_sam.json").write_text(json.dumps(rep, indent=1, default=float) + "\n", encoding="utf-8")
        line = {o: {"iou": round(rep["objects"][o]["iou"], 3),
                    "centroid": round(rep["objects"][o].get("centroid_error") or 0, 4),
                    "onset_err": (rep["objects"][o].get("onset") or {}).get("error_frames")}
                for o, _ in objs if o in rep.get("objects", {})}
        print(run.name, json.dumps(line), "edgeF1", round(rep["edges"]["all"]["f1"], 3) if rep.get("edges") else None,
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
