"""Option 4 without an LLM: refine an LLM-made .blend (condition B) with the repair loop, then score it.

    python scripts/refine_blend.py SCENE.blend CLIP_DIR RUN_DIR [--passes 2] [--attempts 30] [--focal] [--tracks]

The refinement itself is vrecon.run.refine.refine_blend (no ground truth). This script then scores, on the
clip's ground truth: the start (B scene as proxy boxes), the refined proxies, and the refined .blend.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("blend")
    ap.add_argument("clip")
    ap.add_argument("run")
    ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--attempts", type=int, default=30)
    ap.add_argument("--focal", action="store_true")
    ap.add_argument("--tracks", action="store_true")
    a = ap.parse_args()

    from vrecon.evaluate import score, score_blend
    from vrecon.run.refine import refine_blend
    from vrecon.schema import load_project

    clip, run = Path(a.clip).resolve(), Path(a.run).resolve()
    _, h, w = np.load(clip / "gt_passes" / "index.npz")["index"].shape  # image size only
    rec = refine_blend(a.blend, clip, run, (w, h), a.passes, a.attempts, focal=a.focal, tracks=a.tracks)
    rec["score_start"] = score(clip, load_project(run / "start_from_blend.orig.json"))["summary"]
    rec["score_refined"] = score(clip, load_project(run / "start_from_blend.json"))["summary"]
    rec["score_refined_blend"] = score_blend(clip, rec["refined_blend"])["summary"]
    (run / "refine_result.json").write_text(json.dumps(rec, indent=1, default=str) + "\n", encoding="utf-8")
    keys = ("mask_iou_mean", "camera_focal_rel_mean", "timing_abs_error_frames_mean", "timing_missing")
    print(json.dumps({"renamed": rec["renamed"], "accepted": rec["accepted"], "tried": rec["tried"],
                      **{s: {k: rec[s].get(k) for k in keys} for s in ("score_start", "score_refined", "score_refined_blend")}}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
