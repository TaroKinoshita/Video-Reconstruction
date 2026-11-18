"""RQ1 cells L0 (rescored with novel views) and L1 (the fixed hypothesis order on top of each L0 scene).

    python scripts/run_rq1_l1.py [--clips syn2_02 ...] [--l0 runs/rq1/L0] [--out runs/rq1/L1]

For every finished L0 run (<l0>/<clip>/rN/workspace/output/scene.blend, run.json not marked as a usage-limit
failure): score_blend with novel views -> <out>/scores_L0.json; refine_blend (run/refine.py: import, mark the
set, auto repair on the clip's SAM masks, write back) -> <out>/<clip>/rN, refined.blend scored -> scores_L1.json.
Already scored runs are skipped. Ground truth is read only by the scorer.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEV = ["syn2_02", "syn2_04", "syn2_05", "syn2_07", "syn2_09", "syn2_12"]
KEYS = ("mask_iou_all_mean", "novel_iou_10", "novel_iou_30", "camera_focal_rel_mean", "timing_rel_error_frames_mean")


def _load(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}


def _save(p: Path, d: dict) -> None:
    p.write_text(json.dumps(d, indent=1) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", nargs="*", default=DEV)
    ap.add_argument("--l0", type=Path, default=ROOT / "runs" / "rq1" / "L0")
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "rq1" / "L1")
    a = ap.parse_args()

    import numpy as np

    from vrecon.evaluate import score_blend
    from vrecon.run.refine import refine_blend

    a.out.mkdir(parents=True, exist_ok=True)
    s0, s1 = _load(a.out / "scores_L0.json"), _load(a.out / "scores_L1.json")
    for c in a.clips:
        clip = ROOT / "evaluation" / "synthetic_v2" / c
        t, h, w = np.load(clip / "gt_passes" / "index.npz")["index"].shape
        for run in sorted((a.l0 / c).glob("r*/")):
            key = f"{c}/{run.name}"
            blend = run / "workspace" / "output" / "scene.blend"
            if not (run / "score.json").is_file() or not blend.is_file():
                continue
            if key not in s0:
                s = score_blend(clip, blend, novel=True)["summary"]
                s0[key] = {k: s.get(k) for k in KEYS}
                _save(a.out / "scores_L0.json", s0)
            if key not in s1:
                t0 = time.time()
                rd = a.out / c / run.name
                if rd.exists():
                    import shutil

                    shutil.rmtree(rd)
                rec = refine_blend(blend, ROOT / "runs" / "sam_v2" / c / "masks.npz", rd, (w, h))
                s = score_blend(clip, rd / "refined.blend", novel=True)["summary"]
                s1[key] = {**{k: s.get(k) for k in KEYS}, "accepted": rec["accepted"],
                           "seconds": round(time.time() - t0, 1)}
                _save(a.out / "scores_L1.json", s1)
            print(key, "L0", round(s0[key]["mask_iou_all_mean"] or 0, 3), "L1", round(s1[key]["mask_iou_all_mean"] or 0, 3),
                  flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
