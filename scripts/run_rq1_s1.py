"""RQ1 cell S1: the fixed hypothesis order (`policy.auto_repair`, no LLM) from each clip's S0 start scene
(scripts/build_rq1_starts.py), then scored. Deterministic, so one run per clip stands for all its runs.

    python scripts/run_rq1_s1.py [--clips syn2_02 ...] [--start runs/rq1/start] [--out runs/rq1/S1]
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEV = ["syn2_02", "syn2_04", "syn2_05", "syn2_07", "syn2_09", "syn2_12"]
KEYS = ("mask_iou_all_mean", "novel_iou_10", "novel_iou_30", "camera_focal_rel_mean", "timing_rel_error_frames_mean")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", nargs="*", default=DEV)
    ap.add_argument("--start", type=Path, default=ROOT / "runs" / "rq1" / "start")
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "rq1" / "S1")
    a = ap.parse_args()

    from vrecon.evaluate import score
    from vrecon.run import policy, service
    from vrecon.schema import load_project

    rows = {}
    for c in a.clips:
        src = a.start / c / "run"
        cfg = service.load_config(src)
        run = a.out / c
        if run.exists():
            shutil.rmtree(run)
        t0 = time.time()
        service.init_run(cfg["project"], str(run), cfg["masks"], max_attempts=cfg.get("max_attempts", 30),
                         focal=cfg.get("focal"))
        mgr, obs, pc = service._manager(run)
        policy.auto_repair(mgr, obs, pc["pairs"], mgr.tracks, 2, 60, focal=cfg.get("focal", False))
        snaps = sorted((run / "snapshots").glob("rev_*.json"))
        adopted = snaps[-1] if snaps else Path(cfg["project"])
        s = score(ROOT / "evaluation" / "synthetic_v2" / c, load_project(adopted), novel=True)["summary"]
        rows[c] = {**{k: s.get(k) for k in KEYS}, "seconds": round(time.time() - t0, 1)}
        print(c, json.dumps(rows[c]), flush=True)
    old = json.loads((a.out / "scores.json").read_text(encoding="utf-8")) if (a.out / "scores.json").is_file() else {}
    (a.out / "scores.json").write_text(json.dumps({**old, **rows}, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
