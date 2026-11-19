"""Re-run the fixed hypothesis order (condition S1) on the v1 clips from the stored 4RC start scenes, with the
current code, and score the result (scorer v3 with novel views). No LLM, no GPU observer.

    python scripts/rerun_fixed_order.py OUT_DIR [--clips syn_01 ...] [--no-camera-path]

Each clip: a new run folder initialised exactly like runs/bench_rays/<clip>_base (same project, masks, 4RC file,
focal setting, read from its config.json), then `vrecon run auto`, then the adopted scene scored. Writes
OUT_DIR/<clip>/ (run) and OUT_DIR/scores.json.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--clips", nargs="*", default=[f"syn_0{i}" for i in range(1, 7)])
    ap.add_argument("--no-camera-path", action="store_true", help="(the default in auto_repair is now off; this script turns it on unless given)")
    a = ap.parse_args()

    from vrecon.evaluate import score
    from vrecon.run import policy, service
    from vrecon.run.manager import RunManager  # noqa: F401
    from vrecon.schema import load_project

    rows = {}
    for clip in a.clips:
        base = ROOT / "runs" / "bench_rays" / f"{clip}_base"
        cfg = service.load_config(base)
        run = a.out / clip
        if run.exists():
            import shutil

            shutil.rmtree(run)
        t0 = time.time()
        service.init_run(str(ROOT / "evaluation" / "synthetic" / clip / "gt.json") if False else str(base / "init.json"),
                         str(run), cfg["masks"], max_attempts=cfg.get("max_attempts", 30),
                         video=cfg.get("video"), focal=cfg.get("focal"), tracks=cfg.get("tracks"))
        mgr, obs, pc = service._manager(run)
        recs = policy.auto_repair(mgr, obs, pc["pairs"], mgr.tracks, 2, 60, focal=cfg.get("focal", True),
                                  camera_path=not a.no_camera_path)
        snaps = sorted((run / "snapshots").glob("rev_*.json"))
        adopted = snaps[-1] if snaps else Path(service.load_config(run)["project"])
        rep = score(ROOT / "evaluation" / "synthetic" / clip, load_project(adopted), novel=True)
        s = rep["summary"]
        cam_recs = [r for r in recs if r["hypothesis"].startswith("camera translation")]
        rows[clip] = {k: s.get(k) for k in ("mask_iou_all_mean", "novel_iou_10", "novel_iou_30", "camera_focal_rel_mean",
                                           "camera_center_rel_mean", "timing_rel_error_frames_mean")}
        rows[clip]["camera_path"] = [r["status"] for r in cam_recs]
        rows[clip]["seconds"] = round(time.time() - t0, 1)
        print(clip, json.dumps(rows[clip]), flush=True)
    old = json.loads((a.out / "scores.json").read_text(encoding="utf-8")) if (a.out / "scores.json").is_file() else {}
    (a.out / "scores.json").write_text(json.dumps({**old, **rows}, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
