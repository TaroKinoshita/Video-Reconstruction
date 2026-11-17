"""RQ1 start scenes (cell S0) for the v2 clips: a project per clip (`vrecon new --no-shots`), then a run built from
4RC (`init_run(from_4d=...)`) with the SAM masks of scripts/sam_synthetic_v2.py, then scored (scorer v3, novel views).

    python scripts/build_rq1_starts.py [--clips syn2_01 ...] [--work DIR] [--out runs/rq1/start]

4RC outputs: <work>/<clip>/4rc.npz (4RC's inference.py on the clip's frames; default work dir
%LOCALAPPDATA%/vrecon/work/v2). Writes <out>/<clip>/{new,run}/ and <out>/scores_S0.json.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KEYS = ("mask_iou_all_mean", "novel_iou_10", "novel_iou_30", "camera_focal_rel_mean", "timing_rel_error_frames_mean")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", nargs="*", default=[f"syn2_{i:02d}" for i in range(1, 13)])
    ap.add_argument("--work", type=Path, default=Path(os.environ.get("LOCALAPPDATA", ".")) / "vrecon" / "work" / "v2")
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "rq1" / "start")
    a = ap.parse_args()

    from vrecon.evaluate import score
    from vrecon.run import service
    from vrecon.schema import load_project

    rows = {}
    for c in a.clips:
        clip, d = ROOT / "evaluation" / "synthetic_v2" / c, a.out / c
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
        subprocess.run([sys.executable, "-m", "vrecon.cli", "new", str(clip / "video.mp4"), "--out", str(d / "new"),
                        "--no-shots", "--id", f"proj_{c}"], check=True, capture_output=True, encoding="utf-8")
        cfg = service.init_run(str(d / "new" / "project.json"), str(d / "run"),
                               str(ROOT / "runs" / "sam_v2" / c / "masks.npz"), from_4d=str(a.work / c / "4rc.npz"))
        s = score(clip, load_project(cfg["project"]), novel=True)["summary"]
        rows[c] = {**{k: s.get(k) for k in KEYS}, "pairs": cfg["pairs"]}
        print(c, json.dumps(rows[c]), flush=True)
    (a.out / "scores_S0.json").write_text(json.dumps(rows, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
