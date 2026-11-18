"""Collect the RQ1 cells into one row list for scripts/compare_conditions.py --rows.

    python scripts/rq1_rows.py [--root runs/rq1] [--out runs/rq1/all.json]

Cells: S0 (runs/rq1/start/scores_S0.json), S1 (runs/rq1/S1/scores.json, plus held-out S1 if present),
L0 and L1 (runs/rq1/L1/scores_L0.json / scores_L1.json, one row per Claude run), and edit_success_share where
runs/rq1/edits/edits.json has it. S0 and S1 are deterministic: their one run is entered RUNS_DET times so the
paired bootstrap has two sides (the interval of a deterministic cell has zero width; recorded as "deterministic").
L0 runs that ended without a scene (usage limit, crash) are counted in runs/rq1/failures.json, not as rows.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS_DET = 3
KEYS = ("mask_iou_all_mean", "novel_iou_10", "novel_iou_30", "camera_focal_rel_mean", "timing_rel_error_frames_mean")


def _load(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ROOT / "runs" / "rq1")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    r = a.root
    edits = _load(r / "edits" / "edits.json")
    rows = []

    def add(cell, clip, run, s, det=False):
        row = {"cell": cell, "clip": clip, "run": run, **{k: s.get(k) for k in KEYS}}
        e = edits.get(f"{cell}/{clip}/{run}")
        if e is not None:
            row["edit_success_share"] = e
        if det:
            row["deterministic"] = True
        rows.append(row)

    for cell, path in (("S0", r / "start" / "scores_S0.json"), ("S1", r / "S1" / "scores.json")):
        for clip, s in _load(path).items():
            for i in range(RUNS_DET):
                add(cell, clip, f"r{i + 1}", s, det=True)
    for cell, path in (("L0", r / "L1" / "scores_L0.json"), ("L1", r / "L1" / "scores_L1.json")):
        for key, s in _load(path).items():
            clip, run = key.split("/")
            add(cell, clip, run, s)
    fails = {}
    for d in sorted((r / "L0").glob("*/r*/")):
        if not (d / "score.json").is_file():
            fails[f"{d.parent.name}/{d.name}"] = "no scene scored (usage limit, crash or still running)"
    (r / "failures.json").write_text(json.dumps(fails, indent=1) + "\n", encoding="utf-8")
    out = a.out or r / "all.json"
    out.write_text(json.dumps(rows, indent=1) + "\n", encoding="utf-8")
    print(f"{len(rows)} rows, {len(fails)} L0 runs without a score -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
