"""Which L0 clips need runs 4 and 5 (evaluation/thresholds.yaml attempts_per_condition: 3 by default, 5 where the
development SD exceeds 0.03). Prints the clips for the dev rule ("dev": per clip SD of mask_iou_all_mean over
runs > 0.03) or the number of held-out runs ("held_out": 5 if the mean dev SD of L0 > 0.03, else 3).

    python scripts/rq1_extra_runs.py dev|held_out
"""

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEV = ["syn2_02", "syn2_04", "syn2_05", "syn2_07", "syn2_09", "syn2_12"]
LIMIT = json.loads((ROOT / "evaluation" / "thresholds.yaml").read_text(encoding="utf-8"))["attempts_per_condition"]
s0 = json.loads((ROOT / "runs" / "rq1" / "L1" / "scores_L0.json").read_text(encoding="utf-8"))
sd = {}
for c in DEV:
    v = [s["mask_iou_all_mean"] for k, s in s0.items() if k.startswith(c + "/") and s["mask_iou_all_mean"] is not None]
    sd[c] = statistics.stdev(v) if len(v) > 1 else 0.0
if sys.argv[1] == "dev":
    print(" ".join(c for c, x in sd.items() if x > 0.03))
else:
    print(LIMIT["when_dev_sd_above_0.03"] if statistics.mean(sd.values()) > 0.03 else LIMIT["default"])
