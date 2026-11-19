"""Re-score the novel views of every L0/L1 scene with the .blend's real meshes (evaluate.score_blend now keeps them;
before 2026-11-19 the novel views drew bounding boxes). Updates runs/rq1/L1/scores_L0.json and scores_L1.json in
place (old values kept as novel_iou_*_box), then rebuilds runs/rq1/all.json and the comparison tables.

    python scripts/rescore_rq1_novel.py
"""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "runs" / "rq1"


def main() -> int:
    from vrecon.evaluate import score_blend

    for name, blend_of in (("scores_L0.json", lambda c, r: R / "L0" / c / r / "workspace" / "output" / "scene.blend"),
                           ("scores_L1.json", lambda c, r: R / "L1" / c / r / "refined.blend")):
        p = R / "L1" / name
        d = json.loads(p.read_text(encoding="utf-8"))
        for key, row in d.items():
            if "novel_iou_10_box" in row:
                continue
            c, r = key.split("/")
            s = score_blend(ROOT / "evaluation" / "synthetic_v2" / c, blend_of(c, r), novel=True)["summary"]
            row["novel_iou_10_box"], row["novel_iou_30_box"] = row.get("novel_iou_10"), row.get("novel_iou_30")
            row["novel_iou_10"], row["novel_iou_30"] = s.get("novel_iou_10"), s.get("novel_iou_30")
            p.write_text(json.dumps(d, indent=1) + "\n", encoding="utf-8")
            print(name, key, row["novel_iou_10_box"], "->", row["novel_iou_10"], flush=True)
    subprocess.run([sys.executable, str(ROOT / "scripts" / "rq1_rows.py")], check=True)
    for ref, cand in (("L0", "S1"), ("L1", "S1"), ("L0", "L1"), ("S0", "S1")):
        for split in ("dev", "held_out"):
            subprocess.run([sys.executable, str(ROOT / "scripts" / "compare_conditions.py"), "--rows", str(R / "all.json"),
                            "--ref", ref, "--cand", cand, "--split", split,
                            "--out", str(R / "compare" / f"{ref}_vs_{cand}_{split}.md")], check=True,
                           capture_output=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
