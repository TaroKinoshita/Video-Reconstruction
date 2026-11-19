"""Re-score the 10/30-degree novel views of every L0/L1 .blend by rendering the original file with a scoring camera
(evaluate.novel_from_blend; scorer "novel_blend_v2", 2026-11-19). The v1.0 tables are not touched: results go to
runs/rq1/L1/scores_{L0,L1}_novel_v2.json, one row per run with the .blend's sha256 and the scorer version; a row is
reused only when both match. Then runs/rq1/all_v2.json (v1 rows with these two columns replaced) and
runs/rq1/compare_v2/*.md.

Pairs (ground-truth id -> .blend object) come from the source view: L0 from its score.json; L1 from a source-view
score_blend run once and stored in the row.

    python scripts/rescore_rq1_novel.py
"""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "runs" / "rq1"
VERSION = "novel_blend_v2"


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    from vrecon.evaluate import novel_from_blend, score_blend

    for cell, blend_of in (("L0", lambda c, r: R / "L0" / c / r / "workspace" / "output" / "scene.blend"),
                           ("L1", lambda c, r: R / "L1" / c / r / "refined.blend")):
        keys = json.loads((R / "L1" / f"scores_{cell}.json").read_text(encoding="utf-8"))
        outp = R / "L1" / f"scores_{cell}_novel_v2.json"
        out = json.loads(outp.read_text(encoding="utf-8")) if outp.is_file() else {}
        for key in keys:
            c, r = key.split("/")
            b, clip = blend_of(c, r), ROOT / "evaluation" / "synthetic_v2" / c
            h = sha(b)
            row = out.get(key, {})
            if row.get("scorer") == VERSION and row.get("blend_sha256") == h:
                continue
            if cell == "L0":
                pairs = json.loads((R / "L0" / c / r / "score.json").read_text(encoding="utf-8"))["pairs"]
            else:
                pairs = row.get("pairs") if row.get("blend_sha256") == h else None
                pairs = pairs or score_blend(clip, b)["pairs"]
            nov = novel_from_blend(clip, b, pairs)
            out[key] = {**nov["summary"], "pairs": pairs, "floor_rec": nov["floors"]["rec"], "scorer": VERSION,
                        "blend_sha256": h}
            outp.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
            print(cell, key, nov["summary"], flush=True)

    rows = json.loads((R / "all.json").read_text(encoding="utf-8"))
    v2 = {c: json.loads((R / "L1" / f"scores_{c}_novel_v2.json").read_text(encoding="utf-8")) for c in ("L0", "L1")}
    for row in rows:
        if row["cell"] in v2:
            n = v2[row["cell"]].get(f"{row['clip']}/{row['run']}")
            if n:
                row["novel_iou_10"], row["novel_iou_30"] = n["novel_iou_10"], n["novel_iou_30"]
                row["novel_scorer"] = VERSION
    (R / "all_v2.json").write_text(json.dumps(rows, indent=1) + "\n", encoding="utf-8")
    (R / "compare_v2").mkdir(exist_ok=True)
    for ref, cand in (("L1", "S1"), ("L0", "S1"), ("L0", "L1")):
        for split in ("dev", "held_out"):
            subprocess.run([sys.executable, str(ROOT / "scripts" / "compare_conditions.py"), "--rows", str(R / "all_v2.json"),
                            "--ref", ref, "--cand", cand, "--split", split, "--metric", "novel_iou_10", "novel_iou_30",
                            "--out", str(R / "compare_v2" / f"{ref}_vs_{cand}_{split}.md")], check=True, capture_output=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
