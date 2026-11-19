"""H5: the three scripted edits on every RQ1 cell's scene (vrecon/scene/edit_blend.py).

    python scripts/run_rq1_edits.py [--root runs/rq1]

S1: the adopted project is built into a .blend first (`vrecon build`), as a user of the tool would get it.
L0: Claude's scene.blend. L1: the repaired refined.blend. Writes <root>/edits/edits.json
{"<cell>/<clip>/<run>": edit_success_share} and details.json. Already tested scenes are skipped.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _blends(r: Path):
    for clip_dir in sorted((r / "S1").glob("syn2_*")):
        snaps = sorted((clip_dir / "snapshots").glob("rev_*.json"))
        proj = snaps[-1] if snaps else Path(__import__("vrecon.run.service", fromlist=["load_config"]).load_config(clip_dir)["project"])
        yield "S1", clip_dir.name, "r1", proj
    for b in sorted((r / "L0").glob("syn2_*/r*/workspace/output/scene.blend")):
        run = b.parents[2]
        if (run / "score.json").is_file():
            yield "L0", run.parent.name, run.name, b
    for b in sorted((r / "L1").glob("syn2_*/r*/refined.blend")):
        yield "L1", b.parent.parent.name, b.parent.name, b


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ROOT / "runs" / "rq1")
    a = ap.parse_args()
    from vrecon.scene.edit_blend import run_all

    out = a.root / "edits"
    out.mkdir(parents=True, exist_ok=True)
    share = json.loads((out / "edits.json").read_text(encoding="utf-8")) if (out / "edits.json").is_file() else {}
    details = json.loads((out / "details.json").read_text(encoding="utf-8")) if (out / "details.json").is_file() else {}
    for cell, clip, run, src in _blends(a.root):
        key = f"{cell}/{clip}/{run}"
        if key in share:
            continue
        blend = src
        if cell == "S1":
            blend = out / "s1_blends" / f"{clip}.blend"
            blend.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run([sys.executable, "-m", "vrecon.cli", "build", str(src), "--blend", str(blend)], check=True,
                           capture_output=True, encoding="utf-8")
        rep = run_all(blend)
        share[key] = rep["edit_success_share"]
        details[key] = rep["edits"]
        for k, d in (("edits.json", share), ("details.json", details)):
            (out / k).write_text(json.dumps(d, indent=1) + "\n", encoding="utf-8")
        print(key, rep["edit_success_share"], [(e["edit"], e["problems"][:1]) for e in rep["edits"] if not e["success"]],
              flush=True)
    # S1 is deterministic: its one test stands for each of its replicated runs
    for key in [k for k in share if k.startswith("S1/") and k.endswith("/r1")]:
        for i in (2, 3):
            share[key[:-1] + str(i)] = share[key]
    (out / "edits.json").write_text(json.dumps(share, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
