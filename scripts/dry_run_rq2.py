"""WP-4 dry runs: prove the RQ2 scorer on renders where the answer is known (no generation, no GPU observer).

    python scripts/dry_run_rq2.py [--out runs/rq2_dry] [--frames 124]

For each phase-1a scene (evaluation/phase1a/scenes: base + three edits) the controls are exported with the status
pass (depth, canny, gray; the shaded pass stands in for a generated video). Observer masks are the scene's own
index pass (a perfect segmenter).
    A  each scene's shaded render scored against its own controls: everything near perfect
    B  each edited scene's render scored against the BASE controls (as if the generator had ignored nothing):
       the camera edit must show in camera and in every object's IoU, the size edit only in the hero's IoU,
       the timing edit only in the hero's onset (24 frames); unchanged objects keep their pixels where the camera
       is unchanged
Writes <out>/report.json and prints a summary table.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
P1A = ROOT / "evaluation" / "phase1a"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "rq2_dry")
    a = ap.parse_args()

    from vrecon.compare.generated import load_controls, masks_from_index, score_generated
    from vrecon.control.core import load_frames
    from vrecon.scene.controls import export_controls
    from vrecon.schema import load_project

    edits = json.loads((P1A / "edits.json").read_text(encoding="utf-8"))
    names = [edits["base"]] + [e["id"] for e in edits["edits"]]
    scenes, ctrls, videos = {}, {}, {}
    for n in names:
        scenes[n] = load_project(P1A / "scenes" / f"{n}.json")
        d = a.out / n
        if not (d / "manifest.json").is_file():
            export_controls(scenes[n], d, kinds=("depth", "canny", "gray"), preview=False)
        ctrls[n] = load_controls(d)
        videos[n] = load_frames(sorted((d / "passes" / "shaded").glob("frame_*.png")))

    rep = {"A": {}, "B": {}}
    for n in names:  # A: own controls
        c = ctrls[n]
        r = score_generated(videos[n], c, masks_from_index(c["index"], c["id_map"]), scene=scenes[n], camera=scenes[n])
        rep["A"][n] = r
    base = edits["base"]
    cb = ctrls[base]
    for e in edits["edits"]:  # B: edited render against the base controls
        n = e["id"]
        own = ctrls[n]
        r = score_generated(videos[n], cb, masks_from_index(own["index"], own["id_map"]),
                            base_video=videos[base], unchanged=[u for u in e["unchanged"] if u.startswith("obj_")]
                            if "cam_001" in e["unchanged"] else None,
                            scene=scenes[base], camera=scenes[n])
        rep["B"][n] = r
    (a.out / "report.json").write_text(json.dumps(rep, indent=1, default=float) + "\n", encoding="utf-8")

    def line(tag, n, r):
        objs = r.get("objects", {})
        iou = {k: round(v["iou"], 3) for k, v in objs.items()}
        onset = {k: v["onset"]["error_frames"] for k, v in objs.items() if v.get("onset")}
        cam = r.get("camera", {})
        un = r.get("unchanged", {})
        print(f"{tag} {n:13s} edgesF1 {r['edges']['all']['f1']:.3f}  rot {cam.get('rotation_deg_mean', 0):.2f}deg "
              f"centre {cam.get('center_rel_mean') or 0:.3f}  iou {iou}  onset_err {onset}  "
              f"unchanged_mad {un.get('mad')}")
    for n, r in rep["A"].items():
        line("A", n, r)
    for n, r in rep["B"].items():
        line("B", n, r)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
