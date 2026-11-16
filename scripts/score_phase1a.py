"""Score the phase-1a generations (RQ2) with the measures that need no observer, and with masks when they exist.

    python scripts/score_phase1a.py [--gen runs/phase1a/generated] [--controls runs/rq2_dry] [--masks-name masks.npz]

Per run (<scene>__<kind>__s<seed>): vrecon.compare.generated against the scene's own controls with passes
(runs/rq2_dry/<scene>, made by scripts/dry_run_rq2.py from the same scene files):
    edges       edge F1 between the generation and the control's edges (how closely the layout is followed)
    unchanged   for an edit that keeps the camera: mean absolute difference to the BASE generation of the same
                seed on the objects the edit must leave alone (evaluation/phase1a/edits.json)
    objects     IoU / onset per object, only when <run>/masks.npz exists (segmentation of the generated video)
Writes <run>/score.json and prints a table; <gen>/scores.json collects them.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", type=Path, default=ROOT / "runs/phase1a/generated")
    ap.add_argument("--controls", type=Path, default=ROOT / "runs/rq2_dry")
    ap.add_argument("--masks-name", default="masks.npz")
    a = ap.parse_args()

    from vrecon.compare.generated import load_controls, load_masks, score_generated
    from vrecon.control.core import read_video_rgb
    from vrecon.schema import load_project

    edits = json.loads((ROOT / "evaluation/phase1a/edits.json").read_text(encoding="utf-8"))
    by_id = {e["id"]: e for e in edits["edits"]}
    ctrls, rows = {}, []
    runs = sorted(d for d in a.gen.iterdir() if (d / "video.mp4").is_file())
    for d in runs:
        scene, kind, seed = d.name.split("__")
        if scene not in ctrls:
            ctrls[scene] = load_controls(a.controls / scene)
        e = by_id.get(scene)
        base_dir = a.gen / f"{edits['base']}__{kind}__{seed}"
        keep_cam = e is not None and "cam_001" in e["unchanged"]
        masks = load_masks(d / a.masks_name) if (d / a.masks_name).is_file() else None
        rep = score_generated(
            read_video_rgb(d / "video.mp4"), ctrls[scene], masks=masks,
            base_video=read_video_rgb(base_dir / "video.mp4") if keep_cam and (base_dir / "video.mp4").is_file()
            else None,
            unchanged=[u for u in e["unchanged"] if u.startswith("obj_")] if keep_cam else None,
            scene=load_project(ROOT / "evaluation/phase1a/scenes" / f"{scene}.json"))
        rep["run"] = d.name
        (d / "score.json").write_text(json.dumps(rep, indent=1, default=float) + "\n", encoding="utf-8")
        ed = rep["edges"]
        row = {"run": d.name, "scene": scene, "kind": kind, "seed": seed,
               "edge_f1": ed["all"]["f1"] if ed.get("all") else None,
               "edge_precision": ed["all"]["precision"] if ed.get("all") else None,
               "edge_recall": ed["all"]["recall"] if ed.get("all") else None,
               "unchanged_mad": rep.get("unchanged", {}).get("mad"),
               "unchanged_psnr": rep.get("unchanged", {}).get("psnr"),
               "onset_abs_error": rep.get("onset_abs_error_mean")}
        rows.append(row)
        print(" | ".join(f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in row.items()), flush=True)
    (a.gen / "scores.json").write_text(json.dumps(rows, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
