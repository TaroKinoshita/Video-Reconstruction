"""H5, second half: score each edited scene against the ground truth edited the same way (REVIEW F-13).

    python scripts/edit_scores.py [--root runs/rq1]

For every scene the editability test passed an edit on (runs/rq1/edits/details.json): the reconstruction is edited
by the same script (scene/edit_blend.run_edit) and saved; the ground truth scene (gt.json, built with its assets) is
edited with the SAME edit on the object the scorer pairs with the reconstruction's target, its passes rendered into
<root>/edit_scores/gt/<clip>/<edit>/ (cached); then score_blend(edited gt, edited reconstruction). Edits are relative
(camera truck as a share of the camera-to-target distance, size factors, a 1 s delay), so units do not matter.
Writes <root>/edit_scores/scores.json {"<cell>/<clip>/<run>": {edit: mask_iou_all_mean}} and the unedited score.
Ground truth is read here only to score, as everywhere.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _gt_edited(clip: Path, edit: str, gt_target: str, out: Path) -> Path:
    import bpy  # noqa: F401

    from vrecon.scene.builder import build_scene, save_blend
    from vrecon.scene.edit_blend import run_edit
    from vrecon.scene.passes import render_passes
    from vrecon.schema import load_project

    d = out / clip.name / f"{edit}__{gt_target}"
    if (d / "gt_passes" / "index.npz").is_file():
        return d
    d.mkdir(parents=True, exist_ok=True)
    gt = load_project(clip / "gt.json")
    build_scene(gt, root=clip)
    base = save_blend(d / "gt.blend")
    r = run_edit(base, edit, gt_target)
    if not r["success"]:
        raise RuntimeError(f"ground truth edit failed: {r}")
    render_passes(d / "gt_passes", shaded=False)
    return d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ROOT / "runs" / "rq1")
    a = ap.parse_args()
    from vrecon.evaluate import score_blend
    from vrecon.scene.edit_blend import run_edit
    import bpy

    details = json.loads((a.root / "edits" / "details.json").read_text(encoding="utf-8"))
    out = a.root / "edit_scores"
    out.mkdir(parents=True, exist_ok=True)
    sp = out / "scores.json"
    scores = json.loads(sp.read_text(encoding="utf-8")) if sp.is_file() else {}
    for key, rows in details.items():
        cell, clip_name, run = key.split("/")
        if key in scores or not any(r["success"] for r in rows):
            continue
        clip = ROOT / "evaluation" / "synthetic_v2" / clip_name
        if cell == "S1":
            blend = a.root / "edits" / "s1_blends" / f"{clip_name}.blend"
        elif cell == "L0":
            blend = a.root / "L0" / clip_name / run / "workspace" / "output" / "scene.blend"
        else:
            blend = a.root / "L1" / clip_name / run / "refined.blend"
        base = score_blend(clip, blend)
        inv = {rec: ref for ref, rec in base["pairs"].items()}
        row = {"unedited": base["summary"]["mask_iou_all_mean"]}
        for r in rows:
            if not r["success"]:
                row[r["edit"]] = None
                continue
            gt_target = inv.get(r["target"])
            if gt_target is None:
                row[r["edit"]] = None  # the edited object matches nothing in the ground truth
                continue
            gdir = _gt_edited(clip, r["edit"], gt_target, out / "gt")
            shutil.copy(clip / "gt.json", gdir / "gt.json")
            run_edit(blend, r["edit"], r["target"])
            ed = out / "tmp_edited.blend"
            bpy.ops.wm.save_as_mainfile(filepath=str(ed), copy=True)
            row[r["edit"]] = score_blend(gdir, ed)["summary"]["mask_iou_all_mean"]
        scores[key] = row
        sp.write_text(json.dumps(scores, indent=1) + "\n", encoding="utf-8")
        print(key, json.dumps(row), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
