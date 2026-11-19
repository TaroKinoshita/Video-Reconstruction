"""Recipe runner for the scene -> V2V part of a clip (docs/REVIEW_INSTRUCTIONS.md WP-5, the stages added for V2V).

    vrecon pipeline-v2v RECIPE.json [--from STAGE] [--to STAGE] [--dry-run]

A recipe (evaluation/recipes/<clip>/v2v.json) names, for one clip: the finished scene folder, the observer files
(object masks, floor masks), the window, the camera moves, the reference-video kind, the look/action texts and the
generation profile. Stages, in order, each writing a new folder under the recipe's "work" folder (inputs are never
overwritten):

    shapes   analysis/shape_library.apply_templates          box-only objects -> category templates
    floor    scripts/fix_floor_contact.py                    off-floor objects moved behind what hides their feet
    export   scripts/export_scene_controls.py (per move)     reference videos for the window
    crops    scripts/make_ref_crops.py                       reference pictures with context
    prompts  scripts/make_prompts.py (per move)              prompts from the scene (inventory, pictures)
    generate gpu/h3_control/comfy_jobs.py                    only with --server (a running ComfyUI)

Every stage appends one line to <work>/pipeline_log.jsonl: stage, parameters, sha256 of its input and output
files, seconds, exit status. Per-clip values live only in the recipe, each with a "basis".
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
STAGES = ["shapes", "floor", "export", "crops", "prompts", "generate"]


def _sha(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _run(cmd: list[str]) -> int:
    print("  $", " ".join(str(c) for c in cmd), flush=True)
    return subprocess.run([str(c) for c in cmd], cwd=ROOT).returncode


def run(recipe_path: Path, start: str | None = None, stop: str | None = None, server: str | None = None,
        dry: bool = False, scene: str | None = None) -> int:
    """scene: the finished scene folder to start from (e.g. the output of `vrecon pipeline-scene`); default: the
    recipe's "scene". Generation reuses a stored video only when its inputs' fingerprint matches (comfy_jobs)."""
    r = json.loads(recipe_path.read_text(encoding="utf-8"))
    work = ROOT / r["work"]
    work.mkdir(parents=True, exist_ok=True)
    log = work / "pipeline_log.jsonl"
    py = sys.executable
    v = lambda k: r["values"][k]["value"]  # noqa: E731  every per-clip value carries a basis
    scene0 = Path(scene).resolve() if scene else ROOT / r["scene"]
    d_shapes, d_floor, d_ctrl, d_crops = work / "scene_shapes", work / "scene_floor", work / "controls", work / "refs"
    jobs = work / "jobs.json"
    todo = STAGES[STAGES.index(start) if start else 0: STAGES.index(stop) + 1 if stop else None]
    for stage in todo:
        t0, status, ins, outs = time.time(), 0, [], []
        print(f"[{stage}]", flush=True)
        if dry:
            continue
        if stage == "shapes":
            from vrecon.analysis.shape_library import apply_templates
            from vrecon.schema import load_project, save_project
            import shutil

            shutil.copytree(scene0, d_shapes, dirs_exist_ok=True)
            p, rep = apply_templates(load_project(scene0 / "project.json"), d_shapes)
            save_project(p, d_shapes / "project.json")
            (d_shapes / "shapes.json").write_text(json.dumps(rep, indent=1) + "\n", encoding="utf-8")
            ins, outs = [scene0 / "project.json"], [d_shapes / "project.json"]
        elif stage == "floor":
            maps = [f"{k}={x}" for k, x in v("mask_map").items()]
            status = _run([py, "scripts/fix_floor_contact.py", d_shapes, ROOT / r["masks"], ROOT / r["floor_masks"],
                           d_floor, "--map", *maps])
            ins, outs = [d_shapes / "project.json", ROOT / r["masks"], ROOT / r["floor_masks"]], [d_floor / "project.json"]
        elif stage == "export":
            for mv in v("moves"):
                status |= _run([py, "scripts/export_scene_controls.py", d_floor / "project.json", d_floor, d_ctrl,
                                "--moves", mv, "--frames", v("frames"), "--start-time", v("start_time"),
                                "--kinds", v("reference_kind")])
            ins = [d_floor / "project.json"]
            outs = [d_ctrl / f"controls_{mv}" / f"control_{v('reference_kind')}.mp4" for mv in v("moves")]
        elif stage == "crops":
            maps = [f"{k}={x}" for k, x in v("crop_map").items()]
            status = _run([py, "scripts/make_ref_crops.py", ROOT / r["video"], ROOT / r["masks"], d_floor / "project.json",
                           d_crops, "--start", v("start_time"), "--end", v("start_time") + v("frames") / 24.0,
                           "--map", *maps])
            ins, outs = [ROOT / r["video"], ROOT / r["masks"]], [d_crops / "crops.json"]
        elif stage == "prompts":
            if jobs.is_file():
                jobs.unlink()
            for mv in v("moves"):
                name = r["move_names"].get(mv, mv)
                status |= _run([py, "scripts/make_prompts.py", "--project", d_floor / "project.json",
                                "--look", ROOT / r["look"], "--action", ROOT / r["action"], "--main", v("main_object"),
                                "--main-role", v("main_role"), "--variant", f"{v('reference_kind')}={d_ctrl / f'controls_{mv}'}",
                                "--crops", d_crops / "crops.json", "--name", name, "--group", "main", "--out", jobs])
            ins, outs = [d_floor / "project.json", d_crops / "crops.json"], [jobs]
        elif stage == "generate":
            if not server:
                print("  skipped: no --server")
                continue
            status = _run([py, "gpu/h3_control/comfy_jobs.py", "--server", server, "--profile", ROOT / r["profile"],
                           "--jobs", jobs, "--out", work / "generated", "--group", "main", "--seeds", *map(str, v("seeds"))])
            ins, outs = [jobs], []
        rec = {"stage": stage, "status": status, "seconds": round(time.time() - t0, 1),
               "inputs": {str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p): _sha(p) for p in ins},
               "outputs": {str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p): _sha(p) for p in outs},
               "recipe": str(recipe_path)}
        with open(log, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
        if status:
            print(f"stage {stage} failed ({status})")
            return status
    return 0


__all__ = ["STAGES", "run"]
