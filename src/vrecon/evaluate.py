"""Score a reconstruction against a reference clip folder (PLAN.md 11.3).

Reference folder layout (synthetic clips from scripts/make_synthetic_clips.py, or annotated real clips):
    gt_passes/index.npz, gt_passes/id_map.json   object masks (required)
    gt_passes/depth.npz, gt.json                 ground-truth depth and project (optional: 3D measures)
    meta.json                                    {"events": [{"object", "onset_frame"}]} (optional: timing)

The reconstruction is rendered with the same pass renderer, so both sides are measured the same way.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

from vrecon.metrics import (
    anchored_reprojection_error,
    camera_errors,
    centroid_error,
    mask_iou,
    match_objects,
    onset_frame_relative,
    timing_errors,
)
from vrecon.schema import Project, load_project


def _load_map(path: Path) -> dict[int, str]:
    return {int(k): v for k, v in json.loads(path.read_text(encoding="utf-8")).items()}


def render_reconstruction_passes(rec: Project, shot_id: str | None, out_dir: Path):
    from vrecon.scene.builder import build_scene
    from vrecon.scene.passes import render_passes

    build_scene(rec, shot_id)
    render_passes(out_dir, shaded=False)
    return (np.load(out_dir / "index.npz")["index"], np.load(out_dir / "depth.npz")["depth"],
            _load_map(out_dir / "id_map.json"))


def _mean(values) -> float | None:
    vals = [v for v in values if v is not None]
    return float(np.mean(vals)) if vals else None


def score(reference_dir: str | Path, rec: Project, shot_id: str | None = None,
          rec_passes: tuple | None = None) -> dict:
    ref_dir = Path(reference_dir)
    ref_index = np.load(ref_dir / "gt_passes" / "index.npz")["index"]
    ref_map = _load_map(ref_dir / "gt_passes" / "id_map.json")
    gt = load_project(ref_dir / "gt.json") if (ref_dir / "gt.json").is_file() else None
    meta = json.loads((ref_dir / "meta.json").read_text(encoding="utf-8")) if (ref_dir / "meta.json").is_file() else {}
    shot_id = shot_id or rec.shots[0].id

    if rec_passes is None:
        with tempfile.TemporaryDirectory() as tmp:
            rec_index, _, rec_map = render_reconstruction_passes(rec, shot_id, Path(tmp))
    else:
        rec_index, _, rec_map = rec_passes
    if rec_index.shape != ref_index.shape:
        raise ValueError(f"reconstruction renders {rec_index.shape}, reference is {ref_index.shape}: "
                         "camera size or frame count differs")

    pairs = match_objects(ref_index, rec_index, ref_map, rec_map)
    set_ids = {o.id for o in gt.reconstruction.objects if o.kind == "set"} if gt else set()
    iou = mask_iou(ref_index, rec_index, ref_map, rec_map, pairs)
    cen = centroid_error(ref_index, rec_index, ref_map, rec_map, pairs)
    report: dict = {
        "reference": str(ref_dir),
        "reconstruction": rec.id,
        "revision": rec.revision,
        "shot_id": shot_id,
        "pairs": pairs,
        "unmatched_reference": sorted(set(ref_map.values()) - set(pairs)),
        "mask_iou": iou,
        "centroid_error": cen,
    }
    if gt is not None:
        report["camera"] = camera_errors(gt, rec, shot_id)
        if (ref_dir / "gt_passes" / "depth.npz").is_file():
            depth = np.load(ref_dir / "gt_passes" / "depth.npz")["depth"]
            report["anchored_reprojection"] = anchored_reprojection_error(
                gt, rec, shot_id, depth, ref_index, ref_map, pairs)
    if meta.get("events"):
        annotated = {e["object"]: e["onset_frame"] for e in meta["events"]}
        report["timing"] = timing_errors(annotated, rec, shot_id, pairs)
        # the same events by the relative rule (metrics.onset_frame_relative): independent of scene units and
        # of wobble before the start. The reference onset is measured the same way on the ground truth when it
        # is there, so the rule's small delay is on both sides.
        ref_rel = {a: onset_frame_relative(gt, a, shot_id) for a in annotated} if gt is not None else annotated
        report["timing_relative"] = timing_errors({a: v for a, v in ref_rel.items() if v is not None}, rec, shot_id,
                                                  pairs, rule="relative")

    movers = [a for a in ref_map.values() if a not in set_ids]
    matched = sum(a in pairs for a in movers)
    summary = {
        "objects": len(movers),
        "matched": matched,
        "matched_share": matched / len(movers) if movers else None,
        # over the matched objects only: a reconstruction that lost an object is not penalised here
        "mask_iou_mean": _mean(iou[a]["mean"] for a in movers if a in iou),
        # over every reference object, a lost one counting 0 (docs/REVIEW_INSTRUCTIONS.md F-14)
        "mask_iou_all_mean": float(np.mean([(iou[a]["mean"] or 0.0) if a in iou else 0.0 for a in movers]))
        if movers else None,
        "centroid_error_mean": _mean(cen[a]["mean"] for a in movers if a in cen),
    }
    if "anchored_reprojection" in report:
        rp = report["anchored_reprojection"]
        summary["reprojection_error_mean"] = _mean(rp[a]["mean"] for a in movers if a in rp)
        summary["reprojection_error_all_mean"] = _mean(v["mean"] for v in rp.values())
    if "camera" in report:
        summary["camera_rotation_deg_mean"] = report["camera"]["rotation_deg_mean"]
        summary["camera_focal_rel_mean"] = report["camera"]["focal_rel_mean"]
        summary["camera_center_rel_mean"] = report["camera"]["center_rel_mean"]
    if "timing" in report:
        errs = [abs(v["error_frames"]) for v in report["timing"].values() if v["error_frames"] is not None]
        summary["timing_abs_error_frames_mean"] = float(np.mean(errs)) if errs else None
        summary["timing_missing"] = sum(v["error_frames"] is None for v in report["timing"].values())
        rel = report["timing_relative"]
        errs = [abs(v["error_frames"]) for v in rel.values() if v["error_frames"] is not None]
        summary["timing_rel_error_frames_mean"] = float(np.mean(errs)) if errs else None
        summary["timing_rel_found_share"] = len(errs) / len(rel) if rel else None
    report["summary"] = summary
    report["measures"] = {
        "scorer_version": 2,  # 2: mask_iou_all_mean, matched_share, timing_relative added (older keys unchanged)
        "note": "image-space measures use rendered masks on both sides; 3D measures use ground truth "
                "(evaluation only, PLAN.md 11.8)",
        "normalisation": "pixel distances are divided by the image diagonal",
    }
    return report


def score_blend(reference_dir: str | Path, blend: str | Path) -> dict:
    """Score any .blend (e.g. an LLM's scene.blend) against a reference clip folder."""
    from vrecon.scene.blend_import import import_blend

    ref_dir = Path(reference_dir)
    ref_index = np.load(ref_dir / "gt_passes" / "index.npz")["index"]
    t, h, w = ref_index.shape
    with tempfile.TemporaryDirectory() as tmp:
        project, warnings = import_blend(blend, Path(tmp), project_id="proj_blend", size=(w, h))
        passes = (np.load(Path(tmp) / "index.npz")["index"], np.load(Path(tmp) / "depth.npz")["depth"],
                  _load_map(Path(tmp) / "id_map.json"))
    if passes[0].shape[0] != t:
        raise ValueError(f".blend has {passes[0].shape[0]} frames, reference has {t}")
    report = score(ref_dir, project, rec_passes=passes)
    report["reconstruction"] = str(blend)
    report["import_warnings"] = warnings
    return report


def score_novel(reference_dir: str | Path, rec: Project, shot_id: str | None = None, root: str | Path | None = None,
                pairs: dict[str, str] | None = None) -> dict:
    """Scores from the novel views of a synthetic clip (gt_passes/novel/*, metrics/novel.py). pairs: ground-truth
    id -> reconstruction id (default: matched on the source view, as score() does)."""
    from vrecon.metrics.novel import carry_relative, novel_scores
    from vrecon.scene.builder import build_scene
    from vrecon.scene.passes import render_passes
    from vrecon.schema import Camera

    ref_dir = Path(reference_dir)
    gt = load_project(ref_dir / "gt.json")
    shot_id = shot_id or rec.shots[0].id
    if pairs is None:
        pairs = score(ref_dir, rec, shot_id)["pairs"]
    set_ids = {o.id for o in gt.reconstruction.objects if o.kind == "set"}
    out = {"views": {}}
    for view in sorted((ref_dir / "gt_passes" / "novel").glob("*")):
        info = json.loads((view / "camera.json").read_text(encoding="utf-8"))
        cam = carry_relative(Camera.model_validate(info["camera"]), gt.reconstruction.cameras[0],
                             rec.reconstruction.cameras[0])
        r = rec.model_copy(deep=True)
        r.reconstruction.cameras = [cam.model_copy(update={"id": rec.reconstruction.cameras[0].id})]
        build_scene(r, shot_id, root=root)
        with tempfile.TemporaryDirectory() as tmp:
            render_passes(tmp, shot_frames=info["frames"], shaded=False)
            rec_index, rec_map = np.load(Path(tmp) / "index.npz")["index"], _load_map(Path(tmp) / "id_map.json")
        ref_index, ref_map = np.load(view / "index.npz")["index"], _load_map(view / "id_map.json")
        out["views"][view.name] = novel_scores(ref_index, ref_map, rec_index, rec_map, pairs, set_ids)
    vals = [v["iou_movers"] for v in out["views"].values() if v["iou_movers"] is not None]
    out["summary"] = {"iou_movers_mean": float(np.mean(vals)) if vals else None,
                      "covered_mean": float(np.mean([v["covered"] for v in out["views"].values()])),
                      "extra_mean": float(np.mean([v["extra"] for v in out["views"].values()]))}
    return out
