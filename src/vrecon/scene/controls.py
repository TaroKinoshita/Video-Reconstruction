"""Export control videos for one shot: build -> geometry passes -> control mp4s -> manifest.json."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import bpy

from vrecon.control.core import (
    FPS,
    canny_control,
    check_h3_constraints,
    depth_control,
    encode_mp4,
    load_frames,
)
from vrecon.schema import Project

from .builder import build_scene, render_frames
from .passes import render_passes


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def export_controls(
    project: Project,
    out_dir: str | Path,
    shot_id: str | None = None,
    kinds: tuple[str, ...] = ("depth", "canny"),
    depth_mode: Literal["clip", "frame", "disparity"] = "clip",
    preview: bool = True,
    strict_h3: bool = False,
    root: str | Path | None = None,
    hypothesis: str | tuple[str, ...] = "full",
) -> dict:
    """Write control videos and a manifest for one shot. Returns the manifest dict.

    kinds: depth, canny, gray (geometry passes) and clay, id, look (colour passes, scene/color_passes.py; with
    id, legend.json maps each identity colour to its object). root: folder asset paths are relative to.
    hypothesis: "full" | "soft" | "drop" or several (control/status.py). Every kind is also written per extra mode
    as control_<kind>__<mode>.mp4 ("full" keeps the plain name); passes/status.npz holds the status pass and the
    manifest the share of pixels per status per frame."""
    import numpy as np

    from vrecon.control.status import apply_mode, shares, status_pass

    modes = (hypothesis,) if isinstance(hypothesis, str) else tuple(hypothesis)

    out = Path(out_dir).resolve()
    report = build_scene(project, shot_id, root=root)
    scene = bpy.context.scene
    w, h = scene.render.resolution_x, scene.render.resolution_y
    n_frames = scene.frame_end - scene.frame_start + 1
    fps = scene.render.fps / scene.render.fps_base
    problems = check_h3_constraints(w, h, n_frames, fps)
    if problems and strict_h3:
        raise ValueError("scene does not fit H3 constraints:\n  - " + "\n  - ".join(problems))

    passes = render_passes(out / "passes", shaded=bool({"canny", "gray"} & set(kinds)))
    index = np.load(out / "passes" / "index.npz")["index"]
    status = status_pass(index, passes.id_map, project, report.shot_id)
    np.savez_compressed(out / "passes" / "status.npz", status=status)
    files: dict[str, dict] = {}

    def write(kind: str, frames, params: dict, extra: dict | None = None) -> None:
        for mode in modes:
            name = f"control_{kind}.mp4" if mode == "full" else f"control_{kind}__{mode}.mp4"
            path = encode_mp4(apply_mode(kind, frames, status, mode), out / name, fps=FPS)
            key = kind if mode == "full" else f"{kind}__{mode}"
            files[key] = {"path": path.name, "sha256": _sha256(path), "params": params, "hypothesis": mode,
                          **(extra or {})}

    if "depth" in kinds:
        depth = np.load(out / "passes" / "depth.npz")["depth"]
        frames, info = depth_control(depth, mode=depth_mode)
        write("depth", frames, info)
    if "canny" in kinds:
        shaded = sorted((out / "passes" / "shaded").glob("frame_*.png"))
        frames, info = canny_control(load_frames(shaded))
        write("canny", frames, info)

    if "gray" in kinds:  # the shaded proxy render itself, as a grayscale control (rendered-proxy conditioning)
        shaded = sorted((out / "passes" / "shaded").glob("frame_*.png"))
        rgb = load_frames(shaded)
        gray = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).round().astype(np.uint8)
        write("gray", np.repeat(gray[..., None], 3, axis=-1), {"source": "workbench shaded pass"})

    if "pose" in kinds:  # OpenPose-style skeletons of the characters (control/pose.py; no rendering)
        from vrecon.control.pose import pose_control

        frames, info = pose_control(project, report.shot_id, n_frames, (w, h), root=root)
        write("pose", frames, info)

    colour_kinds = [k for k in ("clay", "id", "look") if k in kinds]
    if colour_kinds:
        from .color_passes import identity_colors, render_color_pass

        colors = identity_colors(project)
        shot_frames = list(range(n_frames))
        for kind in colour_kinds:
            paths = render_color_pass(kind, out / "passes" / kind, shot_frames, colors)
            write(kind, load_frames(paths), {"source": f"workbench {kind} pass"},
                  {"legend": "legend.json"} if kind == "id" else None)
        if "id" in colour_kinds:
            (out / "legend.json").write_text(json.dumps(colors, indent=2) + "\n", encoding="utf-8")

    if "coverage" in kinds:  # which faces the source video saw (scene/coverage.py), from this same camera
        from .coverage import render_coverage

        step = max(n_frames // 24, 1)
        paths, summary = render_coverage(project, out / "passes" / "coverage", list(range(0, n_frames, step)),
                                         view_frames=list(range(n_frames)), root=root)
        path = encode_mp4(load_frames(paths), out / "control_coverage.mp4", fps=FPS)
        files["coverage"] = {"path": path.name, "sha256": _sha256(path), "params": {"source_frames_step": step},
                             "summary": summary, "note": "not a V2V control: shows what the video saw (green), saw "
                                                         "edge-on (yellow), never saw (red)"}

    previews = []
    if preview:
        # A small shaded render of the first, middle and last frame, for a quick look.
        build_scene(project, shot_id, root=root)
        picks = sorted({0, n_frames // 2, n_frames - 1})
        previews = [
            p.relative_to(out).as_posix()
            for p in render_frames(out / "preview", picks, engine="BLENDER_WORKBENCH", resolution_percentage=25)
        ]

    manifest = {
        "project_id": project.id,
        "revision": project.revision,
        "shot_id": report.shot_id,
        "blender": bpy.app.version_string,
        "width": w,
        "height": h,
        "frames": n_frames,
        "fps": fps,
        "h3_problems": problems,
        "controls": files,
        "passes": {
            "depth": "passes/depth.npz",
            "index": "passes/index.npz",
            "id_map": {str(k): v for k, v in passes.id_map.items()},
            "status": "passes/status.npz",
        },
        "hypothesis_modes": list(modes),
        "status_codes": {"0": "empty", "1": "observed (observed/estimated/human_confirmed)", "2": "standard",
                         "3": "hypothesis"},
        "status_share": shares(status),
        "previews": previews,
        "warnings": report.warnings,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
