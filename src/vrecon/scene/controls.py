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
    depth_mode: Literal["clip", "frame"] = "clip",
    preview: bool = True,
    strict_h3: bool = False,
    root: str | Path | None = None,
) -> dict:
    """Write control videos and a manifest for one shot. Returns the manifest dict.

    kinds: depth, canny, gray (geometry passes) and clay, id, look (colour passes, scene/color_passes.py; with
    id, legend.json maps each identity colour to its object). root: folder asset paths are relative to."""
    import numpy as np

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
    files: dict[str, dict] = {}
    if "depth" in kinds:
        depth = np.load(out / "passes" / "depth.npz")["depth"]
        frames, info = depth_control(depth, mode=depth_mode)
        path = encode_mp4(frames, out / "control_depth.mp4", fps=FPS)
        files["depth"] = {"path": path.name, "sha256": _sha256(path), "params": info}
    if "canny" in kinds:
        shaded = sorted((out / "passes" / "shaded").glob("frame_*.png"))
        frames, info = canny_control(load_frames(shaded))
        path = encode_mp4(frames, out / "control_canny.mp4", fps=FPS)
        files["canny"] = {"path": path.name, "sha256": _sha256(path), "params": info}

    if "gray" in kinds:  # the shaded proxy render itself, as a grayscale control (rendered-proxy conditioning)
        shaded = sorted((out / "passes" / "shaded").glob("frame_*.png"))
        rgb = load_frames(shaded)
        gray = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).round().astype(np.uint8)
        path = encode_mp4(np.repeat(gray[..., None], 3, axis=-1), out / "control_gray.mp4", fps=FPS)
        files["gray"] = {"path": path.name, "sha256": _sha256(path), "params": {"source": "workbench shaded pass"}}

    colour_kinds = [k for k in ("clay", "id", "look") if k in kinds]
    if colour_kinds:
        from .color_passes import identity_colors, render_color_pass

        colors = identity_colors(project)
        shot_frames = list(range(n_frames))
        for kind in colour_kinds:
            paths = render_color_pass(kind, out / "passes" / kind, shot_frames, colors)
            path = encode_mp4(load_frames(paths), out / f"control_{kind}.mp4", fps=FPS)
            files[kind] = {"path": path.name, "sha256": _sha256(path), "params": {"source": f"workbench {kind} pass"}}
        if "id" in colour_kinds:
            (out / "legend.json").write_text(json.dumps(colors, indent=2) + "\n", encoding="utf-8")
            files["id"]["legend"] = "legend.json"

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
        },
        "previews": previews,
        "warnings": report.warnings,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
