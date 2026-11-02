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
) -> dict:
    """Write control videos and a manifest for one shot. Returns the manifest dict."""
    import numpy as np

    out = Path(out_dir)
    report = build_scene(project, shot_id)
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

    previews = []
    if preview:
        # A small shaded render of the first, middle and last frame, for a quick look.
        build_scene(project, shot_id)
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
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
