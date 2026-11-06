"""Render geometry passes from the current Blender scene (built by builder.build_scene).

Two renders per frame:
  data pass    Cycles, 1 sample, no pixel filter, material override emitting
               (R, G, B) = (view-Z depth in scene units, object pass index, 1 = covered).
               Saved as float EXR and read back exactly. Background is (0, 0, 0).
  shaded pass  Workbench with flat per-object colours, for edge (Canny) controls.

Outputs in out_dir:
  depth.npz   float16 [T, H, W] view-Z depth (CONVENTIONS.md: planar depth, scene units), 0 = background
  index.npz   uint16  [T, H, W] object number, 0 = background; numbers map to ids via id_map.json
  id_map.json {"1": "obj_box", ...}
  shaded/frame_XXXXXX.png
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

import bpy
import numpy as np

from .builder import ID_PROP


@dataclass
class PassesResult:
    out_dir: Path
    shot_frames: list[int]
    width: int
    height: int
    id_map: dict[int, str]


def _assign_pass_indices() -> dict[int, str]:
    meshes = sorted((o for o in bpy.data.objects if o.get(ID_PROP) and o.type == "MESH"), key=lambda o: o[ID_PROP])
    if len(meshes) > 65535:
        raise ValueError("too many objects for a uint16 index pass")
    id_map = {}
    for i, obj in enumerate(meshes, start=1):
        obj.pass_index = i
        id_map[i] = obj[ID_PROP]
    for obj in bpy.data.objects:
        if obj.type == "MESH" and not obj.get(ID_PROP):
            obj.pass_index = 0
    return id_map


def _data_material() -> bpy.types.Material:
    mat = bpy.data.materials.get("vrecon_data") or bpy.data.materials.new("vrecon_data")
    nt = mat.node_tree  # Blender 5 materials always have a node tree
    nt.nodes.clear()
    cam = nt.nodes.new("ShaderNodeCameraData")
    info = nt.nodes.new("ShaderNodeObjectInfo")
    comb = nt.nodes.new("ShaderNodeCombineXYZ")
    emit = nt.nodes.new("ShaderNodeEmission")
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(cam.outputs["View Z Depth"], comb.inputs[0])
    nt.links.new(info.outputs["Object Index"], comb.inputs[1])
    comb.inputs[2].default_value = 1.0
    nt.links.new(comb.outputs[0], emit.inputs["Color"])
    emit.inputs["Strength"].default_value = 1.0
    nt.links.new(emit.outputs[0], out.inputs["Surface"])
    return mat


def _setup_data_render(scene) -> dict:
    """Configure exact data rendering; returns previous settings to restore."""
    prev = {
        "engine": scene.render.engine,
        "override": scene.view_layers[0].material_override,
        "view_transform": scene.view_settings.view_transform,
        "look": scene.view_settings.look,
        "world": scene.world,
        "percent": scene.render.resolution_percentage,
    }
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"  # measured: faster than the GPU for this 1-sample render, same index values
    scene.cycles.samples = 1
    scene.cycles.use_denoising = False
    scene.cycles.filter_width = 0.01  # no anti-aliasing: every pixel is one surface
    scene.cycles.max_bounces = 0
    scene.render.resolution_percentage = 100
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    world = bpy.data.worlds.get("vrecon_black") or bpy.data.worlds.new("vrecon_black")
    world.color = (0.0, 0.0, 0.0)
    bg = world.node_tree.nodes.get("Background") if world.node_tree else None
    if bg is not None:
        bg.inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
        bg.inputs["Strength"].default_value = 0.0
    scene.world = world
    scene.view_layers[0].material_override = _data_material()
    s = scene.render.image_settings
    s.file_format = "OPEN_EXR"
    s.color_depth = "32"
    s.exr_codec = "ZIP"
    return prev


def _restore(scene, prev: dict) -> None:
    scene.render.engine = prev["engine"]
    scene.view_layers[0].material_override = prev["override"]
    scene.view_settings.view_transform = prev["view_transform"]
    scene.view_settings.look = prev["look"]
    scene.world = prev["world"]
    scene.render.resolution_percentage = prev["percent"]


def _read_exr(path: Path) -> np.ndarray:
    img = bpy.data.images.load(str(path))
    try:
        w, h = img.size
        buf = np.empty(w * h * 4, dtype=np.float32)
        img.pixels.foreach_get(buf)
        return buf.reshape(h, w, 4)[::-1]  # Blender stores rows bottom-up
    finally:
        bpy.data.images.remove(img)


def render_passes(out_dir: str | Path, shot_frames: list[int] | None = None, shaded: bool = True) -> PassesResult:
    scene = bpy.context.scene
    if shot_frames is None:
        shot_frames = list(range(scene.frame_end - scene.frame_start + 1))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    id_map = _assign_pass_indices()
    w, h = scene.render.resolution_x, scene.render.resolution_y

    depth = np.zeros((len(shot_frames), h, w), dtype=np.float16)
    index = np.zeros((len(shot_frames), h, w), dtype=np.uint16)
    prev = _setup_data_render(scene)
    with tempfile.TemporaryDirectory() as tmp:
        exr = Path(tmp) / "data.exr"
        for i, f in enumerate(shot_frames):
            scene.frame_set(scene.frame_start + int(f))
            scene.render.filepath = str(exr)
            bpy.ops.render.render(write_still=True)
            px = _read_exr(exr)
            covered = px[..., 2] > 0.5
            depth[i] = np.where(covered, px[..., 0], 0.0).astype(np.float16)
            index[i] = np.where(covered, np.rint(px[..., 1]), 0).astype(np.uint16)
    _restore(scene, prev)
    np.savez_compressed(out / "depth.npz", depth=depth)
    np.savez_compressed(out / "index.npz", index=index)
    (out / "id_map.json").write_text(json.dumps({str(k): v for k, v in id_map.items()}, indent=2) + "\n", encoding="utf-8")

    if shaded:
        _render_shaded(scene, out / "shaded", shot_frames)
    return PassesResult(out_dir=out, shot_frames=shot_frames, width=w, height=h, id_map=id_map)


def _render_shaded(scene, out_dir: Path, shot_frames: list[int]) -> None:
    """Flat-lit, per-object random colours: edges between objects and between faces stay visible."""
    out_dir.mkdir(parents=True, exist_ok=True)
    prev_engine = scene.render.engine
    scene.render.engine = "BLENDER_WORKBENCH"
    shading = scene.display.shading
    shading.light = "STUDIO"
    shading.color_type = "RANDOM"
    shading.show_cavity = False
    shading.show_shadows = False
    shading.background_type = "VIEWPORT"
    shading.background_color = (0.0, 0.0, 0.0)
    scene.render.resolution_percentage = 100
    s = scene.render.image_settings
    s.file_format = "PNG"
    s.color_mode = "RGB"
    s.color_depth = "8"
    scene.view_settings.view_transform = "Standard"
    for f in shot_frames:
        scene.frame_set(scene.frame_start + int(f))
        scene.render.filepath = str(out_dir / f"frame_{f:06d}.png")
        bpy.ops.render.render(write_still=True)
    scene.render.engine = prev_engine
