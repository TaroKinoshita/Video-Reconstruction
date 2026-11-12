"""Colour passes of a built scene for V2V conditioning (Workbench, fast). Three kinds:

    clay  one neutral grey, studio light and shadows: shape, shading, motion
    id    one fixed identity colour per object, flat (no light): which object is where. A symbol, not the look;
          the exact identity is the numeric index pass (passes.py), the colours only make it readable
    look  each object's rough appearance (SceneObject.look, per-vertex colours of mesh assets), studio light

Identity colours come from the object id and kind (deterministic: the same object keeps its colour across
renders and shots) and are never stored in the project. Rules: people get the most saturated colours, one each;
moving things saturated; static props muted; set pieces (floor, walls, background) greys; neighbouring entries
of each list alternate in brightness, so a grey (luminance) control still separates them.
"""

from __future__ import annotations

import colorsys
from pathlib import Path

import bpy
import numpy as np

from vrecon.schema import Project

from .builder import ID_PROP, LOOK_ATTR

ID_ATTR = "vrecon_identity"

# hue (0..1), saturation, value; alternate bright / dark
PEOPLE = [(0.08, 1.0, 1.0), (0.52, 1.0, 0.65), (0.86, 0.9, 1.0), (0.16, 1.0, 0.6), (0.33, 0.9, 1.0), (0.72, 0.8, 0.6)]
MOVERS = [(0.60, 0.9, 1.0), (0.98, 0.8, 0.55), (0.45, 0.9, 0.9), (0.04, 0.6, 0.5), (0.25, 0.8, 0.95), (0.78, 0.7, 0.5)]
PROPS = [(0.12, 0.35, 0.8), (0.55, 0.35, 0.45), (0.95, 0.35, 0.75), (0.38, 0.35, 0.4), (0.68, 0.35, 0.7)]
SET_GREYS = {"floor": 0.8, "wall": 0.5, "background": 0.35}
OTHER_SET = [0.65, 0.25, 0.9]


def _rgb(hsv) -> tuple[float, float, float]:
    return tuple(float(c) for c in colorsys.hsv_to_rgb(*hsv))


def _role(obj, project: Project) -> str:
    if obj.kind == "character":
        return "person"
    if obj.kind == "set":
        return "set"
    return "mover" if obj.keys else "prop"


def identity_colors(project: Project) -> dict[str, dict]:
    """id -> {"rgb", "role"}; order within a role by id, so adding an object of another role changes nothing."""
    out = {}
    objs = sorted(project.reconstruction.objects, key=lambda o: o.id)
    counters = {"person": 0, "mover": 0, "prop": 0, "set": 0}
    for o in objs:
        role = _role(o, project)
        i = counters[role]
        counters[role] += 1
        if role == "person":
            rgb = _rgb(PEOPLE[i % len(PEOPLE)])
        elif role == "mover":
            rgb = _rgb(MOVERS[i % len(MOVERS)])
        elif role == "prop":
            rgb = _rgb(PROPS[i % len(PROPS)])
        else:
            key = next((k for k in SET_GREYS if k in o.id), None)
            g = SET_GREYS[key] if key else OTHER_SET[i % len(OTHER_SET)]
            rgb = (g, g, g * (1.05 if key == "wall" else 1.0))
            rgb = tuple(min(c, 1.0) for c in rgb)
        lum = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
        out[o.id] = {"rgb": [round(c, 4) for c in rgb], "role": role, "luminance": round(lum, 4), "name": o.name or o.id}
    return out


def _meshes():
    return [o for o in bpy.data.objects if o.type == "MESH" and o.get(ID_PROP)]


def _set_identity(colors: dict[str, dict]) -> None:
    for o in _meshes():
        rgb = colors.get(o[ID_PROP], {"rgb": [0.5, 0.5, 0.5]})["rgb"]
        mesh = o.data
        attr = mesh.color_attributes.get(ID_ATTR) or mesh.color_attributes.new(ID_ATTR, "FLOAT_COLOR", "POINT")
        attr.data.foreach_set("color", np.tile([*rgb, 1.0], len(mesh.vertices)).tolist())


def _activate(name: str) -> None:
    for o in _meshes():
        ca = o.data.color_attributes
        if ca.get(name) is not None:
            ca.active_color_name = name
            ca.render_color_index = ca.find(name)


def render_color_pass(kind: str, out_dir: str | Path, shot_frames: list[int], colors: dict[str, dict]) -> list[Path]:
    """Render one of clay / id / look for the current scene; PNG per shot frame. Leaves the scene's render settings
    as they were."""
    from .builder import render_frames

    scene = bpy.context.scene
    sh = scene.display.shading
    keep = (scene.render.engine, sh.light, sh.color_type, sh.show_shadows, sh.show_cavity, tuple(sh.single_color),
            sh.background_type, tuple(sh.background_color), scene.display_settings.display_device,
            scene.view_settings.view_transform)
    scene.render.engine = "BLENDER_WORKBENCH"
    sh.show_cavity = False
    sh.background_type = "VIEWPORT"
    if kind == "clay":
        sh.light, sh.color_type, sh.show_shadows = "STUDIO", "SINGLE", True
        sh.single_color = (0.7, 0.7, 0.7)
        sh.background_color = (0.15, 0.15, 0.15)
    elif kind == "id":
        _set_identity(colors)
        _activate(ID_ATTR)
        sh.light, sh.color_type, sh.show_shadows = "FLAT", "VERTEX", False
        sh.background_color = (0.0, 0.0, 0.0)
        scene.view_settings.view_transform = "Standard"  # identity colours as given, no filmic curve
    elif kind == "look":
        _activate(LOOK_ATTR)
        sh.light, sh.color_type, sh.show_shadows = "STUDIO", "VERTEX", True
        sh.background_color = (0.15, 0.15, 0.15)
    else:
        raise ValueError(f"unknown colour pass {kind!r}")
    try:
        return render_frames(Path(out_dir).resolve(), shot_frames, engine="BLENDER_WORKBENCH")
    finally:
        (scene.render.engine, sh.light, sh.color_type, sh.show_shadows, sh.show_cavity, sh.single_color,
         sh.background_type, sh.background_color, scene.display_settings.display_device,
         scene.view_settings.view_transform) = keep


__all__ = ["ID_ATTR", "identity_colors", "render_color_pass"]
