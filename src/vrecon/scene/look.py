"""Give a built scene a textured, lit look so it can stand in for a real reference video.

Used for synthetic ground-truth clips (evaluation only). Textures matter: point trackers and
segmenters need surface detail, and plain grey boxes would make measurement unrealistically hard.
"""

from __future__ import annotations

import random

import bpy

from .builder import ID_PROP


def _material(name: str, rng: random.Random, kind: str) -> bpy.types.Material:
    mat = bpy.data.materials.new(name)
    nt = mat.node_tree
    bsdf = nt.nodes.get("Principled BSDF")
    if bsdf is None:
        nt.nodes.clear()
        bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
        out = nt.nodes.new("ShaderNodeOutputMaterial")
        nt.links.new(bsdf.outputs[0], out.inputs["Surface"])
    base = [rng.uniform(0.15, 0.9) for _ in range(3)]
    dark = [c * rng.uniform(0.25, 0.5) for c in base]
    coord = nt.nodes.new("ShaderNodeTexCoord")
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = (*dark, 1.0)
    ramp.color_ramp.elements[1].color = (*base, 1.0)
    if kind == "checker":
        tex = nt.nodes.new("ShaderNodeTexChecker")
        tex.inputs["Scale"].default_value = rng.uniform(3, 8)
        nt.links.new(coord.outputs["Object"], tex.inputs["Vector"])
        nt.links.new(tex.outputs["Fac"], ramp.inputs["Fac"])
    elif kind == "brick":
        tex = nt.nodes.new("ShaderNodeTexBrick")
        tex.inputs["Scale"].default_value = rng.uniform(2, 5)
        nt.links.new(coord.outputs["Object"], tex.inputs["Vector"])
        nt.links.new(tex.outputs["Fac"], ramp.inputs["Fac"])
    else:  # noise
        tex = nt.nodes.new("ShaderNodeTexNoise")
        tex.inputs["Scale"].default_value = rng.uniform(4, 12)
        tex.inputs["Detail"].default_value = 6.0
        nt.links.new(coord.outputs["Object"], tex.inputs["Vector"])
        nt.links.new(tex.outputs["Fac"], ramp.inputs["Fac"])
    nt.links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = rng.uniform(0.4, 0.9)
    return mat


def apply_look(seed: int = 0) -> None:
    """Materials per object (by vrecon_id, deterministic), a key light, a fill light and a sky colour."""
    rng = random.Random(seed)
    scene = bpy.context.scene
    for obj in sorted((o for o in bpy.data.objects if o.type == "MESH" and o.get(ID_PROP)), key=lambda o: o[ID_PROP]):
        vid = obj[ID_PROP]
        kind = "checker" if "floor" in vid else "brick" if "wall" in vid else rng.choice(["noise", "checker", "noise"])
        obj.data.materials.clear()
        obj.data.materials.append(_material(f"look_{vid}", rng, kind))

    for o in [o for o in bpy.data.objects if o.type == "LIGHT"]:
        bpy.data.objects.remove(o)
    key = bpy.data.objects.new("look_key", bpy.data.lights.new("look_key", "SUN"))
    key.data.energy = 3.0
    key.data.angle = 0.1
    key.rotation_euler = (rng.uniform(0.5, 0.9), 0.0, rng.uniform(-1.0, 1.0))
    scene.collection.objects.link(key)
    fill = bpy.data.objects.new("look_fill", bpy.data.lights.new("look_fill", "SUN"))
    fill.data.energy = 0.6
    fill.rotation_euler = (1.2, 0.0, key.rotation_euler[2] + 3.0)
    scene.collection.objects.link(fill)
    world = bpy.data.worlds.new("look_world")
    world.color = (0.35, 0.38, 0.42)
    bg = world.node_tree.nodes.get("Background") if world.node_tree else None
    if bg is not None:  # Blender 5 worlds render through their node tree
        bg.inputs["Color"].default_value = (0.35, 0.38, 0.42, 1.0)
        bg.inputs["Strength"].default_value = 1.0
    scene.world = world
    scene.view_settings.view_transform = "AgX"
