"""A short motion-graphics clip with deliberate easing (PLAN v2 P1 material): coloured shapes on a floor in front of a
back wall; a cube slides in with ease-in-out and stops, a sphere bounces twice (squash-free, eased arcs), a cylinder
turns 90 degrees with ease-out, the camera dollies in slowly. 96 frames at 24 fps, 640x352, Cycles.

    python scripts/make_mograph_clip.py OUT_DIR

Writes OUT_DIR/video.mp4 (and frames/). No ground truth is written: this clip is for looking, not scoring.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import bpy


def ease(fc_owner, data_path, index, keys, interp="BEZIER", easing="EASE_IN_OUT"):
    for f, v in keys:
        setattr_path = getattr(fc_owner, data_path)
        setattr_path[index] = v
        fc_owner.keyframe_insert(data_path, index=index, frame=f)
    from bpy_extras import anim_utils

    ad = fc_owner.animation_data
    bag = anim_utils.action_get_channelbag_for_slot(ad.action, ad.action_slot)
    for fc in bag.fcurves:
        if fc.data_path == data_path and fc.array_index == index:
            for kp in fc.keyframe_points:
                kp.interpolation = interp
                kp.easing = easing


def material(name, rgb):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    bsdf = m.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*rgb, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.45
    return m


def main() -> int:
    out = Path(sys.argv[1]).resolve()
    out.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.frame_start, sc.frame_end = 1, 96
    sc.render.fps = 24
    sc.render.resolution_x, sc.render.resolution_y = 640, 352
    sc.render.engine = "CYCLES"
    sc.cycles.samples = 32
    sc.world = bpy.data.worlds.new("w")
    sc.world.use_nodes = True
    sc.world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.9, 0.9, 0.92, 1)
    sc.world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.6

    bpy.ops.mesh.primitive_plane_add(size=20)
    bpy.context.object.data.materials.append(material("floor", (0.85, 0.78, 0.66)))
    bpy.ops.mesh.primitive_plane_add(size=20, location=(0, 6, 5), rotation=(math.pi / 2, 0, 0))
    bpy.context.object.data.materials.append(material("wall", (0.35, 0.55, 0.75)))

    bpy.ops.mesh.primitive_cube_add(size=1, location=(-6, 0, 0.5))
    cube = bpy.context.object
    cube.data.materials.append(material("cube", (0.9, 0.3, 0.2)))
    ease(cube, "location", 0, [(8, -6.0), (40, -1.6)])  # slides in, eased stop

    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.4, location=(1.2, 0.5, 0.4))
    ball = bpy.context.object
    ball.data.materials.append(material("ball", (0.95, 0.8, 0.2)))
    bpy.ops.object.shade_smooth()
    for f0, h in ((30, 2.0), (54, 1.1)):  # two eased bounces: up (ease-out), down (ease-in)
        ease(ball, "location", 2, [(f0, 0.4)], "BEZIER", "EASE_OUT")
        ease(ball, "location", 2, [(f0 + 12, 0.4 + h)], "BEZIER", "EASE_OUT")
        ease(ball, "location", 2, [(f0 + 24, 0.4)], "BEZIER", "EASE_IN")

    bpy.ops.mesh.primitive_cylinder_add(radius=0.45, depth=1.4, location=(3.2, 1.5, 0.7))
    cyl = bpy.context.object
    cyl.data.materials.append(material("cyl", (0.25, 0.7, 0.4)))
    bpy.ops.mesh.primitive_cube_add(size=0.25, location=(3.65, 1.5, 1.2))  # a notch so the turn is visible
    notch = bpy.context.object
    notch.data.materials.append(material("notch", (0.1, 0.1, 0.1)))
    notch.parent = cyl
    notch.matrix_parent_inverse = cyl.matrix_world.inverted()
    ease(cyl, "rotation_euler", 2, [(60, 0.0), (84, math.pi / 2)], "BEZIER", "EASE_OUT")

    bpy.ops.object.light_add(type="SUN", rotation=(0.8, 0.2, 0.6))
    bpy.context.object.data.energy = 3.0
    cam_data = bpy.data.cameras.new("cam")
    cam_data.lens = 35
    cam = bpy.data.objects.new("cam", cam_data)
    sc.collection.objects.link(cam)
    sc.camera = cam
    cam.rotation_euler = (math.radians(72), 0, 0)
    ease(cam, "location", 1, [(1, -11.0), (96, -9.0)], "BEZIER", "EASE_IN_OUT")
    cam.location.x, cam.location.z = 0.6, 3.2
    cam.keyframe_insert("location", index=0, frame=1)
    cam.keyframe_insert("location", index=2, frame=1)

    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
        prefs.compute_device_type = "OPTIX"
        prefs.get_devices()
        for d in prefs.devices:
            d.use = True
        sc.cycles.device = "GPU"
    except Exception:  # noqa: BLE001
        pass
    sc.render.image_settings.file_format = "PNG"
    sc.render.filepath = str(out / "frames" / "f_")
    bpy.ops.render.render(animation=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(out / "source.blend"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
