"""Joint positions of a skeleton character as the scene renders it (same rule as scene/character.py, no bpy).

The rig's bones rest along the armature axes (roll 0), so each pose bone's quaternion is its rotation relative to
its parent in plain armature axes; the root follows the object's keys (translation in scene units, rotation), the
other joints the skeleton file's local rotations, and the rig object carries the uniform scale (rest.scale).
Props held by a hand must use these positions, not the per-frame GVHMR ray points (analysis/place_person.py): the
rendered body keeps GVHMR's joint rotations and a smoothed root, so its hand can sit 0.2-0.5 m away in depth from
the ray point even where both project to the same pixel (tennis, 2026-11-14).
"""

from __future__ import annotations

import numpy as np

from vrecon.scene.convert import quat_to_matrix


def posed_joints(obj, d: dict) -> np.ndarray:
    """[T, J, 3] scene-unit joint positions per key of `obj` (a character SceneObject) with skeleton `d`."""
    parents = [int(p) for p in d["parents"]]
    rest = np.asarray(d["rest_joints"], float)
    rot = np.asarray(d["local_rot"], float)
    sc = float(obj.rest.scale[0])
    out = np.zeros((len(obj.keys), len(parents), 3))
    for i, key in enumerate(obj.keys):
        g = [None] * len(parents)
        p = [None] * len(parents)
        for j, pa in enumerate(parents):
            q = key.pose.rotation if pa < 0 else tuple(float(x) for x in rot[min(i, len(rot) - 1), j])
            r = np.array(quat_to_matrix(tuple(float(x) for x in q)))
            if pa < 0:
                g[j], p[j] = r, np.array(key.pose.translation) / sc
            else:
                g[j], p[j] = g[pa] @ r, p[pa] + g[pa] @ (rest[j] - rest[pa])
        out[i] = np.array(p) * sc
    return out


__all__ = ["posed_joints"]
