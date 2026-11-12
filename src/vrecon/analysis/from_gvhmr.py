"""A person from GVHMR (gpu/observe/run_gvhmr.py -> gvhmr_skeleton.npz) as a vrecon SceneObject.

    kind      "character"
    asset     the skeleton file (joint rotations per frame; the builder makes an armature + simple body from it)
    keys      one per video frame, time = start_time + frame / fps of the video GVHMR read (seconds, so a project
              at another frame rate still lines up): the pelvis position and the body's orientation
    size      the box around the joints over the clip (for overviews only; the body is not a box)

Placement, two ways:
    camera=None   GVHMR's own world (gravity-aligned, floor at the lowest body vertex). Its path is built from
                  velocities and drifts from what the camera sees (tennis demo: 10-35 px, up to 0.35 m by frame 300).
    camera=Camera GVHMR's in-camera estimate put through the scene's camera, frame by frame: lines up with the
                  image (as well as GVHMR's in-camera estimate does). Depth along the view ray comes from the body
                  model's size, so it agrees with the scene only if the scene's scale is metric.
The joint rotations below the pelvis are the same either way; only the root (keys) changes.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from vrecon.schema import Camera, Evidence, FileRef, Pose, SceneObject, TransformKey


def load_skeleton(path: str | Path) -> dict:
    with np.load(path) as d:
        out = {k: d[k] for k in d.files}
    out["meta"] = json.loads(str(out["meta"]))
    out["joint_names"] = [str(n) for n in out["joint_names"]]
    return out


def character_from_gvhmr(path: str | Path, rel_path: str, obj_id: str = "obj_person_01", start_time: float = 0.0,
                         name: str = "", camera: Camera | None = None, first_shot_frame: int = 0) -> SceneObject:
    """path: the skeleton file; rel_path: how the project refers to it (relative to the project file).
    camera: place through this camera; skeleton frame i is the camera's shot frame first_shot_frame + i."""
    from vrecon.scene.convert import matrix_to_quat, quat_to_matrix

    d = load_skeleton(path)
    fps = float(d["fps"])
    n = int(d["frames"])
    if camera is None:
        pos = d["root_pos"]
        rot = [tuple(float(x) for x in q) for q in d["local_rot"][:, 0]]
        j = d["joints_world"]
    else:
        from vrecon.metrics.poses import camera_pose

        if "root_rot_cam" not in d:
            raise ValueError("skeleton file has no root_rot_cam; re-export it with gpu/observe/run_gvhmr.py")
        pos, rot, js = [], [], []
        for i in range(n):
            r, c, _ = camera_pose(camera, first_shot_frame + i)
            pos.append(c + r @ d["joints_cam"][i, 0])
            js.append(d["joints_cam"][i] @ r.T + c)
            m = r @ np.array(quat_to_matrix(tuple(float(x) for x in d["root_rot_cam"][i])))
            rot.append(tuple(matrix_to_quat(m.tolist())))
        pos, j = np.array(pos), np.array(js)
    keys = [TransformKey(time=start_time + i / fps,
                         pose=Pose(translation=tuple(float(x) for x in pos[i]), rotation=rot[i]))
            for i in range(n)]
    lo, hi = j.reshape(-1, 3).min(0), j.reshape(-1, 3).max(0)
    return SceneObject(
        id=obj_id, name=name or obj_id, kind="character",
        asset=FileRef(id=f"file_{obj_id[4:]}_skeleton", path=rel_path, media_type="application/x-vrecon-skeleton+npz"),
        size=tuple(float(x) for x in np.maximum(hi - lo, 0.05)),
        rest=keys[0], keys=keys, status="estimated",
        evidence=[Evidence(note=f"GVHMR {d['meta'].get('commit', '')[:7]}: "
                                + "; ".join(d["meta"].get("assumptions", [])))],
    )


__all__ = ["character_from_gvhmr", "load_skeleton"]
