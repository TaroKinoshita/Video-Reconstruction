"""Object paths from mask centres: at each key, the object's centre is put where the ray through its mask's
centroid meets the horizontal plane at half the object's height.

Uses only the camera, the floor (z = 0) and the masks, not the start scene's per-frame boxes, which can
under-move (4RC start: travel 1.0 against 4.0 on syn_03). Frames where the mask is empty or touches the image
border keep their key.
"""

from __future__ import annotations

import numpy as np

from vrecon.metrics.poses import camera_pose
from vrecon.run.manager import Change, Proposal
from vrecon.scene.convert import scene_time_to_shot_frame
from vrecon.schema import Pose, Project


def _centroid(mask: np.ndarray):
    if not mask.any() or mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any():
        return None
    v, u = np.nonzero(mask)
    return float(u.mean()), float(v.mean())


def ray_keys(project: Project, obs, pairs: dict[str, str], oid: str, shot_id: str = "shot_001"):
    """New keys for an animated object (or None when it has no keys or no usable mask)."""
    obj = next(o for o in project.reconstruction.objects if o.id == oid)
    if not obj.keys:
        return None
    nums = [n for n, label in obs.id_map.items() if pairs.get(label) == oid]
    if not nums:
        return None
    cam = next(c for c in project.reconstruction.cameras if c.shot_id == shot_id)
    mapping = next((m for m in project.reconstruction.shot_mapping if m.shot_id == shot_id), None)
    start = mapping.scene_time_start if mapping else 0.0
    half = obj.size[2] * obj.rest.scale[2] / 2
    out, moved = [], 0
    for k in obj.keys:
        f = int(round(scene_time_to_shot_frame(k.time, start, project.source.fps)))
        c = None
        if 0 <= f < len(obs.masks):
            c = _centroid(np.isin(obs.masks[f], nums))
        if c is None:
            out.append(k)
            continue
        r, centre, kk = camera_pose(cam, f)
        d = r @ np.array([(c[0] - kk[0, 2]) / kk[0, 0], (c[1] - kk[1, 2]) / kk[1, 1], 1.0])
        if abs(d[2]) < 1e-9 or (half - centre[2]) / d[2] <= 0:
            out.append(k)
            continue
        p = centre + (half - centre[2]) / d[2] * d
        out.append(k.model_copy(update={"pose": Pose(translation=(float(p[0]), float(p[1]), k.pose.translation[2]),
                                                     rotation=k.pose.rotation)}))
        moved += 1
    return out if moved else None


def ray_proposal(project: Project, obs, pairs: dict[str, str], oid: str, proposal_id: str):
    keys = ray_keys(project, obs, pairs, oid)
    if keys is None:
        return None
    return Proposal(id=proposal_id, hypothesis=f"{oid} path from mask centres (rays to half height)",
                    changes=[Change(target_id=oid, field="keys", value=keys), Change(target_id=oid, field="rest", value=keys[0])],
                    source="optimizer")
