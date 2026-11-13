"""Put a GVHMR person (skeleton file) into a reconstructed scene (4RC cameras, floor at z = 0, model units).

Per frame and joint: the pixel where GVHMR sees the joint, a ray through the SCENE camera at that pixel, and a
point on it at GVHMR's depth (metres) times one scale s (scene units per metre). So the body lines up with the
image whatever the two cameras' focal lengths are (the depth is rescaled by f_scene / f_gvhmr: GVHMR's focal length
is a default from the image size, e.g. 1085 px against 4RC's 1872 px on the tennis clip, and a depth from the
wrong focal length makes the body 1.7x too large through the scene camera), and s is the one number that has to
come from the scene:

    s  chosen so that the feet stand on the floor: per frame, the lowest foot joint at its usual standing
       height (foot_height, from GVHMR's own world, where the floor is the lowest body vertex). Frames where the
       person is in the air give a larger s, so a low quantile of the per-frame values is used.

The rig keeps GVHMR's joint rotations (the body's shape is not bent to the rays); its root follows the pelvis
ray point, its orientation is the scene camera's rotation times GVHMR's in-camera orientation, and its scale is s.
Assumptions, recorded in the report: the person stands on the scene's floor at some point; GVHMR's depth is
right up to scale. Not checked against ground truth (no person clip with ground truth yet).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from vrecon.schema import Camera, Evidence, FileRef, Pose, SceneObject, TransformKey

FEET = ("left_ankle", "right_ankle", "left_foot", "right_foot")


def _rays(d: dict, camera: Camera, first_shot_frame: int):
    """Per frame: (R cam->world, centre, joint ray points in scene-camera coords at GVHMR depth [J,3])."""
    from vrecon.metrics.poses import camera_pose

    out = []
    for i in range(int(d["frames"])):
        r, c, k_s = camera_pose(camera, first_shot_frame + i)
        pc = d["joints_cam"][i]
        k_g = d["K"][i]
        uv = (pc[:, :2] / pc[:, 2:]) * [k_g[0, 0], k_g[1, 1]] + [k_g[0, 2], k_g[1, 2]]
        ray = np.c_[(uv - [k_s[0, 2], k_s[1, 2]]) / [k_s[0, 0], k_s[1, 1]], np.ones(len(uv))]
        # GVHMR's depth comes from the body's size in pixels and ITS focal length; through a camera with another
        # focal length the same pixel size means depth times f_scene / f_gvhmr
        f_ratio = (k_s[0, 0] + k_s[1, 1]) / (k_g[0, 0] + k_g[1, 1])
        out.append((r, c, ray * pc[:, 2:] * f_ratio))
    return out


def place_in_scene(path: str | Path, rel_path: str, camera: Camera, obj_id: str = "obj_person_01",
                   start_time: float = 0.0, first_shot_frame: int = 0, quantile: float = 0.25,
                   name: str = "", smooth_frames: float = 1.5) -> tuple[SceneObject, dict]:
    from vrecon.analysis.from_gvhmr import load_skeleton
    from vrecon.scene.convert import matrix_to_quat, quat_to_matrix

    d = load_skeleton(path)
    names = d["joint_names"]
    feet = [names.index(f) for f in FEET if f in names]
    # usual height of the lowest foot joint while standing, in GVHMR's world (metres)
    foot_h = float(np.quantile(d["joints_world"][:, feet, 2].min(1), 0.1))
    rays = _rays(d, camera, first_shot_frame)
    s_frame = []
    for r, c, q in rays:
        b = (q[feet] @ r.T)[:, 2].min()  # world z per unit of s, lowest foot
        s_frame.append(c[2] / (foot_h - b) if foot_h - b > 1e-9 else np.nan)
    s_frame = np.array(s_frame)
    ok = np.isfinite(s_frame) & (s_frame > 0)
    if not ok.any():
        raise ValueError("no frame puts the feet on the floor (camera below the floor or feet above the camera?)")
    s = float(np.quantile(s_frame[ok], quantile))
    fps = float(d["fps"])
    keys, joints = [], []
    for i, (r, c, q) in enumerate(rays):
        pw = c + (q * s) @ r.T
        joints.append(pw)
        m = r @ np.array(quat_to_matrix(tuple(float(x) for x in d["root_rot_cam"][i])))
        keys.append(TransformKey(time=start_time + i / fps, pose=Pose(translation=tuple(float(x) for x in pw[0]),
                                                                       rotation=tuple(matrix_to_quat(m.tolist()))),
                                 scale=(s, s, s)))
    joints = np.array(joints)
    # the root's path through the camera's rays shakes (GVHMR's per-frame depth and the camera's own small
    # jitter: up to 0.2 m between frames on tennis): a Gaussian over time (sigma 1.5 frames) on the root position
    # only; the joint rotations stay as GVHMR gave them
    tr = np.array([k.pose.translation for k in keys])
    g = np.exp(-0.5 * (np.arange(-4, 5) / smooth_frames) ** 2) if smooth_frames > 0 else np.array([1.0])
    g /= g.sum()
    pad = np.pad(tr, ((len(g) // 2, len(g) // 2), (0, 0)), mode="edge")
    sm = np.stack([np.convolve(pad[:, a], g, mode="valid") for a in range(3)], 1)
    keys = [k.model_copy(update={"pose": k.pose.model_copy(update={"translation": tuple(float(x) for x in sm[i])})})
            for i, k in enumerate(keys)]
    lowest = joints[:, feet, 2].min(1)
    lo, hi = joints.reshape(-1, 3).min(0), joints.reshape(-1, 3).max(0)
    report = {"scale_units_per_metre": s, "scale_per_frame_quantiles": np.quantile(s_frame[ok], [0.1, 0.25, 0.5, 0.75, 0.9]).tolist(),
              "foot_height_m": foot_h, "lowest_foot_minus_standing_height_units": np.quantile(lowest - foot_h * s, [0.1, 0.5, 0.9]).tolist(),
              "body_height_units_median": float(np.median(joints[:, names.index("head"), 2] - lowest)) if "head" in names else None,
              "assumptions": ["person stands on the scene floor at some point (low quantile of per-frame scales)",
                              "GVHMR depth right up to scale", "feet joints usual standing height from GVHMR's world"]}
    obj = SceneObject(
        id=obj_id, name=name or obj_id, kind="character",
        asset=FileRef(id=f"file_{obj_id[4:]}_skeleton", path=rel_path, media_type="application/x-vrecon-skeleton+npz"),
        size=tuple(float(x) for x in np.maximum(hi - lo, 1e-3)),
        rest=TransformKey(time=keys[0].time, pose=keys[0].pose, scale=(s, s, s)), keys=keys, status="estimated",
        evidence=[Evidence(note=f"GVHMR {d['meta'].get('commit', '')[:7]} placed through the scene camera; scale {s:.4g} "
                                "units/m from feet on the floor")])
    return obj, report


__all__ = ["FEET", "place_in_scene"]
