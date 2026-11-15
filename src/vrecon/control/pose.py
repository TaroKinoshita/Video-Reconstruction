"""OpenPose-style pose control from the scene's rig (runbook 10.3; Fun ControlNet Union lists Pose as a DWPose
skeleton). No bpy: joints from analysis/rig_fk.posed_joints, projected through the scene camera.

Joint mapping, SMPL 22 (GVHMR) -> OpenPose COCO-18 (index: source):
    0 nose: head (moved 0.1 m forward along the body's facing is not done; the head joint is used as is)
    1 neck: neck          2 r_shoulder: right_shoulder  3 r_elbow: right_elbow  4 r_wrist: right_wrist
    5 l_shoulder: left_shoulder  6 l_elbow: left_elbow  7 l_wrist: left_wrist
    8 r_hip: right_hip    9 r_knee: right_knee  10 r_ankle: right_ankle
    11 l_hip: left_hip   12 l_knee: left_knee  13 l_ankle: left_ankle
    14-17 eyes, ears: not in the rig, not drawn
Drawing follows the OpenPose body colours and limb order (limbs as ellipses, joints as dots) on black.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

COCO_FROM_SMPL = {0: "head", 1: "neck", 2: "right_shoulder", 3: "right_elbow", 4: "right_wrist", 5: "left_shoulder",
                  6: "left_elbow", 7: "left_wrist", 8: "right_hip", 9: "right_knee", 10: "right_ankle",
                  11: "left_hip", 12: "left_knee", 13: "left_ankle"}
LIMBS = [(1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13), (1, 0)]
COLORS = [(255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0), (85, 255, 0), (0, 255, 0),
          (0, 255, 85), (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255), (0, 0, 255), (85, 0, 255),
          (170, 0, 255), (255, 0, 255), (255, 0, 170), (255, 0, 85)]


def character_joints(project, root: str | Path | None, times: np.ndarray) -> list[np.ndarray]:
    """Per character with a skeleton file: COCO-18 world joints [T, 18, 3] at `times` (NaN where missing)."""
    from vrecon.analysis.from_gvhmr import load_skeleton
    from vrecon.analysis.rig_fk import posed_joints

    out = []
    for o in project.reconstruction.objects:
        if o.kind != "character" or o.asset is None or not o.keys:
            continue
        path = Path(o.asset.path)
        if not path.is_absolute():
            path = Path(root or ".") / path
        if not path.is_file():
            continue
        d = load_skeleton(path)
        j = posed_joints(o, d)  # [K, 22, 3] at the keys
        kt = np.array([k.time for k in o.keys])
        names = d["joint_names"]
        coco = np.full((len(times), 18, 3), np.nan)
        for ci, name in COCO_FROM_SMPL.items():
            if name in names:
                ji = names.index(name)
                for a in range(3):
                    coco[:, ci, a] = np.interp(times, kt, j[:, ji, a], left=np.nan, right=np.nan)
        out.append(coco)
    return out


def draw_pose(frame_shape: tuple[int, int], pts: list[np.ndarray], stick: int = 4) -> np.ndarray:
    """pts: per person [18, 2] pixels (NaN = missing). uint8 RGB [H, W, 3]."""
    import cv2

    h, w = frame_shape
    canvas = np.zeros((h, w, 3), np.uint8)
    for p in pts:
        for i, (a, b) in enumerate(LIMBS):
            if not (np.isfinite(p[a]).all() and np.isfinite(p[b]).all()):
                continue
            (x0, y0), (x1, y1) = p[a], p[b]
            mx, my = (x0 + x1) / 2, (y0 + y1) / 2
            length = float(np.hypot(x1 - x0, y1 - y0)) / 2
            angle = float(np.degrees(np.arctan2(y1 - y0, x1 - x0)))
            poly = cv2.ellipse2Poly((int(mx), int(my)), (int(length), stick), int(angle), 0, 360, 1)
            layer = canvas.copy()
            cv2.fillConvexPoly(layer, poly, COLORS[i])
            canvas = cv2.addWeighted(canvas, 0.4, layer, 0.6, 0)
        for i in range(18):
            if np.isfinite(p[i]).all():
                cv2.circle(canvas, (int(p[i][0]), int(p[i][1])), 4, COLORS[i], -1)
    return canvas


def pose_control(project, shot_id: str | None, n_frames: int, size: tuple[int, int],
                 root: str | Path | None = None) -> tuple[np.ndarray, dict]:
    """uint8 RGB [T, H, W, 3]: every character's skeleton through the shot camera, frame f at frame_time(f)."""
    from vrecon.metrics.poses import camera_pose, project_points
    from vrecon.schema import frame_time

    shot_id = shot_id or project.shots[0].id
    cam = next(c for c in project.reconstruction.cameras if c.shot_id == shot_id)
    times = np.array([frame_time(project, f, shot_id) for f in range(n_frames)])
    people = character_joints(project, root, times)
    w, h = size
    frames = np.zeros((n_frames, h, w, 3), np.uint8)
    for f in range(n_frames):
        r, c, k = camera_pose(cam, f)
        pts = []
        for coco in people:
            uv, z = project_points(np.nan_to_num(coco[f]), r, c, k)
            uv[(~np.isfinite(coco[f]).all(1)) | (z <= 0)] = np.nan
            pts.append(uv)
        frames[f] = draw_pose((h, w), pts)
    return frames, {"source": "rig joints (analysis/rig_fk) -> OpenPose COCO-18", "people": len(people),
                    "mapping": {str(k): v for k, v in COCO_FROM_SMPL.items()}}


__all__ = ["COCO_FROM_SMPL", "draw_pose", "pose_control"]
