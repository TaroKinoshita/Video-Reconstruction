"""Run GVHMR on one video and write a skeleton file in vrecon's conventions. Runs in WSL, gvhmr env:

    cd ~/vrecon/GVHMR && ~/vrecon/bin/micromamba run -r ~/vrecon/mamba -n gvhmr \
        python /mnt/c/.../gpu/observe/run_gvhmr.py VIDEO OUT_DIR [--moving-camera]

OUT_DIR/gvhmr_skeleton.npz (all lengths in metres, world Z up, frame i = video frame i):
    fps, frames, width, height
    joint_names [22], parents [22] (-1 = root)              SMPL-X body joints (pelvis ... wrists)
    rest_joints [22,3]     joints of this person's body (betas) in the rest pose, pelvis at the origin
    local_rot [T,22,4]     (w,x,y,z) rotation of each joint relative to its parent, in the rest frame's axes
                           (row 0 = the whole body's orientation in the world)
    root_pos [T,3]         world position of the pelvis
    joints_world [T,22,3]  world joint positions (from the body model, for checking)
    joints_cam [T,22,3]    joint positions in the camera frame (OpenCV axes: x right, y down, z forward)
    root_rot_cam [T,4]     (w,x,y,z) body orientation in the camera frame, acting on the Z-up rest skeleton
                           (camera-frame placement: world rot = R_cam_to_world @ root_rot_cam)
    K [T,3,3]              intrinsics GVHMR used (estimated, not measured)
    floor_z_before         height of the lowest body vertex over the clip, moved to z = 0
    meta (json)            tool, commit, assumptions

GVHMR's world is y-up and gravity-aligned; it is rotated here to Z up (x, y, z) -> (x, -z, y), every rotation is
conjugated by the same turn, and the floor is put at the lowest body vertex over the clip (assumption: the person
stands on the floor at some point). Scale comes from the body model (an average body for the estimated shape).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

A = np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])  # y-up -> z-up
JOINTS = ["pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee", "spine2", "left_ankle",
          "right_ankle", "spine3", "left_foot", "right_foot", "neck", "left_collar", "right_collar", "head",
          "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist"]
PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]


def aa_to_mat(aa: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    return Rotation.from_rotvec(aa.reshape(-1, 3)).as_matrix().reshape(*aa.shape[:-1], 3, 3)


def mat_to_quat_wxyz(m: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    q = Rotation.from_matrix(m.reshape(-1, 3, 3)).as_quat()  # x y z w
    q = np.concatenate([q[:, 3:], q[:, :3]], 1).reshape(*m.shape[:-2], 4)
    # neighbouring keys on the same side (q and -q are the same turn but interpolate the long way)
    for t in range(1, q.shape[0]):
        flip = (q[t] * q[t - 1]).sum(-1) < 0
        q[t, flip] *= -1
    return q


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("out")
    ap.add_argument("--moving-camera", action="store_true", help="estimate camera motion (default: static, -s)")
    a = ap.parse_args()
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    root = Path.cwd()  # GVHMR checkout
    cmd = [sys.executable, "tools/demo/demo.py", f"--video={a.video}", f"--output_root={out}"]
    if not a.moving_camera:
        cmd.append("-s")
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    name = Path(a.video).stem
    res_path = out / name / "hmr4d_results.pt"
    if not res_path.exists():  # the demo's final rendering may fail (pytorch3d without GPU); results come first
        print(proc.stdout[-3000:], proc.stderr[-3000:])
        return 1
    pred = torch.load(res_path, map_location="cpu")

    sys.path.insert(0, str(root))
    from hmr4d.utils.smplx_utils import make_smplx

    bm = make_smplx("supermotion")
    g = {k: v.float() for k, v in pred["smpl_params_global"].items()}
    c = {k: v.float() for k, v in pred["smpl_params_incam"].items()}
    with torch.no_grad():
        og = bm(**g)
        oc = bm(**c)
        zero = {k: torch.zeros_like(v[:1]) for k, v in g.items()}
        zero["betas"] = g["betas"][:1]
        orest = bm(**zero)
    jw = og.joints[:, :22].numpy() if og.joints.dim() == 3 else og.joints[0, :, :22].numpy()
    jc = oc.joints[:, :22].numpy() if oc.joints.dim() == 3 else oc.joints[0, :, :22].numpy()
    vw = og.vertices.numpy() if og.vertices.dim() == 3 else og.vertices[0].numpy()
    rest = (orest.joints[:, :22] if orest.joints.dim() == 3 else orest.joints[0, :, :22])[0].numpy()
    rest = rest - rest[0]

    floor = float(vw[..., 1].min())
    to_z = lambda p: p @ A.T  # noqa: E731
    jw_z = to_z(jw - np.array([0, floor, 0]))
    rest_z = to_z(rest)
    rots = np.concatenate([g["global_orient"].numpy()[:, None], g["body_pose"].numpy().reshape(-1, 21, 3)], 1)
    mats = A @ aa_to_mat(rots) @ A.T  # same turns, seen in the Z-up axes
    t_n = len(jw)

    import cv2

    cap = cv2.VideoCapture(a.video)
    fps, w, h = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    meta = {"tool": "GVHMR", "commit": commit, "static_camera": not a.moving_camera, "video": a.video,
            "assumptions": ["floor = lowest body vertex over the clip", "scale from the body model (estimated)",
                            "intrinsics estimated by GVHMR"], "frames_match_video": True}
    np.savez_compressed(out / "gvhmr_skeleton.npz", fps=fps, frames=t_n, width=w, height=h,
                        joint_names=np.array(JOINTS), parents=np.array(PARENTS), rest_joints=rest_z,
                        local_rot=mat_to_quat_wxyz(mats), root_pos=jw_z[:, 0],
                        root_rot_cam=mat_to_quat_wxyz(aa_to_mat(c["global_orient"].numpy()) @ A.T), joints_world=jw_z, joints_cam=jc,
                        K=pred["K_fullimg"].numpy(), floor_z_before=floor, meta=json.dumps(meta))
    print(json.dumps({"out": str(out / "gvhmr_skeleton.npz"), "frames": t_n, "fps": fps,
                      "height_m": float(jw_z[..., 2].max() - jw_z[..., 2].min())}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
