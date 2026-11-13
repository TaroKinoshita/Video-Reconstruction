"""Move a reconstruction into the calibrated court frame (metres, origin at the court centre, X across, Y along,
Z up; analysis/court.py). Pure numpy.

The reconstruction's world (4RC units, floor z = 0) and the court frame are related by a similarity found from
the cameras: the rotation that best maps the reconstruction's camera orientations onto the calibrated ones over
all calibrated frames (SVD), the scale from the camera heights over the floor (both floors at z = 0), the
translation from the camera centres. The tilt between the two floors is reported (it should be under a degree).
The court calibration then replaces the camera (per-frame rotation, one centre, its focal length).
"""

from __future__ import annotations

import numpy as np

from vrecon.metrics.poses import _rot, camera_pose
from vrecon.scene.convert import matrix_to_quat
from vrecon.schema import Camera, CameraKey, Evidence, Intrinsics, Pose, Project


def court_cameras(calibration: dict, camera_id: str, shot_id: str) -> Camera:
    w, h = calibration["width"], calibration["height"]
    f = calibration["focal"]["median_px"] if not calibration["focal"]["zoom"] else None
    keys = []
    for x in calibration["frames"]:
        R = np.asarray(x["R"], float)
        t = np.asarray(x["t"], float)
        intr = None
        if f is None:
            fx = x["focal_px"]
            intr = Intrinsics(width=w, height=h, fx=fx, fy=fx, cx=(w - 1) / 2, cy=(h - 1) / 2)
        keys.append(CameraKey(shot_frame=int(x["frame"]), intrinsics=intr,
                              pose=Pose(translation=tuple(map(float, -R.T @ t)), rotation=matrix_to_quat(R.T.tolist()))))
    f0 = f or calibration["frames"][0]["focal_px"]
    return Camera(id=camera_id, shot_id=shot_id, status="estimated",
                  intrinsics=Intrinsics(width=w, height=h, fx=f0, fy=f0, cx=(w - 1) / 2, cy=(h - 1) / 2), keys=keys,
                  evidence=[Evidence(note="court lines matched to the ITF court (analysis/court.py, scripts/calibrate_court.py)")])


def similarity_to_court(project: Project, calibration: dict):
    """(A rotation, k scale = metres per unit, T translation, report) with x_court = k A x + T."""
    cam = project.reconstruction.cameras[0]
    S = np.zeros((3, 3))
    cs, cc = [], []
    for x in calibration["frames"]:
        R = np.asarray(x["R"], float)
        t = np.asarray(x["t"], float)
        Rs, c, _ = camera_pose(cam, x["frame"])
        S += R.T @ np.asarray(Rs).T
        cs.append(np.asarray(c, float))
        cc.append(-R.T @ t)
    U, _, Vt = np.linalg.svd(S)
    A = U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt
    cs, cc = np.array(cs), np.array(cc)
    k = float(cc[:, 2].mean() / cs[:, 2].mean())
    T = (cc - k * cs @ A.T).mean(0)
    rep = {"floor_tilt_deg": float(np.degrees(np.arccos(np.clip(A[2, 2], -1, 1)))),
           "yaw_deg": float(np.degrees(np.arctan2(A[1, 0], A[0, 0]))), "metres_per_unit": k, "units_per_metre": 1 / k,
           "translation_m": T.tolist(),
           "camera_centre_residual_m": float(np.linalg.norm(cc - (k * cs @ A.T + T), axis=1).max())}
    return A, k, T, rep


def transform_project(project: Project, A: np.ndarray, k: float, T: np.ndarray, camera: Camera | None = None) -> Project:
    """The project in the court frame: poses moved (x -> k A x + T, rotations A R), box sizes and mesh scales
    multiplied by k, characters' scale (units per metre) multiplied by k; the camera replaced if given."""
    p = project.model_copy(deep=True)

    def pose(ps: Pose) -> Pose:
        t = k * A @ np.asarray(ps.translation, float) + T
        q = matrix_to_quat((A @ np.asarray(_rot(ps.rotation))).tolist())
        return Pose(translation=tuple(map(float, t)), rotation=q)

    for o in p.reconstruction.objects:
        if o.kind == "character":
            sc = tuple(v * k for v in o.rest.scale)
            o.rest = o.rest.model_copy(update={"pose": pose(o.rest.pose), "scale": sc})
            o.keys = [kk.model_copy(update={"pose": pose(kk.pose), "scale": sc}) for kk in o.keys]
            continue
        mesh = o.asset is not None and o.asset.path.endswith(".npz")
        upd = {"pose": pose(o.rest.pose)}
        if mesh:
            upd["scale"] = tuple(v * k for v in o.rest.scale)
        o.rest = o.rest.model_copy(update=upd)
        o.keys = [kk.model_copy(update={"pose": pose(kk.pose), **({"scale": tuple(v * k for v in kk.scale)} if mesh else {})})
                  for kk in o.keys]
        o.size = tuple(v * k for v in o.size)
    if camera is not None:
        p.reconstruction.cameras = [camera]
    p.reconstruction.scale = p.reconstruction.scale.model_copy(update={
        "source": "estimated", "note": "metres from the court lines (ITF court dimensions); court frame: origin at the "
                                      "court centre, X across, Y along, Z up"})
    return p




def warp_model(m, A4: np.ndarray, k: float, camera: Camera):
    """The 4RC model seen through the calibrated camera: each point keeps its source pixel and its distance from
    the camera (times k: metres), and moves onto the calibrated camera's ray through that pixel. A plain
    similarity keeps 4RC's own field of view (focal 1872 px against the court's 2104 px on tennis), so things
    placed from it drift off their pixels towards the picture's edges; this keeps them on their pixels.
    A4: 4RC world -> the reconstruction's world (floor z = 0, from_4d report "alignment"). Returns a Model4D in
    the court frame (alignment = identity)."""
    from dataclasses import replace

    pts = np.empty_like(m.pts)
    c2w = np.empty_like(m.c2w)
    Ks = np.empty_like(m.K)
    u = m.src_uv[..., 0].ravel()
    v = m.src_uv[..., 1].ravel()
    for i, f in enumerate(m.frames):
        P = m.pts[i].reshape(-1, 3) @ A4[:3, :3].T + A4[:3, 3]
        Cs = (A4 @ m.c2w[i])[:3, 3]
        d = np.linalg.norm(P - Cs, axis=1)
        R, C, K = camera_pose(camera, int(f))
        R = np.asarray(R, float)
        ray = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], 1) @ R.T
        ray /= np.linalg.norm(ray, axis=1, keepdims=True)
        pts[i] = (np.asarray(C) + (k * d)[:, None] * ray).reshape(m.pts[i].shape)
        c2w[i] = np.eye(4)
        c2w[i][:3, :3], c2w[i][:3, 3] = R, C
        Ks[i] = K
    return replace(m, pts=pts, c2w=c2w, K=Ks)


__all__ = ["court_cameras", "similarity_to_court", "transform_project", "warp_model"]
