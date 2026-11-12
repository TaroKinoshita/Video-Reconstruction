"""Rule-based completion (plan step 4): fill what the video never showed with simple, named assumptions.

Every operation returns a new project and a record; what it adds or changes gets status "assumed" and an evidence
note "rule: <name> ...", so it can be told apart from what the video showed (scene/coverage.py) and swapped out.
Nothing here reads ground truth. Operations:

    extend_set_to_view   grow a floor (horizontal) or wall (vertical) plane until it fills a camera's view
    mirror               a copy of an object mirrored across a vertical plane (symmetric layouts: a court, a desk)
    repeat               copies of an object at a fixed offset (rows of seats, shelves)

A camera given to extend_set_to_view is the camera the user wants to render from (a new camera move), not ground
truth. Which operation fits a scene is a judgement (the LLM proposes, plan step 6); the operations themselves do
not decide that.
"""

from __future__ import annotations

import numpy as np

from vrecon.metrics.poses import _rot, camera_pose
from vrecon.schema import Camera, Evidence, Project, SceneObject, TransformKey


def _plane_axes(o: SceneObject):
    """(thin axis index, normal in world, other two axes) of a flat set piece."""
    s = np.array(o.size)
    thin = int(np.argmin(s))
    r = np.asarray(_rot(o.rest.pose.rotation))
    return thin, r[:, thin], [a for a in range(3) if a != thin], r


def _view_rays(cam: Camera, frame: int, samples: int = 9):
    r, c, k = camera_pose(cam, frame)
    w, h = cam.intrinsics.width, cam.intrinsics.height
    us, vs = np.meshgrid(np.linspace(0, w - 1, samples), np.linspace(0, h - 1, samples))
    d = np.stack([(us.ravel() - k[0, 2]) / k[0, 0], (vs.ravel() - k[1, 2]) / k[1, 1], np.ones(us.size)], 1)
    return c, d @ r.T


def extend_set_to_view(project: Project, obj_id: str, cam: Camera, frames: list[int] | None = None,
                       max_factor: float = 20.0, margin: float = 0.05, sideways_only: bool | None = None,
                       protect: Camera | None = None, protect_frames: list[int] | None = None) -> tuple[Project, dict]:
    """Grow a flat set piece in its plane so that it covers what `cam` sees of that plane (rays through a grid of
    the picture hitting the plane). The thin axis and the plane's position stay; the far side is capped at
    max_factor times the current size (a floor seen up to the horizon would grow without end).
    Standing pieces (normal more than 45 deg from vertical) do not grow below the floor (z = 0): grown down its
    slope, the tennis stands plane ran under the floor and in front of the camera and filled the whole view
    (2026-11-13). sideways_only: grow only along the plane's horizontal axis (off by default: on the synthetic
    clips it lost the wall's top in views from further back).
    protect: the source camera. Growth stops where it would show in that camera's picture (protect_frames): what
    the source saw is the observed reconstruction, and an assumed extension must not cover it (the tennis stands
    grown down to the floor hid the observed wall, 2026-11-13). Each of the four in-plane directions is cut back
    on its own (bisection on the new strip)."""
    p = project.model_copy(deep=True)
    o = next(x for x in p.reconstruction.objects if x.id == obj_id)
    thin, n, other, r = _plane_axes(o)
    centre = np.array(o.rest.pose.translation)
    size = np.array(o.size, float)
    n_frames = cam.keys[-1].shot_frame + 1
    frames = frames or sorted({0, n_frames // 2, n_frames - 1})
    pts = []
    for f in frames:
        c, d = _view_rays(cam, f)
        den = d @ n
        ok = np.abs(den) > 1e-9
        t = ((centre - c) @ n) / np.where(ok, den, 1.0)
        hit = c + t[:, None] * d
        pts.append(hit[ok & (t > 0)])
    standing = abs(n[2]) < np.cos(np.radians(45))
    pts = np.concatenate(pts) if pts else np.zeros((0, 3))
    if standing:
        pts = pts[pts[:, 2] >= 0.0]
    if len(pts) == 0:
        return p, {"op": "extend_set_to_view", "target": obj_id, "changed": False, "why": "the view does not see the plane"}
    local = (pts - centre) @ r  # in the object's axes
    lo = np.minimum(local.min(0), -size / 2)
    hi = np.maximum(local.max(0), size / 2)
    cap = max_factor * size / 2
    lo, hi = np.maximum(lo, -cap), np.minimum(hi, cap)
    lo[thin], hi[thin] = -size[thin] / 2, size[thin] / 2
    keep = max(other, key=lambda ax: abs(r[2, ax]))
    if sideways_only:
        keep = max(other, key=lambda ax: abs(r[2, ax]))  # the in-plane axis closest to vertical keeps its extent
        lo[keep], hi[keep] = -size[keep] / 2, size[keep] / 2
    span = (hi - lo) * (1 + margin)
    span[thin] = size[thin]
    mid = (hi + lo) / 2
    mid[thin] = 0.0
    cut = {}
    if protect is not None:
        # drop the view's hits that the source camera shows (outside the current piece): what the source saw is the
        # observed reconstruction. Then grow each in-plane axis only from hits beside the current piece (within its
        # extent on the other axis), so hits in one direction do not stretch the rectangle into a protected one.
        pf = protect_frames or sorted({k.shot_frame for k in protect.keys})
        pw, ph = protect.intrinsics.width, protect.intrinsics.height
        vis = np.zeros(len(pts), bool)
        for f in pf:
            rr, cc, kk = camera_pose(protect, f)
            q = (pts - cc) @ np.asarray(rr)
            z = np.where(q[:, 2] > 1e-9, q[:, 2], np.nan)
            uu, vv = kk[0, 0] * q[:, 0] / z + kk[0, 2], kk[1, 1] * q[:, 1] / z + kk[1, 2]
            vis |= (uu >= 0) & (uu < pw) & (vv >= 0) & (vv < ph)
        inside = np.all(np.abs(local) <= size / 2 + 1e-9, axis=1)
        free = ~vis | inside
        ax2 = [x for x in range(3) if x != thin]
        lo, hi = -size / 2, size / 2
        lo, hi = lo.copy(), hi.copy()
        for A, B in (ax2, ax2[::-1]):
            sel = free & (np.abs(local[:, B]) <= size[B] / 2)
            if sel.any():
                lo[A] = max(min(lo[A], local[sel, A].min()), -cap[A])
                hi[A] = min(max(hi[A], local[sel, A].max()), cap[A])
            cut["dropped_hits"] = int((vis & ~inside).sum())
        if sideways_only:
            lo[keep], hi[keep] = -size[keep] / 2, size[keep] / 2
        grow = (hi - lo) - size  # margin only on the grown part, so it does not creep into the source view
        span = size + grow * (1 + margin)
        span[thin] = size[thin]
        mid = (hi + lo) / 2
        mid[thin] = 0.0
    new_centre = centre + r @ mid
    o.size = tuple(float(x) for x in span)
    o.rest = o.rest.model_copy(update={"pose": o.rest.pose.model_copy(update={"translation": tuple(map(float, new_centre))})})
    o.status = "assumed"
    o.evidence = list(o.evidence) + [Evidence(note=f"rule: extend_set_to_view from camera {cam.id}, size {np.round(size, 4).tolist()} "
                                                   f"-> {np.round(span, 4).tolist()}")]
    return p, {"op": "extend_set_to_view", "target": obj_id, "changed": True, "size_before": size.tolist(),
               "size_after": span.tolist(), "protect_cut": cut, "capped": bool(np.any((np.abs(lo) >= cap - 1e-9) | (np.abs(hi) >= cap - 1e-9)))}


def _reflect_pose(t: np.ndarray, rmat: np.ndarray, point: np.ndarray, normal: np.ndarray):
    from vrecon.scene.convert import matrix_to_quat

    m = np.eye(3) - 2 * np.outer(normal, normal)
    t2 = point + m @ (t - point)
    # a reflected rotation is improper; flip the object's own axis along the mirror normal to keep it a rotation
    r2 = m @ rmat
    local_n = rmat.T @ normal
    axis = int(np.argmax(np.abs(local_n)))
    flip = np.eye(3)
    flip[axis, axis] = -1
    return t2, matrix_to_quat((r2 @ flip).tolist())


def mirror(project: Project, obj_id: str, point, normal, new_id: str | None = None) -> tuple[Project, dict]:
    """A mirrored copy of an object across the plane through `point` with `normal` (boxes are symmetric, so
    flipping one local axis keeps the copy a proper box)."""
    from vrecon.schema import Pose

    p = project.model_copy(deep=True)
    o = next(x for x in p.reconstruction.objects if x.id == obj_id)
    point, normal = np.asarray(point, float), np.asarray(normal, float) / np.linalg.norm(normal)

    def key(k: TransformKey) -> TransformKey:
        t, q = _reflect_pose(np.array(k.pose.translation), np.asarray(_rot(k.pose.rotation)), point, normal)
        return k.model_copy(update={"pose": Pose(translation=tuple(map(float, t)), rotation=q)})

    new = o.model_copy(deep=True, update={"id": new_id or f"{obj_id}_mirror", "name": f"{o.name or o.id} (mirror)",
                                          "rest": key(o.rest), "keys": [key(k) for k in o.keys], "status": "assumed",
                                          "evidence": [Evidence(note=f"rule: mirror of {obj_id} across plane through "
                                                                     f"{point.round(4).tolist()} normal {normal.round(4).tolist()}")]})
    p.reconstruction.objects.append(new)
    return p, {"op": "mirror", "target": obj_id, "new": new.id}


def repeat(project: Project, obj_id: str, offset, count: int) -> tuple[Project, dict]:
    """count copies of an object, the i-th moved by i * offset."""
    from vrecon.schema import Pose

    p = project.model_copy(deep=True)
    o = next(x for x in p.reconstruction.objects if x.id == obj_id)
    offset = np.asarray(offset, float)
    new_ids = []
    for i in range(1, count + 1):
        def key(k: TransformKey, i=i) -> TransformKey:
            return k.model_copy(update={"pose": Pose(translation=tuple(map(float, np.array(k.pose.translation) + i * offset)),
                                                     rotation=k.pose.rotation)})
        nid = f"{obj_id}_rep{i:02d}"
        p.reconstruction.objects.append(o.model_copy(deep=True, update={
            "id": nid, "name": f"{o.name or o.id} (repeat {i})", "rest": key(o.rest), "keys": [key(k) for k in o.keys],
            "status": "assumed", "evidence": [Evidence(note=f"rule: repeat of {obj_id}, offset {offset.round(4).tolist()} x{i}")]}))
        new_ids.append(nid)
    return p, {"op": "repeat", "target": obj_id, "new": new_ids}


__all__ = ["extend_set_to_view", "mirror", "repeat"]
