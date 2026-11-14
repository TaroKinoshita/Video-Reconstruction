"""Rule-based completion (plan step 4): fill what the video never showed with simple, named assumptions.

Every operation returns a new project and a record; what it adds or changes gets status "hypothesis" and an evidence
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


def covers_source(points: np.ndarray, protect: Camera, frames: list[int], depth: dict | None,
                  depth_tol: float = 0.005, empty_is_seen: bool = False) -> np.ndarray:
    """Per world point: would it show in front of what the source camera's depth shows (depth[f]; 0 = nothing
    built there, which counts as covered only with empty_is_seen)? Without depth, any point in the picture counts.
    empty_is_seen is for new pieces (enclose copies): an empty pixel of the source view still showed something
    (often a masked thing cut out of the background), and a copy standing there would hide what grows into it."""
    pw, ph = protect.intrinsics.width, protect.intrinsics.height
    bad = np.zeros(len(points), bool)
    for f in frames:
        rr, cc, kk = camera_pose(protect, f)
        q = (points - cc) @ np.asarray(rr)
        z = np.where(q[:, 2] > 1e-9, q[:, 2], np.nan)
        uu, vv = kk[0, 0] * q[:, 0] / z + kk[0, 2], kk[1, 1] * q[:, 1] / z + kk[1, 2]
        inf = (uu >= 0) & (uu < pw) & (vv >= 0) & (vv < ph)
        if depth is None:
            bad |= inf
            continue
        dm = depth[f]
        ui = np.clip(np.nan_to_num(uu * dm.shape[1] / pw).astype(int), 0, dm.shape[1] - 1)
        vi = np.clip(np.nan_to_num(vv * dm.shape[0] / ph).astype(int), 0, dm.shape[0] - 1)
        d = dm[vi, ui].astype(float)
        bad |= inf & (((d > 0) & (z < d * (1 - depth_tol))) | ((d <= 0) & empty_is_seen))
    return bad


def box_face_points(o: SceneObject, n: int = 24) -> np.ndarray:
    """A grid of points on a flat piece's plane (its two long axes), in world coordinates."""
    r = np.asarray(_rot(o.rest.pose.rotation))
    size = np.array(o.size, float)
    thin = int(np.argmin(size))
    ax = [a for a in range(3) if a != thin]
    g = np.linspace(-0.5, 0.5, n)
    u, v = np.meshgrid(g, g)
    loc = np.zeros((u.size, 3))
    loc[:, ax[0]] = u.ravel() * size[ax[0]]
    loc[:, ax[1]] = v.ravel() * size[ax[1]]
    return np.array(o.rest.pose.translation) + loc @ r.T


def extend_set_to_view(project: Project, obj_id: str, cam: Camera, frames: list[int] | None = None,
                       max_factor: float = 20.0, margin: float = 0.05, sideways_only: bool | None = None,
                       protect: Camera | None = None, protect_frames: list[int] | None = None,
                       protect_depth: dict | None = None, depth_tol: float = 0.005,
                       empty_is_seen: bool = False) -> tuple[Project, dict]:
    """Grow a flat set piece in its plane so that it covers what `cam` sees of that plane (rays through a grid of
    the picture hitting the plane). The thin axis and the plane's position stay; the far side is capped at
    max_factor times the current size (a floor seen up to the horizon would grow without end).
    Standing pieces (normal more than 45 deg from vertical) do not grow below the floor (z = 0): grown down its
    slope, the tennis stands plane ran under the floor and in front of the camera and filled the whole view
    (2026-11-13). sideways_only: grow only along the plane's horizontal axis (off by default: on the synthetic
    clips it lost the wall's top in views from further back).
    protect: the source camera. Growth stops where it would show in that camera's picture (protect_frames): what
    the source saw is the observed reconstruction, and an assumed extension must not cover it (the tennis stands
    grown down to the floor hid the observed wall, 2026-11-13).
    protect_depth: {frame: view-z depth map of the scene before completion, from `protect` (0 = nothing)}. With it
    a point only counts as covering the source view if it lies in front of what that pixel shows (by depth_tol);
    where nothing was built (0) it covers nothing (masked things are cut out of the background, e.g. beside the
    stands); without it, any point in the picture does. Hits that would cover are dropped,
    each in-plane axis grows from the hits beside the piece, and then each of the four new strips is cut back
    (bisection) until no point of it covers the source view: wall pieces grown sideways across a visible gap
    stood in front of the stands and other walls (2026-11-13)."""
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
        pf = protect_frames or sorted({k.shot_frame for k in protect.keys})

        def covers(w_: np.ndarray) -> np.ndarray:
            return covers_source(w_, protect, pf, protect_depth, depth_tol, empty_is_seen)

        inside = np.all(np.abs(local) <= size / 2 + 1e-9, axis=1)
        vis = covers(pts)
        free = ~vis | inside
        ax2 = [x for x in range(3) if x != thin]
        lo, hi = (-size / 2).copy(), (size / 2).copy()
        for A, B in (ax2, ax2[::-1]):
            sel = free & (np.abs(local[:, B]) <= size[B] / 2)
            if sel.any():
                lo[A] = max(min(lo[A], local[sel, A].min()), -cap[A])
                hi[A] = min(max(hi[A], local[sel, A].max()), cap[A])
        cut["dropped_hits"] = int((vis & ~inside).sum())
        if sideways_only:
            lo[keep], hi[keep] = -size[keep] / 2, size[keep] / 2
        # exact check in the source picture: rays through a pixel grid hit the plane; those hits that would cover
        # (in front of the source depth) and lie outside the old piece must stay outside the grown one. Cut the
        # rectangle back along the axis where such a hit sticks out furthest, until none is inside. (A sampled
        # check on the grown rectangle missed thin overlaps once pieces grew hundreds of metres.)
        pw, ph = protect.intrinsics.width, protect.intrinsics.height
        us, vs = np.meshgrid(np.arange(0, pw, 6) + 0.5, np.arange(0, ph, 6) + 0.5)
        bad_loc = []
        for f in pf:
            rr, cc, kk = camera_pose(protect, f)
            dd = np.stack([(us.ravel() - kk[0, 2]) / kk[0, 0], (vs.ravel() - kk[1, 2]) / kk[1, 1], np.ones(us.size)], 1) @ np.asarray(rr).T
            den = dd @ n
            okk = np.abs(den) > 1e-12
            tt = ((centre - cc) @ n) / np.where(okk, den, 1.0)
            hit = cc + tt[:, None] * dd
            sel = okk & (tt > 0)
            if sel.any():
                bb = covers_source(hit[sel], protect, [f], protect_depth, depth_tol, empty_is_seen)
                bad_loc.append((hit[sel][bb] - centre) @ r)
        bad_loc = np.concatenate(bad_loc) if bad_loc else np.zeros((0, 3))
        bad_loc = bad_loc[~np.all(np.abs(bad_loc) <= size / 2 + 1e-9, axis=1)]
        cut["covering_hits"] = int(len(bad_loc))
        for _ in range(256):
            inside_new = np.all((bad_loc >= lo - 1e-12) & (bad_loc <= hi + 1e-12), axis=1)
            if not inside_new.any():
                break
            q = bad_loc[inside_new]
            best = None
            for A in ax2:
                for side in (0, 1):
                    edge = size[A] / 2
                    if (hi[A] if side else -lo[A]) <= edge + 1e-12:
                        continue  # not grown on this side: nothing to cut
                    out = (q[:, A] - edge) if side else (-edge - q[:, A])
                    if (out > 0).any():
                        # cut just inside the nearest offending hit beyond the old edge on this side
                        new = (q[out > 0, A].min() if side else q[out > 0, A].max())
                        loss = abs((hi[A] if side else lo[A]) - new)
                        cand = (loss, A, side, new)
                        best = cand if best is None or cand[0] < best[0] else best
            if best is None:
                break
            _, A, side, new = best
            if side:
                hi[A] = max(new - 1e-9, size[A] / 2)
            else:
                lo[A] = min(new + 1e-9, -size[A] / 2)
        grow = (hi - lo) - size  # margin only on the grown part, so it does not creep into the source view
        span = size + grow * (1 + margin)
        span[thin] = size[thin]
        mid = (hi + lo) / 2
        mid[thin] = 0.0
    new_centre = centre + r @ mid
    o.size = tuple(float(x) for x in span)
    o.rest = o.rest.model_copy(update={"pose": o.rest.pose.model_copy(update={"translation": tuple(map(float, new_centre))})})
    o.status = "hypothesis"
    o.evidence = list(o.evidence) + [Evidence(kind="rule", note=f"rule: extend_set_to_view from camera {cam.id}, size {np.round(size, 4).tolist()} "
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
                                          "rest": key(o.rest), "keys": [key(k) for k in o.keys], "status": "hypothesis",
                                          "evidence": [Evidence(kind="rule", note=f"rule: mirror of {obj_id} across plane through "
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
            "status": "hypothesis", "evidence": [Evidence(kind="rule", note=f"rule: repeat of {obj_id}, offset {offset.round(4).tolist()} x{i}")]}))
        new_ids.append(nid)
    return p, {"op": "repeat", "target": obj_id, "new": new_ids}


def enclose(project: Project, centre, ids: list[str], angles=(90.0, 180.0, 270.0), contain=None,
            margin: float = 0.0) -> tuple[Project, dict]:
    """Build the sides the camera never faced: copies of the boundary pieces (stands, back walls) turned about the
    vertical axis through `centre` (rule "the surroundings go on around": a stadium ring, the walls of a room).
    The copies keep their distance from the centre, but are pushed outwards (along the direction from the centre)
    until every point of `contain` (the cameras, the people and things) is inside by `margin`: a copy turned
    180 deg about the middle of the things otherwise stood between the camera and the player and filled the view
    (tennis, 2026-11-13). The copies are status "hypothesis" with a "rule: enclose" note."""
    from vrecon.scene.convert import matrix_to_quat
    from vrecon.schema import Pose

    p = project.model_copy(deep=True)
    centre = np.asarray(centre, float)
    new = []
    for oid in ids:
        o = next(x for x in p.reconstruction.objects if x.id == oid)
        for ang in angles:
            a = np.radians(ang)
            rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1.0]])
            t = rz @ (np.array(o.rest.pose.translation) - centre) + centre
            u = t[:2] - centre[:2]
            u = np.r_[u / np.linalg.norm(u), 0.0]
            if contain is not None:
                need = max(float((np.asarray(x, float) - centre) @ u) for x in contain) + margin
                now = float((t - centre) @ u) - 0.5 * float(np.min(o.size[:2]))
                if now < need:
                    t = t + u * (need - now)
            q = matrix_to_quat((rz @ np.asarray(_rot(o.rest.pose.rotation))).tolist())
            nid = f"{oid}_encl{int(ang):03d}"
            p.reconstruction.objects.append(o.model_copy(deep=True, update={
                "id": nid, "name": f"{o.name or o.id} (around {int(ang)})", "keys": [], "status": "hypothesis",
                "rest": o.rest.model_copy(update={"pose": Pose(translation=tuple(map(float, t)), rotation=q)}),
                "evidence": [Evidence(kind="rule", note=f"rule: enclose, copy of {oid} turned {ang:g} deg about "
                                           f"{np.round(centre, 4).tolist()}")]}))
            new.append(nid)
    return p, {"op": "enclose", "centre": centre.tolist(), "sources": ids, "new": new}


__all__ = ["box_face_points", "covers_source", "enclose", "extend_set_to_view", "mirror", "repeat"]
