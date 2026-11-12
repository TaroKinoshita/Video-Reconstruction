"""Static background as one mesh from a 4D model's point maps (4RC). Pure numpy.

Per model pixel: the median of its aligned 3D points over the sampled frames where it is confident and not
covered by an excluded box (people, moving things: they move, so the background behind them shows in other
frames), and the median video colour there. Then a grid mesh over the model pixels (every `stride`-th), without
triangles across depth jumps (an edge longer than `jump` times the distance to the camera).

Only for a still camera: with a moving camera the same pixel sees different places, and only the first frame is
used (recorded in the report). Shapes behind what is never uncovered stay missing. Scale: the model's units.
"""

from __future__ import annotations

import numpy as np

from vrecon.analysis.from_4d import Model4D


def background_mesh(m: Model4D, A: np.ndarray, video: np.ndarray, exclude: dict[int, list[tuple[float, float, float, float]]],
                    stride: int = 2, conf_quantile: float = 0.0, jump: float = 0.03, still_tol: float = 0.01,
                    floor_tol: float | None = None):
    """A: model world -> aligned world (from_4d report "alignment"). video: uint8 RGB [T,H,W,3] (source frames).
    exclude: sample index -> boxes (x0, y0, x1, y1) in source pixels.
    conf_quantile: drop points below this quantile of the model's confidence. Default 0 (keep all): on the tennis
    clip the stands, far and low-confidence (mean 1.1 against 4.3), were the part cut, leaving holes.
    floor_tol: drop points closer than this to the floor plane (z = 0), so the floor stays one object (the floor
    plane) instead of fighting with a noisy copy of itself.

    Returns (verts [V,3], faces [F,3], colors [V,3] in 0..1, report)."""
    n, h, w = m.pts.shape[:3]
    centres = m.c2w[:, :3, 3]
    spread = float(np.linalg.norm(centres - centres.mean(0), axis=1).max())
    dist0 = float(np.median(np.linalg.norm(m.pts[0].reshape(-1, 3) - centres[0], axis=1)))
    still = spread < still_tol * dist0
    use = list(range(n)) if still else [0]
    u = np.rint(m.src_uv[..., 0]).astype(int).clip(0, video.shape[2] - 1)
    v = np.rint(m.src_uv[..., 1]).astype(int).clip(0, video.shape[1] - 1)
    pts = np.full((len(use), h, w, 3), np.nan)
    cols = np.full((len(use), h, w, 3), np.nan)
    thr = np.quantile(m.conf[use], conf_quantile)
    for k, i in enumerate(use):
        ok = m.conf[i] >= thr
        for x0, y0, x1, y1 in exclude.get(i, []):
            ok &= ~((m.src_uv[..., 0] >= x0) & (m.src_uv[..., 0] <= x1) & (m.src_uv[..., 1] >= y0) & (m.src_uv[..., 1] <= y1))
        p = m.pts[i] @ A[:3, :3].T + A[:3, 3]
        pts[k][ok] = p[ok]
        cols[k][ok] = video[int(m.frames[i])][v[ok], u[ok]] / 255.0
    import warnings

    with warnings.catch_warnings():  # pixels never seen: all-NaN medians are expected
        warnings.simplefilter("ignore", RuntimeWarning)
        P = np.nanmedian(pts, 0)
        C = np.nanmedian(cols, 0)
    if floor_tol is not None:
        P[np.abs(P[..., 2]) < floor_tol] = np.nan
    cam = (A @ m.c2w[0])[:3, 3]

    ys, xs = np.arange(0, h, stride), np.arange(0, w, stride)
    G = P[np.ix_(ys, xs)]
    GC = C[np.ix_(ys, xs)]
    gh, gw = G.shape[:2]
    valid = ~np.isnan(G[..., 0])
    idx = -np.ones((gh, gw), int)
    idx[valid] = np.arange(valid.sum())
    verts, colors = G[valid], np.nan_to_num(GC[valid], nan=0.5)
    faces = []
    for a, b, c in (((0, 0), (1, 0), (0, 1)), ((1, 0), (1, 1), (0, 1))):
        ia = idx[a[0]:gh - 1 + a[0], a[1]:gw - 1 + a[1]]
        ib = idx[b[0]:gh - 1 + b[0], b[1]:gw - 1 + b[1]]
        ic = idx[c[0]:gh - 1 + c[0], c[1]:gw - 1 + c[1]]
        ok = (ia >= 0) & (ib >= 0) & (ic >= 0)
        tri = np.stack([ia[ok], ib[ok], ic[ok]], 1)
        pa, pb, pc = verts[tri[:, 0]], verts[tri[:, 1]], verts[tri[:, 2]]
        longest = np.max([np.linalg.norm(pa - pb, axis=1), np.linalg.norm(pb - pc, axis=1),
                          np.linalg.norm(pc - pa, axis=1)], 0)
        d = np.linalg.norm((pa + pb + pc) / 3 - cam, axis=1)
        faces.append(tri[longest <= jump * d])
    faces = np.concatenate(faces)
    used = np.unique(faces)
    remap = -np.ones(len(verts), int)
    remap[used] = np.arange(len(used))
    report = {"conf_quantile": conf_quantile, "floor_tol": floor_tol, "still_camera": still, "frames_used": [int(m.frames[i]) for i in use], "camera_spread_rel": spread / dist0,
              "pixels_with_points": float(valid.mean()), "vertices": int(len(used)), "faces": int(len(faces))}
    return verts[used], remap[faces], colors[used], report


__all__ = ["background_mesh"]
