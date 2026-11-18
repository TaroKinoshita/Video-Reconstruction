"""Object shapes for the silhouette objective. Pure numpy.

Every object is drawn from one description: unit vertices U [V,3] and faces (polygons as vertex index lists),
with object-local points = U * size (per axis). A proxy box is U = cube corners in [-0.5, 0.5]^3 and its six
quads. An object with a mesh file (SceneObject.asset, the builder's .npz: verts [V,3], faces [F,3]) is drawn
from that mesh: U = verts / extents, so its `size` still scales it and the search's size parameters act on
the real shape. The mesh keeps its own origin (it need not be centred).

Large meshes (a body, a scanned prop) are reduced to at most MAX_FACES triangles by vertex clustering on a
grid (deterministic, no per-object choices). Nothing here depends on the clip or the object's kind.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

from vrecon.tuning import MAX_FACES  # noqa: F401  (tuned values, one place)

BOX_U = np.array([[x, y, z] for x in (-0.5, 0.5) for y in (-0.5, 0.5) for z in (-0.5, 0.5)])
# box faces as corner indices into BOX_U (index = 4*ix + 2*iy + iz)
BOX_FACES = ((0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3))


def simplify(verts: np.ndarray, faces: np.ndarray, max_faces: int) -> tuple[np.ndarray, np.ndarray]:
    """Vertex clustering: snap vertices to a grid, merge, drop collapsed and repeated triangles. The grid is
    refined until the result first exceeds max_faces, then the last result within it is kept."""
    if len(faces) <= max_faces:
        return verts, faces
    lo, hi = verts.min(0), verts.max(0)
    span = np.maximum(hi - lo, 1e-9)
    best = None
    for cells in range(2, 257):
        q = np.floor((verts - lo) / span * (cells - 1e-9)).astype(np.int64)
        key = (q[:, 0] * cells + q[:, 1]) * cells + q[:, 2]
        uniq, inv = np.unique(key, return_inverse=True)
        f = inv[faces]
        f = f[(f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])]
        f = np.unique(np.sort(f, axis=1), axis=0) if len(f) else f
        if len(f) > max_faces:
            break
        v = np.zeros((len(uniq), 3))
        np.add.at(v, inv, verts)
        v /= np.bincount(inv, minlength=len(uniq))[:, None]
        best = (v, f)
    return best if best is not None else (verts, faces[:max_faces])


@lru_cache(maxsize=256)
def _mesh_file(path: str, mtime: float, max_faces: int) -> tuple[np.ndarray, tuple]:
    with np.load(path) as d:
        verts, faces = np.asarray(d["verts"], float), np.asarray(d["faces"], np.int64)
    ext = np.maximum(verts.max(0) - verts.min(0), 1e-9)
    v, f = simplify(verts, faces, max_faces)
    return v / ext, tuple(tuple(int(i) for i in t) for t in f)


def object_shape(obj, root: str | Path | None = None, max_faces: int | None = None) -> tuple[np.ndarray, tuple]:
    """(U, faces) of a SceneObject: its mesh file if it has one that exists, else the proxy box."""
    if obj.asset is not None:
        p = Path(obj.asset.path)
        if not p.is_absolute():
            p = Path(root or Path.cwd()) / p
        if p.suffix == ".npz" and p.is_file():
            return _mesh_file(str(p), p.stat().st_mtime, max_faces or MAX_FACES)
    return BOX_U, BOX_FACES


def bottom(obj, root: str | Path | None = None) -> float:
    """Lowest unit z of the shape (-0.5 for a box): a height change keeps this level where it was."""
    return float(object_shape(obj, root)[0][:, 2].min())
