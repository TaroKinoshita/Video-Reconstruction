"""Set pieces of the shared world as flat pieces and boxes, from the merged points (after multishot_join /
multishot_closeups).

    python scripts/multishot_layout.py IN.npz OUT.npz

Points of every cut in the shared world (slot 0), except close-up points (those are mostly a person up close):
  planes   sequential RANSAC: the largest plane is taken, its points removed, again, while a plane holds at least
           MIN_SHARE of the points (at most MAX_PLANES). Each plane keeps the cells of a grid on it that hold
           enough points (so a wall with a gap keeps the gap), cells merged into quads; label: horizontal and near
           z = 0 -> ground, other horizontal -> structure (a ledge, a roof), vertical -> structure (a wall)
  boxes    what is left: connected voxel clusters; one with enough points -> its upright bounding box (structure)
  colours  the median source colour of each piece (for a look pass; the control uses the category colours)
Writes layout_quads [N,4,3], layout_quad_cat, layout_quad_col, layout_boxes [M,8] (centre xyz, size xyz, yaw,
spare), layout_box_col, and info["layout"] (counts, shares).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vrecon.analysis.from_4d import fit_plane  # noqa: E402

MIN_SHARE, MAX_PLANES = 0.04, 10
SNAP_DEG = 20.0  # assumption: built surroundings are mostly level floors and upright walls
MAX_BOX = 4.0
BOX_GROUND_GAP = 0.5  # m
MERGE_DEG, MERGE_GAP = 15.0, 0.5  # planes of different cuts that are one surface: angle, offset (m, or 5% of range)
FAR = 15.0
STILL_TRAVEL = 0.5  # m: a camera that moves less gives no depth for the far background
BACKDROP_AT = 25.0  # m: where such background is put  # m from the cameras: depth there is too loose for shape; such a plane becomes a backdrop panel
MIN_AREA = 0.5  # m^2: smaller pieces are dropped  # m: a leftover cluster bigger than this is not one compact thing
CELL_PER_M = 4.0  # grid cells per metre on a plane (0.25 m)
MIN_CELL_PTS = 3
GROUND_Z = 0.25


def plane_quads(p: np.ndarray, n: np.ndarray, cell: float) -> list[np.ndarray]:
    """occupied grid cells of the points on a plane, merged along rows into quads (in world)."""
    u = np.cross(n, [0, 0, 1.0]) if abs(n[2]) < 0.9 else np.cross(n, [1.0, 0, 0])
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    c0 = p.mean(0)
    uv = np.stack([(p - c0) @ u, (p - c0) @ v], 1)
    ij = np.floor(uv / cell).astype(int)
    keys, cnt = np.unique(ij, axis=0, return_counts=True)
    occ = {tuple(k) for k, c in zip(keys, cnt) if c >= MIN_CELL_PTS}
    # close one-cell holes (sparse points on a flat face)
    grown = set(occ)
    for i, j in occ:
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            a, b = (i + di, j + dj), (i + 2 * di, j + 2 * dj)
            if a not in occ and b in occ:
                grown.add(a)
    d = float((c0 @ n))
    quads = []
    for j in sorted({b for _, b in grown}):
        row = sorted(i for i, b in grown if b == j)
        s = row[0]
        for a, b in zip(row, row[1:] + [None]):
            if b is None or b != a + 1:
                q = []
                for x, y in ((s, j), (a + 1, j), (a + 1, j + 1), (s, j + 1)):
                    w = c0 + u * x * cell + v * y * cell
                    q.append(w - (w @ n - d) * n)
                quads.append(np.array(q))
                if b is not None:
                    s = b
    return quads


def snap(n: np.ndarray, deg: float = SNAP_DEG) -> np.ndarray:
    """near-horizontal planes become exactly horizontal, near-vertical ones exactly vertical (gravity is z)."""
    n = n / np.linalg.norm(n)
    if abs(n[2]) > np.cos(np.radians(deg)):
        return np.array([0, 0, np.sign(n[2])])
    if abs(n[2]) < np.sin(np.radians(deg)):
        h = np.array([n[0], n[1], 0.0])
        return h / np.linalg.norm(h)
    return n


def layout_of_cut(pts, col, keep_out, report, raw_planes):
    """sequential RANSAC planes (snapped) and compact leftover boxes of one cut's points."""
    total = len(pts)
    span = np.percentile(np.linalg.norm(pts - np.median(pts, 0), axis=1), 50)
    tol = max(0.02 * span, 0.04)
    quads, qcat, qcol, boxes, bcol = [], [], [], [], []
    rest = np.ones(total, bool)
    for it in range(MAX_PLANES):
        idx = np.flatnonzero(rest)
        if len(idx) < MIN_SHARE * total:
            break
        n0, d0, _ = fit_plane(pts[idx], tol=tol, seed=it)
        raw = idx[np.abs(pts[idx] @ n0 + d0) < tol]
        n = snap(n0)
        near = pts[idx][np.abs(pts[idx] @ n0 + d0) < 3 * tol]
        d = -float(np.median(near @ n)) if len(near) else d0
        inl = idx[np.abs(pts[idx] @ n + d) < tol]
        if len(inl) < 0.7 * len(raw):  # snapping loses the plane: keep it as fitted
            n, d, inl = n0, d0, raw
        if len(inl) < MIN_SHARE * total:
            break
        rest[inl] = False
        horiz = abs(n[2]) > 0.99
        level = float(-d / n[2]) if horiz else None
        cat = "ground" if horiz and abs(level) < GROUND_Z else "structure"
        if cat == "ground":
            n, d = np.array([0, 0, 1.0]), 0.0  # the shared floor is z = 0 by construction
        raw_planes.append({"n": n, "d": d, "pts": pts[inl], "col": np.median(col[inl], 0), "cat": cat})
        report["planes"].append({"points": int(len(inl)), "share": round(len(inl) / total, 3),
                                 "normal": n.round(3).tolist(), "level_z": level, "category": cat})
    left = np.flatnonzero(rest)
    if len(left):
        vox = 2.0 / CELL_PER_M
        key = np.floor(pts[left] / vox).astype(np.int64)
        uk, inv = np.unique(key, axis=0, return_inverse=True)
        inv = inv.ravel()
        lookup = {tuple(k): i for i, k in enumerate(uk)}
        parent = list(range(len(uk)))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        for i, k in enumerate(uk):
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        j = lookup.get((k[0] + dx, k[1] + dy, k[2] + dz))
                        if j is not None:
                            ra, rb = find(i), find(j)
                            if ra != rb:
                                parent[ra] = rb
        comp = np.array([find(i) for i in range(len(uk))])[inv]
        for cid in np.unique(comp):
            sel = left[comp == cid]
            if len(sel) < 0.01 * total:
                continue
            q = pts[sel]
            lo, hi = np.percentile(q, 5, 0), np.percentile(q, 95, 0)
            size = np.maximum(hi - lo, 0.05)
            inside = ((keep_out > lo - 0.3) & (keep_out < hi + 0.3)).all(1).any()
            grounded = lo[2] < BOX_GROUND_GAP  # a thing rests on the floor; a floating cluster is depth noise
            if size.max() > MAX_BOX or inside or not grounded:  # not compact, would swallow a camera/person, or floats
                report["boxes_dropped"] += 1
                continue
            boxes.append(np.concatenate([(lo + hi) / 2, size, [0.0, 0.0]]))
            bcol.append(np.median(col[sel], 0))
    report["plane_share"] = round(float(1 - rest.mean()), 3)
    return quads, qcat, qcol, boxes, bcol


def frame_axes(n: np.ndarray):
    u = np.cross(n, [0, 0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0, 0])
    u = u - (u @ n) * n
    u /= np.linalg.norm(u)
    return u, np.cross(n, u)


def regularize(raw: list, cams: np.ndarray):
    """Measurements give place and extent, not shape: planes from every cut that are one surface (normals within
    MERGE_DEG, offsets within MERGE_GAP or 5% of their distance) are merged; each merged surface becomes ONE
    rectangle spanning its points' 3rd-97th percentiles (gaps filled); an upright wall whose bottom is near the
    floor goes down to it; a surface farther than FAR from the cameras becomes a backdrop panel (a rectangle
    there, its extent widened by half, as depth noise there makes any finer shape meaningless); ground planes are
    dropped (the shared floor is one plane already); pieces under MIN_AREA are dropped."""
    groups = []
    for pl in raw:
        if pl["cat"] == "ground":
            continue
        dist = float(np.median(np.linalg.norm(pl["pts"][:, None] - cams[None, ::max(1, len(cams) // 50)], axis=2).min(1)))
        pl["dist"] = dist
        for g in groups:
            if np.degrees(np.arccos(np.clip(abs(g["n"] @ pl["n"]), -1, 1))) < MERGE_DEG:
                n = g["n"] if g["n"] @ pl["n"] > 0 else -g["n"]
                gap = abs(np.median(pl["pts"] @ n) - np.median(g["pts"] @ n))
                if gap < max(MERGE_GAP, 0.05 * min(dist, g["dist"])):
                    g["pts"] = np.concatenate([g["pts"], pl["pts"]])
                    g["cols"].append(pl["col"])
                    g["dist"] = min(g["dist"], dist)
                    g["cuts"] += 1
                    break
        else:
            groups.append({"n": pl["n"], "pts": pl["pts"], "cols": [pl["col"]], "dist": dist, "cuts": 1})
    quads, cats, cols, rep = [], [], [], []
    for g in groups:
        n = g["n"] / np.linalg.norm(g["n"])
        off = float(np.median(g["pts"] @ n))
        u, v = frame_axes(n)
        a, b = g["pts"] @ u, g["pts"] @ v
        a0, a1 = np.percentile(a, [3, 97])
        b0, b1 = np.percentile(b, [3, 97])
        far = g["dist"] > FAR
        if far:
            ca, cb, ha, hb = (a0 + a1) / 2, (b0 + b1) / 2, (a1 - a0) * 0.75, (b1 - b0) * 0.75
            a0, a1, b0, b1 = ca - ha, ca + ha, cb - hb, cb + hb
        corners = [n * off + u * x + v * y for x, y in ((a0, b0), (a1, b0), (a1, b1), (a0, b1))]
        upright = abs(n[2]) < 0.1
        if upright:  # a wall stands on the floor when its foot is near it (or it is a far backdrop)
            zs = np.array([c[2] for c in corners])
            if zs.min() < 1.5 or far:
                for i in np.argsort(zs)[:2]:
                    corners[i] = corners[i].copy()
                    corners[i][2] = 0.0
        area = float(np.linalg.norm(np.cross(corners[1] - corners[0], corners[3] - corners[0])))
        if area < MIN_AREA:
            continue
        quads.append(np.array(corners))
        cats.append("structure")
        cols.append(np.median(np.array(g["cols"]), 0))
        rep.append({"normal": n.round(2).tolist(), "cuts": g["cuts"], "area_m2": round(area, 1),
                    "distance_m": round(g["dist"], 1), "backdrop": bool(far), "upright": bool(upright)})
    return quads, cats, cols, {"raw_planes": len(raw), "surfaces": len(quads), "pieces": rep}


def main() -> int:
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    z = dict(np.load(src))
    info = json.loads(str(z["info"]))
    keep_out = []
    for k, s in enumerate(info["shots"]):
        if s.get("slot") == 0:
            keep_out.append(z[f"s{k}_cam"][:, :3, 3])
            if len(z[f"s{k}_body"]):
                keep_out.append(z[f"s{k}_body"][:, 0])
    keep_out = np.concatenate(keep_out)
    Q, QC, QL, B, BL, rep, raw = [], [], [], [], [], {}, []
    # per cut: each cut's points agree with themselves; mixing cuts first gives planes no cut saw
    for k, s in enumerate(info["shots"]):
        if s.get("slot") != 0 or s.get("points_only_during_cut") or not len(z[f"s{k}_pts"]):
            continue
        r = {"planes": [], "boxes_dropped": 0}
        n_before = len(raw)
        q, qc, ql, b, bl = layout_of_cut(z[f"s{k}_pts"], z[f"s{k}_col"], keep_out, r, raw)
        cam = z[f"s{k}_cam"][:, :3, 3]
        travel = float(np.linalg.norm(np.ptp(cam, 0)))
        c0 = np.median(cam, 0)
        body = z[f"s{k}_body"]
        person = float(np.median(np.linalg.norm(body[:, 0] - cam[: len(body)], axis=1))) if len(body) else None
        pushed = 0
        for pl in raw[n_before:]:
            # a still camera sees no parallax: depth behind the person is the model's guess and comes out squashed
            # toward the camera; such a surface is pushed out along the camera's rays to backdrop distance, keeping
            # its size in the picture
            if pl["cat"] == "ground" or travel > STILL_TRAVEL:
                continue
            dist = float(np.median(np.linalg.norm(pl["pts"] - c0, axis=1)))
            if person is None or dist > person:
                f = max(BACKDROP_AT / max(dist, 1e-3), 1.0)
                pl["pts"] = c0 + (pl["pts"] - c0) * f
                pl["d"] = -float(np.median(pl["pts"] @ pl["n"]))
                pushed += 1
        r.update({"camera_travel_m": round(travel, 2), "person_distance_m": person, "pushed_to_backdrop": pushed})
        Q += q; QC += qc; QL += ql; B += b; BL += bl
        r["boxes"] = len(b)
        rep[k] = r
    cams = np.concatenate([z[f"s{k}_cam"][:, :3, 3] for k, s_ in enumerate(info["shots"]) if s_.get("slot") == 0])
    q2, c2, l2, reg = regularize(raw, cams)
    Q += q2; QC += c2; QL += l2
    rep["regularized"] = reg
    z["layout_quads"] = np.array(Q).reshape(-1, 4, 3)
    z["layout_quad_cat"] = np.array(QC)
    z["layout_quad_col"] = np.array(QL).reshape(-1, 3)
    z["layout_boxes"] = np.array(B).reshape(-1, 8)
    z["layout_box_col"] = np.array(BL).reshape(-1, 3)
    info["layout"] = rep
    z["info"] = json.dumps(info)
    np.savez_compressed(out, **z)
    print(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
