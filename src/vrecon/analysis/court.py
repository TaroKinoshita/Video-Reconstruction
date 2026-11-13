"""Tennis court calibration (docs/VENUE_PLAN.md phase B): the court's white lines, matched to the ITF court,
give the court plane, the scale in metres and the camera (focal length and pose).

Court frame (metres): origin at the court centre on the ground, X across the court (singles sidelines at
+-4.115, doubles at +-5.485), Y along it (baselines at +-11.885, service lines at +-6.40), Z up.

    white_lines(frame, exclude)            thin bright low-saturation pixels (top-hat), masked things removed
    detect_lines(white)                    long straight lines (rho, theta, support) from Hough segments
    match_court(lines, white, w, h)        assignment of detected lines to court lines, best by white support
    camera_from_homography(H, w, h)        focal length (principal point at the centre), R, t; the court -> image
    calibrate_points(img, court, w, h)     the same from point pairs (manual mode: click intersections)
    project(K, R, t, X)                    court points -> pixels
A match is only a hypothesis: check it on other frames (reprojection of the lines on their white pixels) and
against things on the court (net height, feet on the ground).
"""

from __future__ import annotations

import itertools

import numpy as np

# ITF: doubles 23.77 x 10.97 m, singles width 8.23 m, service line 6.40 m from the net, net 0.914 m at the centre
HALF_L, HALF_DW, HALF_SW, SERVICE = 11.885, 5.485, 4.115, 6.40
NET_CENTRE_H, NET_POST_H, NET_POST_X = 0.914, 1.07, 6.40
COURT_LINES = {  # name: ((x0, y0), (x1, y1)) on the ground
    "baseline_near": ((-HALF_DW, -HALF_L), (HALF_DW, -HALF_L)),
    "baseline_far": ((-HALF_DW, HALF_L), (HALF_DW, HALF_L)),
    "service_near": ((-HALF_SW, -SERVICE), (HALF_SW, -SERVICE)),
    "service_far": ((-HALF_SW, SERVICE), (HALF_SW, SERVICE)),
    "doubles_left": ((-HALF_DW, -HALF_L), (-HALF_DW, HALF_L)),
    "doubles_right": ((HALF_DW, -HALF_L), (HALF_DW, HALF_L)),
    "singles_left": ((-HALF_SW, -HALF_L), (-HALF_SW, HALF_L)),
    "singles_right": ((HALF_SW, -HALF_L), (HALF_SW, HALF_L)),
    "centre_service": ((0.0, -SERVICE), (0.0, SERVICE)),
}
ACROSS = ["baseline_near", "service_near", "service_far", "baseline_far"]  # ordered along +Y
ALONG = ["doubles_left", "singles_left", "centre_service", "singles_right", "doubles_right"]  # ordered along +X


def white_lines(frame: np.ndarray, exclude: np.ndarray | None = None, top: int = 0, tophat: int = 9,
                min_contrast: float = 40, max_sat: float = 80) -> np.ndarray:
    import cv2

    g = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    s = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)[..., 1]
    th = cv2.morphologyEx(g, cv2.MORPH_TOPHAT, np.ones((tophat, tophat), np.uint8))
    w = (th > min_contrast) & (s < max_sat)
    if exclude is not None:
        w &= ~exclude
    w[:top] = False
    return w


def _line_of(seg):
    x1, y1, x2, y2 = seg
    th = np.arctan2(x2 - x1, -(y2 - y1))  # normal angle
    n = np.array([np.cos(th), np.sin(th)])
    return n, float(n @ (x1, y1))


def detect_lines(white: np.ndarray, min_len: int = 80, angle_tol_deg: float = 2.0, dist_tol: float = 6.0):
    """Hough segments merged into lines. Returns a list of dicts: normal n (unit), offset c (n . p = c),
    support (pixels near the line), angle_deg of the line direction from horizontal."""
    import cv2

    segs = cv2.HoughLinesP(white.astype(np.uint8) * 255, 1, np.pi / 360, 80, minLineLength=min_len, maxLineGap=15)
    if segs is None:
        return []
    segs = sorted(segs.reshape(-1, 4).tolist(), key=lambda s: -np.hypot(s[2] - s[0], s[3] - s[1]))
    groups = []
    for s in segs:
        n, c = _line_of(s)
        for gr in groups:
            if abs(n @ gr["n"]) > np.cos(np.radians(angle_tol_deg)):
                sign = np.sign(n @ gr["n"])
                mid = np.array([(s[0] + s[2]) / 2, (s[1] + s[3]) / 2])
                if abs(gr["n"] @ mid - gr["c"]) < dist_tol:
                    gr["segs"].append(s)
                    break
                del sign
        else:
            groups.append({"n": n, "c": c, "segs": [s]})
    ys, xs = np.nonzero(white)
    pts = np.stack([xs, ys], 1).astype(float)
    out = []
    for gr in groups:  # refit on the white pixels near the line (total least squares)
        d = np.abs(pts @ gr["n"] - gr["c"])
        q = pts[d < 3]
        if len(q) < min_len:
            continue
        m = q.mean(0)
        u, s_, vt = np.linalg.svd(q - m, full_matrices=False)
        dirv = vt[0]
        n = np.array([-dirv[1], dirv[0]])
        c = float(n @ m)
        sup = int((np.abs(pts @ n - c) < 2).sum())
        out.append({"n": n, "c": c, "support": sup, "angle_deg": float(np.degrees(np.arctan2(dirv[1], dirv[0])) % 180)})
    return sorted(out, key=lambda x: -x["support"])


def _hom_line(L):
    return np.array([L["n"][0], L["n"][1], -L["c"]])


def _intersect(l1, l2):
    p = np.cross(l1, l2)
    return p[:2] / p[2] if abs(p[2]) > 1e-12 else None


def homography(court_xy: np.ndarray, img_xy: np.ndarray) -> np.ndarray:
    """Direct linear transform (>= 4 point pairs), court (X, Y) -> image (u, v)."""
    A = []
    for (X, Y), (u, v) in zip(court_xy, img_xy):
        A.append([X, Y, 1, 0, 0, 0, -u * X, -u * Y, -u])
        A.append([0, 0, 0, X, Y, 1, -v * X, -v * Y, -v])
    h = np.linalg.svd(np.array(A, float))[2][-1]
    return h.reshape(3, 3) / h[-1]


def camera_from_homography(H: np.ndarray, w: int, h: int):
    """(K, R, t, f) with the principal point at the image centre and square pixels. H maps court (X, Y, 1) to
    pixels; H ~ K [r1 r2 t]. The focal length comes from r1 . r2 = 0 and |r1| = |r2| (averaged where both hold).
    None when neither gives one: a camera looking straight down a court axis sees the across lines parallel, and
    then the court alone does not fix the focal length (use another clue, e.g. a person's height)."""
    cx, cy = (w - 1) / 2, (h - 1) / 2
    T = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1.0]])
    G = T @ H  # pixels relative to the centre: G ~ diag(f, f, 1) [r1 r2 t]
    g1, g2 = G[:, 0], G[:, 1]
    est = []
    den = g1[2] * g2[2]
    if abs(den) > 1e-12:
        v = -(g1[0] * g2[0] + g1[1] * g2[1]) / den
        if v > 0:
            est.append(np.sqrt(v))
    den = g1[2] ** 2 - g2[2] ** 2
    if abs(den) > 1e-12:
        v = -((g1[0] ** 2 + g1[1] ** 2) - (g2[0] ** 2 + g2[1] ** 2)) / den
        if v > 0:
            est.append(np.sqrt(v))
    if not est:
        return None
    f = float(np.median(est))
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
    M = np.linalg.inv(K) @ H
    lam = 2.0 / (np.linalg.norm(M[:, 0]) + np.linalg.norm(M[:, 1]))
    if (lam * M[:, 2])[2] < 0:  # the court in front of the camera
        lam = -lam
    r1, r2, t = lam * M[:, 0], lam * M[:, 1], lam * M[:, 2]
    R = np.c_[r1, r2, np.cross(r1, r2)]
    U, _, Vt = np.linalg.svd(R)  # nearest rotation
    R = U @ Vt
    if np.linalg.det(R) < 0:
        R = U @ np.diag([1, 1, -1]) @ Vt
    return K, R, t, f, est


def project(K, R, t, X: np.ndarray) -> np.ndarray:
    pc = X @ R.T + t
    return (pc @ K.T)[:, :2] / pc[:, 2:]


def ground_to_image(H: np.ndarray, XY: np.ndarray):
    """Court ground points -> (pixels, in front of the camera)."""
    p = np.c_[XY, np.ones(len(XY))] @ H.T
    return p[:, :2] / p[:, 2:], p[:, 2] > 0


def line_support(H: np.ndarray, white: np.ndarray, names=None, samples: int = 6000, tol: int = 2,
                 valid: np.ndarray | None = None):
    """Share of each court line's projected samples (inside the picture, in front of the camera) on white pixels,
    and how many picture pixels the line covers there. Uses the ground homography (exact for lines on the ground).
    valid: pixels where a line could be seen (not covered by people, the net, the stands, or cut off at the top);
    others are not counted either way."""
    import cv2

    hh, ww = white.shape
    wd = cv2.dilate(white.astype(np.uint8), np.ones((2 * tol + 1, 2 * tol + 1), np.uint8)) > 0
    sign = np.sign((np.array([0.0, 0.0, 1.0]) @ H.T)[2]) or 1.0
    out = {}
    for nm in names or COURT_LINES:
        (x0, y0), (x1, y1) = COURT_LINES[nm]
        n = samples
        uv, front = ground_to_image(H * sign, np.c_[np.linspace(x0, x1, n), np.linspace(y0, y1, n)])
        uv = uv[front]
        uv = np.unique(np.round(uv).astype(int), axis=0)  # count picture pixels, not metres
        ins = (uv[:, 0] >= 0) & (uv[:, 0] < ww) & (uv[:, 1] >= 0) & (uv[:, 1] < hh)
        uv = uv[ins].astype(int)
        if valid is not None:
            uv = uv[valid[uv[:, 1], uv[:, 0]]]
        out[nm] = (float(wd[uv[:, 1], uv[:, 0]].mean()) if len(uv) else None, int(len(uv)))
    return out


def match_court(lines: list[dict], white: np.ndarray, w: int, h: int, top_k: int = 6,
                f_range: tuple[float, float] = (0.3, 8.0), valid: np.ndarray | None = None):
    """Try assignments of two 'across' and two 'along' detected lines to court lines; keep the camera whose
    projected court lines lie on the most white pixels. Any two of the strongest lines may be the 'across' pair
    (a low camera sees the sidelines nearly flat: 18-23 deg from horizontal on the tennis clip), the two lines of a
    pair must not cross inside the picture. Returns (best dict, all scored candidates)."""
    lines = lines[:top_k]
    hh, ww = white.shape

    def apart(L1, L2):  # the lines of a pair do not meet inside the picture (court lines of one family)
        p = _intersect(_hom_line(L1), _hom_line(L2))
        return p is None or not (0 <= p[0] < ww and 0 <= p[1] < hh)

    pairs = [(a, b) for a, b in itertools.combinations(lines, 2) if apart(a, b)]
    cands = []
    for (a1, a2), (b1, b2) in itertools.permutations(pairs, 2):
        if len({id(a1), id(a2), id(b1), id(b2)}) < 4:
            continue
        if True:
            for ma in itertools.permutations(ACROSS, 2):
                for mb in itertools.permutations(ALONG, 2):
                    if "centre_service" in mb and any(x.startswith("baseline") for x in ma):
                        continue  # the centre service line does not meet the baselines
                    img, court = [], []
                    for La, na in zip((a1, a2), ma):
                        for Lb, nb in zip((b1, b2), mb):
                            p = _intersect(_hom_line(La), _hom_line(Lb))
                            if p is None:
                                break
                            X = COURT_LINES[nb][0][0]
                            Y = COURT_LINES[na][0][1]
                            img.append(p)
                            court.append((X, Y))
                    if len(img) < 4:
                        continue
                    H = homography(np.array(court), np.array(img))
                    cam = camera_from_homography(H, w, h)
                    if cam is None:
                        continue
                    K, R, t, f, est = cam
                    if not (f_range[0] * w < f < f_range[1] * w):
                        continue
                    centre = -R.T @ t
                    if centre[2] < 0.3:  # camera above the ground
                        continue
                    sup = line_support(H, white, valid=valid)
                    inside = [(s, n) for s, n in sup.values() if s is not None and n > 20]
                    if len(inside) < 3:
                        continue
                    # pixels of court lines that land on white, minus half the ones that land on nothing: a match must
                    # explain the long lines, not squeeze the court into a few pixels
                    on = sum(s * n for s, n in inside)
                    score = float(on - 0.5 * (sum(n for _, n in inside) - on))
                    cands.append({"score": score, "H": H, "f_from_h": f, "f_estimates": est, "img": np.array(img),
                                  "court": np.array(court), "assign": {"across": ma, "along": mb}, "support": sup,
                                  "centre_from_h": centre})
    cands.sort(key=lambda c: -c["score"])
    return (cands[0] if cands else None), cands


def camera_from_points(img_xy, court_xyz, w: int, h: int, f_range=(0.3, 8.0)):
    """Camera (K, R, t, f, mean reprojection error in px) from point pairs on the court (Z may be non-zero, e.g.
    net posts): principal point at the centre, square pixels; the focal length is searched (log grid, then
    golden section) with OpenCV's PnP for each. Works with the 4 line intersections of a match or with clicked
    points (manual mode)."""
    import cv2

    img = np.asarray(img_xy, np.float64).reshape(-1, 1, 2)
    obj = np.asarray(court_xyz, np.float64)
    if obj.shape[1] == 2:
        obj = np.c_[obj, np.zeros(len(obj))]
    cx, cy = (w - 1) / 2, (h - 1) / 2

    def fit(f):
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
        ok, rv, tv = cv2.solvePnP(obj, img, K, None, flags=cv2.SOLVEPNP_IPPE if len(obj) >= 4 and not obj[:, 2].any()
                                  else cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            return np.inf, None
        ok, rv, tv = cv2.solvePnP(obj, img, K, None, rv, tv, useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
        R = cv2.Rodrigues(rv)[0]
        t = tv.ravel()
        err = float(np.mean(np.linalg.norm(project(K, R, t, obj) - img.reshape(-1, 2), axis=1)))
        if (obj @ R.T + t)[:, 2].min() <= 0:
            err = np.inf
        return err, (K, R, t)

    fs = np.geomspace(f_range[0] * w, f_range[1] * w, 60)
    errs = [fit(f)[0] for f in fs]
    i = int(np.argmin(errs))
    lo, hi = fs[max(i - 1, 0)], fs[min(i + 1, len(fs) - 1)]
    g = (np.sqrt(5) - 1) / 2
    for _ in range(40):
        a, b = hi - g * (hi - lo), lo + g * (hi - lo)
        if fit(a)[0] < fit(b)[0]:
            hi = b
        else:
            lo = a
    f = (lo + hi) / 2
    err, cam = fit(f)
    if cam is None:
        return None
    K, R, t = cam
    return K, R, t, float(f), err


def calibrate_points(img_xy, court_xy, w: int, h: int):
    """Manual mode: >= 4 clicked intersections (pixels) with their court positions (metres, X Y or X Y Z)."""
    return camera_from_points(img_xy, court_xy, w, h)


__all__ = ["COURT_LINES", "calibrate_points", "camera_fixed_centre", "camera_from_homography", "camera_from_points", "detect_lines", "match_near",
           "ground_to_image", "homography", "line_support", "match_court", "project", "white_lines"]


def match_near(lines: list[dict], H_prev: np.ndarray, w: int, h: int, max_dist: float = 12.0, max_deg: float = 4.0):
    """Tracking: assign detected lines to the court lines projected by the previous frame's homography (nearest
    in angle and offset); every across x along pair of matched lines gives an intersection. Returns (image points,
    court points, matched names)."""
    proj = {}
    for nm, (p0, p1) in COURT_LINES.items():
        uv, front = ground_to_image(H_prev, np.array([p0, p1], float))
        if not front.all():
            continue
        d = uv[1] - uv[0]
        n = np.array([-d[1], d[0]]) / np.linalg.norm(d)
        proj[nm] = (n, float(n @ uv[0]), uv)
    used = {}
    for L in lines:
        best = None
        for nm, (n, c, uv) in proj.items():
            cosang = abs(n @ L["n"])
            if cosang < np.cos(np.radians(max_deg)):
                continue
            sgn = np.sign(n @ L["n"])
            mid = uv.mean(0)
            off = abs(L["n"] @ mid - L["c"]) if sgn > 0 else abs(-L["n"] @ mid + L["c"])
            if off < max_dist and (best is None or off < best[0]):
                best = (off, nm)
        if best and (best[1] not in used or used[best[1]]["support"] < L["support"]):
            used[best[1]] = L
    img, court = [], []
    acr = [n for n in used if n in ACROSS]
    alo = [n for n in used if n in ALONG]
    for na in acr:
        for nb in alo:
            if nb == "centre_service" and na.startswith("baseline"):
                continue
            p = _intersect(_hom_line(used[na]), _hom_line(used[nb]))
            if p is None:
                continue
            img.append(p)
            court.append((COURT_LINES[nb][0][0], COURT_LINES[na][0][1]))
    return np.array(img), np.array(court), sorted(used)


def camera_fixed_centre(img_xy, court_xyz, centre, w: int, h: int, f_range=(0.3, 8.0), focal: float | None = None):
    """Camera with a known centre (a camera on a tripod that pans and zooms): focal length searched, rotation by
    aligning pixel rays with the directions to the court points (Kabsch). focal: use this focal length (a camera
    that pans without zooming) instead of searching. Returns (K, R, t, f, error px)."""
    img = np.asarray(img_xy, float)
    X = np.asarray(court_xyz, float)
    if X.shape[1] == 2:
        X = np.c_[X, np.zeros(len(X))]
    C = np.asarray(centre, float)
    cx, cy = (w - 1) / 2, (h - 1) / 2
    wd = (X - C) / np.linalg.norm(X - C, axis=1, keepdims=True)

    def fit(f):
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
        cd = np.c_[(img[:, 0] - cx) / f, (img[:, 1] - cy) / f, np.ones(len(img))]
        cd /= np.linalg.norm(cd, axis=1, keepdims=True)
        U, _, Vt = np.linalg.svd(cd.T @ wd)  # R maps world directions to camera directions
        D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
        R = U @ D @ Vt
        t = -R @ C
        err = float(np.mean(np.linalg.norm(project(K, R, t, X) - img, axis=1)))
        return err, (K, R, t)

    if focal is not None:
        err, (K, R, t) = fit(float(focal))
        return K, R, t, float(focal), err
    fs = np.geomspace(f_range[0] * w, f_range[1] * w, 80)
    errs = [fit(f)[0] for f in fs]
    i = int(np.argmin(errs))
    lo, hi = fs[max(i - 1, 0)], fs[min(i + 1, len(fs) - 1)]
    g = (np.sqrt(5) - 1) / 2
    for _ in range(50):
        a, b = hi - g * (hi - lo), lo + g * (hi - lo)
        if fit(a)[0] < fit(b)[0]:
            hi = b
        else:
            lo = a
    f = (lo + hi) / 2
    err, (K, R, t) = fit(f)
    return K, R, t, float(f), err
