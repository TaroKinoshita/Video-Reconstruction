"""Held staffs: a thin straight object seen in a SAM mask is a staff on the frames where it touches a person's HAND
(body-mesh vertices near the hand keypoints, projected), so beams, branches and masks that only cross a body drop
out. Pure numpy/cv2.

  python scripts/v11_staffs.py PREP.npz PEOPLE_ANIM.npz STICKS.masks.npz BODY_DIR OUT.npz [--touch 4] [--gap 12]

Per stick label:
- grip: the hand (left or right of one person) with most vertices touching the mask; the grip point is that hand's
  vertex centre (from the moving body, so it is defined on every frame);
- length: one per label (a staff keeps its length): the 80th percentile of the seen lengths in the picture, back-
  projected at the hand's depth, so a staff seen at a slant is not taken as shorter;
- tilt: with the length known, the far end lies on the ray through its pixel at that distance from the grip; of the
  two solutions the one nearer the grip's depth is taken (the picture cannot tell toward / away); the near end
  continues the line through the grip. Picture-exact on both ends when the mask is whole.
- gaps up to --gap frames with the same hand are filled: direction slerped, grip from the hand on that frame;
- longer gaps (up to --hold) with the same person on both sides: the staff turns with the hand (vertex Kabsch),
  carried from both sides and blended;
- mask pieces on the main piece's line are joined (a hand cuts the mask); the span reaches at least the grip;
- the holder is the majority of nearby sightings (--holder-win), so it does not jump to a passing hand;
- direction smoothed in the hand's own frame (--sigma-dir); the staff stays in the hand --extend frames past
  its first and last sighting;
- with --video, a filled direction is searched in a cone around the carried one for the best line in the
  picture (on-line pixels nearer the staff colour than pixels beside them; with --bg-render, pixels that
  differ more from the rendered background than pixels beside them, outside the rendered people);
- one staff per person and frame (staffs are held two-handed): the label seen on more frames wins (whole-frame and hand-window masks overlap).
Output per label "<label>_t", "<label>_a", "<label>_b" world ends (PREP space), "<label>_r"; "labels".
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v11_mave_prep import read_ply  # noqa: E402

HANDS = {"right": range(21, 42), "left": range(42, 63)}  # MHR70 hand keypoints incl. wrist (41 right, 62 left)


def hand_vertices(body: Path, radius: float = 0.09) -> dict[str, np.ndarray]:
    """Mesh vertex ids near each hand's keypoints (same topology on every body file)."""
    js = next(body.glob("f*_obs_*.json"))
    meta = json.loads(js.read_text(encoding="utf-8"))["metadata"]["people"][0]
    V, _ = read_ply(js.with_name(js.stem + "_0.ply"))
    V = V * [1, -1, -1]
    K = np.array(meta["keypoints_3d"])
    out = {}
    for h, ids in HANDS.items():
        d = np.linalg.norm(V[:, None] - K[list(ids)][None], axis=2).min(1)
        out[h] = np.nonzero(d < radius)[0]
    return out


def body_at(anim, label: str, f: int, max_gap: int = 12):
    ts = anim[f"{label}_t"]
    if f < ts[0] or f > ts[-1]:
        return None
    j = int(np.searchsorted(ts, f))
    if ts[j] == f:
        return anim[f"{label}_v"][j]
    if ts[j] - ts[j - 1] > max_gap:
        return None
    w = (f - ts[j - 1]) / (ts[j] - ts[j - 1])
    return (1 - w) * anim[f"{label}_v"][j - 1] + w * anim[f"{label}_v"][j]


def slerp_dir(a: np.ndarray, b: np.ndarray, w: float) -> np.ndarray:
    c = float(np.clip(a @ b, -1, 1))
    th = np.arccos(c)
    if th < 1e-4:
        return a
    return (np.sin((1 - w) * th) * a + np.sin(w * th) * b) / np.sin(th)


def main() -> int:
    ap = argparse.ArgumentParser()
    for n in ("prep", "anim", "sticks", "body", "out"):
        ap.add_argument(n, type=Path)
    ap.add_argument("--touch", type=int, default=4, help="pixels: how near hand and staff mask must come")
    ap.add_argument("--min-px", type=int, default=40)
    ap.add_argument("--min-hits", type=int, default=3)
    ap.add_argument("--gap", type=int, default=12, help="frames: longest gap filled from the hand's motion")
    ap.add_argument("--hold", type=int, default=72, help="frames: longest gap bridged when the same person holds "
                    "the staff on both sides (the staff turns with the hand)")
    ap.add_argument("--seeds", type=Path, default=None, help="write point seeds (JSON) for gpu/observe/run_sam3_points.py")
    ap.add_argument("--seeds-per", type=int, default=6)
    ap.add_argument("--video", type=Path, default=None, help="film (PREP size): filled staffs are checked against it")
    ap.add_argument("--people-masks", type=Path, default=None)
    ap.add_argument("--bg-render", type=Path, default=None, help="renders r%%04d.png of the scene without people and "
                    "props: the line check then looks at film-minus-background, so trunks and posts do not count")
    ap.add_argument("--people-render", type=Path, default=None, help="renders with people but without props")
    ap.add_argument("--cone", type=float, default=60.0, help="degrees searched around a carried direction")
    ap.add_argument("--turn-cost", type=float, default=0.25, help="score lost for turning the whole cone away")
    ap.add_argument("--extend", type=int, default=10, help="frames a staff stays in the hand past its first/last sighting")
    ap.add_argument("--sigma-dir", type=float, default=2.0, help="frames; smoothing of the direction in the hand's frame")
    ap.add_argument("--holder-win", type=int, default=4, help="sightings on each side for the holder majority")
    ap.add_argument("--max-len", type=float, default=2.5, help="metres; longer is not a hand-held staff")
    a = ap.parse_args()
    prep = np.load(a.prep)
    info = json.loads(str(prep["info"]))
    W, H = info["size"]
    f_img = info["f_img"]
    cam_R, cam_T = prep["cam_R"], prep["cam_T"]
    anim = dict(np.load(a.anim))  # load once: an npz re-reads (and unzips) on every access
    people = json.loads(str(anim["labels"]))
    hv = hand_vertices(a.body)
    sd = np.load(a.sticks)
    masks, id_map = sd["masks"], json.loads(str(sd["id_map"]))
    ker = np.ones((2 * a.touch + 1,) * 2, np.uint8)

    def project(Xw, f):
        Xc = (Xw - cam_T[f]) @ cam_R[f]
        return Xc, f_img * Xc[:, 0] / Xc[:, 2] + W / 2, f_img * Xc[:, 1] / Xc[:, 2] + H / 2

    def grip_at(p, h, f):
        X = body_at(anim, p, f, a.gap)
        return None if X is None else X[hv[h]].mean(0)

    def hand_rot(p, h, f0, f1):
        """Rotation of the hand's vertices from frame f0 to f1 (Kabsch), or None."""
        X0, X1 = body_at(anim, p, f0, a.gap), body_at(anim, p, f1, a.gap)
        if X0 is None or X1 is None:
            return None
        P, Q = X0[hv[h]] - X0[hv[h]].mean(0), X1[hv[h]] - X1[hv[h]].mean(0)
        U, _, Vt = np.linalg.svd(P.T @ Q)
        D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
        return (U @ D @ Vt).T

    def carried(o, f0, f1):
        """World direction of a staff held in o's hand, carried frame by frame from f0 toward f1."""
        step = 1 if f1 > f0 else -1
        d, out_ = o["dirw"], {}
        for k in range(f0 + step, f1, step):
            R = hand_rot(o["p"], o["h"], k - step, k)
            if R is None:
                return None
            d = R @ d
            out_[k] = d
        return out_

    def join_pieces(m):
        """Largest piece plus the pieces on its line: a hand or arm across a staff cuts its mask in two."""
        n_cc, cc = cv2.connectedComponents(m.astype(np.uint8))
        if n_cc <= 2:
            return m
        sizes = np.bincount(cc.ravel())[1:]
        k0 = 1 + int(np.argmax(sizes))
        ys, xs = np.nonzero(cc == k0)
        P = np.stack([xs, ys], 1).astype(float)
        if len(P) < 5:
            return cc == k0
        c = P.mean(0)
        ev, evec = np.linalg.eigh(np.cov((P - c).T))
        d, nrm = evec[:, 1], evec[:, 0]
        tol = max(3.0, 3 * np.sqrt(max(ev[0], 0.25)))
        span = np.ptp((P - c) @ d)
        keep = cc == k0
        for k in range(1, n_cc):
            if k == k0 or sizes[k - 1] < a.min_px / 4:
                continue
            ys, xs = np.nonzero(cc == k)
            Q = np.stack([xs, ys], 1).astype(float) - c
            if np.median(np.abs(Q @ nrm)) < tol and np.abs(Q @ d).min() < span / 2 + span:
                keep |= cc == k
        return keep

    hand_ref = {}

    def hand_frame(p, h, f):
        """Rotation of the hand at frame f relative to its pose on a reference frame (Kabsch of hand vertices)."""
        X = body_at(anim, p, f, a.gap)
        if X is None:
            return None
        Q = X[hv[h]] - X[hv[h]].mean(0)
        if (p, h) not in hand_ref:
            hand_ref[(p, h)] = Q
        P = hand_ref[(p, h)]
        U, _, Vt = np.linalg.svd(P.T @ Q)
        D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
        return (U @ D @ Vt).T

    def extend_ends(frames, seen_fs):
        """A staff stays in the hand for a short while past its first and last sighting of a holder (carried with
        the hand): a fast swing hides it right where the fight is."""
        if a.extend <= 0 or not frames:
            return frames
        have = {f for f, _ in frames}
        add = []
        runs, cur = [], [frames[0]]
        for x in frames[1:]:
            if x[0] == cur[-1][0] + 1 and x[1]["p"] == cur[-1][1]["p"]:
                cur.append(x)
            else:
                runs.append(cur)
                cur = [x]
        runs.append(cur)
        for run in runs:
            for (f0, o), step in ((run[-1], 1), (run[0], -1)):
                d = o["dirw"]
                for k in range(1, a.extend + 1):
                    f = f0 + step * k
                    if f in have or f < 0 or f >= len(masks):
                        break
                    R = hand_rot(o["p"], o["h"], f - step, f)
                    if R is None:
                        break
                    d = R @ d
                    add.append((f, dict(o, dirw=d)))
                    have.add(f)
        return sorted(frames + add, key=lambda x: x[0])

    # film frames and people masks for checking filled staffs against the picture (optional)
    video = None
    if a.video is not None:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
        from vrecon.control.core import read_video_rgb
        video = read_video_rgb(a.video).astype(np.float32)
    pmasks = np.load(a.people_masks)["masks"] if a.people_masks is not None else None
    sph = np.array([[np.cos(t) * np.sqrt(1 - z * z), np.sin(t) * np.sqrt(1 - z * z), z]
                    for z, t in zip(np.linspace(-1, 1, 600), np.arange(600) * np.pi * (3 - np.sqrt(5)))])

    fg_cache = {}

    def fg_at(f):
        """Film-minus-background difference (blurred) and the rendered people's shape on frame f, or (None, None)."""
        if a.bg_render is None:
            return None, None
        if f not in fg_cache:
            def rd(d):
                im = cv2.imdecode(np.fromfile(str(d / f"r{f:04d}.png"), np.uint8), 1)[:, :, ::-1]
                return cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA).astype(np.float32)
            bg = rd(a.bg_render)
            diff = cv2.GaussianBlur(np.linalg.norm(video[f] - bg, axis=2), (0, 0), 1.5)
            ppl = None
            if a.people_render is not None:
                ppl = np.linalg.norm(rd(a.people_render) - bg, axis=2) > 12
                ppl = cv2.dilate(ppl.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
            fg_cache.clear()
            fg_cache[f] = (diff, ppl if ppl is not None else np.zeros((H, W), bool))
        return fg_cache[f]

    def line_score(f, G, dw, length, frac, colour):
        """How much the picture looks like a staff along the projected far part: pixels on the line nearer the
        staff colour than pixels beside it (both sides). Points on people are skipped (arms say nothing)."""
        Xc, u, v = project(np.array([G + dw * length * q * frac for q in np.linspace(0.3, 1.0, 16)]), f)
        if (Xc[:, 2] < 0.1).any():
            return -1.0
        _, gu, gv = project(G[None], f)
        d = np.array([u[-1] - gu[0], v[-1] - gv[0]])
        if np.linalg.norm(d) < 8:
            return -1.0
        n = np.array([-d[1], d[0]]) / np.linalg.norm(d)
        img = video[f]
        fg, ppl = fg_at(f)
        sc, k = 0.0, 0
        for x, y in zip(u, v):
            xi, yi = int(round(x)), int(round(y))
            if not (6 <= xi < W - 6 and 6 <= yi < H - 6):
                continue
            if pmasks is not None and pmasks[f, yi, xi] > 0:
                continue
            if fg is not None:
                # foreground = film minus the rendered background (trunks drop out); people's rendered shapes skipped
                if ppl[yi, xi]:
                    continue
                side = [fg[int(round(y + o * n[1])), int(round(x + o * n[0]))] for o in (-6, 6)]
                sc += (fg[yi, xi] - max(side)) / 30.0
                k += 1
                continue
            on = np.linalg.norm(img[yi, xi] - colour)
            side = [np.linalg.norm(img[int(round(y + o * n[1])), int(round(x + o * n[0]))] - colour) for o in (-6, 6)]
            sc += (min(side) - on) / 40.0
            k += 1
        return sc / k if k >= 5 else -1.0

    def refine_by_image(frames, seen_fs, length, num):
        """Filled frames (not seen in a mask) get the direction that best matches the picture within a cone around
        the carried direction, with a penalty for turning away from it: carried directions drift."""
        if video is None:
            return frames
        pix = [video[f][masks[f] == num] for f in seen_fs if (masks[f] == num).any()]
        if not pix:
            return frames
        colour = np.median(np.concatenate(pix), 0)
        res_ = []
        for f, o in frames:
            if f in seen_fs:
                res_.append((f, o))
                continue
            G = grip_at(o["p"], o["h"], f)
            if G is None:
                res_.append((f, o))
                continue
            d0 = o["dirw"] / np.linalg.norm(o["dirw"])
            Rw = cam_R[f]
            d0c = d0 @ Rw  # world -> camera
            cand = sph[sph @ d0c > np.cos(np.radians(a.cone))]
            best, best_s = d0, None
            for dc in np.vstack([d0c[None], cand]):
                dwc = dc @ Rw.T
                sc = line_score(f, G, dwc, length, o["frac"], colour)
                sc -= a.turn_cost * np.degrees(np.arccos(np.clip(dc @ d0c, -1, 1))) / a.cone
                if best_s is None or sc > best_s:
                    best, best_s = dwc, sc
            res_.append((f, dict(o, dirw=best)))
        return res_

    def smooth_in_hand(frames, seen_fs):
        """Direction smoothed in the hand's own frame (weights: seen 1, filled 0.3), so a staff follows the hand
        smoothly instead of every per-frame mask."""
        if a.sigma_dir <= 0:
            return frames
        loc = []
        for f, o in frames:
            R = hand_frame(o["p"], o["h"], f)
            loc.append(None if R is None else R.T @ (o["dirw"] / np.linalg.norm(o["dirw"])))
        res_ = []
        for i, (f, o) in enumerate(frames):
            if loc[i] is None:
                res_.append((f, o))
                continue
            acc, wsum = np.zeros(3), 0.0
            for j in range(max(0, i - 4 * int(a.sigma_dir) - 1), min(len(frames), i + 4 * int(a.sigma_dir) + 2)):
                g, oj = frames[j]
                if loc[j] is None or (oj["p"], oj["h"]) != (o["p"], o["h"]) or abs(g - f) > 3 * a.sigma_dir:
                    continue
                wj = np.exp(-0.5 * ((g - f) / a.sigma_dir) ** 2) * (1.0 if g in seen_fs else 0.3)
                acc += wj * loc[j]
                wsum += wj
            d = acc / max(np.linalg.norm(acc), 1e-9)
            R = hand_frame(o["p"], o["h"], f)
            res_.append((f, dict(o, dirw=R @ d)))
        return res_

    out, report, staffs = {}, {}, []
    for num, lab in id_map.items():
        seen = {}  # f -> (mask, person, hand)
        for f in range(len(masks)):
            m = masks[f] == int(num)
            if m.sum() < a.min_px:
                continue
            m = join_pieces(m)
            near = cv2.dilate(m.astype(np.uint8), ker).astype(bool)
            best = None
            for p in people:
                X = body_at(anim, p, f, a.gap)
                if X is None:
                    continue
                for h, ids in hv.items():
                    Xc, u, v = project(X[ids], f)
                    ok = (Xc[:, 2] > 0.1) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
                    hits = int(near[v[ok].astype(int), u[ok].astype(int)].sum())
                    if hits >= a.min_hits and (best is None or hits > best[0]):
                        best = (hits, p, h)
            if best is not None:
                seen[f] = (m, best[1], best[2])
        # the holder changes only when most nearby sightings agree (a passing hand of another person does not take it)
        fs_seen = sorted(seen)
        obs = {}
        for i, f in enumerate(fs_seen):
            m, _, _ = seen[f]
            win = [seen[g][1:] for g in fs_seen[max(0, i - a.holder_win):i + a.holder_win + 1]
                   if abs(g - f) <= 2 * a.holder_win]
            p = max({x[0] for x in win}, key=lambda q: sum(x[0] == q for x in win))
            hs = [x[1] for x in win if x[0] == p]
            h = seen[f][2] if seen[f][1] == p else max(set(hs), key=hs.count)
            ys, xs = np.nonzero(m)
            P = np.stack([xs, ys], 1).astype(float)
            c = P.mean(0)
            ev, evec = np.linalg.eigh(np.cov((P - c).T))
            d = evec[:, 1]
            s = (P - c) @ d
            thick = 4 * np.sqrt(max(ev[0], 0.25))
            if np.ptp(s) < 4 * thick:  # not long and thin
                continue
            g = grip_at(p, h, f)
            if g is None:
                continue
            gc, gu, gv = project(g[None], f)
            sg = (np.array([gu[0], gv[0]]) - c) @ d
            # a held staff passes through the hand: the hand hides it there, so the span reaches at least the grip
            s_lo, s_hi = min(s.min(), sg), max(s.max(), sg)
            e_lo, e_hi = c + s_lo * d, c + s_hi * d
            far, nearp = (e_hi, e_lo) if s_hi - sg >= sg - s_lo else (e_lo, e_hi)
            obs[f] = dict(p=p, h=h, gc=gc[0], far=far, near=nearp, lfar=float(np.linalg.norm(far - [gu[0], gv[0]])),
                          lpx=float(s_hi - s_lo), r=thick / 2 * gc[0, 2] / f_img,
                          pts=[(c + (s.min() + q * np.ptp(s)) * d).tolist() for q in (0.15, 0.35, 0.5, 0.65, 0.85)])
        if len(obs) < 5:
            report[lab] = {"frames": len(obs), "kept": False}
            continue
        # one length per staff: seen lengths back-projected at the grip depth (a slant only shortens them)
        L = float(np.percentile([o["lpx"] * o["gc"][2] / f_img for o in obs.values()], 80))
        L = min(L, a.max_len)
        res = {}
        for f, o in obs.items():
            G = o["gc"]
            frac_far = np.clip(o["lfar"] / max(o["lpx"], 1e-6), 0.5, 1.0)
            s_far = L * frac_far
            ray = np.array([(o["far"][0] - W / 2) / f_img, (o["far"][1] - H / 2) / f_img, 1.0])
            ray /= np.linalg.norm(ray)
            b = ray @ G
            disc = b * b - (G @ G - s_far ** 2)
            ts_ = [b] if disc <= 0 else [b - np.sqrt(disc), b + np.sqrt(disc)]
            Xf = min((t * ray for t in ts_ if t > 0.1), key=lambda X: abs(X[2] - G[2]), default=b * ray)
            u = (Xf - G) / max(np.linalg.norm(Xf - G), 1e-6)
            Xn = G - u * (L - s_far)
            Rw = cam_R[f]
            res[f] = dict(p=o["p"], h=o["h"], dirw=u @ Rw.T, frac=float(frac_far), r=o["r"])
        # fill gaps with the same hand; grip from the moving hand on every frame
        fs = sorted(res)
        frames = []
        for i, f in enumerate(fs):
            frames.append((f, res[f]))
            if i + 1 < len(fs):
                f2 = fs[i + 1]
                o1, o2 = res[f], res[f2]
                if 1 < f2 - f <= a.gap and (o1["p"], o1["h"]) == (o2["p"], o2["h"]):
                    for k in range(f + 1, f2):
                        w = (k - f) / (f2 - f)
                        frames.append((k, dict(p=o1["p"], h=o1["h"], dirw=slerp_dir(o1["dirw"], o2["dirw"], w),
                                               frac=(1 - w) * o1["frac"] + w * o2["frac"], r=o1["r"])))
                elif a.gap < f2 - f <= a.hold and o1["p"] == o2["p"]:
                    # longer gap, same person: the staff stays in the hand and turns with it (motion blur hides it)
                    fwd = carried(o1, f, f2)
                    bwd = carried(dict(o2, h=o1["h"]), f2, f)
                    if fwd is None or bwd is None:
                        continue
                    for k in range(f + 1, f2):
                        w = (k - f) / (f2 - f)
                        d = slerp_dir(fwd[k] / np.linalg.norm(fwd[k]), bwd[k] / np.linalg.norm(bwd[k]), w)
                        frames.append((k, dict(p=o1["p"], h=o1["h"], dirw=d,
                                               frac=(1 - w) * o1["frac"] + w * o2["frac"], r=o1["r"])))
        frames.sort(key=lambda x: x[0])
        frames = extend_ends(frames, set(obs))
        frames = refine_by_image(frames, set(obs), L, int(num))
        frames = smooth_in_hand(frames, set(obs))
        staffs.append((len(obs), lab, L, frames, obs))
    # one staff per person: where two labels (e.g. a whole-frame mask and a hand-window mask of the same staff) are
    # held by the same person on a frame (either hand: staffs are held two-handed), the label seen on more frames keeps it
    claimed = set()
    for _, lab, L, frames, obs in sorted(staffs, key=lambda x: -x[0]):
        T, A, B = [], [], []
        for f, o in frames:
            if (f, o["p"]) in claimed:
                continue
            G = grip_at(o["p"], o["h"], f)
            if G is None:
                continue
            claimed.add((f, o["p"]))
            dw = o["dirw"] / np.linalg.norm(o["dirw"])
            T.append(f)
            B.append(G + dw * L * o["frac"])
            A.append(G - dw * L * (1 - o["frac"]))
        if len(T) < 5:
            report[lab] = {"seen_frames": len(obs), "kept": False, "reason": "holder taken by another label"}
            continue
        out[f"{lab}_t"] = np.array(T, np.int32)
        out[f"{lab}_a"] = np.array(A, np.float32)
        out[f"{lab}_b"] = np.array(B, np.float32)
        out[f"{lab}_r"] = np.float32(np.clip(np.median([o["r"] for o in obs.values()]), 0.015, 0.035))
        report[lab] = {"seen_frames": len(obs), "frames_with_fill": len(T), "first": int(min(T)), "last": int(max(T)),
                       "length_m": round(L, 2), "holders": sorted({f"{o['p']}:{o['h']}" for _, o in frames})}
    if a.seeds:
        # point seeds for a tracker: per kept label, held sightings spread over its frames (largest staff first)
        objs = {}
        for n_obs, lab, _, _, obs in sorted(staffs, key=lambda x: -x[0]):
            if f"{lab}_t" not in out:
                continue
            fs = sorted(obs)
            pick = sorted({fs[int(i)] for i in np.linspace(0, len(fs) - 1, min(a.seeds_per, len(fs)))})
            objs[lab] = [{"frame": int(f), "points": obs[f]["pts"]} for f in pick]
        a.seeds.write_text(json.dumps({"objects": objs}), encoding="utf-8")
    out["labels"] = np.array(json.dumps(sorted(k[:-2] for k in out if k.endswith("_t"))))
    np.savez_compressed(a.out, **out)
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
