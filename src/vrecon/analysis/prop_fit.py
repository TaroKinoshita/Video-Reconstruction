"""Props into a scene from masks, for any prop in analysis/prop_table.py (docs/PROPS_PLAN.md).

Built from the tools in analysis/props.py; no bpy, no ground truth. Steps:
    contact_points   where the scene draws each contact joint (rig FK) plus the prop's reach along the bone
    fit_stick        a stick prop held by the contact nearest its mask: pose per frame from the silhouette;
                     frames without a mask keep the nearest fitted pose relative to the hand (marked)
    tracklets        blobs of a sphere's mask linked frame to frame (several alike objects come apart)
    fit_sphere_track held (by a contact, until the release) / free (ballistic from the release, the contact's
                     position and velocity there as priors) per tracklet; a tracklet never by a contact is another
                     object alike and is left out
    carry_on         unseen after a free part: physics until a stick's strike point meets the path (or, when it
                     never does, the stick's highest strike point in the next 1.5 s, a hypothesis), then a
                     hypothesis for where it goes (target and speed from the caller: the rules of the activity)
    fill_unseen      other unseen frames: with the contact that held it most (a hypothesis)
Every frame keeps a note of how its pose was found.
"""

from __future__ import annotations

import cv2
import numpy as np

from vrecon.analysis import props as P

G = 9.81


def project_pts(pts, cam):
    R, c, K = cam
    pc = (np.atleast_2d(pts) - c) @ R
    return pc[:, :2] / pc[:, 2:3] * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]


def contact_points(person, skeleton: dict, joints: list[str], reach: float) -> tuple[dict, float]:
    """{joint: [T,3]} in scene units: the joint where the scene draws it, moved `reach` metres on along its bone
    (from the parent joint), and the scene's units per metre."""
    from vrecon.analysis.rig_fk import posed_joints

    s = float(person.rest.scale[0])
    names = list(skeleton["joint_names"])
    parents = [int(p) for p in skeleton["parents"]]
    pj = posed_joints(person, skeleton)
    out = {}
    for j in joints:
        i = names.index(j)
        bone = pj[:, i] - pj[:, parents[i]]
        bone /= np.linalg.norm(bone, axis=1, keepdims=True) + 1e-9
        out[j] = pj[:, i] + bone * reach * s
    return out, s


def nearest_contact(mask_frames: np.ndarray, contacts: dict, cams: list, min_px: int = 8):
    near = {k: [] for k in contacts}
    for f in range(len(mask_frames)):
        if mask_frames[f].sum() < min_px:
            continue
        ys, xs = np.nonzero(mask_frames[f])
        pts = np.c_[xs, ys]
        for k, h in contacts.items():
            near[k].append(float(np.min(np.linalg.norm(pts - project_pts(h[f], cams[f]), axis=1))))
    best = min(near, key=lambda k: np.median(near[k]) if near[k] else 1e9)
    return best, {k: (round(float(np.median(v)), 1) if v else None) for k, v in near.items()}


def smooth_dirs(v: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Unit vectors [T,3] (NaN = missing) smoothed over time with a Gaussian (sigma in frames), renormalised.
    Each frame's own pose is fitted alone, so its small errors shake from frame to frame; a light Gaussian
    removes the shake and keeps a fast swing (200 degrees in 6 frames moves little under sigma 1)."""
    if sigma <= 0:
        return v.copy()
    k = np.exp(-0.5 * (np.arange(-3, 4) / sigma) ** 2)
    out = np.full_like(v, np.nan)
    ok = np.all(np.isfinite(v), 1)
    for f in range(len(v)):
        if not ok[f]:
            continue
        acc, wsum = np.zeros(3), 0.0
        for j, w in zip(range(f - 3, f + 4), k):
            if 0 <= j < len(v) and ok[j]:
                acc += w * v[j] * np.sign(v[j] @ v[f] + 1e-9)
                wsum += w
        out[f] = acc / (np.linalg.norm(acc) + 1e-9)
    return out


def fit_stick(masks: np.ndarray, contacts: dict, cams: list, entry: dict, s: float, size, shutter: float = 0.5,
              smooth: float = 1.5):
    """Returns (poses [(centre, R)] per frame, how per frame, report)."""
    hand, dists = nearest_contact(masks, contacts, cams)
    grips = contacts[hand]
    L, W, T = (x * s for x in entry["size"])
    tips, sides, how, score = P.held_poses(masks, grips, cams, L, W, T, shutter=shutter, size=size)
    dirs = smooth_dirs((tips - grips) / np.linalg.norm(tips - grips, axis=1, keepdims=True), smooth)
    sides = smooth_dirs(sides, smooth)
    tips = grips + dirs * L
    seen = [f for f in range(len(masks)) if np.all(np.isfinite(tips[f]))]
    poses, notes = [], []
    for f in range(len(masks)):
        if not seen:
            break
        before = max([x for x in seen if x <= f], default=None)
        after = min([x for x in seen if x >= f], default=None)
        if before is not None and after is not None and before != after:
            # unseen between two fitted frames: turn smoothly from one to the other (a frozen pose then a snap
            # reads as a jerk)
            u = (f - before) / (after - before)
            d = P._slerp_dir(dirs[before], dirs[after], u)
            sd = P._slerp_dir(sides[before], sides[after], u)
            note = f"turned between frames {before} and {after} (hypothesis)"
        else:
            g = before if before is not None else after
            d, sd = dirs[g], sides[g]
            note = how[f] if g == f else f"held from frame {g} (hypothesis)"
        d = d / np.linalg.norm(d)
        sd = sd - (sd @ d) * d
        sd /= np.linalg.norm(sd) + 1e-9
        poses.append((grips[f] + d * L / 2, np.c_[d, sd, np.cross(d, sd)]))
        notes.append(note)
    rep = {"contact": hand, "median_px_to_mask": dists,
           "frames": {k: int((how == k).sum()) for k in ("sharp", "swept", "blurred", "unseen")},
           "iou_median": round(float(np.median(score[how != "unseen"])), 3) if (how != "unseen").any() else None}
    return poses, notes, rep


def strike_points(poses, entry: dict, s: float) -> np.ndarray:
    """[T,3] the point along a stick where it strikes (strike_frac from the held end)."""
    L = entry["size"][0] * s
    return np.array([c + R[:, 0] * L * (entry.get("strike_frac", 0.75) - 0.5) for c, R in poses])


def tracklets(mask: np.ndarray, link: float = 60.0, gap: int = 3, min_track: int = 5, min_area: int = 4):
    """[{frame: (centre px, width px)}]: blobs linked to the nearest blob of the last frame within `link` px."""
    tracks, live = [], []
    for f in range(len(mask)):
        n, cc, stats, _ = cv2.connectedComponentsWithStats(mask[f].astype(np.uint8))
        blobs = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area]
        info = {}
        for i in blobs:
            st = P.streak(cc == i)
            info[i] = (np.array(st["centre"]), st["width"])
        used, nxt = set(), []
        for tr in live:
            if f - max(tr) > gap:
                continue
            last = tr[max(tr)][0]
            cand = [(np.linalg.norm(info[i][0] - last), i) for i in blobs if i not in used]
            if cand:
                dist, i = min(cand)
                if dist < link:
                    tr[f] = info[i]
                    used.add(i)
            nxt.append(tr)
        for i in blobs:
            if i not in used:
                tr = {f: info[i]}
                tracks.append(tr)
                nxt.append(tr)
        live = nxt
    return sorted([t for t in tracks if len(t) >= min_track], key=lambda t: min(t))


def fit_sphere_track(tr: dict, t_n: int, contacts: dict, cams: list, times, fps: float, diameter: float, s: float,
                     hand_px: float = 35.0, max_rms: float = 12.0, min_held: int = 3):
    """One tracklet -> (pos [T,3] NaN where not placed, held {frame: contact}, segments).

    States: held where the blob is by a contact (within hand_px; runs shorter than min_held frames inside a free
    part are the ball passing a hand, not a catch); free in between. Each free part is fitted on its own
    (still / ballistic) with the contact's position as depth anchors at both ends (the release and the catch) and
    the releasing contact's velocity as a prior."""
    cen = np.full((t_n, 2), np.nan)
    wid = np.full(t_n, np.nan)
    for f, (c, w) in tr.items():
        cen[f], wid[f] = c, w
    fs = sorted(tr)
    near, dist = {}, {}
    for f in fs:
        k = min(contacts, key=lambda k_: np.linalg.norm(project_pts(contacts[k_][f], cams[f])[0] - cen[f]))
        dist[f] = float(np.linalg.norm(project_pts(contacts[k][f], cams[f])[0] - cen[f]))
        near[f] = k
    lab = np.array([dist[f] < hand_px for f in fs])
    # runs; short held runs between free parts -> free; the blob reaching a little past the hand inside a held run
    # (up to 2 hand_px, one or two frames) stays held
    runs, i = [], 0
    while i < len(fs):
        j = i
        while j < len(fs) and lab[j] == lab[i]:
            j += 1
        runs.append([i, j, bool(lab[i])])
        i = j
    for r_ in runs:
        if r_[2] and r_[1] - r_[0] < min_held and 0 < r_[0] and r_[1] < len(fs):
            r_[2] = False
        if not r_[2] and r_[1] - r_[0] <= 2 and 0 < r_[0] and r_[1] < len(fs) and                 all(dist[fs[x]] < 2 * hand_px for x in range(r_[0], r_[1])):
            r_[2] = True
    merged = []
    for r_ in runs:
        if merged and merged[-1][2] == r_[2]:
            merged[-1][1] = r_[1]
        else:
            merged.append(list(r_))
    pos = np.full((t_n, 3), np.nan)
    held = {}
    segs = []
    for i0, i1, is_held in merged:
        if not is_held:
            continue
        for x in range(i0, i1):
            f = fs[x]
            R, c, K = cams[f]
            ray = P.pixel_ray(cen[f], R, c, K)
            pos[f] = c + ray * float((contacts[near[f]][f] - c) @ ray)
            held[f] = near[f]
        segs.append({"frames": [fs[i0], fs[i1 - 1]], "model": "held", "contacts": sorted({near[fs[x]] for x in range(i0, i1)}),
                     "held_frames": i1 - i0, "rms_px": 0.0, "kept": True})
    for n_, (i0, i1, is_held) in enumerate(merged):
        if is_held:
            continue
        free = [fs[x] for x in range(i0, i1)]
        rel = fs[i0 - 1] if i0 > 0 else None    # last held frame before (the release)
        cat = fs[i1] if i1 < len(fs) else None  # first held frame after (the catch)
        free_cen = np.full((t_n, 2), np.nan)
        anchors, vpriors = {}, {}
        for f in free:
            free_cen[f] = cen[f]
        for f in (rel, cat):
            if f is not None and f in held:
                free_cen[f] = cen[f]
                R, c, K = cams[f]
                anchors[f] = (float((pos[f] - c) @ P.pixel_ray(cen[f], R, c, K)), 0.1 * s)
        if rel is not None and rel in held:
            k = held[rel]
            f0, f1 = max(rel - 2, 0), min(rel + 1, t_n - 1)
            vpriors[rel] = ((contacts[k][f1] - contacts[k][f0]) / max(times[f1] - times[f0], 1e-3), 3.0 * s)
        fpos, _, fsegs = P.ball_fit(free_cen, cams, fps, widths=wid, diameter=diameter, anchors=anchors,
                                    models=("still", "ballistic"), splits=[], times=times, vpriors=vpriors)
        for sg in fsegs:
            lo, hi = sg["frames"]
            sg["kept"] = bool(sg["rms_px"] <= max_rms)
            sg["release"], sg["catch"] = rel, cat
            if sg["kept"]:
                for f in range(lo, hi + 1):
                    if f not in held:
                        pos[f] = fpos[f]
                if sg["model"] == "still":
                    # put down: the hand lingers by it after letting go; frames before where the ball already sits
                    # at its resting pixel are resting too (the release is the first of them)
                    first = next(f for f in range(lo, hi + 1) if np.all(np.isfinite(cen[f])))
                    f = first - 1
                    while f in held and np.linalg.norm(cen[f] - cen[first]) < 2.0:
                        held.pop(f)
                        pos[f] = fpos[first]
                        f -= 1
                    sg["release"] = f
                    sg["frames"] = [f + 1, hi]
            segs.append(sg)
    for sg in segs:  # held ranges after the put-down correction
        if sg["model"] == "held":
            hs = [f for f in held if sg["frames"][0] <= f <= sg["frames"][1]]
            if hs:
                sg["frames"], sg["held_frames"] = [min(hs), max(hs)], len(hs)
    return pos, held, segs


def carry_on(pos: np.ndarray, segs: list, times, fps: float, strikes: np.ndarray | None, s: float, radius: float,
             hit_m: float = 0.35, after: dict | None = None):
    """Unseen frames after each free part: physics on to a strike, then `after` (hypothesis: {"target": xyz
    metres, "speed": m/s, "stop_abs_y": metres, "restitution": 0.7}). Changes pos in place; returns notes."""
    t_n = len(pos)
    notes = []
    for sg in [x for x in segs if x.get("model") == "ballistic" and x.get("kept") and "fit" in x]:
        prm = {**sg["fit"], "p0": np.array(sg["fit"]["p0"]), "v": np.array(sg["fit"]["v_exact"])}
        last = sg["frames"][1]
        f, hit, how = last + 1, None, None
        while f < t_n and not np.all(np.isfinite(pos[f])):
            p = P.trajectory_at(prm, times[f])
            if strikes is not None and np.linalg.norm(p - strikes[f]) < hit_m * s:
                hit, how = f, "strike point within hit_m of the path"
                break
            if p[2] < radius or times[f] - prm["t0"] > 2.0:
                break
            pos[f] = p
            f += 1
        if hit is None and strikes is not None:
            # the free part's depth and the stick's pose mid-swing are both uncertain: take the highest strike
            # point in the next 1.5 s as the contact and bend the unseen arc to meet it (hypothesis)
            lo, hi = last + 1, min(last + int(1.5 * fps), t_n - 1)
            hit = lo + int(np.argmax(strikes[lo:hi + 1, 2]))
            dt = times[hit] - times[last]
            v0 = (strikes[hit] - pos[last] + np.array([0, 0, G * s * dt * dt / 2])) / max(dt, 1e-3)
            arc = {"t0": times[last], "p0": pos[last], "v": v0, "model": "ballistic", "g": G * s}
            for g in range(lo, t_n):
                pos[g] = P.trajectory_at(arc, times[g]) if g < hit else np.nan
            how = "no strike found; the stick's highest strike point within 1.5 s (hypothesis)"
        end = hit if hit is not None else f
        notes.append({"frames": [last + 1, end - 1], "model": "unseen: physics on" + (f" to the strike ({how})" if how else ""),
                      "hypothesis": True})
        if hit is None or not after:
            continue
        p_hit = strikes[hit]
        tgt = np.asarray(after["target"], float) * s
        tof = np.linalg.norm(tgt - p_hit) / (after["speed"] * s)
        v0 = (tgt - p_hit + np.array([0, 0, G * s * tof * tof / 2])) / tof
        pr = {"t0": times[hit], "p0": p_hit, "v": v0, "model": "ballistic", "g": G * s}
        rest = None
        e = after.get("restitution", 0.7)
        stop = after.get("stop_abs_y")
        for g in range(hit, t_n):
            if rest is not None:
                pos[g] = rest
                continue
            q = P.trajectory_at(pr, times[g])
            if stop is not None and abs(q[1]) > stop * s:
                rest = np.r_[q[0], np.sign(q[1]) * stop * s, radius]
                q = rest
            elif q[2] < radius:
                vb = pr["v"] - np.array([0, 0, pr["g"] * (times[g] - pr["t0"])])
                pr = {"t0": times[g], "p0": np.r_[q[:2], radius], "v": vb * [0.9, 0.9, -e], "model": "ballistic", "g": G * s}
                q = pr["p0"]
            pos[g] = q
        notes.append({"frames": [hit, t_n - 1], "model": f"after the strike: towards {after['target']} m at "
                      f"{after['speed']} m/s (hypothesis: {after.get('why', 'given')})", "hypothesis": True,
                      "strike_frame": int(hit)})
    return notes


def fill_unseen(pos: np.ndarray, contacts: dict, held_counts: dict) -> list:
    if not held_counts:
        return []
    k = max(held_counts, key=held_counts.get)
    miss = [f for f in range(len(pos)) if not np.all(np.isfinite(pos[f]))]
    for f in miss:
        pos[f] = contacts[k][f]
    return [{"frames": [min(miss), max(miss)] if miss else [0, -1], "model": f"other unseen frames: with {k} (hypothesis)",
             "hypothesis": True, "count": len(miss)}]


__all__ = ["carry_on", "contact_points", "fill_unseen", "fit_sphere_track", "fit_stick", "nearest_contact", "smooth_dirs",
           "project_pts", "strike_points", "tracklets"]
