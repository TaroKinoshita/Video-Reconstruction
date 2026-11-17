"""Is an object really standing on the floor where its mask ends? (scene-independent; pure numpy)

proxies.ground_box puts every masked object on the floor point under the bottom of its mask. That is wrong for
anything whose lower edge is hidden: a spectator seen above a wall has the wall top as the mask's lower edge, the
ray through that edge meets the floor in front of the wall, and the person is placed on the court side of it
(tennis obj_person_box_04, 2026-11-18: a second "man" beside the line judge in every generated view).

Evidence comes from the VIDEO, not from the scene: a floor mask of the source (SAM 3.1 with a generic prompt such
as "floor, ground", gpu/h3_control/comfy_sam.py). Per sampled frame the foot pixel is the lowest row of the
object's mask (median column); it stands on the floor when the floor mask covers the pixels just below it.
When most frames say it does not, the object is placed where its foot ray first meets a surface of the scene once
the floor pieces are taken out (walls, stands, furniture): the depth at the foot pixel of a render of the scene
without the floor and without the object. Its size is scaled with the distance so its image stays the same.
A first version (2026-11-18 morning) judged from the scene's own depth instead; it took the net in the line of
sight for the occluder and moved a spectator towards the camera. Rejected.
"""

from __future__ import annotations

import numpy as np

BELOW_PX = (2, 6)  # rows below the mask's lowest row that are looked at
MIN_FLOOR = 0.5  # share of those pixels that must be floor
MIN_VOTE = 0.6  # share of sampled frames that must say "not on the floor"
FLOOR_WORDS = ("floor", "ground", "court", "lines", "grass", "road", "terrain")


def foot_pixel(mask: np.ndarray) -> tuple[float, float] | None:
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return None
    y = ys.max()
    return float(np.median(xs[ys >= y - 1])), float(y)


def stands_on_floor(mask: np.ndarray, floor: np.ndarray) -> bool | None:
    """True / False for one frame; None when the mask is empty or ends at the image's bottom edge (feet cut off)."""
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return None
    y = ys.max()
    if y + BELOW_PX[1] >= mask.shape[0]:
        return None
    cols = xs[ys >= y - 1]
    rows = np.arange(y + BELOW_PX[0], y + BELOW_PX[1] + 1)
    under = floor[np.ix_(rows, np.unique(cols))]
    return bool(under.mean() >= MIN_FLOOR)


def check(mask_frames: dict[int, np.ndarray], floor_frames: dict[int, np.ndarray]) -> dict:
    votes = [stands_on_floor(m, floor_frames[f]) for f, m in mask_frames.items() if f in floor_frames]
    votes = [v for v in votes if v is not None]
    off = float(np.mean([not v for v in votes])) if votes else 0.0
    return {"on_floor": not (votes and off >= MIN_VOTE), "off_share": off, "frames": len(votes)}


def surface_depth(mask_frames: dict[int, np.ndarray], depth_frames: dict[int, np.ndarray]) -> float | None:
    """Median view-Z at the foot pixels in a render of the scene without the floor and without the object."""
    zs = []
    for f, m in mask_frames.items():
        fp = foot_pixel(m)
        if fp is None or f not in depth_frames:
            continue
        d = depth_frames[f]
        z = float(d[min(int(fp[1]), d.shape[0] - 1), int(round(fp[0]))])
        if z > 0:
            zs.append(z)
    return float(np.median(zs)) if zs else None


def is_floor_piece(obj) -> bool:
    return any(w in obj.id.lower() for w in FLOOR_WORDS)


__all__ = ["FLOOR_WORDS", "check", "foot_pixel", "is_floor_piece", "stands_on_floor", "surface_depth"]
