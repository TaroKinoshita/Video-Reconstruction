"""Rough appearance colours (SceneObject.look) from the reference video. Pure numpy.

These stand in for the final look in "rough colour" renders; they are medians, so lighting, shading and texture
are averaged away. Identity colours for ID passes are something else (scene/color_passes.py) and are never estimated.
"""

from __future__ import annotations

import numpy as np

BODY_PARTS = ("head", "upper", "lower", "skin", "shoes")  # parts a character's `look` can name (besides "all")
SKIN, SHOES = (0.72, 0.55, 0.45), (0.92, 0.92, 0.9)  # assumed when the look does not name them


def clothed_part(joint: str) -> str:
    """Which look part a skinned body vertex takes from the joint that moves it most: forearms, hands and shins
    show skin, feet are shoes, the rest as body_part (shirt / shorts / head). A whole body in one measured colour
    reads as bare skin (and an image model's content checker refuses it)."""
    if any(k in joint for k in ("elbow", "wrist", "hand", "knee")):
        return "skin"
    if any(k in joint for k in ("ankle", "foot")):
        return "shoes"
    return body_part(joint)


PLAIN = {"upper": (0.55, 0.56, 0.58), "lower": (0.25, 0.26, 0.3)}  # plain shirt / shorts


def skin_like(c) -> bool:
    """A measured clothing colour that is itself skin-toned (warm, r > g > b, moderate saturation): the median
    then mostly sampled arms or legs (crossed arms over the torso, bare legs), not the clothes."""
    r, g, b = (float(x) for x in c)
    return r > g > b and 0.08 < r - b < 0.4 and r > 0.35 and (g - b) < (r - g) * 1.6 + 0.08


def part_colour(look: dict, part: str, default):
    if part in look:
        if part in PLAIN and skin_like(look[part]):
            return PLAIN[part]
        return look[part]
    if part == "skin":
        return SKIN
    if part == "shoes":
        return SHOES
    return look.get("all", default)


def body_part(bone: str) -> str:
    """Which look part a bone's cylinder belongs to: head, upper (torso and arms) or lower (hips and legs)."""
    if bone in ("head", "neck"):
        return "head"
    if any(k in bone for k in ("hip", "knee", "ankle", "foot")) or bone == "pelvis":
        return "lower"
    return "upper"


def _bone_samples(a: np.ndarray, b: np.ndarray, n: int = 7) -> np.ndarray:
    t = np.linspace(0.15, 0.85, n)[:, None]  # away from the joints, where limbs overlap
    return a + (b - a) * t


def character_look(video: np.ndarray, uv: np.ndarray, names: list[str], parents: list[int], frames=None,
                   head_up: float = 0.5) -> dict[str, tuple[float, float, float]]:
    """video uint8 RGB [T,H,W,3]; uv [T,J,2] joint pixels. Median colour per body part over the frames."""
    t_n, h, w = video.shape[:3]
    frames = range(t_n) if frames is None else frames
    acc: dict[str, list] = {}
    for f in frames:
        for j, p in enumerate(parents):
            if p < 0:
                continue
            pts = _bone_samples(uv[f, p], uv[f, j])
            if names[j] == "head":  # the face/hair is above the head joint
                pts = np.vstack([pts, uv[f, j] + (uv[f, j] - uv[f, p]) * head_up])
            q = np.rint(pts).astype(int)
            ok = (q[:, 0] >= 0) & (q[:, 0] < w) & (q[:, 1] >= 0) & (q[:, 1] < h)
            acc.setdefault(body_part(names[p]) if names[j] != "head" else "head", []).append(video[f][q[ok, 1], q[ok, 0]])
    return {k: tuple(float(c) for c in np.median(np.concatenate(v), 0) / 255.0) for k, v in acc.items() if v}


def region_look(video: np.ndarray, mask: np.ndarray, frames=None) -> tuple[float, float, float]:
    """Median colour of the pixels under a mask [H,W] or [T,H,W] over frames."""
    frames = range(len(video)) if frames is None else list(frames)
    if mask.ndim == 2:
        px = np.concatenate([video[f][mask] for f in frames])
    else:
        px = np.concatenate([video[f][mask[f]] for f in frames])
    return tuple(float(c) for c in np.median(px, 0) / 255.0)


__all__ = ["BODY_PARTS", "body_part", "character_look", "region_look"]
