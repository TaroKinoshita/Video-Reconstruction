"""Rough appearance colours (SceneObject.look) from the reference video. Pure numpy.

These stand in for the final look in "rough colour" renders; they are medians, so lighting, shading and texture
are averaged away. Identity colours for ID passes are something else (scene/color_passes.py) and are never estimated.
"""

from __future__ import annotations

import numpy as np

BODY_PARTS = ("head", "upper", "lower")  # parts a character's `look` can name (besides "all")


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
