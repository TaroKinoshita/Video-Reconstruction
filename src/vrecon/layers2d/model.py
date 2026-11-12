"""Layer values over time and rendering. Pure numpy + cv2 (in memory only, no file functions)."""

from __future__ import annotations

import numpy as np

from vrecon.schema import Layer2D, LayerKey

CHANNELS = ("x", "y", "rotation", "scale", "opacity")


def _in_out(p):
    p = np.asarray(p, float)
    return np.where(p < 0.5, 4 * p ** 3, 1 - (-2 * p + 2) ** 3 / 2)


EASINGS = {
    "linear": lambda p: np.asarray(p, float),
    "ease_in": lambda p: np.asarray(p, float) ** 3,
    "ease_out": lambda p: 1 - (1 - np.asarray(p, float)) ** 3,
    "ease_in_out": _in_out,
    "hold": lambda p: np.zeros_like(np.asarray(p, float)),
}


def ease(name: str, p):
    return EASINGS[name](np.clip(p, 0.0, 1.0))


def layer_state(layer: Layer2D, t: float) -> dict[str, float]:
    keys = layer.keys
    if t <= keys[0].time:
        k = keys[0]
        return {c: float(getattr(k, c)) for c in CHANNELS}
    for a, b in zip(keys, keys[1:]):
        if t < b.time:
            p = float(ease(a.easing, (t - a.time) / max(b.time - a.time, 1e-9)))
            return {c: float(getattr(a, c) + (getattr(b, c) - getattr(a, c)) * p) for c in CHANNELS}
    k = keys[-1]
    return {c: float(getattr(k, c)) for c in CHANNELS}


def states(layer: Layer2D, n: int, fps: float) -> np.ndarray:
    """[n, 5] values per frame (x, y, rotation, scale, opacity)."""
    return np.array([[layer_state(layer, i / fps)[c] for c in CHANNELS] for i in range(n)])


def layer_matrix(size: tuple[float, float], s: dict[str, float] | np.ndarray) -> np.ndarray:
    """2x3 affine from sprite pixels to screen pixels."""
    import cv2

    if not isinstance(s, dict):
        s = dict(zip(CHANNELS, s))
    w, h = size
    m = cv2.getRotationMatrix2D((w / 2, h / 2), float(s["rotation"]), float(s["scale"]))
    m[0, 2] += s["x"] - w / 2
    m[1, 2] += s["y"] - h / 2
    return m


def render(layers: list[Layer2D], sprites: dict[str, np.ndarray], n: int, fps: float, size: tuple[int, int],
           background: np.ndarray | None = None, mask_alpha: float = 0.5, min_opacity: float = 0.15,
           frames: list[int] | None = None):
    """sprites: id -> float RGBA [h, w, 4] in 0..1 (only alpha is needed without a background).

    Returns (rgb uint8 [T,H,W,3] or None, index int16 [T,H,W]: 1 + position in `layers` of the front-most layer
    whose sprite alpha > mask_alpha and opacity >= min_opacity, 0 = none)."""
    import cv2

    w, h = size
    order = sorted(range(len(layers)), key=lambda i: layers[i].z)
    frames = list(range(n)) if frames is None else frames
    index = np.zeros((len(frames), h, w), np.int16)
    rgb = None if background is None else np.empty((len(frames), h, w, 3), np.uint8)
    for fi, f in enumerate(frames):
        img = None if background is None else background.astype(np.float32).copy()
        for i in order:
            lay = layers[i]
            spr = sprites[lay.id]
            s = layer_state(lay, f / fps)
            m = layer_matrix((spr.shape[1], spr.shape[0]), s)
            warped = cv2.warpAffine(spr, m, (w, h), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))
            a = warped[..., 3]
            if s["opacity"] >= min_opacity:
                index[fi][a > mask_alpha] = i + 1
            if img is not None:
                alpha = (a * s["opacity"])[..., None]
                img = img * (1 - alpha) + warped[..., :3] * 255.0 * alpha
        if img is not None:
            rgb[fi] = np.clip(img + 0.5, 0, 255).astype(np.uint8)
    return rgb, index


def key(t: float, x: float, y: float, rotation: float = 0.0, scale: float = 1.0, opacity: float = 1.0,
        easing: str = "linear") -> LayerKey:
    return LayerKey(time=t, x=x, y=y, rotation=rotation, scale=scale, opacity=opacity, easing=easing)
