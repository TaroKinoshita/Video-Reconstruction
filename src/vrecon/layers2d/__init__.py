"""2D layers (text, shapes, logos drawn over the picture): evaluation, rendering, estimation from video.

    model     layer_state (keys + easing -> values at a time), render (sprites -> frames, alpha, index)
    estimate  per-frame tracks from masks + LK, then keyframes with easing (fit_keys)
    refine    pattern search on the keys against the masks (silhouette IoU), kept only if better

Screen convention (schema.Layer2D): pixels of the source frame, origin top-left, y down, rotation in degrees
counter-clockwise as seen, centre of the sprite at (x, y).
"""

from .model import EASINGS, ease, layer_matrix, layer_state, render, states

__all__ = ["EASINGS", "ease", "layer_matrix", "layer_state", "render", "states"]
