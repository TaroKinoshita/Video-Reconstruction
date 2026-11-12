"""Repair loop for 2D layers: nudge each key (position, turn, size, time) and keep a change only if the layers'
silhouettes match the masks better (mean IoU over labels and frames). Same accept rule as the 3D repair loop.
Opacity is not searched: the masks only say whether a layer is shown (opacity >= HIDDEN_OPACITY)."""

from __future__ import annotations

import numpy as np

from vrecon.schema import Layer2D

from .model import render

STEPS = {"x": 2.0, "y": 2.0, "rotation": 2.0, "scale": 0.02, "time": 1.0}  # time in frames


def silhouette_loss(layers: list[Layer2D], sprites, index: np.ndarray, id_map: dict[int, str], fps: float,
                    stride: int = 2) -> float:
    n, h, w = index.shape
    frames = list(range(0, n, stride))
    _, rec = render(layers, sprites, n, fps, (w, h), frames=frames)
    lab = {v: k for k, v in id_map.items()}
    ious = []
    for i, lay in enumerate(layers):
        obs = index[frames] == lab[lay.id]
        got = rec == i + 1
        inter = (obs & got).reshape(len(frames), -1).sum(1)
        union = (obs | got).reshape(len(frames), -1).sum(1)
        ok = union > 0
        if ok.any():
            ious.append(float((inter[ok] / union[ok]).mean()))
    return 1.0 - float(np.mean(ious)) if ious else 1.0


def _nudge(lay: Layer2D, k: int, field: str, delta: float, fps: float) -> Layer2D | None:
    keys = list(lay.keys)
    key = keys[k]
    if field == "time":
        t = key.time + delta / fps
        if (k > 0 and t <= keys[k - 1].time + 1e-9) or (k + 1 < len(keys) and t >= keys[k + 1].time - 1e-9):
            return None
        keys[k] = key.model_copy(update={"time": t})
    else:
        val = getattr(key, field) + delta
        if field == "scale" and val <= 0.05:
            return None
        keys[k] = key.model_copy(update={field: val})
    return lay.model_copy(update={"keys": keys})


def refine_layers(layers: list[Layer2D], sprites, index, id_map, fps: float, rounds: int = 3, max_evals: int = 1500):
    """Coordinate pattern search over every key of every layer. Returns (layers, info)."""
    cur = list(layers)
    best = silhouette_loss(cur, sprites, index, id_map, fps)
    start, evals, accepted = best, 1, 0
    scale = 1.0
    for _ in range(rounds):
        improved = False
        for li in range(len(cur)):
            for k in range(len(cur[li].keys)):
                for field, step in STEPS.items():
                    for sign in (1, -1):
                        if evals >= max_evals:
                            break
                        cand = _nudge(cur[li], k, field, sign * step * scale, fps)
                        if cand is None:
                            continue
                        trial = cur[:li] + [cand] + cur[li + 1:]
                        loss = silhouette_loss(trial, sprites, index, id_map, fps)
                        evals += 1
                        if loss < best - 1e-6:
                            cur, best, accepted, improved = trial, loss, accepted + 1, True
                            break
        if not improved:
            scale /= 2
    return cur, {"loss_before": start, "loss_after": best, "evaluations": evals, "accepted": accepted}


__all__ = ["STEPS", "refine_layers", "silhouette_loss"]
