"""Fit a venue layout's numbers to the observations (analysis/observations.py). Pure python/numpy.

Only parts with "fit": LABEL move, and only along what the observation measures well:
    stand   shifted along its outward normal so that its step profile passes through the observed centroid
            (centroid height -> row -> distance behind the front); rows added if the observed top is higher
    wall    shifted along its outward normal onto the observed centroid; height raised to the observed top
    box     centre moved to the observed centroid; size kept unless "fit_size": true
Shifts are clipped to +-2 m: the layout's own numbers (from the picture and the rules) are trusted over far
4RC depths, which stood a stand 3 m in front of its own sign wall on tennis.
Line directions from the observations are not used: far away, 4RC's depth error turns them (tennis stands).
Standard parts (court, net) never move: the court calibration fixed them. Each change is recorded.
"""

from __future__ import annotations

import copy
import math

import numpy as np


def _mean_normal(path):
    from .generate import _outward

    ns = [_outward(a, b) * np.linalg.norm(np.subtract(b, a)) for a, b in zip(path[:-1], path[1:])]
    n = np.sum(ns, 0)
    return n / np.linalg.norm(n)


MAX_SHIFT = 2.0  # metres: far observations carry 4RC depth errors of metres (tennis: net at +1.9 m, stands 3 m short)


def fit_layout(layout: dict, obs: dict, max_shift: float = MAX_SHIFT) -> tuple[dict, list[dict]]:
    lay = copy.deepcopy(layout)
    log = []
    things = obs.get("things", {})
    for p in lay["parts"]:
        lab = p.get("fit")
        if not lab:
            continue
        o = things.get(lab)
        if o is None:
            log.append({"part": p["id"], "fit": lab, "skipped": "no such observation"})
            continue
        c = np.array(o["centroid"], float)
        if p["type"] in ("stand", "wall"):
            key = "front" if p["type"] == "stand" else "line"
            path = np.array(p[key], float)
            n = _mean_normal(path)
            mid = path.mean(0)
            if p["type"] == "stand":
                row = max(0.0, (c[2] - p["base_height"]) / p["rise"] - 0.5)
                behind = row * p["row_depth"]
                shift = float((c[:2] - mid) @ n - behind)
                top = o["z_range"][1]
                need = math.ceil(max(0.0, top - p["base_height"]) / p["rise"])
                if need > p["rows"]:
                    log.append({"part": p["id"], "rows": [p["rows"], need]})
                    p["rows"] = need
            else:
                shift = float((c[:2] - mid) @ n)
                if o["z_range"][1] > p["height"]:
                    log.append({"part": p["id"], "height": [p["height"], o["z_range"][1]]})
                    p["height"] = float(o["z_range"][1])
            wanted = shift
            shift = float(np.clip(shift, -max_shift, max_shift))
            p[key] = (path + shift * n).round(3).tolist()
            log.append({"part": p["id"], "fit": lab, "shift_m": round(shift, 3),
                        **({"wanted_m": round(wanted, 3), "clipped": True} if wanted != shift else {})})
        elif p["type"] == "box":
            old = list(p["centre"])
            p["centre"] = [round(float(c[0]), 3), round(float(c[1]), 3)]
            if p.get("fit_size"):
                ext = np.subtract(o["extent_max"], o["extent_min"])
                p["size"] = [round(float(max(ext[0], 0.2)), 3), round(float(max(ext[1], 0.2)), 3), round(float(max(o["z_range"][1], 0.3)), 3)]
            log.append({"part": p["id"], "fit": lab, "centre": [old, p["centre"]]})
    return lay, log


__all__ = ["fit_layout"]
