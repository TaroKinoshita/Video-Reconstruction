"""Loose clothing below the waist (long coats, dresses, skirts) from the person masks.

    python scripts/multishot_clothing.py IN.npz DIAG_DIR TAG OUT.npz

Per cut with a GVHMR body in the shared world: the bare mannequin's silhouette (thick sticks,
multishot_closeups.silhouette) is compared with the person mask in the band between the projected pelvis and
ankles, on up to 20 frames. When the mask there covers clearly more (median area ratio > MIN_RATIO), the body gets a
flared garment from the waist down to where the mask ends: its hem radius grows with the ratio, its length is the
median share of the waist-to-ankle span the mask still covers. Writes s<k>_garment [hem_radius_m, length_share]
(0 = none) and info["clothing"].
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from multishot_closeups import silhouette  # noqa: E402
from multishot_people import project  # noqa: E402

MIN_RATIO = 1.5
GW, GH = 320, 136


def main() -> int:
    src, diag, tag, out = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], Path(sys.argv[4])
    z = dict(np.load(src))
    info = json.loads(str(z["info"]))
    W, H = int(z["W"]), int(z["H"])
    rep = {}
    for k, s in enumerate(info["shots"]):
        z[f"s{k}_garment"] = np.zeros(2)
        mp = diag / f"{tag}_shot{k:02d}.masks.npz"
        body = z[f"s{k}_body"]
        if s.get("slot") != 0 or s.get("points_only_during_cut") or not len(body) or not mp.exists():
            continue
        masks = np.load(mp)["masks"]
        cam = z[f"s{k}_cam"]
        lens = s["lens_35mm"]
        f = lens / 36.0 * GW
        ratios, lengths = [], []
        T = min(len(masks), len(body))
        for t in np.linspace(0, T - 1, min(20, T), dtype=int):
            m = masks[t] > 0
            m = m[np.arange(GH) * m.shape[0] // GH][:, np.arange(GW) * m.shape[1] // GW]
            if not m.any():
                continue
            uv = project(body[t][[0, 7, 8]], cam[t], f, GW, GH)
            top, bot = int(uv[0, 1]), int(max(uv[1, 1], uv[2, 1]))
            if bot - top < 6 or top < 0 or bot > GH:
                continue  # the band is not in the picture (close framing, or upside down)
            sil = silhouette(body[t], cam[t], lens, GW, GH)
            band = slice(top, bot)
            a_sil = sil[band].sum()
            if a_sil < 20:
                continue
            ratios.append(m[band].sum() / a_sil)
            rows = np.flatnonzero(m[band].any(1))
            lengths.append((rows.max() + 1) / (bot - top) if len(rows) else 0)
        r = {"frames": len(ratios), "area_ratio": float(np.median(ratios)) if ratios else None}
        if ratios and np.median(ratios) > MIN_RATIO:
            hem = float(np.clip(0.12 * np.median(ratios), 0.2, 0.6))
            length = float(np.clip(np.median(lengths), 0.4, 1.0))
            z[f"s{k}_garment"] = np.array([hem, length])
            r.update({"garment": True, "hem_radius_m": hem, "length_share": length})
        else:
            r["garment"] = False
        rep[k] = r
    info["clothing"] = rep
    z["info"] = json.dumps(info)
    np.savez_compressed(out, **z)
    print(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
