"""Clothes for the SMPL bodies (after the smpl step), measured against the person masks of every cut, close-ups too.

    python scripts/multishot_dress.py IN.npz DIAG_DIR TAG OUT.npz

The SMPL surface is the bare body; people in films wear clothes. Per cut with s<k>_smpl and person masks, a small grid
of dressings (scene/mannequin.dressed: surface pushed out by an offset = cloth over the body; optionally a flared
skirt from the waist = long coat / dress) is drawn on up to N_FRAMES frames and the one with the best mean mask IoU
is kept, if it beats the bare body by MIN_GAIN. Writes s<k>_dress [offset_m, hem_radius_m, length_share] and
info["dress"]. The old s<k>_garment (mannequin coat) is the starting hint for the skirt length.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vrecon.scene.mannequin import dressed, mesh_silhouette  # noqa: E402

OFFSETS = (0.0, 0.015, 0.03, 0.05, 0.08)
SKIRTS = ((0.0, 0.0), (0.25, 1.0), (0.35, 1.0), (0.45, 1.0), (0.3, 0.7))
N_FRAMES = 6
MIN_GAIN = 0.01
GW, GH = 192, 80


def main() -> int:
    src, diag, tag, out = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], Path(sys.argv[4])
    z = dict(np.load(src))
    info = json.loads(str(z["info"]))
    faces = z["smpl_faces"]
    rep = {}
    for k, s in enumerate(info["shots"]):
        z[f"s{k}_dress"] = np.zeros(3)
        mp = diag / f"{tag}_shot{k:02d}.masks.npz"
        if f"s{k}_smpl" not in z or not mp.exists():
            continue
        V = z[f"s{k}_smpl"].astype(float)
        J = z[f"s{k}_body"]
        cam = z[f"s{k}_cam"]
        masks = np.load(mp)["masks"]
        T = min(len(V), len(masks), len(cam))
        frames = []
        for t in np.linspace(0, T - 1, min(N_FRAMES * 2, T), dtype=int):
            m = masks[t] > 0
            if m.any():
                frames.append((t, m[np.arange(GH) * m.shape[0] // GH][:, np.arange(GW) * m.shape[1] // GW]))
        frames = frames[:N_FRAMES]
        if not frames:
            continue
        res = {}
        for off in OFFSETS:
            for hem, length in SKIRTS:
                ious = []
                for t, m in frames:
                    v, f = dressed(V[t], faces, J[t], (off, hem, length))
                    sil = mesh_silhouette(v, f, cam[t], s["lens_35mm"], GW, GH)
                    ious.append((sil & m).sum() / max((sil | m).sum(), 1))
                res[(off, hem, length)] = float(np.mean(ious))
        bare = res[(0.0, 0.0, 0.0)]
        best = max(res, key=res.get)
        keep = best if res[best] > bare + MIN_GAIN else (0.0, 0.0, 0.0)
        z[f"s{k}_dress"] = np.array(keep)
        rep[k] = {"frames": len(frames), "bare_iou": round(bare, 3), "iou": round(res[keep], 3),
                  "offset_m": keep[0], "hem_radius_m": keep[1], "length_share": keep[2]}
        print(k, rep[k], flush=True)
    info["dress"] = rep
    z["info"] = json.dumps(info)
    np.savez_compressed(out, **z)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
