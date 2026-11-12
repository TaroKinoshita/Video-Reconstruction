"""How much completion changed the source view: object ids per pixel from the source camera, before and after.

    python scripts/source_view_change.py BEFORE.json AFTER.json [--frames 0,117,234]

Prints the share of pixels whose object changed, the share where an added or grown piece now covers something
that was built before (should be ~0: completion must not hide what the source saw), and the largest swaps.
Mesh files resolve against BEFORE's folder.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("before")
    ap.add_argument("after")
    ap.add_argument("--frames", default="0,117,234")
    a = ap.parse_args()
    from vrecon.scene.builder import build_scene
    from vrecon.scene.passes import render_passes
    from vrecon.schema import load_project

    root = Path(a.before).resolve().parent
    fr = [int(x) for x in a.frames.split(",")]
    res = []
    for path in (a.before, a.after):
        build_scene(load_project(path), root=root)
        with tempfile.TemporaryDirectory() as t:
            render_passes(t, shot_frames=fr, shaded=False)
            idx = np.load(Path(t) / "index.npz")["index"]
            m = {int(k): v for k, v in json.loads((Path(t) / "id_map.json").read_text(encoding="utf-8")).items()}
        res.append(np.vectorize(lambda k, m=m: m.get(int(k), ""))(idx))
    na, nb = res
    ch = na != nb
    out = {"changed": round(float(ch.mean()), 4), "covers_built": round(float((ch & (na != "")).mean()), 4),
           "swaps": [[x, y, c] for (x, y), c in Counter(zip(na[ch].tolist(), nb[ch].tolist())).most_common(6)]}
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
