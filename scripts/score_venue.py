"""Score a candidate venue against the synthetic ground truth on the fixed cameras (evaluation/venue/cameras.json).

    python scripts/score_venue.py CANDIDATE.json [--root DIR] [--clip evaluation/venue/ven_01] [--out SCORES.json]

Renders the candidate's index and depth passes from each fixed camera (gt_passes/<camera>/camera.json) and scores
them with metrics/venue.py. The source camera also gets observed_kept. Mesh files resolve against --root (default:
the candidate's folder). Prints one line per camera and a summary (means over the new cameras).
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("candidate")
    ap.add_argument("--root", default=None)
    ap.add_argument("--clip", default="evaluation/venue/ven_01")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    from vrecon.metrics.venue import category, observed_kept, score_view
    from vrecon.scene.builder import build_scene
    from vrecon.scene.passes import render_passes
    from vrecon.schema import Camera, load_project

    clip = Path(a.clip)
    cand = load_project(a.candidate)
    root = Path(a.root).resolve() if a.root else Path(a.candidate).resolve().parent
    meta = json.loads((clip / "meta.json").read_text(encoding="utf-8"))
    observed = {category(k) for k, v in meta["visible_in_source_px"].items() if v > 500}
    res = {}
    for cd in sorted((clip / "gt_passes").iterdir()):
        info = json.loads((cd / "camera.json").read_text(encoding="utf-8"))
        q = cand.model_copy(deep=True)
        q.reconstruction.cameras = [Camera.model_validate(info["camera"])]
        build_scene(q, root=root)
        with tempfile.TemporaryDirectory() as t:
            render_passes(t, shot_frames=[0], shaded=False)
            ri, rd = np.load(Path(t) / "index.npz")["index"][0], np.load(Path(t) / "depth.npz")["depth"][0]
            rm = {int(k): v for k, v in json.loads((Path(t) / "id_map.json").read_text(encoding="utf-8")).items()}
        gi, gd = np.load(cd / "index.npz")["index"][0], np.load(cd / "depth.npz")["depth"][0]
        gm = {int(k): v for k, v in json.loads((cd / "id_map.json").read_text(encoding="utf-8")).items()}
        s = score_view(gi, gm, gd, ri, rm, rd)
        if cd.name == "source":
            s["observed_kept"] = observed_kept(gi, gm, ri, rm, observed)
        res[cd.name] = s
        print(cd.name, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items() if k != "category_iou"}),
              {k: round(v, 2) for k, v in s["category_iou"].items()}, flush=True)
    new = [v for k, v in res.items() if k != "source"]
    summary = {k: float(np.mean([v[k] for v in new if v[k] is not None])) for k in ("covered", "extra", "same_category", "depth_rel")}
    summary["stands_iou_new"] = float(np.mean([v["category_iou"].get("stands", 0.0) for v in new]))
    summary["source_observed_kept"] = res["source"]["observed_kept"]
    res["summary"] = summary
    print("summary", json.dumps({k: round(v, 3) for k, v in summary.items()}))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
