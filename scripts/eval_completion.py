"""Completion ablation on the synthetic clips (plan steps 5-6): novel-view scores without and with rule-based
completion, on a given reconstruction per clip.

    python scripts/eval_completion.py OUT.json [--recs runs/llm_c4/syn_{c}/snapshots/LAST]

The new cameras are the clip's novel views carried into the reconstruction's world (metrics/novel.carry_relative):
they stand for the camera move a user asks for, which is known when editing; nothing else from ground truth is
used to complete.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--clips", default="01,02,03,04,05,06")
    a = ap.parse_args()
    from vrecon.evaluate import score, score_novel
    from vrecon.metrics.novel import carry_relative
    from vrecon.run.complete import extend_set_to_view
    from vrecon.schema import Camera, load_project

    rows = {}
    for c in a.clips.split(","):
        clip = Path(f"evaluation/synthetic/syn_{c}")
        rec = load_project(sorted(glob.glob(f"runs/llm_c4/syn_{c}/snapshots/*.json"))[-1])
        gt = load_project(clip / "gt.json")
        pairs = score(clip, rec)["pairs"]
        before = score_novel(clip, rec, pairs=pairs)
        done = rec
        records = []
        for view in sorted((clip / "gt_passes" / "novel").glob("*")):
            info = json.loads((view / "camera.json").read_text(encoding="utf-8"))
            cam = carry_relative(Camera.model_validate(info["camera"]), gt.reconstruction.cameras[0], rec.reconstruction.cameras[0])
            for o in [o for o in done.reconstruction.objects if o.kind == "set"]:
                done, r = extend_set_to_view(done, o.id, cam, frames=info["frames"])
                records.append({"view": view.name, **r})
        after = score_novel(clip, done, pairs=pairs)
        rows[f"syn_{c}"] = {"before": before["summary"], "after": after["summary"],
                            "before_set": {k: v["iou_set"] for k, v in before["views"].items()},
                            "after_set": {k: v["iou_set"] for k, v in after["views"].items()}, "ops": records}
        b, f = before["summary"], after["summary"]
        print(f"syn_{c} movers {b['iou_movers_mean']:.3f}->{f['iou_movers_mean']:.3f} covered {b['covered_mean']:.2f}->{f['covered_mean']:.2f} "
              f"extra {b['extra_mean']:.2f}->{f['extra_mean']:.2f}", flush=True)
    Path(a.out).write_text(json.dumps(rows, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
