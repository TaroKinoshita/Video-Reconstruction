"""Plan step 6 on a real clip: none / rules / LLM completion, compared by eye from orbit cameras (a real clip
with a fixed camera has no ground truth for new views; the scored comparison is scripts/run_completion_compare.py).

    python scripts/llm_complete_scene.py SOURCE_PROJECT RULES_PROJECT VIDEO OUT_DIR [--goals faithful,creative]
        [--runs 1] [--model claude-opus-5-5] [--frame 120] [--orbits 30,-30]

The LLM gets only the source picture (one frame) and scene.json (objects in metres: the model units divided by the
project's scale; cameras: the source and the new ones) and proposes mirror / repeat / nothing
(run_completion_compare.ask_llm, Read tool only). Proposed positions come back in metres and are converted.
People (characters) are listed but cannot be copied. Writes OUT_DIR/<goal>_<k>/project.json, llm.log,
OUT_DIR/results.json (ops, cost), OUT_DIR/overview.png (rows none / rules / each LLM run; columns source, orbits).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))


def summary(p, s: float, cams: dict) -> dict:
    from vrecon.metrics.poses import camera_pose

    objs = []
    for o in p.reconstruction.objects:
        if o.id == "obj_background":
            continue
        k = o.keys[0] if o.keys else o.rest
        t = np.array(k.pose.translation) / s
        q = k.pose.rotation
        e = {"id": o.id, "name": o.name or o.id, "centre": [round(float(v), 2) for v in t]}
        if o.kind == "character":
            e["kind"] = "person (pose from video; cannot be copied)"
        else:
            e["kind"] = "floor/wall/stands (flat)" if o.kind == "set" else "object"
            e["size"] = [round(float(v) / s, 2) for v in o.size]
            e["yaw_deg"] = round(math.degrees(2 * math.atan2(q[3], q[0])), 1)
            if o.status == "assumed":
                e["note"] = "already extended by a rule (assumed)"
        objs.append(e)

    def cam(c):
        r, centre, _ = camera_pose(c, 0)
        return {"position": [round(float(v) / s, 2) for v in centre], "looks_along": [round(float(v), 3) for v in np.asarray(r)[:, 2]]}

    out = {"objects": objs, "original_camera": cam(cams["source"])}
    out["new_camera"] = cam(next(v for k, v in cams.items() if k != "source"))
    out["other_new_cameras"] = [cam(v) for k, v in list(cams.items())[2:]]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    for x in ("source", "rules", "video", "out"):
        ap.add_argument(x)
    ap.add_argument("--goals", default="faithful,creative")
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--model", default="claude-opus-5-5")
    ap.add_argument("--frame", type=int, default=120)
    ap.add_argument("--orbits", default="30,-30")
    a = ap.parse_args()
    import cv2
    from coverage_scene import orbit
    from run_completion_compare import apply_ops, ask_llm

    from vrecon.control.core import load_frames, read_video_rgb
    from vrecon.scene.builder import build_scene
    from vrecon.scene.color_passes import render_color_pass
    from vrecon.schema import load_project, save_project

    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    src_p, rules_p = load_project(a.source), load_project(a.rules)
    root = Path(a.source).resolve().parent
    place = json.loads((root / "placement.json").read_text(encoding="utf-8")) if (root / "placement.json").exists() else None
    if place is None:
        place = json.loads((Path(a.source).resolve().parents[1] / "scene" / "placement.json").read_text(encoding="utf-8"))
    s = float(place["scale_units_per_metre"])
    cam = src_p.reconstruction.cameras[0]
    ch = next(o for o in src_p.reconstruction.objects if o.kind == "character")
    pivot = np.array((ch.keys[0] if ch.keys else ch.rest).pose.translation, float)
    pivot[2] = 0.0
    cams = {"source": cam} | {f"orbit{int(x):+d}": orbit(cam, float(x), pivot) for x in a.orbits.split(",")}
    frame = read_video_rgb(a.video)[a.frame]
    png = out / "source.png"
    ok, buf = cv2.imencode(".png", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    buf.tofile(png)

    results, rows = {}, {"none": src_p, "rules": rules_p}
    for goal in a.goals.split(","):
        for k in range(a.runs):
            tag = f"{goal}_{k + 1}"
            d = out / tag
            d.mkdir(exist_ok=True)
            try:
                ops, usage = ask_llm(png, summary(rules_p, s, cams), a.model, d / "llm.log", goal)
            except Exception as e:
                results[tag] = {"error": str(e)}
                continue
            scaled = []
            for op in ops:
                op = dict(op)
                for key in ("point", "offset"):
                    if key in op:
                        op[key] = [float(v) * s for v in op[key]]
                if op.get("target") == ch.id or any(o.id == op.get("target") and o.kind == "character"
                                                    for o in rules_p.reconstruction.objects):
                    op = {"op": "none", "reason": f"refused: {op.get('target')} is a person"}
                scaled.append(op)
            p, done = apply_ops(rules_p, scaled)
            save_project(p, d / "project.json")
            results[tag] = {"ops": ops, "applied": done, "usage": usage}
            rows[tag] = p
            print(tag, json.dumps(done, default=str)[:600], usage, flush=True)
    (out / "results.json").write_text(json.dumps(results, indent=1, default=str) + "\n", encoding="utf-8")

    grid = []
    for tag, p in rows.items():
        row = []
        for name, c in cams.items():
            q = p.model_copy(deep=True)
            q.reconstruction.cameras = [c]
            build_scene(q, root=root)
            x = load_frames(render_color_pass("clay", out / "clay" / f"{tag}_{name}", [a.frame], {}))[0]
            x = cv2.resize(x, (480, int(480 * x.shape[0] / x.shape[1])))
            cv2.putText(x, f"{tag} {name}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            row.append(x)
        grid.append(np.concatenate(row, 1))
    ok, buf = cv2.imencode(".png", cv2.cvtColor(np.concatenate(grid, 0), cv2.COLOR_RGB2BGR))
    buf.tofile(out / "overview.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
