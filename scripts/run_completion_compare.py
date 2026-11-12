"""Completion comparison (plan step 6) on evaluation/completion: none / rules / LLM, scored from the new camera.

    python scripts/run_completion_compare.py OUT_DIR [--llm-runs 3] [--model claude-opus-5-5] [--scenes cmp_01,...]
        [--goal faithful|creative]

    none   the starting scene (what the source view shows)
    rules  floor and wall extended to the new camera's view (run/complete.extend_set_to_view)
    llm    rules + the operations an LLM proposes after looking at the source picture and the scene's numbers
           (mirror / repeat / nothing). Its working folder holds only source.png and scene.json: no ground truth.

Scores (new camera, ground truth render; objects re-matched by overlap in that view, so copies count):
    iou_all      mean IoU over the ground truth's objects (hidden ones score 0 unless something covers them)
    iou_hidden   mean IoU over the objects the source never showed
    added        objects the condition added; unmatched_added: of those, ones that match nothing (made up)
    extra        share of drawn object pixels where the ground truth has none
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

OPS_DOC = """Operations you can propose (JSON list, applied in order):
  {"op": "mirror", "target": OBJECT_ID, "point": [x, y, z], "normal": [nx, ny, 0], "reason": "...", "keep": "...", "hold": "..."}
      a copy of the object mirrored across the vertical plane through point with that normal
  {"op": "repeat", "target": OBJECT_ID, "offset": [dx, dy, 0], "count": N, "reason": "...", "keep": "...", "hold": "..."}
      N copies, the i-th moved by i * offset
  {"op": "none", "reason": "..."}
      add nothing (when nothing about the scene says what is out of view)
reason: what in the picture or the numbers supports it. keep: what must not change. hold: what you are unsure of."""

PROMPT = """You are completing a 3D scene reconstructed from one camera view. The camera saw only part of the scene.
A new camera will show more of it. Decide whether the visible part implies objects outside the original view
(for example a symmetric arrangement, or a regular row that continues), and propose operations to add them.
{goal}

Files in this folder:
  source.png   the original camera's picture
  scene.json   the reconstructed objects (metres, world Z up, floor at z = 0): id, centre, size, yaw; the
               original camera and the new camera (positions and look-at points)

{ops}

Answer with ONLY the JSON list, no other text."""


def scene_summary(p, new_cam) -> dict:
    import math

    from vrecon.metrics.poses import camera_pose

    objs = []
    for o in p.reconstruction.objects:
        q = o.rest.pose.rotation
        objs.append({"id": o.id, "kind": "floor/wall" if o.kind == "set" else "object",
                     "centre": [round(v, 3) for v in o.rest.pose.translation], "size": [round(v, 3) for v in o.size],
                     "yaw_deg": round(math.degrees(2 * math.atan2(q[3], q[0])), 1)})

    def cam(c):
        r, centre, _ = camera_pose(c, 0)
        return {"position": [round(v, 3) for v in centre], "looks_along": [round(v, 3) for v in r[:, 2]]}

    return {"objects": objs, "original_camera": cam(p.reconstruction.cameras[0]), "new_camera": cam(new_cam)}


GOALS = {
    "faithful": "Do not invent objects without support from what is visible: an empty or unexplained area should stay empty.",
    "creative": ("The goal is a natural, complete scene for a new camera move, not a record of what was seen. Where the "
                 "visible part sets up a pattern (symmetry, a regular row, a repeated unit), continue it the way the "
                 "layout most plausibly goes on, up to what the new camera shows, rather than stopping at the edge of "
                 "the original picture. Still add nothing where the visible part implies nothing."),
}


def ask_llm(src_png: Path, summary: dict, model: str, log: Path, goal: str = "faithful") -> list[dict]:
    with tempfile.TemporaryDirectory() as tmp:
        shutil.copyfile(src_png, Path(tmp) / "source.png")
        (Path(tmp) / "scene.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
        (Path(tmp) / "mcp.json").write_text('{"mcpServers": {}}', encoding="utf-8")
        prompt = PROMPT.format(ops=OPS_DOC, goal=GOALS[goal])
        proc = subprocess.run(["claude", "-p", prompt, "--model", model, "--max-turns", "6", "--strict-mcp-config",
                               "--mcp-config", str(Path(tmp) / "mcp.json"), "--tools", "Read", "--allowedTools", "Read",
                               "--output-format", "json"], cwd=tmp, capture_output=True, text=True, encoding="utf-8",
                              timeout=600)
    log.write_text(proc.stdout + "\n--- stderr\n" + proc.stderr, encoding="utf-8")
    out = json.loads(proc.stdout)
    text = out.get("result", "")
    start, end = text.find("["), text.rfind("]")
    ops = json.loads(text[start:end + 1]) if start >= 0 else []
    return ops, {"cost_usd": out.get("total_cost_usd"), "turns": out.get("num_turns")}


def apply_ops(p, ops: list[dict]):
    from vrecon.run.complete import mirror, repeat

    done = []
    for op in ops:
        try:
            if op.get("op") == "mirror":
                p, r = mirror(p, op["target"], op["point"], op["normal"])
            elif op.get("op") == "repeat":
                p, r = repeat(p, op["target"], op["offset"], int(op["count"]))
            else:
                r = {"op": op.get("op", "none")}
            done.append({**r, "reason": op.get("reason"), "hold": op.get("hold")})
        except Exception as e:  # a bad proposal is recorded, not fatal
            done.append({"op": op.get("op"), "error": str(e)})
    return p, done


def score(scene_dir: Path, p, start_ids: set[str]) -> dict:
    from vrecon.evaluate import _load_map
    from vrecon.metrics import match_objects
    from vrecon.scene.builder import build_scene
    from vrecon.scene.passes import render_passes
    from vrecon.schema import Camera

    nd = scene_dir / "gt_passes" / "novel" / "wide"
    info = json.loads((nd / "camera.json").read_text(encoding="utf-8"))
    q = p.model_copy(deep=True)
    q.reconstruction.cameras = [Camera.model_validate(info["camera"])]
    build_scene(q)
    with tempfile.TemporaryDirectory() as tmp:
        render_passes(tmp, shot_frames=[0], shaded=False)
        ri, rm = np.load(Path(tmp) / "index.npz")["index"], _load_map(Path(tmp) / "id_map.json")
    gi, gm = np.load(nd / "index.npz")["index"], _load_map(nd / "id_map.json")
    pairs = match_objects(gi, ri, gm, rm)
    meta = json.loads((scene_dir / "meta.json").read_text(encoding="utf-8"))
    rl = {v: k for k, v in rm.items()}
    ious = {}
    for lab, a in gm.items():
        if a in ("obj_floor", "obj_wall"):
            continue
        g = gi == lab
        r = ri == rl[pairs[a]] if a in pairs else np.zeros_like(g)
        u = (g | r).sum()
        ious[a] = float((g & r).sum() / u) if u else 0.0
    added = [o.id for o in p.reconstruction.objects if o.id not in start_ids]
    matched = set(pairs.values())
    sets = {"obj_floor", "obj_wall"}
    g_obj = np.isin(gi, [k for k, v in gm.items() if v not in sets])
    r_obj = np.isin(ri, [k for k, v in rm.items() if v not in sets])
    return {"iou": ious, "iou_all": float(np.mean(list(ious.values()))),
            "iou_hidden": float(np.mean([ious[h] for h in meta["hidden"]])) if meta["hidden"] else None,
            "added": len(added), "unmatched_added": [a for a in added if a not in matched],
            "extra": float((r_obj & ~g_obj).sum() / max(r_obj.sum(), 1))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--llm-runs", type=int, default=3)
    ap.add_argument("--model", default="claude-opus-5-5")
    ap.add_argument("--scenes", default="cmp_01,cmp_02,cmp_03")
    ap.add_argument("--goal", choices=sorted(GOALS), default="faithful",
                    help="faithful: add only what the visible part supports; creative: continue patterns to fill the new view")
    a = ap.parse_args()
    from vrecon.run.complete import extend_set_to_view
    from vrecon.schema import Camera, load_project

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    results = {}
    for sid in a.scenes.split(","):
        d = Path("evaluation/completion") / sid
        start = load_project(d / "start.json")
        info = json.loads((d / "gt_passes" / "novel" / "wide" / "camera.json").read_text(encoding="utf-8"))
        new_cam = Camera.model_validate(info["camera"])
        ids = {o.id for o in start.reconstruction.objects}
        rules = start
        for o in [o for o in start.reconstruction.objects if o.kind == "set"]:
            rules, _ = extend_set_to_view(rules, o.id, new_cam, frames=[0])
        res = {"none": score(d, start, ids), "rules": score(d, rules, ids), "llm": []}
        for k in range(a.llm_runs):
            log = out / f"{sid}_llm{k + 1}.log"
            try:
                ops, usage = ask_llm(d / "source.png", scene_summary(start, new_cam), a.model, log, a.goal)
            except Exception as e:
                res["llm"].append({"error": str(e)})
                continue
            p, done = apply_ops(rules, ops)
            res["llm"].append({"ops": done, "usage": usage, **score(d, p, ids)})
        results[sid] = res
        line = f"{sid} none {res['none']['iou_all']:.3f} rules {res['rules']['iou_all']:.3f} llm " + \
               " ".join(f"{r.get('iou_all', float('nan')):.3f}(+{r.get('added', '?')})" for r in res["llm"])
        print(line, flush=True)
        (out / "results.json").write_text(json.dumps(results, indent=1, default=str) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
