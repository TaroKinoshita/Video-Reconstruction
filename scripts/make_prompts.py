"""Write r2v jobs whose prompts are built from the scene, not written per clip (docs: gpu/h3_control/RESULTS_2026-11-16.md).

    python scripts/make_prompts.py --project runs/venue/tennis/status_v040/project.json \
        --look evaluation/tennis/prompts/look_tennis.txt --action evaluation/tennis/prompts/action_tennis.txt \
        --main obj_person_01 --main-role "the tennis player serving" \
        --variant clay=evaluation/tennis/controls/clay/controls_dolly-100+crane4 \
        --variant id=evaluation/tennis/controls/colour/controls_dolly-100+crane4 \
        --variant look=evaluation/tennis/controls/colour/controls_dolly-100+crane4 \
        --images refs/frame_full.png,refs/player_full.png,refs/player_upper.png,refs/player_serve.png,refs/venue_stands.png \
        --name dolly --group stage3 --out evaluation/tennis/jobs_stage3.json

Prompt = look (what <Picture N> show; the clip's own text) + how to read <Video 1> (one sentence per render kind,
same for every clip) + the scene inventory (control/inventory.py, from the export's index pass) + action (the
clip's own text). Each --variant is a render kind and the export folder holding control_<kind>.mp4 and passes/.
Paths in the jobs file are relative to the jobs file's folder.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

HOW = {
    "clay": "<Video 1> is a grey clay 3D render of the shot: follow it for the camera, the position, size and outline of "
            "every object and person, and every motion, frame by frame. Do not copy its grey look.",
    "id": "<Video 1> is a colour-coded 3D render of the shot: every flat colour is one separate object or person. Follow it "
          "for the camera, the position, size and outline of every object and person, and every motion, frame by frame. "
          "Do not copy its colours.",
    "idclay": "<Video 1> is a colour-coded clay 3D render of the shot: every colour is one separate object or person, "
              "shaded to show its 3D form. Follow it for the camera, the position, size and outline of every object and "
              "person, and every motion, frame by frame. Do not copy its colours.",
    "look": "<Video 1> is a rough coloured 3D render of the shot: follow it for the camera, the position, size and outline "
            "of every object and person, and every motion, frame by frame. Take the real look from the pictures, not "
            "from its plastic surfaces.",
    "depth": "<Video 1> is a depth render of the shot (near is bright): follow it for the camera, the positions of every "
             "object and person, and every motion, frame by frame. Do not copy its look.",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", type=Path, required=True)
    ap.add_argument("--look", type=Path, required=True)
    ap.add_argument("--action", type=Path, required=True)
    ap.add_argument("--main", default=None)
    ap.add_argument("--main-role", default="the main subject")
    ap.add_argument("--variant", action="append", required=True, help="KIND=EXPORT_DIR")
    ap.add_argument("--images", default="")
    ap.add_argument("--crops", type=Path, help="crops.json from scripts/make_ref_crops.py: pictures and what they show")
    ap.add_argument("--name", required=True)
    ap.add_argument("--group", default="gen")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--no-inventory", action="store_true")
    a = ap.parse_args()

    from vrecon.control.inventory import inventory_sentence, visible_objects
    from vrecon.schema import load_project

    project = load_project(a.project)
    pics, pic_text = [x for x in a.images.split(",") if x], ""
    if a.crops:  # pictures in their order, each said in the prompt: what object and class it shows
        rows = json.loads(a.crops.read_text(encoding="utf-8"))
        rel = os.path.relpath(a.crops.parent.resolve(), a.out.parent.resolve()).replace(os.sep, "/")
        pics = [f"{rel}/{r['file']}" for r in rows]
        words = {"frame": "a full frame of the original video", "person": "a person", "crowd": "spectators",
                 "set:stand": "the stands", "set:net": "the net", "set:court": "the court", "set:wall": "the wall"}
        said = []
        for i, r in enumerate(rows, 1):
            what = a.main_role if r.get("object") == a.main else words.get(r["class"], f"a {r['class']}")
            if r["class"] != "frame":
                what += " with its surroundings" + (" (cut by the frame edge in the video)" if r.get("partial") else "")
            said.append(f"<Picture {i}> shows {what}")
        pic_text = "; ".join(said) + ". Keep the look of every person and object from the pictures."
    look, action = a.look.read_text(encoding="utf-8").strip(), a.action.read_text(encoding="utf-8").strip()
    base = a.out.parent.resolve()
    spec = json.loads(a.out.read_text(encoding="utf-8")) if a.out.is_file() else {
        "about": "r2v jobs written by scripts/make_prompts.py", "defaults": {
            "sample_size": [704, 800], "num_inference_steps": 20, "guidance_scale": 1.0, "control_context_scale": 1.0,
            "seeds": [43], "negative_prompt": ""}, "jobs": []}
    for v in a.variant:
        kind, _, folder = v.partition("=")
        folder = Path(folder).resolve()
        man = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        index = np.load(folder / man["passes"]["index"])["index"]
        vis = visible_objects(index, {int(k): x for k, x in man["passes"]["id_map"].items()}, project)
        inv = "" if a.no_inventory else inventory_sentence(vis, a.main, a.main_role)
        job_id = f"{a.name}__{kind}{'' if a.no_inventory else '_inv'}"
        job = {"id": job_id, "scene": a.name, "control": os.path.relpath(folder / f"control_{kind}.mp4", base).replace("\\", "/"),
               "control_kind": kind, "ref_images": pics,
               "prompt": "\n".join(x for x in (pic_text, look, HOW[kind], inv, action) if x), "inventory": vis, "group": a.group}
        spec["jobs"] = [j for j in spec["jobs"] if j["id"] != job_id] + [job]
        print(job_id, "|", inv)
    a.out.write_text(json.dumps(spec, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
