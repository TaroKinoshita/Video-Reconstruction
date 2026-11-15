"""WP-3 step 5: give the finished tennis scene the 0.4.0 statuses, without re-running any observer or fit.

    python scripts/convert_tennis_status.py [--final runs/venue/tennis/fal2c_props/final]
        [--props runs/venue/tennis/fal2c_props/props.json] [--out runs/venue/tennis/status_v040]

What it rebuilds (stored scene untouched; output in a new folder):
    venue parts   status from the layout part's own word in its evidence note ("venue layout part (standard)"):
                  standard / observed / hypothesis, evidence kind standard / observation / llm
    racket        per-frame notes from the stored evidence (one entry per frame that was not "sharp"), then
                  spans (analysis/prop_spans.stick_labels); per-frame notes to obj_tennis_racket_frames.json
    ball          segment notes from props.json, then spans (analysis/prop_spans.sphere_labels)
    other objects unchanged (people, chair and person boxes: estimated from masks)
Then the status pass of the stored controls (passes/index.npz of each controls_<move>) and its shares:
    <out>/status_report.json
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VENUE = re.compile(r"venue layout part \((standard|observed|hypothesis)\)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", type=Path, default=ROOT / "runs/venue/tennis/fal2c_props/final")
    ap.add_argument("--props", type=Path, default=ROOT / "runs/venue/tennis/fal2c_props/props.json")
    ap.add_argument("--out", type=Path, default=ROOT / "runs/venue/tennis/status_v040")
    a = ap.parse_args()

    from vrecon.analysis import prop_spans as S
    from vrecon.control.status import shares, status_pass
    from vrecon.schema import Evidence, Rational, load_project, save_project

    p = load_project(a.final / "project.json")
    props = json.loads(a.props.read_text(encoding="utf-8"))
    fps = p.source.fps.value
    a.out.mkdir(parents=True, exist_ok=True)
    rep: dict = {"source": a.final.relative_to(ROOT).as_posix(), "objects": {}}
    objs = []
    for o in p.reconstruction.objects:
        note = o.evidence[0].note if o.evidence else ""
        m = VENUE.search(note)
        if m:
            st = m.group(1)
            ev = [e.model_copy(update={"kind": {"standard": "standard", "observed": "observation"}.get(st, "llm")})
                  for e in o.evidence]
            o = o.model_copy(update={"status": st, "evidence": ev})
        elif o.id == "obj_tennis_racket":
            n = len(o.keys)
            notes = ["sharp"] * n
            for e in o.evidence[1:]:
                if len(e.source_frames) == 1 and e.source_frames[0] < n:
                    notes[e.source_frames[0]] = e.note
            labels = S.stick_labels(notes)
            spans, span_ev = S.to_schema(labels, fps)
            (a.out / "obj_tennis_racket_frames.json").write_text(json.dumps(
                {"object": o.id, "notes": notes, "status": [x[0] for x in labels]}) + "\n", encoding="utf-8")
            first = o.evidence[0].model_copy(update={"kind": "observation"})
            o = o.model_copy(update={"status": S.overall_status(labels), "spans": spans, "evidence": [first] + span_ev})
        elif o.id == "obj_tennis_ball":
            notes = props["props"]["tennis ball"]["object_0"]["notes"]
            keyed = {round(k.time * fps) for k in o.keys}
            t_n = p.source.frame_count
            labels = S.sphere_labels(notes, t_n, [f in keyed for f in range(t_n)])
            spans, span_ev = S.to_schema(labels, fps)
            (a.out / "obj_tennis_ball_frames.json").write_text(json.dumps(
                {"object": o.id, "notes": notes, "status": [x[0] if x else None for x in labels]}) + "\n",
                encoding="utf-8")
            first = o.evidence[0].model_copy(update={"kind": "observation"})
            o = o.model_copy(update={"status": S.overall_status(labels), "spans": spans, "evidence": [first] + span_ev})
        else:
            o = o.model_copy(update={"evidence": [e.model_copy(update={"kind": e.kind or "observation"})
                                                  for e in o.evidence]})
        objs.append(o)
        rep["objects"][o.id] = {"status": o.status,
                                "spans": [[round(s.start * fps), round(s.end * fps), s.status, s.basis]
                                          for s in o.spans]}
    p = p.model_copy(deep=True)
    p.reconstruction.objects = objs
    units = a.final.parent / "units.json"  # F-07: the scale goes into the project (not found by a folder walk)
    if units.is_file():
        u = json.loads(units.read_text(encoding="utf-8"))
        p.reconstruction.scale.units_per_metre = float(u["units_per_metre"])
        p.reconstruction.scale.basis = str(u.get("basis") or u.get("source") or "court lines (ITF), units.json")
        shutil.copyfile(units, a.out / "units.json")
    for f in a.final.parent.glob("*.npz"):  # meshes the project refers to
        shutil.copyfile(f, a.out / f.name)
    for f in a.final.glob("*.npz"):
        shutil.copyfile(f, a.out / f.name)
    save_project(p, a.out / "project.json")

    rep["controls"] = {}
    for cdir in sorted(a.final.glob("controls_*")):
        idx = cdir / "passes" / "index.npz"
        if not idx.is_file():
            continue
        man = json.loads((cdir / "manifest.json").read_text(encoding="utf-8"))
        index = np.load(idx)["index"]
        id_map = {int(k): v for k, v in man["passes"]["id_map"].items()}
        q = p.model_copy(deep=True)  # the export's project: 24 fps, frame f at f / 24 s
        q.source.fps = Rational(num=int(man["fps"]), den=1)
        st = status_pass(index, id_map, q)
        np.savez_compressed(a.out / f"status_{cdir.name}.npz", status=st)
        rep["controls"][cdir.name] = shares(st)["clip"]
    (a.out / "status_report.json").write_text(json.dumps(rep, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"ball": rep["objects"].get("obj_tennis_ball"), "racket_status": rep["objects"]
                      ["obj_tennis_racket"]["status"], "controls": rep["controls"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
