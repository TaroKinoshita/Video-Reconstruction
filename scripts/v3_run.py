"""Run one agent iteration (docs/DESIGN_v3.md): evidence scene + the agent's script -> scene, pictures, checks.

    python scripts/v3_run.py EVIDENCE.npz DIAG_DIR TAG SOURCE.mp4 SCRIPT.py OUT_DIR

EVIDENCE.npz is the observation pipeline's last scene file (ms_TAG_gate.npz). OUT_DIR gets:
  base.blend        the locked evidence (built once): cameras, people, floor, backdrop
  scene.blend       base + the script's parts
  look/cutK.jpg     per cut, 2 frames: film left, scene right (what the agent compares)
  report.md         script errors, the checks as sentences (MAVE-style evidence checks + picture checks), scores
  report.json       the same, for programs
The script runs with the API of src/vrecon/scene/agent_api.py in its namespace and may read evidence.json
(written next to base.blend: cameras, people, evidence planes, observed objects).
"""

from __future__ import annotations

import json
import subprocess
import sys
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
COVER_M = 0.6  # an evidence surface counts as covered when a part reaches within this of its centre
FAR_M = 15.0


def evidence_json(z, info) -> dict:
    cams, people = [], []
    for k, s in enumerate(info["shots"]):
        c = z[f"s{k}_cam"]
        a = s["source_frames"][0] + 1
        cams.append({"cut": k, "timeline_frames": [a, a + s["frames"] - 1], "close_up": bool(s.get("points_only_during_cut")),
                     "lens_mm": round(s["lens_35mm"], 1),
                     "position_first_mid_last": [np.round(c[i, :3, 3], 2).tolist() for i in (0, len(c) // 2, -1)],
                     "looking_first_mid_last": [np.round(-c[i, :3, 2], 2).tolist() for i in (0, len(c) // 2, -1)]})
        b = z[f"s{k}_body"]
        if len(b):
            people.append({"name": f"Body{k:02d}", "cut": k, "pelvis_first_mid_last": [np.round(b[i, 0], 2).tolist() for i in (0, len(b) // 2, -1)],
                           "height_m": round(float(b[len(b) // 2, :, 2].max() - b[len(b) // 2, :, 2].min()), 2)})
    planes = [{"category": str(cat), "corners": np.round(q, 2).tolist()} for q, cat in zip(z["layout_quads"], z["layout_quad_cat"])] if "layout_quads" in z else []
    objs = []
    for k, s in enumerate(info["shots"]):
        if f"s{k}_objects" in z:
            for tr, lab in zip(z[f"s{k}_objects"], z[f"s{k}_object_labels"]):
                v = tr[:, 6] > 0
                if v.any():
                    objs.append({"cut": k, "label": str(lab).split("|")[0], "median_position": np.round(np.median(tr[v, :3], 0), 2).tolist(),
                                 "size_whd": np.round(tr[v][0, 3:6], 2).tolist(), "frames_seen": int(v.sum())})
    return {"world": "metres, Z up, floor z = 0, one world for all cuts", "cameras": cams, "people": people,
            "evidence_planes": planes, "observed_objects": objs}


def main() -> int:
    ev, diag, tag, source, script, out = (Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], Path(sys.argv[4]),
                                          Path(sys.argv[5]), Path(sys.argv[6]))
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    import bpy

    from vrecon.scene import agent_api
    from vrecon.scene.multishot import build

    z = np.load(ev)
    info = json.loads(str(z["info"]))
    base = out / "base.blend"
    if not base.exists():
        build(ev.resolve(), base.resolve(), finished=True, evidence_only=True)
        (out / "evidence.json").write_text(json.dumps(evidence_json(z, info), indent=1), encoding="utf-8")
    bpy.ops.wm.open_mainfile(filepath=str(base.resolve()))
    errors = []
    ns = dict(agent_api.API)
    ns["EVIDENCE"] = json.loads((out / "evidence.json").read_text(encoding="utf-8"))
    try:
        exec(compile(script.read_text(encoding="utf-8"), str(script), "exec"), ns)
    except Exception:  # noqa: BLE001 - the agent's script: report, keep what was built
        errors.append(traceback.format_exc(limit=3))
    scene = out / "scene.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(scene.resolve()))
    issues = agent_api.part_checks()
    # evidence coverage: measured structure surfaces near the cameras with no part close to them
    pb = {o.name[2:]: agent_api._world_bbox(o) for o in agent_api.parts()}
    camp = np.concatenate([z[f"s{k}_cam"][:, :3, 3] for k in range(len(info["shots"]))])
    if "layout_quads" in z:
        for q, cat in zip(z["layout_quads"], z["layout_quad_cat"]):
            c = q.mean(0)
            if np.min(np.linalg.norm(camp - c, axis=1)) > FAR_M:
                continue
            near = [n for n, (lo, hi) in pb.items() if np.linalg.norm(np.clip(c, lo, hi) - c) < COVER_M]
            if not near:
                issues.append(f"uncovered: a measured {cat} surface around {np.round(c, 1).tolist()} "
                              f"({np.linalg.norm(q[1] - q[0]):.1f} x {np.linalg.norm(q[3] - q[0]):.1f} m) has no part")
    # pictures: 2 frames per cut, film | scene
    from vrecon.control.core import read_video_rgb
    import cv2

    film = read_video_rgb(source)
    (out / "look").mkdir(exist_ok=True)
    frames = {}
    for k, s in enumerate(info["shots"]):
        a = s["source_frames"][0] + 1
        for f in sorted({a + s["frames"] // 4, a + 3 * s["frames"] // 4}):
            frames[f] = out / "look" / f"_r{f:04d}.png"
    agent_api.render_look(frames)
    for k, s in enumerate(info["shots"]):
        a = s["source_frames"][0] + 1
        rows = []
        for f in sorted({a + s["frames"] // 4, a + 3 * s["frames"] // 4}):
            r = cv2.imdecode(np.fromfile(str(frames[f]), np.uint8), cv2.IMREAD_COLOR)
            l = cv2.resize(film[f - 1][..., ::-1], (r.shape[1], r.shape[0]))
            rows.append(np.concatenate([l, r], 1))
        cv2.imencode(".jpg", np.concatenate(rows, 0))[1].tofile(str(out / "look" / f"cut{k}.jpg"))
    for p in frames.values():
        p.unlink(missing_ok=True)
    # picture checks against the observations (person / object overlap, depth conflicts, teleports)
    chk = out / "check.json"
    subprocess.run([sys.executable, str(ROOT / "scripts/multishot_check.py"), str(ev), str(scene), str(diag), tag,
                    str(chk), "--every", "3"], capture_output=True, text=True, encoding="utf-8", errors="replace")
    cuts = {}
    if chk.exists():
        r = json.loads(chk.read_text(encoding="utf-8"))
        for f in r["flags"]:  # every warning of the check goes to the agent, with its numbers
            nums = ", ".join(f"{q} {v}" for q, v in f.items() if q not in ("cut", "what"))
            where = f"cut {f['cut']}: " if "cut" in f else ""
            issues.append(f"check: {where}{f['what']}" + (f" ({nums})" if nums else ""))
        cuts = {k: {q: c.get(q) for q in ("score", "empty", "person_blocked", "object_blocked")} for k, c in r["cuts"].items()}
    rep = {"errors": errors, "issues": issues, "parts": len(pb), "hidden_people": agent_api.HIDDEN_PEOPLE, "cuts": cuts}
    (out / "report.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    md = ["# report", f"parts: {len(pb)}", "", "## script errors", *(errors or ["none"]), "", "## issues (fix, or say why not)",
          *([f"- {i}" for i in issues] or ["none"]), "", "## per cut (score: person/object match minus empty share)",
          *[f"- cut {k}: " + ", ".join(f"{q} {v:.2f}" for q, v in c.items() if isinstance(v, (int, float))) for k, c in cuts.items()],
          "", "## pictures: look/cut<K>.jpg (film left, scene right)"]
    (out / "report.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
