"""MAVE-style agentic modelling for one shot (arXiv 2611.22464 sections 3.2-3.3): an LLM agent builds the set
with the query - edit - verify tools of src/vrecon/agent, under a time budget; then the scene goes to Blender.

  python scripts/agent_run.py evidence PREP.npz PARTS.npz RC.npz RC_FRAMES.json OUT_EVIDENCE.npz [--people-masks M.npz]
  python scripts/agent_run.py init RUN_DIR EVIDENCE.npz PEOPLE.npz PARTS.npz --video VIDEO --mode empty|system
  python scripts/agent_run.py run RUN_DIR [--model claude-opus-5-5] [--target 15] [--limit 30]
  python scripts/agent_run.py verify RUN_DIR                       (the checks without an agent)
  python scripts/agent_run.py export RUN_DIR OUT_SPEC.json         (for src/vrecon/scene/mave_blend.py --spec)
  python scripts/agent_run.py blend RUN_DIR PREP.npz OUT.blend [--size 1920 816] [--fps 23.976] [--blender EXE]
      [--extra "--staffs S.npz --people-gain R G B"]   (the run's people and parts, its current scene)

people masks (evidence): a run_sam3.py file on a video of the 4RC frames in order (frame i = RC_FRAMES[i]).
mode empty = MAVE eq. 7 (cameras and people only, the agent builds everything); system = start from the system's
parts (ground, trunks, posts, beams, canopies, backdrop) and let the agent correct and complete them.
The agent runs as `claude -p` with only these tools and Read (for its pictures); no ground truth is reachable.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

PROMPT = """You are building an editable 3D blockout of one film shot in Blender terms, as simple named primitives
(boxes, cylinders, planes, ellipsoids), so that the shot's recovered camera, looking at your set with the recovered
people in it, reproduces the shot's spatial layout: what is in front of what, where people stand and what hides them.

Fixed already (do not try to change): the camera on every frame and the people's body poses, in a metric, z-up
world (metres). You control: the environment primitives, the system meshes (ground height field, backdrop surface,
far forest wall: keep or drop them with the "fixed" edit), one scale k per person (it moves the whole track along
the camera rays and keeps the picture), and removal of tracks or segments that are not people or are tracking
failures (with a reason).

{mode_text}

Work in a loop: overview -> structures / points_in_box / pixel_to_world / render -> edit -> verify -> fix or
justify -> submit. Rules:
- Measured evidence beats your impression from the pictures; use the pictures to decide what a cluster is.
- A primitive must not stand in front of a surface the camera observed (video check); observed support, wall and
  ceiling surfaces should be covered by something (coverage check); feet should rest on a support, bodies should
  not sink into primitives, heights should be plausible (human check); nothing floating, duplicated or buried, and
  the camera never inside a primitive (completeness check). Look at orbit / top views once for the whole structure.
- Justify only what is truly right as it is (a jump, a sitting pose, depth noise on far foliage), with the reason.
- Prefer few, larger primitives for large surfaces; name them by what they are.
- Time: aim to submit by {target} minutes; the hard limit is {limit} minutes, after which the last saved scene
  counts. The overview shows elapsed minutes.
{task}
Work on your own until the scene is submitted: nobody will answer questions. When done, call submit; if it lists
open issues, resolve or justify them and submit again. Reply with one line only after an accepted submit."""

MODES = {
    "empty": "The environment starts EMPTY and only primitives are available: build the ground (it may slope: "
             "follow the support layer's height grid with one or a few planes or boxes) and every structure the "
             "camera sees, from the evidence (structures, points_in_box, pixel_to_world on things you see in the "
             "source pictures) and from the pictures themselves.",
    "system": "The environment starts from the system's automatic parts (ground height field, trunks, posts, beams, "
              "canopies, a backdrop surface of what is not split yet, imagined trees outside every view, a far wall). "
              "Check them, correct what the checks and pictures show to be wrong, and complete what is missing.",
}


def cmd_evidence(a) -> int:
    from vrecon.agent import evidence
    frames = json.loads(Path(a.rc_frames).read_text(encoding="utf-8"))
    pm = None
    if a.people_masks:
        pm = np.load(a.people_masks)["masks"] > 0
    rep = evidence.build(Path(a.prep), Path(a.parts), Path(a.rc), frames, pm, Path(a.out), step=a.step)
    print(json.dumps(rep, indent=1))
    return 0


def cmd_init(a) -> int:
    from vrecon.agent.session import init_run
    print(json.dumps(init_run(a.run, a.evidence, a.people, a.parts, a.video, a.mode, a.task, a.fps,
                              a.target, a.limit)))
    return 0


def cmd_verify(a) -> int:
    from vrecon.agent.session import Session
    s = Session(a.run)
    t = time.time()
    res = s.verify(max_issues=a.max)
    res["seconds"] = round(time.time() - t, 1)
    print(json.dumps(res, indent=1, ensure_ascii=False))
    return 0


def cmd_run(a) -> int:
    run = Path(a.run).resolve()
    meta = json.loads((run / "run.json").read_text(encoding="utf-8"))
    meta["started"] = time.time()
    meta["budget_min"] = {"target": a.target, "limit": a.limit}
    meta["model"] = a.model
    (run / "run.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    py = sys.executable
    cfg = {"mcpServers": {"scene": {"command": py, "args": ["-m", "vrecon.agent.tools_mcp"],
                                    "env": {"VRECON_AGENT_RUN": str(run), "PYTHONPATH": str(ROOT / "src"),
                                            "PYTHONUTF8": "1"}}}}
    (run / "mcp.json").write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    prompt = PROMPT.format(mode_text=MODES[meta["mode"]], target=a.target, limit=a.limit,
                           task=("About the shot: " + meta["task"]) if meta.get("task") else "")
    (run / "prompt.txt").write_text(prompt, encoding="utf-8")
    base = ["--model", a.model, "--max-turns", str(a.max_turns), "--strict-mcp-config",
            "--mcp-config", str(run / "mcp.json"), "--tools", "Read", "--allowedTools", "Read,mcp__scene",
            "--output-format", "stream-json", "--verbose"]
    t0 = time.time()
    deadline = t0 + a.limit * 60
    session, rounds, why, code = None, 0, "finished", 0
    cost, turns, result = 0.0, 0, None
    msg = prompt
    # MAVE's loop runs until the scene is submitted or the time is used up: when the agent stops before an
    # accepted submit, the same session is resumed with the open issues and the time left
    with open(run / "agent_stream.jsonl", "w", encoding="utf-8") as log:
        while True:
            rounds += 1
            cmd = ["claude", "-p", msg] + base + (["--resume", session] if session else [])
            mark = log.tell()
            proc = subprocess.Popen(cmd, cwd=run, stdout=log, stderr=subprocess.STDOUT, text=True, encoding="utf-8")
            try:
                code = proc.wait(timeout=max(deadline - time.time(), 1))
            except subprocess.TimeoutExpired:
                proc.kill()
                code, why = -1, "hard time limit"
            log.flush()
            with open(run / "agent_stream.jsonl", encoding="utf-8") as f:
                f.seek(mark)
                for line in f:
                    try:
                        m = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if m.get("type") == "system" and m.get("session_id"):
                        session = m["session_id"]
                    if m.get("type") == "result":
                        cost += m.get("total_cost_usd") or 0
                        turns += m.get("num_turns") or 0
                        result = m.get("result")
            left = (deadline - time.time()) / 60
            if why != "finished" or (run / "final.json").exists() or left < 1.0 or rounds >= a.max_rounds or not session:
                break
            issues = json.loads((run / "issues.json").read_text(encoding="utf-8")) if (run / "issues.json").exists() else {}
            elapsed = (time.time() - t0) / 60
            msg = (f"The scene is not submitted yet. Last verification: {issues.get('counts')} "
                   f"({issues.get('open', '?')} open). Elapsed {elapsed:.1f} min; target {a.target} min, hard limit "
                   f"{a.limit} min. Keep working without asking anything (no one will answer): build what is missing, "
                   "fix or justify each issue, verify, and submit.")
    summary = {"why": why, "exit": code, "minutes": round((time.time() - t0) / 60, 1), "cost_usd": round(cost, 3),
               "turns": turns, "rounds": rounds, "result": result, "submitted": (run / "final.json").exists()}
    (run / "agent_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    return 0


def cmd_export(a) -> int:
    run = Path(a.run)
    sc = json.loads((run / "scene.json").read_text(encoding="utf-8"))
    Path(a.out).write_text(json.dumps(sc, indent=1), encoding="utf-8")
    print(json.dumps({"primitives": len(sc["primitives"]), "fixed": sc.get("fixed"),
                      "people": {k: {"k": v.get("k"), "removed": v.get("removed"), "track_removed": v.get("track_removed")}
                                 for k, v in sc["people"].items()}}))
    return 0


def cmd_blend(a) -> int:
    run = Path(a.run).resolve()
    meta = json.loads((run / "run.json").read_text(encoding="utf-8"))
    spec = run / "spec.json"
    Path(a.out).resolve().parent.mkdir(parents=True, exist_ok=True)
    spec.write_text((run / "scene.json").read_text(encoding="utf-8"), encoding="utf-8")
    cmd = [a.blender, "-b", "--factory-startup", "-P", str(ROOT / "src/vrecon/scene/mave_blend.py"), "--",
           str(Path(a.prep).resolve()), str(Path(a.out).resolve()), "--size", str(a.size[0]), str(a.size[1]),
           "--fps", str(a.fps), "--people-anim", meta["people"], "--parts", meta["parts"], "--spec", str(spec)]
    cmd += a.extra.split() if a.extra else []
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    lines = [l for l in (p.stdout + p.stderr).splitlines() if l.startswith(("saved", "Error", "Traceback")) or "Error" in l]
    print("\n".join(lines[-8:]))
    return p.returncode


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("evidence")
    for n in ("prep", "parts", "rc", "rc_frames", "out"):
        e.add_argument(n)
    e.add_argument("--people-masks", default=None)
    e.add_argument("--step", type=int, default=4)
    i = sub.add_parser("init")
    for n in ("run", "evidence", "people", "parts"):
        i.add_argument(n)
    i.add_argument("--video", default=None)
    i.add_argument("--mode", choices=["empty", "system"], default="empty")
    i.add_argument("--task", default="")
    i.add_argument("--fps", type=float, default=None)
    i.add_argument("--target", type=float, default=15)
    i.add_argument("--limit", type=float, default=30)
    r = sub.add_parser("run")
    r.add_argument("run")
    r.add_argument("--model", default="claude-opus-5-5")
    r.add_argument("--target", type=float, default=15)
    r.add_argument("--limit", type=float, default=30)
    r.add_argument("--max-turns", type=int, default=300)
    r.add_argument("--max-rounds", type=int, default=12, help="resumes when the agent stops before submitting")
    v = sub.add_parser("verify")
    v.add_argument("run")
    v.add_argument("--max", type=int, default=60)
    x = sub.add_parser("export")
    x.add_argument("run")
    x.add_argument("out")
    bl = sub.add_parser("blend")
    bl.add_argument("run")
    bl.add_argument("prep")
    bl.add_argument("out")
    bl.add_argument("--size", type=int, nargs=2, default=[1920, 816])
    bl.add_argument("--fps", type=float, default=23.976)
    bl.add_argument("--blender", default="C:/Program Files/Blender Foundation/Blender 4.5/blender.exe")
    bl.add_argument("--extra", default="")
    a = ap.parse_args()
    os.environ.setdefault("PYTHONUTF8", "1")
    return {"evidence": cmd_evidence, "init": cmd_init, "run": cmd_run, "verify": cmd_verify,
            "export": cmd_export, "blend": cmd_blend}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
