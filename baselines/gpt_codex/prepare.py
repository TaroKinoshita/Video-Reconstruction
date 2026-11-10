"""Prepare condition B or C runs for an agent that is driven by hand (e.g. the Codex app), not by a script.

    python baselines/gpt_codex/prepare.py b CLIP_DIR RUN_DIR      # B: workspace with the video + prompt
    python baselines/gpt_codex/prepare.py c CLIP_NAME RUN_DIR     # C: fresh repair run, same start as runs/llm_c4

Writes RUN_DIR/prompt.txt (paste it into the agent) and RUN_DIR/codex_mcp.toml (the one MCP server the agent
may use). Ground truth never goes into the agent's workspace (PLAN.md 11.8). After the agent finishes:

    python baselines/gpt_codex/finish.py b|c ...   (see README.md)
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PY = REPO / ".venv" / "Scripts" / "python.exe"
if not PY.exists():
    PY = REPO / ".venv" / "bin" / "python"

C_PROMPT = """You are repairing a 3D reconstruction of a short video. The run directory is: {run}

Use only the vrecon tools. Each repair attempt is a hypothesis about ONE thing that is wrong (the camera
focal length, or one object's floor position, size, timing or orientation) tested with run_optimize; the
run manager adopts it only if the loss improves. Read run_state and run_diagnose first, and diagnose again
after each adopted change. The attempt budget is fixed in the run (run_state shows attempts and
max_attempts).
Keep working until the budget is used up. A rejected attempt is not a failure: it only costs one
attempt. When the obvious hypotheses are done, revisit every object's position, size, timing and orientation
again (earlier fits were made with other parts still wrong), and try the camera focal length again.
Finish with a short summary of what you changed and what still looks wrong.
"""


def toml_str(s: str) -> str:
    return json.dumps(str(s).replace("\\", "/"), ensure_ascii=False)


def prepare_b(clip: Path, run: Path) -> None:
    ws = run / "workspace"
    (ws / "input").mkdir(parents=True)
    (ws / "output").mkdir()
    (run / "logs").mkdir()
    shutil.copy(clip / "video.mp4", ws / "input" / "reference.mp4")
    prompt = (REPO / "baselines" / "b_llm_mcp" / "prompt_L1.txt").read_text(encoding="utf-8")
    prompt = prompt.replace("<VIDEO_PATH>", (ws / "input" / "reference.mp4").as_posix())
    prompt = prompt.replace("<OUTPUT_DIR>", (ws / "output").as_posix())
    (run / "prompt.txt").write_text(prompt, encoding="utf-8")
    blend_ai = Path.home() / "Claude" / "blend-ai"
    uv = shutil.which("uv") or "uv"
    (run / "codex_mcp.toml").write_text(
        f"[mcp_servers.blender]\ncommand = {toml_str(uv)}\n"
        f"args = [\"run\", \"--directory\", {toml_str(blend_ai)}, \"blend-ai\"]\n", encoding="utf-8")
    (run / "clip.txt").write_text(str(clip), encoding="utf-8")


def prepare_c(name: str, run: Path) -> None:
    from vrecon.run import service

    src = REPO / "runs" / "llm_c4" / name
    cfg = json.loads((src / "config.json").read_text(encoding="utf-8"))
    service.init_run(str(src / "snapshots" / "rev_0000.json"), str(run), cfg["masks"], cfg["max_attempts"],
                     video=cfg["video"], from_4d=cfg["from_4d"], focal=cfg["focal"])
    new = json.loads((run / "init.json").read_text(encoding="utf-8"))
    old = json.loads((src / "snapshots" / "rev_0000.json").read_text(encoding="utf-8"))
    assert new["reconstruction"] == old["reconstruction"], "start scene differs from runs/llm_c4"
    (run / "prompt.txt").write_text(C_PROMPT.format(run=run.as_posix()), encoding="utf-8")
    (run / "codex_mcp.toml").write_text(
        f"[mcp_servers.vrecon]\ncommand = {toml_str(PY)}\nargs = [\"-m\", \"vrecon.mcp_server\"]\n"
        f"cwd = {toml_str(REPO)}\nenv = {{ PYTHONIOENCODING = \"utf-8\", PYTHONUTF8 = \"1\" }}\n", encoding="utf-8")
    (run / "clip.txt").write_text(str(REPO / "evaluation" / "synthetic" / name), encoding="utf-8")


def main() -> int:
    kind, what, run = sys.argv[1], sys.argv[2], Path(sys.argv[3]).resolve()
    if run.exists() and any(run.iterdir()):
        print(f"{run} is not empty")
        return 2
    if kind == "b":
        prepare_b(Path(what).resolve(), run)
    elif kind == "c":
        prepare_c(what, run)
    else:
        print("first argument: b or c")
        return 2
    print(f"ready: {run / 'prompt.txt'}  /  {run / 'codex_mcp.toml'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
