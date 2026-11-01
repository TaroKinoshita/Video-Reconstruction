# Video-Reconstruction

Reconstruct a short reference video as an editable 3D scene (Blender), then use
structure-conditioned video generation for the final look. Pilot project to test
whether an editable 3D representation beats direct generative editing.

- **Handoff / current status (Japanese): [`docs/HANDOFF.md`](docs/HANDOFF.md)** — start here
- **Plan (current, Japanese): [`docs/PLAN.md`](docs/PLAN.md)** — takes precedence over the proposal
- Proposal: [`docs/proposal_v0.3.pdf`](docs/proposal_v0.3.pdf)
- Architecture draft: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)

## Setup

```bash
bash scripts/setup.sh
```

Installs Mesa EGL on Linux (software OpenGL, no GPU needed), creates `.venv` with
Python 3.13, installs `bpy` 5.2.3 (~400 MB) and this package, then runs a render
smoke test. Safe to rerun. Tests: `.venv/bin/python -m pytest`.

### Windows (PowerShell)

```powershell
winget install --id=astral-sh.uv -e
winget install --id=Gyan.FFmpeg -e
uv venv --python 3.13 .venv
uv pip install --python .venv\Scripts\python.exe -e ".[blender,control,mcp,dev]"
.venv\Scripts\python.exe -m pytest -q
```

Bash scripts (`scripts/*.sh`, `gpu/`, `baselines/`) need WSL or Git Bash.

## Usage (phase 0)

```bash
.venv/bin/vrecon new     clip.mp4 --out myproj                 # import a video, detect shots
.venv/bin/vrecon build   project.json --blend out/scene.blend --render-dir out/frames --frames 0,12
.venv/bin/vrecon handoff project.json --out out/handoff      # then edit out/handoff/scene.blend in Blender
.venv/bin/vrecon ingest  project.json --handoff out/handoff/handoff.json [--dry-run]
.venv/bin/vrecon control project.json --out out/controls       # depth / canny control videos for V2V
.venv/bin/vrecon score   evaluation/synthetic/syn_01 project.json   # score against a ground-truth clip
.venv/bin/vrecon score-blend evaluation/synthetic/syn_01 scene.blend  # score any .blend (e.g. from an LLM)
```

`ingest` imports transforms of objects with a `vrecon_id`, camera settings and their keyframes,
records them as human edits, and bumps the project revision. Other edits are reported as warnings.

## Status
Phase 0 done. Phase 1a in progress: test scenes and control videos in `evaluation/phase1a/`,
GPU generation with MiniMax-H3 + Fun ControlNet Union 2.0 in `gpu/h3_control/` (see its README).
Phase 1b prep: synthetic ground-truth clips (`evaluation/synthetic/`), scoring tools, baseline B kit
(`baselines/b_llm_mcp/`).
