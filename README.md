# Video-Reconstruction

Reconstruct a short reference video as an editable 3D scene (Blender), then use
structure-conditioned video generation for the final look. Pilot project to test
whether an editable 3D representation beats direct generative editing.

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

## Usage (phase 0)

```bash
.venv/bin/vrecon build   project.json --blend out/scene.blend --render-dir out/frames --frames 0,12
.venv/bin/vrecon handoff project.json --out out/handoff      # then edit out/handoff/scene.blend in Blender
.venv/bin/vrecon ingest  project.json --handoff out/handoff/handoff.json [--dry-run]
```

`ingest` imports transforms of objects with a `vrecon_id`, camera settings and their keyframes,
records them as human edits, and bumps the project revision. Other edits are reported as warnings.

## Status
Phase 0 done: project schema, conventions, Blender builder, handoff/ingest. Next: phase 1
(V2V control test and single-shot baseline). See `docs/PLAN.md` section 12.
