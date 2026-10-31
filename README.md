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

## Status
Skeleton only. Next: check Blender (`bpy`) on CPU, then write the shared scene schema.
Stage 1 (manual scene -> generator control test) needs a GPU machine.
