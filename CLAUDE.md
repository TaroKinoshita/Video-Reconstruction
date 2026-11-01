# CLAUDE.md

Research / non-commercial project: reconstruct reference video as an editable Blender scene, then V2V generation.

## Read first
- `docs/PLAN.md` (Japanese, authoritative plan). Section 12 = stages and exit criteria.
- `docs/ARCHITECTURE.md` (responsibilities, repair loop).
- `docs/CONVENTIONS.md` (axes, units, rotations, intrinsics, time). Follow it in all code.

## Environment
- Setup: `bash scripts/setup.sh --no-smoke` (idempotent; skip if `.venv/bin/python -c "import bpy"` works).
- Python 3.13 venv in `.venv` (bpy 5.2.3 needs 3.13). Run tests: `.venv/bin/python -m pytest -q`.
- No GPU in cloud sessions. Render small (e.g. 320x180, Cycles CPU, few samples).
- Blender 5 animation API: read keyframes through `bpy_extras.anim_utils.action_get_channelbag_for_slot`, not `action.fcurves`.

## CLI
- `vrecon build | handoff | ingest | control` (see README). Code: `src/vrecon/cli.py`.
- Control videos for V2V: `scene/passes.py` (depth/index via emission override + EXR, shaded pass) -> `control/core.py` (numpy/cv2/ffmpeg). H3 constraints: 24 fps, W/H multiples of 32, 17n+5 frames.
- Phase 1a: `scripts/make_phase1a_scenes.py`, `evaluation/phase1a/`, GPU side in `gpu/h3_control/` (runs in the VideoX-Fun env, not vrecon).
- Read-back design: baseline = readout of the saved .blend at handoff (`scene/readback.py`), diff in pure Python (`scene/ingest.py`).

## Rules
- Only `src/vrecon/scene` calls `bpy`.
- Scene state is the source of the adopted reconstruction; never treat it as ground truth. Never overwrite human edits or `locked` fields.
- Large media is not committed (`runs/`, videos are git-ignored).

## User
- Explain in Japanese, casual, conclusion first, for a non-expert. No exaggeration.
