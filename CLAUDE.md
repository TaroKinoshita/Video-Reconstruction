# CLAUDE.md

Research / non-commercial project: reconstruct reference video as an editable Blender scene, then V2V generation.

## Read first
- `docs/HANDOFF.md` (Japanese): current status, open issues, next steps. Start here.
- `docs/PLAN.md` (Japanese, authoritative plan). Section 12 = stages and exit criteria.
- `docs/ARCHITECTURE.md` (responsibilities, repair loop).
- `docs/CONVENTIONS.md` (axes, units, rotations, intrinsics, time). Follow it in all code.

## Environment
- Setup: `bash scripts/setup.sh --no-smoke` (idempotent; skip if `.venv/bin/python -c "import bpy"` works).
- Python 3.13 venv in `.venv` (bpy 5.2.3 needs 3.13). Run tests: `.venv/bin/python -m pytest -q`.
- No GPU in cloud sessions. Render small (e.g. 320x180, Cycles CPU, few samples).
- Blender 5 animation API: read keyframes through `bpy_extras.anim_utils.action_get_channelbag_for_slot`, not `action.fcurves`.

## CLI
- `vrecon new | build | handoff | ingest | control | score | score-blend` (see README). Code: `src/vrecon/cli.py`.
- Control videos for V2V: `scene/passes.py` (depth/index via emission override + EXR, shaded pass) -> `control/core.py` (numpy/cv2/ffmpeg). H3 constraints: 24 fps, W/H multiples of 32, 17n+5 frames.
- Phase 1a: `scripts/make_phase1a_scenes.py`, `evaluation/phase1a/`, GPU side in `gpu/h3_control/` (runs in the VideoX-Fun env, not vrecon).
- Scoring: `metrics/` (pure numpy) + `evaluate.py`. Synthetic GT clips: `scripts/make_synthetic_clips.py` -> `evaluation/synthetic/` (gt.json/gt_passes are evaluation-only, never inputs to reconstruction). Baseline B kit: `baselines/b_llm_mcp/` (needs GUI Blender; not runnable in cloud).
- Repair loop: `run/manager.py` (proposals, constraints, accept rule, atomic adoption), `optimize/` (silhouette loss via fast hull raster, pattern search), `run/service.py` shared by `vrecon run ...` and `mcp_server.py` (mcp 2.x `MCPServer`). Masks are a GT stand-in until a segmenter is integrated.
- Read-back design: baseline = readout of the saved .blend at handoff (`scene/readback.py`), diff in pure Python (`scene/ingest.py`).

## Rules
- Only `src/vrecon/scene` calls `bpy`.
- Scene state is the source of the adopted reconstruction; never treat it as ground truth. Never overwrite human edits or `locked` fields.
- Large media is not committed (`runs/`, videos are git-ignored).

## User
- Explain in Japanese, casual, conclusion first, for a non-expert. No exaggeration.
