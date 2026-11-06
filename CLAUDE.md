# CLAUDE.md

Research / non-commercial project: reconstruct reference video as an editable Blender scene, then V2V generation.

## Read first
- `docs/HANDOFF.md` (Japanese): current status, open issues, next steps. Start here.
- `docs/PLAN.md` (Japanese, authoritative plan). Section 12 = stages and exit criteria.
- `docs/ARCHITECTURE.md` (responsibilities, repair loop).
- `docs/CONVENTIONS.md` (axes, units, rotations, intrinsics, time). Follow it in all code.

## Environment
- Main machine: Windows 11, i7-13700H, 32 GB RAM, RTX 4060 Laptop (8 GB VRAM), Blender 5.2 GUI installed. Cloud sessions (Linux, CPU only) are no longer the default.
- Python 3.13 venv in `.venv` (bpy 5.2.3 needs 3.13). On Windows the python is `.venv/Scripts/python.exe` (Linux: `.venv/bin/python`). Run tests: `.venv/Scripts/python.exe -m pytest -q`.
- Setup: `bash scripts/setup.sh --no-smoke` from Git Bash (idempotent). In PowerShell, plain `bash` starts WSL, which is a different (Linux) environment.
- The repo path contains Japanese and is under OneDrive. Never pass paths to OpenCV file functions (`cv2.imread/imwrite/VideoCapture/VideoWriter` fail on non-ASCII paths on Windows): use `control/core.py` (`load_frames`, `read_video_rgb`, `encode_mp4`). Always pass `encoding="utf-8"` for text files and subprocess output (the Windows default is cp932).
- Rendering: `render_frames` uses the GPU when there is one (`VRECON_RENDER_DEVICE` = auto|cpu|optix|cuda). Data passes in `scene/passes.py` stay on CPU: measured faster for 1 sample.
- "Windows fatal exception: code 0xc0000139" in pytest output comes from DLL loading in `import bpy` / Cycles device listing; the process continues and it does not mean a test crashed.
- 8 GB VRAM cannot run MiniMax-H3 (estimated to need about 80 GB): generation stays on a rented GPU.
- Blender 5 animation API: read keyframes through `bpy_extras.anim_utils.action_get_channelbag_for_slot`, not `action.fcurves`.

## CLI
- `vrecon new | build | handoff | ingest | control | score | score-blend` (see README). Code: `src/vrecon/cli.py`.
- Control videos for V2V: `scene/passes.py` (depth/index via emission override + EXR, shaded pass) -> `control/core.py` (numpy/cv2/ffmpeg). H3 constraints: 24 fps, W/H multiples of 32, 17n+5 frames.
- Phase 1a: `scripts/make_phase1a_scenes.py`, `evaluation/phase1a/`, GPU side in `gpu/h3_control/` (runs in the VideoX-Fun env, not vrecon).
- Scoring: `metrics/` (pure numpy) + `evaluate.py`. Synthetic GT clips: `scripts/make_synthetic_clips.py` -> `evaluation/synthetic/` (gt.json/gt_passes are evaluation-only, never inputs to reconstruction). Baseline B kit: `baselines/b_llm_mcp/` (needs GUI Blender; not runnable in cloud).
- Repair loop: `run/manager.py` (proposals, constraints, accept rule, atomic adoption), `optimize/` (silhouette loss via z-buffer raster of proxy boxes, OpenCV LK point-track loss in `optimize/tracks.py`, pattern search; camera proposals are searched and accepted on set-piece evidence only), `run/policy.py` (fixed hypothesis order = no-LLM condition C, `vrecon run auto`), `run/service.py` shared by `vrecon run ...` and `mcp_server.py` (mcp 2.x `MCPServer`). Masks are a GT stand-in until a segmenter is integrated.
- Phase 2 prep: `evaluation/thresholds.yaml` (JSON syntax, draft until frozen), `scripts/compare_conditions.py` (H1a/H1b tables).
- Read-back design: baseline = readout of the saved .blend at handoff (`scene/readback.py`), diff in pure Python (`scene/ingest.py`).

## Rules
- Only `src/vrecon/scene` calls `bpy`.
- Scene state is the source of the adopted reconstruction; never treat it as ground truth. Never overwrite human edits or `locked` fields.
- Large media is not committed (`runs/`, videos are git-ignored).

## User
- Explain in Japanese, casual, conclusion first, for a non-expert. No exaggeration.
