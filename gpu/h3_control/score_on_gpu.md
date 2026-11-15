# Scoring generated videos (WP-4 step 6)

The scorer (`vrecon score-generated`, `src/vrecon/compare/generated.py`) is CPU only. What it needs from the GPU
machine is the observer output for each generated video, in the existing formats.

## What runs where

| Step | Where | Output |
|---|---|---|
| generation | rented GPU, `run_jobs.py` | `runs/<set>/generated/<run>/video.mp4`, `meta.json` |
| segmentation of each generated video | rented GPU, SAM 3.1 (`gpu/observe/run_sam3.py`, same prompts as the scene's object names) | `masks.npz` per run (masks [T,H,W] labels, id_map) |
| camera from the video (edit_camera, moving tennis views) | optional; 4RC on the rented GPU (`envs/4rc`) | a project JSON with that camera |
| annotator depth of the tennis source window (F-31 baseline) | rented GPU, VideoX-Fun's depth annotator | `evaluation/tennis/controls/source_video_depth.mp4` |
| scoring | anywhere, CPU | `score.json` per run |

Do segmentation only after every generation of a stage is written (one GPU job at a time).

## Install order on the rented machine (after setup_gpu.sh)

1. VideoX-Fun env (generation) - setup_gpu.sh.
2. SAM 3.1 env: `gpu/observe/` (weights need the user's HF access; downloads approved per file).
3. Optional 4RC env.
4. vrecon env (`bash scripts/setup.sh --no-smoke`) for `vrecon score-generated`, or copy masks back and score at home.

## Commands

```bash
vrecon score-generated runs/phase1a/generated/base__depth__s43/video.mp4 runs/rq2_dry/base \
    --masks runs/phase1a/generated/base__depth__s43/masks.npz --scene evaluation/phase1a/scenes/base.json \
    --out runs/phase1a/generated/base__depth__s43/score.json
vrecon score-generated runs/phase1a/generated/edit_object__depth__s43/video.mp4 runs/rq2_dry/edit_object \
    --masks .../masks.npz --base runs/phase1a/generated/base__depth__s43/video.mp4 \
    --unchanged obj_pillar,obj_crate,obj_wall,obj_floor --scene evaluation/phase1a/scenes/edit_object.json
```

The controls folders with passes (index, status) are made locally by `scripts/dry_run_rq2.py` (phase 1a) and
`scripts/export_scene_controls.py` (tennis); copy them with the videos.

## Copy back

`meta.json`, `sheet.jpg`, `masks.npz` (small), `score.json` to git; `video.mp4` to the user's disk.
