# Status (2026-11-19): v1.0 done

Read this first. History: `docs/HANDOFF_EN.md` (and the Japanese `docs/HANDOFF.md`). Plan and decisions:
`docs/PLAN.md` (16.6 = review decisions D1-D8). Review and work packages: `docs/REVIEW_INSTRUCTIONS.md`.

## F002 (MAVEBench, forest staff fight) — closed 2026-11-28 at v11

- Final scene: `runs/mave/F002/scene_v11.blend`; film vs render: `runs/mave/F002/review/compare_v11.mp4`;
  work data (git-ignored): `runs/mave/F002/work/` (the Temp copies were partly deleted on 2026-11-28).
- MAVE eval 86.8 (MAVE baseline 76.5); second GPT review 7/10 at v9. Details and history: `docs/PLAN_v1_1.md` 11.1-11.4.
- Pipeline (all from the clip only): `scripts/v11_mave_prep.py` (camera) -> `gpu/body/run_sam3dbody.py` ->
  `scripts/v11_people_anim.py` (fit-weighted smoothing) -> `scripts/v11_parts.py` -> SAM stick masks
  (`gpu/observe/run_sam3.py`, `scripts/v11_hand_windows.py`, `gpu/observe/run_sam3_points.py`) ->
  `scripts/v11_staffs.py` -> `scripts/v11_colour_gain.py` -> `src/vrecon/scene/mave_blend.py`.
- Known gaps left as they are: staff at 00:09.0 (thin, blurred, crossing a body; no local tool segments it),
  staff holder can switch people, sacks are part of the backdrop, imagined trunks too regular.

## Where things are

- **v1.0 = tennis case study (V2V, RQ2) + RQ1 2x3 study** (user, 2026-11-18). RQ1 runs unattended:
  `bash scripts/rq1_resume.sh` (starts `scripts/rq1_all.sh` unless running/finished; log `runs/rq1/progress.log`).
  Cells S0/S1 (4RC start, fixed-order repair) and L0/L1 (Claude-built scene, + repair); H4 (S2/L2) dropped.
  Done 2026-11-19: verdicts in `evaluation/RESULTS.md` section 8 (H1 supported for 4RC starts only; H2, H3, H5
  not supported by the rule). Tables `runs/rq1/compare/`.
- **V2V works.** ComfyUI on a RunPod pod (RTX 4090 24 GB or 5090 32 GB; network volume in euro-3 with MiniMax-H3
  int8 weights, SAM 3.1). Method: the ComfyUI **r2v** template (`MiniMaxH3ReferenceToVideo`, no ControlNet):
  `<Video 1>` = a clay render of the 3D scene (camera, layout, motion), pictures = frames and crops of the source
  video (look), prompt built from the scene. 1024x896, 30 steps, seed 43. Main runs: `runs/tennis/main/`
  (source camera nearly identical to the source; small dolly and a 90-degree orbit follow the 3D camera).
  Fun ControlNet on proxy depth was coarse and is kept only as `control_depth_legacy`.
- **RQ2 (edits survive generation)**: on phase 1a all three edits (camera, object size, timing) show with both the
  ControlNet and r2v (`gpu/h3_control/RESULTS_2026-11-16.md`).
- **M1 done**: scorer v3, synthetic v2 (12 clips), thresholds frozen at `ff8c6b0`, one time base (WP-2).
- **M2 done**: hypothesis spans and status pass (WP-3), RQ2 scorer `vrecon score-generated` (WP-4).

## How to run tennis V2V

1. Pod: RunPod, euro-3, network volume, RTX 4090/5090, the runpod-slim ComfyUI template. Right after start, run
   the auto-stop command once (`docs/HANDOFF_EN.md`, "pod"); never kill its `sleep`.
2. `vrecon pipeline-v2v evaluation/recipes/tennis/v2v.json --server https://<pod>-8188.proxy.runpod.net`
   (stages: shapes, floor, export, crops, prompts, generate; log in `<work>/pipeline_log.jsonl`).
3. Score: `scripts/score_generated_sam.py` (SAM 3.1 masks on the pod -> IoU, centroid, onset vs the 3D scene).

## Generic pieces added for V2V (scene-independent)

| Piece | File |
|---|---|
| ComfyUI job runner, settings profiles | `gpu/h3_control/comfy_jobs.py`, `gpu/h3_control/comfy/profiles/` |
| SAM 3.1 on the pod (text prompt -> masks npz) | `gpu/h3_control/comfy_sam.py` |
| Scene inventory for prompts (who/what is in shot, cut by the edge) | `src/vrecon/control/inventory.py`, `scripts/make_prompts.py` |
| Reference crops with context | `scripts/make_ref_crops.py` |
| Category shapes for box-only objects | `src/vrecon/analysis/shape_library.py` |
| Floor-contact check from a floor mask | `src/vrecon/analysis/floor_contact.py`, `scripts/fix_floor_contact.py` |
| Colour-coded clay | `control/core.shaded_identity`, export kind `idclay` |
| Pose control from the rig, status pass, hypothesis treatments | `control/pose.py`, `control/status.py` |

## Open

- Labels for unusual objects (how far SAM gets; label-free fallback). Unknown categories keep their box.
- A white shape at the frame edge was drawn as a sneaker once; prompts now say edge-cut parts continue outside.
- Tennis observation -> scene: `vrecon pipeline-scene evaluation/recipes/tennis/scene.json --check` rebuilds
  calibrate, person, proxies, observe, venue, props and matches the stored results (recipe "rebuild_check");
  and further export and status, to the finished scene (RESULTS 10).
- H5 has both halves: edits applied without repair, and scores after the edit (RESULTS 9).
- Repair loop: objects are drawn from their own mesh when they have one (`optimize/shape.py`); a mesh object keeps
  its proportions.

## Rules (unchanged, see CLAUDE.md)

Only `src/vrecon/scene` calls bpy; ground truth is for scorers only; `runs/` is not in git; commit when the user
asks; the user approves downloads and anything sent to outside services.
