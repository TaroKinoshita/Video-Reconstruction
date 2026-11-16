# Before the rental (runbook docs/REVIEW_INSTRUCTIONS.md 10, checked 2026-11-15)

## 10.2 Facts about the generator (looked up, not run)

| Question | Answer | Source |
|---|---|---|
| Example script at the pinned commit | `examples/minimax_h3_fun/predict_v2v_control.py` (and `predict_v2v_control_inpaint.py`) | github.com/aigc-apps/VideoX-Fun/tree/e8c314f/examples/minimax_h3_fun |
| Control branch config | `config/minimax_h3/minimax_h3_control_inpaint_post_norm.yaml` (the v1 config with the 2.0 checkpoint builds half the branch and fails silently) | same script; comfyui-wiki 2026-09-22 |
| Accepted control kinds (Union 2.0) | Canny, Depth, HED, MLSD, Pose (DWPose skeleton), Scribble, Layout (colour boxes per subject on white), Gray (luminance; one source only) | comfyui-wiki 2026-09-22 (secondary; the model card itself could not be fetched) |
| clay / id / look | **not accepted kinds**: removed from the job lists. `id` could become a Layout control (boxes from the index pass); not built | same |
| Preprocessing | none: the control video is used as given (resized/cropped to `sample_size`, `keep_aspect_ratio=True`) | predict_v2v_control.py |
| Reference image with a control video | **no** in this script (`ref_image=None`); the ref2va transformer is a separate model file, not set up | predict_v2v_control.py |
| Control strength | `control_context_scale` (1.0 strongest, 0 off), one number for the whole video | predict_v2v_control.py |
| Per-pixel control weight | **none**. The inpaint script takes `inpaint_video` + `inpaint_video_mask` (white = repaint, black = keep): a possible later route for "keep the source pixels where observed" | predict_v2v_control_inpaint.py |
| Memory modes | `model_group_offload` (default), `model_cpu_offload`, `model_cpu_offload_and_qfloat8`, `sequential_cpu_offload` | predict script, README |
| Length, rate | 24 fps fixed; frames snapped to 17n+5; at most 15 s (the README's 243-frame cap is the script default) | comfyui-wiki |

Consequence for F-24 (hypotheses): with no per-pixel weight, the treatments stay what `control/status.py` does to the
control pixels (`full` / `soft` / `drop`).

## 10.3 Pose control

Built: `control/pose.py` (rig joints -> OpenPose COCO-18, mapping table in the module), export kind `pose`.

## 10.4 Job files

- `evaluation/phase1a/jobs.json` (exists). `run_jobs.py --dry-run` lists the 12 main runs; the control mp4s are
  committed under `evaluation/phase1a/controls/`.
- `evaluation/tennis/jobs.json`: stages E and F, a 124-frame window (scene seconds 5.26-10.43, the second serve).
  Prompts approved by the user (2026-11-15). Copy `runs/venue/tennis/v2v_ready/*` to `evaluation/tennis/controls/`
  on the rented machine. `source__video_depth` needs the annotator depth of the source window, made there.

## 10.5 Depth check

Player's grey-level span (2-98%) in the source view, `controls_source` of the finished scene: **4 levels** with
the clip mode (median over frames, 2-7) and 4 with the frame mode; the stands at 30-60 m take the range. Under the
runbook's 20-level line. Added `depth_control(mode="disparity")` (inverse depth, as monocular estimators output):
**18 levels** (10-35). Tennis jobs carry both (`source__depth`, `source__depth_disparity`).

## 10.6 Cost cap and stop rule

User, 2026-11-15: expects about 1-2 dollars per rented hour. Not yet fixed: the total cap and the stop rule.
Note against the user's "24-32 GB VRAM": the README estimate is about 80 GB (transformer bf16 ~62 GB); 24-32 GB
would need `model_cpu_offload_and_qfloat8` or `sequential_cpu_offload` plus lots of RAM, untested and much slower.
Stage A (one job) decides this on the machine.
User, 2026-11-15: start with the smallest, a 32 GB GPU, and verify there (stage A with
`--memory-mode model_cpu_offload_and_qfloat8`; RAM as large as offered, disk 250 GB). Move to 80 GB if it does not
load or one job is far too slow.

## 10.7 D8

Recorded in `docs/PLAN.md` 16.6: sending to Claude, fal and the rented GPU allowed for research; results not
published.

## Stage A, measured (2026-11-16, RunPod RTX 4090 24 GB, 124 GB RAM, ComfyUI v0.44.0)

ComfyUI template "MiniMax H3 Fun ControlNet Union" (`gpu/h3_control/comfy/template_ui_stageA.json`), unchanged
except the video VAE (fp16): `minimax_h3_ref2va_pruned_int8_convrot` + `minimax_h3_fun_controlnet_union_pruned_int8_convrot`
(model_patches, 2.14 GB) + text encoder `qwen3vl_32b_minimax_h3_nvfp4_awq` + video VAE fp16. The template's own pose
example (SDPose extraction from `dancer_field_pose.mp4`), 1344x768, 124 frames, 20 steps `res_multistep`, turbo LoRA
switched off, control strength 1. **Ran to the end on 24 GB**: 245 s for the first run including all loading and
the pose extraction (32 s); sampling 80 s (4.0 s/step). No out-of-memory. Generation stays in ComfyUI (not
VideoX-Fun): `run_jobs.py` is replaced by a ComfyUI API runner for this machine.
