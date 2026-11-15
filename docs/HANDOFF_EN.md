# Handoff (English, 2026-11-15)

Start here in a new session. This file summarises everything up to 2026-11-15: what the project is, what was
built, what was measured, what was decided, where things are, and what to do next. The detailed records stay
in the Japanese documents listed in section 10; this file points into them.

---

## 0. Where things stand (read this first)

- **The tennis Blender scene is finished for V2V.** Final scene and control videos:
  `runs/venue/tennis/fal2c_props/final/` (`scene.blend`, `project.json`, `controls_<move>/control_{depth,canny,clay,id,look}.mp4`
  for five camera moves; all pass H3's constraints: 24 fps, sides multiples of 32, 17n+5 frames). `runs/` is not in git.
- **An outside design review was done** (Claude Fable 5.2, commit `0a4d0f3`): `docs/REVIEW_INSTRUCTIONS.md`.
  It lists findings F-01..F-31, eight decisions for the user (D1..D8), work packages WP-0..WP-9, a stop/defer list,
  RQ1 protocol v2 and a V2V session runbook. Its section 0 says what was already done (scorer v2, schema 0.4.0).
- **M1 done (2026-11-15).** WP-0, WP-1 (scorer v3, re-score of 100 stored runs in `evaluation/RESULTS.md` 5,
  synthetic v2 in `evaluation/synthetic_v2/`, bootstrap comparison, `src/vrecon/tuning.py`) and WP-2 (padding
  removed at `vrecon new`, `schema.frame_time`). **`evaluation/thresholds.yaml` frozen at `ff8c6b0`** (commit
  `828cc2a`): from now on a change to scoring rules or `tuning.py` must be written into its `changes`.
  Findings recorded in `docs/REVIEW_INSTRUCTIONS.md` 12: L1 ("B + tools .881") used ground-truth masks; the
  "body still one frame in five" claim does not hold; tennis container rate is 60/1 (fixed).
- **Next: M2** (WP-3 hypothesis spans and export variants, WP-4 RQ2 scorer, pre-rental checks of runbook 10.2-10.6).
  Starting points: `scripts/fit_props.py:133,170` still write `status="estimated"` for racket and ball;
  `scene/controls.export_controls` has no status pass; `src/vrecon/compare/` is empty; VideoX-Fun pinned at
  `e8c314f` (`gpu/h3_control/README.md`); phase-1a scenes and `edits.json` in `evaluation/phase1a/`.
- **D1..D8 answered on 2026-11-15**, recorded in `docs/PLAN.md` 16.6. All as recommended except D6 (keep the
  12 stock real clips, no self-shot clips with truth) and D7 (`runs/` and `.venv` stay inside OneDrive, no
  `VRECON_RUNS`). Next: WP-0 (`evaluation/RESULTS.md`).
- The user's own wish was "finish the Blender scene before V2V" (done) and then Phase F (V2V on a rented GPU).
  The review recommends doing WP-1, WP-3 and WP-4 (CPU only) before the GPU session, and running Phase 1a first
  inside that same rental (D4). The user has not decided.
- Tests: 160 pass (`.venv/Scripts/python.exe -m pytest -q`, about 80 s). Working tree clean at `0a4d0f3` + this file.

## 1. The project

- Goal: turn a reference video into an **editable Blender scene** (camera, layout, motion, timing), then use
  control videos rendered from that scene (depth, edges, etc.) for **video-to-video generation** (MiniMax-H3 +
  Fun ControlNet Union 2.0). Research, non-commercial. User: an individual making short CG shots in Blender.
- Research questions (`docs/PLAN.md` ch. 2):
  - RQ1: do structured diagnostics + dedicated repair tools (condition C) reconstruct better than a plain LLM +
    Blender MCP (condition B)?
  - RQ2: how much of an edit made in 3D survives into the generated video?
- Milestone M1 (PLAN 3.1): single-cut RQ1 result (M1-core) + RQ2 measured (M1-V2V). Stages in PLAN ch. 12.

## 2. Environment and rules (also in CLAUDE.md)

- Windows 11, i7-13700H, 32 GB RAM, RTX 4060 Laptop **8 GB VRAM**. Blender 5.2; Python 3.13 venv `.venv` (bpy 5.2.3).
- Repo path contains Japanese and is under OneDrive: never pass paths to OpenCV file functions; use
  `control/core.py` (`load_frames`, `read_video_rgb`, `encode_mp4`) or `cv2.imencode(...).tofile(...)`.
  Always `encoding="utf-8"`.
- Only `src/vrecon/scene` calls `bpy`. Scene state is the adopted reconstruction, never ground truth. Never
  overwrite human edits or `locked` fields.
- Observers run in separate environments: `%LOCALAPPDATA%\vrecon\envs\observe` (SAM 3.1, TAPNext++), `envs\4rc`,
  WSL `/home/taro/vrecon/work` for GVHMR (Hydra rejects non-ASCII paths). H3 needs ~80 GB: rented GPU only.
- **Run only one GPU job at a time.** SAM and Cycles rendering together slowed both by tens of times (SAM went
  from ~10 s/frame to hours per frame). SAM alone on the 312-frame tennis clip: ~50-80 min for a full pass;
  short windows (props_recut) a few minutes.
- Ground truth (`gt.json`, `gt_passes/`, evaluation `meta.json`) is for scorers only; never give it to an LLM or agent.
- Licences: 4RC (S-Lab, non-commercial), SMPL-X body data stays under `runs/`, MoRe has no licence (not used).
- External services: fal (`FAL_KEY` is the user's; only check presence, never read or print it; requests use
  `X-Fal-Store-IO: 0` and `sync_mode`; do not disable the safety checker), `claude -p` for LLM steps. Never read the
  HF token file. Downloads need the user's approval (file, source, size). The user handles registrations/licences.
- Large media are not committed (`runs/`, big videos). Small synthetic evaluation clips are committed.
- **User preferences:** explain in Japanese, casual, conclusion first, for a non-expert, no exaggeration. Give full
  absolute paths when pointing to files. Commit when a piece of work is done (the user often says "コミットして").

## 3. Code layout

```
src/vrecon/ (~10.4k lines)
  schema/     versioned JSON model: Project, SceneObject, Camera, TransformKey, Evidence, status, locked.
              Schema 0.4.0 (review): Status = observed/estimated/standard/hypothesis/human_confirmed,
              SceneObject.spans + status_at(), Evidence.kind, Scale.units_per_metre + basis
  scene/      Blender: builder (build, render, mesh .npz assets), character (SMPL-X rig), passes/color_passes/
              controls (control videos), readback/ingest/handoff (human edits), look, convert, coverage
  run/        manager (proposals, constraints, atomic adoption), policy (fixed order = condition C without LLM),
              service (shared by CLI and MCP; align_timing, hold_still), look/view (tools that show an LLM pictures),
              complete (filling unseen space), refine
  optimize/   silhouette loss (z-buffer raster of boxes), LK track loss, pattern search, rays (off)
  metrics/    numpy scorers (IoU, camera, reprojection, timing, novel views, venue); evaluate.py (scorer v2)
  analysis/   observations -> scene: from_4d, from_gvhmr, place_person, proxies, background, planes, court,
              to_court, observations, appearance, triage, units, props, prop_fit, prop_table, prop_shapes, rig_fk
  venue/      layout JSON, part generation, fit to observations, build
  control/core.py  control video encoding, H3 conformance
  cli.py, mcp_server.py
scripts/ (~5k lines) experiment and pipeline scripts (see section 6)
gpu/observe/   run_sam3, run_gvhmr, gvhmr_body_mesh, run_tapnextpp, run_megasam, run_more, select_movers
gpu/h3_control/ H3 job runner for the rented GPU (never run yet)
baselines/     condition B kits (Claude + MCP, GPT + Codex)
evaluation/    synthetic (6 clips), props (4 clips), phase1a, venue, repair_benchmark, thresholds.yaml (draft)
```

## 4. History (condensed)

| Dates | Work | Result |
|---|---|---|
| 11-01..06 | Plan, schema (4 layers), Blender builder, read-back, control videos, 6 synthetic clips with truth, scorers, repair loop, CLI/MCP | Stage 0 done; numeric search recovers focal length on all 6 |
| 11-07..09 | Observers: SAM 3.1, 4RC start scene, trackers (LK / CoTracker3 / TAPNext++), MegaSaM and MoRe compared | 4RC kept (best object IoU). CoTracker3 removed, LK kept. 4RC focal off by 12-29%; repairing focal from a 4RC start makes it worse, so focal is fixed |
| 11-09..12 | Condition C with Claude and GPT; condition B partly; tool defects fixed (camera accept guard 0.03, fixed pairing, floor/wall fixed) | C: fixed order .837, Claude .827, GPT .830 (mean IoU, 6 clips). B: syn_01 .945 / .940, syn_02 .912 (B well above C). GPT B for 4 clips stopped by a credit limit (invalid, to redo) |
| 11-12 | Real stock clips end to end, 2D layers, triage | Clips where 4RC tilts the floor cannot be repaired |
| 11-13 | Tennis clip (312 frames, nearly static camera): GVHMR + SMPL-X body, SAM, 4RC, proxies, background, completion, court calibration (white lines -> ITF court), venue layout by LLM, control export | Focal fixed at 2104 px, court metres; player height 1.84 m (real 1.85) |
| 11-13..14 | LLM view/look tools, timing (align_timing, hold_still), phase E (fal GPT Image 2.5 Sunburst outside pictures -> LLM layout), look check (fal_restyle), clothing colours, seated crowd, racket and ball, generic props | Timing error 7.8 -> 6.1 frames (6 clips). fal outside pictures: no IoU gain (.70 -> .68), judge 3/7 -> 5-6/7 |
| 11-14 | Review brief (`docs/REVIEW_BRIEF.md`), outside review (`docs/REVIEW_INSTRUCTIONS.md`), scorer v2, schema 0.4.0 | See section 0 |

## 5. Key numbers

- Condition C (6 synthetic clips, mean source IoU): fixed order .837, Claude .827 (3 runs each, SD <= .009),
  GPT .830; budget-forcing .829. Repair benchmark (truth + noise -> repair): mean .906. 4RC start, focal fixed: .840-.890.
- Condition B: syn_01 .945 (Claude) / .940 (GPT), syn_02 .912 (GPT). Review: B .862 -> B + tools .881 (6 of 7 better).
- Timing (6 clips): looker 7.8, viewer 7.8, +align 6.75, +hold 6.08 frames; syn_04 stays ~17 (camera truck not
  reconstructed, objects move instead; tennis is unaffected).
- Venue (synthetic ven_01, stands IoU in new views): visible only .08, LLM .62/.65, hand-made .88. With fal
  outside pictures .68 vs .70 without. Tennis judge (good answers of 7): none 3, manual 2-4, fal 5/6.
- Props (synthetic, `evaluation/props`, `scripts/eval_props.py`): toss 1.6-3 cm, throw-catch flight 1.8 cm
  (release 21 vs truth 20, catch 44 vs 44), at rest after put-down 0.7 cm (put-down 26 vs 30), held still with no
  contact 35-48 cm (size depth only), after a hit 0.5-1.1 m, racket direction 1.2-2.0 deg, frame-to-frame jitter 0.2 deg.
- fal costs so far: outside pictures ~$0.26, restyles ~$0.03-0.04 per 3 views, total well under $1.

## 6. The tennis pipeline (how the final scene was made)

Inputs/stores under `runs/human/tennis/` (video `tennis.mp4`, `masks.npz`, `4rc.npz`, `gvhmr_skeleton.npz`,
`props_masks.npz`, `recut/ball_masks.npz`), venue under `runs/venue/tennis/`. The review notes (F-02, F-03) that this
cannot be rebuilt from the repository alone; WP-5 is to turn it into a recipe.

1. Observe: SAM 3.1 (`gpu/observe/run_sam3.py`), 4RC, GVHMR (WSL) + SMPL-X body (`gvhmr_body_mesh.py`).
2. Calibrate: `scripts/calibrate_court.py` (white lines -> ITF court; focal fixed 2104 px, rotation per frame);
   4RC points moved onto the calibrated rays (`analysis/to_court.py`); units in `units.json`.
3. People: `scripts/place_person.py` / `proxies_scene.py` (GVHMR on scene rays, feet on the floor; others as
   standing/seated figures; clothing colours measured, skin-like colours replaced by plain ones).
4. Venue: `scripts/observe_venue.py` -> `scripts/llm_venue.py` (claude -p, Read only; optional fal outside
   pictures `scripts/fal_outside.py`) -> `scripts/build_venue.py` (layout `runs/venue/tennis/fal2c/layout.json`,
   crowd 0.08). Standard objects (court, net) at regulation positions.
5. Completion and protection of the source view: `scripts/complete_scene.py`, `run/complete.py`.
6. Timing: `align_timing`, `hold_still` (`run/service.py`).
7. Props (`docs/PROPS_PLAN.md`):
   - `scripts/props_recut.py prepare|merge`: windows around a joint (left wrist), auto-proposed where the arm
     is raised or fast (85-158, 187-260 = the two serves); SAM run per window (`--keep-raw` keeps all tracks).
   - `scripts/fit_props.py PROJECT VIDEO OUT --prop "tennis racket=runs/human/tennis/props_masks.npz:obs_0"
     --prop "tennis ball=runs/human/tennis/recut/ball_masks.npz" --one "tennis ball"
     --after runs/venue/tennis/tennis_after.json --link 180` (OUT = `runs/venue/tennis/fal2c_props`).
   - Racket: right hand, pose from silhouette IoU, smoothed over time (sigma 1.5), interpolated over unseen
     frames, racket-shaped mesh (a box was drawn as a board by the image model).
   - Ball: serve 1 held in the left hand throughout (no toss; a shadow serve); serve 2 toss seen 4 frames ->
     physics -> strike at frame 236 (hypothesis: highest racket head; the source shows the racket highest at
     234-236) -> into the service box at 45 m/s (hypothesis, clears the net at 0.99 m). After the strike the ball
     is never visible in the source. Other unseen frames: in the left hand (hypothesis).
8. Export: `scripts/export_scene_controls.py PROJECT ROOT OUT --moves source,dolly-100+crane4,orbit90+dolly-60,crane10,orbit180`.
   Check views: `scripts/clay_views.py ... --kind look`. Look check: `scripts/fal_restyle.py` (all 6 views draw
   the racket as a racket and the ball as a tennis ball).

Known facts about the tennis clip: every 5th frame repeats the previous one (24 -> 30 fps padding). Props use
real frame times (`analysis/props.frame_times`, cadence-based); GVHMR, timing and the 24 fps control export do
not (review F-22, WP-2).

## 7. Decisions made so far (and why)

| Decision | Why |
|---|---|
| 4RC as start scene; focal fixed for 4RC starts | Best IoU vs MegaSaM/MoRe; focal repair trades against other errors |
| LK point tracks | TAPNext++ worse on synthetic, CoTracker3 under-tracks motion |
| Camera proposals judged on floor/wall evidence; accept only if total silhouette loss does not rise (+0.03 tolerance); fixed pairing; floor/wall fixed | Closed loopholes where loss fell while truth got worse |
| Regulation objects at regulation positions | 4RC put the net 1.9 m off |
| Unseen parts marked hypothesis with evidence | Separate observation from guess |
| Image generation only as a reference/look check; geometry comes from the scene | Generated pictures are not observations |
| Props handled by states held / free / unseen | Fitting without states gives false answers (a parabola on a held ball was off by 50+ px) |
| Contact points from the rendered rig (`analysis/rig_fk.py`), not GVHMR ray points | Ray points were 0.2-0.5 m off in depth; the racket floated off the hand in side views |
| One GPU job at a time | Measured slowdowns |

## 8. Open items

From the review (`docs/REVIEW_INSTRUCTIONS.md`), in its recommended order:
- **D1..D8 user decisions** (section 6 of the review): RQ1 claim (factorial start scene x finishing stage),
  timing metric (relative onset), headline accuracy set, GPU session order (Phase 1a first), hypotheses in
  controls (`full`/`soft`/`drop`), real clips (four self-shot with measured truth), where `runs/` and `.venv`
  live (outside OneDrive via `VRECON_RUNS`), rights/sending of the tennis clip.
- WP-0 baseline snapshot; WP-1 measurement hygiene (steps 1-2 done: `mask_iou_all_mean`, relative onset);
  WP-2 one time base (repeated frames); WP-3 status spans and hypothesis-aware export (schema done; props still
  `estimated` as a whole, `units_per_metre` not filled); WP-4 RQ2 scorer; WP-5 tennis as a recipe; WP-6 one GPU
  session (Phase 1a + tennis V2V); WP-7 RQ1 on held-out clips; WP-8 hardening; WP-9 documents.
- Stop/defer list (review section 8): stop fal outside pictures for layout, look/view tools for repair order,
  ray placement, MegaSaM/MoRe; defer venue polish, more sports, LLM-written prop tables, 2D layers, multi-cut.

Older open items still true (WP-0 found GPT condition B complete for all six clips, `runs/gpt/b/*/score.json`;
the "redo syn_03..06" item is closed, see `evaluation/RESULTS.md`): `evaluation/thresholds.yaml`
not frozen; no real test clips with truth; syn_04 camera translation; `docs/DIRECTION_OPTIONS.md` superseded in
practice by the review's protocol v2 but not formally closed.

## 9. Useful commands

```bash
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe scripts/eval_props.py evaluation/props/prop_03 --out runs/props/prop_03.json
.venv/Scripts/python.exe scripts/repair_benchmark.py            # see evaluation/repair_benchmark/
bash scripts/run_llm_repair.sh RUN_DIR --clip syn_01 [--viewer] [--use-budget]
.venv/Scripts/python.exe scripts/fal_restyle.py CONTROLS_ROOT OUT --frame 120 --moves source --reference runs/venue/tennis/source.png
```
Setup: `bash scripts/setup.sh --no-smoke` from Git Bash (PowerShell `bash` starts WSL).

## 10. Documents

| File | Language | What |
|---|---|---|
| `CLAUDE.md` | en | Environment and rules |
| `docs/PLAN.md` | ja | Authoritative plan (ch. 2 questions, 11 evaluation, 12 stages, 16 revisions) |
| `docs/ARCHITECTURE.md`, `docs/CONVENTIONS.md` | en | Responsibilities, repair loop; axes, units, rotations, time |
| `docs/HANDOFF.md` | ja | Detailed running log (last section: state before V2V) |
| `docs/REVIEW_BRIEF.md` | ja | What the reviewer was given |
| `docs/REVIEW_INSTRUCTIONS.md` | en | **The review: findings, decisions, work packages, rules for the executing agent** |
| `docs/DIRECTION_OPTIONS.md` | ja | Earlier options for RQ1 direction |
| `docs/VENUE_PLAN.md` | ja/en | Whole-venue reconstruction, phases A-E, view tools, timing |
| `docs/PROPS_PLAN.md` | ja | Generic props: states, parts, numbers |
| `evaluation/repair_benchmark/NOTES.md`, `evaluation/THRESHOLDS_PROPOSAL.md`, `evaluation/clips/SELECTION.md` | | Benchmarks, threshold proposal, clip selection |

## 11. Suggested first message for the new session

> Read `docs/HANDOFF_EN.md`, then `docs/REVIEW_INSTRUCTIONS.md` sections 0, 2, 6, 7 and 11. Ask me decisions D1..D8
> in Japanese, one at a time, recommendation first, and record them in `docs/PLAN.md` 16.6. Then start WP-0.
