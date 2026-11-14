# Design review and work instructions (2026-11-14)

Response to `docs/REVIEW_BRIEF.md`. Written for the agent (or person) who will carry out the work.
Nothing in the repository was changed by the review except this file. Nothing is committed.

Contents

1. How to use this document
2. Verdict
3. What the review is based on
4. Findings (F-01 ... F-31), each with problem / evidence / fix / weight
5. Answers to the nine review questions
6. Decisions the user has to make (D1 ... D8)
7. Work packages in order (WP-0 ... WP-9)
8. Stop and defer list
9. RQ1 protocol v2 (text to adopt into PLAN.md once D1 is decided)
10. V2V session runbook
11. Rules for the executing agent
12. Not verified by this review

---

## 0. Done so far (2026-11-14, not committed)

Foundation pieces only; every older number and key keeps its meaning. Tests: 160 passed.

| Package | Done | Left |
|---|---|---|
| WP-1 step 1 | `summary.mask_iou_all_mean` (a lost object counts 0) and `matched_share`. Differs from the text below: `mask_iou_mean` was left as it is (matched objects only) so stored numbers do not shift; which one is the headline is decided at the freeze | centroid error over all objects |
| WP-1 step 2 | `metrics.onset_frame_relative`, `report["timing_relative"]`, `summary.timing_rel_error_frames_mean`, `timing_rel_found_share`; the reference onset is measured by the same rule on `gt.json`. `measures.scorer_version = 2` | D2 (which rule is the headline) |
| WP-1 steps 3-8 | nothing | novel views in the report, re-scoring stored runs, held-out set, statistics, freeze |
| WP-3 step 1 | schema 0.4.0: `Status` = observed / estimated / standard / hypothesis / human_confirmed, migration `assumed` -> `hypothesis`, `SceneObject.spans` + `status_at()`, `Evidence.kind`, `Scale.units_per_metre` + `basis` | |
| WP-3 step 2 | `venue/build.py` keeps the layout's status and sets the evidence kind; `run/complete.py` writes `hypothesis` with kind `rule` | `fit_props.py` spans (the ball and racket are still `estimated` as a whole); nothing fills `units_per_metre` yet |
| WP-3 steps 3-5 | nothing | status pass, hypothesis-aware export, converter for the tennis scene |
| WP-8 | `history.jsonl` opened as utf-8; `allow_pickle` removed in `fit_props.py`, `props_recut.py` | the rest |

Check on stored condition-B scenes (GPT, run 1): syn_01 IoU .940 on both IoU measures (nothing lost), timing
2 frames by the old rule and 0 by the relative rule; syn_05 IoU .710 on both, no onset found by either rule
(the objects do not move). On these two, F-14 changes nothing; F-15 does.

## 1. How to use this document

- Section 6 comes first in practice: D1-D4 change what the work packages mean. Ask the user, record the
  answers at the top of `docs/PLAN.md` section 16 (new subsection 16.6), then start WP-0.
- Work packages are ordered by dependency and by cost. WP-0 to WP-5 need no GPU and no external service.
  WP-6 is the rented-GPU session. Do not reorder WP-6 before WP-1, WP-3 and WP-4: the session is the only
  expensive step and those three decide what it must produce.
- Every work package has an acceptance check. A package is done when the check passes, not when the code exists.
- Weights used in section 4: **rebuild** (the current shape cannot reach the goal), **improve** (works, but a
  specific defect costs correctness or reproducibility), **preference** (style, would not block anything).
- User-facing explanations stay in Japanese, casual, conclusion first (CLAUDE.md). This file is the working
  document and stays in English.

## 2. Verdict

1. The engineering core is sound: strict versioned schema, atomic saves, a run manager that rejects undeclared
   and locked changes, one service behind CLI and MCP, observers isolated in their own environments, 156 tests
   passing in 75 s. None of that needs rebuilding.
2. The project has split into two lines that no longer touch. Line P is the study (RQ1 on six synthetic clips).
   Line T is the tennis scene. The final tennis `project.json` has `revision 0`, 0 observations, 0 artifacts,
   0 events, and no evidence entry that points at an observation: the four-layer state and the repair loop, which
   are the subject of RQ1, were not used to build the flagship result (F-01).
3. H1a as written ("same budget, C more accurate than B") is false on the development clips, and the comparison
   that showed it is confounded in at least six ways (F-10). The data do support a different, narrower claim:
   the tools improve whatever scene they are given, cut the spread of bad runs, and cost nothing in LLM tokens
   (B .862 -> B + tools .881, 6 of 7 runs better, none worse). Reframe RQ1 as a factorial of
   start scene x finishing stage (section 9).
4. The LLM earned its keep where it supplied knowledge the pixels do not hold (net length, symmetric chair,
   venue layout: stands IoU .08 -> .62) and added nothing where it only chose the order of numeric repairs
   (.827 / .830 against .837 for the fixed order). Design the LLM's role around the first, not the second.
5. The largest unmeasured risk is V2V. No generation has ever been run. Phase 1a has been ready since
   2026-11-01. The code that would score a generated video against the 3D scene (RQ2) does not exist (F-25).
   Everything built for line T since 11-13 rests on an untested assumption about what the generator does with
   proxy controls.
6. The headline metric has a scoring defect (objects that could not be matched are left out of the mean, F-14),
   the timing metric has a known defect (F-15), the improvement band of +-.02 is far inside B's run-to-run
   spread (SD .07-.13, F-16), and every tuned constant was chosen on the same six clips that produce the
   headline numbers (F-17). Fix these before any number is called a result.
7. "Hypothesis" is not a status in the schema. It is the word `assumed` for whole objects, and free text in
   evidence notes for parts of an object's timeline. Nothing downstream can tell the observed toss from the
   invented flight after the strike (F-05, F-06).
8. The tennis scene cannot be rebuilt from the repository. The recipe is spread over HANDOFF.md prose, about ten
   scripts, files found by walking up parent folders, and a hand-written JSON under `runs/` (F-02, F-03).

Order of work that follows from this: measurement hygiene (WP-1), time base (WP-2), status and hypothesis
handling (WP-3), RQ2 scorer (WP-4), tennis recipe (WP-5), one GPU session (WP-6), RQ1 factorial on held-out
clips (WP-7), then structure and documents (WP-8, WP-9).

## 3. What the review is based on

Read in full: CLAUDE.md, REVIEW_BRIEF.md, PLAN.md, ARCHITECTURE.md, CONVENTIONS.md, HANDOFF.md,
DIRECTION_OPTIONS.md, VENUE_PLAN.md, PROPS_PLAN.md, `evaluation/thresholds.yaml`,
`evaluation/THRESHOLDS_PROPOSAL.md`, `evaluation/repair_benchmark/NOTES.md`, `evaluation/clips/SELECTION.md`,
`gpu/h3_control/README.md`.

Code read in full: `schema/project.py`, `schema/common.py`, `schema/io.py`, `run/manager.py`, `run/policy.py`,
`run/service.py`, `optimize/__init__.py`, `metrics/core.py`, `evaluate.py`, `control/core.py`,
`scene/controls.py`, `scripts/make_synthetic_clips.py`, `scripts/export_scene_controls.py`,
`scripts/fit_props.py`, `scripts/run_llm_repair.sh`, `baselines/b_llm_mcp/run_baseline.sh`, `prompt_L1.txt`.
Read in part: `scripts/proxies_scene.py`, `gpu/h3_control/run_jobs.py`, `analysis/props.py`, CLI and MCP tool lists.

Run: `pytest -q` (156 passed, 74.7 s). A read-only summary of
`runs/venue/tennis/fal2c_props/final/project.json`. Greps for hard-coded ids and imports. No GPU work, nothing
sent to an external service, no key files read, nothing under `runs/` written.

## 4. Findings

### A. Architecture and data (question 1)

**F-01. The flagship scene bypasses the architecture.** Weight: rebuild (of the tennis path, not of the core).
- Evidence: `runs/venue/tennis/fal2c_props/final/project.json`: `revision 0`, `observations: []`,
  `artifacts: []`, `events: []`, no locks or edits; 30 objects; 0 evidence entries with an `observation_id`;
  no run directory, history or snapshot behind it. Each script loads a project, edits it in memory and saves a
  new `project.json` in a new folder (`fit_props.py:176-180`, `export_scene_controls.py:68`).
- Consequence: the scene has no record of which mask file, which 4RC output, which calibration or which LLM
  answer produced an object. PLAN 5.1 (four layers), 6.1 (candidates tried before adoption) and 10.5
  (provenance) do not hold for the one real result.
- Fix: WP-5. Each stage becomes a function `stage(project, inputs, params) -> (project, record)` that registers
  its inputs as `Observation`s (file, sha256, `Provenance`) and its outputs as `Artifact`s, bumps `revision`,
  and is driven by a recipe file per clip. Hypothesis-adding stages (venue, completion, props carry-on) submit
  through `RunManager.try_proposal` with an accept rule that says "observed evidence must not get worse"; the
  source-view protection in `run/complete.py` is already such a rule, written by hand.

**F-02. No reproducible recipe for tennis.** Weight: improve (blocks iteration after V2V).
- Evidence: the commands are in HANDOFF.md paragraphs and script docstrings; `scripts/` is about 5,000 lines; `proxies_scene.py` is 368 lines with closures inside `main()`; five scripts import each other
  through `sys.path.insert`; scripts import private names (`_rot`, `_confident`, `_yaw_quat`, `_rays`,
  `_load_map`); `tests/` exercises three scripts.
- Fix: WP-5 (recipe + stage functions moved into `src/vrecon/pipeline/`), then thin the scripts to argument
  parsing.

**F-03. Per-clip values and generic code are not separated.** Weight: improve.
- Evidence: focal 2104 px fixed, ITF dimensions in `analysis/court.py`, `--kinds obj_net=board,...` defaults in
  `proxies_scene.py:32`, `--person obj_person_01`, `--rise-m 0.45`, crowd share 0.08,
  `runs/venue/tennis/tennis_after.json` (hand-written, git-ignored), `--link 180`, `--hand-px 35`.
  `analysis/units.units_per_metre` finds the scale by walking up parent folders until it meets `units.json` or
  `scene/placement.json`.
- Fix: one `recipe.json` per clip, committed under `evaluation/recipes/<clip>/` (small, no media), holding every
  per-clip value with a `basis` (measured / standard / human / llm / default). Put `units_per_metre` and its
  basis into `Reconstruction.scale` (schema change, F-07) and delete the folder walk.

**F-04. The optimiser's reach is structurally small.** Weight: improve (rebuild only if D1 keeps "C beats B").
- Evidence: `optimize.Param` kinds are translation (one rigid offset for rest and all keys), size, focal, time
  shift, yaw, spin (`optimize/__init__.py:179-190`). No camera position or path parameter, no per-key path, no
  tilt, no add or remove. `"shot_001"` is hard-coded in 11 places in `src/`; `cameras[0]` appears 12 times in
  `src/` and 17 times in `scripts/`.
- Consequence: syn_04 (truck) cannot be fixed by construction: nothing in the search can move the camera
  sideways, so the residual goes into object drift (HANDOFF "syn_04's 15 frames"). The single-shot assumption
  also sits under PLAN stages 3-4.
- Fix: WP-7 step 1 is a diagnosis, not a feature: measure on syn_04 whether 4RC's camera path is wrong
  (`camera_center_rel_mean` of the start scene) or the path is right and the conversion loses it. Only then add
  a low-dimensional camera-path parameter (scale of the translation path; lateral, forward, vertical offsets
  linear in time), judged on set-piece evidence plus track parallax, under `CAMERA_RULE`. Replace the
  hard-coded shot and camera with arguments in the same pass.

**F-05. "Hypothesis" is not a status.** Weight: improve (needed before V2V, see F-24).
- Evidence: `schema/common.py:110`: `Status = Literal["estimated", "assumed", "human_confirmed"]`.
  `venue/build.py:33` maps layout `standard` and `observed` to `estimated` and `hypothesis` to `assumed`
  (standard and observed become indistinguishable). `fit_props.py:133,170` writes `status="estimated"` for the
  racket and the ball although 84 racket frames and the ball's whole flight after the strike are invented;
  the word appears only inside `Evidence.note` strings. In the final scene 12 of 30 objects are `assumed`,
  and 14 carry the word "hypothesis" in a note.
- Fix: WP-3. Schema 0.4.0: `Status = observed | estimated | standard | hypothesis | human_confirmed`
  (migration: `assumed` -> `hypothesis`); `SceneObject.spans: list[StatusSpan]` with `start`, `end` (seconds),
  `status`, `basis` for objects whose status changes over time; `Evidence.kind` =
  `observation | standard | rule | physics | llm | generated_image | human`.

**F-06. Evidence is used as a per-frame log.** Weight: improve.
- Evidence: `obj_tennis_racket` has 136 evidence entries, most of them the single word "swept"
  (`fit_props.py:131`, capped at 400). `project.json` is 0.86 MB.
- Fix: per-frame diagnostics go to an external file referenced by `FileRef` (PLAN 5.1 already says bulk data
  stays out of the JSON); evidence keeps one entry per span.

**F-07. Scene units contradict the conventions.** Weight: improve.
- Evidence: CONVENTIONS section 1 says metres. A 4RC start scene is in the model's own units (about 1/7 m,
  HANDOFF 2026-11-08), B's scenes differ by about 2.5x (DIRECTION_OPTIONS). `Scale` has `source` and `note`
  only. `metrics.core.onset_frame` uses `eps = 0.01` "metres" on such scenes (F-15); `policy.MOVE_STEP` had to
  be made relative to camera distance for the same reason.
- Fix: `Scale.units_per_metre: float | None` and `Scale.basis`; every threshold in scene units is either
  relative or multiplied by it; `None` means "unknown" and forbids absolute thresholds.

**F-08. Stale scaffolding and documents.** Weight: preference, except the status table.
- Evidence: `src/vrecon/planning`, `generation`, `compare` are one-line packages; `configs/modules.yaml` is read
  by nothing; ARCHITECTURE.md lists `run` and `optimize` as planned; HANDOFF.md section 2 still says 88 tests
  and "B not run"; REVIEW_BRIEF section 3 and 6 say GPT's B stopped after two clips while DIRECTION_OPTIONS and
  the end of HANDOFF report all six (.862) plus four repeats. `scripts/make_*_clips.py` import `bpy` directly
  (rule: only `src/vrecon/scene`).
- Fix: WP-9.

**F-09. Small defects in the run service.** Weight: preference.
- `service._manager` reloads masks and recomputes LK tracks on every CLI or MCP call; cache tracks in the run
  directory. `config.json` stores absolute paths (a Japanese OneDrive path), so a run cannot be moved; store
  them relative to the run directory. `manager.py:163,169` open `history.jsonl` without `encoding="utf-8"`.
  The metric named `silhouette_loss` is silhouette plus 50 x track loss (`service.py:138`,
  `optimize.TRACK_WEIGHT`); rename to `fit_loss` and report the parts. `align_timing` and `hold_still` are
  defined after `__all__` and are not exported. Snapshot, project file and history are three separate writes;
  a crash between the second and third leaves the attempt count one short.
- `np.load(..., allow_pickle=True)` in `fit_props.py:44` and `props_recut.py:140` is not needed for these files;
  remove it.

### B. Study design (question 2)

**F-10. B and C differ in six things at once.** Weight: rebuild (of the comparison, not of the code).
- Evidence:

  | | B | C |
  |---|---|---|
  | start scene | built by the LLM from nothing | 4RC + SAM masks |
  | what it perceives | the video, any analysis it writes (vanishing points from the checkerboard) | loss numbers; later a 4-frame sheet |
  | what it can change | anything, through Python in Blender and a shell | six parameter kinds, floor and wall fixed |
  | budget | 60 min, 200 turns (used 14-24 min, about $3) | 30 attempts, 30 min (about $0.45) |
  | Blender MCP | the user's blend-ai, not the pinned mcp-for-blender of PLAN 12.1 | none |
  | isolation from ground truth | shell and file tools, workspace inside the repo; checked by grepping logs | tools only |

  PLAN 11.1 asks for the same LLM, generator and budget. The measured difference cannot be assigned to
  "diagnostics and tools".
- Fix: section 9. Vary one factor at a time: start scene (4RC / LLM-built) x finishing stage (none / fixed-order
  tools / LLM-directed tools). Four of the six cells already have development numbers.

**F-11. The primary metric rewards the wrong thing for this project's purpose.** Weight: improve.
- Evidence: source-view mover IoU: B .862, C .830. Thirty-degree novel view (HANDOFF "補完の評価"): C is
  higher on syn_01 (.678 vs .407), syn_03 (.194 vs .061), syn_05 (.541 vs .254), B on syn_02 and syn_06.
  B's focal error is 2-37% and varies between repeats of one clip (37 / 16 / 37% on syn_01). The scene's stated
  use is camera edits (orbit90, orbit180 are exported for tennis). A scene that matches the source view by
  trading focal length against depth breaks exactly there.
- Fix: report a fixed set, none of them alone: source IoU, novel-view IoU (10 and 30 degrees), focal error,
  timing error (after F-15), spread over runs, failures, cost. Pre-register which one decides each claim
  (section 9).

**F-12. The LLM's measured contribution, by role.** Weight: improve (direction).
- Evidence for "no gain as hypothesis chooser": fixed order .837; Claude .827; GPT .830; looker .836; viewer
  .829 (DIRECTION_OPTIONS, VENUE_PLAN view-tools table). Evidence for "gain as source of priors": completion
  cmp_01 .601 -> .850 in 3 of 3 runs with nothing invented on cmp_03; net length from world knowledge; venue
  stands IoU .08 (seen only) -> .62 / .65 (LLM layout) against .88 hand-made. Evidence for "harm without
  guards": syn_04 collapse to .100 through two loopholes, closed in `3201ab7`.
- Fix: stop investing in look / view tools for repair ordering (section 8). Give the LLM one typed job:
  propose priors (known sizes, standard layouts, symmetry, contact, "this is a tennis court") as records with a
  basis, which the solver uses and the accept rule tests against observations. Court calibration, the prop
  table, venue layout and mirror / repeat completion are four instances of this already; they do not share a
  type. This is a direction for after M1, not a work package now.

**F-13. Editability has never been measured.** Weight: improve.
- Evidence: PLAN 11.3 lists it as the fourth measure; PLAN 15 still has "define the editability tasks"
  unchecked. It is the property that separates a schema-backed scene with ids, keys and read-back from an
  arbitrary `.blend`.
- Fix: WP-7 step 4. Three scripted edits, the same three as phase 1a (`evaluation/phase1a/edits.json`): replace
  the camera path, replace one object's size, delay one object's start by one second. Apply by script to every
  condition's output; score "applied without manual repair" and the scores after the edit against a ground
  truth edited the same way.

### C. Evaluation (question 6)

**F-14. Unmatched objects are dropped from the headline mean.** Weight: improve (changes published numbers).
- Evidence: `evaluate.py:94`: `mask_iou_mean` averages `iou[a]["mean"] for a in movers if a in iou`, and `iou`
  only holds pairs that `match_objects` kept (clip IoU >= 0.05, `metrics/core.py:70`). A reconstruction that
  loses an object scores the mean of the rest. The same holds for `centroid_error_mean`, and for timing, where
  missing onsets are counted separately in `timing_missing` (syn_05 under B: three of three missing).
- Fix: WP-1 step 1. Unmatched reference movers count as IoU 0; report `mask_iou_mean` (all movers) and
  `mask_iou_matched_mean` (old definition) side by side; timing reports the share of events found and the
  error over found events. Re-score every stored run and publish the table of old against new.

**F-15. The timing metric measures wobble and depends on scene units.** Weight: improve.
- Evidence: `metrics.core.onset_frame`: first frame that leaves frame 0 by more than 0.01 scene units or
  0.6 degrees. DIRECTION_OPTIONS "タイミングが直らない理由": most of B's 12-24 frame error is this; with a
  relative rule (5% of the travelled distance) B is within 0-3 frames on four clips. `run/view.py` already uses
  the relative rule for the same purpose. The decision has been waiting on the user since 11-12.
- Fix: WP-1 step 2 (D2). Adopt the relative rule in the scorer, keep the old value as a secondary column,
  re-score.

**F-16. The improvement band is far inside the noise.** Weight: improve.
- Evidence: THRESHOLDS_PROPOSAL sets +-.02 from C's spread (SD <= .009). B's spread, measured later, is
  .069 (syn_01) and .132 (syn_04) over three runs. With three runs per clip the median of B moves by several
  hundredths between repetitions of the whole experiment.
- Fix: WP-1 step 4. Report per-clip paired differences with a bootstrap interval over runs; call a clip
  "better" only when the interval excludes zero; state the count of clips and the median difference as PLAN
  11.3 says. Five runs per clip for any condition whose SD on development clips exceeds .03. Keep PLAN 11.5's
  wording: exploratory, no general claim.

**F-17. All tuning and all headline numbers use the same six clips.** Weight: improve.
- Evidence: comments in the code: `CAMERA_GUARD = 0.03` "chosen on those six synthetic clips", `FOCAL_MIN_RISE`
  "draft", `MOVE_STEP` and `SIZE_STEP` set to reproduce earlier values on the six, `HOLD_TOLERANCE` "draft",
  `_deepen` 0.5 and `ROUND_RATIO` 0.8 "chosen on the synthetic clips", SAM prompt words chosen against ground
  truth and then replaced by an automatic rule tuned on the same clips. `thresholds.yaml` has
  `"clips": {"dev": [], "held_out": []}` and `status: draft`.
- Fix: WP-1 step 3. Generate held-out set v2 from a new generator configuration and do not look at its scores
  until the freeze commit.

**F-18. The synthetic clips are narrow and favour both methods in different ways.** Weight: improve.
- Evidence: `make_synthetic_clips.py`: six seeds; every mover is an upright box; one straight move per object
  with a linear turn; floor plus one wall; 640x352, 73 frames; checkerboard and stripes (`scene/look.py`).
  Boxes are exactly C's proxy shape. The patterns make B's vanishing-point calibration and C's LK tracks easy.
- Fix: v2 varies, by seed: surface look (plain, wood-like noise, tiles, checker), mover shape (box, cylinder,
  sphere, two-box compound), one tilted object, a mover that stops and starts again, a mover present from
  frame 0, zoom, a truck with parallax (the syn_04 case, twice), light direction. Twelve clips: six dev-v2 for
  tuning from now on, six held-out. Old syn_01-06 stay as the regression set and are reported as such.

**F-19. Real footage has no ground truth and costs hours per clip.** Weight: improve.
- Evidence: SAM 3.1 on 8 GB: 76 min for one object on the robot-vacuum clip, 2.6 h for tennis; PLAN 11.7 wants
  12 real clips; none is selected.
- Fix: replace "12 stock clips" with four clips the user shoots, with cheap ground truth: phone intrinsics from
  one checkerboard calibration, tape-measured object sizes and camera height, a visible clap or tap at each
  motion start. That gives focal, scale and onset truth on real pixels. Segmentation for these runs on the
  rented GPU in WP-6 (D6).

**F-20. Venue and props are graded by their author on one scene each.** Weight: improve (label it), not a
blocker for M1.
- Evidence: `ven_01` is one synthetic venue written with the same generator vocabulary as the method; the
  hand-made reference "was written by the same person as the generator" (VENUE_PLAN); the plausibility judge
  has no ground truth and runs were judged at different times; props have four clips with hand positions taken
  from ground truth plus 2 cm.
- Fix: report these as case-study checks, not results. If they are ever to carry a claim: a second venue built
  from published dimensions of a different kind of place, generator parameters drawn by seed, the judge run
  blind in one session with shuffled order and three repeats, and calibrated first on a pair it must separate
  (the ground-truth scene against a deliberately broken one); if it cannot, drop the judge.

**F-21. Format differences leak into scores.** Weight: preference.
- Evidence: `observed_kept` fell .98 -> .84 when the net became a mesh with holes while the ground truth net is
  a board (VENUE_PLAN phase C-D). `repair_benchmark.py --observed` falls back to ground-truth masks silently
  when a file is missing (HANDOFF 2.1).
- Fix: compare category silhouettes after a small dilation; make the fallback an error.

### D. Time, hypotheses, generalisation (questions 5, 7; brief 8.10)

**F-22. Repeated frames are handled in one module only.** Weight: improve (affects every control video).
- Evidence: tennis source is 812x720, 312 frames, `fps 29827/1000`, every fifth frame a copy (24 -> 30 padding).
  `analysis/props.frame_times` detects the cadence and gives real times, used by `fit_props.py` only.
  `from_gvhmr.py:62` keys the body at `i / fps`; camera keys, `align_timing`, `hold_still` and the 24 fps control
  export (`export_scene_controls.py:87-93`) use frame / fps. The body therefore holds still for one frame in
  five, and a 24 fps resample of that is uneven motion in the control video.
- Fix: WP-2. Detect the cadence at `vrecon new`; when one is found, write a de-duplicated constant-rate clip
  (here: about 250 frames at 24 fps) and make it the project's source, with the mapping to original frames kept
  in the project. Observers run on that clip. One function `frame_time(project, shot_frame)` in `schema` or
  `video` is the only place frames turn into seconds. For tennis the canonical clip is then frame-for-frame the
  H3 clip, with no resampling at export.

**F-23. Venue and props are tennis-shaped under generic names.** Weight: improve, after M1.
- Evidence: `venue/layout.py` parts are court / net / ground / stand / wall / box / canopy; `analysis/court.py`
  is an ITF line matcher; `prop_fit` has sphere and stick; the after-strike target is a hand-written JSON.
  What is generic: front-edge line plus stepped profile plus repetition; observation summaries; the
  held / free / unseen state machine; the size table.
- Fix: do not generalise now. In WP-5 move every tennis number into the recipe. The test for "generic" later is
  one second domain run with no code change (a basketball hall or a small theatre as a synthetic scene, and a
  thrown-and-caught object by a different body part). The calibrator generalises by taking a template of line
  segments in metres as data.

**F-24. Hypotheses reach V2V undistinguished, and the right treatment differs by control kind.** Weight: improve.
- Evidence: `scene/controls.export_controls` renders every object alike; `export_scene_controls.py` drops only
  `obj_background_rest`. In orbit180 nearly the whole picture is `assumed`.
- Reasoning: a control video is a hard constraint. For a hypothesised region the intent is "something like this
  is there", not "these exact step edges". Depth states that softly; canny states it as exact lines.
- Fix: WP-3 step 4 and WP-6. Export a status pass per frame (observed / standard / hypothesis / empty). Produce
  three variants of each control: `full`, `soft` (hypothesis regions: depth kept and blurred, edges removed,
  id and look flattened), `drop` (hypothesis regions empty, far plane in depth). Measure, do not decide by
  taste: same move, same seed, three variants. Use the status pass in the RQ2 scorer to report fidelity on
  observed and on hypothesised pixels separately. Whether the pipeline accepts a per-pixel control weight is
  not known; look it up in the pinned VideoX-Fun commit before designing around it.

### E. Tests and code (question 8)

**F-26. Where the tests are thin.** Weight: improve.
- Not imported by any test (by a name search; confirm before acting): `scene/character.py`, `venue/build.py`,
  `analysis/to_court.py`, `analysis/rig_fk.py`, `scene/blend_import.py`, `scene/blend_export.py`,
  `analysis/units.py`. `blend_import` is the path every condition-B score goes through.
- No test runs the tennis chain end to end. No test covers scoring with a missing object (F-14) or a wobbling
  still object (F-15).
- Fix: WP-1 adds the two scorer tests. WP-5 adds one small end-to-end test: `ven_01` observations plus one prop
  clip through calibrate-free stages (build venue, complete, fit props, export two control kinds at 16 frames,
  low resolution), asserting object counts, statuses, H3 conformance and a stable hash of the id pass.
  WP-7 adds a round-trip test for `blend_import` on a scene with a known score.

**F-27. Duplicated loaders and numbers.** Weight: preference.
- Mask npz loading exists in `optimize.load_masks`, `service._start_from_4d`, `fit_props.load_mask`,
  `proxies_scene.py` (inline). `fps.num / fps.den` is spelled out in most modules. Intrinsics rescaling is
  inline in `export_scene_controls.py:78-81`. `snap_h3_frames` exists in `control/core.py` and
  `gpu/h3_control/run_jobs.py` (the second cannot import vrecon; keep both and add a test that feeds both the
  same inputs).
- Fix: one `vrecon.observe.io` module for observer files; helper methods on `Rational` and `Intrinsics`.

**F-28. No dependency lock, tool versions in prose.** Weight: improve.
- Evidence: no `uv.lock`; `pyproject.toml` pins only `bpy`; observer commits (sam3 `f1647e8`, 4RC `a08ca0a`,
  GVHMR `ee960bb`, tapnet `ceedd60`), weights and environment recipes are in HANDOFF paragraphs.
- Fix: commit `uv.lock`; add `gpu/observe/tools.lock.json` (tool, repo, commit, weights file and sha256,
  Python, torch, licence); observers write the matching entry into each output's metadata.

**F-29. Condition B can reach the ground truth.** Weight: improve.
- Evidence: `run_baseline.sh` puts the workspace under `RUN` (in practice `runs/B/...` inside the repository)
  and allows `Bash`, `Read`, `Glob`, `Grep`. `evaluation/synthetic/*/gt.json` is three folders away. The check
  is a search of the log for ground-truth file names.
- Fix: WP-7 step 2. Workspace outside the repository tree, clip copied in under a neutral name, deny rules for
  the repository path in the agent's settings, and the log check kept as a second line.

### F. V2V (question 9)

(F-25 is numbered out of order: it was the first V2V finding and is referred to by that number throughout.)

**F-25. RQ2 has no scorer.** Weight: rebuild (it does not exist).
- Evidence: `gpu/h3_control/README.md`: "クラウド：結果を読んで採点（次の作業）". No module compares a generated
  video with the scene; `src/vrecon/generation` and `compare` are empty. PLAN 11.4 needs, on the generated
  video: masks (IoU against the index pass), camera estimate, onsets, and the unchanged-region check.
- Fix: WP-4, before WP-6, because it decides what the GPU session must also run (segmentation on generated
  video takes hours locally) and what to save.

**F-30. Assumptions about the generator that nobody has checked.** Weight: improve.
- Memory, time and disk figures in the README are estimates from model cards.
- Five control kinds are exported; depth and canny (and gray, pose) are named in PLAN 9.1 as accepted. Whether
  `clay`, `id` and `look` are valid control inputs for Fun ControlNet Union 2.0, or only useful as reference
  images, is not established. Check the model card and the pinned example script before paying for runs.
- The main subject is a person with a 22-joint skeleton in the scene, and pose is a listed control type. No
  pose control is exported.
- Depth is normalised over the clip at the 2nd-85th percentile. With stands at 30-60 m and the player at 12 m,
  check on `controls_sheet.png` how many grey levels the player's body spans; the annotator's per-frame mode is
  the alternative already implemented (`depth_control(mode="frame")`).
- Tennis controls are 800x704 x 243 frames, the maximum length; phase 1a is 1280x704 x 124. Cost scales with
  frames. The second serve (toss to the hit at source frame 236) fits in a 124-frame window.
- Prompts for the tennis jobs, and whether a reference image (first source frame) can be given together with a
  control video, are not written down.

**F-31. No plain-video baseline for the unedited case.** Weight: improve.
- For the source camera, the cheap alternative to the whole 3D pipeline is depth estimated from the source
  video. If scene-derived controls do not at least match that on the unedited view, the gain has to come
  entirely from edits. Add this job to WP-6; it costs one annotator pass.

## 5. Answers to the nine review questions

1. **Overall design.** Keep the schema, the run manager, the bpy isolation, the shared CLI / MCP service and the
   separate observer environments. Rebuild how the real pipeline uses them (F-01, F-02), add a real status
   vocabulary (F-05), numeric scale (F-07) and a single time base (F-22). Widen the optimiser only after the
   syn_04 diagnosis (F-04).
2. **RQ1.** Do not defend "C beats B". Adopt the factorial in section 9. Of the options in DIRECTION_OPTIONS:
   option 4 (B's scene finished by the tools) as the product path, reported inside option 3's honesty (the LLM's
   choice of hypotheses did not help). Option 2 has been tried twice (looker, viewer) with no gain: stop.
   Option 1 only for the camera-path parameter, and only after the diagnosis.
3. **Priorities.** Section 7. Stop list in section 8. The twelve real clips become four self-shot clips with
   measured truth (F-19).
4. **Reproducibility.** Not reproducible today. WP-5.
5. **Observed against hypothesised.** Not sufficient: object-level, lossy mapping, free text (F-05, F-06).
   For V2V: measure three treatments, with different handling for depth and for edges (F-24).
6. **Evaluation.** F-14 to F-21. The four that change numbers: unmatched objects, onset rule, noise band,
   held-out set.
7. **Generalisation.** The mechanisms generalise, the vocabulary and constants do not (F-23). Freeze as a tennis
   case study until M1 is reported.
8. **Code quality.** F-09, F-26 to F-29. The risk is concentrated in `scripts/` and in `blend_import`.
9. **Before V2V.** F-25, F-30, F-31 and the runbook in section 10.

## 6. Decisions the user has to make

Ask these in Japanese, one at a time, with the recommendation first.

| | Decision | Recommendation | Why |
|---|---|---|---|
| D1 | What RQ1 claims | The factorial of section 9; H1a in its old form is reported as not supported on development clips | It is what the data can carry; four of six cells exist |
| D2 | Timing metric | Relative onset (5% of travel, as `run/view.py`), old value kept as a second column | The old rule measures wobble and scene units (F-15) |
| D3 | Headline accuracy | A fixed set (source IoU counting unmatched as 0, novel-view IoU at 10 and 30 degrees, focal error, timing), each claim tied to one of them in advance | Source IoU alone rewards focal-depth trades (F-11) |
| D4 | Order of the GPU session | Phase 1a first (one job, then twelve), tennis after; all in one rental | Phase 1a is the generator's calibration, not the product's V2V; it does not conflict with "finish the Blender scene first", and the scene is declared finished in HANDOFF |
| D5 | Hypotheses in controls | Export `full`, `soft`, `drop`; choose from the measurements of WP-6 | No evidence yet for any choice (F-24) |
| D6 | Real clips | Four self-shot clips with measured truth, segmented on the rented GPU | Truth on real pixels at low cost (F-19) |
| D7 | Where `runs/` and `.venv` live | Outside OneDrive, through `VRECON_RUNS` (default stays `runs/`) | Sync of large files and of files being replaced atomically; open since PLAN 15 |
| D8 | Rights and sending of the tennis clip | Write down what was decided for Claude, fal and the rented GPU, and whether results from this clip may be shown | VENUE_PLAN step 6 asked; the answer is not recorded; the clip is the one shipped with GVHMR |

## 7. Work packages

Each package: goal, steps, files, acceptance. "Re-score" always means: from stored outputs, no LLM, no GPU.

### WP-0. Baseline snapshot (half a day)

Goal: a fixed point to compare against.
1. `evaluation/RESULTS.md`: one table of every number that the documents quote for conditions B, C, fixed
   order, looker, viewer, B + tools, novel views: value, clip, run directory, commit, mask source (SAM or
   ground-truth stand-in), number of runs. Mark every number whose run directory no longer exists.
2. List the disagreements between documents (start with REVIEW_BRIEF section 3 and 6 against
   DIRECTION_OPTIONS on GPT's condition B).
3. Record `git rev-parse HEAD` and the test count.
Acceptance: every number in REVIEW_BRIEF section 6 is either traced to a run directory or marked untraceable.

### WP-1. Measurement hygiene (two to three days, CPU only)

Goal: numbers that can be called results.
1. Scorer, unmatched objects (F-14). `evaluate.score`: add `mask_iou_mean` over all reference movers with
   unmatched = 0, keep the old value as `mask_iou_matched_mean`; same pattern for centroid error (report the
   matched share). Timing: `timing_found_share` and error over found. Tests: a reconstruction missing one of
   two objects scores half of the remaining object's IoU; a perfect copy scores 1.0 on both.
2. Scorer, onset (F-15, D2). `metrics.core.onset_frame(rule="relative", share=0.05, min_units=...)`; rotation
   onset relative to total turn in the same way; old rule available as `rule="absolute"`. Test: a still object
   with 2 degrees of wobble that then moves has its onset at the move.
3. Add novel-view scores to the standard report: `vrecon score --novel` calling `evaluate.score_novel`; render
   the 10-degree views that were added but never scored.
4. Re-score every stored run of WP-0 with steps 1-3. Publish old against new in `RESULTS.md`. State plainly
   which conclusions change.
5. Held-out set v2 (F-17, F-18). Extend `make_synthetic_clips.py` with a `--set v2` configuration driven by
   seed (list in F-18). Twelve clips into `evaluation/synthetic_v2/`. Assign six to dev and six to held-out by a
   seeded shuffle written into the script. Render ground-truth passes. Do not run any method on the held-out
   six.
6. Statistics (F-16). `scripts/compare_conditions.py`: paired per-clip differences, bootstrap interval over
   runs (2,000 resamples, seed fixed), "better / worse / not separated" from the interval, plus the counts and
   median difference PLAN 11.3 asks for. Keep the unfrozen-thresholds warning.
7. Constants (F-17). Move every tuned constant to `src/vrecon/tuning.py` with a comment saying on which clips
   it was chosen. No value changes in this step.
8. Freeze. Fill `evaluation/thresholds.yaml`: metrics of D3, bands from step 6, budgets per condition
   (section 9), runs per condition, dev and held-out clip ids, sha256 of `tuning.py`. Set `status: frozen` and
   `frozen_commit` in the commit after the one that adds the file content (the user commits; propose the two
   commits).
Acceptance: tests added in steps 1-2 pass; `RESULTS.md` has the re-scored table; thresholds frozen; no method
has touched the held-out six.
Do not: tune anything on v2 dev clips before the freeze commit exists.

### WP-2. One time base (one to two days, CPU only)

Goal: frames turn into seconds in one place; padded frames are removed at the door (F-22).
1. Move cadence detection from `analysis/props.frame_times` to `vrecon/video.py`; `props` imports it.
2. `vrecon new`: when a cadence is found, write the de-duplicated clip at the true rate and record in
   `SourceVideo` the original file, the original frame for each kept frame, and the detected period. When none
   is found, behaviour is unchanged.
3. `frame_time(project, shot_frame)`; replace the spelled-out conversions in `from_gvhmr.py`, `service.py`,
   `scene/builder.py`, `scene/controls.py`, `export_scene_controls.py`, `optimize`.
4. Check on tennis without re-running observers: remap existing per-frame outputs (masks, GVHMR, calibration)
   through the frame map and compare the body's root speed in the exported 24 fps keys before and after;
   report the share of zero-motion frames (expected about 20% before, about 0% after).
Acceptance: test with a synthetic 24 -> 30 padded clip: detected period 5, kept frames = original count, key
times strictly increasing; H3 export of that clip needs no resampling.
Do not: re-run SAM, 4RC or GVHMR for this. If remapping proves insufficient, stop and report.

### WP-3. Status, spans and hypothesis-aware export (two to three days, CPU only)

Goal: the scene knows what is observed and the export can act on it (F-05, F-06, F-07, F-24).
1. Schema 0.4.0 with migration and tests: `Status` vocabulary of F-05; `SceneObject.spans`; `Evidence.kind`;
   `Scale.units_per_metre` and `basis`. Old files load; `assumed` becomes `hypothesis`.
2. Writers: `venue/build.py` keeps standard / observed / hypothesis as they are; `run/complete.py` writes
   `hypothesis` with `kind="rule"`; `fit_props.py` writes spans from the notes it already computes (held,
   free, carried on by physics, after the strike, filled) and one evidence entry per span; per-frame notes go
   to `<object>_frames.json` next to the project.
3. `scene/passes.py` or `color_passes.py`: a status pass (four values) rendered with the index pass, per frame,
   honouring spans.
4. `scene/controls.export_controls(..., hypothesis="full" | "soft" | "drop")` as described in F-24; the manifest
   records the mode and the share of pixels per status per frame.
5. A converter script for the existing tennis project (no observers re-run): statuses and spans rebuilt from
   `props.json`, the layout JSON and the evidence notes.
Acceptance: migration test; on the converted tennis scene the ball's span list reads, in order, held / free /
physics / hypothesis after the strike, with frame ranges equal to those in `props.json`; status pass of
orbit180 is mostly hypothesis and of source mostly observed (report the two shares).

### WP-4. RQ2 scorer and dry run (three days, CPU; one short local SAM run allowed with the user's consent)

Goal: PLAN 11.4 as code, proven on data where the answer is known (F-25).
1. `src/vrecon/compare/generated.py` and `vrecon score-generated VIDEO CONTROLS_DIR`: inputs are a video, the
   manifest, the index, depth and status passes of the controls it was generated from. Outputs, split by status
   region: per-object mask IoU and centroid track error against the index pass; onset error per animated
   object; camera agreement (reuse `camera_errors` on a camera estimated from the video); unchanged-region
   measure between a base and an edited generation (objects listed as `unchanged` in `edits.json`).
2. Observer contract for generated videos: the scorer takes masks, tracks and cameras as files in the existing
   formats, so the heavy tools run wherever there is a GPU.
3. Dry run A: score each phase-1a control video's shaded render as if it were the generation. Everything must
   be near perfect; anything else is a scorer bug.
4. Dry run B: score an edited scene's render against the base scene's controls. Each of the three edits must
   show up in its own measure and not in the others.
5. Dry run C (tennis, optional): the source video against the `controls_source` passes, using the masks that
   already exist. This is also the first number for "how well does the finished scene match the video".
6. Write `gpu/h3_control/score_on_gpu.md`: which observers to install on the rented machine, in which order,
   and what to copy back.
Acceptance: A is near perfect, B separates the three edits, the report format is fixed and committed.

### WP-5. Tennis as a recipe (four to five days, CPU; re-uses stored observer outputs)

Goal: one command rebuilds the tennis scene from stored observations, and the scene carries its provenance
(F-01, F-02, F-03).
1. `src/vrecon/pipeline/`: `stages.py` (calibrate, place_person, proxies, venue, complete, timing, props,
   finish, export), each `stage(project, inputs, params) -> (project, record)`, bodies moved from the scripts
   without behaviour changes; `runner.py` reads a recipe, runs stages in order, saves the project after each
   stage with `revision + 1`, registers Observations and Artifacts with sha256, writes `pipeline_log.jsonl`
   (stage, params, input hashes, output hash, seconds).
2. `evaluation/recipes/tennis/recipe.json`: every per-clip value of F-03 with its basis; `tennis_after.json`
   and the venue layout JSON copied in; observer outputs referenced by path and sha256 under `VRECON_RUNS`.
   LLM stages (venue layout) are replayed from the stored answer by default; `--refresh-llm` asks again.
3. `vrecon pipeline run RECIPE --out DIR [--from STAGE] [--to STAGE]`.
4. Move `scripts/camera_moves.py` into `src/vrecon/scene/` or `vrecon/camera_moves.py`; give the private helpers
   that scripts import public names.
5. Scripts become wrappers of at most a screenful; delete the ones the runner replaces.
6. End-to-end test of F-26 on synthetic data.
7. Rebuild tennis with the runner and compare with the current final scene: object ids equal, poses within
   1e-6, control videos' id pass identical for two moves at 16 sampled frames. Explain any difference.
Acceptance: step 7 passes; the rebuilt `project.json` has non-empty observations and artifacts, every object's
evidence names a kind, and `revision` equals the number of stages.
Do not: improve the scene while moving code. Behaviour changes come after the comparison passes.

### WP-6. The GPU session (one rental; cost cap agreed with the user beforehand)

Goal: first measurements for RQ2, and the facts F-30 lists. Runbook in section 10.
Depends on: WP-2 (controls on the right time base), WP-3 (three hypothesis variants), WP-4 (scorer), D4, D5.
Acceptance: phase-1a table (three edits x control kind x three seeds) scored; tennis source view scored against
the plain-video baseline; hypothesis variants scored on two moves; measured time, memory and cost per job
written into `gpu/h3_control/README.md` in place of the estimates.

### WP-7. RQ1 on held-out clips (after the freeze; LLM cost agreed beforehand)

1. syn_04 diagnosis (F-04), on old clips and the two v2 truck clips: camera centre error of the 4RC start scene
   and after conversion. Decide on the camera-path parameter from that; if added, it is added before the
   freeze of WP-1 or the freeze is redone and recorded in `changes`.
2. Isolation for B (F-29).
3. Run the cells of section 9 on dev-v2 (tuning allowed, budgets fixed), then once on held-out.
4. Editability tasks (F-13).
5. Report with failures included, per PLAN 11.5.
Acceptance: the table of section 9 filled for held-out clips, with intervals; a short report stating each
pre-registered claim as supported / not supported / not separated.

### WP-8. Hardening (one to two days, any time after WP-1)

F-09 items, F-27 loaders, F-28 lock files, F-21 fallback error, the `blend_import` round-trip test.
Acceptance: tests pass; a run directory copied to another folder still opens with `vrecon run state`.

### WP-9. Documents (one day, last)

1. `docs/STATUS.md` (current state, at most 80 lines) replaces HANDOFF sections 1-5 as the first thing to read;
   HANDOFF.md becomes `docs/log/` by date, unedited.
2. ARCHITECTURE.md brought in line with the code; empty packages removed or filled (WP-4 fills `compare`).
3. PLAN.md section 16.6: the decisions of section 6, the change of condition B's MCP, the RQ1 protocol v2.
4. `docs/LICENSES.md`: one row per tool and weight file (PLAN 8 requires it).
5. CLAUDE.md "Read first" points to STATUS.md.

## 8. Stop and defer list

Stop (evidence says no gain):
- fal outside pictures for layout: stands IoU .70 without, .68 with; judge-only gain with a judge that has no
  ground truth. Keep `fal_restyle.py` as a look check only.
- Look and view tools for repair ordering: .836 and .829 against .837. Keep `view.events` / `view.track`
  because `align_timing` and `hold_still` use them without an LLM.
- Ray placement (`optimize/rays.py`): default off, worse on two clips. Remove after WP-1 unless v2 says
  otherwise.
- MegaSaM and MoRe wrappers: both lost to 4RC on five or six of six; MoRe has no licence. Archive.
- TAPNext++ as default tracker: worse than LK on all six in the latest benchmark. Keep the loader, drop from
  plans.

Defer until M1 is reported:
- Venue polish (crowd, corners, aisles, canopy), second sports in the prop table, LLM-written prop tables and
  after-strike targets.
- The 2D-layer path and `triage`. They work on synthetic data and are on no path to M1.
- PLAN stages 3-4 (two cuts, multi-cut, events) and OpenTimelineIO.
- A typed priors layer (F-12). Write half a page on it in WP-9; build nothing.
- Differentiable rendering (PLAN 16.5).

Keep going:
- 4RC as the start scene, focal fixed for 4RC starts, LK tracks, the camera accept rule with its guard, fixed
  pairing, floor and wall fixed by default.

## 9. RQ1 protocol v2

Question: what does each part contribute to a single-cut reconstruction that is meant to be edited in 3D?

Factors and cells (each cell: same clips, same masks, same scorer):

| | finish: none | finish: fixed-order tools | finish: LLM-directed tools |
|---|---|---|---|
| start: 4RC + SAM | S0 | S1 (today's fixed order) | S2 (today's condition C) |
| start: LLM builds in Blender (today's B) | L0 | L1 (today's option 4) | L2 (`run_baseline.sh --refine`) |

Budgets, fixed in `thresholds.yaml`: L-start 60 min and 200 turns; tools 30 attempts; LLM-directed finish adds
30 min. Cost in dollars and minutes is an outcome, not something to equalise: the cells differ in cost by design
and the report shows accuracy against cost.

Development numbers already in hand (old scorer, to be re-scored in WP-1): S0 .753, S1 .837, S2 .827 / .830,
L0 .862, L1 .881, L2 not run.

Claims, each tied to one measure before the held-out run:

| | Claim | Measure | Comparison |
|---|---|---|---|
| H1 | The tools improve a scene whatever its origin | source IoU (unmatched = 0) | S1 against S0, L1 against L0, paired per clip |
| H2 | The tools reduce run-to-run spread and failures of an LLM-built scene | SD over runs; share of runs below S0 | L1 against L0 |
| H3 | A 4RC start is more consistent in 3D than an LLM-built start | novel-view IoU at 10 and 30 degrees; focal error | S1 against L1 |
| H4 | Letting the LLM choose repairs adds nothing over the fixed order in this action space | source IoU | S2 against S1, L2 against L1 (reported as "not separated" if the interval contains zero) |
| H5 | A schema-backed scene accepts the three scripted edits without repair more often | edit success share | S1 against L0 and L1 |

Runs: three per cell and clip; five where development SD exceeds .03 (expected: L0, L1, L2). Held-out: the six
v2 clips, plus the four self-shot clips for the measures that have truth there (focal, scale, onset).
Observation masks: SAM for every cell; ground-truth stand-in masks are for machinery tests only and never
appear in the results table.
Reporting: per clip and per cell, all runs, failures counted; intervals as in WP-1 step 6; the old H1a
("C more accurate than B at equal budget") stated as not supported on development clips, with the confounds of
F-10 listed.

## 10. V2V session runbook

Before renting (all local):
1. WP-2, WP-3, WP-4 accepted. `run_jobs.py --dry-run` clean for every job file.
2. From the pinned VideoX-Fun commit and the two model cards, write down, with the line or URL: accepted control
   types; whether a reference image can accompany a control video; whether a per-pixel control weight or mask
   exists; the memory modes. Remove from the job list any control kind that is not an accepted type (F-30).
3. If pose is accepted: export an OpenPose-style pose control from the scene's rig joints for the tennis player
   (projection of the 22 joints; mapping table in the commit).
4. Job files: `evaluation/phase1a/jobs.json` (exists); `evaluation/tennis/jobs.json` with prompts written once
   and reviewed by the user; a 124-frame window around the serve as the default tennis length, 243 only for one
   confirmation job.
5. Depth check on `controls_sheet.png` (F-30); if the player spans fewer than about 20 grey levels, add the
   per-frame mode as a second depth job.
6. A cost cap and a stop rule agreed with the user: stop after stage B below if one job takes more than the
   agreed minutes.
7. D8 recorded.

On the machine, in this order; each stage ends with numbers written to disk before the next starts:

| Stage | Jobs | Purpose |
|---|---|---|
| A | `base__depth`, one seed | does it run; time, memory, disk; decide memory mode |
| B | phase 1a main: 4 scenes x depth x 3 seeds | RQ2 on the three edits |
| C | phase 1a extra: canny, gray (and pose if applicable), 1 seed, then 3 for the best | control kind |
| D | install observers; score A-C with `score-generated` | results exist before tennis starts |
| E | tennis source camera: scene-derived depth; plain-video depth (F-31); 124 frames; 3 seeds | is the scene as good as the video for the unedited view |
| F | tennis, two moves (one small: orbit30 or dolly; one large: orbit90): `full`, `soft`, `drop`; 1 seed, then 3 for the two closest | D5 |
| G | segmentation for the four self-shot clips (D6), if time remains | WP-7 input |
| H | pack: metadata, sheets, scores to git; videos to the user's disk | |

Rules: comparison point 2 never changes the 3D scene (PLAN 6.3). Seeds, prompts, model and commit ids are in
every `meta.json` already; add the hypothesis mode and the control variant. Failed generations are kept and
counted.

## 11. Rules for the executing agent

- Follow CLAUDE.md. Only `src/vrecon/scene` calls `bpy`; while moving code in WP-5, the clip generators' direct
  `bpy` use moves behind `scene/` functions too.
- Ground truth (`gt.json`, `gt_passes/`, `meta.json` of evaluation clips) is read by scorers only. Held-out
  clips are not run through any method before the freeze commit. If that rule is broken, say so in
  `thresholds.yaml` `changes`.
- One GPU job at a time on the 8 GB machine. Ask before any observer run longer than ten minutes.
- Nothing is sent to fal, a rented GPU or an LLM without the user's go-ahead for that step; never read key or
  token files.
- Do not write under `runs/` except into new folders the work package names. Never overwrite a stored run.
- Commit only when the user asks. Propose commits per work package, with the acceptance output in the message.
- A work package that fails its acceptance check is reported as failed with the output. Do not adjust the
  check.
- Report numbers with their run directory and scorer version. A number without both is not written into a
  document.
- When a finding of this review turns out to be wrong on closer reading, write that into section 12 and go on.

## 12. Not verified by this review

- The list of untested modules in F-26 comes from a name search over `tests/`; some may be reached indirectly.
- The effect of repeated frames on the exported controls (F-22) is derived from the code; no control video was
  inspected frame by frame.
- Whether `clay`, `id`, `look` or pose are accepted control types, and whether a per-pixel control weight
  exists (F-24, F-30): not looked up.
- The cause of syn_04 (F-04): stated in the documents as camera translation not reproduced; the split between
  4RC's path error and the conversion was not measured.
- B's access to ground truth (F-29) is a possibility from the script's permissions; the logs were not re-read.
- Numbers quoted from the documents were not recomputed, except the test count (156 passed) and the summary of
  the final tennis `project.json`.
- `runs/` beyond the final tennis folder and the names of top-level folders was not inspected.
