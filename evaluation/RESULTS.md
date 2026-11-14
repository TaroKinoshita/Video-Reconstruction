# Results snapshot (WP-0, 2026-11-15)

Fixed point before WP-1 re-scores anything. Every number below is either read from a stored file under `runs/`
or marked otherwise. Nothing was re-scored, no LLM or GPU was used.

- Commit: `89f285a` (`git rev-parse HEAD` = `89f285ad5caf414bf3a41c7bf899ca77917030a1`), plus PLAN 16.6 and this file.
- Tests: 160 passed in 82 s (`.venv/Scripts/python.exe -m pytest -q`, run 2026-11-15 with this file).
- Scorer: v1 as stored at run time (`mask_iou_mean` = matched objects only; timing = absolute-onset rule).
  Scorer v2 fields (`mask_iou_all_mean`, `timing_relative`) are not in these files; WP-1 step 4 adds them.
- Clips: `evaluation/synthetic/syn_01..06` (the development set; also the only set so far).
- Masks: SAM 3.1 for all condition-C, looker, viewer and fixed-order runs (4RC start); condition B does not use
  masks; the repair benchmark uses ground-truth stand-in masks (`masks_tool: ground_truth_stand_in`) and is a
  machinery test, not a result.
- Regenerate the stored-score tables: `.venv/Scripts/python.exe scripts/collect_results.py`.

## 1. Trace of REVIEW_BRIEF section 6

| Number in REVIEW_BRIEF 6 | Stored value | Source | Status |
|---|---|---|---|
| C fixed order .837 | .837 (.840/.849/.884/.776/.783/.890) | `runs/bench_rays/results.json`, base column | traced |
| C Claude .827 | .827 | `runs/llm_c4/syn_0*/llm/result.json` | traced (1 run) |
| C Claude 3 runs, mean .820, SD <= .009 | .823 / .817 / .819 | `runs/llm_c2`, `llm_c2b`, `llm_c2c` | traced; this is the older prompt without `--use-budget` |
| C GPT .830 | .830 | `runs/gpt/c/syn_0*/llm/result.json` | traced |
| B Claude syn_01 .945 | .945, focal 3.7% | `runs/B/syn_01_blendai_r2/score.json` | traced (1 run, 1 clip) |
| B GPT syn_01 .940, syn_02 .912 | .940, .912 | `runs/gpt/b/syn_01`, `syn_02` `score.json` | traced |
| "GPT B for 4 clips stopped by a credit limit (invalid, to redo)" | syn_03..06: .954 / .712 / .710 / .942 | `runs/gpt/b/syn_03..06/score.json` | **contradicted**: all six exist (mean .861 from the stored values; DIRECTION_OPTIONS says .862 from rounded per-clip values). See section 3 |
| Repair benchmark mean .906 | .906 | `evaluation/repair_benchmark/results.json` (run dir `%LOCALAPPDATA%/vrecon/work/bench_guard03`) | traced to the committed summary; run dir outside the repo, not checked |
| 4RC start, focal fixed .840-.890 | .776-.890 per clip | same as fixed order | traced; the range in the brief (.840-.890) leaves out syn_04 .776 and syn_05 .783 |
| Timing looker -> +align -> +hold: 7.8 -> 6.75 -> 6.08 | | `docs/VENUE_PLAN.md` tables; runs `runs/looker`, `viewer2`, `viewer3` | partly traced: IoU of these runs is stored (below); timing per run is in the same `result.json` files, not re-checked here |
| Venue stands IoU .08 / .62-.65 / .88; fal .68 vs .70 | | `runs/venue/ven_01/{observed_only,llm1..5,manual,fal1..3}/scores.json` | files exist; values not re-read in WP-0 |
| Props (toss 1.6-3 cm etc.) | | `runs/props/prop_0*.json` (`prop_01`, `prop_02` also have `.err` files) | files exist; values not re-read |
| Tennis: court line fit .99, height 1.84 m, racket 228/312 frames | | `runs/human/tennis/`, `runs/venue/tennis/` | not traced (no single score file) |

## 2. Other numbers the documents quote

| Number | Stored value | Source | Status |
|---|---|---|---|
| S0 (4RC start, no finishing) .753 | per-clip .712/.813/.834/.686/.711/.761 in HANDOFF | start scenes of `runs/llm_c4/*` (`snapshots/rev_0000.json`) | **not stored as a score**; reproducible by scoring rev 0 |
| B + tools (option 4) .881 | - | `runs/opt4/syn_0*/refined.blend` (`refine_result.json` has losses only) | **not stored as a score**; reproducible with `vrecon score-blend` |
| B + tools on repeats (syn_01 .911, syn_04 .763) | - | `runs/opt4_rep/*/refined.blend` | not stored as a score; same as above |
| B GPT repeats syn_01 .933/.818, syn_04 .553/.814 | .933 / .818 / .552 / .814 | `runs/gpt/b_repeat/*/score.json` | traced (syn_04 r2 is .552 stored, .553 in the documents) |
| Looker .836, viewer .829 | .836, .829 (viewer2 .822, viewer3 .828) | `runs/looker`, `viewer`, `viewer2`, `viewer3` | traced |
| llm_c (before the guard fix, `3201ab7`) .605/.../.100 | mean .664 | `runs/llm_c/*` | traced; superseded by the fix, keep only as a failure record |
| llm_c3 .829 | .828 | `runs/llm_c3/*` | traced; mixed camera guard (0 for syn_01-04, .03 for 05-06), not comparable |

## 3. Disagreements between documents

1. **GPT condition B.** REVIEW_BRIEF 3 and 6 (and HANDOFF_EN 4, 8: "GPT B for syn_03..06 to redo") say the run
   stopped at two clips. HANDOFF.md line 132 records the stop; DIRECTION_OPTIONS ("GPT 条件B 6本そろった") and the
   stored files show all six were later completed and scored. The stored files win: **condition B GPT run 1 is
   complete (6 clips)**. The "redo syn_03..06" open item is closed.
2. **Claude condition C, ".827" against ".820".** .827 is `llm_c4` (one run, `--use-budget`); .820 is the mean of
   three runs of the older prompt (`llm_c2*`). HANDOFF_EN 5 writes ".827 (3 runs each, SD <= .009)", which mixes
   the two. The three-run spread belongs to `llm_c2*`; `llm_c4` has one run.
3. **4RC start range.** REVIEW_BRIEF 6 ".840-.890" against stored .776-.890 (syn_04, syn_05 below .84).
4. **Condition B's MCP.** PLAN 12.1 pins mcp-for-blender; every B run used the user's blend-ai (HANDOFF.md 141).
   Recorded; to be written into PLAN with WP-9.
5. **Rounding.** B GPT mean .861 (stored) against .862 (documents); syn_04 r2 .552 against .553. Harmless, but
   documents should quote this file from now on.

## 4. Stored-score tables

Generated by `scripts/collect_results.py`; per clip: IoU (matched objects), focal error, timing error (old rule),
and the source file. Re-run the script for the full tables; group means:

| Group | Mean IoU | n |
|---|---|---|
| B Claude (blend-ai), syn_01 only | .945 | 1 |
| B GPT run 1 | .861 | 6 |
| B GPT repeats (syn_01 x2, syn_04 x2) | .779 | 4 |
| C Claude `llm_c` (before guard fix) | .664 | 6 |
| C Claude `llm_c2` / `c2b` / `c2c` | .823 / .817 / .819 | 6 each |
| C Claude `llm_c3` (mixed guard) | .828 | 6 |
| C Claude `llm_c4` | .827 | 6 |
| C GPT | .830 | 6 |
| Looker / viewer / viewer2 / viewer3 | .836 / .829 / .822 / .828 | 6 each |
| Fixed order (4RC start, focal fixed) | .837 | 6 |

## 5. What WP-1 has to fill in

- Score `runs/opt4*/refined.blend` and the rev-0 start scenes so S0, L1 have stored scores.
- Re-score every row above with scorer v2 (`mask_iou_all_mean`, `timing_relative`) and add the novel-view scores.
- Re-read the venue, props and timing values from their files and add them here with their source.
