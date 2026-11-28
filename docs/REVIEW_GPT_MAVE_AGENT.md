# Review instructions for GPT: MAVE reproduction (agent stage and checks), 2026-11-28

You are an outside reviewer. Give an honest second opinion: no flattery, and no fixes applied. Do not edit files in
this repository. Write your answer as one report.

## What this is

A research system builds an editable Blender scene from one film shot. It now reproduces MAVE (arXiv 2611.22464),
whose code is not public. MAVE has three stages:
1. Measure the camera, people and geometric evidence.
2. An LLM agent builds the set from primitives in a query / edit / verify loop.
3. Fixed checks report problems, which the agent fixes or justifies before submitting.

The test shot is MAVEBench F002 (12.8 s, a forest training ground, three men fighting with staffs). Everything
comes from the shot; the benchmark's ground truth is used only for scoring.

## What to read

- `docs/PLAN_v1_1.md` sections 12 to 12.3 (Japanese): how each MAVE part maps to this project, every measured run,
  and what was changed and why.
- Code:
  - `src/vrecon/agent/`
    - `prims.py`: primitives.
    - `evidence.py`: depth, labelled points, structures.
    - `checks.py`: the four checks and which issues may be justified.
    - `gravity.py`: MAVE eq. 3.
    - `session.py` and `tools_mcp.py`: the agent's tools.
  - `scripts/agent_run.py`: the agent loop through `claude -p`, resumed until submit or time out.
  - `scripts/mave_pipeline.py`: video to .blend in one command.
- Pictures (git-ignored, under `runs/mave/F002/review/`):
  - `agent_vs_system.jpg`: film | first agent scene | our own scene parts.
  - `agent_mave_m2.jpg`: the MAVE-faithful agent scene (78.5) from the shot camera, and an orbit view.

## Results so far (MAVEBench total; the paper's MAVE on F002: 76.5)

| Run | Total |
|---|---|
| Our scene parts, no agent (one-command pipeline) | 85.5 |
| Our parts, corrected by the agent | 86.1 |
| MAVE reproduced as a whole (empty start), two runs | 73.9 and 78.5 |
| Agent forced to build every near vertical structure (shape only) | 61.7 / 66.6 |

## Questions

1. **Submit rule.** In MAVE, every issue must be resolved or justified. Here a justification is refused for
   structure a segmenter recognised (trunk, wall, ceiling) near the camera path, the near ground layer, a person's
   height, and completeness issues. Is this a fair reading of MAVE, or a deviation that should be reported as ours?
   Would a different rule stop the two failures we saw: justifying everything away, and boxes placed over people?
2. **Video-consistency check (eq. 12).** The tolerance is 0.1 m + 5 % of depth, and the observed depth is
   min-filtered over 3 pixels, because 4RC smooths thin trunks into the background. Is that defensible, or does
   it hide real errors?
3. **Gravity (eq. 3).** Ground, wall, body and camera cues agree on a 4.6-4.9 degree tilt from the ground-plane
   level used before. The cue weights are our own (the paper gives none). Is applying it right?
4. **Variance.** Two runs with the same setup scored 73.9 and 78.5. What is the minimum evidence (number of shots,
   repeats) before claiming "reproduces MAVE" or "beats MAVE"?
5. **Visuals.** In `agent_mave_m2.jpg`, does the blockout look like a usable previs set for this shot? What is the
   most important thing missing?
6. **Anything that looks like a per-clip patch** rather than a general mechanism.
