# P1: how bare Claude drafts break under camera moves (2026-11-20)

Drafts: `runs/p1/drafts/<clip>/r<N>/workspace/output/scene.blend` (baselines/b_llm_mcp, claude-opus-5-5, 60 min).
Sheets: `runs/p1/clay_<clip>.png` (`scripts/clay_moves_blend.py`: source video on top, then clay from the source
camera, a 45-degree orbit and a 60% pull-back, at the start, middle and end of the clip). Done: tennis r1, r2;
duck r1; mograph r1 (duck r2 and mograph r2 waited on the Claude usage limit).

| Clip | Source camera | Orbit 45 / pull-back | Kind (PLAN_v2 2.2 / 2.3) |
|---|---|---|---|
| tennis | Player as an articulated mannequin with a racket in the hand, moving; stands as box rows; people as small figures | At the end of the orbit the venue stops: dark empty space where the stands end; no court lines or net visible; the pull-back shows the stands as blocks only where the source looked | outside the frame (2.3); body is a mannequin, not a body (2.1 keeps it, a body would be an improvement) |
| duck | Legs as stacked tilted cylinders, the string as a straight line, toys approximated | Orbit: the right side is a bare wall, objects cut by the frame edge become floating blocks in the pull-back; the legs stay a stack of cylinders | outside the frame (2.3); frame-edge objects built only as seen; person-object relation (string to hand) to check over time (2.2) |
| mograph | Shapes, sizes and layout close; the ball bounces | Holds up: simple shapes on a floor; back wall too short at the orbit's edge | easy case for a bare LLM; timing/easing to compare frame by frame |

First reading, before any fix: the bare drafts look right from the source camera (as v1.0 found) and break where the
camera leaves it: empty space past what the source saw, frame-edge objects, crude people. The motion-graphics clip is
the case a bare LLM already handles. So P3/P4 start with the outside of the frame (2.3) and frame-edge objects, then
people (replace a mannequin with an observed body where GVHMR runs), then relations over time.
