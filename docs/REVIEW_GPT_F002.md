# Review instructions for GPT: F002 reconstruction (v9, second round; final scene v11)

You are an outside reviewer. Give an honest second opinion: no flattery, and no fixes applied. Do not edit files in this repository. Write your answer as one report.

## What this is

A research system takes one film clip and builds an editable 3D scene in Blender. That scene later guides a video-to-video generator.

The clip is MAVEBench F002: 12.8 s, about 24 fps, several camera cuts. It shows a forest training ground where four men fight with wooden staffs.

Everything in the scene comes from the clip alone. The benchmark's ground truth is used only for scoring.

## What to look at

- `runs/mave/F002/review/compare_v9.mp4` (main input for the second review; final: `compare_v11.mp4`; not in git; `compare.mp4` is the v4 you reviewed first):
  - left half: the film;
  - right half: the scene rendered from the reconstructed camera on every frame.
- `runs/mave/F002/scene_v9.blend` (optional; Blender 4.5+). Collections:
  - `Environment`: Ground, Backdrop, Trees, SetPieces;
  - `People`: one moving body per person;
  - `Props`: held staffs;
  - `Imagined`: trees and a far forest wall where no camera looked; each object has the property `imagined = 1`.
- Code (optional, to check whether the methods are sound and general):
  - `scripts/v11_mave_prep.py`: camera from 4RC, focal fit, per-frame camera.
  - `gpu/body/run_sam3dbody.py`: local SAM 3D Body per person per frame, using our focal.
  - `scripts/v11_people_anim.py`: bodies in world space, scale fitted to 4RC depth, time smoothing, colours from the film.
  - `scripts/v11_parts.py`: level ground, trunks, posts, beam, backdrop, imagined trees.
  - `scripts/v11_staffs.py`: a stick mask counts as a staff only where it touches a hand vertex; fixed length, tilt, carried with the hand across gaps.
  - `scripts/v11_hand_windows.py`, `gpu/observe/run_sam3_points.py`: crops around the hands; SAM tracking from point seeds.
  - `scripts/v11_colour_gain.py`: people colour gain measured against the film.
  - `src/vrecon/scene/mave_blend.py`: Blender builder.

## What changed since your first review (v4, 6/10)

- Shards at 00:04-00:05.5: backdrop points are kept only if other views confirm them as static.
- Light ground patch: sky sheen on the ground; specular turned off. Unseen ground is tiled from seen ground.
- Staffs:
  - grip uses hand vertices only;
  - one length per staff, with tilt;
  - mask pieces cut by a hand are joined;
  - the holder is a majority over nearby frames;
  - point-seeded SAM tracking and carrying with the hand fill part of the middle of the clip.
- People colour: the render is matched to the film brightness.

## Known weak points (check them, but do not stop at them)

- Staffs are still missing around 00:08.5-00:09.6.
- A staff's holder can still switch between people.
- The sacks are part of the backdrop surface, not separate objects.

## Scores so far (for context only; judge by the picture)

MAVE eval (camera / depth order / people), total score:

| Version | Total |
|---|---|
| MAVE baseline | 76.5 |
| v4 (first review) | 86.5 |
| This v9 | 86.7 |

v9 detail:

| Measure | Value |
|---|---|
| RRA@15 | 1.000 |
| RTA@15 | 0.895 |
| Depth OrdAcc | 0.899 |
| HOTA | 0.778 |
| MaskIoU | 0.696 |

## Questions

0. **Compared with v4:** which of your first-review points are fixed, partly fixed, or not fixed? Give a new score 1-10.
1. **Overall:** how faithful is the right side to the left, frame by frame? Score it 1-10 with a reason. Name the 3 worst moments, with timestamps.
2. **People:**
   - Do position, pose, size and motion match the film?
   - Are there jumps, sliding feet, wrong limbs, or people missing or added?
3. **Held objects:** are the staffs in the hands and pointing the right way? Where are they missing or wrong?
4. **Background:**
   - Do the trunks, the ground shape, the posts and the beam sit where the film has them?
   - Are there floating pieces, holes, stretched surfaces or seams?
5. **Unseen areas:** in wide or turning shots, does the imagined forest look plausible and consistent with what is seen?
6. **Cuts:** does the space stay consistent across camera cuts? For example, is the same post in the same place after a cut?
7. **For video generation:** if the right side guides a video model as depth and layout, what is most likely to go wrong? List the top 3 fixes by expected impact.
8. **Method (only if you read the code):** list any step that is a patch for this one clip and would not work on other videos. List any step with an assumption that is likely wrong.

## Answer format

- Put the answer to each question under its own heading.
- Be concrete and use timestamps (mm:ss).
- Say "can't tell" where you can't.
- Separate what you saw from what you guess.
