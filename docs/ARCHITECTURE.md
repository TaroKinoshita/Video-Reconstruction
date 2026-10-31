# Architecture (draft)

Source: `docs/proposal_v0.3.pdf`. Scope of the initial project = Stages 1-3.

```
reference video -> analysis -> planning -> scene (Blender) -> control passes
                                                                  |
                                      [GPU machine] generation <--+
                                                  |
                                  compare/revise <-+  (loop within budget)
```

## Layout
| Path | Role |
|---|---|
| `src/vrecon/schema` | Shared scene representation (IDs, coordinate system, units, camera, confidence, provenance) |
| `src/vrecon/analysis` | Shots, objects, depth order, camera, motion, event times |
| `src/vrecon/planning` | What goes explicit in 3D vs. left to generation; budget |
| `src/vrecon/scene` | Blender construction, control-pass export, save/resume |
| `src/vrecon/generation` | Generator adapter (files in / files out; runs elsewhere) |
| `src/vrecon/compare` | Aligned-time comparison, mismatch attribution, revision |
| `configs/` | Module selection and versions |
| `runs/` | Execution history (git-ignored, not committed) |
| `evaluation/` | 12 clips (6 dev / 6 held-out), conditions A/B/C, results |

## Workflow between cloud and local GPU (pattern A)
1. Cloud: code + scene + control passes -> push.
2. Local GPU: pull, run generation, push results (small files / manifests only).
3. Cloud: compare and revise.
Large media stays out of git; exchange it via a storage location agreed later.

## Rules
- Application state is kept separate from LLM decisions.
- Pin OSS versions; record model version, seed, settings per generation.
