# Tools, weights and data: licences (PLAN 8; WP-9)

Research, non-commercial. "Checked" means the licence was read at the source; "from notes" means it comes from this
project's earlier notes and must be checked against the source before anything is published.

| Item | Used for | Licence | Status |
|---|---|---|---|
| Blender 5.2 / bpy 5.2.3 | scene build, renders | GPL-3.0 | from notes |
| ComfyUI v0.44.0 (on the pod) | generation, SAM | GPL-3.0 | from notes |
| MiniMax-H3 (ref2va / fl2va int8 by Comfy-Org) | video generation | MiniMax model licence (see the Hugging Face card) | to check |
| MiniMax-H3 Fun ControlNet Union 2.0 (alibaba-pai) | ControlNet runs (legacy) | see the model card | to check |
| VideoX-Fun `e8c314f` | job runner design (not run) | Apache-2.0 | from notes |
| SAM 3.1 (`sam3.1_multiplex_fp16`, Comfy-Org) | masks on the pod and locally (`f1647e8`) | SAM licence (Meta) | to check |
| 4RC `a08ca0a` | 4D start scene | S-Lab licence, non-commercial | from notes |
| GVHMR `ee960bb` | human motion | see repository | to check |
| SMPL-X body model | body mesh (data stays under `runs/`) | SMPL-X licence, non-commercial research | from notes |
| TAPNext++ (tapnet `ceedd60`) | tracker comparison | Apache-2.0 | from notes |
| MegaSaM `a27b4e6` | comparison only | see repository | to check |
| MoRe `f330a34` | comparison only, not used | no licence stated | from notes |
| fal (GPT Image 2.5 Sunburst) | look checks, outside pictures (stopped for layout) | service terms | user's account |
| Tennis clip (shipped with GVHMR) | case study | see GVHMR's terms | D8: may be sent to Claude, fal and the rented GPU for research; results not published |
| Pexels stock clips | real-clip runs | Pexels licence | from notes |

Before any result is shown outside: every "to check" row checked at its source, and D8 revisited for the tennis
clip.
