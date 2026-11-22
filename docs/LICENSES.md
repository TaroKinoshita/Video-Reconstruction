# Tools, weights and data: licences (PLAN 8; WP-9)

Research, non-commercial. "Checked" means the licence was read at the source; "from notes" means it comes from this
project's earlier notes and must be checked against the source before anything is published.

| Item | Used for | Licence | Status |
|---|---|---|---|
| Blender 5.2 / bpy 5.2.3 | scene build, renders | GPL-3.0 | from notes |
| ComfyUI v0.44.0 (on the pod) | generation, SAM | GPL-3.0 | from notes |
| MiniMax-H3 (ref2va / fl2va int8 by Comfy-Org) | video generation | MiniMax H3 Community License: worldwide EXCEPT the EU, UK, Republic of Korea and USA (Japan included); written approval above US$20M yearly revenue; outputs may not train other models; generated content published must be disclosed as machine-generated (AUP 12). Pods ran in RunPod euro-3; the user (Japan, Japanese national) judged this fine (2026-11-19) | checked 2026-11-18 (huggingface.co/minimaxai/minimax-h3 LICENSE) |
| MiniMax-H3 Fun ControlNet Union 2.0 (alibaba-pai) | ControlNet runs (legacy) | MiniMax H3 Community License (derivative; same territory terms) | checked 2026-11-18 (model card) |
| VideoX-Fun `e8c314f` | job runner design (not run) | Apache-2.0 | from notes |
| SAM 3.1 (`sam3.1_multiplex_fp16`, Comfy-Org) | masks on the pod and locally (`f1647e8`) | SAM License (Meta): royalty-free use, research and commercial; acknowledge SAM in publications; pass the licence on when redistributing; no military/weapons use | checked 2026-11-18 (facebookresearch/sam3 LICENSE) |
| 4RC `a08ca0a` | 4D start scene | S-Lab licence, non-commercial | from notes |
| GVHMR `ee960bb` | human motion | custom ZJU 3DV notice: educational, research and non-profit use only | checked 2026-11-18 (zju3dv/GVHMR LICENSE) |
| SMPL-X body model | body mesh (data stays under `runs/`) | SMPL-X licence, non-commercial research | from notes |
| TAPNext++ (tapnet `ceedd60`) | tracker comparison | Apache-2.0 | from notes |
| MegaSaM `a27b4e6` | comparison only | Apache-2.0 (code), CC-BY-4.0 (other materials) | checked 2026-11-18 (repository README) |
| MoRe `f330a34` | comparison only, not used | no licence stated | from notes |
| fal (GPT Image 2.5 Sunburst) | look checks, outside pictures (stopped for layout) | service terms | user's account |
| Tennis clip (shipped with GVHMR) | case study | see GVHMR's terms | D8: may be sent to Claude, fal and the rented GPU for research; input frames, 3D renders and generated videos (labelled machine-generated) shown in the README as research results (user, 2026-11-23) |
| Pexels stock clips | real-clip runs | Pexels licence | from notes |

Before any result is shown outside: every "from notes" row checked at its source, D8 revisited for the tennis
clip.
