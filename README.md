# Video-Reconstruction

Reconstruct a short reference video as an editable 3D scene (Blender), then use
structure-conditioned video generation for the final look. Pilot project to test
whether an editable 3D representation beats direct generative editing.

- Proposal: [`docs/proposal_v0.3.pdf`](docs/proposal_v0.3.pdf)
- Architecture draft: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)

## Status
Skeleton only. Next: check Blender (`bpy`) on CPU, then write the shared scene schema.
Stage 1 (manual scene -> generator control test) needs a GPU machine.
