# Architecture (draft v0.2)

This file follows [`PLAN.md`](PLAN.md) (Japanese, authoritative). If they disagree, PLAN.md wins.

## Responsibilities

| Component | Responsibility | Swappable |
|---|---|---|
| **Run manager** (custom, core) | Pipeline steps, budget, stop criteria, pause/resume, run history, trying candidates in a working copy and committing them atomically | No (core) |
| **Constraint manager** (custom, core) | Human locks, which fields may change, `edited_by_human` vs `locked`, rejecting candidates that touch locked or undeclared fields | No (core) |
| **Scene state** (custom, core) | Four layers: observations, adopted reconstruction, human edits/constraints, artifacts. Versioned JSON (`schema_version`); large data in external files referenced by ID + path + hash | No (core) |
| **Optimizers** (custom) | Numeric search for camera, placement and timing within a range chosen by the LLM | Yes |
| **Diagnostics** (custom) | Reprojection residuals, mask IoU / centroid tracks, event timing, contact/penetration, with uncertainty. Three comparison points (see below) | Yes |
| **Analysis plugins** (OSS) | Shot detection, camera/intrinsics, depth, object reconstruction, human motion. Run in isolated environments; exchange files | Yes |
| **Blender builder** (custom) | Scene state → `.blend` and control passes (depth, edges, object masks, camera data). The only place that calls `bpy` | Yes (per Blender version) |
| **Blender read-back** (custom) | `handoff` snapshot → human edits in Blender → `ingest` diff against that snapshot. Initial scope: ID'd object transforms, camera settings, matching keyframes | No (core) |
| **Generator** (V2V) | Control passes + references → final video. Records model, version, seed, settings | Yes |
| **Connectors** | `vrecon` CLI (primary), `vrecon-mcp` (same functions over MCP for Claude Code / Codex / others) | Yes |
| **LLM** | Reads diagnostics, proposes hypotheses and changes. Holds no state; cannot write scene state directly | Yes |

## Flow

```
reference video
  -> analysis plugins -> observations
  -> adopted reconstruction  (+ human edits/constraints, highest priority)
  -> Blender builder -> .blend, 3D render, control passes -> generator -> final video

comparison points:
  (1) reference <-> 3D render      fixes: camera, placement, animation
  (2) 3D        <-> generated      fixes: control inputs, generation settings, model
  (3) reference <-> generated      fixes: materials, lighting, references, settings
  Rule: errors found at (2) never cause edits to the 3D scene.
```

## Repair loop

```
LLM proposes a hypothesis
  -> run manager checks locks and budget
  -> working copy
  -> optimizer search -> render -> diagnostics            (all on the working copy)
  -> constraints pass? then compare metrics per axis
       pass: commit to adopted state atomically + snapshot
       fail / error / interrupted / out of budget: discard copy, adopted state unchanged
```

Arbitrary Python (escape hatch) runs only on the working copy. After it runs, the full diff against the pre-run state is computed; any change to a locked field or outside the declared scope makes the candidate ineligible.

## Interfaces

```
Claude Code / Codex / other LLM --MCP--> vrecon-mcp --+
human ------------------------------> vrecon CLI ----+--> core --> Blender (pinned)
outputs: .blend, single-file HTML report (comparisons, uncertain items, repair history)
```

## Layout

| Path | Role |
|---|---|
| `src/vrecon/schema` | Scene state types, schema versions, migrations |
| `src/vrecon/analysis` | Adapters to analysis plugins |
| `src/vrecon/planning` | What goes explicit in 3D vs. left to generation |
| `src/vrecon/scene` | Blender builder and read-back |
| `src/vrecon/generation` | Generator adapters |
| `src/vrecon/compare` | Diagnostics at the three comparison points |
| (planned) `src/vrecon/run` | Run manager, constraint manager, working copies |
| (planned) `src/vrecon/optimize` | Numeric search tools |
| (planned) `src/vrecon/mcp` | MCP server |
| `configs/modules.yaml` | Module selection and versions |
| `evaluation/thresholds.yaml` | (planned) Frozen pass/fail thresholds, committed before held-out clips are used |
| `runs/` | Execution history (git-ignored) |

## Compatibility rules

- Contracts include conventions: axes, units, matrix direction, rotation format, frame numbering origin, frame rate, intrinsics (`docs/CONVENTIONS.md`, planned).
- Each module declares inputs, outputs and supported schema versions; pinned via `uv` lockfile or container.
- Provenance (model, weights, version, license, settings) is stored with every result.
- Replacing a module requires a regression run; adoption compares quality, cost, speed and editability separately.

## Development vs. use

Cloud session: write code, test, push. Local PC: Blender GUI, GPU, local Claude Code. Exchange via git (small files and manifests only); large media stays out of git.
