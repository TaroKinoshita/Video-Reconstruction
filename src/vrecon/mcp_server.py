"""vrecon-mcp: the repair-loop operations as MCP tools (PLAN.md 7.1). Thin wrapper over run/service.py.

    claude mcp add vrecon -- /path/to/.venv/bin/python -m vrecon.mcp_server

All state lives in the run directory; the server keeps none.
"""

from __future__ import annotations

import json
import os

from mcp.server.mcpserver import MCPServer

from vrecon.run import service

server = MCPServer(
    name="vrecon",
    instructions=(
        "Repair loop for a video reconstruction. Typical cycle: run_state -> run_diagnose -> pick ONE hypothesis "
        "(camera focal, one object's position, size, timing or orientation) -> run_optimize with matching params "
        "-> run_history. "
        "You cannot edit the project directly; proposals are tried on a copy and adopted only if the "
        "silhouette loss improves and no locked field changes."
    ),
)


@server.tool()
def run_init(project: str, run_dir: str, masks: str, max_attempts: int = 30) -> dict:
    """Start a repair run for a project. masks: clip folder providing object masks."""
    return service.init_run(project, run_dir, masks, max_attempts)


@server.tool()
def run_state(run_dir: str) -> dict:
    """Adopted revision, current loss, attempts left, objects, cameras and locks."""
    return service.state(run_dir)


@server.tool()
def run_diagnose(run_dir: str) -> dict:
    """Per-object silhouette loss and image offsets, with a hint on reading them."""
    return service.diagnose(run_dir)


@server.tool()
def run_optimize(run_dir: str, params: list[str], hypothesis: str, max_evals: int = 120) -> dict:
    """Numeric search over params ('focal:cam_001', 'translation:obj_01:0', 'size:obj_01:2', 'time:obj_01',
    'yaw:obj_01' turn about the up axis, 'spin:obj_01' turn spread over the motion), then try to adopt."""
    return service.run_optimizer(run_dir, params, hypothesis, max_evals)


@server.tool()
def run_auto(run_dir: str, passes: int = 2, max_evals: int = 60) -> dict:
    """Scripted repair (camera first, then each object's position, size, timing). The no-LLM baseline."""
    return service.auto(run_dir, passes, max_evals)


@server.tool()
def run_submit(run_dir: str, proposal: dict) -> dict:
    """Try a hand-written proposal {id, hypothesis, changes:[{target_id, field, value}]}."""
    return service.submit(run_dir, proposal)


@server.tool()
def run_history(run_dir: str, last: int = 20) -> list[dict]:
    """Recent proposals with outcome, reasons and loss after."""
    return service.history(run_dir, last)


@server.tool()
def run_rollback(run_dir: str, revision: int) -> dict:
    """Adopt an earlier snapshot again (as a new revision)."""
    return service.rollback(run_dir, revision)


if os.environ.get("VRECON_LOOKER") == "1":  # option 2 prototype; off so condition C keeps its tool set
    from mcp.server.mcpserver import Image

    from vrecon.run import look

    @server.tool()
    def run_look(run_dir: str, frames: list[int] | None = None) -> list:
        """Picture of the reference video (left) next to the reconstruction's boxes over the same frames (right).
        Look at it and name what is wrong, then call run_fix with one issue."""
        r = look.compare_sheet(run_dir, frames)
        info = {k: r[k] for k in ("revision", "frames", "legend", "how_to_read", "issues")}
        return [json.dumps(info), Image(path=r["path"])]

    from vrecon.run import view

    def _img(r: dict, keys: tuple) -> list:
        return [json.dumps({k: r[k] for k in keys if k in r}), Image(path=r["path"])]

    @server.tool()
    def run_view_overview(run_dir: str, n: int = 16) -> list:
        """First look: n numbered frames over the whole clip and each object's speed over time. Then use
        run_view_events to see where things happen, and run_view_clip to step through those frames."""
        return _img(view.overview(run_dir, n), ("frames", "frame_count", "fps", "how_to_read"))

    @server.tool()
    def run_view_events(run_dir: str) -> dict:
        """Machine-found moments (starts, stops, sharp turns, appear/disappear, occlusion, cuts), relative to the
        background (camera pan removed), each with a frame range to look at. It can miss things."""
        return view.events(run_dir)

    @server.tool()
    def run_view_clip(run_dir: str, start: int, end: int, step: int = 1, region: str = "") -> list:
        """Consecutive frames start..end at the video's own rate (at most 16), numbered. region: an observed
        object's label to crop and enlarge around it (spatial zoom), or empty for the whole picture."""
        return _img(view.clip(run_dir, start, end, step, region=region or None), ("frames", "crop_px", "zoom"))

    @server.tool()
    def run_view_compare(run_dir: str, frames: list[int], region: str = "") -> list:
        """The video (left) and the reconstruction's silhouettes over it (right) on the same frames (at most 12),
        cropped like run_view_clip: where and when they differ."""
        return _img(view.compare(run_dir, frames, region=region or None), ("frames", "crop_px", "legend", "revision"))

    @server.tool()
    def run_view_track(run_dir: str, obj: str, start: int = 0, end: int = -1) -> dict:
        """Numbers to confirm what you saw: per frame the observed centroid and area, the reconstruction's
        centroid, the offset, and the start/stop frames of each. obj: observed label or scene object id."""
        return view.track(run_dir, obj, start, None if end < 0 else end)

    @server.tool()
    def run_fix(run_dir: str, issue: str, target: str = "", note: str = "") -> dict:
        """Fix one thing you saw. issue: camera | misplaced | wrong_size | wrong_facing | wrong_timing.
        target: object id (empty for camera). note: what you saw, in your own words. The tool finds the numbers;
        the change is adopted only if the silhouette loss improves."""
        return look.fix(run_dir, issue, target or None, note)


if os.environ.get("VRECON_REFINE_MASKS"):  # option 4: B agent + the repair loop. Masks and output folder are
    from pathlib import Path   # fixed by the launcher (env), not chosen by the agent.

    @server.tool()
    def refine_scene(blend_path: str) -> dict:
        """Refine a saved .blend with vrecon's repair loop: camera focal length, then each moving object's position,
        size, facing and timing, each change kept only if the silhouettes match the video better. Writes a
        refined copy (meshes and materials kept; transforms, camera and object scale replaced) and returns
        its path, the changes kept and the silhouette loss before/after. Save your scene first. Static
        flat meshes are treated as floor/wall; give moving objects their own mesh objects."""
        from vrecon.optimize import gt_stand_in_masks, load_masks
        from vrecon.run.refine import refine_blend

        masks = os.environ["VRECON_REFINE_MASKS"]
        obs = load_masks(masks) if masks.endswith(".npz") else gt_stand_in_masks(masks)
        _, h, w = obs.masks.shape
        root = Path(os.environ["VRECON_REFINE_DIR"])
        n = len(list(root.glob("refine_*"))) + 1 if root.exists() else 1
        rec = refine_blend(blend_path, masks, root / f"refine_{n:02d}", (w, h))
        return {k: rec[k] for k in ("refined_blend", "renamed", "objects", "accepted", "tried", "metrics_before",
                                    "metrics_after", "changes", "export_notes", "import_warnings") if k in rec}


if __name__ == "__main__":
    server.run()
