"""The agent's tools (session.py) as an MCP server for one run folder (env VRECON_AGENT_RUN).

    VRECON_AGENT_RUN=RUN_DIR python -m vrecon.agent.tools_mcp
"""
from __future__ import annotations

import os

from mcp.server.mcpserver import MCPServer

from vrecon.agent.session import Session

server = MCPServer(
    name="scene",
    instructions=(
        "Tools to build an editable 3D blockout of one video shot in a metric, z-up world. Cameras and body poses "
        "are fixed; you edit primitives, one scale k per person, and track removals. Cycle: overview -> "
        "structures / points_in_box / render -> edit -> verify -> fix or justify -> submit."
    ),
)


def S() -> Session:
    return Session(os.environ["VRECON_AGENT_RUN"])


@server.tool()
def overview() -> dict:
    """Shot length, camera path, people (frames, height at k=1, current k), evidence point counts per class,
    scene contents, elapsed time and budget."""
    return S().overview()


@server.tool()
def structures(cls: str | None = None, max_items: int = 30) -> dict:
    """Clusters of observed static points (cls: support, vertical, ceiling, or all): centre, 5-95 % extent,
    point count, distance, keyframes where seen. Use these to place primitives."""
    return S().structures(cls, max_items)


@server.tool()
def points_in_box(lo: list[float], hi: list[float]) -> dict:
    """Observed static points inside an axis-aligned box [lo, hi] (metres): count per class, median, 5/95 %."""
    return S().points_in_box(lo, hi)


@server.tool()
def ground_height(xy: list[list[float]]) -> dict:
    """Height of the system's fitted ground at each [x, y]."""
    return S().ground_height(xy)


@server.tool()
def pixel_to_world(frame: int, x: float, y: float) -> dict:
    """The observed surface point under video pixel (x, y) (the video is 960 x 408; see the scale in a source
    render's note) at the keyframe nearest to frame: depth and world position."""
    return S().pixel(frame, x, y)


@server.tool()
def person(label: str, every: int = 24) -> dict:
    """Samples of one person's body with the current k: centre, lowest point, gap from feet to support."""
    return S().person(label, every)


@server.tool()
def list_primitives(cls: str | None = None, include_imagined: bool = False, near: list[float] | None = None,
                    radius: float = 5.0, max_items: int = 80) -> dict:
    """Primitives (name, type, class, center, size, rotation); filter by class or by distance to [x, y]."""
    return S().list_primitives(cls, include_imagined, near, radius, max_items)


@server.tool()
def edit(ops: list[dict]) -> dict:
    """Apply edits atomically (all or nothing). Ops:
    {"op": "add", "primitives": [{"name", "type": box|cylinder|plane|ellipsoid, "center": [x,y,z],
      "size": [sx,sy,sz] full extents (cylinder: diameter, diameter, height along its axis), "rotation": [rx,ry,rz]
      degrees (Blender XYZ), "class": e.g. floor|wall|ceiling|trunk|rock|prop, "colour": [r,g,b] linear 0-1}]}
    {"op": "update", "name": N, "set": {field: value}}
    {"op": "remove", "names": [...]} or {"op": "remove", "class": C, "imagined": true|false}
    {"op": "fixed", "ground": bool, "backdrop": bool, "forest_wall": bool}  (the system's meshes)"""
    return S().edit(ops)


@server.tool()
def set_human_scale(label: str, mode: str, value: float | None = None, primitive: str | None = None,
                    frames: list[int] | None = None) -> dict:
    """One scale k per person, about the camera centres (picture kept): mode 'direct' (value = k), 'height'
    (value = metres), 'support' (feet on the ground, or on the named primitive; optional frames [a, b])."""
    return S().set_human_scale(label, mode, value, primitive, frames)


@server.tool()
def remove_track(label: str, reason: str, frames: list[int] | None = None) -> dict:
    """Remove a whole person track, or frames [a, b] of it (non-human object, tracking failure). Needs a reason."""
    return S().remove_track(label, reason, frames)


@server.tool()
def restore_track(label: str) -> dict:
    """Undo removals for one person."""
    return S().restore_track(label)


@server.tool()
def verify(max_issues: int = 40) -> dict:
    """Run the four checks: video consistency (primitives in front of observed surfaces), evidence coverage
    (observed support / vertical / ceiling surfaces not covered), human-environment (height, feet, penetration),
    completeness (floating, duplicated, buried, camera inside). Returns open issues."""
    return S().verify(max_issues)


@server.tool()
def render(view: str = "source", frame: int = 0, azimuth: float = 45.0, elevation: float = 35.0,
           distance: float | None = None, target: list[float] | None = None,
           highlight: list[str] | None = None, hide: list[str] | None = None) -> dict:
    """Draw a diagnostic picture and return its path (open it with Read). view: 'source' (film | scene from the
    shot camera at frame), 'orbit' (around target at azimuth / elevation / distance), 'top' (plan view).
    highlight: primitive names to label in red; hide: classes not to draw (e.g. ["foliage"])."""
    return S().render(view, frame, azimuth, elevation, distance, target, highlight, hide)


@server.tool()
def justify(issue_id: str, reason: str) -> dict:
    """Explain an issue that should stay (e.g. jumping, a sitting pose, depth noise on foliage)."""
    return S().justify(issue_id, reason)


@server.tool()
def submit(note: str = "") -> dict:
    """Finish: accepted when every issue is resolved or justified."""
    return S().submit(note)


if __name__ == "__main__":
    server.run()
