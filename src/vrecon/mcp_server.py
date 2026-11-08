"""vrecon-mcp: the repair-loop operations as MCP tools (PLAN.md 7.1). Thin wrapper over run/service.py.

    claude mcp add vrecon -- /path/to/.venv/bin/python -m vrecon.mcp_server

All state lives in the run directory; the server keeps none.
"""

from __future__ import annotations

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


if __name__ == "__main__":
    server.run()
