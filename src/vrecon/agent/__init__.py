"""Agentic modelling with evidence-based checks (MAVE, arXiv 2611.22464, sections 3.2-3.3), pure numpy.

prims     editable primitives (box, cylinder, plane, ellipsoid): ray casting, signed distance, from v11 parts
evidence  the geometric evidence D of one shot: cameras, observed depth at keyframes, static points with
          support / vertical / ceiling classes, people tracks, ground height field
checks    the four checks (video consistency, evidence coverage, human-environment, completeness)
session   the agent's tools on a run folder (query, edit, verify, render, justify, submit)
mcp       the tools as an MCP server for `claude -p` (scripts/agent_run.py)
"""
