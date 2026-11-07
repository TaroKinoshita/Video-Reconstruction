"""Fixed repair policy: a scripted stand-in for the LLM's choice of hypotheses.

Order per pass: camera focal (searched and judged on set pieces only: outlines + tracks) -> for each
moving object: floor position, size, timing. Used by `vrecon run auto` and the repair benchmark, and as
the "no LLM" ablation of condition C.
"""

from __future__ import annotations

from vrecon.optimize import Param, optimize

from .manager import AcceptRule, RunManager

CAMERA_RULE = AcceptRule(primary="camera_loss", min_gain=1e-4)


def hypotheses(project) -> list[tuple[str, list[Param], str]]:
    hs = [("camera focal length is off (judged on the set: outlines + tracks)", [Param("focal", "cam_001", step=0.1)],
           "set")]
    hs[0][1][0].target_id = project.reconstruction.cameras[0].id
    for o in project.reconstruction.objects:
        if o.kind == "set":
            continue
        hs.append((f"{o.id} misplaced on the floor",
                   [Param("translation", o.id, 0, 0.25), Param("translation", o.id, 1, 0.25)], "all"))
        hs.append((f"{o.id} has the wrong size", [Param("size", o.id, a, 0.15) for a in range(3)], "all"))
        if o.keys:
            hs.append((f"{o.id} moves at the wrong time",  # with position: see optimize._scan_time
                       [Param("time", o.id, step=4.0), Param("translation", o.id, 0, 0.25),
                        Param("translation", o.id, 1, 0.25)], "all"))
    return hs


def auto_repair(mgr: RunManager, obs, pairs: dict[str, str], tracks=None, passes: int = 2, evals: int = 60) -> list[dict]:
    """Run the fixed hypothesis order through the run manager. Returns one record per attempt."""
    set_ids = sorted(o.id for o in mgr.adopted.reconstruction.objects if o.kind == "set")
    set_pairs = {k: v for k, v in pairs.items() if v in set_ids}
    records = []
    for n in range(passes):
        for i, (hyp, params, judged_on) in enumerate(hypotheses(mgr.adopted)):
            pid = f"auto{n}_{i:02d}_{mgr.attempts + 1:03d}"
            if judged_on == "set":
                prop, info = optimize(mgr.adopted, obs, set_pairs, params, hyp, pid, max_evals=evals, tracks=tracks,
                                      track_ids=set_ids or None, use_silhouette=bool(set_pairs))
                out = mgr.try_proposal(prop, rule=CAMERA_RULE)
            else:
                prop, info = optimize(mgr.adopted, obs, pairs, params, hyp, pid, max_evals=evals, tracks=tracks)
                out = mgr.try_proposal(prop)
            records.append({"pass": n, "hypothesis": hyp, "status": out.status, "loss": info["loss_after"]})
            if out.status == "budget_exhausted":
                return records
    return records
