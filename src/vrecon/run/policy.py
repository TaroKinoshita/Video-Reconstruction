"""Fixed repair policy: a scripted stand-in for the LLM's choice of hypotheses.

Order per pass: camera focal (searched and judged on set pieces only: outlines + tracks) -> for each
moving object: floor position, size, timing. Used by `vrecon run auto` and the repair benchmark, and as
the "no LLM" ablation of condition C.
"""

from __future__ import annotations

import numpy as np

from vrecon.optimize import Param, apply_params, combined_loss, optimize

from .manager import AcceptRule, RunManager

# camera proposals are judged on camera evidence, and may not make the whole fit worse: otherwise a focal
# change that pulls objects off their observations is adopted (syn_04, condition C trial 2026-11-09)
# Tolerance: with objects still misplaced, the correct focal can first raise the whole loss a little (up to
# +0.024 on the injected-error benchmark, syn_06); the harmful LLM focal changes raised it by about +0.1.
# 0.03 is a draft value chosen on those six synthetic clips.
CAMERA_GUARD = 0.03
CAMERA_RULE = AcceptRule(primary="camera_loss", min_gain=1e-4, guards={"silhouette_loss": CAMERA_GUARD})
FOCAL_PROBE = 0.1  # relative change of focal length used to test the evidence
FOCAL_MIN_RISE = 0.001  # draft value: smallest loss rise on both sides that counts as evidence


def focal_is_weak(loss, found: float, probe: float = FOCAL_PROBE, min_rise: float = FOCAL_MIN_RISE) -> bool:
    """True when moving the found focal value by +-probe barely raises the loss on at least one side.

    With a locked-off camera, tracks say nothing about focal length and only outlines remain, so tiny mask
    errors can decide the minimum.
    """
    base = loss(found)
    return min(loss((1 + found) * (1 - probe) - 1), loss((1 + found) * (1 + probe) - 1)) - base < min_rise


MOVE_STEP = 0.035  # first search step for position, as a share of the camera-to-object distance
SIZE_STEP = 0.02  # same for size
YAW_STEP = 0.3  # first search step for orientation, radians (about 17 degrees)


def _distance(project, o) -> float:
    """Distance from the first camera to the object at shot frame 0 (the scene's own units)."""
    from vrecon.metrics.poses import camera_pose, object_pose

    cam = project.reconstruction.cameras[0]
    _, centre, _ = camera_pose(cam, 0)
    _, t, _ = object_pose(project, o, cam.shot_id, 0)
    return float(np.linalg.norm(np.asarray(t) - centre))


def hypotheses(project) -> list[tuple[str, list[Param], str]]:
    hs = [("camera focal length is off (judged on the set: outlines + tracks)", [Param("focal", "cam_001", step=0.1)],
           "set")]
    hs[0][1][0].target_id = project.reconstruction.cameras[0].id
    for o in project.reconstruction.objects:
        if o.kind == "set":
            continue
        # steps scale with the scene, not metres: a scene from a 4D model is in its own units. The distance to
        # the camera, not the object's size: a small object can be as misplaced as a large one (syn_04).
        # 0.035 / 0.02 of the ~7 m synthetic distances give the former fixed 0.25 m / 0.15 m.
        ref = _distance(project, o)
        move, grow = MOVE_STEP * ref, SIZE_STEP * ref
        hs.append((f"{o.id} misplaced on the floor",
                   [Param("translation", o.id, 0, move), Param("translation", o.id, 1, move)], "all"))
        hs.append((f"{o.id} has the wrong size", [Param("size", o.id, a, grow) for a in range(3)], "all"))
        # orientation: a box seen mostly from the front gets its facing from few points, and a moving one
        # may turn more (or less) than its per-frame boxes say
        turn = [Param("yaw", o.id, step=YAW_STEP)] + ([Param("spin", o.id, step=YAW_STEP)] if o.keys else [])
        hs.append((f"{o.id} faces the wrong way", turn, "all"))
        if o.keys:
            hs.append((f"{o.id} moves at the wrong time",  # with position: see optimize._scan_time
                       [Param("time", o.id, step=4.0), Param("translation", o.id, 0, move),
                        Param("translation", o.id, 1, move)], "all"))
    return hs


def auto_repair(mgr: RunManager, obs, pairs: dict[str, str], tracks=None, passes: int = 2, evals: int = 60,
                focal: bool = True, rays: bool = False) -> list[dict]:
    """Run the fixed hypothesis order through the run manager. Returns one record per attempt.
    focal=False skips the focal-length hypothesis (e.g. when the start comes from a 4D model whose focal
    error the repair tends to trade against other errors). rays=True first proposes, for each moving object,
    a path from its mask centres (optimize.rays), judged by the usual accept rule."""
    set_ids = sorted(o.id for o in mgr.adopted.reconstruction.objects if o.kind == "set")
    set_pairs = {k: v for k, v in pairs.items() if v in set_ids}
    records = []
    for n in range(passes):
        if rays:
            from vrecon.optimize.rays import ray_proposal

            for o in [o for o in mgr.adopted.reconstruction.objects if o.keys]:
                prop = ray_proposal(mgr.adopted, obs, pairs, o.id, f"auto{n}_ray_{mgr.attempts + 1:03d}")
                if prop is None:
                    continue
                out = mgr.try_proposal(prop)
                records.append({"pass": n, "hypothesis": prop.hypothesis, "status": out.status})
                if out.status == "budget_exhausted":
                    return records
        for i, (hyp, params, judged_on) in enumerate(hypotheses(mgr.adopted)):
            if judged_on == "set" and not focal:
                continue
            pid = f"auto{n}_{i:02d}_{mgr.attempts + 1:03d}"
            if judged_on == "set":
                prop, info = optimize(mgr.adopted, obs, set_pairs, params, hyp, pid, max_evals=evals, tracks=tracks,
                                      track_ids=set_ids or None, use_silhouette=bool(set_pairs))
                found = next(iter(info["values"].values()))

                def camera_loss(v, project=mgr.adopted):
                    return combined_loss(apply_params(project, params, np.array([v]))[0], obs, set_pairs, tracks, set_ids or None,
                                         use_silhouette=bool(set_pairs))

                if focal_is_weak(camera_loss, found):  # keep the focal length and leave the question open
                    records.append({"pass": n, "hypothesis": hyp, "status": "held", "loss": info["loss_after"],
                                    "reason": "camera loss barely changes with focal length", "uncertain": "focal_length"})
                    continue
                out = mgr.try_proposal(prop, rule=CAMERA_RULE)
            else:
                prop, info = optimize(mgr.adopted, obs, pairs, params, hyp, pid, max_evals=evals, tracks=tracks)
                out = mgr.try_proposal(prop)
            records.append({"pass": n, "hypothesis": hyp, "status": out.status, "loss": info["loss_after"]})
            if out.status == "budget_exhausted":
                return records
    return records
