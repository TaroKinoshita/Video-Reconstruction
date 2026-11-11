"""One place for the repair-loop operations; the CLI (`vrecon run ...`) and the MCP server call these.

A run directory holds config.json (fixed at creation: project, mask source, accept rule, budget),
history.jsonl and snapshots/. Every call reloads state from disk, so any client (human, Claude Code,
Codex, ...) can continue a run.
"""

from __future__ import annotations

import json
from pathlib import Path

from vrecon.metrics import match_objects
from vrecon.optimize import (MaskObservation, Param, combined_loss, gt_stand_in_masks, load_masks, optimize, render_silhouettes,
                             silhouette_metrics)
from vrecon.schema import load_project

from .manager import AcceptRule, Proposal, RunManager


def _config(run_dir: Path) -> dict:
    return json.loads((run_dir / "config.json").read_text(encoding="utf-8"))


def init_run(project: str, run_dir: str, masks: str, max_attempts: int = 30, min_gain: float = 0.0,
             video: str | None = None, from_4d: str | None = None, focal: bool | None = None,
             edit_set: bool = False, tracks: str | None = None) -> dict:
    """Create a run.

    masks: a synthetic clip folder (ground-truth stand-in) or a masks .npz from an external segmenter
    (gpu/observe/run_sam3.py). video: reference video for point tracks (default: <clip>/video.mp4 for a clip
    folder). from_4d: a 4RC output .npz; the starting scene is then built from it (analysis/from_4d.py) on
    top of `project` (its source and shots), saved as <run_dir>/init.json. focal: whether the automatic
    repair may change the focal length (default: no when starting from a 4D model, whose focal error the
    search trades against other errors; yes otherwise). edit_set: let proposals change the floor and wall
    (default no: they are the camera's evidence, and moving them hides camera errors).

    The pairing of observed labels with reconstruction objects is computed once here and kept: re-pairing
    on every call let a proposal lower the loss by pulling an object off every observation.
    """
    rd = Path(run_dir)
    if (rd / "config.json").exists():
        raise FileExistsError(f"{rd} already has a run")
    observed = Path(masks).suffix == ".npz"
    if video is None and not observed:
        v = Path(masks) / "video.mp4"
        video = str(v) if v.is_file() else None
    rd.mkdir(parents=True, exist_ok=True)
    start = str(Path(project).resolve())
    report = None
    if from_4d:
        start, report = _start_from_4d(project, from_4d, masks, rd)
    cfg = {"project": start, "masks": str(Path(masks).resolve()), "mask_source": "observed" if observed else "gt_stand_in",
           "video": str(Path(video).resolve()) if video else None,
           "tracks": str(Path(tracks).resolve()) if tracks else None, "from_4d": str(Path(from_4d).resolve()) if from_4d else None,
           "focal": (not from_4d) if focal is None else focal,
           "rule": {"primary": "silhouette_loss", "min_gain": min_gain}, "max_attempts": max_attempts}
    start_project = load_project(start)
    cfg["pairs"] = auto_pairs(start_project, _observation(cfg))
    cfg["fixed"] = [] if edit_set else sorted(o.id for o in start_project.reconstruction.objects if o.kind == "set")
    (rd / "config.json").write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    return {**cfg, "init_report": report} if report else cfg


def _start_from_4d(project: str, npz: str, masks: str, rd: Path) -> tuple[str, dict]:
    """Starting scene from a 4RC run: cameras, floor, wall and object boxes. 4RC samples its input frames
    evenly (np.linspace over the folder), so the sampled frames are recovered from the prediction count."""
    import numpy as np

    from vrecon.analysis.from_4d import initialize, load_4rc
    from vrecon.schema import Observation, Provenance, save_project

    if Path(masks).suffix != ".npz":
        raise ValueError("--from-4d needs observed masks (.npz): the floor, wall and movers labels")
    p = load_project(project)
    with np.load(masks) as z:
        m, id_map = z["masks"], {int(k): v for k, v in json.loads(str(z["id_map"])).items()}
    with np.load(npz) as z:
        n = int(z["n_frames"])
    t, h, w = m.shape
    frames = [int(f) for f in np.linspace(0, t - 1, n, dtype=int)]
    shot = p.shots[0]
    cams, objs, scale, report = initialize(load_4rc(npz, w, h, frames), m, id_map, w, h, shot.id, shot_start=shot.start,
                                           fps=p.source.fps.num / p.source.fps.den, observation_id="obs_4rc")
    p.observations = [o for o in p.observations if o.id != "obs_4rc"] + [Observation(
        id="obs_4rc", kind="camera", shot_id=shot.id, summary={"frames": frames, "model": "4RC"},
        provenance=Provenance(tool="4RC", params={"npz": str(Path(npz).resolve())}))]
    p.reconstruction.cameras, p.reconstruction.objects, p.reconstruction.scale = cams, objs, scale
    p.reconstruction.events = []
    out = rd / "init.json"
    save_project(p, out)
    return str(out.resolve()), report


def _observation(cfg: dict) -> MaskObservation:
    if cfg.get("mask_source") == "observed":
        return load_masks(cfg["masks"])
    return gt_stand_in_masks(cfg["masks"])


def auto_pairs(project, obs: MaskObservation, shot_id: str = "shot_001") -> dict[str, str]:
    """Observed label -> reconstruction object: movers by silhouette overlap, set pieces by label."""
    movers = [o.id for o in project.reconstruction.objects if o.kind != "set"]
    t, h, w = obs.masks.shape
    frames = list(range(0, t, 3))
    sil = render_silhouettes(project, shot_id, frames, movers, (w, h))
    sets = {o.id for o in project.reconstruction.objects if o.kind == "set"}
    obs_map = {k: v for k, v in obs.id_map.items() if v not in sets}
    pairs = match_objects(obs.masks[frames], sil, obs_map, {i + 1: m for i, m in enumerate(movers)})
    # set pieces (floor, wall) pair by label when the observation provides them: they pin the camera
    pairs.update({sid: sid for sid in sorted(sets) if sid in obs.id_map.values()})  # sorted: set order varies per process
    return pairs


def _tracks(cfg: dict):
    if not cfg.get("video"):
        return None
    from vrecon.optimize.tracks import load_tracks, track_points

    if cfg.get("tracks"):  # tracks from an external tracker (e.g. gpu/observe/run_tapnextpp.py)
        return load_tracks(cfg["tracks"])

    return track_points(cfg["video"])  # OpenCV LK on the reference video, about a second


def _manager(run_dir: Path):
    cfg = _config(run_dir)
    obs = _observation(cfg)
    tracks = _tracks(cfg)
    pairs_cache: dict = {"pairs": cfg["pairs"]} if "pairs" in cfg else {}  # runs made before pairs were kept

    def evaluate(p):
        pairs = pairs_cache.setdefault("pairs", auto_pairs(p, obs))
        sets = sorted(o.id for o in p.reconstruction.objects if o.kind == "set")
        set_pairs = {k: v for k, v in pairs.items() if v in sets}
        return {"silhouette_loss": combined_loss(p, obs, pairs, tracks),
                "camera_loss": combined_loss(p, obs, set_pairs, tracks, sets or None, use_silhouette=bool(set_pairs))}

    rule = AcceptRule(**cfg["rule"])
    mgr = RunManager(cfg["project"], run_dir, evaluate, rule, max_attempts=cfg["max_attempts"], fixed=cfg.get("fixed"))
    mgr.tracks = tracks
    return mgr, obs, pairs_cache


def state(run_dir: str) -> dict:
    """Adopted revision, metrics, attempts used, and what can be changed (targets, locks)."""
    mgr, _, pairs = _manager(Path(run_dir))
    p = mgr.adopted
    return {"revision": p.revision, "metrics": mgr.metrics, "attempts": mgr.attempts, "max_attempts": mgr.max_attempts,
            "pairs": pairs.get("pairs"),
            "objects": [{"id": o.id, "kind": o.kind, "size": o.size, "animated": bool(o.keys)}
                        for o in p.reconstruction.objects],
            "cameras": [{"id": c.id, "fx": c.intrinsics.fx, "keys": len(c.keys)} for c in p.reconstruction.cameras],
            "locks": [lk.model_dump(mode="json") for lk in p.human.locks]}


def diagnose(run_dir: str) -> dict:
    """Per-object silhouette loss and mean image offset (where the reconstruction sits vs the observation)."""
    mgr, obs, pc = _manager(Path(run_dir))
    pairs = pc["pairs"]  # fixed for the run (filled when the manager evaluated the adopted state)
    m = silhouette_metrics(mgr.adopted, obs, pairs)
    m["pairs"] = pairs
    if mgr.tracks is not None:
        from vrecon.optimize.tracks import track_metrics

        sets = [o.id for o in mgr.adopted.reconstruction.objects if o.kind == "set"]
        m["tracks_all"] = track_metrics(mgr.adopted, mgr.tracks)
        m["tracks_on_set"] = track_metrics(mgr.adopted, mgr.tracks, ids=sets or None)
    m["hint"] = ("offset_x > 0: reconstruction is right of the observation; offset_y > 0: lower. "
                 "A shared offset/scale across all objects suggests the camera; one object alone suggests that object. "
                 "tracks_on_set high: suspect the camera (focal). Camera proposals are judged on camera evidence only.")
    return m


def run_optimizer(run_dir: str, params: list[str], hypothesis: str, max_evals: int = 120) -> dict:
    """params like 'focal:cam_001', 'translation:obj_01:0', 'size:obj_01:2', 'time:obj_01', 'yaw:obj_01',
    'spin:obj_01' (optional ':axis:step'; the axis is ignored for focal, time, yaw and spin)."""
    mgr, obs, pc = _manager(Path(run_dir))
    parsed = []
    for s in params:
        parts = s.split(":")
        kind, target = parts[0], parts[1]
        axis = int(parts[2]) if len(parts) > 2 and kind not in ("focal", "time", "yaw", "spin") else 0
        step = float(parts[3]) if len(parts) > 3 else {"focal": 0.1, "time": 4.0, "yaw": 0.3, "spin": 0.3}.get(kind, 0.2)
        parsed.append(Param(kind, target, axis, step))
    pairs = pc["pairs"]  # fixed for the run (filled when the manager evaluated the adopted state)
    pid = f"opt_{mgr.attempts + 1:03d}"
    camera_only = all(p.kind == "focal" for p in parsed)
    if camera_only:  # judge camera hypotheses on camera evidence only: set outlines + tracks on the set
        sets = sorted(o.id for o in mgr.adopted.reconstruction.objects if o.kind == "set")
        set_pairs = {k: v for k, v in pairs.items() if v in sets}
        prop, info = optimize(mgr.adopted, obs, set_pairs, parsed, hypothesis, pid, max_evals=max_evals,
                              tracks=mgr.tracks, track_ids=sets or None, use_silhouette=bool(set_pairs))
        from .policy import CAMERA_RULE

        out = mgr.try_proposal(prop, rule=CAMERA_RULE)
    else:
        prop, info = optimize(mgr.adopted, obs, pairs, parsed, hypothesis, pid, max_evals=max_evals, tracks=mgr.tracks)
        out = mgr.try_proposal(prop)
    return {"outcome": out.model_dump(mode="json"), "search": info}


def auto(run_dir: str, passes: int = 2, max_evals: int = 60, rays: bool = False) -> dict:
    """Fixed hypothesis order (camera first, then each object's position, size, timing); no LLM."""
    from .policy import auto_repair

    mgr, obs, pc = _manager(Path(run_dir))
    before = dict(mgr.metrics)
    records = auto_repair(mgr, obs, pc["pairs"], mgr.tracks, passes, max_evals,
                          focal=_config(Path(run_dir)).get("focal", True), rays=rays)
    return {"revision": mgr.adopted.revision, "metrics_before": before, "metrics_after": mgr.metrics,
            "accepted": sum(r["status"] == "accepted" for r in records), "tried": len(records), "attempts": records}


def submit(run_dir: str, proposal: dict) -> dict:
    """An LLM-written proposal: {"id", "hypothesis", "changes": [{"target_id", "field", "value"}]}."""
    mgr, _, _ = _manager(Path(run_dir))
    return mgr.try_proposal(Proposal.model_validate(proposal)).model_dump(mode="json")


def history(run_dir: str, last: int = 20) -> list[dict]:
    path = Path(run_dir) / "history.jsonl"
    if not path.exists():
        return []
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()][-last:]
    return [{"proposal": r["proposal"]["id"], "hypothesis": r["proposal"]["hypothesis"],
             "source": r["proposal"]["source"], **{k: r["outcome"][k] for k in ("status", "reasons", "revision")},
             "after": (r["outcome"].get("metrics_after") or {}).get("silhouette_loss")} for r in rows]


def rollback(run_dir: str, revision: int) -> dict:
    mgr, _, _ = _manager(Path(run_dir))
    p = mgr.rollback(revision)
    return {"revision": p.revision, "metrics": mgr.metrics}


__all__ = ["auto", "auto_pairs", "diagnose", "history", "init_run", "rollback", "run_optimizer", "state", "submit"]
