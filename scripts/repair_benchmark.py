"""Repair benchmark on the synthetic clips: does the repair loop recover known errors?

    python scripts/repair_benchmark.py [--clips syn_01 ...] [--passes 2] [--evals 60]

For each clip: take the ground truth, inject seeded errors (focal length; per moving object: floor
position, size, motion timing), then run a FIXED hypothesis order through the optimizer and the run
manager (no LLM). Score before/after against ground truth.

What this does and does not show: masks are the ground-truth stand-in and the errors are synthetic,
so this tests the repair machinery (search + adoption rules), not the research hypothesis (PLAN.md 11.8).
Outputs: evaluation/repair_benchmark/results.json and results.md; run folders under runs/ (ignored).
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from vrecon.evaluate import score
from vrecon.optimize import combined_loss, gt_stand_in_masks
from vrecon.optimize.tracks import track_points
from vrecon.run import AcceptRule, RunManager
from vrecon.run.policy import auto_repair
from vrecon.run.service import auto_pairs
from vrecon.schema import Pose, load_project, save_project

KEYS = ["camera_focal_rel_mean", "reprojection_error_mean", "mask_iou_mean", "timing_abs_error_frames_mean"]


def perturb(project, seed: int):
    rng = random.Random(seed)
    p = project.model_copy(deep=True)
    injected = {}
    cam = p.reconstruction.cameras[0]
    m = 1 + rng.choice([-1, 1]) * rng.uniform(0.1, 0.2)
    cam.intrinsics = cam.intrinsics.model_copy(update={"fx": cam.intrinsics.fx * m, "fy": cam.intrinsics.fy * m})
    injected["focal_multiplier"] = m
    for o in p.reconstruction.objects:
        if o.kind == "set":
            continue
        dx, dy = rng.uniform(-0.5, 0.5), rng.uniform(-0.5, 0.5)
        sz = tuple(v * rng.uniform(0.7, 1.3) for v in o.size)
        dt = rng.choice([-1, 1]) * rng.randint(2, 6)
        def move(k, dx=dx, dy=dy):
            t = k.pose.translation
            return k.model_copy(update={"pose": Pose(translation=(t[0] + dx, t[1] + dy, t[2]), rotation=k.pose.rotation)})
        o.rest = move(o.rest)
        o.keys = [move(k).model_copy(update={"time": k.time + dt / 24}) for k in o.keys]
        o.size = sz
        o.status = "estimated"
        injected[o.id] = {"dx": dx, "dy": dy, "size": sz, "time_frames": dt}
    return p, injected


def summarize(rep):
    return {k: rep["summary"].get(k) for k in KEYS}


def run_clip(clip: Path, out_root: Path, passes: int, evals: int) -> dict:
    gt = load_project(clip / "gt.json")
    wrong, injected = perturb(gt, seed=int(clip.name.split("_")[-1]))
    obs = gt_stand_in_masks(clip)
    run_dir = out_root / clip.name
    if run_dir.exists():
        import shutil

        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    save_project(wrong, run_dir / "project.json")
    pairs = auto_pairs(wrong, obs)
    set_ids = {o.id for o in wrong.reconstruction.objects if o.kind == "set"}

    tracks = track_points(clip / "video.mp4")  # from the reference video itself (a real observation)

    set_pairs = {k: v for k, v in pairs.items() if v in set_ids}

    def evaluate(p):
        return {"silhouette_loss": combined_loss(p, obs, pairs, tracks),  # silhouette + weighted track loss
                "camera_loss": combined_loss(p, obs, set_pairs, tracks, sorted(set_ids), use_silhouette=bool(set_pairs))}


    mgr = RunManager(run_dir / "project.json", run_dir / "run", evaluate, AcceptRule(primary="silhouette_loss",
                                                                                      min_gain=1e-4), max_attempts=60)
    before = summarize(score(clip, wrong))
    loss0 = mgr.metrics["silhouette_loss"]
    start = time.time()
    outcomes = auto_repair(mgr, obs, pairs, tracks, passes, evals)
    seconds = time.time() - start
    after = summarize(score(clip, mgr.adopted))
    return {"clip": clip.name, "pairs": pairs, "injected": injected, "before": before, "after": after,
            "silhouette_loss": [loss0, mgr.metrics["silhouette_loss"]],
            "accepted": sum(o["status"] == "accepted" for o in outcomes), "tried": len(outcomes),
            "seconds": round(seconds, 1), "outcomes": outcomes}


def fmt(v, pct=False):
    if v is None:
        return "-"
    return f"{v * 100:.1f}%" if pct else f"{v:.3f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", default="evaluation/synthetic")
    ap.add_argument("--clips", nargs="*")
    ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--evals", type=int, default=60)
    ap.add_argument("--out", default="evaluation/repair_benchmark")
    args = ap.parse_args()
    clips = sorted(p for p in Path(args.synthetic).iterdir() if (p / "gt.json").is_file())
    if args.clips:
        clips = [c for c in clips if c.name in args.clips]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for clip in clips:
        r = run_clip(clip, Path("runs/repair_benchmark"), args.passes, args.evals)
        results.append(r)
        write(out, args, results)  # after every clip, so an interrupted run keeps what it finished
        print(f"{r['clip']}: loss {r['silhouette_loss'][0]:.3f} -> {r['silhouette_loss'][1]:.3f}, "
              f"accepted {r['accepted']}/{r['tried']}, {r['seconds']} s", flush=True)
    print((out / "results.md").read_text())
    return 0


def write(out: Path, args, results: list) -> None:
    (out / "results.json").write_text(json.dumps({"settings": vars(args), "results": results}, indent=2) + "\n")
    rows = ["| clip | loss (silhouette + tracks) | focal error | reprojection error | mask IoU | timing error (frames) | accepted |",
            "|---|---|---|---|---|---|---|"]
    for r in results:
        b, a = r["before"], r["after"]
        rows.append(f"| {r['clip']} | {fmt(r['silhouette_loss'][0])} → {fmt(r['silhouette_loss'][1])} "
                    f"| {fmt(b['camera_focal_rel_mean'], True)} → {fmt(a['camera_focal_rel_mean'], True)} "
                    f"| {fmt(b['reprojection_error_mean'], True)} → {fmt(a['reprojection_error_mean'], True)} "
                    f"| {fmt(b['mask_iou_mean'])} → {fmt(a['mask_iou_mean'])} "
                    f"| {fmt(b['timing_abs_error_frames_mean'])} → {fmt(a['timing_abs_error_frames_mean'])} "
                    f"| {r['accepted']}/{r['tried']} |")
    note = ("Masks: ground-truth stand-in. Point tracks: OpenCV Lucas-Kanade on the reference video. "
            "Errors: injected (focal, floor position, size, timing). "
            "Fixed hypothesis order, no LLM. Tests the repair machinery only (PLAN.md 11.8).\n"
            "Reprojection error is a fraction of the image diagonal.\n\n")
    (out / "results.md").write_text("# Repair benchmark\n\n" + note + "\n".join(rows) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
