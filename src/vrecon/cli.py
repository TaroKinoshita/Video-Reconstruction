"""vrecon command line (PLAN.md 7.1). Humans and AI clients use the same commands.

    vrecon build   PROJECT [--shot S] --blend OUT.blend [--render-dir DIR --frames 0,12]
    vrecon handoff PROJECT [--shot S] --out DIR
    vrecon ingest  PROJECT --handoff DIR/handoff.json [--blend EDITED.blend] [--dry-run]
    vrecon control PROJECT [--shot S] --out DIR [--kinds depth,canny,gray,clay,id,look] [--depth-mode clip|frame]
    vrecon new     VIDEO --out DIR [--id proj_x] [--threshold 27] [--no-shots]
    vrecon score   REFERENCE_DIR PROJECT [--shot S] [--out report.json]
    vrecon score-blend REFERENCE_DIR SCENE.blend [--out report.json]
    vrecon run init|state|diagnose|optimize|auto|submit|history|rollback ...   (repair loop; see run/service.py)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from vrecon.schema import load_project, save_project


def _cmd_build(args: argparse.Namespace) -> int:
    from vrecon.scene.builder import build_scene, render_frames, save_blend

    project = load_project(args.project)
    report = build_scene(project, args.shot, root=Path(args.project).resolve().parent)
    for w in report.warnings:
        print(f"warning: {w}")
    print(f"saved {save_blend(args.blend)}")
    if args.render_dir:
        frames = [int(f) for f in args.frames.split(",")]
        for p in render_frames(args.render_dir, frames, engine=args.engine, resolution_percentage=args.percent):
            print(f"rendered {p}")
    return 0


def _cmd_handoff(args: argparse.Namespace) -> int:
    from vrecon.scene.handoff import handoff

    project = load_project(args.project)
    snap = handoff(project, args.out, args.shot)
    print(f"handed off {snap.shot_id} at revision {snap.revision}")
    print(f"  edit:  {snap.blend_path}")
    print(f"  then:  vrecon ingest {args.project} --handoff {args.out}/handoff.json")
    print("  imported: transforms of objects with vrecon_id, camera settings, their keyframes")
    return 0


def _cmd_ingest(args: argparse.Namespace) -> int:
    from vrecon.scene.handoff import ingest_blend, load_snapshot
    from vrecon.scene.ingest import IngestConflict

    project = load_project(args.project)
    try:
        result = ingest_blend(project, load_snapshot(args.handoff), args.blend)
    except IngestConflict as exc:
        print(f"conflict: {exc}", file=sys.stderr)
        return 2
    for w in result.warnings:
        print(f"warning: {w}")
    if result.status == "no_changes":
        print("no importable changes")
        return 0
    for e in result.edits:
        print(f"edit {e.id}: {e.target_id}.{e.field}")
    if args.dry_run:
        print("dry run: project not saved")
    else:
        save_project(result.project, args.project)
        print(f"saved {args.project} (revision {result.project.revision})")
    return 0


def _cmd_control(args: argparse.Namespace) -> int:
    from vrecon.scene.controls import export_controls

    project = load_project(args.project)
    m = export_controls(
        project, args.out, args.shot, kinds=tuple(args.kinds.split(",")), depth_mode=args.depth_mode,
        hypothesis=tuple(args.hypothesis.split(",")),
        preview=not args.no_preview, strict_h3=args.strict, root=Path(args.project).resolve().parent,
    )
    for p in m["h3_problems"]:
        print(f"warning (H3): {p}")
    for kind, f in m["controls"].items():
        print(f"{kind}: {args.out}/{f['path']}")
    print(f"manifest: {args.out}/manifest.json ({m['frames']} frames, {m['width']}x{m['height']})")
    return 0


def _cmd_new(args: argparse.Namespace) -> int:
    from vrecon.video import new_project

    p = new_project(args.video, args.out, args.id, shots=not args.no_shots, threshold=args.threshold,
                    dedup=not args.keep_padding)
    src = p.source
    print(f"created {args.out}/project.json ({p.id})")
    print(f"  video: {src.width}x{src.height}, {src.fps.num}/{src.fps.den} fps, {src.frame_count} frames")
    if src.cadence:
        c = src.cadence
        print(f"  padding removed: 1 of every {c.period} frames repeated; {c.original_frame_count} -> "
              f"{src.frame_count} frames ({c.original_fps.num}/{c.original_fps.den} -> {src.fps.num}/{src.fps.den} fps)")
    for s in p.shots:
        print(f"  {s.id}: frames {s.start}-{s.end - 1} ({s.end - s.start} frames)")
    return 0


def _cmd_score(args: argparse.Namespace) -> int:
    import json

    from vrecon.evaluate import score

    if args.cmd == "score-blend":
        from vrecon.evaluate import score_blend

        report = score_blend(args.reference, args.project, novel=args.novel)
    else:
        report = score(args.reference, load_project(args.project), args.shot, novel=args.novel)
    text = json.dumps(report, indent=2)
    if args.out:
        from pathlib import Path

        Path(args.out).write_text(text + "\n", encoding="utf-8")
        print(f"wrote {args.out}")
    print(json.dumps(report["summary"], indent=2))
    return 0


def _cmd_score_generated(args: argparse.Namespace) -> int:
    import json
    from pathlib import Path

    from vrecon.compare.generated import load_controls, load_masks, score_generated
    from vrecon.control.core import read_video_rgb

    rep = score_generated(
        read_video_rgb(args.video), load_controls(args.controls),
        masks=load_masks(args.masks) if args.masks else None,
        base_video=read_video_rgb(args.base) if args.base else None,
        unchanged=[x for x in args.unchanged.split(",") if x] or None,
        scene=load_project(args.scene) if args.scene else None,
        camera=load_project(args.camera) if args.camera else None)
    rep["inputs"] = {k: getattr(args, k) for k in ("video", "controls", "masks", "base", "unchanged", "scene", "camera")}
    text = json.dumps(rep, indent=1, default=float)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
        print(f"wrote {args.out}")
    print(json.dumps({k: rep[k] for k in ("edges", "objects_by_status", "onset_abs_error_mean", "unchanged", "camera")
                      if k in rep}, indent=1, default=float))
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    import json

    from vrecon.run import service

    a = args.run_cmd
    if a == "init":
        out = service.init_run(args.project, args.run_dir, args.masks, args.max_attempts, video=args.video,
                               from_4d=args.from_4d, focal=args.focal, edit_set=args.edit_set, tracks=args.tracks)
    elif a == "state":
        out = service.state(args.run_dir)
    elif a == "diagnose":
        out = service.diagnose(args.run_dir)
    elif a == "optimize":
        out = service.run_optimizer(args.run_dir, args.param, args.hypothesis, args.max_evals)
    elif a == "auto":
        out = service.auto(args.run_dir, args.passes, args.max_evals)
    elif a == "submit":
        from pathlib import Path

        out = service.submit(args.run_dir, json.loads(Path(args.proposal).read_text(encoding="utf-8")))
    elif a == "history":
        out = service.history(args.run_dir, args.last)
    else:
        out = service.rollback(args.run_dir, args.revision)
    print(json.dumps(out, indent=2, default=str))
    return 0


def _cmd_triage(args) -> int:
    import json

    from vrecon.analysis.triage import triage

    out = triage(args.video)
    text = json.dumps(out, indent=2, default=str)
    if args.out:
        from pathlib import Path

        Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="vrecon")
    sub = parser.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="build a shot as a .blend (and optionally render)")
    b.add_argument("project")
    b.add_argument("--shot")
    b.add_argument("--blend", required=True)
    b.add_argument("--render-dir")
    b.add_argument("--frames", default="0")
    b.add_argument("--engine", default="CYCLES")
    b.add_argument("--percent", type=int, default=100)
    b.set_defaults(func=_cmd_build)

    h = sub.add_parser("handoff", help="build a .blend for a human to edit and save the baseline")
    h.add_argument("project")
    h.add_argument("--shot")
    h.add_argument("--out", required=True)
    h.set_defaults(func=_cmd_handoff)

    i = sub.add_parser("ingest", help="import human edits from the handed-off .blend")
    i.add_argument("project")
    i.add_argument("--handoff", required=True)
    i.add_argument("--blend", help="edited .blend (default: the handed-off file)")
    i.add_argument("--dry-run", action="store_true")
    i.set_defaults(func=_cmd_ingest)

    c = sub.add_parser("control", help="export control videos (depth, canny) for V2V")
    c.add_argument("project")
    c.add_argument("--shot")
    c.add_argument("--out", required=True)
    c.add_argument("--kinds", default="depth,canny,gray")
    c.add_argument("--depth-mode", choices=["clip", "frame", "disparity"], default="clip")
    c.add_argument("--hypothesis", default="full", help="full, soft, drop or several comma-separated (control/status.py)")
    c.add_argument("--no-preview", action="store_true")
    c.add_argument("--strict", action="store_true", help="fail if the shot does not fit H3 constraints")
    c.set_defaults(func=_cmd_control)

    n = sub.add_parser("new", help="create a project from a reference video (with shot detection)")
    n.add_argument("video")
    n.add_argument("--out", required=True)
    n.add_argument("--id")
    n.add_argument("--threshold", type=float, default=27.0, help="PySceneDetect content threshold")
    n.add_argument("--no-shots", action="store_true", help="treat the whole video as one shot")
    n.add_argument("--keep-padding", action="store_true",
                   help="keep repeated frames (frame-rate padding) instead of removing them at import")
    n.set_defaults(func=_cmd_new)

    t = sub.add_parser("triage", help="cut a video into shots and say per shot which path fits (camera still/moving)")
    t.add_argument("video")
    t.add_argument("--out")
    t.set_defaults(func=_cmd_triage)

    sc = sub.add_parser("score", help="score a reconstruction against a reference clip folder")
    sc.add_argument("reference")
    sc.add_argument("project")
    sc.add_argument("--shot")
    sc.add_argument("--out")
    sc.add_argument("--novel", action="store_true", help="also score the clip's novel views (gt_passes/novel)")
    sc.set_defaults(func=_cmd_score)

    sg = sub.add_parser("score-generated", help="score a generated video against the controls it came from (RQ2)")
    sg.add_argument("video")
    sg.add_argument("controls", help="export_controls folder (manifest.json, passes/)")
    sg.add_argument("--masks", help="observer masks of the video (SAM-style npz)")
    sg.add_argument("--base", help="generation from the unedited scene (unchanged-region measure)")
    sg.add_argument("--unchanged", default="", help="comma-separated object ids an edit must leave alone")
    sg.add_argument("--scene", help="project the controls came from (movers, camera reference)")
    sg.add_argument("--camera", help="project with a camera estimated from the video")
    sg.add_argument("--out")
    sg.set_defaults(func=_cmd_score_generated)

    pv = sub.add_parser("pipeline-v2v", help="run a clip's scene -> V2V recipe (evaluation/recipes/<clip>/v2v.json)")
    pv.add_argument("recipe")
    pv.add_argument("--from", dest="start")
    pv.add_argument("--to", dest="stop")
    pv.add_argument("--server", help="ComfyUI server for the generate stage")
    pv.add_argument("--dry-run", action="store_true")
    pv.add_argument("--scene", help="finished scene folder to start from (default: the recipe's)")
    pv.set_defaults(func=lambda args: __import__("vrecon.pipeline.v2v", fromlist=["run"]).run(
        Path(args.recipe), args.start, args.stop, args.server, args.dry_run, args.scene))

    ps = sub.add_parser("pipeline-scene", help="run a clip's observation -> scene recipe (evaluation/recipes/<clip>/scene.json)")
    ps.add_argument("recipe")
    ps.add_argument("--from", dest="start")
    ps.add_argument("--to", dest="stop")
    ps.add_argument("--dry-run", action="store_true")
    ps.add_argument("--check", action="store_true", help="compare each stage's outputs with the stored results")
    ps.set_defaults(func=lambda args: __import__("vrecon.pipeline.scene", fromlist=["run"]).run(
        Path(args.recipe), args.start, args.stop, args.dry_run, args.check))

    sb = sub.add_parser("score-blend", help="score any .blend (e.g. from an LLM) against a reference clip folder")
    sb.add_argument("reference")
    sb.add_argument("project", metavar="blend")
    sb.add_argument("--out")
    sb.add_argument("--novel", action="store_true", help="also score the clip's novel views (gt_passes/novel)")
    sb.set_defaults(func=_cmd_score)

    r = sub.add_parser("run", help="repair loop: try proposals on a copy, adopt only improvements")
    r.set_defaults(func=_cmd_run)
    rs = r.add_subparsers(dest="run_cmd", required=True)
    x = rs.add_parser("init")
    x.add_argument("project")
    x.add_argument("run_dir")
    x.add_argument("--masks", required=True,
                   help="clip folder (ground-truth stand-in masks) or masks .npz from gpu/observe/run_sam3.py")
    x.add_argument("--video", help="reference video for point tracks (default: <clip>/video.mp4)")
    x.add_argument("--from-4d", help="4RC output .npz: build the starting scene from it (needs --masks .npz)")
    x.add_argument("--focal", action=argparse.BooleanOptionalAction, default=None,
                   help="let auto repair change the focal length (default: off with --from-4d, on otherwise)")
    x.add_argument("--tracks", help="point tracks .npz from an external tracker (default: OpenCV LK on --video)")
    x.add_argument("--edit-set", action="store_true", help="let proposals change the floor and wall (default: kept fixed)")
    x.add_argument("--max-attempts", type=int, default=30)
    for name in ("state", "diagnose"):
        rs.add_parser(name).add_argument("run_dir")
    x = rs.add_parser("optimize")
    x.add_argument("run_dir")
    x.add_argument("--param", action="append", required=True, help="focal:cam_001 | translation:obj:axis | size:obj:axis | time:obj")
    x.add_argument("--hypothesis", required=True)
    x.add_argument("--max-evals", type=int, default=120)
    x = rs.add_parser("auto", help="fixed hypothesis order, no LLM (ablation of condition C)")
    x.add_argument("run_dir")
    x.add_argument("--passes", type=int, default=2)
    x.add_argument("--max-evals", type=int, default=60)
    x = rs.add_parser("submit")
    x.add_argument("run_dir")
    x.add_argument("proposal", help="JSON file")
    x = rs.add_parser("history")
    x.add_argument("run_dir")
    x.add_argument("--last", type=int, default=20)
    x = rs.add_parser("rollback")
    x.add_argument("run_dir")
    x.add_argument("revision", type=int)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
