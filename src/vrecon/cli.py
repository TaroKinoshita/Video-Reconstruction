"""vrecon command line (PLAN.md 7.1). Humans and AI clients use the same commands.

    vrecon build   PROJECT [--shot S] --blend OUT.blend [--render-dir DIR --frames 0,12]
    vrecon handoff PROJECT [--shot S] --out DIR
    vrecon ingest  PROJECT --handoff DIR/handoff.json [--blend EDITED.blend] [--dry-run]
    vrecon control PROJECT [--shot S] --out DIR [--kinds depth,canny] [--depth-mode clip|frame]
    vrecon new     VIDEO --out DIR [--id proj_x] [--threshold 27] [--no-shots]
"""

from __future__ import annotations

import argparse
import sys

from vrecon.schema import load_project, save_project


def _cmd_build(args: argparse.Namespace) -> int:
    from vrecon.scene.builder import build_scene, render_frames, save_blend

    project = load_project(args.project)
    report = build_scene(project, args.shot)
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
        preview=not args.no_preview, strict_h3=args.strict,
    )
    for p in m["h3_problems"]:
        print(f"warning (H3): {p}")
    for kind, f in m["controls"].items():
        print(f"{kind}: {args.out}/{f['path']}")
    print(f"manifest: {args.out}/manifest.json ({m['frames']} frames, {m['width']}x{m['height']})")
    return 0


def _cmd_new(args: argparse.Namespace) -> int:
    from vrecon.video import new_project

    p = new_project(args.video, args.out, args.id, shots=not args.no_shots, threshold=args.threshold)
    src = p.source
    print(f"created {args.out}/project.json ({p.id})")
    print(f"  video: {src.width}x{src.height}, {src.fps.num}/{src.fps.den} fps, {src.frame_count} frames")
    for s in p.shots:
        print(f"  {s.id}: frames {s.start}-{s.end - 1} ({s.end - s.start} frames)")
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
    c.add_argument("--kinds", default="depth,canny")
    c.add_argument("--depth-mode", choices=["clip", "frame"], default="clip")
    c.add_argument("--no-preview", action="store_true")
    c.add_argument("--strict", action="store_true", help="fail if the shot does not fit H3 constraints")
    c.set_defaults(func=_cmd_control)

    n = sub.add_parser("new", help="create a project from a reference video (with shot detection)")
    n.add_argument("video")
    n.add_argument("--out", required=True)
    n.add_argument("--id")
    n.add_argument("--threshold", type=float, default=27.0, help="PySceneDetect content threshold")
    n.add_argument("--no-shots", action="store_true", help="treat the whole video as one shot")
    n.set_defaults(func=_cmd_new)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
