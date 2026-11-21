"""Jobs for the staged V2V A/B of one clip (scripts/prep_ab.sh), then a dry run of every group.

    python scripts/make_ab_jobs.py CLIP

Reads runs/p5/<clip>/: look.txt, source_refs/ (the source pictures shared by every job), timeline_improved.txt,
<cond>_<move>/control_sem.mp4 and improved_<move>/first_frame.png. Writes runs/p5/<clip>/jobs.json with groups s1-s4 (only s1 when the s2/s4 inputs are absent)
(see prep_ab.sh). Every job has seed 43 and the same base prompt; a stage adds exactly one thing to s1's improved job.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def control_size(d: Path) -> str:
    """WxH of the clip's controls (every control of a clip has the same size)."""
    v = next(d.glob("*_*/control_sem.mp4"))
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v", "-show_entries", "stream=width,height",
                          "-of", "csv=p=0", str(v)], capture_output=True, text=True, encoding="utf-8").stdout
    w, h = out.strip().split(",")[:2]
    return f"{w}x{h}"


def main() -> int:
    from first_frame_ref import LEGEND

    clip = sys.argv[1]
    d = ROOT / "runs" / "p5" / clip
    look = (d / "look.txt").read_text(encoding="utf-8").strip()
    src = [f"source_refs/{p.name}" for p in sorted((d / "source_refs").glob("*.png"))]
    tl = d / "timeline_improved.txt"
    newer = [c for c in d.glob("improved_*/control_sem.mp4") if c.stat().st_mtime > tl.stat().st_mtime + 60]
    if newer:  # the scene was rebuilt after the timeline was written: the timeline may describe things no longer there
        raise SystemExit(f"{tl} is older than {newer[0]}: run scripts/scene_timeline.py on the improved .blend again")
    timeline = tl.read_text(encoding="utf-8").strip()
    legend = LEGEND.replace("as in image 2", "as in the pictures")

    def prompt(pictures: str, extra: str = "") -> str:
        return (f"{pictures} Keep the look of every person and object from the pictures: {look}\n"
                f"<Video 1> is a colour-coded clay 3D render of the shot: follow it for the camera, the position, size "
                f"and outline of every object and person, and every motion, frame by frame. Its colours are labels, not "
                f"looks: {legend}. Do not copy its colours or clay look. Anything cut by the frame edge continues outside "
                f"the picture. Do not add any person.\n{extra}Realistic footage, sharp, natural colours.")

    src_pics = ", ".join(f"<Picture {i + 1}>" for i in range(len(src))) + " show the original video."
    jobs = []
    for mv in ("orbit45", "dolly-60"):
        for k in ("bare", "improved"):
            jobs.append({"id": f"s1_{k}_{mv}", "group": "s1", "control": f"{k}_{mv}/control_sem.mp4",
                         "ref_images": src, "prompt": prompt(src_pics)})
        if not (d / f"improved_{mv}/first_frame.png").is_file() and not (d / f"improved_coarse_{mv}").is_dir():
            continue  # s1-only prep (prep_ab.sh S1_ONLY=1): no first frame, no coarse control
        ff = [f"improved_{mv}/first_frame.png"] + src
        pics2 = "<Picture 1> is the first frame of this shot from the new camera; " + ", ".join(
            f"<Picture {i + 2}>" for i in range(len(src))) + " show the original video."
        jobs.append({"id": f"s2_improved_firstframe_{mv}", "group": "s2", "control": f"improved_{mv}/control_sem.mp4",
                     "ref_images": ff, "prompt": prompt(pics2)})
        jobs.append({"id": f"s3_improved_timeline_{mv}", "group": "s3", "control": f"improved_{mv}/control_sem.mp4",
                     "ref_images": src, "prompt": prompt(src_pics, f"Timeline: {timeline}\n")})
        jobs.append({"id": f"s4_improved_coarse_{mv}", "group": "s4", "control": f"improved_coarse_{mv}/control_sem.mp4",
                     "ref_images": src, "prompt": prompt(src_pics)})
    for j in jobs:
        j["scene"] = j["id"]
        j["control_kind"] = "semantic_clay"
    (d / "jobs.json").write_text(json.dumps({"about": "staged A/B (scripts/prep_ab.sh)", "defaults": {"seeds": [43]},
                                             "jobs": jobs}, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    for g in sorted({j["group"] for j in jobs}):
        r = subprocess.run([sys.executable, str(ROOT / "gpu/h3_control/comfy_jobs.py"), "--jobs", str(d / "jobs.json"),
                            "--out", str(d / "out"), "--profile", str(ROOT / "gpu/h3_control/comfy/profiles/r2v_refs.json"),
                            "--size", control_size(d), "--group", g, "--dry-run", "--server", "http://none"],
                           capture_output=True, encoding="utf-8", errors="replace")
        lines = [x for x in r.stdout.splitlines() if "planned" in x or "problem" in x]
        print(g, lines[:3])
        if r.returncode != 0 or any("problem" in x for x in lines) or not any("planned" in x for x in lines):
            print(f"dry run of {g} failed (exit {r.returncode}):\n{r.stderr[-2000:]}", file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
