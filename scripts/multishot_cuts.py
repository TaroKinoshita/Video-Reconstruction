"""Multi-cut pipeline, step 1: a film clip -> picture area, cuts, one video and frame folder per cut.

    python scripts/multishot_cuts.py VIDEO OUT_DIR TAG [--width 1280]

Black bars are found with ffmpeg cropdetect and cut away; the picture is scaled to WIDTH (height even). Cuts:
vrecon.video.detect_shots_fine (a transition of at most 3 frames between two cuts belongs to no shot). Writes
OUT_DIR/TAG.mp4, shots_TAG.json, TAG_shotNN.mp4, rc_TAGNN/frames/%05d.png.
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vrecon.control.core import read_video_rgb  # noqa: E402
from vrecon.video import detect_shots_fine  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("video")
ap.add_argument("out", type=Path)
ap.add_argument("tag")
ap.add_argument("--width", type=int, default=1280)
a = ap.parse_args()
a.out.mkdir(parents=True, exist_ok=True)
log = subprocess.run(["ffmpeg", "-hide_banner", "-i", a.video, "-vf", "cropdetect=24:2:0", "-frames:v", "200", "-f",
                      "null", "-"], capture_output=True, text=True, encoding="utf-8", errors="replace").stderr
crops = re.findall(r"crop=(\d+:\d+:\d+:\d+)", log)
crop = max(set(crops), key=crops.count) if crops else None
vf = (f"crop={crop}," if crop else "") + f"scale={a.width}:-2"
clip = a.out / f"{a.tag}.mp4"
subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", a.video, "-vf", vf, "-an", "-c:v", "libx264", "-crf", "14",
                str(clip)], check=True)
frames = read_video_rgb(clip)
shots, params = detect_shots_fine(frames)
trans = [tuple(t) for t in params.get("transitions", [])]
shots = [s for s in shots if tuple(s) not in trans]
(a.out / f"shots_{a.tag}.json").write_text(json.dumps({"shots": shots, "transitions": trans, "crop": crop,
                                                      "params": params}, indent=1, default=str), encoding="utf-8")
fps = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v", "-show_entries", "stream=r_frame_rate", "-of",
                      "csv=p=0", str(clip)], capture_output=True, text=True, encoding="utf-8").stdout.strip()
for k, (s0, s1) in enumerate(shots):
    v = a.out / f"{a.tag}_shot{k:02d}.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(clip), "-vf", f"select='between(n,{s0},{s1 - 1})',setpts=N/({fps})/TB",
                    "-an", "-c:v", "libx264", "-crf", "14", str(v)], check=True)
    fd = a.out / f"rc_{a.tag}{k:02d}" / "frames"
    fd.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(v), str(fd / "%05d.png")], check=True)
print(json.dumps({"crop": crop, "frames": len(frames), "shots": shots, "transitions": trans}))
