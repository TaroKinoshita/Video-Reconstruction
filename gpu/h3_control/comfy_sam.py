"""Text-prompted SAM 3.1 video segmentation on a ComfyUI server (the "Video Segment SAM3" template's nodes), saved
as a masks npz in the run_sam3.py format (masks [T,H,W] int labels, id_map). Scene-independent.

    python gpu/h3_control/comfy_sam.py --server URL VIDEO --prompt "floor, ground, court" --label floor OUT.npz
        [--threshold 0.4] [--max-objects 8]

All tracked objects of the prompt are merged into one label (e.g. the floor is several pieces). The mask comes
back through a video (MaskToImage -> CreateVideo -> SaveVideo) and is thresholded at half.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from comfy_jobs import Server, output_files  # noqa: E402

SAM_CKPT = "sam3.1_multiplex_fp16.safetensors"


def graph(video_name: str, prompt: str, threshold: float, max_objects: int, fps: float, prefix: str,
          object_indices: str = "") -> dict:
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": SAM_CKPT}},
        "2": {"class_type": "LoadVideo", "inputs": {"file": video_name}},
        "3": {"class_type": "GetVideoComponents", "inputs": {"video": ["2", 0]}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["1", 1]}},
        "5": {"class_type": "SAM3_VideoTrack", "inputs": {"images": ["3", 0], "model": ["1", 0],
                                                          "detection_threshold": threshold, "max_objects": max_objects,
                                                          "detect_interval": 1, "conditioning": ["4", 0]}},
        "6": {"class_type": "SAM3_TrackToMask", "inputs": {"track_data": ["5", 0], "object_indices": object_indices}},
        "7": {"class_type": "MaskToImage", "inputs": {"mask": ["6", 0]}},
        "8": {"class_type": "CreateVideo", "inputs": {"images": ["7", 0], "fps": float(fps)}},
        "20": {"class_type": "SaveVideo", "inputs": {"video": ["8", 0], "filename_prefix": prefix, "format": "auto",
                                                     "codec": "auto"}},
    }


def masks_for(srv: Server, video_name: str, prompt: str, threshold: float, max_objects: int, fps: float, prefix: str,
              n_frames: int, size: tuple[int, int], tmp: Path, per_object: bool = False) -> list[np.ndarray]:
    """Bool masks [T,H,W]: one merged mask, or one per tracked object (tracking is cached by ComfyUI between the
    per-object graphs; an index past the last object ends the list)."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from vrecon.control.core import read_video_rgb

    out = []
    for idx in ([str(i) for i in range(max_objects)] if per_object else [""]):
        try:
            pid = srv.queue(graph(video_name, prompt, threshold, max_objects, fps, f"{prefix}_{idx or 'all'}", idx))
        except RuntimeError:
            break
        entry = srv.wait(pid)
        files = output_files(entry)
        if not files:
            break
        f = tmp / f"{prefix.replace('/', '_')}_{idx or 'all'}.mp4"
        srv.download(files[0], f)
        m = read_video_rgb(f)[..., 0] > 127
        if per_object and m.mean() == 0:
            break
        out.append(m[:n_frames])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", required=True)
    ap.add_argument("video", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--threshold", type=float, default=0.4)
    ap.add_argument("--max-objects", type=int, default=8)
    a = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from vrecon.control.core import read_video_rgb
    from vrecon.video import frame_rate, probe

    srv = Server(a.server)
    name = srv.upload(a.video, name=f"sam__{a.video.name}")
    t0 = time.time()
    pid = srv.queue(graph(name, a.prompt, a.threshold, a.max_objects, float(frame_rate(probe(a.video))),
                          f"vrecon/sam_{a.label}"))
    print("queued", pid, flush=True)
    entry = srv.wait(pid)
    files = output_files(entry)
    if not files:
        print(json.dumps(entry.get("status", {}).get("messages", [])[-3:])[:3000])
        return 2
    tmp = a.out.with_suffix(".mask.mp4")
    srv.download(files[0], tmp)
    m = read_video_rgb(tmp)[..., 0] > 127
    n_src = len(read_video_rgb(a.video))
    if len(m) != n_src:
        print(f"warning: mask has {len(m)} frames, video {n_src}")
    masks = np.where(m, 1, 0).astype(np.uint8)
    np.savez_compressed(a.out, masks=masks, id_map=json.dumps({"1": a.label}))
    meta = {"video": str(a.video), "prompt": a.prompt, "label": a.label, "threshold": a.threshold,
            "max_objects": a.max_objects, "model": SAM_CKPT, "server": a.server, "prompt_id": pid,
            "seconds": round(time.time() - t0, 1), "frames": int(len(masks)), "coverage_mean": float(m.mean())}
    a.out.with_suffix(".json").write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(meta))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
