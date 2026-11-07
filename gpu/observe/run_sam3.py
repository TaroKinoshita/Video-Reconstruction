"""Object masks with SAM 3.1 video text prompts. Runs in the separate observe env, not in vrecon's .venv.

    python gpu/observe/run_sam3.py VIDEO OUT.npz [--movers box] [--set floor=obj_floor --set wall=obj_wall]

Reads only the video and the text prompts. Writes masks (uint16 [T, H, W], 0 = background), id_map (JSON,
mask number -> label) and meta (JSON); vrecon reads this with optimize.load_masks. Each mover instance gets
its own label obs_N; every instance of a set prompt shares the given label. Movers are painted over set pieces.
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import inspect
import json
import subprocess
import sys
import time
import types
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.attention


def _install_edt_without_triton() -> None:
    """sam3.model.edt needs Triton (no official Windows build) only to mimic cv2.distanceTransform; use the original."""
    def edt_triton(data: torch.Tensor) -> torch.Tensor:
        out = [cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 0) for m in data.detach().cpu().numpy()]
        return torch.from_numpy(np.stack(out)).to(data.device)

    module = types.ModuleType("sam3.model.edt")
    module.edt_triton = edt_triton
    sys.modules["sam3.model.edt"] = module


def frame_count_and_size(path: str) -> tuple[int, int, int]:
    probe = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
                            "stream=width,height,nb_read_frames", "-of", "csv=p=0", path],
                           capture_output=True, encoding="utf-8", errors="replace", check=True)
    w, h, t = (int(v) for v in probe.stdout.strip().split(",")[:3])
    return t, h, w


def segment(predictor, video: str, prompt: str, frames: tuple[int, int, int], prompt_frame: int = 0) -> dict[int, np.ndarray]:
    """obj id -> bool [T, H, W]. Tracking runs both ways from prompt_frame."""
    sid = predictor.handle_request({"type": "start_session", "resource_path": video, "offload_video_to_cpu": True})["session_id"]
    predictor.handle_request({"type": "add_prompt", "session_id": sid, "frame_index": prompt_frame, "text": prompt})
    found: dict[int, np.ndarray] = {}
    for out in predictor.handle_stream_request({"type": "propagate_in_video", "session_id": sid}):
        res = out["outputs"]
        for oid, mask in zip(res["out_obj_ids"], res["out_binary_masks"]):
            found.setdefault(int(oid), np.zeros(frames, bool))[out["frame_index"]] = mask
    predictor.handle_request({"type": "close_session", "session_id": sid})
    return found


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("out")
    ap.add_argument("--movers", default="box", help="text prompt for separately labelled objects")
    ap.add_argument("--new-det-thresh", type=float, default=0.3, help="score needed to start a new object (sam3 default 0.65)")
    ap.add_argument("--prompt-frame", type=int, default=0, help="frame where the movers prompt is given: pick one where they are apart")
    ap.add_argument("--det-thresh", type=float, default=0.2, help="score for a detection to count at all (sam3 default 0.4)")
    ap.add_argument("--max-objects", type=int, default=4, help="fewer tracked objects use less VRAM")
    ap.add_argument("--set", dest="sets", action="append", default=[], metavar="PROMPT=LABEL")
    args = ap.parse_args()

    _install_edt_without_triton()
    # Windows PyTorch has no Flash Attention, and sam3 forces that backend: let PyTorch choose (before sam3 is imported).
    original_kernel = torch.nn.attention.sdpa_kernel
    torch.nn.attention.sdpa_kernel = functools.wraps(original_kernel)(lambda *a, **k: contextlib.nullcontext())
    from sam3.model_builder import build_sam3_predictor

    predictor = build_sam3_predictor(version="sam3.1", use_fa3=False, async_loading_frames=False, max_num_objects=args.max_objects)
    # This sam3 commit's base predictor always passes options the multiplex model's init_state does not take.
    init_state = predictor.model.init_state
    accepted = set(inspect.signature(init_state).parameters)
    predictor.model.init_state = lambda **kw: init_state(**{k: v for k, v in kw.items() if k in accepted})
    predictor.model.batched_grounding_batch_size = 1  # the default of 16 frames at once overflows 8 GB of VRAM
    predictor.model.new_det_thresh = args.new_det_thresh
    predictor.model.masklet_confirmation_enable = False
    # Weak detector scores must not end a track: otherwise one object is dropped and found again under a new id.
    predictor.model.score_threshold_detection = args.det_thresh
    predictor.model.suppress_unmatched_only_within_hotstart = True
    frames = frame_count_and_size(args.video)
    masks = np.zeros(frames, np.uint16)
    id_map: dict[int, str] = {}
    start = time.time()
    for spec in args.sets:
        prompt, label = spec.split("=", 1)
        number = len(id_map) + 1
        id_map[number] = label
        for m in segment(predictor, args.video, prompt, frames).values():
            masks[m] = number
    for oid, m in sorted(segment(predictor, args.video, args.movers, frames, args.prompt_frame).items()):
        number = len(id_map) + 1
        id_map[number] = f"obs_{oid}"
        masks[m] = number
    meta = {"tool": "sam3.1_multiplex", "movers": args.movers, "new_det_thresh": args.new_det_thresh, "det_thresh": args.det_thresh, "sets": args.sets, "seconds": round(time.time() - start, 2),
            "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 2**20) if torch.cuda.is_available() else 0}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as f:
        np.savez_compressed(f, masks=masks, id_map=json.dumps(id_map), meta=json.dumps(meta))
    print(json.dumps({**meta, "labels": list(id_map.values())}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
