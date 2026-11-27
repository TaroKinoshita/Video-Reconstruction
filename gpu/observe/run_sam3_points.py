"""Track objects from point seeds with SAM 3.1 (no text): each seed is a few positive points on one object on one
frame (e.g. along a staff where its mask was clean); SAM's tracker carries the object through frames where a text
prompt finds nothing (motion blur). Runs in the observe env.

    python gpu/observe/run_sam3_points.py VIDEO SEEDS.json OUT.npz [--max-objects 4]

SEEDS.json: {"objects": {"<label>": [{"frame": f, "points": [[x, y], ...]}, ...]}} in pixels of VIDEO.
Writes the run_sam3.py format: masks (uint16 [T, H, W]), id_map (number -> label), meta.
"""
from __future__ import annotations

import argparse
import contextlib
import functools
import inspect
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.attention

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_sam3 import _install_edt_without_triton, frame_count_and_size  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("seeds")
    ap.add_argument("out")
    ap.add_argument("--max-objects", type=int, default=4)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
    _install_edt_without_triton()
    original_kernel = torch.nn.attention.sdpa_kernel
    torch.nn.attention.sdpa_kernel = functools.wraps(original_kernel)(lambda *a, **k: contextlib.nullcontext())
    from sam3.model_builder import build_sam3_predictor

    predictor = build_sam3_predictor(version="sam3.1", use_fa3=False, async_loading_frames=False,
                                     max_num_objects=args.max_objects)
    init_state = predictor.model.init_state
    accepted = set(inspect.signature(init_state).parameters)
    predictor.model.init_state = lambda **kw: init_state(**{k: v for k, v in kw.items() if k in accepted})
    predictor.model.batched_grounding_batch_size = 1

    # With point seeds only (no text), no full detector pass ever fills the per-frame output cache, and the tracker's
    # masks are dropped on every frame that has no cache entry: start those frames from an empty cache instead.
    model = predictor.model
    build = model._build_sam2_output

    def _build(inference_state, frame_idx, refined_obj_id_to_mask=None):
        inference_state["cached_frame_outputs"].setdefault(frame_idx, {})
        return build(inference_state, frame_idx, refined_obj_id_to_mask)
    model._build_sam2_output = _build
    if args.debug:
        pp = model._postprocess_output

        def _pp(inference_state, out, suppressed_obj_ids=None, **kw):
            print("PP", {int(k): int(v.sum()) for k, v in out["obj_id_to_mask"].items()},
                  {int(k): float(v) for k, v in out.get("obj_id_to_sam2_score", {}).items()},
                  "supp", suppressed_obj_ids, flush=True)
            return pp(inference_state, out, suppressed_obj_ids=suppressed_obj_ids, **kw)
        model._postprocess_output = _pp
    frames = frame_count_and_size(args.video)
    T, H, W = frames
    seeds = json.loads(Path(args.seeds).read_text(encoding="utf-8"))["objects"]
    labels = list(seeds)[: args.max_objects]
    start = time.time()
    sid = predictor.handle_request({"type": "start_session", "resource_path": args.video,
                                    "offload_video_to_cpu": True})["session_id"]
    for oid, lab in enumerate(labels, 1):
        for s in seeds[lab]:
            pts = [[x / W, y / H] for x, y in s["points"]]
            predictor.handle_request({"type": "add_prompt", "session_id": sid, "frame_index": int(s["frame"]),
                                      "points": pts, "point_labels": [1] * len(pts), "obj_id": oid,
                                      "clear_old_points": True})
    masks = np.zeros(frames, np.uint16)
    req = {"type": "propagate_in_video", "session_id": sid, "evict_cached_frame_outputs": True,
           "force_tracker_propagation": True,
           "start_frame_index": min(int(s["frame"]) for lab in labels for s in seeds[lab])}
    for out in predictor.handle_stream_request(req):
        res = out["outputs"]
        if args.debug:
            print("DBG", out["frame_index"], list(map(int, res["out_obj_ids"])),
                  [int(np.asarray(m).sum()) for m in res["out_binary_masks"]], flush=True)
        for oid, m in zip(res["out_obj_ids"], res["out_binary_masks"]):
            if 1 <= int(oid) <= len(labels):
                masks[out["frame_index"]][np.asarray(m, bool) & (masks[out["frame_index"]] == 0)] = int(oid)
    predictor.handle_request({"type": "close_session", "session_id": sid})
    id_map = {i: f"pt_{lab}" for i, lab in enumerate(labels, 1)}
    meta = {"tool": "sam3.1_points", "labels": labels, "seconds": round(time.time() - start, 2)}
    with open(args.out, "wb") as f:
        np.savez_compressed(f, masks=masks, id_map=json.dumps(id_map), meta=json.dumps(meta))
    print(json.dumps({**meta, "present": {lab: int((masks == i).any((1, 2)).sum()) for i, lab in id_map.items()}}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
