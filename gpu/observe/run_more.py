"""Cameras and point maps with MoRe (HellexF/MoRe, full-attention model). Runs in its own env, not in vrecon's .venv.

    python gpu/observe/run_more.py VIDEO OUT.npz --repo <MoRe checkout> [--frames 30]

Samples --frames frames evenly (np.linspace, as 4RC does, so `vrecon run init --from-4d` recovers them) and
writes the same keys as a 4RC output: n_frames, pred_<i>_extrinsic (camera-to-world, OpenCV axes, world =
first camera), pred_<i>_intrinsic (model pixels), pred_<i>_pts ([1, h, w, 3] world points), pred_<i>_conf,
pred_<i>_motion (MoRe's motion probability) and model_to_source = (sx, sy, crop_x, crop_y), with
source_px + 0.5 = (model_px + 0.5 + crop) / s per axis. Sky masking from the repo's demo is skipped.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def read_video(path: str) -> np.ndarray:
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                            "-of", "csv=p=0", path], capture_output=True, encoding="utf-8", errors="replace", check=True)
    w, h = (int(v) for v in probe.stdout.strip().split(",")[:2])
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("out")
    ap.add_argument("--repo", required=True)
    ap.add_argument("--ckpt", default="pretrained/more_full.pt", help="relative to --repo")
    ap.add_argument("--frames", type=int, default=30)
    args = ap.parse_args()
    repo = Path(args.repo).resolve()
    sys.path.insert(0, str(repo))
    try:
        import magi_attention  # noqa: F401  (needed only by the stream model's block attention)
    except ImportError:  # stub it: vggt/layers/attention.py imports it unconditionally
        import types

        def _unavailable(*a, **k):
            raise RuntimeError("MagiAttention is not installed (only the stream model needs it)")

        for name in ("magi_attention", "magi_attention.functional", "magi_attention.functional.flex_flash_attn",
                     "magi_attention.common", "magi_attention.common.enum"):
            sys.modules[name] = types.ModuleType(name)
        sys.modules["magi_attention.functional.flex_flash_attn"].flex_flash_attn_func = _unavailable
        sys.modules["magi_attention.common.enum"].AttnMaskType = type("AttnMaskType", (), {})
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    from vggt.utils.geometry import unproject_depth_map_to_point_map

    video = read_video(args.video)
    t, src_h, src_w = video.shape[:3]
    frames = [int(f) for f in np.linspace(0, t - 1, args.frames, dtype=int)]
    tmp = Path(tempfile.mkdtemp(prefix="more_"))
    paths = []
    for i, f in enumerate(frames):
        p = tmp / f"{i:05d}.png"
        Image.fromarray(video[f]).save(p)
        paths.append(str(p))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    with initialize_config_dir(version_base=None, config_dir=str(repo / "training" / "config")):
        cfg = compose(config_name="omniworld_full")
    model = instantiate(cfg.model, _recursive_=False)
    ck = torch.load(repo / args.ckpt, map_location="cpu")
    missing, unexpected = model.load_state_dict(ck.get("model", ck), strict=False)
    model = model.to(device).eval()

    images = load_and_preprocess_images(paths).to(device)  # crop mode: width 518, height a multiple of 14
    h, w = images.shape[-2:]
    new_h = round(src_h * (518 / src_w) / 14) * 14
    sx, sy, ox, oy = w / src_w, new_h / src_h, 0.0, float(max(new_h - h, 0) // 2)

    start = time.time()
    with torch.no_grad(), torch.autocast(device.type, dtype=dtype):
        x = images[None]
        tokens, ps_idx, _ = model.aggregator(x)
        pose_enc = model.camera_head(tokens, first_token=True)[-1]
        w2c, K = pose_encoding_to_extri_intri(pose_enc, x.shape[-2:])
        depth, conf = model.depth_head(tokens, x, ps_idx)
        motion = torch.sigmoid(model.motion_head(tokens, x, ps_idx)[0, ..., 0].float())
    seconds = time.time() - start
    w2c, K = w2c[0].float().cpu().numpy(), K[0].float().cpu().numpy()
    pts = unproject_depth_map_to_point_map(depth[0].float(), torch.from_numpy(w2c), torch.from_numpy(K))
    conf = conf[0].float().cpu().numpy()
    motion = motion.cpu().numpy()

    out = {"n_frames": np.array(len(frames)), "model_to_source": np.array([sx, sy, ox, oy])}
    for i in range(len(frames)):
        m = np.eye(4)
        m[:3] = w2c[i]
        out[f"pred_{i}_extrinsic"] = np.linalg.inv(m).astype(np.float32)
        out[f"pred_{i}_intrinsic"] = K[i].astype(np.float32)
        out[f"pred_{i}_pts"] = np.asarray(pts[i], np.float32)[None]
        out[f"pred_{i}_conf"] = conf[i][None].astype(np.float32)
        out[f"pred_{i}_motion"] = motion[i][None].astype(np.float32)
    meta = {"tool": "more_full", "frames": frames, "seconds": round(seconds, 2), "model_size": [int(w), int(h)],
            "missing_keys": len(missing), "unexpected_keys": len(unexpected),
            "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 2**20) if device.type == "cuda" else 0}
    out["meta"] = np.array(json.dumps(meta))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as fh:
        np.savez_compressed(fh, **out)
    print(json.dumps(meta), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
