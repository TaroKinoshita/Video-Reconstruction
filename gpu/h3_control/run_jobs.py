"""Run phase-1a generation jobs with MiniMax-H3 + Fun ControlNet Union 2.0 (VideoX-Fun).

Runs on the GPU machine, inside the VideoX-Fun environment (see README.md here). It does not import vrecon.

    python gpu/h3_control/run_jobs.py --jobs evaluation/phase1a/jobs.json \
        --videox third_party/VideoX-Fun --out runs/phase1a/generated [--group main] [--dry-run]

Based on VideoX-Fun examples/minimax_h3_fun/predict_v2v_control.py (commit e8c314f): same model loading,
same pipeline call. Differences: loads the model once and loops over jobs x seeds, skips runs that already
have an output (resumable), and writes one metadata JSON per run plus results_manifest.json.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

H3_FPS = 24
H3_MAX_FRAMES = 243


def snap_num_frames(actual: int, max_frames: int) -> int:
    n = min(actual, max_frames)
    return max((n - 5) // 17 * 17 + 5, 5)


def git_commit(path: Path) -> str | None:
    try:
        return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, encoding="utf-8", errors="replace",
                              check=True).stdout.strip()
    except Exception:
        return None


def probe(path: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
         "-show_entries", "stream=width,height,r_frame_rate,nb_read_frames", "-of", "json", str(path)],
        capture_output=True, encoding="utf-8", errors="replace", check=True,
    )
    s = json.loads(out.stdout)["streams"][0]
    num, den = (int(v) for v in s["r_frame_rate"].split("/"))
    return {"width": s["width"], "height": s["height"], "fps": num / den, "frames": int(s["nb_read_frames"])}


def plan_runs(jobs_file: Path, group: str | None, only: list[str] | None, seeds: list[int] | None) -> list[dict]:
    spec = json.loads(jobs_file.read_text(encoding="utf-8"))
    base = jobs_file.parent
    defaults = spec["defaults"]
    runs = []
    for job in spec["jobs"]:
        if group and job.get("group") != group:
            continue
        if only and job["id"] not in only:
            continue
        settings = {**defaults, **{k: v for k, v in job.items() if k in defaults}}
        for seed in seeds or settings["seeds"]:
            runs.append({
                "run_id": f"{job['id']}__s{seed}",
                "job": job,
                "control_path": str((base / job["control"]).resolve()),
                "seed": seed,
                "settings": {k: v for k, v in settings.items() if k != "seeds"},
            })
    return runs


def check_runs(runs: list[dict]) -> list[str]:
    problems = []
    for r in runs:
        p = Path(r["control_path"])
        if not p.is_file():
            problems.append(f"{r['run_id']}: control video missing: {p}")
            continue
        info = probe(p)
        h, w = r["settings"]["sample_size"]
        if (info["width"], info["height"]) != (w, h):
            problems.append(f"{r['run_id']}: control is {info['width']}x{info['height']}, sample_size is {w}x{h} "
                            "(the pipeline would resize/crop)")
        if abs(info["fps"] - H3_FPS) > 1e-3:
            problems.append(f"{r['run_id']}: control is {info['fps']} fps, H3 uses {H3_FPS}")
        if snap_num_frames(info["frames"], H3_MAX_FRAMES) != info["frames"]:
            problems.append(f"{r['run_id']}: {info['frames']} frames is not 17n+5; H3 would drop frames")
        r["control_info"] = info
    return problems


def contact_sheet(video: Path, out: Path, frames: int) -> None:
    picks = sorted({0, frames // 2, frames - 1})
    sel = "+".join(f"eq(n\\,{i})" for i in picks)
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-y", "-i", str(video), "-vf",
         f"select='{sel}',scale=426:-2,tile={len(picks)}x1", "-frames:v", "1", "-q:v", "4", str(out)],
        check=False,
    )


def load_pipeline(args):
    """Model loading, following predict_v2v_control.py (single GPU)."""
    import torch
    from omegaconf import OmegaConf
    from videox_fun.models import (AutoencoderKLMiniMaxH3, AutoencoderKLMiniMaxH3Audio,
                                   MiniMaxH3ControlTransformer3DModel, Qwen2TokenizerFast,
                                   Qwen3VLForConditionalGeneration, Qwen3VLProcessor)
    from videox_fun.pipeline import MiniMaxH3ControlPipeline
    from videox_fun.utils import MiniMaxH3Scheduler, apply_gpu_memory_mode, convert_model_weight_to_float8

    dtype = torch.bfloat16
    device = "cuda"
    cfg = OmegaConf.load(args.config)
    kwargs = OmegaConf.to_container(cfg["transformer_additional_kwargs"], resolve=True)
    transformer = MiniMaxH3ControlTransformer3DModel.from_pretrained(
        args.model, subfolder="transformer", low_cpu_mem_usage=True, torch_dtype=dtype, **kwargs)
    from safetensors.torch import load_file
    state = load_file(args.controlnet)
    state = state.get("state_dict", state)
    missing, unexpected = transformer.load_state_dict(state, strict=False)
    print(f"controlnet loaded: missing {len(missing)}, unexpected {len(unexpected)}", flush=True)
    if unexpected:
        raise SystemExit(f"controlnet has {len(unexpected)} unexpected keys: wrong --config for this checkpoint?")

    vae = AutoencoderKLMiniMaxH3.from_pretrained(args.model, subfolder="vae", low_cpu_mem_usage=True, torch_dtype=dtype)
    audio_vae = AutoencoderKLMiniMaxH3Audio.from_pretrained(args.model, subfolder="audio_vae", low_cpu_mem_usage=True,
                                                            torch_dtype=dtype)
    tokenizer = Qwen2TokenizerFast.from_pretrained(os.path.join(args.model, "tokenizer"))
    processor = Qwen3VLProcessor.from_pretrained(os.path.join(args.model, "processor"))
    text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
        os.path.join(args.model, "text_encoder"), low_cpu_mem_usage=True, torch_dtype=dtype).eval()
    scheduler = MiniMaxH3Scheduler.from_pretrained(args.model, subfolder="scheduler")
    audio_scheduler = MiniMaxH3Scheduler.from_pretrained(args.model, subfolder="audio_scheduler")
    pipe = MiniMaxH3ControlPipeline(vae=vae, audio_vae=audio_vae, text_encoder=text_encoder, tokenizer=tokenizer,
                                    processor=processor, transformer=transformer, scheduler=scheduler,
                                    audio_scheduler=audio_scheduler)
    if "qfloat8" in args.memory_mode:
        convert_model_weight_to_float8(transformer, exclude_module_name=[
            "proj_in", "audio_proj_in", "context_embedder", "time_embedder", "time_proj",
            "token_refiner", "norm_out", "proj_out", "audio_proj_out"], device=device)
    apply_gpu_memory_mode(pipe, args.memory_mode, device, dtype,
                          quant_tag="qfloat8_preconverted" if args.memory_mode.endswith("_and_qfloat8") else None,
                          exclude_module_name=[])
    return pipe, (missing, unexpected)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--videox", type=Path, default=Path("third_party/VideoX-Fun"))
    ap.add_argument("--model", default="models/Diffusion_Transformer/MiniMax-H3",
                    help="diffusers layout or an original partition such as .../MiniMax-H3/FL2VA")
    ap.add_argument("--controlnet", default="models/Diffusion_Transformer/MiniMax-H3-Fun-Controlnet-Union-2.0/"
                                            "MiniMax-H3-Fun-Controlnet-Union-2.0.safetensors")
    ap.add_argument("--config", default=None, help="control branch yaml (default: the 2.0 inpaint layout)")
    ap.add_argument("--memory-mode", default="model_group_offload",
                    choices=["model_full_load", "model_full_load_and_qfloat8", "model_cpu_offload",
                             "model_cpu_offload_and_qfloat8", "model_group_offload", "sequential_cpu_offload"])
    ap.add_argument("--group", default="main", help="job group to run ('' for all)")
    ap.add_argument("--only", nargs="*", help="job ids to run")
    ap.add_argument("--seeds", nargs="*", type=int)
    ap.add_argument("--dry-run", action="store_true", help="check inputs and list runs; no torch needed")
    args = ap.parse_args()
    if args.config is None:
        args.config = str(args.videox / "config/minimax_h3/minimax_h3_control_inpaint_post_norm.yaml")

    runs = plan_runs(args.jobs, args.group or None, args.only, args.seeds)
    problems = check_runs(runs)
    for p in problems:
        print(f"problem: {p}")
    todo = [r for r in runs if not (args.out / r["run_id"] / "video.mp4").is_file()]
    print(f"{len(runs)} runs planned, {len(runs) - len(todo)} already done, {len(todo)} to run")
    if args.dry_run:
        for r in todo:
            print(f"  {r['run_id']}")
        return 1 if problems else 0
    if problems:
        print("fix the problems above first")
        return 1

    sys.path.insert(0, str(args.videox))
    import torch
    from videox_fun.utils import get_video_to_video_latent, save_videos_with_audio_grid

    load_start = time.time()
    pipe, (missing, unexpected) = load_pipeline(args)
    load_seconds = time.time() - load_start
    env = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_count": torch.cuda.device_count(),
        "videox_commit": git_commit(args.videox),
        "model": args.model,
        "controlnet": args.controlnet,
        "config": args.config,
        "memory_mode": args.memory_mode,
        "controlnet_missing_keys": len(missing),
        "load_seconds": round(load_seconds, 1),
    }
    print(json.dumps(env, indent=2), flush=True)

    for r in todo:
        s = r["settings"]
        run_dir = args.out / r["run_id"]
        run_dir.mkdir(parents=True, exist_ok=True)
        h, w = s["sample_size"]
        torch.cuda.reset_peak_memory_stats()
        start = time.time()
        with torch.no_grad():
            control, _, _, _ = get_video_to_video_latent(
                r["control_path"], video_length=H3_MAX_FRAMES, sample_size=[h, w], fps=H3_FPS, ref_image=None,
                keep_aspect_ratio=True)
            num_frames = snap_num_frames(control.shape[2], H3_MAX_FRAMES)
            out = pipe(
                prompt=r["job"]["prompt"],
                negative_prompt=s["negative_prompt"],
                control_video=control,
                control_context_scale=s["control_context_scale"],
                mask_video=None,
                inpaint_video=None,
                height=h,
                width=w,
                num_frames=num_frames,
                num_inference_steps=s["num_inference_steps"],
                guidance_scale=s["guidance_scale"],
                flow_shift=None,
                audio_flow_shift=None,
                generator=torch.Generator(device="cuda").manual_seed(r["seed"]),
                output_type="pt",
            )
        seconds = time.time() - start
        tmp = run_dir / "video.tmp.mp4"
        save_videos_with_audio_grid(out.videos, out.audio, str(tmp), fps=H3_FPS,
                                    audio_sample_rate=out.sampling_rate)
        os.replace(tmp, run_dir / "video.mp4")  # a crash never leaves a "done" run behind
        contact_sheet(run_dir / "video.mp4", run_dir / "sheet.jpg", num_frames)
        meta = {
            "run_id": r["run_id"],
            "job": r["job"],
            "seed": r["seed"],
            "settings": s,
            "num_frames": num_frames,
            "control_info": r.get("control_info"),
            "seconds": round(seconds, 1),
            "peak_gpu_memory_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
            "finished_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "env": env,
        }
        (run_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"done {r['run_id']} in {seconds:.0f} s", flush=True)

    summary = []
    for d in sorted(p for p in args.out.iterdir() if (p / "meta.json").is_file()):
        m = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        summary.append({k: m[k] for k in ("run_id", "seed", "num_frames", "seconds", "peak_gpu_memory_gb")}
                       | {"scene": m["job"]["scene"], "control_kind": m["job"]["control_kind"]})
    (args.out / "results_manifest.json").write_text(json.dumps({"env": env, "runs": summary}, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out / 'results_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
