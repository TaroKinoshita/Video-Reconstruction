"""Run generation jobs (evaluation/*/jobs.json) on a ComfyUI server through its HTTP API.

    python gpu/h3_control/comfy_jobs.py --server https://<pod-id>-8188.proxy.runpod.net \
        --jobs evaluation/phase1a/jobs.json --out runs/phase1a/generated [--group main] [--only ID ...]
        [--seeds 43] [--steps 20] [--turbo] [--dry-run]

Standard library only (runs anywhere: this PC, the pod). Per run: the control video is uploaded to ComfyUI's input
folder, the graph below is queued, the result is downloaded, and next to it meta.json records the job, seed,
settings, the graph sent, timings and the server's system stats. Runs whose video.mp4 exists are skipped.

The graph is the ComfyUI template "MiniMax H3 Fun ControlNet Union" (gpu/h3_control/comfy/template_ui_stageA.json,
stage A on an RTX 4090 24 GB, 2026-11-16) with the template's pose extraction replaced by our control video:
    UNETLoader(ref2va int8) -> [LoraLoaderModelOnly turbo, --turbo] -> MiniMaxH3FunControlNetApply(model patch,
    video VAE, control video frames, strength) -> BasicScheduler(steps) / BasicGuider(MiniMaxH3ReferenceToVideo
    conditioning) -> SamplerCustomAdvanced(res_multistep, seed) -> VAEDecode + VAEDecodeAudio -> CreateVideo(24)
    -> SaveVideo
width, height and length follow the control video (H3 constraints are checked before uploading).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
import time
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

MODELS = {
    "unet": "minimax_h3_ref2va_pruned_int8_convrot.safetensors",
    "patch": "minimax_h3_fun_controlnet_union_pruned_int8_convrot.safetensors",
    "clip": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
    "vae": "minimax_h3_video_vae_fp16.safetensors",
    "audio_vae": "minimax_h3_audio_vae_fp32.safetensors",
    "turbo_lora": "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors",
}
SAMPLER = "res_multistep"
FPS = 24


# ---------------------------------------------------------------- jobs

def _file_sha(p: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fingerprint(r: dict, a, steps: int, profile: dict) -> str:
    """sha256 over everything a generation depends on."""
    import hashlib

    base = a.jobs.parent
    files = [r["control_path"]] + ([Path(a.ref_image)] if a.ref_image else []) +         [(base / x).resolve() for x in r["job"].get("ref_images", []) + r["job"].get("extra_ref_videos", [])]
    prompt = a.prompt_file.read_text(encoding="utf-8") if a.prompt_file else r["job"].get("prompt", "")
    data = {"files": [_file_sha(f) if Path(f).is_file() else f"missing:{f}" for f in files], "prompt": prompt,
            "profile": profile, "models": dict(MODELS), "seed": r["seed"], "steps": steps, "settings": r["settings"],
            "mode": a.mode, "scheduler": a.scheduler, "size": a.size, "strength": a.strength,
            "ref_image_size": a.ref_image_size, "turbo": a.turbo}
    return hashlib.sha256(json.dumps(data, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def plan_runs(jobs_file: Path, group: str | None, only: list[str] | None, seeds: list[int] | None) -> list[dict]:
    spec = json.loads(jobs_file.read_text(encoding="utf-8"))
    defaults = spec["defaults"]
    runs = []
    for job in spec["jobs"]:
        if group and job.get("group") != group:
            continue
        if only and job["id"] not in only:
            continue
        if str(job.get("prompt", "")).startswith("@"):
            job = {**job, "prompt": spec[job["prompt"][1:]]}
        settings = {**defaults, **{k: v for k, v in job.items() if k in defaults}}
        for seed in seeds or settings["seeds"]:
            runs.append({"run_id": f"{job['id']}__s{seed}", "job": job, "seed": seed,
                         "control_path": (jobs_file.parent / job["control"]).resolve(),
                         "settings": {k: v for k, v in settings.items() if k != "seeds"}})
    return runs


def probe(path: Path) -> dict:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries",
                          "stream=width,height,r_frame_rate,nb_read_frames", "-of", "json", str(path)],
                         capture_output=True, encoding="utf-8", errors="replace", check=True)
    s = json.loads(out.stdout)["streams"][0]
    num, _, den = s["r_frame_rate"].partition("/")
    return {"width": int(s["width"]), "height": int(s["height"]), "fps": int(num) / int(den or 1),
            "frames": int(s["nb_read_frames"])}


def check(info: dict) -> list[str]:
    p = []
    if info["width"] % 32 or info["height"] % 32:
        p.append(f"size {info['width']}x{info['height']} is not a multiple of 32")
    if (info["frames"] - 5) % 17:
        p.append(f"{info['frames']} frames is not 17n+5")
    if abs(info["fps"] - FPS) > 1e-3:
        p.append(f"{info['fps']} fps, H3 is {FPS}")
    return p


# ---------------------------------------------------------------- graph

def build_graph(control_name: str, prompt: str, seed: int, width: int, height: int, length: int, steps: int,
                strength: float, turbo: bool, prefix: str, ref_image_name: str | list[str] | None = None, mode: str = "control",
                scheduler: str = "simple", ref_image_size: str = "match", extra_videos: list[str] | None = None) -> dict:
    """mode "control": Fun ControlNet Union on the control video. mode "r2v": no ControlNet; the video goes in as
    the reference video of MiniMaxH3ReferenceToVideo (<Video 1> in the prompt), as the ComfyUI r2v template does."""
    g = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": MODELS["unet"], "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": MODELS["clip"], "type": "minimax", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": MODELS["vae"]}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": MODELS["audio_vae"]}},
        "5": {"class_type": "ModelPatchLoader", "inputs": {"name": MODELS["patch"]}},
        "6": {"class_type": "LoadVideo", "inputs": {"file": control_name}},
        "7": {"class_type": "GetVideoComponents", "inputs": {"video": ["6", 0]}},
        "10": {"class_type": "MiniMaxH3FunControlNetApply",
               "inputs": {"model": ["9", 0] if turbo else ["1", 0], "model_patch": ["5", 0], "vae": ["3", 0],
                          "control_video": ["7", 0], "strength": strength, "start_percent": 0.0,
                          "end_percent": 1.0}},
        "11": {"class_type": "MiniMaxH3ReferenceToVideo",
               "inputs": {"clip": ["2", 0], "vae": ["3", 0], "audio_vae": ["4", 0], "prompt": prompt,
                          "width": width, "height": height, "length": length, "ref_image_size": ref_image_size}},
        "12": {"class_type": "BasicScheduler", "inputs": {"model": ["10", 0], "scheduler": scheduler, "steps": steps,
                                                          "denoise": 1.0}},
        "13": {"class_type": "BasicGuider", "inputs": {"model": ["10", 0], "conditioning": ["11", 0]}},
        "14": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": SAMPLER}},
        "15": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "16": {"class_type": "SamplerCustomAdvanced",
               "inputs": {"noise": ["15", 0], "guider": ["13", 0], "sampler": ["14", 0], "sigmas": ["12", 0],
                          "latent_image": ["11", 1]}},
        "17": {"class_type": "VAEDecode", "inputs": {"samples": ["16", 0], "vae": ["3", 0]}},
        "18": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["16", 0], "vae": ["4", 0]}},
        "19": {"class_type": "CreateVideo", "inputs": {"images": ["17", 0], "audio": ["18", 0], "fps": float(FPS)}},
        "20": {"class_type": "SaveVideo", "inputs": {"video": ["19", 0], "filename_prefix": prefix,
                                                     "format": "auto", "codec": "auto"}},
    }
    if turbo:
        g["9"] = {"class_type": "LoraLoaderModelOnly",
                  "inputs": {"model": ["1", 0], "lora_name": MODELS["turbo_lora"], "strength_model": 1.0}}
    images = [ref_image_name] if isinstance(ref_image_name, str) else list(ref_image_name or [])
    for i, name in enumerate(images[:9]):  # <Picture 1..> in the order given
        g[f"2{i + 1}"] = {"class_type": "LoadImage", "inputs": {"image": name}}
        g["11"]["inputs"][f"ref_images.ref_image_{i}"] = [f"2{i + 1}", 0]
    if mode == "r2v":
        base = ["9", 0] if turbo else ["1", 0]
        del g["5"], g["10"]
        g["12"]["inputs"]["model"] = base
        g["13"]["inputs"]["model"] = base
        g["11"]["inputs"]["ref_videos.ref_video_0"] = ["7", 0]
        for i, name in enumerate((extra_videos or [])[:2]):  # <Video 2>, <Video 3>
            g[f"3{i}"] = {"class_type": "LoadVideo", "inputs": {"file": name}}
            g[f"4{i}"] = {"class_type": "GetVideoComponents", "inputs": {"video": [f"3{i}", 0]}}
            g["11"]["inputs"][f"ref_videos.ref_video_{i + 1}"] = [f"4{i}", 0]
    return g


# ---------------------------------------------------------------- server

class Server:
    def __init__(self, url: str):
        self.url = url.rstrip("/")
        self.client_id = str(uuid.uuid4())

    def _req(self, path: str, data: bytes | None = None, headers: dict | None = None, timeout: float = 120):
        # the RunPod proxy (Cloudflare) answers 403 to the default Python user agent
        req = urllib.request.Request(self.url + path, data=data, headers={"User-Agent": "Mozilla/5.0", **(headers or {})})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read()

    def get_json(self, path: str):
        return json.loads(self._req(path))

    def upload(self, path: Path, subfolder: str = "vrecon", name: str | None = None) -> str:
        """Into ComfyUI's input folder; returns the name LoadVideo / LoadImage take."""
        boundary = uuid.uuid4().hex
        parts = []
        # (bug until 2026-11-16 12:00: this loop reused "name", so every upload was called "overwrite" and replaced
        # the previous one: a run's reference image was replaced by its reference video)
        filename = name or path.name
        for field, value in (("subfolder", subfolder), ("type", "input"), ("overwrite", "true")):
            parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{field}"\r\n\r\n{value}\r\n'.encode())
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="{filename}"\r\n'
                     f'Content-Type: application/octet-stream\r\n\r\n'.encode() + path.read_bytes() + b"\r\n")
        parts.append(f"--{boundary}--\r\n".encode())
        res = json.loads(self._req("/upload/image", b"".join(parts),
                                   {"Content-Type": f"multipart/form-data; boundary={boundary}"}, timeout=600))
        return f"{res['subfolder']}/{res['name']}" if res.get("subfolder") else res["name"]

    def queue(self, graph: dict) -> str:
        body = json.dumps({"prompt": graph, "client_id": self.client_id}).encode()
        try:
            res = json.loads(self._req("/prompt", body, {"Content-Type": "application/json"}))
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"ComfyUI refused the graph: {e.read().decode(errors='replace')[:3000]}") from e
        return res["prompt_id"]

    def wait(self, prompt_id: str, poll: float = 5.0, timeout: float = 7200) -> dict:
        t0 = time.time()
        while time.time() - t0 < timeout:
            try:
                h = self.get_json(f"/history/{prompt_id}")
            except (OSError, ValueError) as e:  # the RunPod proxy drops idle connections now and then
                print(f"  (poll failed: {type(e).__name__}; retrying)", flush=True)
                time.sleep(poll)
                continue
            if prompt_id in h:
                entry = h[prompt_id]
                st = entry.get("status", {})
                if st.get("completed") or st.get("status_str") in ("success", "error"):
                    return entry
            time.sleep(poll)
        raise TimeoutError(f"{prompt_id} not finished after {timeout} s")

    def download(self, info: dict, dest: Path) -> None:
        q = urllib.parse.urlencode({"filename": info["filename"], "subfolder": info.get("subfolder", ""),
                                    "type": info.get("type", "output")})
        dest.write_bytes(self._req(f"/view?{q}", timeout=600))


SAVE_NODE = "20"


def output_files(entry: dict) -> list[dict]:
    """Files written by the SaveVideo node only: LoadVideo also lists its input as an output (type "input"), and
    taking that one returned the control video itself (2026-11-16)."""
    out = entry.get("outputs", {}).get(SAVE_NODE, {})
    files = [f for key in ("videos", "images", "gifs") for f in out.get(key, [])]
    return [f for f in files if f.get("type", "output") == "output"
            and str(f.get("filename", "")).endswith((".mp4", ".webm", ".mov", ".mkv"))]


# ---------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://127.0.0.1:8188")
    ap.add_argument("--jobs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--group", default="main", help="job group ('' for all)")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--seeds", nargs="*", type=int)
    ap.add_argument("--steps", type=int, default=20, help="sampling steps (template default 20; turbo uses 4)")
    ap.add_argument("--turbo", action="store_true", help="turbo LoRA, 4 steps")
    ap.add_argument("--ref-image", type=Path, help="reference image for every run (e.g. the source's first frame)")
    ap.add_argument("--clip", default=MODELS["clip"], help="text encoder file (e.g. qwen3vl_32b_minimax_h3_int8_convrot"
                    ".safetensors: 8-bit; the default nvfp4 is emulated on Ada GPUs)")
    ap.add_argument("--strength", type=float, help="control strength for every run (default: the job's "
                    "control_context_scale)")
    ap.add_argument("--unet", default=MODELS["unet"], help="H3 transformer file (ref2va or fl2va)")
    ap.add_argument("--size", help="generate at WxH instead of the control's size (control is resized by ComfyUI)")
    ap.add_argument("--ref-image-size", default="match", choices=["match", "max"])
    ap.add_argument("--mode", default="control", choices=["control", "r2v"])
    ap.add_argument("--scheduler", default="simple", help="simple (template) or beta (the r2v template's advice)")
    ap.add_argument("--prompt-file", type=Path, help="use this prompt instead of the job's (e.g. one with <Video 1>)")
    ap.add_argument("--tag", default="", help="suffix for run folders, to keep setting variants apart")
    ap.add_argument("--profile", type=Path, help="JSON settings profile (gpu/h3_control/comfy/profiles/*.json): keys "
                    "are these option names (steps, mode, clip, ...) plus prompt_prefix / prompt_suffix; command-line "
                    "options given explicitly still win")
    ap.add_argument("--dry-run", action="store_true")
    pre, _ = ap.parse_known_args()
    profile = {}
    if pre.profile:
        profile = json.loads(pre.profile.read_text(encoding="utf-8"))
        known = {x.dest for x in ap._actions}
        ap.set_defaults(**{k.replace("-", "_"): (Path(v) if k.replace("-", "_") in ("ref_image", "prompt_file") and v
                                                else v)
                           for k, v in profile.items() if k.replace("-", "_") in known})
    a = ap.parse_args()
    steps = 4 if a.turbo else a.steps
    MODELS["clip"] = a.clip
    MODELS["unet"] = a.unet

    runs = plan_runs(a.jobs, a.group or None, a.only, a.seeds)
    for r in runs:
        r["run_id"] += a.tag
    problems = []
    for r in runs:
        if not r["control_path"].is_file():
            problems.append(f"{r['run_id']}: control missing: {r['control_path']}")
            continue
        r["control_info"] = probe(r["control_path"])
        problems += [f"{r['run_id']}: {p}" for p in check(r["control_info"])]
    for r in runs:
        r["fingerprint"] = fingerprint(r, a, steps, profile)

    def reusable(r) -> bool:
        """A stored video counts only when everything that went into it is the same (control and reference files
        by content, prompt, models, settings, seed); otherwise it is generated again."""
        d = a.out / r["run_id"]
        fp = d / "inputs.sha256"
        return (d / "video.mp4").is_file() and fp.is_file() and fp.read_text(encoding="utf-8").strip() == r["fingerprint"]

    todo = [r for r in runs if not reusable(r)]
    print(f"{len(runs)} runs planned, {len(runs) - len(todo)} done, {len(todo)} to run; steps {steps}"
          f"{' (turbo)' if a.turbo else ''}")
    for p in problems:
        print("problem:", p)
    if a.dry_run or problems:
        for r in todo:
            print("  ", r["run_id"], r.get("control_info"))
        return 1 if problems else 0

    srv = Server(a.server)
    stats = srv.get_json("/system_stats")
    ref_name = srv.upload(a.ref_image) if a.ref_image else None
    for r in todo:
        info, s = r["control_info"], r["settings"]
        run_dir = a.out / r["run_id"]
        run_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        # unique name: every scene's control is called control_<kind>.mp4, and uploads overwrite
        ctrl = srv.upload(r["control_path"], name=f"{r['job']['id']}__{r['control_path'].name}")
        # per-job references (jobs.json "ref_images", "extra_ref_videos"; paths relative to the jobs file)
        base_dir = a.jobs.parent
        job_imgs = [srv.upload((base_dir / x).resolve(), name=f"ref__{Path(x).name}") for x in r["job"].get("ref_images", [])]
        job_vids = [srv.upload((base_dir / x).resolve(), name=f"{r['job']['id']}__v{i + 2}__{Path(x).name}")
                    for i, x in enumerate(r["job"].get("extra_ref_videos", []))]
        images_all = ([ref_name] if ref_name else []) + job_imgs
        gw, gh = (int(x) for x in a.size.split("x")) if a.size else (info["width"], info["height"])
        prompt = a.prompt_file.read_text(encoding="utf-8").strip() if a.prompt_file else r["job"]["prompt"]
        prompt = "\n".join(x for x in (profile.get("prompt_prefix", ""), prompt, profile.get("prompt_suffix", "")) if x)
        graph = build_graph(ctrl, prompt, r["seed"], gw, gh, info["frames"], steps,
                            a.strength if a.strength is not None else float(s.get("control_context_scale", 1.0)), a.turbo, f"vrecon/{r['run_id']}", images_all, mode=a.mode, scheduler=a.scheduler, ref_image_size=a.ref_image_size,
                            extra_videos=job_vids)
        pid = srv.queue(graph)
        print(f"queued {r['run_id']} ({pid})", flush=True)
        entry = srv.wait(pid)
        seconds = time.time() - t0
        files = output_files(entry)
        status = entry.get("status", {})
        meta = {"run_id": r["run_id"], "job": r["job"], "profile": str(a.profile) if a.profile else None,
                "profile_settings": profile, "prompt_sent": prompt, "mode": a.mode, "scheduler": a.scheduler,
                "ref_image_size": a.ref_image_size, "seed": r["seed"], "steps": steps, "turbo": a.turbo,
                "settings": s, "control_info": info, "control": str(r["control_path"]), "ref_image": str(a.ref_image)
                if a.ref_image else None, "models": MODELS, "sampler": SAMPLER, "prompt_id": pid,
                "status": status.get("status_str"), "messages": status.get("messages", [])[-6:],
                "seconds_wall": round(seconds, 1), "server": a.server, "system_stats": stats, "graph": graph,
                "finished_at": dt.datetime.now(dt.timezone.utc).isoformat()}
        if files:
            srv.download(files[0], run_dir / "video.mp4")
            (run_dir / "inputs.sha256").write_text(r["fingerprint"] + "
", encoding="utf-8")
        else:
            meta["error"] = "no video in the outputs"
        (run_dir / "meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"{'done' if files else 'FAILED'} {r['run_id']} in {seconds:.0f} s", flush=True)
        if not files:
            print(json.dumps(meta["messages"], ensure_ascii=False)[:2000])
            return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
