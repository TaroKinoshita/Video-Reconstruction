"""One entry point: a single shot (video) -> editable Blender scene, the MAVE way (arXiv 2611.22464) with this
project's tools. Stages run in order; a stage whose output exists is skipped (delete it to redo).

  python scripts/mave_pipeline.py VIDEO WORK_DIR [--keyframes 10] [--body-step 2] [--agent sonnet|opus|none]
      [--people mave|own] [--mode empty|system] [--size W H] [--fps F]

WORK_DIR must be an ASCII path (the GPU tools read files with cv2). Stages:
  video      the shot at <= 960 px wide (video.mp4), frame count, fps
  frames     keyframes (evenly spaced) and the 4RC frames (keyframes + two between each pair, <= 30)
  4rc        camera, depth and point maps of the 4RC frames (env 4rc)              [MAVE: DA3]
  people     SAM 3.1 people tracks on the whole shot (env observe)                [MAVE: SAM 3]
  semantic   SAM 3.1 text prompts on the 4RC frames: ground, wall, ceiling, trunk, post, beam, foliage
                                                                                   [MAVE: Mask2Former]
  focal      one focal length fitted to the 4RC point maps
  body       SAM 3D Body per person on every --body-step frame and the keyframes (env body)
  prep       camera per frame, people at the keyframes (scripts/v11_mave_prep.py)
  anim       people as moving bodies (scripts/v11_people_anim.py); --people mave adds MAVE eq. 4-5
             (scripts/v11_people_place.py)
  parts      ground height field, trunks, posts, beams, backdrop (scripts/v11_parts.py): the world's level and the
             system parts (used by --mode system, and for the ground in the evidence)
  evidence   geometric evidence with the estimated up (eq. 3) and the semantic labels (scripts/agent_run.py)
  agent      the agent loop (scripts/agent_run.py run), 15 min target, 30 min limit
  blend      scene.blend at --size (scripts/agent_run.py blend)
Everything comes from the video; nothing from a benchmark's ground truth.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
LOCAL = Path(os.environ.get("LOCALAPPDATA", "")) / "vrecon"
ENV = {k: LOCAL / "envs" / k / "Scripts" / "python.exe" for k in ("4rc", "observe", "body")}
PY = sys.executable
SEM_SETS = ["ground=ground", "floor=ground", "wall=wall", "ceiling=ceiling", "tree trunk=trunk", "pole=post",
            "beam=beam", "foliage=foliage"]


def run(cmd: list, log: Path, cwd: Path | None = None) -> None:
    t = time.time()
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    with open(log, "w", encoding="utf-8") as f:
        p = subprocess.run([str(c) for c in cmd], cwd=cwd, stdout=f, stderr=subprocess.STDOUT, env=env)
    print(f"  {log.stem}: exit {p.returncode}, {time.time() - t:.0f} s", flush=True)
    if p.returncode != 0:
        raise SystemExit(f"stage failed, see {log}")


def fit_focal(rc_path: Path, W: int, H: int, n: int) -> float:
    """The square-pixel focal of scripts/v11_mave_prep.py, at W x H."""
    rc = np.load(rc_path)
    num = den = 0.0
    for i in range(n):
        P = rc[f"pred_{i}_pts"].reshape(rc[f"pred_{i}_pts"].shape[-3:])
        h4, w4 = P.shape[:2]
        K = rc[f"pred_{i}_intrinsic"].reshape(3, 3)
        w2c = np.linalg.inv(rc[f"pred_{i}_extrinsic"].reshape(4, 4))
        X = P.reshape(-1, 3) @ w2c[:3, :3].T + w2c[:3, 3]
        gu, gv = np.meshgrid(np.arange(w4) - K[0, 2], np.arange(h4) - K[1, 2])
        xn, yn = X[:, 0] / X[:, 2], X[:, 1] / X[:, 2]
        num += (xn * gu.ravel()).sum() + (yn * gv.ravel()).sum()
        den += (xn * xn).sum() + (yn * yn).sum()
    return float(num / den / (512 / max(W, H)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=Path)
    ap.add_argument("work", type=Path)
    ap.add_argument("--keyframes", type=int, default=10)
    ap.add_argument("--body-step", type=int, default=2)
    ap.add_argument("--people", choices=["mave", "own"], default="mave")
    ap.add_argument("--mode", choices=["empty", "system"], default="empty")
    ap.add_argument("--agent", choices=["sonnet", "opus", "none"], default="sonnet")
    ap.add_argument("--size", type=int, nargs=2, default=None, help="output picture size (default: the input's)")
    ap.add_argument("--fps", type=float, default=None)
    a = ap.parse_args()
    from vrecon.control.core import encode_mp4, read_video_rgb
    import cv2

    w = a.work.resolve()
    w.mkdir(parents=True, exist_ok=True)
    if not str(w).isascii():
        raise SystemExit("WORK_DIR must be an ASCII path")
    logs = w / "logs"
    logs.mkdir(exist_ok=True)
    state_p = w / "state.json"
    st = json.loads(state_p.read_text(encoding="utf-8")) if state_p.exists() else {}

    def save():
        state_p.write_text(json.dumps(st, indent=1), encoding="utf-8")

    def stage(name: str, out: Path) -> bool:
        if out.exists():
            print(f"[{name}] done ({out.name})")
            return False
        print(f"[{name}]", flush=True)
        return True

    vid = w / "video.mp4"
    if stage("video", vid):
        v = read_video_rgb(str(a.video))
        cap = cv2.VideoCapture(str(a.video)) if str(a.video).isascii() else None
        fps = a.fps or (cap.get(cv2.CAP_PROP_FPS) if cap else 24.0)
        H0, W0 = v.shape[1:3]
        s = min(1.0, 960 / W0)
        W, H = int(round(W0 * s / 2) * 2), int(round(H0 * s / 2) * 2)
        if s < 1:
            v = np.stack([cv2.resize(f, (W, H), interpolation=cv2.INTER_AREA) for f in v])
        encode_mp4(v, vid, fps=fps)
        st.update({"frames": int(len(v)), "size": [W, H], "source_size": [W0, H0], "fps": fps})
        save()
    N, (W, H) = st["frames"], st["size"]

    rc_dir = w / "rc"
    if stage("frames", w / "frames.json"):
        kf = sorted({int(round(x)) for x in np.linspace(0, N - 1, a.keyframes)})
        fr = set(kf)
        for p_, q in zip(kf, kf[1:]):
            fr |= {p_ + (q - p_) // 3, p_ + 2 * (q - p_) // 3}
        fr = sorted(fr)[:30]
        v = read_video_rgb(str(vid))
        (rc_dir / "frames").mkdir(parents=True, exist_ok=True)
        for i in fr:
            cv2.imwrite(str(rc_dir / "frames" / f"{i:04d}.png"), v[i][:, :, ::-1])
        encode_mp4(v[fr], rc_dir / "rc_frames.mp4", fps=24)
        (rc_dir / "frames.json").write_text(json.dumps(fr), encoding="utf-8")
        (w / "frames.json").write_text(json.dumps({"keyframes": kf, "rc": fr}), encoding="utf-8")
    fj = json.loads((w / "frames.json").read_text(encoding="utf-8"))
    kf, fr = fj["keyframes"], fj["rc"]

    if stage("4rc", rc_dir / "4rc.npz"):
        run([ENV["4rc"], "inference.py", "--input", rc_dir / "frames", "--save", rc_dir / "4rc.npz"],
            logs / "4rc.log", cwd=LOCAL / "src" / "4RC")
    people_m = w / "people.masks.npz"
    if stage("people", people_m):
        run([ENV["observe"], ROOT / "gpu/observe/run_sam3.py", vid, people_m, "--movers", "person",
             "--max-objects", "6"], logs / "people.log")
    sem_m = rc_dir / "sem.masks.npz"
    if stage("semantic", sem_m):
        cmd = [ENV["observe"], ROOT / "gpu/observe/run_sam3.py", rc_dir / "rc_frames.mp4", sem_m, "--movers",
               "person", "--keep-raw", "--max-objects", "4"]
        for s_ in SEM_SETS:
            cmd += ["--set", s_]
        run(cmd, logs / "semantic.log")
    if "focal" not in st:
        st["focal"] = fit_focal(rc_dir / "4rc.npz", W, H, len(fr))
        save()
        print(f"[focal] {st['focal']:.1f} px")
    body = w / "body"
    if stage("body", body / "done"):
        frames = sorted(set(range(0, N, a.body_step)) | set(kf))
        run([ENV["body"], ROOT / "gpu/body/run_sam3dbody.py", vid, people_m, body, "--focal", st["focal"],
             "--frames", *frames], logs / "body.log")
        (body / "done").write_text("ok", encoding="utf-8")
    prep = w / "prep.npz"
    if stage("prep", prep):
        run([PY, ROOT / "scripts/v11_mave_prep.py", rc_dir / "4rc.npz", rc_dir / "frames.json", people_m, body, prep,
             "--size", W, H, "--keyframes", *kf, "--frames", N], logs / "prep.log")
    anim = w / "people_own.npz"
    if stage("anim", anim):
        run([PY, ROOT / "scripts/v11_people_anim.py", prep, rc_dir / "4rc.npz", rc_dir / "frames.json", people_m,
             body, vid, anim], logs / "anim.log")
    parts = w / "parts.npz"
    if stage("parts", parts):
        run([PY, ROOT / "scripts/v11_parts.py", prep, rc_dir / "4rc.npz", rc_dir / "frames.json",
             rc_dir / "rc_frames.mp4", parts, "--labels", f"{sem_m}=ground,trunk,beam,foliage,post",
             "--people", people_m, "--keyframes", *kf, "--ground-masks", sem_m], logs / "parts.log")
    people = anim
    if a.people == "mave":
        people = w / "people_mave.npz"
        if stage("place (MAVE eq. 4-5)", people):
            run([PY, ROOT / "scripts/v11_people_place.py", prep, anim, body, people], logs / "place.log")
    ev = w / "evidence.npz"
    if stage("evidence", ev):
        run([PY, ROOT / "scripts/agent_run.py", "evidence", prep, parts, rc_dir / "4rc.npz", rc_dir / "frames.json",
             ev, "--people-masks", rc_dir / "sem.masks.npz", "--up", "multi", "--people", people,
             "--semantic", sem_m, "--no-calibrate"], logs / "evidence.log")
    agent_dir = w / f"agent_{a.mode}_{a.people}"
    if a.agent != "none" and stage("agent", agent_dir / "agent_summary.json"):
        if not (agent_dir / "run.json").exists():
            run([PY, ROOT / "scripts/agent_run.py", "init", agent_dir, ev, people, parts, "--video", vid, "--mode",
                 a.mode, "--fps", st["fps"]], logs / "init.log")
        model = {"sonnet": "claude-sonnet-5-5", "opus": "claude-opus-5-5"}[a.agent]
        run([PY, ROOT / "scripts/agent_run.py", "run", agent_dir, "--model", model], logs / "agent.log")
    out = w / f"scene_{a.mode}_{a.people}.blend"
    if a.agent != "none" and stage("blend", out):
        size = a.size or st["source_size"]
        run([PY, ROOT / "scripts/agent_run.py", "blend", agent_dir, prep, out, "--size", *size, "--fps", st["fps"]],
            logs / "blend.log")
    print(json.dumps({"scene": str(out), "state": st}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
