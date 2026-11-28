"""One entry point: a single shot (video) -> editable Blender scene, the MAVE way (arXiv 2611.22464) with this
project's tools. Stages run in order; a stage whose output exists is skipped (delete it to redo).

  python scripts/mave_pipeline.py VIDEO WORK_DIR [--keyframes 10] [--body-step 1] [--max-people 4]
      [--agent sonnet|opus|none]
      [--people mave|own] [--mode empty|system] [--size W H] [--fps F]

WORK_DIR must be an ASCII path (the GPU tools read files with cv2). Stages:
  video      the shot at <= 960 px wide (video.mp4), frame count, fps
  frames     keyframes (evenly spaced) and the 4RC frames (keyframes + two between each pair, <= 30)
  4rc / da3  camera, depth and point maps of the 4RC frames (env 4rc, or env da3 with --depth da3, as MAVE)
  people     SAM 3.1 people tracks on the whole shot (env observe)                [MAVE: SAM 3]
  semantic   SAM 3.1 text prompts on the 4RC frames: ground, wall, ceiling, trunk, post, beam, foliage (parts)
  mask2former  Mask2Former (ADE20K) labels for the evidence, as MAVE (env sem; --labels sam uses the SAM ones)
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
ENV = {k: LOCAL / "envs" / k / "Scripts" / "python.exe" for k in ("4rc", "observe", "body", "sem", "da3")}
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


def link_chunks(parts: list, N: int, hw: tuple, overlap: int) -> np.ndarray:
    """Masks of overlapping chunks [(start, end, npz)] as one track set: a chunk's track takes the global id of the
    previous chunk's track it overlaps most (IoU over the shared frames > 0.3), else a new id."""
    out = np.zeros((N,) + hw, np.uint16)
    nxt = 1
    for c, (s0, s1, d) in enumerate(parts):
        m = d["masks"]
        ids = [int(k) for k in json.loads(str(d["id_map"]))]
        mapping = {}
        if c > 0:
            sh = range(s0, min(s1, parts[c - 1][1]))
            scores = []
            for i in ids:
                for g in range(1, nxt):
                    a_ = np.stack([m[t - s0] == i for t in sh])
                    b_ = np.stack([out[t] == g for t in sh])
                    u = (a_ | b_).sum()
                    if u:
                        scores.append(((a_ & b_).sum() / u, i, g))
            for iou, i, g in sorted(scores, reverse=True):
                if iou > 0.3 and i not in mapping and g not in mapping.values():
                    mapping[i] = g
        for i in ids:
            if i not in mapping:
                mapping[i] = nxt
                nxt += 1
        start = s0 if c == 0 else s0 + overlap // 2  # the overlap's second half comes from the new chunk
        for t in range(start, s1):
            fr = m[t - s0]
            out[t] = 0
            for i, g in mapping.items():
                out[t][fr == i] = g
    return out


def merge_tracks(m: np.ndarray, video: np.ndarray | None = None, max_gap: int = 16, max_dist: float = 0.2,
                 min_frames: int = 15, edge_gap: int = 60, max_colour: float = 0.25) -> np.ndarray:
    """Join track pieces of one person, then drop pieces seen on fewer than min_frames frames. Two passes:
    1. continuations: the later piece starts within max_gap frames after the earlier one ends, its first mask
       centre within max_dist x picture width of the earlier one's last (nearest first);
    2. re-entries: the earlier piece left through the picture's edge and a piece starts within edge_gap frames
       (the longest such piece: short fragments are often false detections).
    With the video, pieces must also look alike (hue-saturation histogram distance < max_colour; this rejects
    objects taken for people, not people dressed alike). Ids of kept tracks stay (the earliest id of a chain)."""
    import cv2
    H, W = m.shape[1:]
    m = m.copy()

    def info(i):
        fr = np.where((m == i).any((1, 2)))[0]
        if not len(fr):
            return None
        hist = None
        if video is not None:
            hist = np.zeros((16, 8))
            for t in fr[::max(1, len(fr) // 20)]:
                hsv = cv2.cvtColor(video[t], cv2.COLOR_RGB2HSV)[m[t] == i]
                hist += np.histogram2d(hsv[:, 0], hsv[:, 1], bins=[16, 8], range=[[0, 180], [0, 256]])[0]
            hist /= max(hist.sum(), 1)
        return fr[0], fr[-1], np.argwhere(m[fr[0]] == i).mean(0), np.argwhere(m[fr[-1]] == i).mean(0), len(fr), hist

    def alike(x, y):
        return x[5] is None or 1 - np.sqrt(x[5] * y[5]).sum() < max_colour

    for edge_pass in (False, True):
        while True:
            ids = [i for i in range(1, int(m.max()) + 1) if (m == i).any()]
            inf = {i: info(i) for i in ids}
            best = None
            for a_ in ids:
                ea, last = inf[a_][1], inf[a_][3]
                at_edge = min(last[1], W - last[1]) < 0.05 * W or min(last[0], H - last[0]) < 0.05 * H
                for b_ in ids:
                    sb = inf[b_][0]
                    if a_ == b_ or not alike(inf[a_], inf[b_]):
                        continue
                    if not edge_pass:
                        d = np.linalg.norm(last - inf[b_][2]) / W
                        if ea < sb <= ea + max_gap and d < max_dist:
                            score = d
                        else:
                            continue
                    elif at_edge and ea < sb <= ea + edge_gap:
                        score = -inf[b_][4]
                    else:
                        continue
                    if best is None or score < best[0]:
                        best = (score, a_, b_)
            if best is None:
                break
            m[m == best[2]] = best[1]
    for i in range(1, int(m.max()) + 1):
        if 0 < (m == i).any((1, 2)).sum() < min_frames:
            m[m == i] = 0
    return m


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
    ap.add_argument("--body-step", type=int, default=1)
    ap.add_argument("--people-width", type=int, default=480, help="picture width for people tracking")
    ap.add_argument("--chunk", type=int, default=80, help="frames per people-tracking chunk")
    ap.add_argument("--overlap", type=int, default=8)
    ap.add_argument("--max-people", type=int, default=4, help="tracked at once (8 GB VRAM: 6 overflows at 960 px)")
    ap.add_argument("--people", choices=["mave", "own"], default="mave")
    ap.add_argument("--depth", choices=["4rc", "da3"], default="4rc", help="cameras, depth and points (MAVE: da3)")
    ap.add_argument("--labels", choices=["sam", "m2f"], default="m2f",
                    help="semantic labels for the evidence (MAVE: Mask2Former); the system parts use SAM")
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

    if a.depth == "4rc" and stage("4rc", rc_dir / "4rc.npz"):
        run([ENV["4rc"], "inference.py", "--input", rc_dir / "frames", "--save", rc_dir / "4rc.npz"],
            logs / "4rc.log", cwd=LOCAL / "src" / "4RC")
    people_m = w / "people.masks.npz"
    if stage("people", people_m):
        # tracked in overlapping chunks on a small copy (SAM 3.1's memory grows along the video: on 8 GB a 306-frame
        # run slowed from 10 to 40 s a frame), people linked across chunks by mask overlap, masks scaled back up
        v = read_video_rgb(str(vid))
        ws = min(W, a.people_width)
        hs = int(round(H * ws / W / 2) * 2)
        small = np.stack([cv2.resize(f, (ws, hs), interpolation=cv2.INTER_AREA) for f in v])
        chunks = [(s0, min(s0 + a.chunk, N)) for s0 in range(0, N, a.chunk - a.overlap) if s0 == 0 or s0 + a.overlap < N]
        parts_m = []
        for c, (s0, s1) in enumerate(chunks):
            cv_ = w / f"people_c{c}.mp4"
            cm = w / f"people_c{c}.masks.npz"
            if not cm.exists():
                encode_mp4(small[s0:s1], cv_, fps=st["fps"])
                run([ENV["observe"], ROOT / "gpu/observe/run_sam3.py", cv_, cm, "--movers", "person",
                     "--max-objects", a.max_people], logs / f"people_c{c}.log")
            parts_m.append((s0, s1, np.load(cm)))
        big = link_chunks(parts_m, N, (hs, ws), a.overlap)
        big = merge_tracks(np.stack([cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST) for m in big]), v)
        n_ids = int(big.max())
        np.savez_compressed(people_m, masks=big, id_map=json.dumps({str(i): f"obs_{i - 1}" for i in range(1, n_ids + 1)
                                                                    if (big == i).any()}),
                            meta=json.dumps({"tool": "sam3.1 chunked", "chunks": chunks}))
    if a.depth == "da3" and stage("da3", rc_dir / "da3.npz"):
        run([ENV["da3"], ROOT / "gpu/observe/run_da3.py", rc_dir / "frames", rc_dir / "da3.npz"], logs / "da3.log")
    depth = rc_dir / f"{a.depth}.npz"
    tag = "" if a.depth == "4rc" else f"_{a.depth}"
    m2f = rc_dir / "m2f.masks.npz"
    if a.labels == "m2f" and stage("mask2former", m2f):
        run([ENV["sem"], ROOT / "gpu/observe/run_mask2former.py", rc_dir / "rc_frames.mp4", m2f], logs / "m2f.log")
    sem_m = rc_dir / "sem.masks.npz"
    if stage("semantic", sem_m):
        cmd = [ENV["observe"], ROOT / "gpu/observe/run_sam3.py", rc_dir / "rc_frames.mp4", sem_m, "--movers",
               "person", "--keep-raw", "--max-objects", "4"]
        for s_ in SEM_SETS:
            cmd += ["--set", s_]
        run(cmd, logs / "semantic.log")
    if "focal" not in st:
        st["focal"] = fit_focal(depth, W, H, len(fr))
        save()
        print(f"[focal] {st['focal']:.1f} px")
    body = w / "body"
    if stage("body", body / "done"):
        frames = sorted(set(range(0, N, a.body_step)) | set(kf))
        d = np.load(people_m)
        pm = d["masks"]
        labels = [v for k, v in json.loads(str(d["id_map"])).items() if (pm == int(k)).any((1, 2)).sum() >= 0.35 * N]
        run([ENV["body"], ROOT / "gpu/body/run_sam3dbody.py", vid, people_m, body, "--focal", st["focal"],
             "--frames", *frames, "--labels", *labels], logs / "body.log")
        (body / "done").write_text("ok", encoding="utf-8")
    prep = w / f"prep{tag}.npz"
    if stage("prep", prep):
        run([PY, ROOT / "scripts/v11_mave_prep.py", depth, rc_dir / "frames.json", people_m, body, prep,
             "--size", W, H, "--keyframes", *kf, "--frames", N], logs / "prep.log")
    anim = w / f"people_own{tag}.npz"
    if stage("anim", anim):
        run([PY, ROOT / "scripts/v11_people_anim.py", prep, depth, rc_dir / "frames.json", people_m,
             body, vid, anim], logs / "anim.log")
    parts = w / f"parts{tag}.npz"
    if stage("parts", parts):
        run([PY, ROOT / "scripts/v11_parts.py", prep, depth, rc_dir / "frames.json",
             rc_dir / "rc_frames.mp4", parts, "--labels", f"{sem_m}=ground,trunk,beam,foliage,post",
             "--people", people_m, "--keyframes", *kf, "--ground-masks", sem_m], logs / "parts.log")
    people = anim
    if a.people == "mave":
        people = w / f"people_mave{tag}.npz"
        if stage("place (MAVE eq. 4-5)", people):
            run([PY, ROOT / "scripts/v11_people_place.py", prep, anim, body, people], logs / "place.log")
    ev = w / f"evidence{tag}_{a.labels}.npz"
    if stage("evidence", ev):
        run([PY, ROOT / "scripts/agent_run.py", "evidence", prep, parts, depth, rc_dir / "frames.json",
             ev, "--people-masks", rc_dir / "sem.masks.npz", "--up", "multi", "--people", people,
             "--semantic", m2f if a.labels == "m2f" else sem_m, "--no-calibrate"], logs / f"evidence{tag}_{a.labels}.log")
    variant = f"{a.mode}_{a.people}{tag}_{a.labels}"
    agent_dir = w / f"agent_{variant}"
    if a.agent != "none" and stage("agent", agent_dir / "agent_summary.json"):
        if not (agent_dir / "run.json").exists():
            run([PY, ROOT / "scripts/agent_run.py", "init", agent_dir, ev, people, parts, "--video", vid, "--mode",
                 a.mode, "--fps", st["fps"]], logs / f"init_{variant}.log")
        model = {"sonnet": "claude-sonnet-5-5", "opus": "claude-opus-5-5"}[a.agent]
        run([PY, ROOT / "scripts/agent_run.py", "run", agent_dir, "--model", model], logs / f"agent_{variant}.log")
    out = w / f"scene_{variant}.blend"
    if a.agent != "none" and stage("blend", out):
        size = a.size or st["source_size"]
        run([PY, ROOT / "scripts/agent_run.py", "blend", agent_dir, prep, out, "--size", *size, "--fps", st["fps"]],
            logs / f"blend_{variant}.log")
    print(json.dumps({"scene": str(out), "state": st}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
