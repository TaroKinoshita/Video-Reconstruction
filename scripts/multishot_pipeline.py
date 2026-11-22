"""Multi-cut pipeline: a film clip -> one Blender world with every cut's camera work -> semantic control video.

    python scripts/multishot_pipeline.py VIDEO OUT_DIR TAG [--redo STEP]

Every step skips when its output exists (rerun after any stop); --redo STEP reruns STEP and everything after it.
Steps (all local; GPU ones run one at a time):
  cuts      picture area, cuts, per-cut videos and frames                       (multishot_cuts.py)
  sam       people masks per cut (SAM 3.1; long cuts at half size, 8 GB VRAM)  (gpu/observe/run_sam3.py)
  objects   things from one fixed word list, tracked per cut (SAM 3.1)          (gpu/observe/find_objects_sam3.py)
  gvhmr     body per cut where a person is seen whole enough (WSL)              (gpu/observe/run_gvhmr.py)
  fab       body per frame where GVHMR cannot follow (close-ups), FAB-L        (gpu/observe/run_fab.py)
  4rc       camera path and points per cut                                     (4RC inference.py)
  joint     one 4RC run over keyframes of the cuts with a body                  (4RC inference.py)
  prep      per-cut cameras, bodies, points in metres                           (multishot_prep.py)
  join      the cuts with keyframes in one world                                (multishot_join.py)
  closeups  close-ups placed against the matched person                         (multishot_closeups.py)
  layout    flat set pieces and boxes                                           (multishot_layout.py)
  people    figures for people GVHMR did not follow; carve what blocks them     (multishot_people.py)
  place_objects  held things follow a wrist, loose ones stand on the floor      (multishot_objects.py)
  clothing  loose clothing below the waist from the masks                       (multishot_clothing.py)
  gate      every per-frame placed thing held to what the world does (no teleports, still props, no flashes)
                                                                                 (multishot_gate.py)
  who       man / woman / child per cut by an image-reading model (fal; --fake without FAL_KEY) (multishot_who.py)
  smpl      SMPL body surfaces fitted to every cut's joints (fab env)          (gpu/observe/fit_smpl.py)
  dress     clothes on the SMPL bodies (cloth offset, long-coat skirt) by mask fit (multishot_dress.py)
  render    finished .blend + semantic control (24 fps, /32 sizes, 17n+5)      (multishot_render.py)
  check     category labels vs the SAM masks, sense checks -> check_TAG.json    (multishot_check.py)
Needs: the observe env (SAM 3.1), GVHMR in WSL (~/vrecon), the 4rc env and src under %LOCALAPPDATA%/vrecon.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
LOCAL = Path(os.environ.get("LOCALAPPDATA", ""))
OBS = LOCAL / "vrecon/envs/observe/Scripts/python.exe"
RC_PY = LOCAL / "vrecon/envs/4rc/Scripts/python.exe"
RC_SRC = LOCAL / "vrecon/src/4RC"
FAB_PY = LOCAL / "vrecon/envs/fab/Scripts/python.exe"
FAB_SRC = LOCAL / "vrecon/src/FAB"
STEPS = ["cuts", "sam", "objects", "gvhmr", "fab", "4rc", "joint", "prep", "join", "closeups", "layout", "people",
         "place_objects", "clothing", "gate", "who", "smpl", "dress", "render", "check"]
HALF_ABOVE = 120  # frames: longer cuts are masked at half size
JOINT_N = 30  # 4RC's frame budget


def run(cmd, cwd=None, log: Path | None = None) -> int:
    print("+", " ".join(str(c) for c in cmd), flush=True)
    if log:
        with open(log, "w", encoding="utf-8") as fh:
            return subprocess.run([str(c) for c in cmd], cwd=cwd, stdout=fh, stderr=subprocess.STDOUT).returncode
    return subprocess.run([str(c) for c in cmd], cwd=cwd).returncode


def wsl_path(p: Path) -> str:
    """C:\a\b -> /mnt/c/a/b (wslpath through wsl.exe loses non-ASCII paths)."""
    p = Path(p).resolve()
    return "/mnt/" + p.drive[0].lower() + p.as_posix()[2:]


def person_seen_whole(masks: np.ndarray) -> bool:
    """a person GVHMR can follow: some mask in most frames, not so close that it fills the picture."""
    on = masks > 0
    present = on.any((1, 2)).mean()
    area = np.median(on.mean((1, 2))[on.any((1, 2))]) if on.any() else 0
    return present > 0.5 and 0.005 < area < 0.3


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("out", type=Path)
    ap.add_argument("tag")
    ap.add_argument("--redo", choices=STEPS)
    a = ap.parse_args()
    d, tag = a.out.resolve(), a.tag
    d.mkdir(parents=True, exist_ok=True)
    redo = set(STEPS[STEPS.index(a.redo):]) if a.redo else set()
    S = lambda name: ROOT / "scripts" / name  # noqa: E731

    def need(step: str, out: Path) -> bool:
        return step in redo or not out.exists()

    if need("cuts", d / f"shots_{tag}.json"):
        if run([PY, S("multishot_cuts.py"), a.video, d, tag]):
            return 1
    shots = json.loads((d / f"shots_{tag}.json").read_text(encoding="utf-8"))["shots"]
    K = range(len(shots))
    for k in K:
        v, m = d / f"{tag}_shot{k:02d}.mp4", d / f"{tag}_shot{k:02d}.masks.npz"
        if need("sam", m):
            src = v
            if shots[k][1] - shots[k][0] > HALF_ABOVE:
                src = d / f"{tag}_shot{k:02d}_half.mp4"
                run(["ffmpeg", "-v", "error", "-y", "-i", v, "-vf", "scale=iw/2:-2", "-c:v", "libx264", "-crf", "16", src])
            extra = ["--max-objects", "2", "--new-det-thresh", "0.5", "--quick"] if src != v else []
            run([OBS, ROOT / "gpu/observe/run_sam3.py", src, m, "--movers", "person", "--prompt-frame", "auto",
                 "--min-present", "0.1", "--max-area", "0.9", *extra], log=d / f"{tag}_shot{k:02d}.sam.log")
    if need("objects", d / f"{tag}_objects.json"):
        run([OBS, ROOT / "gpu/observe/find_objects_sam3.py", d, tag], log=d / "objects.log")
    wsl_root, wsl_d = wsl_path(ROOT), wsl_path(d)
    bodies = []
    for k in K:
        m = d / f"{tag}_shot{k:02d}.masks.npz"
        g = d / f"gvhmr_{tag}{k:02d}" / "gvhmr_skeleton.npz"
        if not m.exists() or not person_seen_whole(np.load(m)["masks"]):
            continue
        if need("gvhmr", g):
            run(["wsl", "-e", "bash", "-lc", f"cd ~/vrecon/GVHMR && ~/vrecon/bin/micromamba run -r ~/vrecon/mamba -n gvhmr "
                 f"python '{wsl_root}/gpu/observe/run_gvhmr.py' '{wsl_d}/{tag}_shot{k:02d}.mp4' '{wsl_d}/gvhmr_{tag}{k:02d}'"])
        if g.exists():
            bodies.append(k)
    for k in K:  # close-ups: a body from each image (FAB), used for the pose shape (multishot_closeups.py)
        m = d / f"{tag}_shot{k:02d}.masks.npz"
        fo = d / f"fab_{tag}_shot{k:02d}.npz"
        if FAB_PY.exists() and m.exists() and k not in bodies and need("fab", fo):
            fr = d / f"rc_{tag}{k:02d}" / "frames"
            if fr.exists():
                # the lens is not known before 4RC: a 39 mm-equivalent guess (4RC's lens on film cuts so far); only the
                # pose's shape is used downstream, which a wrong focal length changes little
                focal = 39.0 / 36.0 * int(json.loads((d / f"shots_{tag}.json").read_text(encoding="utf-8")).get("width", 1280))
                run([FAB_PY, ROOT / "gpu/observe/run_fab.py", fr, m, focal, fo], cwd=FAB_SRC, log=d / f"fab_{tag}{k:02d}.log")
    for k in K:
        o = d / f"rc_{tag}{k:02d}" / "4rc.npz"
        if need("4rc", o):
            run([RC_PY, "inference.py", "--input", o.parent / "frames", "--save", o], cwd=RC_SRC, log=o.parent / "4rc.log")
    jd = d / f"rc_{tag}_joint"
    if need("joint", jd / "4rc.npz") and len(bodies) >= 2:
        (jd / "frames").mkdir(parents=True, exist_ok=True)
        for f in (jd / "frames").glob("*.png"):
            f.unlink()
        lens = {k: shots[k][1] - shots[k][0] for k in bodies}
        w = {k: np.sqrt(n) for k, n in lens.items()}
        alloc = {k: max(3, int(round(JOINT_N * w[k] / sum(w.values())))) for k in bodies}
        while sum(alloc.values()) > JOINT_N:
            alloc[max(alloc, key=alloc.get)] -= 1
        rows, i = [], 0
        import shutil

        for k in bodies:
            for f in np.linspace(0, lens[k] - 1, alloc[k], dtype=int):
                shutil.copy(d / f"rc_{tag}{k:02d}" / "frames" / f"{f + 1:05d}.png", jd / "frames" / f"{i:05d}.png")
                rows.append([k, int(f)])
                i += 1
        (jd / "frames.json").write_text(json.dumps(rows), encoding="utf-8")
        run([RC_PY, "inference.py", "--input", jd / "frames", "--save", jd / "4rc.npz"], cwd=RC_SRC, log=jd / "4rc.log")
    n = {s: d / f"ms_{tag}_{s}.npz" for s in ("prep", "join", "closeups", "layout", "people", "objects", "clothing", "gate", "who", "smpl", "dress")}
    if need("prep", n["prep"]):
        run([PY, S("multishot_prep.py"), d, f"{tag}_shot", n["prep"]], log=d / "prep.log")
    last = n["prep"]
    if (jd / "4rc.npz").exists():
        if need("join", n["join"]):
            run([PY, S("multishot_join.py"), n["prep"], jd, n["join"], tag], log=d / "join.log")
        if need("closeups", n["closeups"]):
            run([PY, S("multishot_closeups.py"), n["join"], d, f"{tag}_shot", n["closeups"]], log=d / "closeups.log")
        if need("layout", n["layout"]):
            run([PY, S("multishot_layout.py"), n["closeups"], n["layout"]], log=d / "layout.log")
        if need("people", n["people"]):
            run([PY, S("multishot_people.py"), n["layout"], d, f"{tag}_shot", n["people"]], log=d / "people.log")
        if need("place_objects", n["objects"]):
            run([PY, S("multishot_objects.py"), n["people"], d, tag, n["objects"]], log=d / "place_objects.log")
        if need("clothing", n["clothing"]):
            run([PY, S("multishot_clothing.py"), n["objects"], d, tag, n["clothing"]], log=d / "clothing.log")
        if need("gate", n["gate"]):
            run([PY, S("multishot_gate.py"), n["clothing"], n["gate"]], log=d / "gate.log")
        last = n["gate"]
        if FAB_PY.exists():
            if need("who", n["who"]):
                run([PY, ROOT / "scripts/multishot_who.py", n["gate"], d, tag, n["who"]] + ([] if os.environ.get("FAL_KEY") else ["--fake"]),
                    log=d / "who.log")
            fit_in = n["who"] if n["who"].exists() else n["gate"]
            if need("smpl", n["smpl"]):
                run([FAB_PY, ROOT / "gpu/observe/fit_smpl.py", fit_in, n["smpl"]], cwd=FAB_SRC, log=d / "smpl.log")
            if n["smpl"].exists():
                last = n["smpl"]
            if n["smpl"].exists() and need("dress", n["dress"]):
                run([PY, ROOT / "scripts/multishot_dress.py", n["smpl"], d, tag, n["dress"]], log=d / "dress.log")
            if n["dress"].exists():
                last = n["dress"]
    ctl = d / f"{tag}_control_sem.mp4"
    if need("render", ctl):
        run([PY, S("multishot_render.py"), last, d / f"{tag}_final.blend", ctl], log=d / "render.log")
    chk = d / f"check_{tag}.json"
    if need("check", chk) and (d / f"{tag}_final.blend").exists():
        run([PY, S("multishot_check.py"), last, d / f"{tag}_final.blend", d, tag, chk], log=d / "check.log")
    print(json.dumps({"blend": str(d / f"{tag}_final.blend"), "control": str(ctl), "cuts": len(shots),
                      "bodies": bodies}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
