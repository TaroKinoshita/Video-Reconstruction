"""PLAN v4 phase 3: check every cut's camera and every person's body against the picture, refit what the picture
supports, keep the rest (and say what stays open).

    python scripts/v4_refine.py SCENE.npz DIAG TAG OUT_DIR

1 per cut: background tracks and the camera check (diagnose_camera.py). A camera-only refit (refit_camera.py) is
  kept when it lowers the check-track error and the cut has enough check pairs (MIN_PAIRS); otherwise the camera
  stays and the cut is listed as "camera undecided" or "camera kept".
2 per person (info["who"] identity): one shared body over all its cuts, fitted to the 2D keypoints
  (fit_person.py, which itself keeps the current body for a cut when the held-out points do not improve).
3 clothes (multishot_dress.py) and the measures (measure_v4.py).
Writes OUT_DIR/refined.npz, refine.json (every decision with its numbers), measure.json/.md.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
FAB_PY = Path(os.environ.get("LOCALAPPDATA", "")) / "vrecon/envs/fab/Scripts/python.exe"
MIN_PAIRS = 20
GAIN = 0.9  # a refit camera is kept when its check error is below GAIN x the current one


def run(cmd, log=None):
    r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, encoding="utf-8", errors="replace")
    if log:
        Path(log).write_text(r.stdout + r.stderr, encoding="utf-8")
    if r.returncode:
        print(r.stderr[-1500:], flush=True)
    return r.returncode == 0


def main() -> int:
    src, diag, tag, out = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], Path(sys.argv[4])
    out.mkdir(parents=True, exist_ok=True)
    z = np.load(src)
    info = json.loads(str(z["info"]))
    cur = out / "work.npz"
    cur.write_bytes(src.read_bytes())
    dec = {"cameras": {}, "people": {}}
    for k in range(len(info["shots"])):
        d0 = out / f"cam{k}_before.json"
        run([PY, ROOT / "scripts/diagnose_camera.py", cur, diag, tag, k, d0])
        b = json.loads(d0.read_text(encoding="utf-8"))
        r = {"before": b["overall_scene"], "picture": b["overall_picture"], "check_pairs": b["check_pairs"]}
        if b["check_pairs"] < MIN_PAIRS or b["overall_scene"] is None:
            r["decision"] = "camera undecided (too little background in the picture)"
        else:
            cand = out / f"cam{k}_refit.npz"
            ok = run([FAB_PY, ROOT / "gpu/observe/refit_camera.py", cur, k, d0.with_suffix(".tracks.npz"), cand],
                     log=out / f"cam{k}_refit.log")
            if ok:
                d1 = out / f"cam{k}_after.json"
                run([PY, ROOT / "scripts/diagnose_camera.py", cand, diag, tag, k, d1])
                a = json.loads(d1.read_text(encoding="utf-8"))
                r["after"] = a["overall_scene"]
                if a["overall_scene"] is not None and a["overall_scene"] < GAIN * b["overall_scene"]:
                    r["decision"] = "camera refit kept"
                    cur.write_bytes(cand.read_bytes())
                else:
                    r["decision"] = "camera kept (refit not better on check tracks)"
            else:
                r["decision"] = "camera kept (refit failed)"
        dec["cameras"][k] = r
        print("camera", k, r, flush=True)
    who = info.get("who", {})
    for ident in sorted({v.get("identity") for v in who.values() if v.get("identity") not in (None, "?")}):
        nxt = out / f"person_{ident}.npz"
        ok = run([FAB_PY, ROOT / "gpu/observe/fit_person.py", cur, diag, tag, ident, nxt], log=out / f"person_{ident}.log")
        if ok:
            cur.write_bytes(nxt.read_bytes())
            dec["people"][ident] = json.loads(str(np.load(nxt)["info"]))["fit_person"][ident]
            print("person", ident, {k: (v["choice"], v["before_heldout"]["all"], v["after_heldout"]["all"])
                                    for k, v in dec["people"][ident]["cuts"].items()}, flush=True)
    refined = out / "refined.npz"
    run([PY, ROOT / "scripts/multishot_dress.py", cur, diag, tag, refined], log=out / "dress.log")
    run([PY, ROOT / "scripts/measure_v4.py", refined, diag, tag, out / "measure.json"])
    (out / "refine.json").write_text(json.dumps(dec, indent=1), encoding="utf-8")
    print((out / "measure.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
