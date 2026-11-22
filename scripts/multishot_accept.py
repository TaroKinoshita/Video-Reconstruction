"""Accept step for pieces a model added (scripts/multishot_llm.py): keep a piece only if the scene checks no worse
with it than without it.

    python scripts/multishot_accept.py IN.npz SCENE.blend DIAG_DIR TAG OUT.npz

Measure: scripts/multishot_check.py's clip score minus the frame-weighted share of person and object pixels the set
hides (depth conflicts). Each piece is hidden in turn (--hide LLM_<name>); a piece whose absence raises the measure by
more than TOL is dropped. The kept pieces are written back (llm_set) with the report in info["llm"]["accept"];
rebuild the .blend afterwards (scripts/multishot_render.py).
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TOL = 0.002


def measure(npz: Path, blend: Path, diag: Path, tag: str, hide: list, out: Path) -> float:
    subprocess.run([sys.executable, str(ROOT / "scripts/multishot_check.py"), str(npz), str(blend), str(diag), tag,
                    str(out), "--every", "3", "--hide", *hide], capture_output=True, text=True, encoding="utf-8",
                   errors="replace", check=True)
    r = json.loads(out.read_text(encoding="utf-8"))
    info = json.loads(str(np.load(npz)["info"]))
    w = np.array([s["frames"] for s in info["shots"]], float)
    blocked = np.array([np.mean([c.get("person_blocked", 0.0), c.get("object_blocked", 0.0)])
                        for c in (r["cuts"].get(str(k), r["cuts"].get(k, {})) for k in range(len(w)))])
    return float(r["score"] - (blocked * w).sum() / w.sum())


def main() -> int:
    src, blend, diag, tag, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4], Path(sys.argv[5])
    z = dict(np.load(src))
    info = json.loads(str(z["info"]))
    pieces = json.loads(str(z["llm_set"]))
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / "c.json"
        base = measure(src, blend, diag, tag, [], tmp)
        none = measure(src, blend, diag, tag, ["LLM_"], tmp)
        rep, keep = [], []
        for p in pieces:
            m = measure(src, blend, diag, tag, [f"LLM_{p['name']}"], tmp)
            ok = m <= base + TOL
            rep.append({"name": p["name"], "without": round(m, 4), "with_all": round(base, 4), "kept": ok})
            print(p["name"], round(m - base, 4), "keep" if ok else "drop", flush=True)
            if ok:
                keep.append(p)
    z["llm_set"] = json.dumps(keep)
    info["llm"]["accept"] = {"measure_all": base, "measure_none": none, "pieces": rep}
    z["info"] = json.dumps(info)
    np.savez_compressed(out, **z)
    print(json.dumps({"all": base, "none": none, "kept": len(keep), "of": len(pieces)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
