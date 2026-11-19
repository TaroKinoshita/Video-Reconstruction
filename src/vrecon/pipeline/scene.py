"""Recipe runner for the observation -> scene half of a clip (WP-5 front half). Clip-independent.

    vrecon pipeline-scene RECIPE.json [--from STAGE] [--to STAGE] [--dry-run] [--check]

The recipe's "run" list holds the stages in order. Each stage: {"stage": name, "argv": [...], "outputs": [...],
"stored": {...}}. In argv and outputs, "{work}" is the recipe's work folder (a NEW folder; stored results are never
overwritten), "{in:NAME}" an entry of "inputs" (observer files, recorded LLM answers), "{val:NAME}" a value of
"values" (each with its basis). A stage that calls an LLM or a paid service is not re-run: its recorded answer
is an input, so a rebuild is deterministic and free.

Every stage appends a line to <work>/pipeline_log.jsonl: stage, argv, sha256 of the files named in argv and of
its outputs, seconds, exit status. --check compares each stage's outputs with the stored results named in the
stage's "stored" map (JSON files compared as parsed data; other files by sha256) and writes <work>/check.json.
Each stored file may have an "allow" entry: {"tolerance": metres-or-units, "paths": [fnmatch JSON paths], "why"}.
A missing output, or a difference not covered by its allowance, fails the run (exit 3).
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def _sha(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _same_json(a, b, tol: float = 1e-6) -> bool:
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same_json(a[k], b[k], tol) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_same_json(x, y, tol) for x, y in zip(a, b))
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        return math.isclose(a, b, rel_tol=tol, abs_tol=tol)
    return a == b


def _expand(s: str, r: dict, work: Path) -> str:
    s = s.replace("{work}", str(work))
    for k, v in r.get("inputs", {}).items():
        s = s.replace("{in:" + k + "}", str(ROOT / v["path"]))
    for k, v in r.get("values", {}).items():
        if "value" not in v:
            continue
        s = s.replace("{val:" + k + "}", str(v["value"]) if not isinstance(v["value"], (dict, list)) else json.dumps(v["value"]))
    return s


def _max_diff(a, b) -> float | None:
    """Largest numeric difference between two parsed JSON trees; None when their structure or text differs."""
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            return None
        ds = [_max_diff(a[k], b[k]) for k in a]
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return None
        ds = [_max_diff(x, y) for x, y in zip(a, b)]
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        return abs(float(a) - float(b))
    else:
        return 0.0 if a == b else None
    return None if any(d is None for d in ds) else max(ds, default=0.0)


def _load_json(p: Path):
    """A project file is read through the schema (an older stored version is migrated), anything else as JSON."""
    raw = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "schema_version" in raw and "reconstruction" in raw:
        from vrecon.schema import load_project

        return json.loads(load_project(p).model_dump_json())
    return raw


def _diffs(a, b, path: str, out: list, tol: float) -> None:
    """Every place where two parsed JSON trees differ: (path, kind); numbers within tol are equal. List items get
    their index in the path, so allowance patterns use [*] (fnmatch)."""
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append((f"{path}.{k}", "only in one"))
            else:
                _diffs(a[k], b[k], f"{path}.{k}", out, tol)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append((path, f"length {len(a)} vs {len(b)}"))
        for i, (x, y) in enumerate(zip(a, b)):
            _diffs(x, y, f"{path}[{i}]", out, tol)
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        if abs(float(a) - float(b)) > tol:
            out.append((path, f"{a} vs {b}"))
    elif a != b:
        out.append((path, "text differs"))


def compare(got: Path, want: Path, tol: float = 1e-6, allowed: list[str] | None = None) -> dict:
    """{"result": same | allowed | different | missing, "unexplained": [...], "allowed": n}.
    tol: largest numeric difference counted as equal (recipe: per stored file, with its basis).
    allowed: fnmatch patterns of JSON paths whose differences are known and accepted (recipe, with the reason)."""
    import fnmatch

    if not got.exists():
        return {"result": "missing", "unexplained": [str(got)], "allowed": 0}
    if got.suffix == ".json" and want.suffix == ".json":
        a, b = (_strip_paths(_load_json(p)) for p in (got, want))
        out: list = []
        _diffs(a, b, "", out, tol)
        ok = [d for d in out if any(fnmatch.fnmatchcase(d[0], pat) for pat in (allowed or []))]
        bad = [f"{p}: {k}" for p, k in out if (p, k) not in ok]
        res = "same" if not out else ("allowed" if not bad else "different")
        return {"result": res, "unexplained": bad[:20], "unexplained_count": len(bad), "allowed": len(ok)}
    same = _sha(got) == _sha(want)
    return {"result": "same" if same else "different", "unexplained": [] if same else ["file content"], "allowed": 0}


def _strip_paths(x):
    """File paths differ between a rebuild and the stored run by design: compare the file names only."""
    if isinstance(x, dict):
        return {k: (Path(v).name if isinstance(v, str) and ("/" in v or "\\" in v) and Path(v).suffix else _strip_paths(v))
                for k, v in x.items()}
    if isinstance(x, list):
        return [_strip_paths(v) for v in x]
    return x


def run(recipe_path: Path, start: str | None = None, stop: str | None = None, dry: bool = False,
        check: bool = False) -> int:
    r = json.loads(recipe_path.read_text(encoding="utf-8"))
    work = ROOT / r["work"]
    names = [s["stage"] for s in r["run"]]
    todo = r["run"][names.index(start) if start else 0: names.index(stop) + 1 if stop else None]
    work.mkdir(parents=True, exist_ok=True)
    log = work / "pipeline_log.jsonl"
    results = {}
    for st in todo:
        argv = [_expand(a, r, work) for a in st["argv"]]
        argv = [sys.executable if a == "{python}" else a for a in argv]
        outs = [Path(_expand(o, r, work)) for o in st.get("outputs", [])]
        print(f"[{st['stage']}]", " ".join(argv), flush=True)
        if dry:
            continue
        t0 = time.time()
        code = subprocess.run(argv, cwd=ROOT, encoding="utf-8").returncode
        ins = {a: _sha(Path(a)) for a in argv[1:] if Path(a).is_file()}
        rec = {"stage": st["stage"], "argv": argv, "inputs_sha256": ins,
               "outputs_sha256": {str(o): _sha(o) for o in outs}, "seconds": round(time.time() - t0, 1), "exit": code}
        with open(log, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
        if code != 0:
            print(f"stage {st['stage']} failed (exit {code})")
            return code
        if check:
            for got, want in st.get("stored", {}).items():
                g, w = Path(_expand(got, r, work)), ROOT / want
                key = st["stage"] + ": " + Path(got).name
                rule = st.get("allow", {}).get(got, {})
                results[key] = compare(g, w, float(rule.get("tolerance", 1e-6)), rule.get("paths"))
                print(f"  check {key}: {results[key]['result']}"
                      + (f" ({results[key].get('unexplained_count')} unexplained, e.g. {results[key]['unexplained'][:2]})"
                         if results[key]["result"] in ("different", "missing") else ""), flush=True)
            (work / "check.json").write_text(json.dumps(results, indent=1) + "\n", encoding="utf-8")
    if check and any(v["result"] in ("different", "missing") for v in results.values()):
        print("check FAILED: outputs missing or differing beyond the recipe's allowances")
        return 3
    return 0
