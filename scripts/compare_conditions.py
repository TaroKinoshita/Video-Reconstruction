"""Compare two conditions per clip with a paired bootstrap over runs (PLAN.md 11.3, docs/REVIEW_INSTRUCTIONS.md F-16).

    python scripts/compare_conditions.py --rows runs/rescore_v3/all.json --ref S1 --cand S2 [--key cell]
    python scripts/compare_conditions.py RESULTS_DIR --ref B --cand C            (older layout, below)
        [--split dev|held_out|all] [--metric NAME ...] [--out report.md]

Inputs:
  --rows FILE   a JSON list of rows, one per run: {<key>: condition, "clip": ..., <metric>: value, ...}
                (scripts/rescore_runs.py writes one as all.json; --key picks the condition field, default "cell")
  RESULTS_DIR   RESULTS_DIR/<condition>/<clip>/<attempt>/score.json (+ optional run.json from a baseline run:
                "valid", "agent.result.total_cost_usd", "agent.result.duration_ms"); invalid runs excluded, counted.

Per clip: the difference of the run means (cand - ref) and a 95% bootstrap interval (runs resampled with
replacement on each side, BOOT resamples, fixed seed). "better" / "worse" only when the interval excludes zero
in that direction; otherwise "not separated". A clip with one run on either side has no interval and is reported
"one run" (never better or worse). Then the counts and the median difference over clips (PLAN 11.3).
Metrics and their direction come from evaluation/thresholds.yaml; using held-out clips while the file is not frozen
prints a warning (PLAN.md 11.6).
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BOOT = 2000
SEED = 20261115


def load_runs(results: Path) -> dict:
    """{condition: {clip: [ {metrics..., cost, seconds, valid} ]}} from RESULTS_DIR/<cond>/<clip>/<attempt>/score.json"""
    out: dict = {}
    for score in sorted(results.glob("*/*/*/score.json")):
        attempt = score.parent
        clip, cond = attempt.parent.name, attempt.parent.parent.name
        rep = json.loads(score.read_text(encoding="utf-8"))
        row = dict(rep.get("summary", {}))
        row["error"] = rep.get("error")
        run = attempt / "run.json"
        if run.is_file():
            r = json.loads(run.read_text(encoding="utf-8"))
            res = (r.get("agent") or {}).get("result") or {}
            row.update(valid=r.get("valid", True), cost=res.get("total_cost_usd"),
                       seconds=(res.get("duration_ms") or 0) / 1000 or None)
        else:
            row.setdefault("valid", True)
        out.setdefault(cond, {}).setdefault(clip, []).append(row)
    return out


def load_rows(path: Path, key: str) -> dict:
    out: dict = {}
    for r in json.loads(path.read_text(encoding="utf-8")):
        if r.get(key) in (None, "-"):
            continue
        r = dict(r)
        r.setdefault("valid", not r.get("error"))
        out.setdefault(r[key], {}).setdefault(r["clip"], []).append(r)
    return out


def median(vals):
    vals = [v for v in vals if v is not None]
    return statistics.median(vals) if vals else None


def paired_bootstrap(a: list[float], b: list[float], n: int = BOOT, seed: int = SEED) -> tuple[float, float, float]:
    """mean(b) - mean(a) and its 95% interval, resampling each side's runs with replacement."""
    rng = random.Random(seed)
    diff = statistics.fmean(b) - statistics.fmean(a)
    sims = sorted(statistics.fmean(rng.choices(b, k=len(b))) - statistics.fmean(rng.choices(a, k=len(a)))
                  for _ in range(n))
    return diff, sims[int(0.025 * n)], sims[min(int(0.975 * n), n - 1)]


def compare(runs: dict, ref: str, cand: str, metric: str, lower_better: bool, clips: list[str] | None) -> dict:
    rows, diffs = [], []
    counts = {"better": 0, "worse": 0, "not separated": 0, "one run": 0}
    names = sorted(set(runs.get(ref, {})) & set(runs.get(cand, {})))
    if clips:
        names = [c for c in names if c in clips]
    for clip in names:
        a = [r[metric] for r in runs[ref][clip] if r.get("valid") and r.get(metric) is not None]
        b = [r[metric] for r in runs[cand][clip] if r.get("valid") and r.get(metric) is not None]
        if not a or not b:
            continue
        diff, lo, hi = paired_bootstrap(a, b)
        if len(a) < 2 or len(b) < 2:
            verdict = "one run"
        elif lo > 0 or hi < 0:  # the interval excludes zero
            verdict = "better" if (lo > 0) != lower_better else "worse"
        else:
            verdict = "not separated"
        counts[verdict] += 1
        diffs.append(diff)
        rows.append((clip, len(a), statistics.fmean(a), len(b), statistics.fmean(b), diff, lo, hi, verdict))
    return {"rows": rows, "counts": counts, "median_diff": median(diffs), "compared": len(diffs)}


def fmt(v):
    return "-" if v is None else f"{v:.4g}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("results", type=Path, nargs="?")
    ap.add_argument("--rows", type=Path)
    ap.add_argument("--key", default="cell")
    ap.add_argument("--ref", default="B")
    ap.add_argument("--cand", default="C")
    ap.add_argument("--metric", nargs="*", help="default: every metric in thresholds.yaml lower_is_better")
    ap.add_argument("--split", default="all", choices=["all", "dev", "held_out"])
    ap.add_argument("--thresholds", type=Path, default=ROOT / "evaluation/thresholds.yaml")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    th = json.loads(args.thresholds.read_text(encoding="utf-8"))
    if args.split == "held_out" and th.get("status") != "frozen":
        print("WARNING: scoring held-out clips while thresholds.yaml is not frozen (PLAN.md 11.6)")
    clips = None if args.split == "all" else th["clips"][args.split]
    if args.rows:
        runs = load_rows(args.rows, args.key)
    elif args.results:
        runs = load_runs(args.results)
    else:
        ap.error("give RESULTS_DIR or --rows")
    lines = [f"# {args.cand} vs {args.ref} ({args.split})", "",
             f"Paired bootstrap over runs, {BOOT} resamples, seed {SEED}, 95% interval of mean({args.cand}) - "
             f"mean({args.ref}). thresholds.yaml status: {th.get('status')}.", ""]
    excluded = {c: sum(not r.get("valid") for rs in v.values() for r in rs) for c, v in runs.items()}
    lines.append("Runs per condition: " + ", ".join(f"{c}: {sum(len(x) for x in v.values())} "
                                                    f"(invalid {excluded[c]})" for c, v in sorted(runs.items())))
    lines.append("")
    metrics = th["lower_is_better"]
    for metric in args.metric or list(metrics):
        lower = metrics.get(metric, False)
        res = compare(runs, args.ref, args.cand, metric, lower, clips)
        c = res["counts"]
        tag = " (primary)" if metric == th.get("primary_metric") else ""
        lines += [f"## {metric}{tag}", "",
                  f"{res['compared']} clips: better {c['better']}, worse {c['worse']}, not separated "
                  f"{c['not separated']}, one run {c['one run']}; median difference {fmt(res['median_diff'])} "
                  f"({'lower' if lower else 'higher'} is better)", "",
                  f"| clip | {args.ref} runs | {args.ref} mean | {args.cand} runs | {args.cand} mean | diff | 95% interval | verdict |",
                  "|---|---|---|---|---|---|---|---|"]
        lines += [f"| {clip} | {na} | {fmt(ma)} | {nb} | {fmt(mb)} | {fmt(d)} | [{fmt(lo)}, {fmt(hi)}] | {v} |"
                  for clip, na, ma, nb, mb, d, lo, hi, v in res["rows"]]
        lines.append("")
    if any("cost" in r for v in runs.values() for x in v.values() for r in x):
        lines += ["## Cost and time", "", "| condition | median cost (USD) | median seconds |", "|---|---|---|"]
        for cnd, v in sorted(runs.items()):
            rs = [r for x in v.values() for r in x if r.get("valid")]
            lines.append(f"| {cnd} | {fmt(median(r.get('cost') for r in rs))} | "
                         f"{fmt(median(r.get('seconds') for r in rs))} |")
    text = "\n".join(lines) + "\n"
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
