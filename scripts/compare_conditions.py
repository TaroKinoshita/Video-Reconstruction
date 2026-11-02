"""Compare conditions (e.g. B vs C) per PLAN.md 11.3 and write a markdown report.

    python scripts/compare_conditions.py RESULTS_DIR [--split held_out|dev|all] [--out report.md]

Layout: RESULTS_DIR/<condition>/<clip>/<attempt>/score.json  (+ optional run.json from a baseline run:
"valid", "agent.result.total_cost_usd", "agent.result.duration_ms"). Invalid runs are excluded and counted.

H1a (primary): per clip, median over attempts of the primary metric; count clips where the candidate beats
the reference condition and report the median difference. H1b: median cost and time per condition.
Clip split and primary metric come from evaluation/thresholds.yaml; using held-out clips while the file is
not frozen prints a warning (PLAN.md 11.6).
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_runs(results: Path) -> dict:
    """{condition: {clip: [ {metrics..., cost, seconds, valid} ]}}"""
    out: dict = {}
    for score in sorted(results.glob("*/*/*/score.json")):
        attempt = score.parent
        clip, cond = attempt.parent.name, attempt.parent.parent.name
        rep = json.loads(score.read_text())
        row = dict(rep.get("summary", {}))
        row["error"] = rep.get("error")
        run = attempt / "run.json"
        if run.is_file():
            r = json.loads(run.read_text())
            res = (r.get("agent") or {}).get("result") or {}
            row.update(valid=r.get("valid", True), cost=res.get("total_cost_usd"),
                       seconds=(res.get("duration_ms") or 0) / 1000 or None)
        else:
            row.setdefault("valid", True)
        out.setdefault(cond, {}).setdefault(clip, []).append(row)
    return out


def median(vals):
    vals = [v for v in vals if v is not None]
    return statistics.median(vals) if vals else None


def compare(runs: dict, ref: str, cand: str, metric: str, lower_better: bool, clips: list[str] | None) -> dict:
    rows, wins, diffs = [], 0, []
    names = sorted(set(runs.get(ref, {})) | set(runs.get(cand, {})))
    if clips:
        names = [c for c in names if c in clips]
    for clip in names:
        a = median(r.get(metric) for r in runs.get(ref, {}).get(clip, []) if r.get("valid"))
        b = median(r.get(metric) for r in runs.get(cand, {}).get(clip, []) if r.get("valid"))
        better = None
        if a is not None and b is not None:
            better = b < a if lower_better else b > a
            wins += better
            diffs.append(b - a)
        rows.append((clip, a, b, better))
    return {"rows": rows, "wins": wins, "compared": len(diffs), "median_diff": median(diffs)}


def fmt(v):
    return "-" if v is None else f"{v:.4g}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("results", type=Path)
    ap.add_argument("--ref", default="B")
    ap.add_argument("--cand", default="C")
    ap.add_argument("--split", default="all", choices=["all", "dev", "held_out"])
    ap.add_argument("--thresholds", type=Path, default=ROOT / "evaluation/thresholds.yaml")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    th = json.loads(args.thresholds.read_text())
    if args.split == "held_out" and th.get("status") != "frozen":
        print("WARNING: scoring held-out clips while thresholds.yaml is not frozen (PLAN.md 11.6)")
    clips = None if args.split == "all" else th["clips"][args.split]
    runs = load_runs(args.results)
    lines = [f"# {args.cand} vs {args.ref} ({args.split})", ""]
    excluded = {c: sum(not r.get("valid") for rs in v.values() for r in rs) for c, v in runs.items()}
    lines.append("Runs per condition: " + ", ".join(f"{c}: {sum(len(x) for x in v.values())} "
                                                    f"(invalid {excluded[c]})" for c, v in sorted(runs.items())))
    lines.append("")
    primary = th["primary_metric"]
    for metric, lower in th["lower_is_better"].items():
        res = compare(runs, args.ref, args.cand, metric, lower, clips)
        tag = " (primary, H1a)" if metric == primary else ""
        lines += [f"## {metric}{tag}", "",
                  f"{args.cand} better on {res['wins']}/{res['compared']} clips; median difference "
                  f"({args.cand} - {args.ref}) {fmt(res['median_diff'])} ({'lower' if lower else 'higher'} is better)", "",
                  f"| clip | {args.ref} | {args.cand} | {args.cand} better |", "|---|---|---|---|"]
        lines += [f"| {c} | {fmt(a)} | {fmt(b)} | {'-' if w is None else ('yes' if w else 'no')} |" for c, a, b, w in res["rows"]]
        lines.append("")
    lines += ["## Cost and time (H1b)", "", "| condition | median cost (USD) | median seconds |", "|---|---|---|"]
    for c, v in sorted(runs.items()):
        rs = [r for x in v.values() for r in x if r.get("valid")]
        lines.append(f"| {c} | {fmt(median(r.get('cost') for r in rs))} | {fmt(median(r.get('seconds') for r in rs))} |")
    text = "\n".join(lines) + "\n"
    if args.out:
        args.out.write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
