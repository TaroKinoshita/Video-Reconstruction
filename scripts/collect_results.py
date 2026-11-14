"""WP-0: collect the scores that are already stored under runs/ (no re-scoring, no LLM, no GPU).

Prints one markdown table per group: clip, mask IoU (matched objects, scorer v1 as stored), focal error,
timing error, and the file the value came from. Groups whose scores are not stored are listed with the
reason. Usage: python scripts/collect_results.py [RUNS_DIR] > table.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

CLIPS = [f"syn_0{i}" for i in range(1, 7)]

# group name -> (folder pattern relative to runs/, kind)
GROUPS = [
    ("B Claude (blend-ai)", "B/syn_01_blendai_r2/score.json", "score"),
    ("B GPT run 1", "gpt/b/{clip}/score.json", "score"),
    ("B GPT repeats", "gpt/b_repeat/*/score.json", "score_glob"),
    ("C Claude llm_c (before 3201ab7)", "llm_c/{clip}/llm/result.json", "llm"),
    ("C Claude llm_c2", "llm_c2/{clip}/llm/result.json", "llm"),
    ("C Claude llm_c2b", "llm_c2b/{clip}/llm/result.json", "llm"),
    ("C Claude llm_c2c", "llm_c2c/{clip}/llm/result.json", "llm"),
    ("C Claude llm_c3 (use-budget, mixed guard)", "llm_c3/{clip}/llm/result.json", "llm"),
    ("C Claude llm_c4 (use-budget)", "llm_c4/{clip}/llm/result.json", "llm"),
    ("C GPT", "gpt/c/{clip}/llm/result.json", "llm"),
    ("Looker", "looker/{clip}/llm/result.json", "llm"),
    ("Viewer", "viewer/{clip}/llm/result.json", "llm"),
    ("Viewer2", "viewer2/{clip}/llm/result.json", "llm"),
    ("Viewer3", "viewer3/{clip}/llm/result.json", "llm"),
]


def _load(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _summary(d):
    if not isinstance(d, dict):
        return None
    if "summary" in d and isinstance(d["summary"], dict):
        return d["summary"]
    if "score" in d and isinstance(d["score"], dict):
        return _summary(d["score"]) or d["score"]
    if "mask_iou_mean" in d:
        return d
    return None


def _fmt(v, pct=False):
    if v is None:
        return "-"
    return f"{100 * v:.1f}%" if pct else f"{v:.3f}"


def _row(label, path: Path, root: Path):
    s = _summary(_load(path)) if path.exists() else None
    rel = path.relative_to(root).as_posix()
    if s is None:
        return f"| {label} | missing | - | - | `{rel}` |", None
    iou = s.get("mask_iou_mean")
    timing = s.get("timing_abs_error_frames_mean")
    return (f"| {label} | {_fmt(iou)} | {_fmt(s.get('camera_focal_rel_mean'), pct=True)} | "
            f"{'-' if timing is None else f'{timing:.1f}'} | `{rel}` |"), iou


def main(root: Path) -> None:
    print(f"Stored scores under `{root.name}/` (as written at run time; scorer v1 unless the file says otherwise)\n")
    for name, pattern, kind in GROUPS:
        rows, vals = [], []
        if kind == "score_glob":
            paths = sorted(root.glob(pattern))
            items = [(p.parent.name, p) for p in paths]
        elif "{clip}" in pattern:
            items = [(c, root / pattern.format(clip=c)) for c in CLIPS]
        else:
            p = root / pattern
            items = [(p.parent.name, p)]
        for label, p in items:
            line, v = _row(label, p, root)
            rows.append(line)
            if v is not None:
                vals.append(v)
        mean = f"{sum(vals) / len(vals):.3f} (n={len(vals)})" if vals else "-"
        print(f"### {name}\n\nmean IoU: {mean}\n")
        print("| clip | IoU (matched) | focal err | timing (f) | source |\n|---|---|---|---|---|")
        print("\n".join(rows) + "\n")
    bench = _load(root / "bench_rays/results.json")
    if bench:
        print("### Fixed order, 4RC start, focal fixed (`bench_rays/results.json`, base column)\n")
        vals = [r[1][0] for r in bench]
        print("| clip | IoU |\n|---|---|")
        for r in bench:
            print(f"| {r[0]} | {r[1][0]:.3f} |")
        print(f"\nmean IoU: {sum(vals) / len(vals):.3f} (n={len(vals)})\n")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "runs")
