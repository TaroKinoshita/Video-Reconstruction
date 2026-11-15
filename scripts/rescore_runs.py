"""WP-1 step 4: re-score every stored run with the current scorer (v3: unmatched objects count 0, relative onset,
novel views). Reads stored scenes only; no LLM, no GPU observer. Writes one score file per run into a NEW folder
(stored runs are never touched), then a summary table.

    python scripts/rescore_runs.py [--runs runs] [--out runs/rescore_v3] [--only GROUP ...] [--no-novel]

Groups and cells (docs/PLAN.md 16.6, RQ1 protocol v2):
    S0  4RC start scene, no finishing           bench_rays/<clip>_base/snapshots/rev_0000.json
    S1  fixed order                             bench_rays/<clip>_base, last snapshot
    S2  LLM-directed (Claude, GPT)               llm_c2, llm_c2b, llm_c2c, llm_c4, gpt/c, last snapshot
    L0  LLM builds in Blender                   B/*, gpt/b, gpt/b_repeat  (workspace/output/scene.blend)
    L1  L0 finished by the fixed order           opt4, opt4_rep (refined.blend)  -- mask_source recorded
    other: llm_c (before the guard fix), llm_c3 (mixed guard), looker, viewer, viewer2, viewer3
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SYN = ROOT / "evaluation" / "synthetic"
CLIP = re.compile(r"syn_0\d")


def _clip_of(path: Path) -> str:
    m = CLIP.search(path.as_posix())
    if not m:
        raise ValueError(f"no clip name in {path}")
    return m.group(0)


def _last_snapshot(run: Path) -> Path:
    snaps = sorted((run / "snapshots").glob("rev_*.json"))
    if not snaps:
        raise FileNotFoundError(f"no snapshots in {run}")
    return snaps[-1]


def _mask_source(run: Path) -> str | None:
    cfg = run / "config.json"
    if cfg.is_file():
        return json.loads(cfg.read_text(encoding="utf-8")).get("mask_source")
    return None


def discover(runs: Path) -> list[dict]:
    """[{group, cell, run, kind: project|blend, path, clip, mask_source}]"""
    items: list[dict] = []

    def add_project(group, cell, run: Path, snap: Path | None = None):
        if not (run / "snapshots").is_dir():
            return
        items.append(dict(group=group, cell=cell, run=run, kind="project", path=snap or _last_snapshot(run),
                          clip=_clip_of(run), mask_source=_mask_source(run)))

    for run in sorted(runs.glob("bench_rays/syn_0*_base")):
        add_project("S0 4RC start", "S0", run, run / "snapshots" / "rev_0000.json")
        add_project("S1 fixed order", "S1", run)
    for g in ("llm_c2", "llm_c2b", "llm_c2c", "llm_c4"):
        for run in sorted((runs / g).glob("syn_0*")):
            add_project(f"S2 Claude {g}", "S2", run)
    for run in sorted((runs / "gpt" / "c").glob("syn_0*")):
        add_project("S2 GPT", "S2", run)
    for g in ("llm_c", "llm_c3", "looker", "viewer", "viewer2", "viewer3"):
        for run in sorted((runs / g).glob("syn_0*")):
            if run.is_dir():
                add_project(f"other {g}", "-", run)
    for group, pattern in (("L0 B Claude", "B/syn_0*"), ("L0 B GPT", "gpt/b/syn_0*"),
                           ("L0 B GPT repeats", "gpt/b_repeat/syn_0*")):
        for run in sorted(runs.glob(pattern)):
            blend = run / "workspace" / "output" / "scene.blend"
            if blend.is_file():
                items.append(dict(group=group, cell="L0", run=run, kind="blend", path=blend, clip=_clip_of(run),
                                  mask_source=None))
    for group, pattern in (("L1 B GPT + tools", "opt4/syn_0*"), ("L1 B GPT repeats + tools", "opt4_rep/syn_0*")):
        for run in sorted(runs.glob(pattern)):
            blend = run / "refined.blend"
            if blend.is_file():
                items.append(dict(group=group, cell="L1", run=run, kind="blend", path=blend, clip=_clip_of(run),
                                  mask_source=_mask_source(run)))
    return items


def _name(item: dict, runs: Path) -> str:
    return item["run"].relative_to(runs).as_posix().replace("/", "__") + ("__rev0" if item["path"].name ==
                                                                          "rev_0000.json" and item["cell"] == "S0"
                                                                          else "")


SUMMARY_KEYS = ("mask_iou_mean", "mask_iou_all_mean", "matched_share", "camera_focal_rel_mean",
                "timing_abs_error_frames_mean", "timing_rel_error_frames_mean", "timing_rel_found_share",
                "novel_iou_10", "novel_iou_30", "novel_iou_back")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, default=ROOT / "runs")
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "rescore_v3")
    ap.add_argument("--only", nargs="*", help="group name prefixes")
    ap.add_argument("--no-novel", action="store_true")
    ap.add_argument("--redo", action="store_true", help="score again even when the output file exists")
    a = ap.parse_args()

    from vrecon.evaluate import score, score_blend
    from vrecon.schema import load_project

    a.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for item in discover(a.runs):
        if a.only and not any(item["group"].startswith(p) for p in a.only):
            continue
        dest = a.out / f"{_name(item, a.runs)}.json"
        if dest.is_file() and not a.redo:
            rep = json.loads(dest.read_text(encoding="utf-8"))
        else:
            t0 = time.time()
            ref = SYN / item["clip"]
            try:
                if item["kind"] == "blend":
                    rep = score_blend(ref, item["path"], novel=not a.no_novel)
                else:
                    rep = score(ref, load_project(item["path"]), novel=not a.no_novel)
            except Exception as e:  # noqa: BLE001  keep going, record the failure
                rep = {"error": f"{type(e).__name__}: {e}", "summary": {}}
            rep["rescore"] = {"group": item["group"], "cell": item["cell"], "clip": item["clip"],
                              "source": item["path"].relative_to(a.runs).as_posix(),
                              "mask_source": item["mask_source"], "seconds": round(time.time() - t0, 1)}
            dest.write_text(json.dumps(rep, indent=1) + "\n", encoding="utf-8")
            print(f"{item['group']:28s} {item['clip']} {rep['summary'].get('mask_iou_all_mean')} "
                  f"{rep.get('error', '')} ({rep['rescore']['seconds']} s)", flush=True)
        rows.append(rep)
    (a.out / "all.json").write_text(json.dumps(
        [{**r["rescore"], **{k: r["summary"].get(k) for k in SUMMARY_KEYS}, "error": r.get("error")} for r in rows],
        indent=1) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} scores to {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
