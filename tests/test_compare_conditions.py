"""scripts/compare_conditions.py: per-clip verdicts from a bootstrap over runs (docs/REVIEW_INSTRUCTIONS.md F-16)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("compare_conditions", ROOT / "scripts" / "compare_conditions.py")
cc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cc)


def _runs(ref: dict, cand: dict, metric: str = "m") -> dict:
    return {c: {clip: [{metric: v, "valid": True} for v in vals] for clip, vals in d.items()}
            for c, d in (("A", ref), ("B", cand))}


def test_verdicts():
    runs = _runs({"clear": [0.70, 0.71, 0.72], "noisy": [0.60, 0.90, 0.75], "one": [0.5], "worse": [0.9, 0.91, 0.92]},
                 {"clear": [0.80, 0.81, 0.82], "noisy": [0.70, 0.95, 0.80], "one": [0.9], "worse": [0.8, 0.81, 0.79]})
    res = cc.compare(runs, "A", "B", "m", lower_better=False, clips=None)
    v = {r[0]: r[-1] for r in res["rows"]}
    assert v == {"clear": "better", "noisy": "not separated", "one": "one run", "worse": "worse"}
    assert res["counts"] == {"better": 1, "worse": 1, "not separated": 1, "one run": 1}
    low = cc.compare(runs, "A", "B", "m", lower_better=True, clips=["clear"])
    assert [r[-1] for r in low["rows"]] == ["worse"]


def test_bootstrap_is_reproducible():
    a, b = [0.7, 0.75, 0.8], [0.72, 0.78, 0.85]
    assert cc.paired_bootstrap(a, b) == cc.paired_bootstrap(a, b)
    d, lo, hi = cc.paired_bootstrap(a, b)
    assert lo <= d <= hi
