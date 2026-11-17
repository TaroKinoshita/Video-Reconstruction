"""H5 editability test: apply the three scripted edits (vrecon/edits.py) to each .blend and record success.

    python scripts/edit_test.py OUT.json A.blend [B.blend ...]

A vrecon project is tested through its built .blend (`vrecon build`), so every condition is edited the same
way. Writes {blend: {edits: [...], edit_success_share}} and prints one line per file.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    from vrecon.scene.edit_blend import run_all

    out = Path(sys.argv[1])
    rows = {}
    for b in sys.argv[2:]:
        rep = run_all(b)
        rows[b] = rep
        print(b, rep["edit_success_share"], [(r["edit"], r["problems"][:2]) for r in rep["edits"] if not r["success"]],
              flush=True)
    out.write_text(json.dumps(rows, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
