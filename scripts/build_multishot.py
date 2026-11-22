"""Build the multi-cut .blend (src/vrecon/scene/multishot.py) and render one frame per check point.

    python scripts/build_multishot.py PREP.npz OUT.blend [RENDER_DIR]
"""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from vrecon.scene.multishot import build, render_check  # noqa: E402

r = build(Path(sys.argv[1]), Path(sys.argv[2]).resolve())
if len(sys.argv) > 3:
    r["renders"] = render_check(Path(sys.argv[3]).resolve())
print(json.dumps(r))
