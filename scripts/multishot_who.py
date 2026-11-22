"""Who is each cut's person: man, woman or child, so the body fit uses the matching body model.

    python scripts/multishot_who.py IN.npz DIAG_DIR TAG OUT.npz [--model openai/gpt-6.1-sol] [--fake]

Per cut with a person mask: a crop around the largest person mask on two frames (quarter, three quarters). All
crops go in ONE picture, labelled by cut, to an image-reading model (fal openrouter/router/vision, as
multishot_judge.py). It answers per cut: man | woman | child | unknown, a confidence, and an identity letter (the
same letter for the same person in different cuts). Answers are unified per identity by vote, so one person keeps
one body model over all cuts. Low confidence (< MIN_CONF) becomes "unknown" (neutral body).
Writes info["who"] = {cut: {"type", "confidence", "identity"}} into OUT.npz. Needs FAL_KEY; sends film frames to an
outside service; one call (~0.02 USD). --fake writes "unknown" everywhere without a call.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from multishot_judge import ask, parse  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
MIN_CONF = 0.6
TILE = 256
TYPES = ("man", "woman", "child", "unknown")
QUESTION = """The picture is a grid of crops from a film, two crops per cut, each labelled "cut K".
Each crop is centred on the main person of that cut (it may be a close-up of a hand, a back, or blurred).
For every cut, say who the main person is. Return JSON only:
{"cuts": {"K": {"type": "man" | "woman" | "child" | "unknown", "confidence": 0.0-1.0,
                "identity": "A" | "B" | ...  (the same letter for the same person seen in different cuts)}}}
Use "unknown" when the crop does not show enough of the person to tell."""


def crops(diag: Path, tag: str, k: int) -> list[np.ndarray]:
    from vrecon.control.core import read_video_rgb

    mp, vp = diag / f"{tag}_shot{k:02d}.masks.npz", diag / f"{tag}_shot{k:02d}.mp4"
    if not (mp.exists() and vp.exists()):
        return []
    masks = np.load(mp)["masks"]
    film = read_video_rgb(vp)
    T = min(len(masks), len(film))
    out = []
    for t in sorted({T // 4, 3 * T // 4}):
        m = masks[t] > 0
        if not m.any():
            continue
        img = film[t]
        ys, xs = np.nonzero(m)
        sy, sx = img.shape[0] / m.shape[0], img.shape[1] / m.shape[1]
        y0, y1, x0, x1 = ys.min() * sy, (ys.max() + 1) * sy, xs.min() * sx, (xs.max() + 1) * sx
        c, h = np.array([(x0 + x1) / 2, (y0 + y1) / 2]), max(y1 - y0, x1 - x0) * 1.15
        a = np.clip((c - h / 2).astype(int), 0, None)
        b = np.minimum((c + h / 2).astype(int), [img.shape[1], img.shape[0]])
        crop = img[a[1]:b[1], a[0]:b[0]]
        s = TILE / max(crop.shape[:2])
        crop = cv2.resize(crop, (max(1, int(crop.shape[1] * s)), max(1, int(crop.shape[0] * s))))
        tile = np.zeros((TILE, TILE, 3), np.uint8)
        tile[:crop.shape[0], :crop.shape[1]] = crop
        cv2.putText(tile, f"cut {k}", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        out.append(tile)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("diag", type=Path)
    ap.add_argument("tag")
    ap.add_argument("out", type=Path)
    ap.add_argument("--model", default="openai/gpt-6.1-sol")
    ap.add_argument("--fake", action="store_true")
    a = ap.parse_args()
    z = dict(np.load(a.src))
    info = json.loads(str(z["info"]))
    cuts = [k for k in range(len(info["shots"])) if len(z.get(f"s{k}_body", []))]
    who = {str(k): {"type": "unknown", "confidence": 0.0, "identity": "?"} for k in cuts}
    if not a.fake:
        rows = [np.concatenate(c + [np.zeros_like(c[0])] * (2 - len(c)), 1) for k in cuts if (c := crops(a.diag, a.tag, k))]
        if rows:
            pairs = [np.concatenate(rows[i:i + 2] + [np.zeros_like(rows[0])] * (2 - len(rows[i:i + 2])), 1)
                     for i in range(0, len(rows), 2)]
            sheet = np.concatenate(pairs, 0)
            cv2.imencode(".jpg", sheet[..., ::-1])[1].tofile(str(a.out.with_suffix(".who.jpg")))
            import multishot_judge as J

            J.QUESTION, J.SYSTEM = QUESTION, "You look at film crops and answer with JSON only."
            r = ask(a.model, sheet)
            ans = parse(r.get("output", "")) or {}
            print("cost", r.get("usage", {}).get("cost"), flush=True)
            for k, v in (ans.get("cuts") or {}).items():
                if str(k) in who and isinstance(v, dict):
                    t = v.get("type") if v.get("type") in TYPES else "unknown"
                    conf = float(v.get("confidence") or 0)
                    who[str(k)] = {"type": t if conf >= MIN_CONF else "unknown", "confidence": conf,
                                   "identity": str(v.get("identity", "?"))}
    # one body model per identity: the most common confident answer
    for ident in {w["identity"] for w in who.values() if w["identity"] != "?"}:
        votes = Counter(w["type"] for w in who.values() if w["identity"] == ident and w["type"] != "unknown")
        if votes:
            for w in who.values():
                if w["identity"] == ident:
                    w["type"] = votes.most_common(1)[0][0]
    info["who"] = who
    z["info"] = json.dumps(info)
    np.savez_compressed(a.out, **z)
    print(json.dumps(who, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
