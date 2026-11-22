"""Pipeline + AI: a vision model adds what the measurements cannot give (set pieces never measured: buildings,
towers, parapets, the unseen sides), on top of a measured scene it may not change.

    python scripts/multishot_llm.py IN.npz SOURCE.mp4 CONTROL.mp4 SHOTS.json OUT.npz [--model openai/gpt-6.1-sol]
                                   

1 ask     one call: per cut a picture (film keyframes left, the current render right), plus a text summary of the
          measured scene (cameras: place and view direction; people: where they stand; set pieces; the floor z = 0).
          The answer is JSON: named set pieces (boxes or planes, metres, the same world) to ADD. Measured cameras,
          people, objects and the floor are locked: the answer cannot touch them.
2 guard   a piece is dropped when it contains a camera or a body joint of any cut (it would hide the shot or cut a
          person), or when it is larger than MAX_SIZE, or under the floor.
3 write   kept pieces go to llm_set (JSON list) in the scene file; scene/multishot.py builds each as its own object
          LLM_<name>, so scripts/multishot_check.py --hide LLM_<name> can measure whether it helps (accept step).
Needs FAL_KEY. Sends film frames to an outside service (fal -> the model's vendor).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))
from multishot_judge import ENDPOINT, data_uri  # noqa: E402

MAX_SIZE = 200.0  # m: larger is a mistake, not a building
KEEP_CLEAR = 0.3  # m around cameras and joints

SYSTEM = ("You complete a 3D blockout of a film scene in Blender. The measured parts (cameras, people, objects, the "
          "floor at z = 0) are locked and correct; you add the surroundings the measurements missed, as simple boxes "
          "and planes, in the same world (metres, Z up). Answer with JSON only.")


def summary(z, info) -> str:
    lines = ["World: metres, Z up, the floor is z = 0.", "Cameras (one per cut; position, view direction, lens):"]
    for k, s in enumerate(info["shots"]):
        c = z[f"s{k}_cam"]
        mid = c[len(c) // 2]
        kind = "close-up" if s.get("points_only_during_cut") else "wide"
        lines.append(f"  cut {k} ({kind}, {s['frames']} frames): at {np.round(mid[:3, 3], 1).tolist()}, looking "
                     f"{np.round(-mid[:3, 2], 2).tolist()}, lens {s['lens_35mm']:.0f} mm, moves "
                     f"{np.linalg.norm(np.ptp(c[:, :3, 3], 0)):.1f} m")
    lines.append("People (pelvis position at the cut's middle frame):")
    for k, s in enumerate(info["shots"]):
        b = z[f"s{k}_body"]
        if len(b) and not s.get("points_only_during_cut"):
            lines.append(f"  cut {k}: {np.round(b[len(b) // 2, 0], 1).tolist()}, height {b[len(b) // 2, :, 2].max():.1f} m")
    lines.append("Set pieces already there (category, corners):")
    for q, cat in zip(z["layout_quads"], z["layout_quad_cat"]):
        lines.append(f"  {cat}: {np.round(q, 1).tolist()}")
    return "\n".join(lines)


QUESTION = """Each picture is one cut: left the film, right the current blockout from the same camera (orange people,
green ground, blue-grey structures, pale sky). Add the set pieces the blockout lacks so each cut's background matches
the film: rooftop edges and parapets, railings, walls, water tanks, nearby and distant buildings, towers. Use the camera
list to put them where each camera sees them in the film. Do not put anything between a camera and its people.
Return JSON: {"pieces": [{"name": "short_snake_case", "type": "box" or "plane", "category": "structure" or "ground",
"center": [x, y, z], "size": [sx, sy, sz], "yaw_deg": 0, "seen_in_cuts": [k, ...], "why": "few words"}]}
A plane is flat in its own x-y (size z ignored); boxes stand on their bottom face. At most 20 pieces.

Measured scene:
"""


def ask(model: str, text: str, images: list) -> dict:
    import os
    import urllib.request

    body = {"model": model, "prompt": text, "system_prompt": SYSTEM, "image_urls": [data_uri(i) for i in images],
            "temperature": 0, "max_tokens": 8000, "reasoning": True}
    req = urllib.request.Request(ENDPOINT, data=json.dumps(body).encode(), method="POST", headers={
        "Authorization": "Key " + os.environ["FAL_KEY"], "Content-Type": "application/json", "X-Fal-Store-IO": "0"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read().decode("utf-8"))


def box_contains(p: dict, pts: np.ndarray, pad: float) -> bool:
    c, s = np.array(p["center"], float), np.array(p["size"], float)
    th = np.radians(p.get("yaw_deg", 0))
    R = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
    local = (pts - c) @ R
    half = s / 2 + pad
    if p.get("type") == "plane":
        half[2] = pad
    return bool((np.abs(local) < half).all(1).any())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", type=Path)
    ap.add_argument("source", type=Path)
    ap.add_argument("control", type=Path)
    ap.add_argument("shots", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--model", default="openai/gpt-6.1-sol")
    a = ap.parse_args()
    from vrecon.control.core import read_video_rgb

    z = dict(np.load(a.npz))
    info = json.loads(str(z["info"]))
    src, ctl = read_video_rgb(a.source), read_video_rgb(a.control)
    shots = json.loads(a.shots.read_text(encoding="utf-8"))["shots"]
    pics = []
    for s0, s1 in shots:
        rows = []
        for f in np.linspace(s0, s1 - 1, 4, dtype=int)[1:-1]:
            if f >= len(ctl):
                continue
            l = cv2.resize(src[f], (480, int(480 * src.shape[1] / src.shape[2])))
            rows.append(np.concatenate([l, cv2.resize(ctl[f], (480, l.shape[0]))], 1))
        if rows:
            pics.append(np.concatenate(rows, 0))
    res = ask(a.model, QUESTION + summary(z, info), pics)
    cost = float(res.get("usage", {}).get("cost") or 0)
    text = res.get("output", "")
    s, e = text.find("{"), text.rfind("}")
    try:
        pieces = json.loads(text[s:e + 1])["pieces"]
    except (ValueError, KeyError):
        print("no JSON in the answer:", text[:500])
        return 1
    guard = [z[f"s{k}_cam"][:, :3, 3] for k in range(len(info["shots"]))]
    guard += [z[f"s{k}_body"].reshape(-1, 3) for k in range(len(info["shots"])) if len(z[f"s{k}_body"])]
    guard = np.concatenate(guard)
    kept, dropped = [], []
    for p in pieces:
        try:
            size = np.array(p["size"], float)
            center = np.array(p["center"], float)
        except (KeyError, ValueError, TypeError):
            dropped.append({"piece": p, "why": "malformed"})
            continue
        if size.shape != (3,) or center.shape != (3,) or (size <= 0).any() and p.get("type") != "plane":
            dropped.append({"name": p.get("name"), "why": "bad size"})
        elif size.max() > MAX_SIZE:
            dropped.append({"name": p.get("name"), "why": "too large"})
        elif center[2] + (size[2] / 2 if p.get("type") != "plane" else 0) < -0.05:
            dropped.append({"name": p.get("name"), "why": "under the floor"})
        elif box_contains(p, guard, KEEP_CLEAR):
            dropped.append({"name": p.get("name"), "why": "contains a camera or a person"})
        else:
            p["name"] = "".join(ch if ch.isalnum() else "_" for ch in str(p.get("name", f"piece{len(kept)}")))[:40]
            kept.append(p)
    z["llm_set"] = json.dumps(kept)
    info["llm"] = {"model": a.model, "usd": cost, "asked": len(pieces), "kept": [p["name"] for p in kept],
                   "dropped": dropped}
    z["info"] = json.dumps(info)
    np.savez_compressed(a.out, **z)
    print(json.dumps(info["llm"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
