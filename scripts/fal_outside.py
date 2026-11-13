"""Phase E: the picture outside the frame, drawn by an image model on fal (docs/VENUE_PLAN.md, "フェーズEの画面外の絵").

    python scripts/fal_outside.py SOURCE.png OUT_DIR [--quality medium] [--n 2] [--styles natural,structure]
        [--scale-w 2.0] [--scale-h 1.6] [--long 1280] [--model openai/gpt-image-2.5/sunburst/edit] [--fake]

Same camera, same perspective: the source is placed at the centre of a wider canvas (its position and scale are
recorded); the model fills the rest (mask: white = draw, black = keep). Afterwards the source pixels are pasted
back (the mask is a guide, not a guarantee) and the change inside the source area is measured. Writes per
candidate: <style>_<i>.png (pasted back), <style>_<i>.raw.png, plus canvas.png, mask.png, outside.json (prompt,
model, settings, placement, inside-area change, cost estimate). Needs FAL_KEY in the environment; the key is
never printed. Requests ask fal not to store inputs/outputs (X-Fal-Store-IO: 0, sync_mode) - not a guarantee.
--fake writes a grey fill instead of calling fal (plumbing check only; not for quality comparisons).
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

PROMPTS = {
    "natural": (
        "Extend this photograph outward beyond its edges: the same camera position, the same lens and perspective, "
        "the same lighting and time of day. The centre of the canvas is the original photo; keep it exactly. Fill "
        "the grey area with what plausibly surrounds it at this venue (continue the ground, court surroundings, "
        "stands, walls, fences, sky) so lines that leave the photo continue straight across the border. Do not add "
        "a roof, a second court or new buildings unless the photo already shows one. No fisheye, no panorama "
        "warp, no bird's-eye view. Keep it plain: no readable text or logos, no detailed faces; people in the "
        "stands as simple small figures."),
    "structure": (
        "Extend this photograph outward beyond its edges with the same camera, lens and perspective, showing the "
        "venue's structure clearly. The centre is the original photo; keep it exactly. Fill the grey area with "
        "the plausible rest of the venue drawn as clean, flat-shaded surfaces with distinct colours per kind: "
        "ground, court, seating tiers (each row visible as a step), aisles/stairs, walls and fences, sky. Lines "
        "leaving the photo must continue straight across the border. No roof, second court or new buildings "
        "unless the photo shows one. No fisheye, panorama or bird's-eye view, no text, no faces."),
}
PRICE = {"low": 0.004, "medium": 0.009, "high": 0.036, "xhigh": 0.064, "max": 0.144}  # fal table, 1024x768, USD


def _read(p):
    return cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)


def _write(p, img):
    ok, buf = cv2.imencode(".png", img)
    buf.tofile(str(p))


def _data_uri(img):
    return "data:image/png;base64," + base64.b64encode(cv2.imencode(".png", img)[1].tobytes()).decode()


def canvas_for(src, scale_w, scale_h, long_side):
    """(canvas, mask, placement): the source resized so the canvas' long side is long_side (multiples of 16)."""
    h, w = src.shape[:2]
    cw, ch = w * scale_w, h * scale_h
    k = long_side / max(cw, ch)
    CW, CH = int(round(cw * k / 16) * 16), int(round(ch * k / 16) * 16)
    sw, sh = int(round(w * k)), int(round(h * k))
    x0, y0 = (CW - sw) // 2, (CH - sh) // 2
    canvas = np.full((CH, CW, 3), 128, np.uint8)
    canvas[y0:y0 + sh, x0:x0 + sw] = cv2.resize(src, (sw, sh), interpolation=cv2.INTER_AREA)
    mask = np.full((CH, CW, 3), 255, np.uint8)
    mask[y0:y0 + sh, x0:x0 + sw] = 0
    return canvas, mask, {"canvas_wh": [CW, CH], "source_box_xywh": [x0, y0, sw, sh], "source_scale": k,
                          "source_wh": [w, h]}


def call_fal(model, prompt, canvas, mask, quality, timeout=600):
    body = {"prompt": prompt, "image_urls": [_data_uri(canvas)], "mask_url": _data_uri(mask), "quality": quality,
            "image_size": {"width": canvas.shape[1], "height": canvas.shape[0]}, "num_images": 1,
            "output_format": "png", "sync_mode": True}
    req = urllib.request.Request(f"https://fal.run/{model}", data=json.dumps(body).encode(), method="POST", headers={
        "Authorization": "Key " + os.environ["FAL_KEY"], "Content-Type": "application/json", "X-Fal-Store-IO": "0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.loads(r.read().decode("utf-8"))
    url = out["images"][0]["url"]
    if url.startswith("data:"):
        raw = base64.b64decode(url.split(",", 1)[1])
    else:
        with urllib.request.urlopen(url, timeout=120) as r:
            raw = r.read()
    return cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)


def align(gen, canvas, place):
    """The model redraws the whole canvas and may shift or rescale the source part: estimate a similarity
    (scale, rotation, shift) from SIFT matches inside the source box and warp the generated image onto the
    canvas. Returns (aligned, info)."""
    x0, y0, sw, sh = place["source_box_xywh"]
    sift = cv2.SIFT_create(4000)
    g1, g2 = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY), cv2.cvtColor(gen, cv2.COLOR_BGR2GRAY)
    m = np.zeros(g1.shape, np.uint8)
    m[y0:y0 + sh, x0:x0 + sw] = 255
    k1, d1 = sift.detectAndCompute(g1, m)
    k2, d2 = sift.detectAndCompute(g2, None)
    if d1 is None or d2 is None:
        return gen, {"aligned": False}
    good = [a for a, b in cv2.BFMatcher().knnMatch(d2, d1, k=2) if a.distance < 0.75 * b.distance]
    if len(good) < 12:
        return gen, {"aligned": False, "matches": len(good)}
    src = np.float32([k2[g.queryIdx].pt for g in good])
    dst = np.float32([k1[g.trainIdx].pt for g in good])
    M, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if M is None:
        return gen, {"aligned": False}
    out = cv2.warpAffine(gen, M, (canvas.shape[1], canvas.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    sc = float(np.hypot(M[0, 0], M[1, 0]))
    return out, {"aligned": True, "matches": len(good), "inliers": int(inl.sum()), "scale": round(sc, 4),
                 "shift_px": [round(float(M[0, 2]), 1), round(float(M[1, 2]), 1)]}


def paste_back(gen, canvas, place):
    CW, CH = place["canvas_wh"]
    if gen.shape[:2] != (CH, CW):
        gen = cv2.resize(gen, (CW, CH), interpolation=cv2.INTER_AREA)
    gen, al = align(gen, canvas, place)
    x0, y0, sw, sh = place["source_box_xywh"]
    inside = np.abs(gen[y0:y0 + sh, x0:x0 + sw].astype(float) - canvas[y0:y0 + sh, x0:x0 + sw]).mean() / 255
    out = gen.copy()
    out[y0:y0 + sh, x0:x0 + sw] = canvas[y0:y0 + sh, x0:x0 + sw]
    # seam: mean colour step across the source border (outside ring vs inside ring, 3 px)
    ring = 3
    o = np.concatenate([out[y0 - ring:y0, x0:x0 + sw].reshape(-1, 3), out[y0 + sh:y0 + sh + ring, x0:x0 + sw].reshape(-1, 3),
                        out[y0:y0 + sh, x0 - ring:x0].reshape(-1, 3), out[y0:y0 + sh, x0 + sw:x0 + sw + ring].reshape(-1, 3)])
    i = np.concatenate([out[y0:y0 + ring, x0:x0 + sw].reshape(-1, 3), out[y0 + sh - ring:y0 + sh, x0:x0 + sw].reshape(-1, 3),
                        out[y0:y0 + sh, x0:x0 + ring].reshape(-1, 3), out[y0:y0 + sh, x0 + sw - ring:x0 + sw].reshape(-1, 3)])
    seam = float(np.abs(o.astype(float).mean(0) - i.astype(float).mean(0)).mean() / 255)
    return out, {"inside_change": round(float(inside), 4), "seam_step": round(seam, 4), "alignment": al}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("out_dir")
    ap.add_argument("--quality", default="medium", choices=list(PRICE))
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--styles", default="natural,structure")
    ap.add_argument("--scale-w", type=float, default=2.0)
    ap.add_argument("--scale-h", type=float, default=1.6)
    ap.add_argument("--long", type=int, default=1280)
    ap.add_argument("--model", default="openai/gpt-image-2.5/sunburst/edit")
    ap.add_argument("--tag", default="")
    ap.add_argument("--fake", action="store_true")
    a = ap.parse_args()
    if not a.fake and not os.environ.get("FAL_KEY"):
        raise SystemExit("FAL_KEY is not set")
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    src = _read(a.source)
    canvas, mask, place = canvas_for(src, a.scale_w, a.scale_h, a.long)
    _write(out / "canvas.png", canvas)
    _write(out / "mask.png", mask)
    log_path = out / "outside.json"
    log = json.loads(log_path.read_text(encoding="utf-8")) if log_path.exists() else {"candidates": []}
    log.update({"source": str(a.source), "placement": place, "prompts": PROMPTS})
    for style in a.styles.split(","):
        for i in range(a.n):
            name = f"{style}_{a.quality}{a.tag}_{i}"
            t0 = time.time()
            try:
                gen = (np.full_like(canvas, 100) if a.fake else call_fal(a.model, PROMPTS[style], canvas, mask, a.quality))
            except Exception as e:  # keep going; record the failure (no key or body in the message)
                log["candidates"].append({"name": name, "error": f"{type(e).__name__}: {str(e)[:200]}"})
                print(name, "failed:", type(e).__name__, str(e)[:200])
                continue
            _write(out / f"{name}.raw.png", gen)
            final, chk = paste_back(gen, canvas, place)
            _write(out / f"{name}.png", final)
            rec = {"name": name, "style": style, "quality": a.quality, "model": "fake" if a.fake else a.model,
                   "seconds": round(time.time() - t0, 1), "gen_wh": [gen.shape[1], gen.shape[0]],
                   "cost_estimate_usd": 0 if a.fake else PRICE[a.quality] * (canvas.shape[0] * canvas.shape[1]) / (1024 * 768),
                   **chk}
            log["candidates"].append(rec)
            print(json.dumps(rec))
    log["cost_estimate_total_usd"] = round(sum(c.get("cost_estimate_usd", 0) for c in log["candidates"]), 3)
    log_path.write_text(json.dumps(log, indent=1) + "\n", encoding="utf-8")
    print("estimated total so far: $", log["cost_estimate_total_usd"])
    return 0





def redo(out_dir):
    """Re-run the paste-back on saved raw images (after changing align/paste_back)."""
    out = Path(out_dir)
    canvas = _read(out / "canvas.png")
    log = json.loads((out / "outside.json").read_text(encoding="utf-8"))
    for c in log["candidates"]:
        if "error" in c:
            continue
        final, chk = paste_back(_read(out / f"{c['name']}.raw.png"), canvas, log["placement"])
        _write(out / f"{c['name']}.png", final)
        c.update(chk)
        print(c["name"], json.dumps(chk))
    (out / "outside.json").write_text(json.dumps(log, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--redo":
        redo(sys.argv[2])
        sys.exit(0)
    sys.exit(main())
