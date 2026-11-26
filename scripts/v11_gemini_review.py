"""Send one short video to the Gemini API (Interactions API, inline base64) with a fixed frame rate, save the raw reply.

Standard library only. The key is read from the GEMINI_API_KEY environment variable and never printed.
Request shape from https://aistudio.google.com/docs/video-understanding ("processing": {"type": "static", "fps": N},
fps in (0, 24]). The page's examples use other model ids; check the reply for an error if the model id is refused.

  python scripts/v11_gemini_review.py VIDEO.mp4 PROMPT.txt OUT.json --model gemini-3.8-flash --fps 24
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

URL = "https://generativelanguage.googleapis.com/v1beta/interactions"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("prompt_file")
    ap.add_argument("out")
    ap.add_argument("--model", default="gemini-3.8-flash")
    ap.add_argument("--fps", type=float, default=24.0)
    ap.add_argument("--api", choices=["interactions", "generate"], default="interactions")
    ap.add_argument("--retries", type=int, default=6, help="retries on 429/5xx, waiting 1, 2, 4, 8, 10, 10 min")
    ap.add_argument("--wait", type=int, default=0, help="fixed seconds between retries (0 = 1, 2, 4, 8, 10 min...)")
    ap.add_argument("--resolution", choices=["low", "medium", "high"], default=None,
                    help="per-item media resolution (interactions API; documented with image examples only)")
    a = ap.parse_args()

    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        sys.exit("GEMINI_API_KEY is not set")
    data = Path(a.video).read_bytes()
    if len(data) > 15 * 1024 * 1024:
        sys.exit(f"video is {len(data) / 1e6:.1f} MB; inline requests are documented for under 20 MB in total")
    b64 = base64.b64encode(data).decode("ascii")
    prompt = Path(a.prompt_file).read_text(encoding="utf-8")
    if a.api == "interactions":
        url = URL
        body = {
            "model": a.model,
            "input": [
                {"type": "video", "data": b64, "mime_type": "video/mp4",
                 "processing": {"type": "static", "fps": a.fps}},
                {"type": "text", "text": prompt},
            ],
        }
        if a.resolution:
            body["input"][0]["resolution"] = a.resolution
    else:  # generateContent with videoMetadata.fps (older documented form; not re-checked against current docs)
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{a.model}:generateContent"
        body = {"contents": [{"parts": [
            {"inline_data": {"mime_type": "video/mp4", "data": b64}, "video_metadata": {"fps": a.fps}},
            {"text": prompt},
        ]}]}
    req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), method="POST",
                                 headers={"x-goog-api-key": key, "Content-Type": "application/json"})
    attempts = []
    for k in range(a.retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                reply = json.loads(r.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as e:
            reply = {"http_error": e.code, "body": e.read().decode("utf-8", errors="replace")}
        attempts.append({"attempt": k + 1, "time": time.strftime("%H:%M:%S"), "http_error": reply["http_error"]})
        if reply["http_error"] not in (429, 500, 503, 504) or k == a.retries:
            break
        wait = a.wait if a.wait else min(60 * 2 ** k, 600)
        print(f"attempt {k + 1}: http {reply['http_error']}, retry in {wait} s", flush=True)
        time.sleep(wait)
    reply["failed_attempts"] = attempts
    out = {"model": a.model, "fps": a.fps, "api": a.api, "resolution": a.resolution, "video": str(a.video), "video_bytes": len(data), "reply": reply}
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("saved", a.out, "(http error)" if "http_error" in reply else "")


if __name__ == "__main__":
    main()
