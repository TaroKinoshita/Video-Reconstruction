"""Objects in each cut of a multi-cut clip: one fixed vocabulary (the same for every video), SAM 3.1 text prompts.
Runs in the observe env.

    python gpu/observe/find_objects_sam3.py OUT_DIR TAG

1 search   3 keyframes per cut (first, middle, last) go into one short video; every vocabulary word is prompted on
           it once (detection runs on each frame). A word counts for a cut when it is found in at least one of the
           cut's keyframes with a mask of at most MAX_AREA of the picture (bigger: a person or a wall, not a thing).
2 track    each (cut, word) found is tracked through the whole cut (long cuts at half size).
Writes OUT_DIR/TAG_shotNN.objects.npz: masks uint16 [T, H, W] (0 = none), id_map JSON {id: word}; and
OUT_DIR/TAG_objects.json with the search hits.
"""

from __future__ import annotations

import contextlib
import functools
import inspect
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.attention

sys.path.insert(0, str(Path(__file__).parent))
from run_sam3 import _install_edt_without_triton, frame_count_and_size, segment  # noqa: E402

VOCAB = ["gun", "knife", "phone", "bag", "suitcase", "umbrella", "bottle", "cup", "book", "box", "ball", "racket",
         "bat", "chair", "table", "bench", "sofa", "bed", "lamp", "car", "bicycle", "motorcycle", "truck", "dog",
         "cat", "horse", "bird", "laptop", "television", "clock", "sign", "flag", "bucket", "barrel", "crate",
         "trash can", "traffic cone", "hat", "helmet", "plant", "guitar", "microphone", "camera", "bullet casing",
         "debris", "rope", "door", "ladder", "pipe", "railing"]
MAX_AREA = 0.25
HALF_ABOVE = 120


def predictor(max_objects: int = 3):
    _install_edt_without_triton()
    original = torch.nn.attention.sdpa_kernel
    torch.nn.attention.sdpa_kernel = functools.wraps(original)(lambda *a, **k: contextlib.nullcontext())
    from sam3.model_builder import build_sam3_predictor

    p = build_sam3_predictor(version="sam3.1", use_fa3=False, async_loading_frames=False, max_num_objects=max_objects)
    init_state = p.model.init_state
    ok = set(inspect.signature(init_state).parameters)
    p.model.init_state = lambda **kw: init_state(**{k: v for k, v in kw.items() if k in ok})
    p.model.batched_grounding_batch_size = 1
    p.model.new_det_thresh = 0.5  # precision: a word must be clearly there
    p.model.masklet_confirmation_enable = False
    p.model.score_threshold_detection = 0.4
    p.model.suppress_unmatched_only_within_hotstart = True
    return p


def main() -> int:
    out, tag = Path(sys.argv[1]), sys.argv[2]
    shots = json.loads((out / f"shots_{tag}.json").read_text(encoding="utf-8"))["shots"]
    pred = predictor()
    with tempfile.TemporaryDirectory() as td:
        keys = []
        for k, (a, b) in enumerate(shots):
            n = b - a
            for f in sorted({0, n // 2, n - 1}):
                keys.append((k, f))
                src = out / f"rc_{tag}{k:02d}" / "frames" / f"{f + 1:05d}.png"
                (Path(td) / f"{len(keys) - 1:05d}.png").write_bytes(src.read_bytes())
        kv = str(Path(td) / "keys.mp4")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", "1", "-i", str(Path(td) / "%05d.png"),
                        "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", kv], check=True)
        frames = frame_count_and_size(kv)
        hits: dict[int, set] = {}
        for w in VOCAB:
            found = segment(pred, kv, w, frames, 0)
            for m in found.values():
                for i in np.flatnonzero(m.any((1, 2))):
                    if m[i].mean() <= MAX_AREA:
                        hits.setdefault(keys[i][0], set()).add(w)
            print(w, {k: sorted(v) for k, v in hits.items() if w in v}, flush=True)
    (out / f"{tag}_objects.json").write_text(json.dumps({k: sorted(v) for k, v in hits.items()}, indent=1), encoding="utf-8")
    for k, words in sorted(hits.items()):
        v = out / f"{tag}_shot{k:02d}.mp4"
        if shots[k][1] - shots[k][0] > HALF_ABOVE and (out / f"{tag}_shot{k:02d}_half.mp4").exists():
            v = out / f"{tag}_shot{k:02d}_half.mp4"
        fr = frame_count_and_size(str(v))
        masks = np.zeros(fr, np.uint16)
        id_map = {}
        for w in sorted(words):
            for m in segment(pred, str(v), w, fr, 0).values():
                if m.any() and m[m.any((1, 2))].mean() <= MAX_AREA:
                    num = len(id_map) + 1
                    id_map[num] = w
                    masks[m & (masks == 0)] = num
        np.savez_compressed(out / f"{tag}_shot{k:02d}.objects.npz", masks=masks, id_map=json.dumps(id_map))
        print(k, id_map, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
