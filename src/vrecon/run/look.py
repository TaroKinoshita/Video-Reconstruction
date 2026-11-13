"""The "looker" prototype (docs/DIRECTION_OPTIONS.md, option 2): the LLM looks, the tools adjust.

`compare_sheet` draws the reference video next to the adopted reconstruction (proxy boxes over the same
frame), so an LLM can say what looks wrong in plain terms. `fix` turns one such complaint (an issue word and
a target) into the matching search; the LLM never writes numbers. The result is judged by the same accept
rule as every other proposal.

Only the reference video and the adopted scene are drawn: no ground truth.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

ISSUES = {
    "camera": "the whole picture is zoomed in or out (focal length)",
    "misplaced": "an object sits in the wrong place on the floor",
    "wrong_size": "an object is too big, too small, too long or too flat",
    "wrong_facing": "an object faces or turns the wrong way",
    "wrong_timing": "a moving object moves too early or too late",
}
_HYP = {"misplaced": "misplaced on the floor", "wrong_size": "has the wrong size", "wrong_facing": "faces the wrong way",
        "wrong_timing": "moves at the wrong time"}
_COLOURS = [(255, 80, 80), (80, 200, 255), (255, 200, 40), (120, 255, 120), (220, 120, 255), (255, 150, 200)]


def _video(cfg: dict) -> np.ndarray:
    from vrecon.control.core import read_video_rgb

    if not cfg.get("video"):
        raise ValueError("this run has no reference video")
    return read_video_rgb(cfg["video"])


def compare_sheet(run_dir: str, frames: list[int] | None = None, n: int = 4, max_width: int = 1280) -> dict:
    """Write <run_dir>/look/rev_XXX.png: one row per frame, left = reference, right = reference with the
    reconstruction's boxes drawn over it (filled, labelled). Returns the path and the legend."""
    import cv2

    from vrecon.optimize import render_silhouettes

    from .service import _config, _manager

    rd = Path(run_dir)
    cfg = _config(rd)
    mgr, obs, _ = _manager(rd)
    p = mgr.adopted
    video = _video(cfg)
    t, h, w = obs.masks.shape
    t = min(t, len(video))
    if frames is None:
        frames = [int(f) for f in np.linspace(0, t - 1, n).round()]
    ids = [o.id for o in p.reconstruction.objects if o.kind != "set"]
    sets = [o.id for o in p.reconstruction.objects if o.kind == "set"]
    shot = p.reconstruction.cameras[0].shot_id
    sil = render_silhouettes(p, shot, frames, sets + ids, (w, h))
    rows = []
    for i, f in enumerate(frames):
        ref = cv2.resize(video[f], (w, h), interpolation=cv2.INTER_AREA) if video[f].shape[:2] != (h, w) else video[f]
        over = (ref * 0.55).astype(np.uint8)
        for n_, oid in enumerate(sets + ids, start=1):
            m = sil[i] == n_
            if not m.any():
                continue
            if oid in sets:  # set pieces: outline only, grey
                cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(over, cs, -1, (200, 200, 200), 1)
                continue
            col = np.array(_COLOURS[ids.index(oid) % len(_COLOURS)])
            over[m] = (over[m] * 0.4 + col * 0.6).astype(np.uint8)
            cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(over, cs, -1, tuple(int(c) for c in col), 2)
            v, u = np.nonzero(m)
            cv2.putText(over, oid, (int(u.min()), max(int(v.min()) - 4, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        tuple(int(c) for c in col), 1, cv2.LINE_AA)
        row = np.concatenate([ref, np.full((h, 6, 3), 255, np.uint8), over], axis=1)
        cv2.putText(row, f"frame {f}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1, cv2.LINE_AA)
        rows.append(row)
    sheet = np.concatenate(rows, axis=0)
    if sheet.shape[1] > max_width:
        s = max_width / sheet.shape[1]
        sheet = cv2.resize(sheet, (max_width, int(sheet.shape[0] * s)), interpolation=cv2.INTER_AREA)
    out = rd / "look" / f"rev_{p.revision:03d}.png"
    out.parent.mkdir(exist_ok=True)
    ok, buf = cv2.imencode(".png", cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))  # not imwrite: non-ASCII paths
    if not ok:
        raise RuntimeError("png encode failed")
    buf.tofile(str(out))
    return {"path": str(out), "revision": p.revision, "frames": frames,
            "legend": {oid: list(_COLOURS[i % len(_COLOURS)]) for i, oid in enumerate(ids)},
            "how_to_read": "left: reference video; right: the reconstruction's boxes over the same frame "
                           "(coloured = objects, grey outline = floor/wall). Compare position, size, facing, "
                           "timing and overall zoom.",
            "issues": ISSUES}


def fix(run_dir: str, issue: str, target: str | None = None, note: str = "", max_evals: int = 60) -> dict:
    """Run the search that matches one visual complaint. issue: one of ISSUES; target: object id (not for
    'camera'). note: the looker's own words, kept in the history."""
    from vrecon.optimize import Param

    from .policy import hypotheses
    from .service import _manager, run_optimizer

    if issue not in ISSUES:
        raise ValueError(f"issue must be one of {sorted(ISSUES)}")
    mgr, _, _ = _manager(Path(run_dir))
    hs = hypotheses(mgr.adopted)
    if issue == "camera":
        params = hs[0][1]
    else:
        if not target:
            raise ValueError("this issue needs a target object id")
        want = f"{target} {_HYP[issue]}"
        match = [h for h in hs if h[0] == want]
        if not match:
            raise ValueError(f"no such hypothesis for {target} (static objects have no timing)")
        params = match[0][1]
    if issue == "wrong_timing":  # first the machine's onsets, applied directly; the search only if that fails
        from .service import align_timing

        al = align_timing(run_dir, target, note)
        if any(t.get("status") == "accepted" for t in al.get("tries", [])):
            return {"aligned_from_onsets": al}
    specs = [_spec(p) for p in params]
    text = f"[looker] {issue}" + (f" {target}" if target else "") + (f": {note}" if note else "")
    return run_optimizer(run_dir, specs, text, max_evals=max_evals)


def _spec(p) -> str:
    return f"{p.kind}:{p.target_id}:{p.axis}:{p.step}"


__all__ = ["ISSUES", "compare_sheet", "fix"]
