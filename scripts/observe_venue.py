"""Observation summary for venue layout (analysis/observations.py), in court metres.

    python scripts/observe_venue.py tennis VIDEO 4RC.npz MASKS.npz SCENE_DIR CALIBRATION.json OUT.json
    python scripts/observe_venue.py synthetic CLIP_DIR OUT.json

tennis: 4RC points of every sampled frame, labelled by the SAM masks, moved onto the calibrated camera's rays
(analysis/to_court.warp_model; SCENE_DIR = the place_person output with the 4RC alignment).
synthetic: the clip's observed.npz (source-view points with labels; no ground truth).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    mode = sys.argv[1]
    from vrecon.analysis.observations import summarize

    if mode == "synthetic":
        clip, out = Path(sys.argv[2]), Path(sys.argv[3])
        d = np.load(clip / "observed.npz")
        gt_cam = json.loads((clip / "gt_passes" / "source" / "camera.json").read_text(encoding="utf-8"))["camera"]
        centre = gt_cam["keys"][0]["pose"]["translation"]
        s = summarize(d["points"].astype(float), d["labels"], [str(x) for x in d["label_names"]], centre)
    else:
        video, rc, masks_p, scene, cal_p, out = sys.argv[2:8]
        out = Path(out)
        import cv2

        from vrecon.analysis.from_4d import labels_at_model_pixels, load_4rc
        from vrecon.analysis.to_court import court_cameras, similarity_to_court, warp_model
        from vrecon.control.core import read_video_rgb
        from vrecon.metrics.poses import camera_pose
        from vrecon.schema import load_project

        v = read_video_rgb(video)
        t, h, w = v.shape[:3]
        z = np.load(masks_p)
        masks = z["masks"]
        if masks.shape[1:] != (h, w):
            masks = np.stack([cv2.resize(x, (w, h), interpolation=cv2.INTER_NEAREST) for x in masks])
        id_map = {int(k): n for k, n in json.loads(str(z["id_map"])).items()}
        scene = Path(scene)
        place = json.loads((scene / "placement.json").read_text(encoding="utf-8"))
        base = load_project(scene / "project.json")
        cal = json.loads(Path(cal_p).read_text(encoding="utf-8"))
        with np.load(rc) as zz:
            n = int(zz["n_frames"])
        frames = [int(f) for f in np.linspace(0, t - 1, n, dtype=int)]
        m = load_4rc(rc, w, h, frames)
        labels = labels_at_model_pixels(m, masks)
        A, k, T, _ = similarity_to_court(base, cal)
        cam = court_cameras(cal, "cam_001", base.reconstruction.cameras[0].shot_id)
        m = warp_model(m, np.array(place["4rc"]["alignment"]), k, cam)
        keep = np.ones(m.conf.shape, bool)  # no confidence cut: 4RC is least confident on the far stands, the thing we need
        names = sorted(set(id_map.values()))
        lab = np.full(labels.shape, -1)
        for kk, nm in id_map.items():
            lab[labels == kk] = names.index(nm)
        sel = keep & (lab >= 0)
        s = summarize(m.pts[sel], lab[sel], names, camera_pose(cam, 0)[1])
        s["note"] = "4RC depth moved onto the calibrated camera; far things carry 4RC's depth error (the net came out at Y = +1.9 m)"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(s, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk in ("centroid", "z_range", "line")} for k, v in s["things"].items()}, indent=1)[:3000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
