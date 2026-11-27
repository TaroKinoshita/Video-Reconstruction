"""People placement as in MAVE (arXiv 2611.22464, eq. 4-5 and 10), applied to bodies from scripts/v11_people_anim.py.
Pure numpy; the MAVE code itself is not public (README: "coming soon", checked 2026-11-28), so this follows the paper.

  python scripts/v11_people_place.py PREP.npz PEOPLE.npz BODY_DIR OUT.npz [--parts PARTS.npz]

eq. 4-5: per frame the neck (MHR70 keypoint 69) sits at its depth on the camera ray (f_t, taken from the input bodies,
whose depth came from the scene depth), and the body around the neck is scaled by one factor per person,
m = median_t f_t; so the body keeps one size relative to SAM 3D Body's own prediction.
eq. 10 (optional, --parts): one factor lambda per person scales the whole track about the camera centres (picture
kept); here lambda is chosen by foot support (MAVE's agent may also use height or direct assignment): the median over
frames of the factor that puts the lowest vertices on the ground height field.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from v11_mave_prep import read_ply  # noqa: E402

NECK = 69  # MHR70: the neck, about the middle of the shoulders


def main() -> int:
    ap = argparse.ArgumentParser()
    for n in ("prep", "people", "body", "out"):
        ap.add_argument(n, type=Path)
    ap.add_argument("--parts", type=Path, default=None, help="v11_parts.py output: ground for the foot-support lambda")
    a = ap.parse_args()
    prep = np.load(a.prep)
    Rc, T = prep["cam_R"], prep["cam_T"]
    d = dict(np.load(a.people))
    ground = None
    if a.parts is not None:
        pz = np.load(a.parts)
        parts = json.loads(str(pz["parts"]))
        GR, Gt, gv = np.array(parts["world_R"]), np.array(parts["world_t"]), pz["ground_v"]
        ground = (GR, Gt, gv)

    def kp(label, t):
        js = a.body / f"f{int(t):04d}_{label}.json"
        return np.array(json.loads(js.read_text(encoding="utf-8"))["metadata"]["people"][0]["keypoints_3d"])

    report = {}
    for label in json.loads(str(d["labels"])):
        ts, X = d[f"{label}_t"], d[f"{label}_v"]
        mid = int(ts[len(ts) // 2])
        V, _ = read_ply(a.body / f"f{mid:04d}_{label}_0.ply")
        V = V * [1, -1, -1]
        neck_v = np.argsort(np.linalg.norm(V - kp(label, mid)[NECK], axis=1))[:20]  # same topology on every frame
        Y = np.einsum("tvk,tkl->tvl", X - T[ts][:, None], Rc[ts])  # world -> camera
        yn = Y[:, neck_v].mean(1)
        f = np.array([np.linalg.norm(yn[j]) / np.linalg.norm(kp(label, t)[NECK]) for j, t in enumerate(ts)])
        m = float(np.median(f))
        Y2 = yn[:, None] + (m / f)[:, None, None] * (Y - yn[:, None])  # eq. 5
        Xw = np.einsum("tvl,tkl->tvk", Y2, Rc[ts]) + T[ts][:, None]
        lam = 1.0
        if ground is not None:  # eq. 10 with foot support
            GR, Gt, gv = ground
            lams = []
            for j, t in enumerate(ts):
                W = Xw[j] @ GR.T + Gt
                o = T[t] @ GR.T + Gt
                low = W[np.argsort(W[:, 2])[:30]].mean(0)
                L = 1.0
                for _ in range(5):
                    q = o + L * (low - o)
                    near = np.argsort(((gv[:, :2] - q[:2]) ** 2).sum(1))[:8]
                    L = (gv[near, 2].mean() - o[2]) / (low[2] - o[2])
                lams.append(L)
            lam = float(np.median(lams))
            Xw = T[ts][:, None] + lam * (Xw - T[ts][:, None])
        d[f"{label}_v"] = Xw.astype(np.float32)
        report[label] = {"f_t_p5_50_95": [round(float(x), 3) for x in np.percentile(f, [5, 50, 95])], "m": round(m, 3),
                         "lambda": round(lam, 3)}
    np.savez_compressed(a.out, **d)
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
