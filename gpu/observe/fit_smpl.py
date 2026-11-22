"""A real body surface for every person in the scene: SMPL fitted to the scene's joints (whatever made them: GVHMR,
FAB, the close-up fit, the arm refit). Runs in the fab env (torch + FAB's SMPL layer, the SMPL model of
~/.cache/fab/body_models/smpl).

    <fab env>/python gpu/observe/fit_smpl.py IN.npz OUT.npz

Per cut with a body [T, 22, 3] (world, metres): one shape (betas) for the cut, per frame the 22 joint rotations and
the root translation, fitted to the joints (inverse kinematics) with a small pose prior and temporal smoothness.
Writes s<k>_smpl [T, 6890, 3] float16 vertices (world) and smpl_faces [13776, 3]; info["smpl"] has the joint error
per cut. The scene builder draws these instead of the stick mannequin when present.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ITERS = 400
# a child has no SMPL model of its own: the neutral body, sized by the fit to the measured joints
BODY_MODEL = {"man": "MALE", "woman": "FEMALE", "child": "NEUTRAL", "unknown": "NEUTRAL"}
W_SMOOTH, W_POSE, W_BETA = 2.0, 1e-3, 1e-2  # betas stay free enough to size a child; measured: they stay near 0 for adults


def fit(smpl, J: np.ndarray, dev) -> tuple[np.ndarray, float, list]:
    T = len(J)
    tgt = torch.tensor(J, dtype=torch.float32, device=dev)
    go = torch.zeros(T, 3, device=dev, requires_grad=True)
    bp = torch.zeros(T, 69, device=dev, requires_grad=True)
    betas = torch.zeros(1, 10, device=dev, requires_grad=True)
    tr = tgt[:, 0].clone().requires_grad_(True)
    # face the body the way its hips face: start the root rotation from the hip line (SMPL: x = left, z = forward, y = up)
    with torch.no_grad():
        hip = tgt[:, 1] - tgt[:, 2]
        up = tgt[:, 12] - tgt[:, 0]
        x = hip / hip.norm(dim=1, keepdim=True)
        y = up - (up * x).sum(1, keepdim=True) * x
        y = y / y.norm(dim=1, keepdim=True)
        z = torch.cross(x, y, dim=1)
        R = torch.stack([x, y, z], 2)
        go[:] = matrix_to_axis_angle(R)
    opt = torch.optim.Adam([go, bp, betas, tr], lr=0.03)
    for it in range(ITERS):
        if it == ITERS // 2:
            for g in opt.param_groups:
                g["lr"] = 0.01
        _, joints = smpl(go, bp, betas.expand(T, -1), transl=tr)
        loss = ((joints[:, :22] - tgt) ** 2).sum(-1).mean()
        if T > 2:
            loss = loss + W_SMOOTH * ((bp[2:] - 2 * bp[1:-1] + bp[:-2]) ** 2).mean() * 1e-2
        loss = loss + W_POSE * (bp ** 2).mean() + W_BETA * (betas ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        verts, joints = smpl(go, bp, betas.expand(T, -1), transl=tr)
        err = float((joints[:, :22] - tgt).norm(dim=-1).mean())
    return verts.cpu().numpy(), err, betas.detach().cpu().numpy().ravel().round(2).tolist()


def matrix_to_axis_angle(R: torch.Tensor) -> torch.Tensor:
    cos = ((R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]) - 1) / 2
    ang = torch.arccos(cos.clamp(-1 + 1e-6, 1 - 1e-6))
    ax = torch.stack([R[:, 2, 1] - R[:, 1, 2], R[:, 0, 2] - R[:, 2, 0], R[:, 1, 0] - R[:, 0, 1]], 1)
    ax = ax / (2 * torch.sin(ang)[:, None]).clamp(min=1e-6)
    return ax * ang[:, None]


def main() -> int:
    import fab.paths
    from bench.body_models import SMPL

    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    z = dict(np.load(src))
    info = json.loads(str(z["info"]))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    models = {}

    def model(kind: str):
        if kind not in models:
            models[kind] = SMPL(fab.paths.body_models_dir() / "smpl" / f"SMPL_{kind}.pkl").to(dev)
        return models[kind]

    who = info.get("who", {})  # scripts/multishot_who.py: man | woman | child | unknown per cut
    z["smpl_faces"] = model("NEUTRAL").faces.cpu().numpy().astype(np.int32)  # same topology for all three
    rep = {}
    for k in range(len(info["shots"])):
        J = z[f"s{k}_body"]
        if not len(J):
            continue
        kind = BODY_MODEL.get(who.get(str(k), {}).get("type", "unknown"), "NEUTRAL")
        v, err, b = fit(model(kind), J, dev)
        z[f"s{k}_smpl"] = v.astype(np.float16)
        rep[k] = {"frames": len(J), "joint_error_m": round(err, 3), "betas": b, "model": kind}
        print(k, rep[k], flush=True)
    info["smpl"] = rep
    z["info"] = json.dumps(info)
    np.savez_compressed(out, **z)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
