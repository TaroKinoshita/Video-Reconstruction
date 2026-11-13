"""Add the SMPL-X body mesh (this person's shape, rest pose) with skinning weights to a gvhmr_skeleton.npz, so
Blender can draw a real body instead of cylinders. Runs in WSL, gvhmr env, from the GVHMR checkout:

    cd ~/vrecon/GVHMR && ~/vrecon/bin/micromamba run -r ~/vrecon/mamba -n gvhmr \
        python .../gpu/observe/gvhmr_body_mesh.py HMR4D_RESULTS.pt SKELETON.npz

Adds to SKELETON.npz (same conventions as run_gvhmr.py: metres, Z up, pelvis at the origin in the rest pose):
    body_verts [V,3]     rest-pose vertices with this person's betas
    body_faces [F,3]
    body_weights [V,22]  linear-blend-skinning weights on the 22 body joints; SMPL-X's jaw and eyes are folded
                         into head, each hand's finger joints into its wrist (the skeleton has no finger joints)
With the skeleton's local rotations, Blender's armature deform (linear blend skinning) then gives the SMPL-X body
without pose correctives (pose blend shapes are not applied: elbows and knees thin a little when bent).
The SMPL-X model is licensed (non-commercial); the npz stays under runs/ (not committed).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

A = np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])  # y-up -> z-up, as run_gvhmr.py


def main() -> int:
    res_path, sk_path = sys.argv[1], Path(sys.argv[2])
    sys.path.insert(0, str(Path.cwd()))
    from hmr4d.utils.smplx_utils import make_smplx

    pred = torch.load(res_path, map_location="cpu")
    bm = make_smplx("supermotion")
    g = {k: v.float() for k, v in pred["smpl_params_global"].items()}
    zero = {k: torch.zeros_like(v[:1]) for k, v in g.items()}
    zero["betas"] = g["betas"][:1]
    with torch.no_grad():
        o = bm(**zero)
    joints = (o.joints[:, :22] if o.joints.dim() == 3 else o.joints[0, :, :22])[0].numpy()
    verts = (o.vertices if o.vertices.dim() == 3 else o.vertices[0])[0].numpy() - joints[0]
    w = bm.bm.lbs_weights.numpy().copy()  # [V, 55]
    w22 = w[:, :22].copy()
    w22[:, 15] += w[:, 22:25].sum(1)  # jaw, eyes -> head
    w22[:, 20] += w[:, 25:40].sum(1)  # left hand fingers -> left wrist
    w22[:, 21] += w[:, 40:55].sum(1)  # right hand fingers -> right wrist
    d = dict(np.load(sk_path))
    d["body_verts"] = (verts @ A.T).astype(np.float32)
    d["body_faces"] = np.asarray(bm.faces, np.int32)
    d["body_weights"] = w22.astype(np.float32)
    np.savez_compressed(sk_path, **d)
    print({"verts": verts.shape, "faces": d["body_faces"].shape, "weight_sum": float(w22.sum(1).mean())})
    return 0


if __name__ == "__main__":
    sys.exit(main())
