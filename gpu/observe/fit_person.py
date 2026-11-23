"""One body per person, fitted to the picture (docs/PLAN_v4.md phase 1). Runs in the fab env (torch + SMPL).

    <fab env>/python gpu/observe/fit_person.py IN.npz DIAG TAG IDENTITY OUT.npz [--cuts 0,2] [--iters 600]

For the person IDENTITY (info["who"], scripts/multishot_who.py) over the given cuts (default: all cuts of that
person): ONE shape (betas) shared by every cut, and per cut and frame the joint rotations and the root position.
The cameras stay as they are. The fit pulls the body's joints onto the 2D keypoints of the video
(DIAG/kp2d_TAGNN.npz, ViTPose: shoulders, elbows, wrists, hips, knees, ankles, and the face from SMPL surface points),
weighted by their confidence, with a weak pull to the scene's current 3D joints (depth, and whatever the picture
does not show), smooth motion and a pose prior.

Held-out check points: per keypoint, a third of the frames in blocks of HOLD_BLOCK (shifted per keypoint) are not
used by the fit. Two candidates per cut, "current" (the scene's joints) and "body" (this fit), are compared on
these points only: the fit is chosen when the held-out median goes down; each part is reported better / same /
worse / no evidence. Parts the picture rarely shows, and parts that got worse, are listed as "unobserved"; a
close-up's camera distance is listed as "uncertain" (docs/PLAN_v4.md: what the picture cannot decide is passed
on as open). Wide cuts start from GVHMR's own joint rotations when they exist. Writes OUT.npz = IN with s<k>_body (new joints), s<k>_smpl, smpl_faces for those cuts, and
info["fit_person"][IDENTITY] (errors, betas, model).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

KP_CONF = 0.5
# SMPL joint -> COCO keypoint; face keypoints come from SMPL surface vertices (as used by SPIN / VIBE)
J2C = [(16, 5), (17, 6), (18, 7), (19, 8), (20, 9), (21, 10), (1, 11), (2, 12), (4, 13), (5, 14), (7, 15), (8, 16)]
V2C = [(332, 0), (2800, 1), (6260, 2), (583, 3), (4071, 4)]  # nose, left eye, right eye, left ear, right ear
PART = {0: "face", 1: "face", 2: "face", 3: "face", 4: "face", 5: "shoulders", 6: "shoulders", 7: "elbows", 8: "elbows",
        9: "wrists", 10: "wrists", 11: "hips", 12: "hips", 13: "knees", 14: "knees", 15: "ankles", 16: "ankles"}
KP_W = np.ones(17, np.float32)
KP_W[[11, 12]] = 0.5  # SMPL hip joints sit inside the pelvis, COCO hips on its side
BODY_MODEL = {"man": "MALE", "woman": "FEMALE"}
REF_W = 1280.0
HOLD_BLOCK = 4
WORSE_PX = 3.0  # a part counts as worse when its held-out median grows by more than this
SEEN_SHARE = 0.3  # a part is observed in a cut when confident in at least this share of frames
W_2D, W_3D, W_SMOOTH, W_TR, W_POSE, W_BETA = 1.0, 0.3, 5.0, 50.0, 0.02, 0.01
W_BETA_GIRTH = 1.0


TORSO = [0, 1, 2, 3, 6, 9, 12, 13, 14, 15, 16, 17]


def similarity(src: np.ndarray, dst: np.ndarray):
    """s, R, t with dst ~ s R src + t (Umeyama)."""
    ms, md = src.mean(0), dst.mean(0)
    a, b = src - ms, dst - md
    U, S, Vt = np.linalg.svd(b.T @ a)
    D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    R = U @ D @ Vt
    s = (S * np.diag(D)).sum() / (a ** 2).sum()
    return s, R, md - s * R @ ms


def own_motion(J3: np.ndarray, fab_path: Path) -> tuple[np.ndarray, str]:
    """a close-up drawn with one pose copied to every frame: put FAB's per-frame joints (the close-up's own motion)
    where that pose stands, by the similarity that best maps one FAB frame's torso onto it."""
    if not fab_path.exists() or not np.allclose(J3, J3[:1], atol=1e-5):
        return J3, "scene"
    f = np.load(fab_path)
    ok = np.flatnonzero(f["ok"])
    if not len(ok):
        return J3, "scene"
    best = min(ok, key=lambda t: np.linalg.norm((lambda s, R, tt: (s * f["joints"][t][TORSO] @ R.T + tt))(
        *similarity(f["joints"][t][TORSO], J3[0][TORSO])) - J3[0][TORSO]))
    s, R, t = similarity(f["joints"][best][TORSO], J3[0][TORSO])
    out = J3.copy()
    for i in range(min(len(J3), len(f["ok"]))):
        src = f["joints"][i] if f["ok"][i] else f["joints"][ok[np.argmin(np.abs(ok - i))]]
        out[i] = s * src @ R.T + t
    return out, f"fab (frame {int(best)} anchors the place)"


ALIGN_CLOSEUP = False  # tried on Matrix A: cut 2 held-out 62 -> 111 px with it, so off (kept for later tests)
MIN_POINTS = 15  # a cut with fewer confident keypoints keeps its current body and is marked "too little evidence"


def align_closeup(J3: np.ndarray, cam: np.ndarray, lens: float, kp: np.ndarray, w: int, h: int) -> np.ndarray:
    """a close-up's start: slide the body across the picture (at its current depth) so that its shoulders and head
    land on the keypoints' shoulders and face; per frame, then a running median over frames. The depth stays (the
    picture cannot tell it, docs/PLAN_v4.md: camera distance stays open)."""
    f = lens / 36.0 * w
    T = min(len(J3), len(cam), len(kp))
    shift = np.full((T, 3), np.nan)
    for t in range(T):
        w2c = np.linalg.inv(cam[t])
        src, dst = [], []
        for sj, cj in ((16, 5), (17, 6), (15, 0), (15, 1), (15, 2), (18, 7), (19, 8)):
            if kp[t, cj, 2] >= KP_CONF:
                src.append(J3[t, sj])
                dst.append(kp[t, cj, :2])
        if not src:
            continue
        pc = np.array(src) @ w2c[:3, :3].T + w2c[:3, 3]
        d = np.median(-pc[:, 2])
        if d <= 0.05:
            continue
        uv = np.stack([w / 2 + f * pc[:, 0] / -pc[:, 2], h / 2 - f * pc[:, 1] / -pc[:, 2]], 1)
        du, dv = np.median(np.array(dst) - uv, 0)
        shift[t] = cam[t, :3, :3] @ np.array([du * d / f, -dv * d / f, 0.0])
    ok = np.isfinite(shift[:, 0])
    if not ok.any():
        return J3
    idx = np.flatnonzero(ok)
    shift = shift[idx[np.abs(np.arange(T)[:, None] - idx[None]).argmin(1)]]
    sm = np.array([np.median(shift[max(0, t - 2):t + 3], 0) for t in range(T)])
    out = J3.copy()
    out[:T] += sm[:, None]
    return out


def matrix_to_axis_angle(R: torch.Tensor) -> torch.Tensor:
    cos = ((R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]) - 1) / 2
    ang = torch.arccos(cos.clamp(-1 + 1e-6, 1 - 1e-6))
    ax = torch.stack([R[:, 2, 1] - R[:, 1, 2], R[:, 0, 2] - R[:, 2, 0], R[:, 1, 0] - R[:, 0, 1]], 1)
    return ax / (2 * torch.sin(ang)[:, None]).clamp(min=1e-6) * ang[:, None]


class Cut:
    def __init__(self, k, J3, cam, lens, kp, w, h, dev):
        T = min(len(J3), len(cam), len(kp))
        self.k, self.T, self.w, self.h = k, T, w, h
        self.J3 = torch.tensor(J3[:T], dtype=torch.float32, device=dev)
        w2c = np.linalg.inv(cam[:T])
        self.R = torch.tensor(w2c[:, :3, :3], dtype=torch.float32, device=dev)
        self.t = torch.tensor(w2c[:, :3, 3], dtype=torch.float32, device=dev)
        self.f = lens / 36.0 * w
        kp = kp[:T].copy()
        kp[kp[..., 2] < KP_CONF, 2] = 0
        self.kp = torch.tensor(kp, dtype=torch.float32, device=dev)
        # held-out check points: per keypoint, blocks of HOLD_BLOCK frames, a third of them, shifted per keypoint so
        # every frame keeps most of its keypoints for the fit
        held = ((np.arange(T)[:, None] // HOLD_BLOCK + np.arange(17)[None]) % 3 == 0)
        self.fit_mask = torch.tensor(~held, device=dev)
        self.go = torch.zeros(T, 3, device=dev, requires_grad=True)
        self.bp = torch.zeros(T, 69, device=dev, requires_grad=True)
        self.tr = self.J3[:, 0].clone().requires_grad_(True)
        with torch.no_grad():  # root rotation from the hip line (SMPL: x left, y up, z forward)
            x = self.J3[:, 1] - self.J3[:, 2]
            x = x / x.norm(dim=1, keepdim=True)
            up = self.J3[:, 12] - self.J3[:, 0]
            y = up - (up * x).sum(1, keepdim=True) * x
            y = y / y.norm(dim=1, keepdim=True)
            self.go[:] = matrix_to_axis_angle(torch.stack([x, y, torch.cross(x, y, dim=1)], 2))

    def start_from_gvhmr(self, path: Path) -> bool:
        """wide cuts: start the body pose from GVHMR's own joint rotations (scene/conventions: GVHMR's local_rot are
        conjugated to Z-up; undo it for SMPL's y-up frame). Joints 1..21 of SMPL-X and SMPL share their meaning."""
        if not path.exists():
            return False
        g = np.load(path)
        q = g["local_rot"][: self.T]
        if len(q) < self.T:
            return False
        w, x, y, zq = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
        R = np.stack([np.stack([1 - 2 * (y * y + zq * zq), 2 * (x * y - zq * w), 2 * (x * zq + y * w)], -1),
                      np.stack([2 * (x * y + zq * w), 1 - 2 * (x * x + zq * zq), 2 * (y * zq - x * w)], -1),
                      np.stack([2 * (x * zq - y * w), 2 * (y * zq + x * w), 1 - 2 * (x * x + y * y)], -1)], -2)
        A = np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])
        R = A.T @ R @ A
        aa = matrix_to_axis_angle(torch.tensor(R[:, 1:22].reshape(-1, 3, 3), dtype=torch.float32)).reshape(self.T, 21 * 3)
        with torch.no_grad():
            self.bp[:, :63] = aa.to(self.bp.device)
        return True

    def project(self, P):  # [T, N, 3] world -> [T, N, 2] pixels, depth
        pc = torch.einsum("tij,tnj->tni", self.R, P) + self.t[:, None]
        d = (-pc[..., 2]).clamp(min=0.05)
        return torch.stack([self.w / 2 + self.f * pc[..., 0] / d, self.h / 2 - self.f * pc[..., 1] / d], -1), d

    def coco(self, verts, joints):
        """the body's 17 COCO points [T, 17, 3] (world)."""
        P = torch.zeros(self.T, 17, 3, device=verts.device)
        for sj, c in J2C:
            P[:, c] = joints[:, sj]
        for v, c in V2C:
            P[:, c] = verts[:, v]
        return P

    def px_error(self, P, mask) -> dict:
        """pixel error (1280-wide) of COCO points P against the keypoints, over mask & confident, per part."""
        uv, _ = self.project(P)
        e = (uv - self.kp[..., :2]).norm(dim=-1) * (REF_W / self.w)
        ok = mask & (self.kp[..., 2] > 0)
        out, allv = {}, []
        for part in sorted(set(PART.values())):
            idx = [c for c, p in PART.items() if p == part]
            v = e[:, idx][ok[:, idx]].detach().cpu().numpy()
            out[part] = round(float(np.median(v)), 1) if len(v) else None
            allv += v.tolist()
        out["all"] = round(float(np.median(allv)), 1) if allv else None
        out["n"] = len(allv)
        return out


def main() -> int:
    import fab.paths
    from bench.body_models import SMPL

    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("diag", type=Path)
    ap.add_argument("tag")
    ap.add_argument("identity")
    ap.add_argument("out", type=Path)
    ap.add_argument("--cuts", default="")
    ap.add_argument("--iters", type=int, default=600)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    z = dict(np.load(a.src))
    info = json.loads(str(z["info"]))
    who = info.get("who", {})
    cuts = [int(c) for c in a.cuts.split(",")] if a.cuts else [int(k) for k, v in who.items() if v.get("identity") == a.identity]
    kinds = [who.get(str(k), {}).get("type", "unknown") for k in cuts]
    model = BODY_MODEL.get(max(set(kinds), key=kinds.count), "NEUTRAL")
    smpl = SMPL(fab.paths.body_models_dir() / "smpl" / f"SMPL_{model}.pkl").to(dev)
    C, rep_start = [], {}
    for k in cuts:
        kp = np.load(a.diag / f"kp2d_{a.tag}{k:02d}.npz")["kp2d"]
        J3, start = own_motion(z[f"s{k}_body"], a.diag / f"fab_{a.tag}_shot{k:02d}.npz")
        if ALIGN_CLOSEUP and info["shots"][k].get("points_only_during_cut"):
            J3 = align_closeup(J3, z[f"s{k}_cam"], info["shots"][k]["lens_35mm"], kp, int(z["W"]), int(z["H"]))
            start += ", slid onto the face and shoulders"
        c = Cut(k, J3, z[f"s{k}_cam"], info["shots"][k]["lens_35mm"], kp, int(z["W"]), int(z["H"]), dev)
        if start == "scene" and c.start_from_gvhmr(a.diag / f"gvhmr_{a.tag}{k:02d}" / "gvhmr_skeleton.npz"):
            start = "gvhmr joint rotations"
        rep_start[k] = start
        c.Jcur = torch.tensor(z[f"s{k}_body"][: c.T], dtype=torch.float32, device=dev)  # the "current" candidate
        C.append(c)
    kw = torch.tensor(KP_W, device=dev)
    betas = torch.zeros(1, 10, device=dev, requires_grad=True)

    # the shape comes from the wide cuts (whole body seen); close-ups use it as it is (their few points would push
    # it anywhere). Joints show bone lengths, not girth: every shape coefficient but the first (size) is held near 0.
    closeup = {c.k: bool(info["shots"][c.k].get("points_only_during_cut")) for c in C}
    if all(closeup.values()):
        closeup = {k: False for k in closeup}
    beta_w = torch.full((10,), W_BETA_GIRTH, device=dev)
    beta_w[0] = W_BETA

    def run(c):
        b = betas.detach() if closeup[c.k] else betas
        return smpl(c.go, c.bp, b.expand(c.T, -1), transl=c.tr)

    rep = {"model": model, "cuts": {}, "start": rep_start}
    with torch.no_grad():
        for c in C:  # before: the scene's current joints (faces: the head joint stands in)
            P0 = torch.zeros(c.T, 17, 3, device=dev)
            for sj, cc in J2C:
                P0[:, cc] = c.Jcur[:, sj]
            P0[:, :5] = c.Jcur[:, 15:16]
            rep["cuts"][c.k] = {"before_heldout": c.px_error(P0, ~c.fit_mask), "before_fit": c.px_error(P0, c.fit_mask)}
    params = [betas] + [p for c in C for p in (c.go, c.bp, c.tr)]
    # stage 1: to the current 3D joints (a start); stage 2: to the picture
    for stage, iters in ((1, 200), (2, a.iters)):
        opt = torch.optim.Adam(params, lr=0.02)
        for it in range(iters):
            loss = (beta_w * betas[0] ** 2).sum() / 10
            for c in C:
                verts, joints = run(c)
                l3 = ((joints[:, :22] - c.J3) ** 2).sum(-1).mean()
                if stage == 1:
                    loss = loss + l3
                else:
                    uv, _ = c.project(c.coco(verts, joints))
                    r = (uv - c.kp[..., :2]) / c.w  # in picture widths
                    wgt = c.kp[..., 2] * kw * c.fit_mask
                    rho = (r ** 2).sum(-1)
                    rho = rho / (rho + 0.02 ** 2)  # robust (Geman-McClure, ~2 % of the width)
                    loss = loss + W_2D * (wgt * rho).sum() / wgt.sum().clamp(min=1) + W_3D * 0.1 * l3
                if c.T > 2:
                    loss = loss + W_SMOOTH * ((c.bp[2:] - 2 * c.bp[1:-1] + c.bp[:-2]) ** 2).mean() \
                        + W_SMOOTH * ((c.go[2:] - 2 * c.go[1:-1] + c.go[:-2]) ** 2).mean() \
                        + W_TR * ((c.tr[2:] - 2 * c.tr[1:-1] + c.tr[:-2]) ** 2).sum(-1).mean()
                loss = loss + W_POSE * (c.bp ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        print("stage", stage, "loss", round(float(loss), 5), flush=True)
    with torch.no_grad():
        for c in C:
            verts, joints = run(c)
            P = c.coco(verts, joints)
            r = rep["cuts"][c.k]
            r["after_heldout"], r["after_fit"] = c.px_error(P, ~c.fit_mask), c.px_error(P, c.fit_mask)
            r["frames"] = c.T
            # candidates: "current" (the scene's joints) and "body" (this fit); decided on the held-out points only
            b, f = r["before_heldout"], r["after_heldout"]
            parts = [q for q in b if q not in ("all", "n")]
            verdict = {}
            for q in parts:
                if b[q] is None or f[q] is None:
                    verdict[q] = "no evidence"
                elif f[q] > b[q] + WORSE_PX:
                    verdict[q] = "worse"
                elif f[q] < b[q] - WORSE_PX:
                    verdict[q] = "better"
                else:
                    verdict[q] = "same"
            r["parts"] = verdict
            enough = int((c.kp[..., 2] > 0).sum()) >= MIN_POINTS
            accept = enough and b["all"] is not None and f["all"] is not None and f["all"] < b["all"]
            if not enough:
                r["evidence"] = "too little: the picture shows too few body points to choose (current body kept)"
            r["choice"] = "body" if accept else "current"
            # what the picture does not show stays an open question for later steps
            seen = (c.kp[..., 2] > 0).float().mean(0).cpu().numpy()
            r["unobserved"] = sorted({PART[i] for i in range(17) if seen[i] < SEEN_SHARE})
            if verdict and any(v == "worse" for v in verdict.values()):
                r["unobserved"] = sorted(set(r["unobserved"]) | {q for q, v in verdict.items() if v == "worse"})
            r["uncertain"] = (["camera distance (close-up: moving the camera or the person gives the same picture)"]
                              if info["shots"][c.k].get("points_only_during_cut") else [])
            if accept:
                z[f"s{c.k}_body"] = joints[:, :22].cpu().numpy().astype(np.float32)
                z[f"s{c.k}_smpl"] = verts.cpu().numpy().astype(np.float16)
                # the rig's animation for this cut (scene/multishot._person_rig): SMPL axis-angle per joint, root shift
                z[f"s{c.k}_pose"] = torch.cat([c.go, c.bp], 1).cpu().numpy().astype(np.float32)
                z[f"s{c.k}_transl"] = c.tr.cpu().numpy().astype(np.float32)
                z[f"s{c.k}_person"] = np.array(a.identity)
            print(c.k, json.dumps(r), flush=True)
    z["smpl_faces"] = smpl.faces.cpu().numpy().astype(np.int32)
    with torch.no_grad():  # the person's one body: rest surface, rest joints, skin weights (one rig for all cuts)
        vs = smpl.v_template + torch.einsum("l,vcl->vc", betas[0, : smpl.shapedirs.shape[-1]], smpl.shapedirs)
        z[f"person_{a.identity}_rest_verts"] = vs.cpu().numpy().astype(np.float32)
        z[f"person_{a.identity}_rest_joints"] = (smpl.J_regressor @ vs).cpu().numpy().astype(np.float32)
        z[f"person_{a.identity}_weights"] = smpl.weights.cpu().numpy().astype(np.float32)
        z["smpl_parents"] = np.asarray(smpl.parents.cpu() if torch.is_tensor(smpl.parents) else smpl.parents, np.int32)
    rep["betas"] = betas.detach().cpu().numpy().ravel().round(2).tolist()
    info.setdefault("fit_person", {})[a.identity] = rep
    z["info"] = json.dumps(info)
    np.savez_compressed(a.out, **z)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
