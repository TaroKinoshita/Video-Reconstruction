"""The agent's tools on one run folder (MAVE section 3.2: query - edit - verify, then submit).

Run folder:
  run.json        inputs (evidence, people, parts, video, prep), mode, time budget, start time
  scene.json      the current scene: primitives, kept system meshes ("fixed"), people scale k and removals,
                  justifications; saved after every successful edit (revisions/ keeps each one)
  history.jsonl   every tool call with its result summary
  issues.json     the last verification
  renders/        diagnostic pictures

Camera and body poses stay fixed (MAVE eq. 9): the agent edits the environment, one scale k per person (eq. 10)
and removals of whole tracks or segments (each with a reason).
"""
from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import cv2
import numpy as np

from vrecon.agent import checks, prims, render
from vrecon.agent import evidence as evmod

_CACHE: dict = {}


def _j(x):
    if isinstance(x, dict):
        return {k: _j(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_j(v) for v in x]
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, float):
        return round(x, 4)
    return x


class Session:
    def __init__(self, run_dir: str | Path):
        self.dir = Path(run_dir)
        self.run = json.loads((self.dir / "run.json").read_text(encoding="utf-8"))
        key = str(self.dir.resolve())
        if key not in _CACHE:
            _CACHE[key] = self._load()
        self.c = _CACHE[key]

    # ---------- loading
    def _load(self) -> dict:
        r = self.run
        ev = evmod.load(Path(r["evidence"]))
        pz = np.load(r["parts"])
        parts = json.loads(str(pz["parts"]))
        GR, Gt = np.array(parts["world_R"]), np.array(parts["world_t"])
        UR = ev.get("up_R", np.eye(3))  # evidence world = parts world turned so the estimated up is +z
        an = np.load(r["people"])
        people = {}
        for label in json.loads(str(an["labels"])):
            ts = an[f"{label}_t"].astype(int)
            V = ((an[f"{label}_v"].astype(np.float64) @ GR.T + Gt) @ UR.T).astype(np.float32)
            ext = V[:, :, 2].max(1) - V[:, :, 2].min(1)
            people[label] = {"t": ts, "v": V, "height": float(np.percentile(ext, 95))}
        bd = pz["backdrop_v"].astype(np.float64) @ UR.T
        return {"ev": ev, "people": people, "backdrop_v": bd, "backdrop_vox": checks.mesh_voxels(bd),
                "video": None}

    def scene(self) -> dict:
        return json.loads((self.dir / "scene.json").read_text(encoding="utf-8"))

    def _save(self, sc: dict, what: str) -> None:
        rev = self.dir / "revisions"
        rev.mkdir(exist_ok=True)
        n = len(list(rev.glob("scene_*.json")))
        txt = json.dumps(sc, indent=1)
        (rev / f"scene_{n:04d}.json").write_text(txt, encoding="utf-8")
        (self.dir / "scene.json").write_text(txt, encoding="utf-8")
        self._log("save", {"revision": n, "what": what})

    def _log(self, tool: str, data) -> None:
        with open(self.dir / "history.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"t": round(time.time() - self.run["started"], 1), "tool": tool, "data": _j(data)})
                    + "\n")

    def people_now(self, sc: dict) -> dict:
        """Bodies with the current k (eq. 10, about the camera centres) and without removed frames."""
        ev = self.c["ev"]
        out = {}
        for label, P in self.c["people"].items():
            st = sc["people"].get(label, {})
            if st.get("track_removed"):
                continue
            keep = np.ones(len(P["t"]), bool)
            for a, b, *_ in st.get("removed", []):
                keep &= ~((P["t"] >= a) & (P["t"] <= b))
            ts = P["t"][keep]
            o = ev["cam_T"][ts][:, None].astype(np.float32)
            k = np.float32(st.get("k", 1.0))
            out[label] = {"t": ts, "v": o + k * (P["v"][keep] - o), "height": P["height"]}
        return out

    def elapsed_min(self) -> float:
        return (time.time() - self.run["started"]) / 60

    # ---------- query
    def overview(self) -> dict:
        ev, sc = self.c["ev"], self.scene()
        cams = ev["cam_T"]
        by = {}
        for p in sc["primitives"]:
            key = f"{p['type']}/{p.get('class')}" + (" (imagined)" if p.get("imagined") else "")
            by[key] = by.get(key, 0) + 1
        res = {
            "task": self.run.get("task", ""),
            "frames": int(len(cams)), "fps": self.run.get("fps"), "picture_size": [int(v) for v in ev["size"]],
            "keyframes_with_depth": [int(v) for v in ev["kf"]],
            "camera": {"start": cams[0].round(2).tolist(), "end": cams[-1].round(2).tolist(),
                       "path_length_m": round(float(np.linalg.norm(np.diff(cams, axis=0), axis=1).sum()), 2),
                       "height_above_ground_m": round(float(cams[0, 2] - ev["ground"](cams[:1, 0], cams[:1, 1])[0]), 2)},
            "people": {l: {"frames": [int(P["t"][0]), int(P["t"][-1])], "n_frames": int(len(P["t"])),
                           "height_at_k1_m": round(P["height"], 2),
                           **{k: v for k, v in sc["people"].get(l, {}).items()}} for l, P in self.c["people"].items()},
            "evidence_points": {c: int((ev["Q_cls"] == k).sum()) for k, c in enumerate(evmod.CLASSES)},
            "scene": {"primitives": len(sc["primitives"]), "by_type_class": by, "fixed": sc.get("fixed", {})},
            "elapsed_min": round(self.elapsed_min(), 1), "budget_min": self.run.get("budget_min"),
        }
        self._log("overview", {})
        return res

    def structures(self, cls: str | None = None, max_items: int = 30) -> dict:
        """Clusters of all observed static points per class (support / vertical / ceiling), cached."""
        ev = self.c["ev"]
        cache = self.dir / "structures.json"
        if cache.exists():
            allc = json.loads(cache.read_text(encoding="utf-8"))
        else:
            allc = evmod.extract_structures(ev)
            cache.write_text(json.dumps(allc, indent=1), encoding="utf-8")
        out = [c for c in allc if cls is None or c["class"] == cls][:max_items]
        self._log("structures", {"cls": cls, "n": len(out)})
        return {"structures": out, "total": sum(1 for c in allc if cls is None or c["class"] == cls),
                "note": "support: fitted layers with a 2.5 m height grid; vertical / ceiling: plan-view groups "
                "(column = narrow and tall, e.g. a trunk or post; mass = wide, e.g. a wall, a bush or foliage). "
                "Points are a sample of the observed surfaces (only the sides the camera saw); 5-95 % extents."}

    def points_in_box(self, lo: list[float], hi: list[float]) -> dict:
        ev = self.c["ev"]
        X = ev["Q"]
        m = ((X >= np.asarray(lo)) & (X <= np.asarray(hi))).all(1)
        res = {"points": int(m.sum())}
        if m.any():
            P = X[m]
            res.update({"by_class": {c: int((ev["Q_cls"][m] == k).sum()) for k, c in enumerate(evmod.CLASSES)},
                        "median": np.median(P, 0).round(2).tolist(), "p5": np.percentile(P, 5, 0).round(2).tolist(),
                        "p95": np.percentile(P, 95, 0).round(2).tolist()})
        self._log("points_in_box", {"lo": lo, "hi": hi, "n": res["points"]})
        return res

    def ground_height(self, xy: list[list[float]]) -> dict:
        P = np.asarray(xy, float).reshape(-1, 2)
        z = self.c["ev"]["ground"](P[:, 0], P[:, 1])
        return {"heights": [round(float(v), 3) for v in z],
                "note": "the system's fitted ground height field (from observed ground points, extended)"}

    def pixel(self, frame: int, x: float, y: float) -> dict:
        """World point of the observed depth at a video pixel, on the nearest keyframe with depth."""
        ev = self.c["ev"]
        j = int(np.argmin(np.abs(ev["kf"] - frame)))
        f = int(ev["kf"][j])
        step = int(ev["step"])
        u, v = int(x / step), int(y / step)
        h, w = ev["depth"].shape[1:]
        if not (0 <= u < w and 0 <= v < h):
            return {"error": "outside the picture"}
        win = ev["depth"][j, max(v - 1, 0):v + 2, max(u - 1, 0):u + 2]
        d = float(np.min(win)) if np.isfinite(win).any() else None
        if d is None:
            return {"keyframe": f, "depth": None, "note": "no observed static depth here (person or sky)"}
        o, dirs = checks.rays(ev, f, np.array([u], float), np.array([v], float))
        X = o + d * dirs[0]
        return {"keyframe": f, "depth_m": round(d, 2), "world": X.round(2).tolist()}

    def person(self, label: str, every: int = 24) -> dict:
        sc = self.scene()
        P = self.people_now(sc).get(label)
        if P is None:
            return {"error": f"no person {label} (removed or unknown)"}
        rows = []
        for j in range(0, len(P["t"]), every):
            V = P["v"][j]
            low = V[np.argsort(V[:, 2])[:30]].mean(0)
            sup = checks.support_height(self.c["ev"], sc, low[None, :2], low[None, 2])[0]
            rows.append({"frame": int(P["t"][j]), "centre": V.mean(0).round(2).tolist(),
                         "lowest": low.round(2).tolist(), "foot_gap_m": round(float(low[2] - sup), 3),
                         "vertical_extent_m": round(float(V[:, 2].max() - V[:, 2].min()), 2)})
        return {"person": label, "k": sc["people"].get(label, {}).get("k", 1.0), "samples": rows}

    def list_primitives(self, cls: str | None = None, include_imagined: bool = False, near: list[float] | None = None,
                        radius: float = 5.0, max_items: int = 80) -> dict:
        sc = self.scene()
        rows = []
        for p in sc["primitives"]:
            if cls and p.get("class") != cls:
                continue
            if p.get("imagined") and not include_imagined:
                continue
            if near is not None and np.linalg.norm(np.asarray(p["center"][:2]) - np.asarray(near[:2])) > radius:
                continue
            rows.append({k: _j(p[k]) for k in ("name", "type", "class", "center", "size", "rotation") if k in p})
        n_im = sum(1 for p in sc["primitives"] if p.get("imagined"))
        return {"primitives": rows[:max_items], "listed": min(len(rows), max_items), "matching": len(rows),
                "imagined_total": n_im}

    def get_primitive(self, name: str) -> dict:
        for p in self.scene()["primitives"]:
            if p["name"] == name:
                return p
        return {"error": f"no primitive {name}"}

    # ---------- edit
    def edit(self, ops: list[dict]) -> dict:
        """Apply edit operations atomically: add / update / remove primitives, keep or drop system meshes."""
        sc = self.scene()
        new = copy.deepcopy(sc)
        names = {p["name"]: i for i, p in enumerate(new["primitives"])}
        problems = []
        for n, op in enumerate(ops):
            kind = op.get("op")
            if kind == "add":
                for p in op.get("primitives", [op.get("primitive")]):
                    p = dict(p or {})
                    p.setdefault("rotation", [0, 0, 0])
                    p.setdefault("class", "prop")
                    p.setdefault("colour", [0.5, 0.5, 0.5])
                    p.setdefault("imagined", False)
                    pr = prims.check(p)
                    if p.get("name") in names:
                        pr.append(f"name {p.get('name')} exists")
                    if pr:
                        problems.append({"op": n, "name": p.get("name"), "problems": pr})
                        continue
                    names[p["name"]] = len(new["primitives"])
                    new["primitives"].append(p)
            elif kind == "update":
                i = names.get(op.get("name"))
                if i is None:
                    problems.append({"op": n, "problems": [f"no primitive {op.get('name')}"]})
                    continue
                p = dict(new["primitives"][i])
                p.update({k: v for k, v in op.get("set", {}).items() if k != "name"})
                pr = prims.check(p)
                if pr:
                    problems.append({"op": n, "name": p["name"], "problems": pr})
                    continue
                new["primitives"][i] = p
            elif kind == "remove":
                drop = set(op.get("names", []))
                if op.get("class"):
                    drop |= {p["name"] for p in new["primitives"] if p.get("class") == op["class"]
                             and (op.get("imagined") is None or bool(p.get("imagined")) == op["imagined"])}
                missing = drop - set(names)
                if missing:
                    problems.append({"op": n, "problems": [f"no primitive {m}" for m in sorted(missing)][:10]})
                new["primitives"] = [p for p in new["primitives"] if p["name"] not in drop]
                names = {p["name"]: i for i, p in enumerate(new["primitives"])}
            elif kind == "fixed" and self.run.get("mode") == "empty":
                problems.append({"op": n, "problems": ["the system's meshes are not available in this run: build the "
                                                       "environment from primitives"]})
            elif kind == "fixed":
                for k in ("ground", "backdrop", "forest_wall"):
                    if k in op:
                        new.setdefault("fixed", {})[k] = bool(op[k])
            else:
                problems.append({"op": n, "problems": [f"unknown op {kind!r} (add, update, remove, fixed)"]})
        if problems:
            self._log("edit", {"ops": len(ops), "rejected": problems})
            return {"applied": False, "problems": problems, "note": "nothing was changed"}
        self._save(new, f"edit {len(ops)} ops")
        self._log("edit", {"ops": ops})
        return {"applied": True, "primitives": len(new["primitives"])}

    def set_human_scale(self, label: str, mode: str, value: float | None = None, primitive: str | None = None,
                        frames: list[int] | None = None) -> dict:
        """k for one person (eq. 10): mode 'direct' (k = value), 'height' (value = metres), 'support' (feet on
        the ground or on the named primitive; median over frames, or over frames [a, b])."""
        sc = self.scene()
        P = self.c["people"].get(label)
        if P is None:
            return {"error": f"no person {label}"}
        ev = self.c["ev"]
        if mode == "direct":
            k = float(value)
        elif mode == "height":
            k = float(value) / P["height"]
        elif mode == "support":
            sub = copy.deepcopy(sc)
            if primitive:
                sub["primitives"] = [p for p in sc["primitives"] if p["name"] == primitive]
                sub["fixed"] = {"ground": False}
                if not sub["primitives"]:
                    return {"error": f"no primitive {primitive}"}
            sel = np.ones(len(P["t"]), bool) if not frames else (P["t"] >= frames[0]) & (P["t"] <= frames[1])
            ks = []
            for j in np.where(sel)[0][::2]:
                V = P["v"][j]
                low = V[np.argsort(V[:, 2])[:30]].mean(0).astype(np.float64)
                o = ev["cam_T"][P["t"][j]]
                k = 1.0
                for _ in range(6):
                    q = o + k * (low - o)
                    s = checks.support_height(ev, sub, q[None, :2], np.array([q[2] + 5.0]))[0]
                    if not np.isfinite(s) or abs(low[2] - o[2]) < 1e-6:
                        break
                    k = (s - o[2]) / (low[2] - o[2])
                if np.isfinite(k) and 0.3 < k < 3:
                    ks.append(k)
            if not ks:
                return {"error": "no frame where the feet are over that support"}
            k = float(np.median(ks))
        else:
            return {"error": "mode must be direct, height or support"}
        if not 0.3 < k < 3:
            return {"error": f"k {k:.3f} out of range (0.3-3)"}
        sc["people"].setdefault(label, {})["k"] = round(k, 4)
        self._save(sc, f"k {label} {k:.4f} ({mode})")
        return {"person": label, "k": round(k, 4), "height_m": round(k * P["height"], 2)}

    def remove_track(self, label: str, reason: str, frames: list[int] | None = None) -> dict:
        sc = self.scene()
        if label not in self.c["people"]:
            return {"error": f"no person {label}"}
        if not reason:
            return {"error": "a reason is required"}
        st = sc["people"].setdefault(label, {})
        if frames:
            st.setdefault("removed", []).append([int(frames[0]), int(frames[1]), reason])
        else:
            st["track_removed"] = reason
        self._save(sc, f"remove {label} {frames or 'all'}")
        return {"person": label, "removed": frames or "whole track"}

    def restore_track(self, label: str) -> dict:
        sc = self.scene()
        st = sc["people"].setdefault(label, {})
        st.pop("track_removed", None)
        st["removed"] = []
        self._save(sc, f"restore {label}")
        return {"person": label, "restored": True}

    # ---------- verify
    def verify(self, max_issues: int = 40) -> dict:
        sc = self.scene()
        res = checks.verify(self.c["ev"], sc, self.people_now(sc),
                            self.c["backdrop_vox"] if sc.get("fixed", {}).get("backdrop") else None)
        (self.dir / "issues.json").write_text(json.dumps(_j(res), indent=1), encoding="utf-8")
        self._log("verify", {"counts": res["counts"], "open": res["open"]})
        out = dict(res)
        out["issues"] = [{k: v for k, v in i.items()} for i in res["issues"] if not i["justified"]][:max_issues]
        out["justified"] = sum(1 for i in res["issues"] if i["justified"])
        return _j(out)

    def justify(self, issue_id: str, reason: str) -> dict:
        if not reason:
            return {"error": "a reason is required"}
        last = self.dir / "issues.json"
        if last.exists():
            for i in json.loads(last.read_text(encoding="utf-8")).get("issues", []):
                if i["id"] == issue_id and not i.get("justifiable", True):
                    return {"error": f"{issue_id} cannot be justified: fix it with an edit (or set_human_scale)"}
        sc = self.scene()
        sc.setdefault("justified", {})[issue_id] = reason
        self._save(sc, f"justify {issue_id}")
        return {"justified": issue_id}

    # ---------- render
    def render(self, view: str = "source", frame: int = 0, azimuth: float = 45.0, elevation: float = 35.0,
               distance: float | None = None, target: list[float] | None = None, highlight: list[str] | None = None,
               hide: list[str] | None = None, width: int = 960) -> dict:
        ev, sc = self.c["ev"], self.scene()
        W, H = (int(v) for v in ev["size"])
        frame = int(np.clip(frame, 0, len(ev["cam_T"]) - 1))
        ppl = self.people_now(sc)
        pts = []
        for label in self.c["people"]:
            P = ppl.get(label)
            if P is None or frame not in set(P["t"].tolist()):
                pts.append(None)
                continue
            pts.append(P["v"][int(np.where(P["t"] == frame)[0][0])][::4])
        ground = ev["ground"] if sc.get("fixed", {}).get("ground") else None
        bd = self.c["backdrop_v"][::3] if sc.get("fixed", {}).get("backdrop") else None
        if view == "source":
            w = width // 2
            h = int(round(w * H / W))
            v = render.View(ev["cam_R"][frame], ev["cam_T"][frame], float(ev["f"]) * w / W, w, h)
            img = render.draw(v, sc, ground, pts, bd, label_names=highlight or (), hide=tuple(hide or ()))
            film = self._film(frame, w, h)
            img = np.concatenate([film, img], 1) if film is not None else img
        else:
            h = int(width * 0.6)
            cams = ev["cam_T"]
            tgt = np.asarray(target, float) if target is not None else cams.mean(0) + ev["cam_R"][0][:, 2] * 6
            if view == "top":
                dist = distance or 30.0
                R = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1.0]])
                v = render.View(R, tgt + [0, 0, 60], 1.0, width, h, ortho=2 * dist / width)
            else:
                dist = distance or 18.0
                a, e = np.radians(azimuth), np.radians(elevation)
                eye = tgt + dist * np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
                v = render.View(render.look_at(eye, tgt), eye, width * 0.8, width, h)
            img = render.draw(v, sc, ground, pts, bd, cam_path=cams, cam_now=cams[frame], label_names=highlight or (),
                              hide=tuple(hide or ()))
        rd = self.dir / "renders"
        rd.mkdir(exist_ok=True)
        n = len(list(rd.glob("*.png")))
        path = rd / f"{n:03d}_{view}_f{frame:04d}.png"
        ok, buf = cv2.imencode(".png", img[:, :, ::-1])
        path.write_bytes(buf.tobytes())
        self._log("render", {"view": view, "frame": frame, "path": path.name})
        scale = (W / (width // 2)) if view == "source" else None
        return {"path": str(path), "note": (f"left: the film; right: the scene from the shot's camera; a point (x, y) "
                                            f"in either half is video pixel ({scale:.2f} x, {scale:.2f} y) for "
                                            f"pixel_to_world" if view == "source"
                                            else "yellow: camera path, dot: camera at this frame") +
                "; colours: brown trunks/posts, green foliage, grey other, coloured dots people"}

    def _film(self, frame: int, w: int, h: int):
        if self.c["video"] is None and self.run.get("video"):
            from vrecon.control.core import read_video_rgb
            self.c["video"] = read_video_rgb(self.run["video"])
        vid = self.c["video"]
        if vid is None or frame >= len(vid):
            return None
        return cv2.resize(vid[frame], (w, h), interpolation=cv2.INTER_AREA)

    # ---------- submit
    def submit(self, note: str = "") -> dict:
        res = self.verify(max_issues=200)
        sc = self.scene()
        unexplained = [i["id"] for i in res["issues"]]
        for label, st in sc["people"].items():
            for seg in st.get("removed", []):
                if len(seg) < 3 or not seg[2]:
                    unexplained.append(f"removal without reason: {label} {seg[:2]}")
        if unexplained and self.elapsed_min() < self.run.get("budget_min", {}).get("limit", 30):
            self._log("submit", {"accepted": False, "open": len(unexplained)})
            return {"accepted": False, "open_issues": unexplained[:40],
                    "note": "resolve each issue by an edit, or explain it with justify(issue_id, reason)"}
        final = {"scene": sc, "note": note, "open_issues": unexplained, "elapsed_min": round(self.elapsed_min(), 1)}
        (self.dir / "final.json").write_text(json.dumps(final, indent=1), encoding="utf-8")
        self._log("submit", {"accepted": True, "open": len(unexplained)})
        return {"accepted": True, "open_issues_recorded": len(unexplained)}


def init_run(run_dir: str | Path, evidence: str, people: str, parts: str, video: str | None, mode: str,
             task: str = "", fps: float | None = None, target_min: float = 15, limit_min: float = 30) -> dict:
    """mode 'empty': cameras and people only, as MAVE eq. 7; 'system': start from the system's parts."""
    d = Path(run_dir)
    d.mkdir(parents=True, exist_ok=True)
    pz = np.load(parts)
    an = np.load(people)
    labels = json.loads(str(an["labels"]))
    if mode == "system":
        sc = {"primitives": prims.from_parts(json.loads(str(pz["parts"]))),
              "fixed": {"ground": True, "backdrop": True, "forest_wall": True}}
    else:
        sc = {"primitives": [], "fixed": {"ground": False, "backdrop": False, "forest_wall": False}}
    sc["people"] = {l: {"k": 1.0, "removed": []} for l in labels}
    sc["justified"] = {}
    ab = lambda x: str(Path(x).resolve()) if x else None  # noqa: E731  (the tool server runs in the run folder)
    run = {"evidence": ab(evidence), "people": ab(people), "parts": ab(parts), "video": ab(video), "mode": mode,
           "task": task, "fps": fps, "budget_min": {"target": target_min, "limit": limit_min}, "started": time.time()}
    (d / "run.json").write_text(json.dumps(run, indent=1), encoding="utf-8")
    (d / "scene.json").write_text(json.dumps(sc, indent=1), encoding="utf-8")
    (d / "revisions").mkdir(exist_ok=True)
    (d / "revisions" / "scene_0000.json").write_text(json.dumps(sc, indent=1), encoding="utf-8")
    return {"run": str(d), "mode": mode, "primitives": len(sc["primitives"])}
