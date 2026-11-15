"""Every constant that was tuned by looking at results (docs/REVIEW_INSTRUCTIONS.md F-17, WP-1 step 7).

One place, so the freeze in evaluation/thresholds.yaml can record a hash of this file: a change here after the
freeze is a change of the method and must be written into thresholds.yaml "changes".

"chosen on" says which data the value was looked at against. "v1 dev" = evaluation/synthetic/syn_01..06 (the
development and, until 2026-11-15, the only clips; their ground truth was used while tuning). Nothing here was
chosen on evaluation/synthetic_v2. Modules import these names; the values were moved here unchanged.
"""

from __future__ import annotations

# --- repair loop: accept rules and search (run/policy.py) -------------------------------------------------------
# Tolerance for camera proposals: may raise the whole silhouette loss by at most this much. The correct focal can
# first raise it by up to +0.024 (injected-error benchmark, syn_06); harmful LLM focal changes raised it by ~+0.1.
CAMERA_GUARD = 0.03  # chosen on: v1 dev (repair benchmark and condition-C runs, 2026-11-10)
FOCAL_PROBE = 0.1  # relative change of focal length used to test the evidence; not tuned (round value)
FOCAL_MIN_RISE = 0.001  # draft; chosen on: v1 dev
MOVE_STEP = 0.035  # first position step, share of camera-to-object distance; chosen on: v1 dev (= 0.25 m at ~7 m)
SIZE_STEP = 0.02  # same for size; chosen on: v1 dev (= 0.15 m at ~7 m)
YAW_STEP = 0.3  # first orientation step, radians; chosen on: v1 dev

# --- loss (optimize/__init__.py) --------------------------------------------------------------------------------
TRACK_WEIGHT = 50.0  # track loss (~0.001 of the diagonal) onto the silhouette scale (~0.1); chosen on: v1 dev

# --- timing hold (run/service.py) -------------------------------------------------------------------------------
HOLD_TOLERANCE = 0.002  # draft: silhouette loss a hold may cost; chosen on: v1 dev

# --- 4D start scene (analysis/from_4d.py) -----------------------------------------------------------------------
MIN_DEPTH_RATIO = 0.5  # hidden side at least this share of the other side (_deepen); chosen on: v1 dev
STILL_PROGRESS = 0.1  # share of travel below which a key counts as still at an end; chosen on: v1 dev
TOP_DOWN = 0.3  # view within ~17 deg of straight down: heading from the image's up; chosen on: one real clip
ROUND_RATIO = 0.8  # draft: footprint sides within this ratio -> no facing; chosen on: v1 dev and one real clip
MIN_FLOOR_POINTS = 500  # draft: fewer floor-mask points -> floor plane not trusted; chosen on: real clips
MOVE_FRACTION = 0.5  # box centres spread more than this share of size -> the object moves; chosen on: v1 dev

# --- importing an outside scene (run/refine.py) -----------------------------------------------------------------
FLAT = 0.02  # thinnest/longest side below this: a plane; chosen on: GPT condition-B scenes of v1 dev

__all__ = [
    "CAMERA_GUARD", "FLAT", "FOCAL_MIN_RISE", "FOCAL_PROBE", "HOLD_TOLERANCE", "MIN_DEPTH_RATIO",
    "MIN_FLOOR_POINTS", "MOVE_FRACTION", "MOVE_STEP", "ROUND_RATIO", "SIZE_STEP", "STILL_PROGRESS", "TOP_DOWN",
    "TRACK_WEIGHT", "YAW_STEP",
]
