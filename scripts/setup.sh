#!/usr/bin/env bash
# Set up the development environment (idempotent: safe to run again).
#
#   bash scripts/setup.sh            # full setup + smoke test
#   bash scripts/setup.sh --no-smoke # skip the render smoke test
#
# What it does:
#   1. Linux only: installs Mesa EGL so Workbench/EEVEE can render without a GPU
#      (Cycles on CPU works without it). Skipped if already present.
#   2. Creates .venv with Python 3.13 (required by bpy 5.2.3) using uv.
#   3. Installs this package with the "blender" and "dev" extras.
#   4. Runs scripts/smoke_render.py to check that Blender can build, render,
#      save and reload a scene.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
RUN_SMOKE=1
[[ "${1:-}" == "--no-smoke" ]] && RUN_SMOKE=0

log() { printf '\n[setup] %s\n' "$*"; }

# 1. Software OpenGL (Linux, no GPU) ------------------------------------------
if [[ "$(uname -s)" == "Linux" ]]; then
  if ldconfig -p 2>/dev/null | grep -q 'libEGL.so.1' \
     && ldconfig -p 2>/dev/null | grep -q 'libEGL_mesa.so.0'; then
    log "Mesa EGL already installed"
  elif command -v apt-get >/dev/null 2>&1; then
    SUDO=""; [[ "$(id -u)" -ne 0 ]] && SUDO="sudo"
    log "Installing Mesa EGL (software OpenGL for Workbench/EEVEE)"
    PKGS="libegl1 libegl-mesa0 libgl1-mesa-dri libgbm1"
    $SUDO apt-get install -y -q $PKGS >/dev/null \
      || { $SUDO apt-get update -q >/dev/null && $SUDO apt-get install -y -q $PKGS >/dev/null; }
  else
    log "WARNING: no apt-get; Workbench/EEVEE may fail. Cycles (CPU) still works."
  fi
fi

# 2. uv + Python 3.13 venv ------------------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
  log "Installing uv"
  python3 -m pip install --user -q uv
  export PATH="$HOME/.local/bin:$PATH"
fi

if [[ -x .venv/bin/python ]] && .venv/bin/python -c 'import sys; sys.exit(sys.version_info[:2] != (3, 13))'; then
  log ".venv (Python 3.13) already exists"
else
  log "Creating .venv with Python 3.13"
  rm -rf .venv
  uv venv --python 3.13 .venv
fi

# 3. Install package ------------------------------------------------------------
log "Installing vrecon[blender,control,mcp,dev] (bpy is ~400 MB on first install)"
uv pip install --python .venv/bin/python -q -e ".[blender,control,mcp,dev]"
.venv/bin/python -c 'import bpy; print("[setup] Blender", bpy.app.version_string)'

# 4. Smoke test -------------------------------------------------------------------
if [[ "$RUN_SMOKE" -eq 1 ]]; then
  log "Running render smoke test"
  .venv/bin/python scripts/smoke_render.py --out runs/smoke
fi

log "Done. Activate with: source .venv/bin/activate"
