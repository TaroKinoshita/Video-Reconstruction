#!/usr/bin/env bash
# GVHMR (human motion, world-grounded) in WSL Ubuntu, isolated from any other conda install.
#   wsl -d Ubuntu -- bash /mnt/c/.../gpu/observe/setup_gvhmr.sh
# Layout: ~/vrecon/bin/micromamba, ~/vrecon/mamba (envs), ~/vrecon/GVHMR (code + inputs/checkpoints)
# Licenses: GVHMR code/weights educational, research and nonprofit only; SMPL/SMPL-X need the user's own
# registration (not downloaded here). pytorch3d is a Linux-only wheel, hence WSL.
set -euo pipefail
ROOT=~/vrecon
mkdir -p "$ROOT/bin"
export MAMBA_ROOT_PREFIX="$ROOT/mamba"
if [ ! -x "$ROOT/bin/micromamba" ]; then
  curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj -C "$ROOT" bin/micromamba
fi
MM="$ROOT/bin/micromamba"
if [ ! -d "$ROOT/GVHMR" ]; then
  git clone https://github.com/zju3dv/GVHMR "$ROOT/GVHMR"
fi
cd "$ROOT/GVHMR"
git log -1 --format='GVHMR commit %H %cd'
if [ ! -d "$MAMBA_ROOT_PREFIX/envs/gvhmr" ]; then
  "$MM" create -y -n gvhmr -c conda-forge python=3.10
fi
# pytorch3d: the pinned wheel's host (dl.fbaipublicfiles.com) served an expired certificate on 2026-11-12, so it
# is built from the GitHub tag instead, CPU-only (GVHMR's inference uses its rotation/geometry code in Python;
# the CUDA rasteriser is only for the demo's rendering)
# chumpy's setup.py imports pip, which an isolated build env lacks: install it without isolation first
grep -v -e '^pytorch3d @' -e '^chumpy' requirements.txt > /tmp/gvhmr_req.txt
"$MM" run -n gvhmr pip install "numpy==1.23.5" "setuptools>=68.0" wheel
"$MM" run -n gvhmr pip install --no-build-isolation chumpy
"$MM" run -n gvhmr pip install -r /tmp/gvhmr_req.txt
if ! "$MM" run -n gvhmr python -c "import pytorch3d" 2>/dev/null; then
  "$MM" install -y -n gvhmr -c conda-forge "gxx_linux-64=11.*"
  FORCE_CUDA=0 "$MM" run -n gvhmr pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.6"
fi
"$MM" run -n gvhmr pip install -e .
"$MM" run -n gvhmr pip install gdown
mkdir -p inputs/checkpoints/{body_models/smpl,body_models/smplx,dpvo,gvhmr,hmr2,vitpose,yolo} outputs
"$MM" run -n gvhmr python -c "import torch, pytorch3d; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
echo SETUP_DONE
