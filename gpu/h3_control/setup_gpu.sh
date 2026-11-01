#!/usr/bin/env bash
# Set up a GPU machine (e.g. a RunPod pod) for phase-1a generation with MiniMax-H3 + Fun ControlNet Union 2.0.
# Run from the repository root:   bash gpu/h3_control/setup_gpu.sh [--skip-models] [--source hf|modelscope]
#
# Needs: NVIDIA GPU with 80 GB (A100/H100 80GB), plenty of system RAM for offloading, and roughly
# 250 GB of disk for the weights (see README.md). Safe to rerun.
set -euo pipefail

VIDEOX_COMMIT="e8c314f"   # the commit run_jobs.py was written against
SOURCE="hf"
SKIP_MODELS=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-models) SKIP_MODELS=1 ;;
    --source) SOURCE="$2"; shift ;;
    *) echo "unknown option $1"; exit 2 ;;
  esac
  shift
done

log() { printf '\n[gpu-setup] %s\n' "$*"; }
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || { echo "no NVIDIA GPU found"; exit 1; }
free -g | head -2
df -h . | tail -1

# 1. VideoX-Fun at a pinned commit -------------------------------------------------------------
mkdir -p third_party
if [[ ! -d third_party/VideoX-Fun/.git ]]; then
  log "Cloning VideoX-Fun"
  git clone https://github.com/aigc-apps/VideoX-Fun third_party/VideoX-Fun
fi
git -C third_party/VideoX-Fun fetch -q origin
git -C third_party/VideoX-Fun checkout -q "$VIDEOX_COMMIT" \
  || log "WARNING: commit $VIDEOX_COMMIT not found; staying on $(git -C third_party/VideoX-Fun rev-parse --short HEAD)"

# 2. Python environment (VideoX-Fun README, option 2) ---------------------------------------------
log "Installing Python packages"
python -m pip install -q -r third_party/VideoX-Fun/requirements.txt
python -m pip uninstall -q -y opencv-python opencv-contrib-python || true
python -m pip install -q opencv-python-headless "huggingface_hub[cli]" modelscope

# 3. Weights -----------------------------------------------------------------------------------
MODELS=models/Diffusion_Transformer
if [[ "$SKIP_MODELS" -eq 0 ]]; then
  mkdir -p "$MODELS"
  log "Downloading weights from $SOURCE (large: base model + text encoder + ControlNet)"
  if [[ "$SOURCE" == "hf" ]]; then
    huggingface-cli download MiniMaxAI/MiniMax-H3 --local-dir "$MODELS/MiniMax-H3"
    huggingface-cli download alibaba-pai/MiniMax-H3-Fun-Controlnet-Union-2.0 \
      --local-dir "$MODELS/MiniMax-H3-Fun-Controlnet-Union-2.0"
  else
    modelscope download --model MiniMax/MiniMax-H3 --local_dir "$MODELS/MiniMax-H3"
    modelscope download --model PAI/MiniMax-H3-Fun-Controlnet-Union-2.0 \
      --local_dir "$MODELS/MiniMax-H3-Fun-Controlnet-Union-2.0"
  fi
fi

log "Layout check"
ls "$MODELS/MiniMax-H3" | head -20 || true
ls "$MODELS/MiniMax-H3-Fun-Controlnet-Union-2.0" | head -20 || true
cat <<'EOF'

If MiniMax-H3 holds partitions (e.g. FL2VA/, REF2VA/) instead of transformer/, pass the partition
to run_jobs.py:  --model models/Diffusion_Transformer/MiniMax-H3/FL2VA
(VideoX-Fun converts original partitions while loading.)

Next:
  python gpu/h3_control/run_jobs.py --jobs evaluation/phase1a/jobs.json --out runs/phase1a/generated --dry-run
EOF
