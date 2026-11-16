#!/bin/bash
# Verification run on NCSA Delta: the official LIBERO training from the Wan2.2 base, stopped after
# 150 steps (one 8x H200 node, 1 hour limit). Not yet run on Delta.
# Pass: the logged loss starts near 1.7 and falls steadily.
#
# Submit from the repo root:
#   sbatch --account=<project>-delta-gpu scripts/delta_libero_150steps.sbatch
#
# Shorter wait: 4 GPUs on the interactive H200 partition (1 hour cap, double rate, so the same cost
# per hour as 8 GPUs here). Training uses however many GPUs the job was given:
#   sbatch --account=<project>-delta-gpu --partition=gpuH200x8-interactive \
#          --gpus-per-node=4 --cpus-per-task=48 --mem=400G scripts/delta_libero_150steps.sbatch
#
# Must be in place first (official default locations, relative to the repo root):
#   ../fastwam-env/        conda env from the README, plus conda-forge "ffmpeg>=7,<8"
#                          (or set FASTWAM_ENV=<env dir> when submitting)
#   checkpoints/Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model*.safetensors            video model
#   checkpoints/DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.2_VAE.safetensors
#   checkpoints/DiffSynth-Studio/Wan-Series-Converted-Safetensors/models_t5_umt5-xxl-enc-bf16.safetensors
#   checkpoints/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl/                               tokenizer
#   data/libero_mujoco3.3.2/libero_{spatial,object,goal,10}_no_noops_lerobot/
# The text encoder and tokenizer are only used by the two preparation commands below.
#
#SBATCH --job-name=fastwam-libero-150
#SBATCH --partition=gpuH200x8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --gpus-per-node=8
#SBATCH --mem=800G
#SBATCH --constraint="work"
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=slurm-%j.out

set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
ENV_DIR="${FASTWAM_ENV:-$PWD/../fastwam-env}"
export PATH="$ENV_DIR/bin:$PATH"
export DIFFSYNTH_SKIP_DOWNLOAD=true  # stop if a weight file is missing instead of downloading on GPU time

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)  # the GPUs this job was given
echo "[gpus] training on $NUM_GPUS GPUs"
ls -dcheckpoints/Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model*.safetensors \
      checkpoints/DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.2_VAE.safetensors \
      checkpoints/DiffSynth-Studio/Wan-Series-Converted-Safetensors/models_t5_umt5-xxl-enc-bf16.safetensors \
      checkpoints/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl \
      data/libero_mujoco3.3.2/libero_*_no_noops_lerobot

# Read the environment, weights and dataset once so the commands below get them from memory.
# Measured on Delta's /work/hdd on 2026-11-15: loading the 11 GB text encoder cold took 12 minutes;
# after reading the files once like this, a script that loads 33 GB of weights took under 2 minutes.
# The text encoder is left out: only the two preparation commands use it, and they are normally
# already done (if they are not, they still work, only slowly).
echo "[read-once] start $(date +%T)"
find "$ENV_DIR" checkpoints data/libero_mujoco3.3.2 data/text_embeds_cache -type f \
  ! -name '*.tar.gz' ! -name 'models_t5_*' -print0 | xargs -0 -P 48 -n 16 cat > /dev/null || true
echo "[read-once] end $(date +%T)"

# README "Model Preparation", step 2: build the starting action weights from Wan2.2 (skipped if present).
ACTION_INIT=checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt
[ -e "$ACTION_INIT" ] || python scripts/preprocess_action_dit_backbone.py \
  --model-config configs/model/fastwam.yaml --output "$ACTION_INIT" --device cuda --dtype bfloat16

# README "Training", step 1: cache the text embeddings of the task instructions (skipped if present).
[ -d data/text_embeds_cache/libero ] || python scripts/precompute_text_embeds.py task=libero_uncond_2cam224_1e-4

# README "Training", step 2: the official LIBERO command, limited to 150 steps.
bash scripts/train_zero1.sh "$NUM_GPUS" task=libero_uncond_2cam224_1e-4 max_steps=150
