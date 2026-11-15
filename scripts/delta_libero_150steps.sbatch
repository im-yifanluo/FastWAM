#!/bin/bash
# Verification run on NCSA Delta: the official LIBERO training from the Wan2.2 base, stopped after
# 150 steps (one 8x H200 node, 1 hour limit). Not yet run on Delta.
# Pass: the logged loss starts near 1.7 and falls steadily.
#
# Submit from the repo root:
#   sbatch --account=<project>-delta-gpu scripts/delta_libero_150steps.sbatch
#
# Must be in place first (official default locations, relative to the repo root):
#   ../fastwam-env/        conda env from the README, plus conda-forge "ffmpeg>=7,<8"
#                          (or set FASTWAM_ENV=<env dir> when submitting)
#   checkpoints/Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model*.safetensors
#   checkpoints/DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.2_VAE.safetensors
#   data/libero_mujoco3.3.2/libero_{spatial,object,goal,10}_no_noops_lerobot/
#   data/text_embeds_cache/libero/
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
export PATH="${FASTWAM_ENV:-$PWD/../fastwam-env}/bin:$PATH"
export DIFFSYNTH_SKIP_DOWNLOAD=true  # stop if a weight file is missing instead of downloading on GPU time

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
ls -d checkpoints/Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model*.safetensors \
      checkpoints/DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.2_VAE.safetensors \
      data/libero_mujoco3.3.2/libero_*_no_noops_lerobot data/text_embeds_cache/libero

# README "Model Preparation", step 2: build the starting action weights from Wan2.2 (once).
ACTION_INIT=checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt
[ -e "$ACTION_INIT" ] || python scripts/preprocess_action_dit_backbone.py \
  --model-config configs/model/fastwam.yaml --output "$ACTION_INIT" --device cuda --dtype bfloat16

# README "Training": the official LIBERO command, limited to 150 steps.
bash scripts/train_zero1.sh 8 task=libero_uncond_2cam224_1e-4 max_steps=150
