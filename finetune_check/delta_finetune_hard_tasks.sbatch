#!/bin/bash
# Short fine-tune of the released LIBERO checkpoint on the hard-task subset, one 8x H200 node.
# NOT yet run on Delta: written from the Delta docs and checked locally only as far as the
# Hydra overrides (they compose) and the pieces listed in the plan. Edit the two CHANGE_ME values.
#
#   sbatch finetune_check/delta_finetune_hard_tasks.sbatch                 # 1000 steps
#   MAX_STEPS=30 sbatch finetune_check/delta_finetune_hard_tasks.sbatch    # quick end-to-end run
#
#SBATCH --job-name=fastwam-ft-hard
#SBATCH --account=CHANGE_ME-delta-gpu
#SBATCH --partition=gpuH200x8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --gpus-per-node=8
#SBATCH --mem=800G
#SBATCH --constraint="work"
#SBATCH --time=04:00:00
#SBATCH --no-requeue
#SBATCH --output=slurm-%j.out

set -eo pipefail

ROOT=/work/hdd/CHANGE_ME/$USER/fastwam      # holds: FastWAM/ (repo), envs/fastwam, ckpt/, data/
MAX_STEPS=${MAX_STEPS:-1000}
LR=${LR:-1e-5}

CKPT=$ROOT/ckpt/libero_uncond_2cam224.pt
STATS=$ROOT/ckpt/libero_uncond_2cam224_dataset_stats.json
SUBSET=$ROOT/data/libero_hard_subset
TEXT_CACHE=$ROOT/data/text_embeds_cache/libero

module reset
source "$ROOT/miniforge3/etc/profile.d/conda.sh"
conda activate "$ROOT/envs/fastwam"
cd "$ROOT/FastWAM"

# The frozen VAE is read from $DIFFSYNTH_MODEL_BASE_PATH/DiffSynth-Studio/Wan-Series-Converted-Safetensors/.
export DIFFSYNTH_MODEL_BASE_PATH=$ROOT/ckpt
export DIFFSYNTH_SKIP_DOWNLOAD=true

echo "node: $(hostname)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'gpus', torch.cuda.device_count())"

# Without this fix the loaded checkpoint is lost after the first optimizer step (see resume_pt_fix.patch).
grep -q "refresh_fp32_params" src/fastwam/trainer.py || { echo "Apply finetune_check/resume_pt_fix.patch first"; exit 1; }
for f in "$CKPT" "$STATS" "$SUBSET/libero_10/meta/info.json" "$DIFFSYNTH_MODEL_BASE_PATH/DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.2_VAE.safetensors"; do
  [ -e "$f" ] || { echo "missing: $f"; exit 1; }
done

# What to look for in the log:
#   steps 10-30: loss about 0.08 (loss_video about 0.07, loss_action about 0.006). A loss near 1 or
#                higher means the checkpoint weights were not kept or a setting does not match.
#   speed line:  samples/s and ETA.
bash scripts/train_zero1.sh 8 task=libero_uncond_2cam224_1e-4 \
  "resume=$CKPT" \
  model.skip_dit_load_from_pretrain=true \
  model.action_dit_pretrained_path=null \
  model.action_scheduler.train_shift=5.0 \
  model.action_scheduler.infer_shift=5.0 \
  "data.train.dataset_dirs=[$SUBSET/libero_10,$SUBSET/libero_goal,$SUBSET/libero_spatial]" \
  "+data.train.pretrained_norm_stats=$STATS" \
  "data.train.text_embedding_cache_dir=$TEXT_CACHE" \
  "learning_rate=$LR" \
  "max_steps=$MAX_STEPS" \
  save_every=0 eval_every=100 log_every=10

# The run ends by writing weights (12 GB, .../checkpoints/weights/step_*.pt) and a full training
# state (.../checkpoints/state/step_*/, roughly 85 GB by estimate). The state is only needed to
# continue this run; delete it once the weights file is copied.
