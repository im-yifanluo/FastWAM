#!/usr/bin/env bash
# Miniature end-to-end run of the REAL training script on the hard-task subset.
#
# The model is shrunk to a few million parameters so it fits a 16 GB desktop GPU; everything
# else is the real thing: scripts/train.py, accelerate + DeepSpeed ZeRO-1 in bf16, the LIBERO
# subset, the released normalisation stats, the cached text embeddings, the in-training eval and
# the checkpoint writer. It then resumes from the weights file it wrote, twice:
#   B) with the repo's code as is, C) with a patched copy of the code (resume_pt_fix.patch).
# and reports how far each run's weights ended up from the checkpoint it resumed from.
#
# Usage (from the repo root, fastwam env active):
#   DIFFSYNTH_MODEL_BASE_PATH=<dir with the Wan VAE> \
#   bash finetune_check/mini_end_to_end.sh <work dir without spaces> <patched code dir> <stats.json>
set -eo pipefail

WORK="${1:?work dir}"; PATCHED="${2:?dir holding patched src/ scripts/ configs/}"; STATS="${3:?dataset_stats.json}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$WORK"
ln -sfn "$REPO/data" "$WORK/data"          # Hydra overrides are simpler without spaces in paths
cp "$STATS" "$WORK/stats.json"
D="$WORK/data/libero_hard_subset"

TINY=(
  model.video_dit_config.hidden_dim=192 model.video_dit_config.ffn_dim=384
  model.video_dit_config.num_heads=4 model.video_dit_config.attn_head_dim=48 model.video_dit_config.num_layers=2
  model.action_dit_config.hidden_dim=96 model.action_dit_config.ffn_dim=192
  model.action_dit_config.num_heads=4 model.action_dit_config.attn_head_dim=48 model.action_dit_config.num_layers=2
)
COMMON=(
  task=libero_uncond_2cam224_1e-4
  model.skip_dit_load_from_pretrain=true model.action_dit_pretrained_path=null
  model.action_scheduler.train_shift=5.0 model.action_scheduler.infer_shift=5.0
  "data.train.dataset_dirs=[$D/libero_10,$D/libero_goal,$D/libero_spatial]"
  "+data.train.pretrained_norm_stats=$WORK/stats.json"
  "data.train.text_embedding_cache_dir=$WORK/data/text_embeds_cache/libero"
  batch_size=2 num_workers=2 learning_rate=1e-3 save_every=0
)

launch() {  # launch <code dir> <run name> <extra overrides...>
  local code="$1" name="$2"; shift 2
  ( cd "$code" && PYTHONPATH="$code/src:${PYTHONPATH:-}" accelerate launch \
      --config_file scripts/accelerate_configs/accelerate_zero1_ds.yaml --num_processes 1 \
      scripts/train.py "${COMMON[@]}" "${TINY[@]}" "output_dir=$WORK/$name" "$@" ) > "$WORK/$name.log" 2>&1 \
    || { echo "run $name FAILED; last lines:"; tail -25 "$WORK/$name.log"; exit 1; }
  echo "run $name finished"
  # The logger drops the "[train]"/"[eval]" tags, so match on the fields instead.
  { grep -a "epoch=.*step=\|val_loss=\|Refreshed DeepSpeed\|Loading weight" "$WORK/$name.log" || true; } \
    | sed 's/  */ /g; s/ trainer.py:[0-9]*//' | cut -c1-200 | tail -8
}

rm -rf "$WORK/A" "$WORK/B" "$WORK/C"
echo "=== A: train the miniature model from scratch (repo code) ==="
launch "$REPO" A max_steps=40 eval_every=20 log_every=10
CKPT="$WORK/A/checkpoints/weights/step_000040.pt"

echo "=== B: resume from A's weights file, repo code as is ==="
launch "$REPO" B "resume=$CKPT" max_steps=3 eval_every=0 log_every=1
echo "=== C: resume from A's weights file, patched code ==="
launch "$PATCHED" C "resume=$CKPT" max_steps=3 eval_every=0 log_every=1

python - "$CKPT" "$WORK/B/checkpoints/weights/step_000003.pt" "$WORK/C/checkpoints/weights/step_000003.pt" <<'PY'
import sys, torch
def flat(path):
    p = torch.load(path, map_location="cpu")
    return torch.cat([v.float().flatten() for _, v in sorted(p["mot"].items())])
a, b, c = (flat(p) for p in sys.argv[1:4])
rel = lambda x: float((x - a).norm() / a.norm())
print("RESULT relative distance from the resumed checkpoint after 3 steps:")
print(f"RESULT   repo as is : {rel(b):.3f}   (about 1 or more means unrelated weights, i.e. the checkpoint was lost)")
print(f"RESULT   patched    : {rel(c):.3f}   (small means training continued from the checkpoint)")
PY
