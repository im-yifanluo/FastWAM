# Short fine-tune check: released LIBERO checkpoint on a few hard tasks

Written overnight on 2026-11-14. Nothing here has run on Delta. The Delta commands are built from
the Delta docs and from what worked on the desktop; treat them as untested until the first job.

## Read this first: `resume=<file.pt>` is broken as shipped

Fine-tuning from a FastWAM weights file uses `resume=<file.pt>`. As shipped, the loaded weights
are lost at the first optimizer step: DeepSpeed keeps its own fp32 copy of the weights, made
before the file is loaded, and copies it back over the model. The run then continues from
whatever the model was initialised with (Wan2.2, or random), without an error.

- Fix: `finetune_check/resume_pt_fix.patch` (8 lines in `src/fastwam/trainer.py`). **Not applied**;
  apply it with `git apply finetune_check/resume_pt_fix.patch`.
- Resuming a full training state directory (`resume=<.../checkpoints/state/step_*>`) is a
  different code path and is not affected.

Evidence, all on the desktop with the real launcher config (DeepSpeed ZeRO-1, bf16, one process):

| Test | Repo as is | With the patch |
|---|---|---|
| Real trainer class, stand-in model whose checkpoint has zero loss (`trainer_resume_check.py`) | loss 0.00008 at step 1, then 11 to 19 | loss stays at 0.00005 to 0.0001 |
| Real `scripts/train.py`, miniature FastWAM on the LIBERO subset (`mini_end_to_end.sh`) | weights end 1.00 away from the checkpoint (relative) | 0.027 away |

## What else was checked locally

| Check | Result |
|---|---|
| Model built from the fine-tune config (no Wan2.2 DiT needed) and the released checkpoint | all 6.021B trainable weights matched, no missing or unexpected keys |
| Released checkpoint's training loss on 256 random LIBERO samples, action shift 5.0, released stats | video 0.074, action 0.0057 (authors quote about 0.07 and 0.01 as final values) |
| Same, with each video paired with another sample's actions (control) | action 1.29, so the number does react to a mismatch |
| Same, on the 7 hard tasks only (128 samples) | video 0.068, action 0.0040 |
| Task-subset tool | 7 tasks, 267 demos, 52,591 frames, 777 MB; 18 of 18 random samples identical to the original data through the repo's loader |
| Exact fine-tune overrides | compose; 411 steps per epoch at global batch 128 |
| Real training script end to end at miniature model size | trains (loss 3.1 to 0.7 in 40 steps), in-training eval and videos run, weights and state are written |
| Full-size forward and backward pass on the CPU | finite gradients on every trainable parameter; about 1.6 to 1.8 GB of memory per sample |

The last row puts one GPU at roughly 33 GB + 16 x 1.7 GB = 60 GB for the official batch of 16 on
8 GPUs. That fits an H200 (141 GB) with room to spare and rules out Delta's 40 GB A100s and
48 GB A40s. It is a CPU measurement, so treat it as approximate.

Not checked, because the desktop GPU has 16 GB: an optimizer step of the full model, and any speed
number for Delta.

## The 7 tasks

Lowest success for the released checkpoint in `model-comparison/results/benchmark/fastwam_official`:

| LIBERO task | Baseline | Demos |
|---|---|---|
| libero_10 task 4, two mugs on two plates | 82% | 37 |
| libero_goal task 3, open top drawer and put the bowl inside | 90% | 33 |
| libero_goal task 9, wine bottle on the rack | 90% | 38 |
| libero_spatial task 5, bowl on the ramekin to the plate | 90% | 38 |
| libero_10 task 0, soup and tomato sauce in the basket | 92% | 38 |
| libero_10 task 6, mug on the plate, pudding to its right | 92% | 38 |
| libero_goal task 2, wine bottle on top of the cabinet | 92% | 45 |

What to expect: probably no measurable change in success. The checkpoint already fits these demos
as well as the rest (loss above), and two existing runs of the same checkpoint differ by up to
14 points on a single task (2.6 on average). This run checks that the Delta pipeline works and
that a fine-tuned checkpoint is usable; it is unlikely to show an improvement by itself.

## Runbook

Placeholders: `<project>` is the Delta project, `<user>` the NCSA username.

1. On the desktop, apply the fix and copy code and inputs (14 GB) to Delta:

   ```bash
   cd "/home/yifan/RoboTouch Lab/FastWAM"
   git apply finetune_check/resume_pt_fix.patch
   DEST=<user>@login.delta.ncsa.illinois.edu:/work/hdd/<project>/<user>/fastwam
   rsync -avP --exclude data --exclude runs --exclude checkpoints --exclude third_party --exclude .git ./ $DEST/FastWAM/
   rsync -avPL data/delta_bundle/ $DEST/      # checkpoint, stats, VAE, subset, text cache
   ```

2. On a Delta login node, build the environment:

   ```bash
   export ROOT=/work/hdd/<project>/$USER/fastwam && cd $ROOT
   wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
   bash Miniforge3-Linux-x86_64.sh -b -p $ROOT/miniforge3
   source $ROOT/miniforge3/etc/profile.d/conda.sh
   conda create -y -p $ROOT/envs/fastwam python=3.10 && conda activate $ROOT/envs/fastwam
   pip install -U pip
   pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 --extra-index-url https://download.pytorch.org/whl/cu128
   cd $ROOT/FastWAM && pip install -e . --extra-index-url https://download.pytorch.org/whl/cu128
   conda install -y -c conda-forge "ffmpeg>=7,<8"     # torchcodec needs it
   ```

3. Preflight on one cheap GPU (a few minutes). It repeats the desktop loss check on Delta:

   ```bash
   srun --account=<project>-delta-gpu --partition=gpuA40x4-interactive --nodes=1 --gpus-per-node=1 \
        --cpus-per-task=16 --mem=64g --time=00:30:00 --constraint="work" --pty bash
   source $ROOT/miniforge3/etc/profile.d/conda.sh && conda activate $ROOT/envs/fastwam && cd $ROOT/FastWAM
   export DIFFSYNTH_MODEL_BASE_PATH=$ROOT/ckpt DIFFSYNTH_SKIP_DOWNLOAD=true S=$ROOT/data/libero_hard_subset
   python finetune_check/check_released_ckpt.py --ckpt $ROOT/ckpt/libero_uncond_2cam224.pt \
     --stats $ROOT/ckpt/libero_uncond_2cam224_dataset_stats.json \
     --dataset-dirs $S/libero_10 $S/libero_goal $S/libero_spatial \
     --text-cache $ROOT/data/text_embeds_cache/libero --num-samples 32 --out $ROOT/runs/preflight
   ```

   Pass: video loss about 0.07, action loss about 0.004 to 0.006.

4. Fine-tune (edit the two `CHANGE_ME` values first):

   ```bash
   cd $ROOT/FastWAM && sbatch finetune_check/delta_finetune_hard_tasks.sbatch
   ```

   1,000 steps is 2.4 passes over the subset at learning rate 1e-5. The learning rate is my
   choice for restarting a converged model, not an official value. Estimated at 100 minutes of
   training from the authors' H20 speed; the log's speed line gives the real figure.

   In the log, steps 10 to 30 should show a loss of about 0.08 (video about 0.07, action about
   0.005). A loss near 1 means the checkpoint was not kept or a setting does not match: cancel.

5. Copy `runs/libero_uncond_2cam224_1e-4/<run_id>/checkpoints/weights/step_001000.pt` (12 GB)
   back and evaluate it with `model-comparison/evals/fastwam/fastwam_common.py --ckpt <file>`,
   keeping the same `--stats`, seeds and episode ids. Re-run the released checkpoint on the same
   episodes for the "before" number: the 7 tasks were picked for scoring low, so their old
   numbers are biased low. Your records show about 3 to 5 seconds per episode, so 350 episodes
   is about 20 minutes.

   Pass: the fine-tuned checkpoint is within noise of the released one on the 7 tasks, pooled.

## Files

| File | Purpose |
|---|---|
| `resume_pt_fix.patch` | the trainer fix |
| `toy_deepspeed_resume_check.py`, `trainer_resume_check.py` | reproduce the bug and the fix on small models |
| `mini_end_to_end.sh` | real training script at miniature model size, then resume with and without the fix |
| `check_released_ckpt.py` | load the checkpoint as a fine-tune would and measure its loss on real samples |
| `make_task_subset.py` | cut a LeRobot v2.1 dataset down to chosen tasks |
| `delta_finetune_hard_tasks.sbatch` | the Delta job |

Local data made tonight, all under the git-ignored `data/` and `runs/`: the LIBERO dataset
(`data/libero_mujoco3.3.2`), the subset (`data/libero_hard_subset`), the text cache
(`data/text_embeds_cache/libero`), the transfer bundle of symlinks (`data/delta_bundle`), and the
check results (`runs/finetune_check*/check_released_ckpt.json`).
