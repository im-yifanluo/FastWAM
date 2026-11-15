"""Local check of the fine-tune-from-released-checkpoint input path (no training).

It builds the model and dataset from the same Hydra config and overrides a fine-tune would use,
loads the released LIBERO checkpoint the way `resume=<file.pt>` does, and measures the training
loss on real LIBERO samples. The authors give the final LIBERO training losses as roughly
video 0.07 and action 0.01 (upstream issue #31), so a correct setup should land near those.

What a pass tells you: checkpoint keys, normalisation stats, action noise shift, text cache and
video decoding all line up. What it does not tell you: anything about optimisation on Delta.

The 12 GB of trainable weights go on the GPU in bfloat16; the frozen VAE stays on the CPU so this
fits a 16 GB card. Run from the repo root:

    DIFFSYNTH_MODEL_BASE_PATH=<dir with Wan VAE> python finetune_check/check_released_ckpt.py \
        --ckpt <libero_uncond_2cam224.pt> --stats <libero_uncond_2cam224_dataset_stats.json>
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

REPO = Path(__file__).resolve().parents[1]


def q(path) -> str:
    """Quote a path for a Hydra override (paths here may contain spaces)."""
    return "'" + str(path).replace("'", "\\'") + "'"


def build_cfg(args):
    from fastwam.utils.config_resolvers import register_default_resolvers

    register_default_resolvers()
    overrides = [
        f"task={args.task}",
        # Fine-tuning from a FastWAM checkpoint does not need the Wan2.2 DiT or the interpolated action expert.
        "model.skip_dit_load_from_pretrain=true",
        "model.action_dit_pretrained_path=null",
        # The original released checkpoints were trained with action shift 5.0 (the repo default is now 1.0).
        f"model.action_scheduler.train_shift={args.action_shift}",
        f"model.action_scheduler.infer_shift={args.action_shift}",
        # Reuse the checkpoint's normalisation stats instead of recomputing them on a subset.
        f"+data.train.pretrained_norm_stats={q(args.stats)}",
        f"data.train.text_embedding_cache_dir={q(args.text_cache)}",
        f"output_dir={q(args.out)}",
    ]
    if args.dataset_dirs:
        overrides.append("data.train.dataset_dirs=[" + ",".join(q(d) for d in args.dataset_dirs) + "]")
    with initialize_config_dir(config_dir=str(REPO / "configs"), version_base="1.3"):
        return compose(config_name="train", overrides=overrides), overrides


def build_model(cfg, ckpt_path: str, device: str):
    """Build the training-config model, then load the checkpoint strictly (the repo loads with strict=False)."""
    import torch.nn.init as init

    # Every trainable weight is overwritten by the checkpoint below; the strict load proves that,
    # so the slow random init of 6B parameters is skipped.
    saved = {}
    for name in ["normal_", "uniform_", "trunc_normal_", "kaiming_uniform_", "kaiming_normal_",
                 "xavier_uniform_", "xavier_normal_", "zeros_", "ones_", "constant_"]:
        saved[name] = getattr(init, name)
        setattr(init, name, lambda t, *a, **k: t)
    try:
        model = instantiate(cfg.model, model_dtype=torch.bfloat16, device="cpu")
    finally:
        for name, fn in saved.items():
            setattr(init, name, fn)

    payload = torch.load(ckpt_path, map_location="cpu", mmap=True, weights_only=False)
    report = {"ckpt_step": payload.get("step"), "ckpt_dtype": payload.get("torch_dtype")}
    result = model.mot.load_state_dict(payload["mot"], strict=False)
    report["mot_missing_keys"] = list(result.missing_keys)
    report["mot_unexpected_keys"] = list(result.unexpected_keys)
    model.proprio_encoder.load_state_dict(payload["proprio_encoder"], strict=True)
    report["proprio_encoder"] = "loaded (strict)"
    report["trainable_params_B"] = round(
        (sum(p.numel() for p in model.mot.parameters()) + sum(p.numel() for p in model.proprio_encoder.parameters())) / 1e9, 3
    )
    del payload

    # Trainable parts on the GPU; frozen VAE on the CPU in float32.
    model.mot.to(device)
    model.proprio_encoder.to(device)
    model.vae.to("cpu")
    model.vae.model.float()
    model.device = torch.device(device)
    scale = [s.detach().to("cpu", torch.float32) if torch.is_tensor(s) else s for s in model.vae.scale]

    @torch.no_grad()
    def encode_on_cpu(video_tensor, tiled=False, **_):
        z = model.vae.model.encode(video_tensor.detach().to("cpu", torch.float32), scale)
        return z.to(device=device, dtype=torch.bfloat16)

    model._encode_video_latents = encode_on_cpu
    model.eval()
    return model, report


def to_batch(samples):
    batch = {}
    for key in samples[0]:
        vals = [s[key] for s in samples]
        batch[key] = torch.stack(vals) if torch.is_tensor(vals[0]) else vals
    return batch


@torch.no_grad()
def mean_loss(model, dataset, indices, batch_size, seed, shuffle_actions=False):
    """Average the real `training_loss` over the given samples (random noise levels, fixed seed)."""
    torch.manual_seed(seed)
    sums = defaultdict(float)
    n_batches = 0
    for start in range(0, len(indices), batch_size):
        chunk = indices[start:start + batch_size]
        if len(chunk) < 2 and shuffle_actions:
            continue
        batch = to_batch([dataset[i] for i in chunk])
        if shuffle_actions:  # control: pair each video with another sample's actions
            batch["action"] = torch.roll(batch["action"], shifts=1, dims=0)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, parts = model.training_loss(batch)
        for k, v in parts.items():
            sums[k] += float(v)
        n_batches += 1
    return {k: v / max(n_batches, 1) for k, v in sums.items()}, n_batches


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--stats", required=True)
    ap.add_argument("--task", default="libero_uncond_2cam224_1e-4")
    ap.add_argument("--action-shift", type=float, default=5.0)
    ap.add_argument("--text-cache", default=str(REPO / "data/text_embeds_cache/libero"))
    ap.add_argument("--dataset-dirs", nargs="*", default=None, help="override data.train.dataset_dirs")
    ap.add_argument("--num-samples", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(REPO / "runs/finetune_check"))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    from fastwam.utils import misc

    cfg, overrides = build_cfg(args)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    misc.register_work_dir(args.out)
    report = {"overrides": overrides}

    t0 = time.time()
    dataset = instantiate(cfg.data.train)
    report["dataset"] = {"num_samples": len(dataset), "load_s": round(time.time() - t0, 1)}
    sample = dataset[0]
    report["sample"] = {
        k: (list(v.shape) if torch.is_tensor(v) else str(v)[:120]) for k, v in sample.items()
    }
    report["sample"]["action_minmax"] = [round(float(sample["action"].min()), 3), round(float(sample["action"].max()), 3)]
    report["sample"]["video_minmax"] = [round(float(sample["video"].min()), 3), round(float(sample["video"].max()), 3)]
    print(json.dumps(report, indent=2), flush=True)

    t0 = time.time()
    model, load_report = build_model(cfg, args.ckpt, args.device)
    load_report["build_and_load_s"] = round(time.time() - t0, 1)
    load_report["gpu_mem_after_load_GB"] = round(torch.cuda.memory_allocated() / 1e9, 2)
    report["model"] = load_report
    print(json.dumps({"model": load_report}, indent=2), flush=True)
    if load_report["mot_missing_keys"] or load_report["mot_unexpected_keys"]:
        raise SystemExit("Checkpoint keys do not match the model built from this config; stopping.")

    gen = torch.Generator().manual_seed(args.seed)
    indices = torch.randperm(len(dataset), generator=gen)[: args.num_samples].tolist()

    t0 = time.time()
    loss, nb = mean_loss(model, dataset, indices, args.batch_size, args.seed)
    report["loss_released_ckpt"] = {**{k: round(v, 4) for k, v in loss.items()}, "batches": nb,
                                    "samples": len(indices), "seconds": round(time.time() - t0, 1)}
    print(json.dumps({"loss_released_ckpt": report["loss_released_ckpt"]}, indent=2), flush=True)

    ctrl, nb = mean_loss(model, dataset, indices[: max(args.num_samples // 4, 8)], args.batch_size, args.seed,
                         shuffle_actions=True)
    report["loss_control_mismatched_actions"] = {**{k: round(v, 4) for k, v in ctrl.items()}, "batches": nb}
    report["reference_final_losses_upstream_issue_31"] = {"loss_video": 0.07, "loss_action": 0.01}
    report["gpu_peak_GB"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)

    out_file = Path(args.out) / "check_released_ckpt.json"
    out_file.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ["loss_released_ckpt", "loss_control_mismatched_actions",
                                             "reference_final_losses_upstream_issue_31", "gpu_peak_GB"]}, indent=2))
    print(f"wrote {out_file}")


if __name__ == "__main__":
    main()
