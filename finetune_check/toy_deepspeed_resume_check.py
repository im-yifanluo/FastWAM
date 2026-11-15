"""Toy reproduction: are weights loaded AFTER `accelerator.prepare()` kept by DeepSpeed ZeRO?

FastWAM's trainer handles `resume=<file.pt>` by loading the weights into the model after
`accelerator.prepare()` (src/fastwam/trainer.py, `_resume_or_load_checkpoint`). With DeepSpeed ZeRO
and bf16, the optimiser keeps its own fp32 master copy of the weights, made inside `prepare()`,
and copies it back into the model on every step. This script mimics the trainer's order on a tiny
model and reports where the weights end up after one optimiser step.

Run from the repo root with the same launcher config the real training uses:

    accelerate launch --config_file scripts/accelerate_configs/accelerate_zero1_ds.yaml \
        --num_processes 1 finetune_check/toy_deepspeed_resume_check.py [--fix]
"""
import argparse
import json

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader, TensorDataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fix", action="store_true", help="refresh DeepSpeed's fp32 master weights after loading")
    args = ap.parse_args()

    torch.manual_seed(0)
    accelerator = Accelerator(gradient_accumulation_steps=1, mixed_precision="bf16", step_scheduler_with_optimizer=False)
    model = torch.nn.Sequential(torch.nn.Linear(16, 64), torch.nn.GELU(), torch.nn.Linear(64, 16)).to(torch.bfloat16)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2, betas=(0.9, 0.95))
    loader = DataLoader(TensorDataset(torch.randn(64, 16), torch.randn(64, 16)), batch_size=8)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=10)
    model, optimizer, loader, scheduler = accelerator.prepare(model, optimizer, loader, scheduler)

    unwrapped = accelerator.unwrap_model(model)
    init_w = {k: v.detach().float().clone() for k, v in unwrapped.state_dict().items()}

    # "Checkpoint" weights: clearly different from the initial ones (every value shifted by +1).
    ckpt = {k: (v + 1.0).to(torch.bfloat16) for k, v in init_w.items()}
    unwrapped.load_state_dict(ckpt)  # same call order as the trainer: load after prepare()
    ckpt_w = {k: v.float() for k, v in ckpt.items()}

    if args.fix:
        ds_optimizer = accelerator.deepspeed_engine_wrapped.engine.optimizer
        ds_optimizer.refresh_fp32_params()

    x, y = next(iter(loader))
    with accelerator.accumulate(model):
        with accelerator.autocast():
            loss = torch.nn.functional.mse_loss(model(x.to(torch.bfloat16)).float(), y.float())
        accelerator.backward(loss)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

    after = {k: v.detach().float().cpu() for k, v in unwrapped.state_dict().items()}
    dist_to_ckpt = max(float((after[k] - ckpt_w[k].cpu()).abs().max()) for k in after)
    dist_to_init = max(float((after[k] - init_w[k].cpu()).abs().max()) for k in after)
    verdict = "KEPT the loaded weights" if dist_to_ckpt < dist_to_init else "REVERTED to the pre-load weights"
    print("RESULT " + json.dumps({
        "fix_applied": bool(args.fix),
        "distributed_type": str(accelerator.distributed_type),
        "zero_stage": accelerator.state.deepspeed_plugin.deepspeed_config.get("zero_optimization", {}).get("stage"),
        "max_abs_distance_to_loaded_weights_after_1_step": round(dist_to_ckpt, 5),
        "max_abs_distance_to_pre_load_weights_after_1_step": round(dist_to_init, 5),
        "verdict": verdict,
    }))


if __name__ == "__main__":
    main()
