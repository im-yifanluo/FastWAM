"""Run the real FastWAM trainer on a tiny stand-in model to see what `resume=<file.pt>` does.

The stand-in learns y = x @ W. The "checkpoint" holds the exact W, so a run that truly starts
from it has a loss of about zero at every step. If the loaded weights are lost, the loss is about
zero at step 1 (the first forward still sees them) and jumps at step 2.

Run from the repo root with the real launcher config:

    accelerate launch --config_file scripts/accelerate_configs/accelerate_zero1_ds.yaml \
        --num_processes 1 finetune_check/trainer_resume_check.py --out <scratch dir> \
        [--trainer-file <path to a patched copy of src/fastwam/trainer.py>]
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.utils.data import Dataset

DIM = 16


def load_trainer_cls(trainer_file):
    if trainer_file is None:
        from fastwam.trainer import Wan22Trainer
        return Wan22Trainer
    import fastwam  # noqa: F401  (parent package for the relative imports)
    spec = importlib.util.spec_from_file_location("fastwam.trainer_under_test", trainer_file)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.Wan22Trainer


class ToyData(Dataset):
    def __init__(self, w_true, n=256):
        g = torch.Generator().manual_seed(1)
        self.x = torch.randn(n, DIM, generator=g)
        self.y = self.x @ w_true.float()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return {"x": self.x[i], "y": self.y[i]}


class ToyPolicy(torch.nn.Module):
    """Has the attributes the trainer expects from FastWAM: `dit`, `training_loss`, save/load_checkpoint."""

    def __init__(self):
        super().__init__()
        self.dit = torch.nn.Linear(DIM, DIM, bias=False)
        self.proprio_encoder = None
        self.compile_training_denoise = False
        self.losses = []

    def training_loss(self, sample):
        w = self.dit.weight
        pred = self.dit(sample["x"].to(device=w.device, dtype=w.dtype))
        loss = torch.nn.functional.mse_loss(pred.float(), sample["y"].to(w.device).float())
        self.losses.append(float(loss.detach()))
        return loss, {"loss_action": float(loss.detach())}

    def save_checkpoint(self, path, optimizer=None, step=None):
        torch.save({"mot": self.dit.state_dict(), "step": step}, path)

    def load_checkpoint(self, path, optimizer=None):
        self.dit.load_state_dict(torch.load(path, map_location="cpu")["mot"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--trainer-file", default=None)
    ap.add_argument("--steps", type=int, default=6)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(0)
    w_true = torch.randn(DIM, DIM).to(torch.bfloat16)
    ckpt = out / "released_like.pt"
    torch.save({"mot": {"weight": w_true.t().contiguous()}}, ckpt)  # nn.Linear stores W transposed

    cfg = OmegaConf.create({
        "output_dir": str(out), "learning_rate": 1e-4, "weight_decay": 1e-2, "batch_size": 8, "num_workers": 0,
        "num_epochs": 1, "max_steps": args.steps, "log_every": 1, "save_every": 0, "eval_every": 0,
        "eval_num_inference_steps": 1, "gradient_accumulation_steps": 1, "max_grad_norm": 1.0, "seed": 42,
        "resume": str(ckpt), "mixed_precision": "bf16", "lr_scheduler_type": "constant", "wandb": {"enabled": False},
    })
    model = ToyPolicy().to(torch.bfloat16)  # random init, like a model built before the checkpoint is loaded
    trainer_cls = load_trainer_cls(args.trainer_file)
    trainer = trainer_cls(model=model, train_dataset=ToyData(w_true), val_dataset=None, cfg=cfg)
    trainer.train()

    losses = [round(v, 5) for v in model.losses]
    kept = max(losses[1:]) < 10 * max(losses[0], 1e-4)
    print("RESULT " + json.dumps({
        "trainer": "patched copy" if args.trainer_file else "repo as is",
        "loss_per_step": losses,
        "verdict": "checkpoint weights KEPT" if kept else "checkpoint weights LOST after step 1",
    }))


if __name__ == "__main__":
    main()
