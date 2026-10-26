"""Supervised training on shards from prepare_data.py.

Quick CPU run with the tiny model:
    python train.py --model tiny --batch-size 256 --steps 3000 --out checkpoints/tiny

Full run on a Kaggle GPU:
    python train.py --model full --batch-size 1024 --epochs 2 --out checkpoints/full

Interrupted? Re-run the same command with --resume checkpoints/<run>/latest.pt.
Writes latest.pt (every eval), best.pt (lowest validation loss) and metrics.csv.
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from chessnet.checkpoint import save_checkpoint
from chessnet.data import load_shard, unpack_planes_torch
from chessnet.model import ChessNet, ModelConfig, count_parameters

MODEL_PRESETS = {"tiny": ModelConfig.tiny(), "full": ModelConfig()}


class InMemoryData:
    """All shards of one split held bit-packed in RAM (144 B/position); unpacked per batch."""

    def __init__(self, paths: list[Path], limit: int | None = None):
        if not paths:
            raise FileNotFoundError("no shards found")
        shards = [load_shard(p) for p in paths]
        self.planes = np.concatenate([s["planes"] for s in shards])[:limit]
        self.move = np.concatenate([s["move"] for s in shards])[:limit].astype(np.int64)
        self.result = np.concatenate([s["result"] for s in shards])[:limit].astype(np.int64)

    def __len__(self) -> int:
        return len(self.move)

    def batch(self, indices: np.ndarray, device: torch.device) -> tuple[torch.Tensor, ...]:
        # Ship the packed bytes and unpack on the device: 8x less host->GPU traffic.
        planes = unpack_planes_torch(torch.from_numpy(self.planes[indices]).to(device))
        move = torch.from_numpy(self.move[indices]).to(device)
        result = torch.from_numpy(self.result[indices]).to(device)
        return planes, move, result


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def lr_at(step: int, total: int, peak: float, warmup: int) -> float:
    """Linear warm-up, then cosine decay to 5% of peak."""
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return peak * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0))))


@torch.no_grad()
def evaluate(model: ChessNet, data: InMemoryData, device: torch.device, batch_size: int) -> dict[str, float]:
    model.eval()
    totals = {"policy_loss": 0.0, "value_loss": 0.0, "policy_acc": 0.0, "value_acc": 0.0}
    for start in range(0, len(data), batch_size):
        planes, move, result = data.batch(np.arange(start, min(start + batch_size, len(data))), device)
        policy, wdl = model(planes)
        totals["policy_loss"] += F.cross_entropy(policy, move, reduction="sum").item()
        totals["value_loss"] += F.cross_entropy(wdl, result, reduction="sum").item()
        totals["policy_acc"] += (policy.argmax(1) == move).sum().item()
        totals["value_acc"] += (wdl.argmax(1) == result).sum().item()
    model.train()
    return {k: v / len(data) for k, v in totals.items()}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=Path("data/processed"))
    parser.add_argument("--out", type=Path, default=Path("checkpoints/run"))
    parser.add_argument("--model", choices=MODEL_PRESETS, default="tiny")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--steps", type=int, default=None, help="total optimiser steps (overrides --epochs)")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup", type=int, default=200, help="warm-up steps")
    parser.add_argument("--value-weight", type=float, default=1.0, help="value loss multiplier")
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--max-train", type=int, default=None, help="use only the first N training positions")
    parser.add_argument("--max-val", type=int, default=50_000)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--max-hours", type=float, default=None,
                        help="stop cleanly (evaluate + save) after this long, e.g. before Kaggle's 12 h limit")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = pick_device(args.device)
    use_amp = device.type == "cuda"
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True  # input shape never changes, so let cuDNN pick the fastest kernels
        print(f"GPU: {torch.cuda.get_device_name(device)}")

    train_data = InMemoryData(sorted(args.data.glob("train-*.npz")), args.max_train)
    val_data = InMemoryData(sorted(args.data.glob("val-*.npz")), args.max_val)
    steps_per_epoch = max(1, len(train_data) // args.batch_size)
    total_steps = args.steps or max(1, int(args.epochs * steps_per_epoch))

    model = ChessNet(MODEL_PRESETS[args.model]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    step, best_val = 0, float("inf")

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        step, best_val = checkpoint["step"], checkpoint["best_val"]
        print(f"resumed from {args.resume} at step {step}")

    print(f"device {device}  model {args.model} ({count_parameters(model):,} params)  "
          f"train {len(train_data):,}  val {len(val_data):,}  steps {total_steps:,}")

    args.out.mkdir(parents=True, exist_ok=True)
    metrics_path = args.out / "metrics.csv"
    new_log = not metrics_path.exists()
    log = open(metrics_path, "a", newline="")
    writer = csv.writer(log)
    if new_log:
        writer.writerow(["step", "lr", "train_loss", "val_policy_loss", "val_value_loss",
                         "val_policy_acc", "val_value_acc", "positions_per_s"])

    def checkpoint_extra() -> dict:
        # Paths stored as strings: plain types load the same in every torch/Python version.
        run_args = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
        return {"optimizer": optimizer.state_dict(), "step": step, "best_val": best_val, "args": run_args}

    model.train()
    order = rng.permutation(len(train_data))
    cursor = (step * args.batch_size) % len(train_data)
    # Accumulated on the device: calling loss.item() every step would make the CPU wait for the GPU.
    running_loss, running_n, last_time = torch.zeros((), device=device), 0, time.time()
    deadline = time.time() + args.max_hours * 3600 if args.max_hours is not None else None

    while step < total_steps:
        if cursor + args.batch_size > len(train_data):  # new epoch
            order, cursor = rng.permutation(len(train_data)), 0
        planes, move, result = train_data.batch(order[cursor:cursor + args.batch_size], device)
        cursor += args.batch_size

        lr = lr_at(step, total_steps, args.lr, args.warmup)
        for group in optimizer.param_groups:
            group["lr"] = lr

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            policy, wdl = model(planes)
            loss = F.cross_entropy(policy, move) + args.value_weight * F.cross_entropy(wdl, result)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        step += 1
        running_loss += loss.detach().float()
        running_n += 1
        out_of_time = deadline is not None and time.time() > deadline

        if step % args.eval_every == 0 or step == total_steps or out_of_time:
            train_loss = running_loss.item() / running_n  # also waits for the GPU, so the timing below is honest
            speed = running_n * args.batch_size / (time.time() - last_time)
            val = evaluate(model, val_data, device, args.batch_size * 4)
            val_loss = val["policy_loss"] + args.value_weight * val["value_loss"]
            print(f"step {step:>7,}/{total_steps:,}  lr {lr:.2e}  train {train_loss:.3f}  "
                  f"val policy {val['policy_loss']:.3f} (acc {val['policy_acc']:.1%})  "
                  f"value {val['value_loss']:.3f} (acc {val['value_acc']:.1%})  {speed:,.0f} pos/s", flush=True)
            writer.writerow([step, f"{lr:.6g}", f"{train_loss:.4f}",
                             *(f"{val[k]:.4f}" for k in ("policy_loss", "value_loss", "policy_acc", "value_acc")),
                             f"{speed:.0f}"])
            log.flush()

            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(args.out / "best.pt", model, **checkpoint_extra())
            save_checkpoint(args.out / "latest.pt", model, **checkpoint_extra())
            running_loss, running_n, last_time = torch.zeros((), device=device), 0, time.time()

            if out_of_time:
                print(f"stopping at step {step:,}: --max-hours {args.max_hours} reached "
                      f"(resume with --resume {args.out / 'latest.pt'})")
                break

    log.close()
    print(f"done. best validation loss {best_val:.3f}; checkpoints in {args.out}")


if __name__ == "__main__":
    main()
