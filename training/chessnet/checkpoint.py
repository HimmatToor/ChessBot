"""Saving and loading model checkpoints (weights + the config needed to rebuild them)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from chessnet.model import ChessNet, ModelConfig


def save_checkpoint(path: Path, model: ChessNet, **extra: Any) -> None:
    """Writes atomically, so a Kaggle session dying mid-save can't corrupt the last good file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save({"config": model.config.to_dict(), "model": model.state_dict(), **extra}, tmp)
    tmp.replace(path)


def load_checkpoint(path: Path, device: str | torch.device = "cpu") -> tuple[ChessNet, dict[str, Any]]:
    """Returns (model in eval mode, the full checkpoint dict)."""
    # Our own files, which also hold the optimiser state and run settings, so not
    # weights-only (newer torch defaults to weights_only=True and would refuse them).
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = ChessNet(ModelConfig(**checkpoint["config"])).to(device)
    model.load_state_dict(checkpoint["model"])
    return model.eval(), checkpoint
