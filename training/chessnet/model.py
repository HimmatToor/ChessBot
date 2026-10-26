"""Policy + value residual network (AlphaZero / Leela style).

Input:  (batch, 18, 8, 8) float32 planes from chessnet.encode.encode_board.
Output: (policy_logits, wdl_logits)
    policy_logits  (batch, 1858)  one logit per move slot in chessnet.encode.MOVES;
                                  illegal moves are masked by the caller.
    wdl_logits     (batch, 3)     win / draw / loss for the side to move.

Only ops that export cleanly to ONNX and run in onnxruntime-web are used.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn

from chessnet.encode import NUM_MOVES, NUM_PLANES


@dataclass(frozen=True)
class ModelConfig:
    blocks: int = 10
    channels: int = 128
    se_ratio: int = 4  # squeeze-excitation bottleneck; 0 disables SE
    head_channels: int = 32
    value_hidden: int = 128

    @classmethod
    def tiny(cls) -> ModelConfig:
        """Small enough to train on a laptop CPU in minutes; for tests and smoke runs."""
        return cls(blocks=2, channels=32, se_ratio=4, head_channels=8, value_hidden=32)

    def to_dict(self) -> dict:
        return asdict(self)


def _conv_bn(in_ch: int, out_ch: int, kernel: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel, padding=kernel // 2, bias=False),
        nn.BatchNorm2d(out_ch),
    )


class SqueezeExcitation(nn.Module):
    """Reweights channels using a summary of the whole board."""

    def __init__(self, channels: int, ratio: int):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // ratio),
            nn.ReLU(inplace=True),
            nn.Linear(channels // ratio, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.fc(x.mean(dim=(2, 3)))
        return x * scale[:, :, None, None]


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, se_ratio: int):
        super().__init__()
        self.conv1 = _conv_bn(channels, channels, 3)
        self.conv2 = _conv_bn(channels, channels, 3)
        self.se = SqueezeExcitation(channels, se_ratio) if se_ratio else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.conv1(x))
        out = self.se(self.conv2(out))
        return self.relu(out + x)


class ChessNet(nn.Module):
    def __init__(self, config: ModelConfig = ModelConfig()):
        super().__init__()
        self.config = config
        c, h = config.channels, config.head_channels

        self.stem = nn.Sequential(_conv_bn(NUM_PLANES, c, 3), nn.ReLU(inplace=True))
        self.tower = nn.Sequential(*(ResidualBlock(c, config.se_ratio) for _ in range(config.blocks)))

        self.policy_head = nn.Sequential(
            _conv_bn(c, h, 1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(h * 64, NUM_MOVES),
        )
        self.value_head = nn.Sequential(
            _conv_bn(c, h, 1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(h * 64, config.value_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(config.value_hidden, 3),
        )

    def forward(self, planes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.tower(self.stem(planes))
        return self.policy_head(x), self.value_head(x)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
