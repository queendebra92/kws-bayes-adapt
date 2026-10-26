"""TC-ResNet8 (Choi et al., Interspeech 2019), own PyTorch re-implementation.

Input: 40 MFCC coefficients x T frames, treated as a 1-D sequence with 40 channels.
Stem: Conv1d(40 -> 16, k=3). Three residual blocks (16->24->32->48 channels, k=9, stride 2
in the first conv of each block, 1x1 strided shortcut), global average pooling over time,
linear classifier. About 66k parameters.
"""
from __future__ import annotations

import torch
from torch import nn


class Block(nn.Module):
    def __init__(self, cin: int, cout: int, k: int = 9):
        super().__init__()
        self.c1 = nn.Conv1d(cin, cout, k, stride=2, padding=k // 2, bias=False)
        self.b1 = nn.BatchNorm1d(cout)
        self.c2 = nn.Conv1d(cout, cout, k, padding=k // 2, bias=False)
        self.b2 = nn.BatchNorm1d(cout)
        self.short = nn.Sequential(nn.Conv1d(cin, cout, 1, stride=2, bias=False),
                                   nn.BatchNorm1d(cout))

    def forward(self, x):
        y = torch.relu(self.b1(self.c1(x)))
        y = self.b2(self.c2(y))
        return torch.relu(y + self.short(x))


class TCResNet8(nn.Module):
    def __init__(self, n_mfcc: int = 40, n_classes: int = 12, widths=(16, 24, 32, 48)):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(n_mfcc, widths[0], 3, padding=1, bias=False),
                                  nn.BatchNorm1d(widths[0]), nn.ReLU())
        self.blocks = nn.Sequential(*[Block(widths[i], widths[i + 1]) for i in range(3)])
        self.head = nn.Linear(widths[3], n_classes)

    def forward(self, x):  # x: [B, n_mfcc, T]
        x = self.blocks(self.stem(x))
        return self.head(x.mean(dim=2))


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def time_shift(x: torch.Tensor, max_shift: int, gen: torch.Generator | None = None) -> torch.Tensor:
    """Shift each sample in time by an integer in [-max_shift, max_shift], zero fill.

    x: [B, C, T] (already normalised, so zero = the training mean).
    """
    b, c, t = x.shape
    s = torch.randint(-max_shift, max_shift + 1, (b,), generator=gen)
    idx = torch.arange(t)[None, :] - s[:, None]
    valid = ((idx >= 0) & (idx < t))[:, None, :]
    out = x.gather(2, idx.clamp(0, t - 1)[:, None, :].expand(-1, c, -1))
    return out * valid
