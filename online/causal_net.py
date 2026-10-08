"""Causal ("forward only") network for online saccade detection.

Output at time t depends only on the input at times <= t, so the last sample of the
newest time bin is labelled without any lookahead, and the net can run as a stateful
streaming filter (cost per sample independent of the window length).

Same input as U'n'Eye (eye velocity dX, dY per sample), plus the speed sqrt(dX^2+dY^2),
which is rotation invariant and makes the first layer's job easier.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv(nn.Module):
    def __init__(self, cin, cout, k, dilation=1):
        super().__init__()
        self.pad = (k - 1) * dilation  # left padding only -> no information from the future
        self.conv = nn.Conv1d(cin, cout, k, dilation=dilation)

    def forward(self, x):
        return self.conv(F.pad(x, (self.pad, 0)))


class Block(nn.Module):
    """causal conv -> ReLU -> BatchNorm (same order as U'n'Eye) with a residual connection"""
    def __init__(self, c, k, dilation):
        super().__init__()
        self.conv = CausalConv(c, c, k, dilation)
        self.bn = nn.BatchNorm1d(c)

    def forward(self, x):
        return x + self.bn(F.relu(self.conv(x)))


class CausalTCN(nn.Module):
    def __init__(self, classes=2, channels=24, ks=3, dilations=(1, 2, 4, 8, 16), stem_ks=5, in_ch=3):
        super().__init__()
        self.classes, self.channels, self.ks = classes, channels, ks
        self.dilations, self.stem_ks, self.in_ch = tuple(dilations), stem_ks, in_ch
        self.stem = CausalConv(in_ch, channels, stem_ks)
        self.stem_bn = nn.BatchNorm1d(channels)
        self.blocks = nn.ModuleList([Block(channels, ks, d) for d in dilations])
        self.head = nn.Conv1d(channels, classes, 1)

    @property
    def receptive_field(self):
        return 1 + (self.stem_ks - 1) + sum((self.ks - 1) * d for d in self.dilations)

    def forward(self, v):
        """v: (batch, 2, T) velocities (dX, dY) -> (batch, classes, T) softmax probabilities"""
        speed = torch.sqrt(v[:, :1] ** 2 + v[:, 1:2] ** 2 + 1e-12)
        x = torch.cat([v, speed], 1)
        x = self.stem_bn(F.relu(self.stem(x)))
        for b in self.blocks:
            x = b(x)
        return torch.softmax(self.head(x), 1)
