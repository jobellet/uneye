"""Foundation model for eye-movement segmentation: a causal TCN encoder -> unit-norm embedding -> cosine classifier.

The cosine classifier's weight rows ARE class prototypes: p(class) = softmax(scale * cos(z, w_k)). In the C++ engine the same
prototypes can be updated online from a user's clicks (new class = new prototype) without retraining the encoder.
Inputs per sample (all velocities, no absolute position): vx, vy (deg/s / 100), the same minus their causal 100 ms running median
(saccades as fast DEVIATIONS from the ongoing slow movement: catches microsaccades against pursuit), and a validity flag.
Causal with a fixed lookahead: the label of sample t is emitted at t + LOOKAHEAD (the encoder sees t + LOOKAHEAD at most).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LOOKAHEAD = 20                     # samples at 1 kHz (20 ms)
IN_CH, CH, DIM = 5, 64, 32
DILATIONS = (1, 2, 4, 8, 16, 32, 64)


def inputs(pos, fs=1000.0):
    """pos (n, T, 2) deg with NaN -> (n, 5, T) float32. Velocity = 1-sample difference (as U'n'Eye), 0 at and after invalid samples."""
    p = np.asarray(pos, np.float64)
    valid = np.isfinite(p).all(2)
    v = np.zeros_like(p); v[:, 1:] = (p[:, 1:] - p[:, :-1]) * fs
    ok = valid.copy(); ok[:, 1:] &= valid[:, :-1]
    v[~ok] = 0.0
    v = np.clip(v, -2000.0, 2000.0)
    # causal running median over the last 100 samples (computed with a strided window; edge = shorter window)
    w = 100; n, T, _ = v.shape
    pad = np.concatenate([np.repeat(v[:, :1], w - 1, 1), v], 1)
    med = np.median(np.lib.stride_tricks.sliding_window_view(pad, w, axis=1), axis=-1)       # (n, T, 2)
    d = v - med
    x = np.concatenate([v / 100.0, d / 100.0, ok[..., None].astype(float)], 2)
    return np.ascontiguousarray(x.transpose(0, 2, 1)).astype(np.float32)


class CConv(nn.Module):
    def __init__(self, cin, cout, k, d=1):
        super().__init__(); self.pad = (k - 1) * d; self.conv = nn.Conv1d(cin, cout, k, dilation=d)
    def forward(self, x): return self.conv(F.pad(x, (self.pad, 0)))


class Block(nn.Module):                                    # same block as online/causal_net.py: conv -> ReLU -> BN, residual
    def __init__(self, c, d):
        super().__init__(); self.conv = CConv(c, c, 3, d); self.bn = nn.BatchNorm1d(c)
    def forward(self, x): return x + self.bn(F.relu(self.conv(x)))


class Foundation(nn.Module):
    def __init__(self, classes=5, ch=CH, dim=DIM, dilations=DILATIONS):
        super().__init__()
        self.stem = CConv(IN_CH, ch, 5); self.stem_bn = nn.BatchNorm1d(ch)
        self.blocks = nn.ModuleList([Block(ch, d) for d in dilations])
        self.emb = nn.Conv1d(ch, dim, 1)
        self.proto = nn.Parameter(torch.randn(classes, dim) * 0.1)      # class prototypes (cosine classifier)
        self.scale = 16.0

    @property
    def receptive_field(self):
        return 1 + 4 + sum(2 * d for d in self.dilations)

    @property
    def dilations(self): return tuple(b.conv.conv.dilation[0] for b in self.blocks)

    def embed(self, x):                                    # (B, 5, T) -> (B, dim, T) unit norm; output t describes sample t - LOOKAHEAD
        h = self.stem_bn(F.relu(self.stem(x)))
        for b in self.blocks: h = b(h)
        return F.normalize(self.emb(h), dim=1)

    def forward(self, x):                                  # logits (B, K, T)
        z = self.embed(x)
        return self.scale * torch.einsum("bdt,kd->bkt", z, F.normalize(self.proto, dim=1))
