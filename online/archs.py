"""Causal backbones for online saccade detection. Output at time t depends only on inputs <= t.

All backbones map velocities (B, 2, T) [dX, dY per sample] -> features (B, C, T).
`SaccadeNet` adds a 1x1 softmax head.  `JEPA` (jepa.py) pretrains a backbone without labels.

  tcn       causal dilated conv + residual (the model of the previous PR), receptive field 131 samples
  tcn_lite  depthwise-separable causal convs (about 3x fewer MACs)
  s4d       diagonal state-space layers (S4D-Lin): linear recurrence, O(1) state per step, unlimited memory
  gru       causal conv stem + 2-layer GRU: classic recurrent baseline
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def make_input(v):
    """(B,2,T) velocities -> (B,3,T): dX, dY and speed (rotation invariant)."""
    return torch.cat([v, torch.sqrt(v[:, :1] ** 2 + v[:, 1:2] ** 2 + 1e-12)], 1)


class CausalConv(nn.Module):
    def __init__(self, cin, cout, k, dilation=1, groups=1):
        super().__init__()
        self.pad = (k - 1) * dilation
        self.conv = nn.Conv1d(cin, cout, k, dilation=dilation, groups=groups)

    def forward(self, x):
        return self.conv(F.pad(x, (self.pad, 0)))


# ----------------------------------------------------------------------------- TCN
class _TCNBlock(nn.Module):
    def __init__(self, c, k, d, lite):
        super().__init__()
        if lite:  # depthwise causal conv + pointwise mixing
            self.conv = nn.Sequential(CausalConv(c, c, k, d, groups=c), nn.Conv1d(c, c, 1))
        else:
            self.conv = CausalConv(c, c, k, d)
        self.bn = nn.BatchNorm1d(c)

    def forward(self, x):
        return x + self.bn(F.relu(self.conv(x)))


class TCNBackbone(nn.Module):
    def __init__(self, channels=48, ks=3, dilations=(1, 2, 4, 8, 16, 32), stem_ks=5, lite=False):
        super().__init__()
        self.channels = channels
        self.receptive_field = 1 + (stem_ks - 1) + sum((ks - 1) * d for d in dilations)
        self.stem = CausalConv(3, channels, stem_ks)
        self.stem_bn = nn.BatchNorm1d(channels)
        self.blocks = nn.ModuleList([_TCNBlock(channels, ks, d, lite) for d in dilations])

    def forward(self, v):
        x = self.stem_bn(F.relu(self.stem(make_input(v))))
        for b in self.blocks:
            x = b(x)
        return x


# ----------------------------------------------------------------------------- S4D
class S4DLayer(nn.Module):
    """Diagonal SSM with N complex states per channel (S4D-Lin init, ZOH). Parallel (FFT) and recurrent forms."""

    def __init__(self, h, n=16, dt_min=1e-3, dt_max=1e-1):
        super().__init__()
        self.h, self.n = h, n
        self.log_dt = nn.Parameter(torch.rand(h) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        self.log_a_re = nn.Parameter(torch.log(0.5 * torch.ones(h, n)))
        self.a_im = nn.Parameter(math.pi * torch.arange(n).float().repeat(h, 1))
        self.c = nn.Parameter(torch.randn(h, n, 2) * (0.5 ** 0.5))
        self.d = nn.Parameter(torch.randn(h))

    def _disc(self):
        a = -torch.exp(self.log_a_re) + 1j * self.a_im                 # (h,n)
        dt = torch.exp(self.log_dt)[:, None]
        da = torch.exp(a * dt)                                          # exp(dt A)
        db = (da - 1) / a                                               # ZOH, B = 1
        return da, db, torch.view_as_complex(self.c.contiguous())

    def kernel(self, length):
        da, db, c = self._disc()
        k = torch.arange(length, device=da.device)
        powers = torch.exp(torch.log(da)[:, :, None] * k)               # (h,n,L)
        return 2 * torch.einsum("hn,hn,hnl->hl", c, db, powers).real    # (h,L)

    def forward(self, u):  # (B,H,T) -> (B,H,T), causal
        t = u.shape[-1]
        k = self.kernel(t)
        y = torch.fft.irfft(torch.fft.rfft(u, 2 * t) * torch.fft.rfft(k, 2 * t), 2 * t)[..., :t]
        return y + self.d[None, :, None] * u

    @torch.no_grad()
    def step(self, u, state):  # u (B,H), state (B,H,N) complex -> y (B,H), state
        da, db, c = self._disc()
        state = da * state + db * u[:, :, None]
        return 2 * (state * c).sum(-1).real + self.d * u, state


class S4DBackbone(nn.Module):
    def __init__(self, channels=48, layers=4, n=16):
        super().__init__()
        self.channels = channels
        self.receptive_field = None  # unlimited (decaying) memory
        self.stem = nn.Conv1d(3, channels, 1)
        self.stem_bn = nn.BatchNorm1d(channels)
        self.ssm = nn.ModuleList([S4DLayer(channels, n) for _ in range(layers)])
        self.mix = nn.ModuleList([nn.Conv1d(channels, 2 * channels, 1) for _ in range(layers)])
        self.bn = nn.ModuleList([nn.BatchNorm1d(channels) for _ in range(layers)])

    def forward(self, v):
        x = self.stem_bn(self.stem(make_input(v)))
        for ssm, mix, bn in zip(self.ssm, self.mix, self.bn):
            y = F.glu(mix(F.gelu(ssm(x))), dim=1)
            x = bn(x + y)
        return x


# ----------------------------------------------------------------------------- GRU
class GRUBackbone(nn.Module):
    def __init__(self, channels=48, layers=2, stem_ks=5):
        super().__init__()
        self.channels = channels
        self.receptive_field = None
        self.stem = CausalConv(3, channels, stem_ks)
        self.stem_bn = nn.BatchNorm1d(channels)
        self.gru = nn.GRU(channels, channels, layers, batch_first=True)
        self.bn = nn.BatchNorm1d(channels)

    def forward(self, v):
        x = self.stem_bn(F.relu(self.stem(make_input(v))))
        y, _ = self.gru(x.transpose(1, 2))
        return self.bn(x + y.transpose(1, 2))


BACKBONES = {
    "tcn": lambda c=48: TCNBackbone(c),
    "tcn_lite": lambda c=48: TCNBackbone(c, lite=True),
    "s4d": lambda c=48: S4DBackbone(c),
    "gru": lambda c=48: GRUBackbone(c),
}


class SaccadeNet(nn.Module):
    def __init__(self, backbone, classes=2):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Conv1d(backbone.channels, classes, 1)

    def features(self, v):
        return self.backbone(v)

    def forward(self, v):
        return torch.softmax(self.head(self.backbone(v)), 1)


def build(name, channels=48, classes=2):
    return SaccadeNet(BACKBONES[name](channels), classes)


def init_like_uneye(model):
    """weights_init of uneye/functions.py for conv layers (N(0, 0.02))."""
    for m in model.modules():
        if isinstance(m, nn.Conv1d):
            m.weight.data.normal_(0.0, 0.02)
