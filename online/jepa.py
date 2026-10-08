"""Self-supervised pretraining of a causal backbone WITHOUT labels.

JEPA (joint-embedding predictive architecture) adapted to a causal online encoder:
  context encoder f   sees a CORRUPTED copy of the velocity signal (random blocks masked, noise, random rotation)
  target encoder f'   (EMA of f, no gradient) sees the CLEAN signal
  predictors g_h      from f's embedding at time t predict f'(clean)'s embedding at time t+h, for h in HORIZONS
                      (h = 0 : fill in what the masked / noisy past contained; h > 0 : forecast the near future)
Predicting in latent space lets the model ignore unpredictable measurement noise and keep the structure
(saccade onsets, fixation drift).  Collapse is prevented by stop-gradient + EMA target and a VICReg-style
variance term on the embeddings.

`target="raw"` replaces the latent target by the raw future velocity (classic predictive / reconstruction SSL) as an ablation.
"""
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def rotate_v(v, angles):
    c, s = torch.cos(angles)[:, None], torch.sin(angles)[:, None]
    return torch.stack([v[:, 0] * c + v[:, 1] * s, -v[:, 0] * s + v[:, 1] * c], 1)


def random_rotate(v):
    return rotate_v(v, torch.rand(v.shape[0], device=v.device) * 2 * np.pi)


def corrupt(v, mask_frac=0.3, block=(8, 40), noise=0.005):
    """zero random blocks of velocity (about mask_frac of the time axis) and add white noise. Fully vectorised (no Python loop)."""
    b, _, t = v.shape
    nb = max(1, int(round(mask_frac * t / np.mean(block))))
    ln = torch.randint(block[0], block[1] + 1, (b, nb, 1), device=v.device)
    st = (torch.rand(b, nb, 1, device=v.device) * (t - ln).clamp(min=1)).long()
    pos = torch.arange(t, device=v.device)
    masked = ((pos >= st) & (pos < st + ln)).any(1, keepdim=True)          # (b,1,t)
    return v * (~masked) + noise * torch.rand(b, 1, 1, device=v.device) * torch.randn_like(v)


class JEPA(nn.Module):
    def __init__(self, backbone, horizons=(0, 5, 10, 20), ema=0.99, target="latent"):
        super().__init__()
        self.ctx, self.horizons, self.ema, self.target_kind = backbone, tuple(horizons), ema, target
        c = backbone.channels
        out = c if target == "latent" else 3
        self.pred = nn.ModuleList([nn.Sequential(nn.Conv1d(c, 2 * c, 1), nn.GELU(), nn.Conv1d(2 * c, out, 1)) for _ in horizons])
        if target == "latent":
            self.tgt = copy.deepcopy(backbone)
            for p in self.tgt.parameters():
                p.requires_grad = False

    @torch.no_grad()
    def update_target(self):
        tp, cp = list(self.tgt.parameters()), [p.detach() for p in self.ctx.parameters()]
        torch._foreach_mul_(tp, self.ema)                       # fused: one kernel launch per list, not per tensor
        torch._foreach_add_(tp, cp, alpha=1 - self.ema)
        torch._foreach_copy_(list(self.tgt.buffers()), list(self.ctx.buffers()))

    def loss(self, v, mask_frac=0.3):
        v = random_rotate(v)
        z = self.ctx(corrupt(v, mask_frac))                                   # (B,C,T)
        t = v.shape[-1]
        if self.target_kind == "latent":
            with torch.no_grad():
                self.tgt.train()                                                # batch statistics, no gradient
                tz = self.tgt(v)
                tz = F.layer_norm(tz.transpose(1, 2), (tz.shape[1],)).transpose(1, 2)
        else:
            from archs import make_input
            tz = make_input(v)
            tz = tz / (tz.flatten(0, 2).std() + 1e-8)
        total = 0.0
        for h, g in zip(self.horizons, self.pred):
            p = g(z)
            total = total + F.smooth_l1_loss(p[:, :, :t - h], tz[:, :, h:])
        total = total / len(self.horizons)
        # variance regulariser (VICReg): every embedding channel must keep a spread over batch*time
        zf = z.transpose(0, 1).reshape(z.shape[1], -1)
        std = torch.sqrt(zf.var(1) + 1e-4)
        var_loss = F.relu(1.0 - std).mean()
        return total + 0.1 * var_loss, torch.stack([total.detach(), var_loss.detach(), std.mean().detach()])  # stats stay on the device (no sync)
