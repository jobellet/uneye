"""EXPLORATORY (one quick CPU run, not yet part of the notebook): gaze-position forecasting. Self-supervised: the target is the future eye position itself, no human labels.

A causal network reads the velocity stream and, at every sample t, outputs for each horizon h (ms) a Laplace distribution over the displacement
x(t+h) - x(t) (location and log-scale per axis): a predictive distribution, i.e. an uncertainty. The Laplace likelihood (L1-like) is robust to the
rare jumps that remain in eye-tracker data; a Gaussian one is dominated by them.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import archs
import experiment as E
from experiment import random_rotate

HORIZONS_MS = (5, 10, 20)


def artifact_mask(X, Y, fs, max_speed_deg_s=1000.0, dilate_ms=30):
    """blink / dropout samples: non-finite positions or a jump faster than any saccade (1000 deg/s), dilated by +-30 ms.
    This is also a causal, rule-based blink flag (it needs no network)."""
    from scipy.ndimage import binary_dilation
    X, Y = np.asarray(X, float), np.asarray(Y, float)
    bad = ~np.isfinite(X) | ~np.isfinite(Y)
    step = np.zeros_like(bad)
    step[:, 1:] = np.hypot(np.diff(np.nan_to_num(X, nan=0, posinf=0, neginf=0), axis=1), np.diff(np.nan_to_num(Y, nan=0, posinf=0, neginf=0), axis=1)) > max_speed_deg_s / fs
    k = int(round(dilate_ms * fs / 1000.0))
    return binary_dilation(bad | step, structure=np.ones((1, 2 * k + 1), bool))


def displacement_targets(X, Y, fs, horizons_ms=HORIZONS_MS):
    """(n, 2*H, T) future displacement x(t+h) - x(t) in degrees (x then y per horizon); NaN in the last h samples and around blinks"""
    art = artifact_mask(X, Y, fs)
    X, Y = np.where(art, np.nan, np.asarray(X, float)), np.where(art, np.nan, np.asarray(Y, float))
    n, T = X.shape
    out = np.full((n, 2 * len(horizons_ms), T), np.nan, np.float32)
    for j, h in enumerate(horizons_ms):
        k = max(1, int(round(h * fs / 1000.0)))
        out[:, 2 * j, :T - k] = X[:, k:] - X[:, :T - k]
        out[:, 2 * j + 1, :T - k] = Y[:, k:] - Y[:, :T - k]
    return out


class ForecastNet(nn.Module):
    """causal backbone + 1x1 head: per horizon (mean_x, mean_y, logstd_x, logstd_y)"""

    def __init__(self, backbone="tcn", channels=48, horizons=HORIZONS_MS):
        super().__init__()
        self.backbone, self.horizons = archs.BACKBONES[backbone](channels), tuple(horizons)
        self.head = nn.Conv1d(channels, 4 * len(horizons), 1)

    def forward(self, v):
        o = self.head(self.backbone(v)).view(v.shape[0], len(self.horizons), 4, -1)
        return o[:, :, :2], o[:, :, 2:].clamp(-8, 3)       # mean, log std (deg), each (B, H, 2, T)


def rotate_pairs(y, angles):
    """rotate every (x, y) pair of a target tensor (B, 2H, T) by the same angles as the velocity"""
    c, s = torch.cos(angles)[:, None, None], torch.sin(angles)[:, None, None]
    yx, yy = y[:, 0::2], y[:, 1::2]
    out = torch.empty_like(y)
    out[:, 0::2], out[:, 1::2] = yx * c + yy * s, -yx * s + yy * c
    return out


def nll(mean, logstd, target):
    """masked Laplace negative log-likelihood (logstd = log of the scale b); target (B, 2H, T) with NaN = ignore"""
    B, H, _, T = mean.shape
    tgt = target.view(B, H, 2, T)
    m = torch.isfinite(tgt)
    tgt = torch.nan_to_num(tgt)
    l = (tgt - mean).abs() / logstd.exp() + logstd
    return (l * m).sum() / m.sum().clamp(min=1)


def train_forecaster(data, backbone="tcn", channels=48, horizons=HORIZONS_MS, groups=None, epochs=80, steps=30, batch=32, crop=700, lr=1e-3, log=print, min_epochs=30, patience=6):
    dev = E.DEVICE
    if groups is None:   # future displacement at the horizons (self-supervised)
        groups = [(data.tr[s][0], displacement_targets(*data.raw[s][:2], data.raw[s][2], horizons)) for s in data.sets]
    groups = [(torch.as_tensor(V, device=dev), torch.as_tensor(Yt, device=dev)) for V, Yt in groups]
    nv = 30
    net = ForecastNet(backbone, channels, horizons).to(dev)
    for m in net.modules():
        if isinstance(m, nn.Conv1d): m.weight.data.normal_(0, 0.02)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    n = np.array([g[0].shape[0] - nv for g in groups], float); p = n / n.sum()
    T = min(crop, min(g[0].shape[2] for g in groups)); pos0 = torch.arange(T, device=dev)
    ch2, chY = torch.arange(2, device=dev)[None, :, None], torch.arange(2 * len(horizons), device=dev)[None, :, None]

    def sample():
        vs, ys = [], []
        for k, c in enumerate(np.random.multinomial(batch, p)):
            if c == 0: continue
            V, Yt = groups[k]
            i = nv + torch.randint(V.shape[0] - nv, (c,), device=dev); s = torch.randint(V.shape[2] - T + 1, (c,), device=dev)
            pos = s[:, None] + pos0
            vs.append(V[i[:, None, None], ch2, pos[:, None, :]]); ys.append(Yt[i[:, None, None], chY, pos[:, None, :]])
        v, y = torch.cat(vs), torch.cat(ys)
        a = torch.rand(v.shape[0], device=dev) * 2 * np.pi
        return random_rotate_with(v, a), rotate_pairs(y, a)

    def val_loss():
        net.eval()
        with torch.no_grad():
            tot = 0.0
            for V, Yt in groups:
                mu, ls = net(V[:nv]); tot += float(nll(mu, ls, Yt[:nv]))
        return tot / len(groups)

    best, best_w, bad = None, None, 0
    for ep in range(1, epochs + 1):
        net.train()
        for _ in range(steps):
            v, y = sample()
            mu, ls = net(v)
            loss = nll(mu, ls, y)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        vl = val_loss()
        if best is None or vl < best: best, best_w, bad = vl, {k: x.clone() for k, x in net.state_dict().items()}, 0
        elif ep > min_epochs:
            bad += 1; net.load_state_dict(best_w)
            for g in opt.param_groups: g["lr"] *= 0.5
        if log and ep % 5 == 0: log(f"  epoch {ep:3d} validation NLL {vl:.4f} (best {best:.4f})")
        if bad > patience: break
    net.load_state_dict(best_w)
    return net


def random_rotate_with(v, a):
    c, s = torch.cos(a)[:, None], torch.sin(a)[:, None]
    return torch.stack([v[:, 0] * c + v[:, 1] * s, -v[:, 0] * s + v[:, 1] * c], 1)


def situations(L, k, since_ms=10, post=20):
    """label every sample of (n,T) human labels: 0 stable fixation, 1 saccade starts within the horizon (k samples), 2 first `since_ms`
    of a saccade, 3 later in a saccade, -1 just after a saccade (excluded)"""
    n, T = L.shape
    ph = np.zeros((n, T), int)
    for i in range(n):
        l = L[i] > 0
        d = np.diff(np.r_[0, l.astype(int)]); on, off = np.where(d == 1)[0], np.where(d == -1)[0]
        for o in on: ph[i, max(o - k, 0):o] = 1
        since = np.full(T, 10 ** 6); last = -10 ** 6
        for t in range(T):
            if d[t] == 1: last = t
            since[t] = t - last if l[t] else 10 ** 6
        ph[i][l & (since <= since_ms)] = 2; ph[i][l & (since > since_ms)] = 3
        for f in off: ph[i, f:f + post] = np.where(ph[i, f:f + post] == 0, -1, ph[i, f:f + post])
    return ph


SITUATION = {0: "stable fixation", 1: "saccade about to start", 2: "saccade, first 10 ms", 3: "saccade, later"}


@torch.no_grad()
def evaluate_forecast(net, data, sets=("1", "2")):
    """RMSE of the 2-D position error (deg) of the network, of 'stay where you are' and of 'keep the current velocity', per situation and
    horizon, plus calibration of the predicted uncertainty (share of errors inside the predicted 80 % / 95 % interval: ideal 0.80 / 0.95)."""
    import pandas as pd
    net.eval(); rows = []
    for j, h in enumerate(net.horizons):
        acc = {c: dict(net=[], stay=[], cv=[], z=[], sig=[]) for c in SITUATION}
        for s in sets:
            V, L, fs = data.te[s]; X, Y = data.te_pos[s]
            k = max(1, int(round(h * fs / 1000)))
            tgt = displacement_targets(X, Y, fs, net.horizons)[:, 2 * j:2 * j + 2]
            mu, ls = net(torch.from_numpy(V).to(next(net.parameters()).device))
            mu, sg = mu[:, j].cpu().numpy(), ls[:, j].exp().cpu().numpy()
            cv = np.zeros_like(V)
            for a in range(3): cv[:, :, 2:] += V[:, :, 2 - a:V.shape[2] - a] / 3
            ph = situations(L, k); ok = np.isfinite(tgt).all(1)
            for c in SITUATION:
                m = ok & (ph == c)
                acc[c]["net"].append(np.linalg.norm(mu - tgt, axis=1)[m]); acc[c]["stay"].append(np.linalg.norm(tgt, axis=1)[m])
                acc[c]["cv"].append(np.linalg.norm(cv * k - tgt, axis=1)[m]); acc[c]["sig"].append(np.linalg.norm(sg, axis=1)[m] / np.sqrt(2))
                acc[c]["z"].append(((mu - tgt) / sg)[np.repeat(m[:, None], 2, 1)])
        for c, name in SITUATION.items():
            r = {k_: np.concatenate(v) for k_, v in acc[c].items()}
            if len(r["net"]) == 0: continue
            rms = lambda a: float(np.sqrt(np.mean(a ** 2)))
            rows.append(dict(horizon_ms=h, situation=name, n=len(r["net"]), rmse_stay=rms(r["stay"]), rmse_const_velocity=rms(r["cv"]), rmse_network=rms(r["net"]),
                             within_80pct_interval=float(np.mean(np.abs(r["z"]) < 1.609)), within_95pct_interval=float(np.mean(np.abs(r["z"]) < 2.996)), mean_pred_scale=float(r["sig"].mean())))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- where will an ongoing saccade land?
def landing_targets(X, Y, L, fs):
    """(n, 2, T): for every sample INSIDE a labeled saccade, the displacement still to come until its last labeled sample (degrees);
    NaN elsewhere and around blinks. Supervised by the onset/offset of the human labels only (no extra labeling)."""
    art = artifact_mask(X, Y, fs)
    X, Y = np.where(art, np.nan, np.asarray(X, float)), np.where(art, np.nan, np.asarray(Y, float))
    n, T = X.shape
    out = np.full((n, 2, T), np.nan, np.float32)
    for i in range(n):
        d = np.diff(np.r_[0, (L[i] > 0).astype(int), 0])
        for on, off in zip(np.where(d == 1)[0], np.where(d == -1)[0] - 1):
            out[i, 0, on:off + 1] = X[i, off] - X[i, on:off + 1]
            out[i, 1, on:off + 1] = Y[i, off] - Y[i, on:off + 1]
    return out


@torch.no_grad()
def evaluate_landing(net, data, sets=("1", "2"), bins_ms=(0, 5, 10, 15, 20, 30, 50, 1000)):
    """error of the predicted remaining displacement, and micro (< 1 deg) vs larger classification, against the time since saccade onset.
    The displacement already made since onset is taken from the data (in a live system: from the moment the detector has flagged the onset)."""
    import pandas as pd
    net.eval(); rows = []
    ts, err, err0, so_far, rem_true, rem_pred = [], [], [], [], [], []
    for s in sets:
        V, L, fs = data.te[s]; X, Y = data.te_pos[s]
        tgt = landing_targets(X, Y, L, fs)
        mu, _ = net(torch.from_numpy(V).to(next(net.parameters()).device)); mu = mu[:, 0].cpu().numpy()      # (n,2,T)
        art = artifact_mask(X, Y, fs)
        for i in range(V.shape[0]):
            d = np.diff(np.r_[0, (L[i] > 0).astype(int), 0])
            for on, off in zip(np.where(d == 1)[0], np.where(d == -1)[0] - 1):
                if art[i, max(on - 5, 0):off + 5].any(): continue
                for t in range(on, off + 1):
                    ts.append((t - on) * 1000.0 / fs); err.append(np.linalg.norm(mu[i, :, t] - tgt[i, :, t])); err0.append(np.linalg.norm(tgt[i, :, t]))
                    sf = np.hypot(X[i, t] - X[i, max(on - 1, 0)], Y[i, t] - Y[i, max(on - 1, 0)])
                    so_far.append(sf); rem_true.append(tgt[i, :, t]); rem_pred.append(mu[i, :, t])
    ts, err, err0, so_far = map(np.array, (ts, err, err0, so_far)); rem_true, rem_pred = np.array(rem_true), np.array(rem_pred)
    # total amplitude = displacement so far + remaining (vector sum from the onset position)
    final_true = np.linalg.norm(rem_true, axis=1)
    for a, b in zip(bins_ms[:-1], bins_ms[1:]):
        m = (ts >= a) & (ts < b)
        if m.sum() == 0: continue
        rows.append(dict(ms_since_onset=f"{a}-{b if b < 1000 else '...'}", n=int(m.sum()), rmse_remaining_if_nothing_more=float(np.sqrt((err0[m] ** 2).mean())),
                         rmse_remaining_network=float(np.sqrt((err[m] ** 2).mean())), median_abs_err_network=float(np.median(err[m]))))
    return pd.DataFrame(rows)
