"""Figures inspired by Bellet et al. 2019 (J Neurophysiol): learning curves (Fig 7A), per-metric boxplots with a reference line
(Fig 4), example traces with label bars (Fig 3), main-sequence plots of hits / false alarms / misses (Fig 5)."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from metrics import match_events, runs

COL = {"scratch": "#0072B2", "weak_ft": "#E69F00", "distill": "#009E73", "gru": "#56B4E9", "unet_lastbin": "#CC79A7", "unet_offline": "#7f7f7f",
       "human": "#222222", "fix": "#0072B2", "sacc": "#E69F00"}
LABEL = {"scratch": "from scratch", "weak_ft": "weak-label pretraining", "distill": "distilled from U'n'Eye teacher",
         "unet_lastbin": "original U'n'Eye, online (last bin)", "unet_offline": "original U'n'Eye, offline (uses the future)"}
NICE = {"kappa@0.5": "Cohen's kappa", "mcc@0.5": "MCC", "ev_f1@0.5": "event F1", "alarm_delay_ms@0.5": "alarm delay (ms)",
        "onset_err_ms@0.5": "|onset error| (ms)", "offset_err_ms@0.5": "|offset error| (ms)", "false_alarms_per_min@0.5": "false alarms / min",
        "kappa@tuned": "Cohen's kappa (tuned threshold)", "pr_auc@0.5": "PR-AUC"}


def style():
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 9, "axes.titlesize": 10,
                         "figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.25, "legend.frameon": False})


def _name(bb, st):
    return f"{bb} - {LABEL.get(st, st)}" if bb != "unet" else LABEL[f"unet_{st}"]


def _color(bb, st, i=0):
    if bb == "unet": return COL[f"unet_{st}"]
    return COL.get(st, "k") if bb == "tcn" else (["#D55E00", "#56B4E9", "#F0E442"][i % 3] if st == "scratch" else COL.get(st, "k"))


def learning_curves(df, metric="kappa@0.5", candidates=(("tcn", "scratch"), ("tcn", "weak_ft"), ("tcn", "distill")), lookahead=0.0,
                    refs=(("unet", "lastbin"), ("unet", "offline")), ax=None, seeds_dots=True):
    """metric vs number of labeled trials; line = median over seeds, band = inter-quartile range, dots = single seeds (cf. Fig 7A)"""
    style(); ax = ax or plt.subplots(figsize=(5.6, 4))[1]
    d = df[np.isclose(df.lookahead_ms, lookahead)]
    for i, (bb, st) in enumerate(list(refs) + list(candidates)):
        g = d[(d.backbone == bb) & (d.strategy == st)]
        if g.empty: continue
        by = g.groupby("n_labels")[metric]
        x, med, q1, q3 = by.median().index.values, by.median().values, by.quantile(.25).values, by.quantile(.75).values
        c, ls = _color(bb, st, i), ("--" if bb == "unet" else "-")
        ax.plot(x, med, ls, color=c, marker="o", ms=4, label=_name(bb, st)); ax.fill_between(x, q1, q3, color=c, alpha=0.15)
        if seeds_dots and bb != "unet": ax.scatter(g.n_labels * (1 + 0.03 * np.random.RandomState(0).randn(len(g))), g[metric], s=6, color=c, alpha=0.3)
    ax.set_xscale("log"); ax.set_xlabel("labeled trials (about 1 s each)"); ax.set_ylabel(NICE.get(metric, metric))
    from matplotlib.ticker import FixedLocator, ScalarFormatter
    ax.xaxis.set_major_locator(FixedLocator(sorted(d.n_labels.unique()))); ax.xaxis.set_major_formatter(ScalarFormatter()); ax.minorticks_off()
    ax.legend(fontsize=7, loc="lower right"); return ax


def boxes(df, n, candidates, metrics=("kappa@0.5", "ev_f1@0.5", "onset_err_ms@0.5", "offset_err_ms@0.5", "alarm_delay_ms@0.5", "false_alarms_per_min@0.5"),
          lookahead=0.0, ref=("unet", "offline")):
    """one boxplot panel per metric, one box per candidate over the seeds, like Fig 4; the red line = reference"""
    style(); fig, axes = plt.subplots(1, len(metrics), figsize=(2.6 * len(metrics), 3.4))
    d = df[(df.n_labels == n) & np.isclose(df.lookahead_ms, lookahead)]
    for ax, m in zip(np.atleast_1d(axes), metrics):
        data, cols, names = [], [], []
        for i, (bb, st) in enumerate(candidates):
            g = d[(d.backbone == bb) & (d.strategy == st)][m].dropna()
            data.append(g.values); cols.append(_color(bb, st, i)); names.append(f"{bb}\n{st}" if bb != "unet" else f"U-Net\n{st}")
        bp = ax.boxplot(data, patch_artist=True, widths=0.6, medianprops=dict(color="k"))
        for p, c in zip(bp["boxes"], cols): p.set_facecolor(c); p.set_alpha(0.55)
        r = d[(d.backbone == ref[0]) & (d.strategy == ref[1])][m].median()
        if np.isfinite(r): ax.axhline(r, color="#D55E00", lw=1.2)
        ax.set_xticks(range(1, len(names) + 1)); ax.set_xticklabels(names, fontsize=6.5); ax.set_title(NICE.get(m, m), fontsize=9)
    fig.suptitle(f"N = {n} labeled trials, {len(d.seed.unique())} seeds. Red line: {ref[0]} {ref[1]} (median)", fontsize=9, y=1.02)
    fig.tight_layout(); return fig


def delay_curve(df, backbone="tcn", strategy="scratch", n=100):
    """accuracy and alarm delay against the deliberate lookahead L (the network output at t describes the sample t-L)"""
    style(); fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    g = df[(df.backbone == backbone) & (df.strategy == strategy) & (df.n_labels == n)]
    ref = df[(df.backbone == "unet") & (df.strategy == "offline") & (df.n_labels == n)]
    for ax, m in zip(axes, ("kappa@0.5", "ev_f1@0.5", "alarm_delay_ms@0.5")):
        by = g.groupby("lookahead_ms")[m]
        ax.errorbar(by.median().index, by.median(), [by.median() - by.quantile(.25), by.quantile(.75) - by.median()], marker="o", capsize=2, color=COL["scratch"])
        if m != "alarm_delay_ms@0.5" and len(ref): ax.axhline(ref[m].median(), color=COL["unet_offline"], ls="--", label="original U'n'Eye offline")
        ax.set_xlabel("lookahead L (ms)"); ax.set_title(NICE[m])
    axes[0].legend(fontsize=7); fig.suptitle(f"{backbone}, {strategy}, N = {n}", fontsize=9, y=1.02); fig.tight_layout(); return fig


def _clean(x):
    """blink / dropout samples (inf, absurd values) -> NaN so that plots and amplitudes ignore them"""
    x = np.asarray(x, float)
    return np.where(np.isfinite(x) & (np.abs(np.nan_to_num(x, nan=0, posinf=1e30, neginf=-1e30)) < 1e3), x, np.nan)


def align_pred(P, k):
    """prediction made at time t for the sample t-k -> put it back at t-k (pad the end with 0)"""
    return P if k <= 0 else np.concatenate([P[:, k:], np.zeros((P.shape[0], k), P.dtype)], 1)


def trace_gallery(data, preds, thr=0.5, n_per_set=2, window=400, seed=0):
    """example traces with label bars (cf. Fig 3). preds: {name: {set: P aligned to the human labels (n,T)}}. The first model in
    `preds` is the main one: per set it shows the trial with its worst false alarm and the trial with its worst miss."""
    style(); names = list(preds); main = names[0]
    rows = []
    for s in data.sets:
        V, L, fs = data.te[s]; P = preds[main][s]
        ev = match_events(P, L, thr)
        pick = []
        for kind in ("fn", "fp"):
            if ev[kind]: pick.append((kind, max(ev[kind], key=lambda e: e[2] - e[1])))
        for k in range(len(pick), n_per_set):                      # fill with a typical (well detected) event
            if ev["tp"]: pick.append(("hit", ev["tp"][len(ev["tp"]) // (k + 2)]))
        rows += [(s, kind, e) for kind, e in pick[:n_per_set]]
    fig, axes = plt.subplots(len(rows), 1, figsize=(8.5, 2.25 * len(rows)), squeeze=False)
    for ax, (s, kind, (k, a, b)) in zip(axes[:, 0], rows):
        V, L, fs = data.te[s]; X, Y = (_clean(a) for a in data.te_pos[s])
        t0 = int(np.clip((a + b) // 2 - window // 2, 0, X.shape[1] - window)); sl = slice(t0, t0 + window); t = np.arange(window) * 1000 / fs
        rng = float(np.nanmax([np.nanmax(X[k, sl]) - np.nanmin(X[k, sl]), np.nanmax(Y[k, sl]) - np.nanmin(Y[k, sl]), 0.2]))
        ax.plot(t, X[k, sl] - np.nanmean(X[k, sl]), color="#9467bd", lw=1); ax.plot(t, Y[k, sl] - np.nanmean(Y[k, sl]), color="#2ca02c", lw=1)
        ax.set_ylim(-0.75 * rng, 0.75 * rng + 0.6 * rng * (len(names) + 1)); ax.set_xlim(0, t[-1]); ax.set_yticks([])
        bars = [("human", L[k, sl] > 0)] + [(n_, preds[n_][s][k, sl] > thr) for n_ in names]
        for j, (lab, b_) in enumerate(bars):
            y0 = 0.75 * rng + 0.6 * rng * j
            ax.fill_between([0, t[-1]], y0, y0 + 0.5 * rng, color=COL["fix"], lw=0)
            for on, off in runs(b_):
                ax.fill_between([t[on], t[min(off + 1, window - 1)]], y0, y0 + 0.5 * rng, color=COL["sacc"], lw=0)
            ax.text(t[-1] + 4, y0 + 0.2 * rng, lab, fontsize=7, va="center", clip_on=False)
        ax.set_title(f"dataset {s} ({fs} Hz), recording {k}: " + {"fn": "missed saccade", "fp": "false alarm", "hit": "typical detection"}[kind], fontsize=8, loc="left")
        ax.grid(False); ax.spines["left"].set_visible(False)
    axes[-1, 0].set_xlabel("time (ms)  [purple: horizontal, green: vertical eye position; blue = fixation, orange = saccade]")
    fig.tight_layout(); return fig


def main_sequence(data, P, thr=0.5, min_event=3):
    """peak velocity against amplitude of true positives / false positives / false negatives (cf. Fig 5), all datasets pooled
    by sampling rate; amplitude in degrees, peak velocity in deg/s from the 3-sample-smoothed position difference"""
    style(); out = {"tp": [], "fp": [], "fn": []}
    for s in data.sets:
        V, L, fs = data.te[s]; X, Y = (_clean(a) for a in data.te_pos[s])
        ev = match_events(P[s], L, thr, min_event)
        for kind in out:
            for k, a, b in ev[kind]:
                a0 = max(a - 1, 0)
                amp = float(np.hypot(X[k, b] - X[k, a0], Y[k, b] - Y[k, a0]))
                sp = np.hypot(np.diff(X[k, a0:b + 1]), np.diff(Y[k, a0:b + 1])) * fs
                if len(sp) and np.isfinite(amp) and np.isfinite(sp).all(): out[kind].append((amp, float(np.convolve(sp, np.ones(3) / 3, "same").max())))
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), sharex=True, sharey=True)
    titles = {"tp": "found by network and human", "fp": "network only (false alarm)", "fn": "human only (missed)"}
    for ax, kind in zip(axes, ("tp", "fp", "fn")):
        a = np.array(out[kind]) if out[kind] else np.zeros((0, 2))
        if len(a):
            h = ax.hist2d(np.clip(a[:, 0], 1e-2, None), a[:, 1], bins=[np.geomspace(0.02, 15, 30), np.geomspace(5, 700, 30)], cmin=1, norm=LogNorm(), cmap="viridis")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_title(f"{titles[kind]} (n = {len(a)})", fontsize=8); ax.set_xlabel("amplitude (deg)")
    axes[0].set_ylabel("peak velocity (deg/s)"); fig.tight_layout(); return fig
