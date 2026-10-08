"""Figures for the label-free saccade notebook."""
import numpy as np
import matplotlib.pyplot as plt
from . import detectors as D, intrinsic as I

COL = ["#d95f02", "#1b9e77", "#7570b3", "#e7298a", "#66a61e", "#e6ab02", "#a6761d", "#666666"]


def trace_gallery(win, labels, n=6, seed=0, T=None, human=None, title=""):
    """n windows; for each: x/y gaze (top) and one colored band per method (below). labels: {name: bool (n_win, T)}"""
    rng = np.random.RandomState(seed)
    sp, valid = D.speed_of(win.pos, win.fs)
    busy = np.argsort(-np.nan_to_num(np.nanpercentile(np.where(valid, sp, np.nan), 99, axis=1)))[: max(n * 3, n)]
    idx = rng.choice(busy, n, replace=False)
    names = ([("human", human)] if human is not None else []) + list(labels.items())
    fig, axs = plt.subplots(n, 1, figsize=(11, 2.1 * n + 0.2), squeeze=False)
    t = np.arange(win.pos.shape[1]) / win.fs * 1000
    for a, i in zip(axs[:, 0], idx):
        a.plot(t, win.pos[i, :, 0], "k", lw=0.8); a.plot(t, win.pos[i, :, 1], color="0.5", lw=0.8)
        lo, hi = np.nanmin(win.pos[i]), np.nanmax(win.pos[i]); span = max(hi - lo, 0.2)
        for j, (nm, L) in enumerate(names):
            y0 = lo - span * (0.12 + 0.1 * j)
            for s, e in D.runs_1d(L[i]): a.fill_between(t[[s, min(e + 1, len(t) - 1)]], y0, y0 - span * 0.08, color=("k" if nm == "human" else COL[j % len(COL)]), lw=0)
            a.text(t[-1] * 1.005, y0 - span * 0.04, nm, fontsize=7, va="center")
        a.set_xlim(t[0], t[-1]); a.set_ylabel("deg"); a.margins(x=0)
    axs[-1, 0].set_xlabel("ms"); axs[0, 0].set_title(title)
    fig.tight_layout(); return fig


def main_sequence_plot(win, labels, cols=4):
    names = list(labels); n = len(names); r = int(np.ceil(n / cols))
    fig, axs = plt.subplots(r, cols, figsize=(3.4 * cols, 3.1 * r), squeeze=False)
    for ax, nm in zip(axs.ravel(), names):
        amp, vp, du = I.main_sequence_table(win.pos, labels[nm], win.fs)
        if len(amp) < 5: ax.set_title(f"{nm}: {len(amp)} events"); continue
        ax.loglog(np.maximum(amp, 1e-3), vp, ".", ms=3, alpha=0.4, color="k")
        rho = I._spear(amp, vp)
        ax.set_title(f"{nm}  n={len(amp)}  rho={rho:.2f}", fontsize=9); ax.set_xlabel("amplitude (deg)"); ax.set_ylabel("peak speed (deg/s)")
        ax.grid(alpha=0.3, which="both")
    for ax in axs.ravel()[n:]: ax.axis("off")
    fig.tight_layout(); return fig


def duration_plot(win, labels):
    """run-length distributions: saccade durations and the gaps between two saccade labels. Real saccades: no gaps < 20 ms."""
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.2)); fs = win.fs
    for j, (nm, L) in enumerate(labels.items()):
        d, g = [], []
        for k in range(L.shape[0]):
            r = D.runs_1d(L[k]); d += [(e - s + 1) * 1000 / fs for s, e in r]; g += [(s2 - e1 - 1) * 1000 / fs for (_, e1), (s2, _) in zip(r[:-1], r[1:])]
        if d: axs[0].hist(d, bins=np.arange(0, 80, 2), histtype="step", color=COL[j % len(COL)], label=nm, density=True)
        if g: axs[1].hist(g, bins=np.arange(0, 200, 4), histtype="step", color=COL[j % len(COL)], label=nm, density=True)
    axs[0].set_xlabel("saccade label duration (ms)"); axs[1].set_xlabel("gap between two saccade labels (ms)"); axs[1].axvline(20, color="r", ls="--", lw=0.8)
    axs[0].legend(fontsize=7); fig.tight_layout(); return fig


def scores_heatmap(df, cols=("ILS", "persistence", "main_sequence", "coverage", "precision", "stereotypy", "invariance", "inj_kappa"), title=""):
    d = df[[c for c in cols if c in df.columns]]
    fig, ax = plt.subplots(figsize=(1.0 * d.shape[1] + 3, 0.45 * d.shape[0] + 1.2))
    im = ax.imshow(d.values.astype(float), vmin=0, vmax=1, cmap="viridis", aspect="auto")
    ax.set_xticks(range(d.shape[1])); ax.set_xticklabels(d.columns, rotation=40, ha="right"); ax.set_yticks(range(d.shape[0])); ax.set_yticklabels(d.index)
    for i in range(d.shape[0]):
        for j in range(d.shape[1]):
            v = d.values[i, j]
            if np.isfinite(v): ax.text(j, i, f"{v:.2f}", ha="center", va="center", color="w" if v < 0.6 else "k", fontsize=8)
    ax.set_title(title); fig.colorbar(im, fraction=0.03); fig.tight_layout(); return fig


def validation_scatter(df, x="ILS", y="kappa", hue="dataset"):
    """label-free score vs real kappa on the human-labeled recordings: does the score rank methods like the truth?"""
    from scipy.stats import spearmanr
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    for j, (g, d) in enumerate(df.groupby(hue)): ax.scatter(d[x], d[y], label=g, color=COL[j % len(COL)], s=28)
    r = spearmanr(df[x], df[y])[0]
    within = [spearmanr(d[x], d[y])[0] for _, d in df.groupby(hue) if len(d) > 3]
    r_in = float(np.nanmean(within)) if within else np.nan
    ax.set_xlabel(x + " (no labels)"); ax.set_ylabel("Cohen's kappa vs human labels"); ax.set_title(f"pooled rho = {r:.2f} | within-dataset rho = {r_in:.2f}", fontsize=9)
    ax.legend(fontsize=7); ax.grid(alpha=0.3); fig.tight_layout(); return fig, (r, r_in)


def embedding_plot(emb, win, labels=None, n=8000, seed=0):
    from sklearn.decomposition import PCA
    sp, valid = D.speed_of(win.pos, win.fs); X = emb.reshape(-1, emb.shape[-1]); ok = valid.reshape(-1)
    idx = np.where(ok)[0]; idx = np.random.RandomState(seed).choice(idx, min(n, len(idx)), replace=False)
    Z = PCA(2).fit_transform(X[idx]); ls = np.log10(sp.reshape(-1)[idx] + 1)
    fig, axs = plt.subplots(1, 2 if labels is not None else 1, figsize=(9 if labels is not None else 4.6, 4), squeeze=False)
    sc = axs[0, 0].scatter(Z[:, 0], Z[:, 1], c=ls, s=3, cmap="magma"); fig.colorbar(sc, ax=axs[0, 0], label="log10 speed"); axs[0, 0].set_title("embedding (PCA), color = speed")
    if labels is not None:
        axs[0, 1].scatter(Z[:, 0], Z[:, 1], c=labels.reshape(-1)[idx], s=3, cmap="coolwarm"); axs[0, 1].set_title("color = label (red = saccade)")
    fig.tight_layout(); return fig


def bars(df, col, title=None, ref=None):
    fig, ax = plt.subplots(figsize=(7, 0.35 * len(df) + 1))
    ax.barh(df.index, df[col], color="#4c72b0"); ax.invert_yaxis()
    if ref is not None: ax.axvline(ref, color="r", ls="--", lw=0.8)
    ax.set_title(title or col); ax.grid(alpha=0.3, axis="x"); fig.tight_layout(); return fig
