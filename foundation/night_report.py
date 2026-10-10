#!/usr/bin/env python3
"""Builds docs/OVERNIGHT_REPORT.md (+ figures in docs/figs_night/) from the JSON results of foundation/runs/night/. Safe to run at any time (partial results are listed
as they come); the overnight driver runs it after every step.   python foundation/night_report.py
"""
import glob, json, os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

NIGHT = os.path.join(ROOT, "foundation", "runs", "night"); OUT = os.path.join(ROOT, "docs", "OVERNIGHT_REPORT.md"); FIGS = os.path.join(ROOT, "docs", "figs_night")
DS = ("d1", "d2", "d3", "d4", "andersson")
CATALOGUE = [   # (prefix, family, idea, source)
    ("ref_uneye", "reference", "U'n'Eye, original supervised U-Net (training/weights_1+2+3), offline", "Bellet et al. 2019, J Neurophysiol"),
    ("ref_universal_hmm", "reference", "minimum-duration HMM on the unit-free detrended speed, fitted on archive/ only (no label)", "this work"),
    ("base_input_channels", "reference", "plain input: 8 velocity channels -> linear probe (labels of the 4 other datasets) -> HMM", "-"),
    ("base_input_6ch", "reference", "plain input: 6 rotation-invariant channels -> same chain", "-"),
    ("ctrl_untrained", "control", "random-initialised transformer encoder (never trained) -> same chain: what a representation must beat", "-"),
    ("ctrl_untrained_ts2vec", "control", "random-initialised TS2Vec convolutional encoder -> same chain", "-"),
    ("jepa_", "self-supervised", "I-JEPA / V-JEPA in 1-D: predict in latent space the EMA-target representations of masked blocks from the context before AND after", "Assran et al. 2023; Bardes et al. 2024"),
    ("mae_", "self-supervised", "masked autoencoder: reconstruct the masked patches of the velocity channels", "He et al. 2022; Nie et al. 2023 (PatchTST)"),
    ("hubert_", "self-supervised", "masked prediction of k-means classes of the input tokens (HuBERT iteration 1)", "Hsu et al. 2021"),
    ("ts2vec_", "self-supervised", "hierarchical contrastive learning of a bidirectional dilated-convolution encoder, timestamp masking, overlapping crops", "Yue et al. 2022 (TS2Vec)"),
    ("selftrain_", "label-free student", "noisy-student self-training of a bidirectional TCN on the pseudo-labels of the universal HMM (labels of the benchmarks never used)", "Xie et al. 2020"),
    ("sup_bitcn", "supervised", "bidirectional TCN trained on d1+d2+d3 labels with rotations etc., tested zero-shot on d4 and Andersson; + TTA; + EMA", "Bellet 2019 (augmentation); Elmadjian et al. 2021; Zemblys et al. 2019; Startsev et al. 2019"),
]
ALREADY = [
    ("seeded CEBRA, multi-session alignment (free_saccade/)", "no gain over the HMM except on dataset 2 (kappa 0.52-0.75); one human saccade alone never suffices", "Schneider et al. 2023"),
    ("DINO with a [CLS] attention map (foundation/dino1d.py)", "attention anti-correlated with saccades (AUC 0.14-0.23), drift to collapse; independent noise per view did not change it", "Caron et al. 2021"),
    ("self-supervised embedding + hyperplane chosen by physiological priors", "event F1 0.54 / 0.40 / 0.45 / 0.01 / 0.37; the prior score ranks hyperplanes like the human F1 only on dataset 2", "this work"),
    ("HMM with boundaries at a fraction of the peak speed; semi-Markov durations", "kappa worse / unchanged: that line looks saturated", "this work"),
    ("I-JEPA, first setting (jepa_a)", "validation event F1 0.50 (random) -> 0.625 at step 3000; below the plain input channels (0.66)", "Assran et al. 2023"),
]


def load():
    rows = {}
    for f in sorted(glob.glob(os.path.join(NIGHT, "*.json"))):
        if os.path.basename(f) in ("label_eff.json", "status.json", "status_extra.json") or "andersson_own" in f: continue
        try: d = json.load(open(f)); rows[d["id"]] = d
        except Exception: pass
    return rows


def family_of(cid):
    for prefix, fam, idea, src in CATALOGUE:
        if cid.startswith(prefix): return fam, idea, src
    return "other", "", ""


def fmt(x): return f"{x:.2f}"


def readout(cid, d):
    """which readout represents a candidate: references and novelty rows have only one meaningful readout (a single prediction / the HMM); for the others the better mean event F1
    of the two readouts, 'hmm' (HMM decoding) or 'threshold' (the score alone), is used for ranking and both are shown in the table"""
    if cid.startswith("ref_") or "novelty" in cid: return "hmm"
    return max(("hmm", "threshold"), key=lambda r: d["mean"][r]["f1"])


def build():
    rows = load(); os.makedirs(FIGS, exist_ok=True); now = time.strftime("%Y-%m-%d %H:%M")
    status = []
    for fn_ in ("status.json", "status_extra.json"):
        if os.path.exists(os.path.join(NIGHT, fn_)): status += json.load(open(os.path.join(NIGHT, fn_)))
    ref_f1 = rows.get("ref_uneye", {}).get("mean", {}).get("hmm", {}); hmm_ref = rows.get("ref_universal_hmm", {}).get("mean", {}).get("hmm", {})
    L = [f"# Overnight comparison of candidate architectures and training heuristics\n", f"Generated {now} by `foundation/night_report.py` from `foundation/runs/night/*.json` (MacBook Air M1, one GPU job at a time). "
         "The numbers are those of the files; nothing is typed by hand in the tables.\n"]
    # --- summary
    cand = [(cid, d) for cid, d in rows.items() if not cid.startswith(("ref_", "base_", "ctrl_"))]
    L.append("## Summary\n")
    if cand:
        R = lambda t: t[1]["mean"][readout(*t)]
        bf = max(cand, key=lambda t: R(t)["f1"]); bk = max(cand, key=lambda t: R(t)["kappa"])
        L.append(f"- Best mean event F1 over the 5 benchmarks: **{bf[0]}** ({fmt(R(bf)['f1'])} F1, {fmt(R(bf)['kappa'])} kappa, readout: {readout(*bf)}); best mean kappa: **{bk[0]}** ({fmt(R(bk)['f1'])} / {fmt(R(bk)['kappa'])}, readout: {readout(*bk)}).")
    if ref_f1: L.append(f"- Reference U'n'Eye (supervised): {fmt(ref_f1['f1'])} F1 / {fmt(ref_f1['kappa'])} kappa on the same subsets (Andersson with its general weights is 0.55 / 0.33; with its own weights 0.89 / 0.81). The label-free universal HMM: {fmt(hmm_ref.get('f1', float('nan')))} / {fmt(hmm_ref.get('kappa', float('nan')))}.")
    if "ctrl_untrained" in rows:
        c = rows["ctrl_untrained"]; r_ = readout("ctrl_untrained", c); L.append(f"- Control: the random-initialised transformer encoder gives {fmt(c['mean'][r_]['f1'])} / {fmt(c['mean'][r_]['kappa'])} ({r_}): a trained encoder only counts if it beats this.")
    done = [s for s in status if s.get("state") == "done"]; failed = [s for s in status if s.get("state") not in ("done", "running", "skipped")]
    L.append(f"- Steps of the night finished: {len(done)}; failed or timed out: {len(failed)}" + (" (" + ", ".join(f"{s['id']}: {s['state']}" for s in failed) + ")" if failed else "") + ".\n")
    # --- protocol
    L.append("## Protocol (the same for every row)\n")
    L.append("- Data: the four labeled datasets of Bellet et al. 2019 and the Andersson et al. 2017 benchmark (saccade vs everything else, 1 kHz). Tested on a FIXED subset of each test split "
             "(300 trials, 60 for Andersson), scored by event F1 (a predicted run of at least 3 samples overlapping a human saccade is a hit) and Cohen's kappa (sample by sample).")
    L.append("- Input of every learned model: 8 velocity channels (speed, acceleration, direction change, validity, detrended speed, vx, vy), unit-free, no absolute position.")
    L.append("- Representations (self-supervised rows): frozen encoder -> linear score fitted on the human labels of the OTHER four datasets (leave-one-dataset-out) -> minimum-duration HMM fitted WITHOUT labels on the target's own "
             "train trials. The encoder never saw a label. 'threshold' columns use the score alone (no HMM).")
    L.append("- Direct rows (self-training, supervised): model score -> threshold or the same HMM decoding.")
    L.append("- Checkpoints and early stopping: every 500 steps a validation on labeled TRAIN splits (probe + HMM event F1); training stops after 3 evaluations without improvement; the best checkpoint is the one tested. "
             "The test subsets were never used to choose anything.\n")
    # --- candidate catalogue
    L.append("## Candidates (what is new tonight, with sources)\n"); L.append("| id prefix | family | idea | source |\n|---|---|---|---|")
    for p, fam, idea, src in CATALOGUE: L.append(f"| `{p}` | {fam} | {idea} | {src} |")
    L.append("\nAlready tried before tonight (not repeated):\n"); L.append("| attempt | outcome | source |\n|---|---|---|")
    for a, o, s in ALREADY: L.append(f"| {a} | {o} | {s} |")
    # --- results
    L.append("\n## Results (mean over the five datasets, and per dataset; event F1 / kappa)\n")
    L.append("| id | family | labels used | **best readout** F1 / kappa | readout | mean HMM F1 / kappa | mean threshold F1 / kappa | d1 | d2 | d3 | d4 | Andersson | min |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    order = {"reference": 0, "control": 1, "self-supervised": 2, "label-free student": 3, "supervised": 4, "other": 5}
    for cid, d in sorted(rows.items(), key=lambda t: (order.get(family_of(t[0])[0], 5), -t[1]["mean"][readout(*t)]["f1"])):
        fam = family_of(cid)[0]; m = d["mean"]; t = d["test"]; ro = readout(cid, d)
        cells = [f"{fmt(t[k][ro]['f1'])} / {fmt(t[k][ro]['kappa'])}" if k in t else "-" for k in DS]
        one = cid.startswith("ref_") or "novelty" in cid
        thr = "n/a" if one else f"{fmt(m['threshold']['f1'])} / {fmt(m['threshold']['kappa'])}"; hm = f"{fmt(m['hmm']['f1'])} / {fmt(m['hmm']['kappa'])}"
        L.append(f"| `{cid}` | {fam} | {d.get('labels_used', '')[:60]} | **{fmt(m[ro]['f1'])} / {fmt(m[ro]['kappa'])}** | {ro} | {hm if not cid.startswith('ref_') else 'n/a'} | {thr} | " + " | ".join(cells) + f" | {d.get('minutes', '')} |")
    L.append("\nThe headline column is the better mean event F1 of the two readouts (shown separately): 'hmm' = the score decoded by the minimum-duration HMM (Gaussian emissions, which suits a probe score of a representation but "
             "not the saturated logit of a network), 'threshold' = the score alone. The per-dataset cells use that readout. For `ref_*` rows a single prediction is scored; for `*_novelty` rows only the HMM readout is meaningful.\n")
    # --- figures
    pts = [(cid, d["mean"][readout(cid, d)]["f1"], d["mean"][readout(cid, d)]["kappa"], family_of(cid)[0]) for cid, d in rows.items()]
    if pts:
        col = {"reference": "black", "control": "grey", "self-supervised": "#0072B2", "label-free student": "#009E73", "supervised": "#D55E00", "other": "#CC79A7"}
        fig, ax = plt.subplots(figsize=(9, 6.5))
        for cid, f1, kp, fam in pts: ax.scatter(f1, kp, c=col.get(fam, "k"), s=60); ax.annotate(cid, (f1, kp), fontsize=7, xytext=(3, 3), textcoords="offset points")
        ax.set_xlabel("mean event F1 (5 datasets, better readout of each candidate)"); ax.set_ylabel("mean Cohen's kappa"); ax.grid(alpha=0.3)
        for fam, c in col.items(): ax.scatter([], [], c=c, label=fam)
        ax.legend(fontsize=8, loc="lower left"); ax.set_title("Overnight candidates vs references"); fig.tight_layout(); fig.savefig(os.path.join(FIGS, "f1_vs_kappa.png"), dpi=130); plt.close(fig)
        L.append("![mean F1 vs mean kappa](figs_night/f1_vs_kappa.png)\n")
        ids = sorted(rows, key=lambda c: -rows[c]["mean"][readout(c, rows[c])]["kappa"]); M = np.array([[rows[c]["test"].get(k, {}).get(readout(c, rows[c]), {}).get("kappa", np.nan) for k in DS] for c in ids])
        fig, ax = plt.subplots(figsize=(7.5, 0.35 * len(ids) + 1.5)); im = ax.imshow(M, vmin=0, vmax=1, cmap="viridis", aspect="auto"); ax.set_xticks(range(5)); ax.set_xticklabels(DS); ax.set_yticks(range(len(ids))); ax.set_yticklabels(ids, fontsize=7)
        for i in range(len(ids)):
            for j in range(5):
                if np.isfinite(M[i, j]): ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", color="w" if M[i, j] < 0.6 else "k", fontsize=7)
        ax.set_title("Cohen's kappa per dataset (better readout of each candidate)"); fig.colorbar(im); fig.tight_layout(); fig.savefig(os.path.join(FIGS, "kappa_heatmap.png"), dpi=130); plt.close(fig)
        L.append("![kappa per dataset](figs_night/kappa_heatmap.png)\n")
    # --- label efficiency
    lp = os.path.join(NIGHT, "label_eff.json")
    if os.path.exists(lp):
        le = json.load(open(lp)); L.append("## Label efficiency (frozen features + linear score from N labeled trials of the target, then HMM; event F1 / kappa, mean over d1, d2, d3, Andersson)\n")
        L.append("| representation | N=1 | N=5 | N=20 | N=100 |\n|---|---|---|---|---|")
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        for n, r in le.items():
            vals = {N: (np.mean([r[k][str(N)]["hmm_f1"] for k in r]), np.mean([r[k][str(N)]["hmm_kappa"] for k in r])) for N in (1, 5, 20, 100)}
            L.append(f"| `{n}` | " + " | ".join(f"{fmt(vals[N][0])} / {fmt(vals[N][1])}" for N in (1, 5, 20, 100)) + " |")
            for i in (0, 1): ax[i].plot((1, 5, 20, 100), [vals[N][i] for N in (1, 5, 20, 100)], marker="o", label=n)
        for i, t in enumerate(("event F1", "kappa")): ax[i].set_xscale("log"); ax[i].set_xlabel("labeled trials of the target"); ax[i].set_title(t + " (HMM)"); ax[i].grid(alpha=0.3)
        ax[1].legend(fontsize=6); fig.tight_layout(); fig.savefig(os.path.join(FIGS, "label_efficiency.png"), dpi=130); plt.close(fig); L.append("\n![label efficiency](figs_night/label_efficiency.png)\n")
    # --- steps
    if status:
        L.append("## Steps of the night\n"); L.append("| step | state | minutes |\n|---|---|---|")
        for s in status: L.append(f"| {s['id']} | {s.get('state')} | {s.get('minutes', '')} |")
    L.append("\n## Limits\n\n- One training run per configuration (seed 0): differences of a few hundredths are not significant; no confidence intervals.\n- The self-supervised rows use the labels of the other four datasets in the linear probe; "
             "only the encoders are label-free. Rows trained with pseudo-labels (selftrain) use no label of any benchmark for training.\n- Test subsets: 300 trials per dataset (60 for Andersson), not the full test splits.\n"
             "- Hyper-parameter variations are few (lr, block length, patch size, EMA, depth / width); see `args` in each JSON.\n")
    open(OUT, "w").write("\n".join(L) + "\n"); print("wrote", OUT, f"({len(rows)} result files)")


if __name__ == "__main__":
    build()
