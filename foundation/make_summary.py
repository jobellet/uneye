#!/usr/bin/env python3
"""Writes docs/BENCHMARK_SUMMARY.md: everything tried so far against the U'n'Eye benchmark, as tables built from the result files (nothing typed by hand in the tables).
Best score of every column in bold, NA = not tested.   python foundation/make_summary.py
"""
import json, os
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); N_ = os.path.join(ROOT, "foundation", "runs", "night"); DS = ["d1", "d2", "d3", "d4", "andersson"]
J = lambda n: json.load(open(os.path.join(N_, n + ".json"))) if os.path.exists(os.path.join(N_, n + ".json")) else None


def fmt(cells, best):
    """cells: list of (f1, kappa) or None; best: (best f1, best kappa) of the column -> list of strings"""
    out = []
    for c, b in zip(cells, best):
        if c is None: out.append("NA"); continue
        f, k = c; out.append(("**%.2f**" if f >= b[0] - 1e-9 else "%.2f") % f + " / " + ("**%.2f**" if k >= b[1] - 1e-9 else "%.2f") % k)
    return out


def per_dataset(jid, readout=None):
    d = J(jid)
    if d is None: return None
    t = d["test"]; get = lambda v, r: (v[r]["f1"], v[r]["kappa"])
    if readout is None:                                                 # better mean F1 of the two readouts, as in the overnight report
        readout = max(["hmm", "threshold"], key=lambda r: np.mean([get(t[k], r)[0] for k in DS if k in t and r in t[k]]))
    c = [get(t[k], readout) for k in DS]; return c + [(float(np.mean([x[0] for x in c])), float(np.mean([x[1] for x in c])))]


def rowcells(jid, readout=None): return per_dataset(jid, readout) or [None] * 6


# ---------------------------------------------------------------- table 1: all five datasets
uneye = J("ref_uneye")["test"]; own = J("ref_uneye_andersson_own"); un = [(uneye[k]["threshold"]["f1"], uneye[k]["threshold"]["kappa"]) for k in DS[:4]] + [(own["f1"], own["kappa"])]
un += [(float(np.mean([x[0] for x in un])), float(np.mean([x[1] for x in un])))]
ung = [(uneye[k]["threshold"]["f1"], uneye[k]["threshold"]["kappa"]) for k in DS]; ung += [(float(np.mean([x[0] for x in ung])), float(np.mean([x[1] for x in ung])))]
lodo = J("lodo_bitcn"); lo = [(lodo[k]["f1"], lodo[k]["kappa"]) for k in DS] if lodo and all(k in lodo for k in DS) else None
if lo: lo += [(float(np.mean([x[0] for x in lo])), float(np.mean([x[1] for x in lo])))]
H = ["method", "labels used for training", "d1", "d2", "d3", "d4", "Andersson", "mean"]
t1 = [("**U'n'Eye (benchmark)**", "d1+d2+d3 labels (weights 1+2+3); d4 unseen; Andersson with its own weights", un),
      ("Universal HMM", "none (fitted on archive/ only)", rowcells("ref_universal_hmm")),
      ("Noisy-student TCN of the HMM", "none", rowcells("selftrain_b_r1_tta", "threshold")),
      ("U'n'Eye, general weights everywhere (same labels as the BiTCN rows)", "d1+d2+d3 labels; d4 and Andersson unseen", ung),
      ("Input channels + probe + HMM", "labels of the 4 other datasets", rowcells("base_input_channels")),
      ("BiTCN 8 channels, supervised", "d1+d2+d3 labels; d4, Andersson unseen", rowcells("sup_bitcn_b_tta", "threshold")),
      ("BiTCN 2 channels (vx, vy), supervised, with 8-pass test-time augmentation", "d1+d2+d3 labels; d4, Andersson unseen", rowcells("sup_bitcn_v2_tta", "threshold")),
      ("BiTCN 2 channels, supervised, single pass (what the C++ engine runs)", "d1+d2+d3 labels; d4, Andersson unseen", rowcells("sup_bitcn_v2", "threshold")),
      ("BiTCN 2 channels, leave-one-dataset-out", "the 4 OTHER datasets (zero-shot on the tested one)", lo or [None] * 6)]
s1 = "| " + " | ".join(H) + " |\n|" + "---|" * len(H) + "\n"
best = [(max(r[2][j][0] for r in t1 if r[2][j]), max(r[2][j][1] for r in t1 if r[2][j])) for j in range(6)]
for lab, lbl, cells in t1: s1 += f"| {lab} | {lbl} | " + " | ".join(fmt(cells, best)) + " |\n"

# ---------------------------------------------------------------- table 2: self-supervised representations
ssl = [("Input channels (8) + probe + HMM", "base_input_channels"), ("Input channels (6) + probe + HMM", "base_input_6ch"), ("Untrained transformer (control)", "ctrl_untrained"), ("Untrained TS2Vec conv (control)", "ctrl_untrained_ts2vec"),
       ("I-JEPA a", "jepa_a"), ("I-JEPA b (lr 1e-4)", "jepa_b"), ("I-JEPA d (EMA 0.99, block 6)", "jepa_d"), ("I-JEPA e (patch 8)", "jepa_e"), ("MAE a", "mae_a"), ("MAE b", "mae_b"), ("HuBERT a", "hubert_a"),
       ("TS2Vec a", "ts2vec_a"), ("TS2Vec b (deeper)", "ts2vec_b"), ("TS2Vec a, novelty curve (no probe)", "ts2vec_a_novelty")]
t2 = [(l, "labels of the 4 other datasets (linear probe only)" if "novelty" not in i else "none", rowcells(i)) for l, i in ssl]
s2 = "| " + " | ".join(H) + " |\n|" + "---|" * len(H) + "\n"
best = [(max(r[2][j][0] for r in t2 + [("", "", un)] if r[2][j]), max(r[2][j][1] for r in t2 + [("", "", un)] if r[2][j])) for j in range(6)]
for lab, lbl, cells in t2 + [("**U'n'Eye (benchmark, for reference)**", "see table 1", un)]: s2 += f"| {lab} | {lbl} | " + " | ".join(fmt(cells, best)) + " |\n"

# ---------------------------------------------------------------- table 3: few labels, dataset 1
NS = (10, 20, 50, 100, 200, 300, 1000)
def fl(d, pref, N, nmax=10):
    v = [d[k][:2] for k in d if k.startswith(f"{pref}{N}_")] if d else []
    return (float(np.mean([x[0] for x in v])), float(np.mean([x[1] for x in v]))) if v else None
def ndraw(d, pref, N): return len([k for k in d if k.startswith(f"{pref}{N}_")]) if d else 0
UN = json.load(open(os.path.join(ROOT, "docs", "figs_paper", "results_uneye.json")))["A"]; BI = json.load(open(os.path.join(ROOT, "docs", "figs_paper", "results.json")))["A"]
sc = J("unet_ssl_finetune"); ms = J("unet_ms_finetune"); fr = J("unet_layers_freeze"); pf = J("unet_peft")
srcs = [("**U'n’Eye retrained (benchmark)**", UN, ""), ("BiTCN 2 channels from scratch", BI, ""), ("Wider U-Net, masked-velocity pre-trained, all layers", sc, "pre_"), ("Same wider U-Net from scratch", sc, "scratch_"),
        ("Multi-scale input (k = 1, 2, 4, 8, 16), pre-trained", ms, "pre_"), ("Multi-scale input, from scratch", ms, "scratch_"),
        ("Pre-trained, only the 3 blocks that change most (c6, c5, up2)", fr, "top3_"), ("Pre-trained, only the 3 blocks that change least (c0, c1, c2)", fr, "bottom3_"), ("Pre-trained, new head only", fr, "head_only_"),
        ("Pre-trained + LoRA (rank 4)", pf, "lora_"), ("Pre-trained + BitFit (biases only)", pf, "bitfit_"), ("Pre-trained + batch-norm scale/shift only", pf, "bn_only_"), ("Pre-trained + LP-FT (head, then all)", pf, "lp_ft_")]
rows = []; prov = []
for lab, d, pref in srcs:
    cells = []; pv = []
    for N in NS:
        key = (lambda k: k) if pref else (lambda k: k)
        if pref == "" : v = [x for k, x in d.items() if k.split("_")[0] == str(N)]; c = (float(np.mean([a[0] for a in v])), float(np.mean([a[1] for a in v]))) if v else None
        else: c = fl(d, pref, N)
        nd_ = (len([k for k in d if k.split("_")[0] == str(N)]) if pref == "" else ndraw(d, pref, N)) if d else 0
        cells.append(c); pv.append(c is not None and nd_ < 3)
    rows.append((lab, cells)); prov.append(pv)
best = [(max(r[1][j][0] for r, pv in zip(rows, prov) if r[1][j] and not pv[j]), max(r[1][j][1] for r, pv in zip(rows, prov) if r[1][j] and not pv[j])) for j in range(len(NS))]
s3 = "| method (mean over the draws available) | " + " | ".join(f"N = {n}" for n in NS) + " | draws (N=10/50) |\n|" + "---|" * (len(NS) + 2) + "\n"
for (lab, cells), (_, d, pref), pv in zip(rows, srcs, prov):
    nd = (lambda N: len([k for k in d if k.split("_")[0] == str(N)]) if pref == "" else ndraw(d, pref, N)) if d else (lambda N: 0)
    strs = [("(%.2f / %.2f, provisional)" % c if p_ else f_) for c, p_, f_ in zip(cells, pv, fmt(cells, best))]
    s3 += f"| {lab} | " + " | ".join(strs) + f" | {nd(10)} / {nd(50)} |\n"

# ---------------------------------------------------------------- table 4: between subjects (dataset 4)
B = json.load(open(os.path.join(ROOT, "docs", "figs_paper", "results.json")))["B"]; subj = [k for k in B if k != "all"]
diag = float(np.mean([B[k][i][0] for i, k in enumerate(sorted(subj, key=float))])); off = float(np.mean([B[k][j][0] for i, k in enumerate(sorted(subj, key=float)) for j in range(len(B[k])) if j != i]))
diagk = float(np.mean([B[k][i][1] for i, k in enumerate(sorted(subj, key=float))])); offk = float(np.mean([B[k][j][1] for i, k in enumerate(sorted(subj, key=float)) for j in range(len(B[k])) if j != i]))
allf, allk = float(np.mean([v[0] for v in B["all"]])), float(np.mean([v[1] for v in B["all"]]))
worst = min(subj, key=lambda k: np.mean([v[0] for v in B[k]]))
s4 = ("| method | trained on one subject, tested on the same subject | trained on one subject, tested on the 9 others | trained on 3 trials of every subject, tested on all |\n|---|---|---|---|\n"
      f"| BiTCN 2 channels (33 trials per subject) | {diag:.2f} / {diagk:.2f} | {off:.2f} / {offk:.2f} | **{allf:.2f} / {allk:.2f}** |\n| U'n'Eye retrained | NA | NA | NA |\n")

# ---------------------------------------------------------------- table 5: layer changes
st = J("unet_layers_stats"); blocks = ["c0", "c1", "c2", "c3", "up1", "c4", "up2", "c5", "c6"]
M = np.array([[np.mean([v[b]["rel_w"] for k, v in st.items() if k.startswith(ds + "_")]) for b in blocks] for ds in DS]); npar = st["d1_20_0"]
s5 = "| block | " + " | ".join(blocks) + " |\n|" + "---|" * (len(blocks) + 1) + "\n| parameters | " + " | ".join(str(npar[b]["n_params"]) for b in blocks) + " |\n"
s5 += "| mean relative weight change over 5 datasets | " + " | ".join(f"{x:.3f}" for x in M.mean(0)) + " |\n| spread between datasets (std) | " + " | ".join(f"{x:.3f}" for x in M.std(0)) + " |\n"

doc = f"""# Benchmark summary: everything tried against U'n'Eye

Generated by `foundation/make_summary.py` from the result files in `foundation/runs/night/` and `docs/figs_paper/` (no number typed by hand in the tables). **Bold = best of the column.** NA = not tested. Scores are event F1 / Cohen's kappa
(`foundation/compare.py`: a predicted run of at least 3 samples overlapping a human saccade is a hit; kappa is sample-wise, saccade vs the rest, at 1 kHz). Test sets: a fixed subset of set B of each dataset (300 trials, 60 for Andersson); nothing was chosen on them.
Goal: match or beat U'n'Eye on BOTH F1 and kappa. If a quick test does not beat the benchmark, the idea is dropped and the result that made us stop is kept here. Details and negative results: `docs/FOUNDATION_ROADMAP.md`, `docs/OVERNIGHT_REPORT.md`.

## 1. All five datasets (no or little target data)
{s1}
Notes: the first row mixes the general weights (d1-d3; d4 unseen by it as well) and its own weights for Andersson (with the general weights Andersson is 0.55 / 0.33). U'n'Eye is trained on the labels of the datasets it is tested on, except d4 (and Andersson with general weights); the leave-one-dataset-out row is the only fully zero-shot row.
Verdict: on the datasets whose labels they were trained on or that are similar (d1, d2, d3) and on d4, the 2-channel BiTCN trained on d1+d2+d3 labels is at least as good as U'n'Eye (F1 0.94 / 0.95 / 0.94 / 0.92 against 0.90 / 0.94 / 0.93 / 0.92, kappa 0.87 / 0.91 / 0.85 / 0.84 against 0.85 / 0.88 / 0.82 / 0.85), but on Andersson it is clearly below U'n'Eye trained with its own weights (0.76 / 0.56 against 0.89 / 0.81), so the five-dataset MEAN stays with U'n'Eye (0.92 / 0.84 against 0.90 / 0.81). (A mean of 0.85 / 0.75 for U'n'Eye, used earlier in the project, was computed with its general weights on Andersson, which is unfavourable to it.) With nothing from the tested dataset (leave-one-dataset-out row) it does not beat U'n'Eye: clear losses on d2, d3, Andersson.

## 2. Self-supervised representations (frozen encoder, linear probe on other datasets, HMM or threshold readout): no gain
{s2}
Verdict: no encoder (JEPA, MAE, HuBERT, TS2Vec) beats the plain input channels or U'n'Eye; several equal the untrained control (MAE b is exactly the control: its best checkpoint was step 0). DINO attention maps and a CEBRA-style seeded embedding were also tried before (table 6).

## 3. Few labels, dataset 1 (train on N labeled trials of set B, test on 300 trials of set A)
{s3}
N = number of labeled 1-second trials; 10 draws of the trials for N = 10 / 20 / 50 (pre-trained U-Nets, U'n'Eye), 3 draws for the other cells. The last column gives the number of draws behind each row (N = 10 / 50). Cells with fewer than 3 draws are shown in parentheses (provisional, never bold): the experiment was still running when this file was generated.
Verdict: the pre-trained wider U-Net ties U'n'Eye in F1 at N <= 20 and loses at N = 50, but has a better kappa at every N; the pre-training itself adds about 0.01 over the same network from scratch. Multi-scale input does not help. Freezing: training only the blocks that change most is clearly worse than training everything (F1 about 0.8 against 0.9), and the control that trains only the blocks that change least is as bad; the head alone gives F1 0.15 (probably under-trained: 600 steps at lr 3e-4 on a frozen backbone, not tuned). So 'freeze what hardly changes' did not help here. LoRA / BitFit / batch-norm-only / LP-FT: provisional (1 draw) until the run is finished.

## 4. Generalization between subjects (dataset 4, 10 subjects)
{s4}
(F1 / kappa; weakest training subject: subject {int(float(worst))}.) U'n'Eye was not retrained for this analysis (stopped when the few-label comparison showed it was not beaten).

## 5. Which layers change when the pre-trained U-Net is fine-tuned (30 fine-tunings: 5 datasets x N = 20, 50 x 3 draws)
{s5}
The change grows from the input to the output (first block 0.03, last block 0.33) and the ranking is the same on all datasets (rank correlation >= 0.98): early layers are nearly general, the last ones adapt.

## 6. Tried and abandoned (reason that stopped each)
| idea | what we saw | source |
|---|---|---|
| Seeded CEBRA / multi-session alignment | no gain over the HMM except on dataset 2 (kappa 0.52-0.75); one human saccade alone never suffices | `free_saccade/` |
| DINO with a [CLS] attention map | attention anti-correlated with saccades (AUC 0.14-0.23), drift to collapse; independent noise per view did not change it | `foundation/dino1d.py` |
| Embedding + a hyperplane chosen by physiological priors | event F1 0.54 / 0.40 / 0.45 / 0.01 / 0.37 (d1 / d2 / d3 / d4 / Andersson) | `foundation/prior_hyperplane.py` |
| HMM with boundaries at a fraction of the peak speed, semi-Markov durations | kappa worse or unchanged | `foundation/hmm_improvements.py` |
| Zero-shot BiTCN (leave-one-dataset-out) | loses on d2, d3, Andersson (table 1) | `foundation/lodo_bitcn.py` |
| Wider U-Net + masked-velocity pre-training, multi-scale inputs | marginal / no gain in F1 (table 3) | `foundation/unet_ssl.py`, `unet_ms.py` |
| Training U'n'Eye / the BiTCN on 1000 trials, other-algorithm curves (Sheynikhovich, Otero-Millan, Engbert-Mergenthaler) | NA: not run | |

## 7. Known limitations of this comparison (found by an independent read-only review with Antigravity `agy`, each point checked in the code)
- **Not the same labels on Andersson.** The benchmark row uses U'n'Eye's own Andersson weights (in-domain); the BiTCN rows never saw Andersson. Row "U'n'Eye, general weights everywhere" is the like-for-like comparison: there the BiTCN is better or equal on every dataset except the kappa of d4 (0.84 against 0.85), with a mean of 0.90 / 0.81 against 0.85 / 0.75 (the gap comes mostly from Andersson, 0.76 / 0.56 against 0.55 / 0.33).
- **Test-time augmentation is not in the C++ engine.** The headline BiTCN rows use 8 passes (4 rotations x mirror); `cpp/src/bitcn.cpp` runs one pass, i.e. the "single pass" row (F1 0.87 / kappa 0.80), not the headline.
- **Readout chosen on the test sets.** In tables 1-2 each row uses the better of the HMM and the threshold readout, decided after seeing both on the test subsets (two options only; the network rows all use the threshold). Table 2 is affected the most.
- **Unlabeled test positions in the pre-training pool (table 3).** The pool of `foundation/unet_ssl.py` / `unet_ms.py` contains windows of the unlabeled set-A trials of dataset 1, which is also the few-label TEST set (trained on set B, tested on set A, as in the article). No label was used, but the pre-trained rows may be slightly optimistic; the from-scratch rows and U'n'Eye are not affected. The pre-trained U-Net did not beat U'n'Eye anyway, so the conclusion is unchanged (the bias is in the direction of the benchmark being under-estimated).
- **U'n'Eye was not evaluated leave-one-dataset-out** (it is trained on d1+d2+d3 labels), so the zero-shot row has no matched baseline except the "general weights" row.
- **Event F1 ignores predicted runs shorter than 3 samples** (`online/metrics.py`, `min_event = 3`): the same for every method, but 1-2-sample spurious detections are not penalised.
- Checked and not material: U'n'Eye on Andersson with a threshold of 0.5 on the saccade probability instead of the argmax (as in `uneye/classifier.py`): F1 0.890 / kappa 0.815 against 0.883 / 0.817.

Reproduce: `.venv/bin/python foundation/make_summary.py`.
"""
open(os.path.join(ROOT, "docs", "BENCHMARK_SUMMARY.md"), "w").write(doc); print("written", len(doc))
