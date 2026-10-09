# Local results (repository recordings, classic detectors only)

Run on 60 labeled trials per set (u1-u4), 500 Hz. The SSL models and the self-training have **not** been evaluated yet
(run `UnEye_label_free.ipynb`; the smoke test of the SSL part did not finish locally).

Cohen's kappa against human labels, no training labels used:

| set | EK λ=6 | adaptive I-VT | HMM |
|---|---|---|---|
| u1 | 0.69 | 0.65 | 0.74 |
| u2 (pursuit data) | -0.02 | 0.36 | 0.64 |
| u3 | 0.56 | 0.51 | 0.77 |
| u4 | 0.78 | 0.82 | 0.74 |

Does a label-free score rank methods like the real kappa? Spearman correlation between score and kappa over 6 detector settings
(EK λ = 3, 6, 12, 20, I-VT, HMM), per dataset, then averaged over 4 datasets:

| score | mean within-dataset rho |
|---|---|
| ILS (mean of 5 parts) | +0.21 |
| coverage (clearly moving samples are labeled) | +0.43 |
| persistence | +0.07 |
| main sequence | -0.31 |
| stereotypy | -0.30 |
| precision | -0.45 |

Conclusion so far: the label-free scores are **weak rankers**. They flag clearly broken settings (EK λ=3 has precision 0.1-0.3),
but main sequence, stereotypy and precision reward detectors that only keep large, clean saccades, and so prefer a detector that
misses small ones. Only coverage (a recall proxy) points the right way, and only on average. With 4 datasets x 6 methods these
numbers are noisy. The human labels themselves score 0.67-0.84 on ILS, not higher than the detectors. Do not pick a winner from
ILS alone: use the injection benchmark (known truth, your own noise) and invariance, and confirm the final choice with a few
human labels.

# Contrastive embeddings after CEBRA (Schneider, Lee & Mathis, Nature 2023), 2026-10-09, MacBook Air M1 (mps)

All on set B of the four labeled datasets, resampled to 500 Hz; encoders trained on set A. Cohen's kappa (samples). Scripts:
`benchmark_cebra.py`, `separability.py`, `cebra_align.py` (results CSV in `free_results/`, not in git).

| method | human labels used | d1 | d2 | d3 | d4 |
|---|---|---|---|---|---|
| HMM (unsupervised) | 0 | **0.78** | 0.60 | **0.77** | **0.83** |
| Engbert-Kliegl lambda=6 | 0 | 0.72 | -0.03 | 0.56 | 0.82 |
| seeds only (EK on detrended velocity, strict) | 0 | 0.63 | 0.72 | 0.20 | 0.66 |
| seeded CEBRA-Hybrid + kNN (one dataset) | 0 | 0.73 | 0.68 | 0.54 | 0.62 |
| multi-session aligned, zero-shot (seeds) | 0 of target | 0.72 | 0.65 | 0.56 | 0.62 |
| multi-session aligned + 1 saccade + seeds (3 draws) | 1 of target | 0.75 | **0.73** | 0.55 | 0.69 |
| multi-session aligned + 1 saccade, no seeds | 1 of target | 0.35 | 0.52 | 0.19 | 0.09 |

Findings:
- Detrending the velocity (minus a 100 ms running median) is what makes saccades during pursuit findable: on dataset 2, 310
  human saccades go against the pursuit; raw Engbert-Kliegl finds 22 % of them, the detrended version 100 % (false alarms 0.27 ->
  0.51 per 1 s trial).
- No contrastive variant beats the plain unsupervised HMM except on the pursuit dataset. Event F1 of the seeded CEBRA is higher
  than Engbert-Kliegl on all four sets; sample kappa (onset / offset precision) is the weak point.
- Dataset 3, CEBRA-Time with no hint at all: no saccade cluster emerges (k-means / GMM / HDBSCAN, ARI <= 0.12), but a linear probe
  separates the classes with AUC 0.989 (input features 0.904): the information is in the embedding, not as separate clusters.
- Cross-dataset decoding without any target label (sources aligned with human labels): kappa 0.52-0.74, event F1 0.86-0.95.
  One human saccade alone never suffices; with the label-free seeds it helps on 3 of 4 sets.
