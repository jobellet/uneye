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
