# free_saccade/ — label-free saccade detection (before the foundation study)
Classic detectors, a minimum-duration HMM, seeded CEBRA embeddings and label-free quality scores; benchmarked on the 4 datasets of the 2019 article. Findings summarized in `docs/FOUNDATION_ROADMAP.md`.
| File | Role |
|---|---|
| `detectors.py` | Engbert-Kliegl velocity detector, `HMM` (2 Gaussian states, saccade = chain of 6), `velocity`, `robust_sigma`, `fill` (used by `foundation/`) |
| `cebra_seed.py` | `feats` (6 rotation-invariant channels) + seeded contrastive encoder (after Schneider et al. 2023) |
| `cebra_align.py`, `benchmark_cebra.py` | multi-session alignment and benchmark of the seeded CEBRA |
| `intrinsic.py`, `suite.py`, `separability.py` | label-free quality scores, the test suite for a detector, separability of saccades vs fixations |
| `ssl_models.py`, `data.py`, `check_units.py`, `viz.py`, `run_label_free.py`, `make_notebook.py` | embeddings, data access, unit-invariance check, figures, notebook builder |
Outputs go to `free_results/` (gitignored). Prefer `foundation/` for anything new.
