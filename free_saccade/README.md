# Label-free saccade / microsaccade detection (plug and play)

Goal: no expert labels. Put in a recording, get saccades and microsaccades. This folder tests whether **self-supervised embeddings**
(dense DINO, 1-D JEPA, dense contrastive) and **physics-regularised self-training** can do it, and how to *measure* quality on
recordings that have no labels.

Run `UnEye_label_free.ipynb` (Kaggle / Colab / local). It reads your h5 layout (`x`: `(n_sequences, L, 3) = (x, y, dt)` +
`subject_h5_metadata.csv`) and falls back to a demo dataset built from the repository recordings.

| file | content |
|---|---|
| `data.py` | h5 reader, audit table, resampling to 500 Hz, windows; human-labeled repository sets (sanity check only) |
| `detectors.py` | Engbert-Kliegl, adaptive I-VT, HMM with minimum-duration chain, consensus pseudo-labels, event extraction |
| `ssl_models.py` | rotation-invariant features, dense encoder, DINO / JEPA / InfoNCE objectives, k-means detector, noisy-student self-training |
| `intrinsic.py` | label-free scores: persistence (no gap < 20 ms), main sequence, coverage, precision, stereotypy, straightness; invariances; synthetic-saccade injection benchmark |
| `suite.py`, `viz.py` | glue and figures |
| `make_notebook.py` | builds the notebook (`python3 make_notebook.py`) |

Label-free tests, all computed on held-out windows of *your* data: see the notebook introduction. The notebook also scores the
repository's human-labeled recordings with the same tests, to check whether the label-free score ranks methods like the real
Cohen's kappa. Results are in the notebook output and in `RESULTS.md` (local run on the repository recordings).
