# References behind the label-free approaches (foundation/lost1d.py and the ideas proposed around it)

Every method implemented or proposed in the label-free study, with the URL of the paper that inspired it.

## The emergent attention of self-supervised ViTs (the starting point)

- **DINO** — Caron et al., "Emerging Properties in Self-Supervised Vision Transformers", ICCV 2021.
  The [CLS] attention of the last layer segments the dominant object without any label (45.9 IoU on VOC vs 27.3 supervised). Implemented for eye traces in `foundation/dino1d.py`; the attention was anti-correlated with saccades, which motivated everything below.
  Paper: https://arxiv.org/abs/2104.14294
  Code: https://github.com/facebookresearch/dino

- **Survey** — Siméoni et al., "Unsupervised Object Localization in the Era of Self-Supervised ViTs: A Survey", 2023.
  States explicitly that the [CLS] attention is the weakest of the maps, and that the field moved to patch keys and spectral partitioning.
  Paper: https://arxiv.org/abs/2310.12904
  Curated list: https://github.com/valeoai/Awesome-Unsupervised-Object-Localization

## Implemented in foundation/lost1d.py (patch-patch readouts, no [CLS] attention)

- **LOST** — Siméoni et al., "LOST and Found: A Two-stream Network for Unsupervised Object Discovery", 2021.
  Seed = the patch with the smallest degree of the similarity graph of the last block's keys; map = similarity to the seed. Implemented as the `lost` readout.
  Paper: https://arxiv.org/abs/2102.13457
  Code: https://github.com/valeoai/LOST

- **TokenCut** — Wang et al., "TokenCut: Segmenting Objects in Images and Videos with Self-supervised Transformer and Normalized Cut", 2022.
  Normalized cut of the token graph built from DINO keys; the Fiedler vector (second-smallest eigenvector of the normalised Laplacian) is the continuous relaxation of the partition. Implemented as the `tokencut` readout.
  Paper: https://arxiv.org/abs/2209.00383
  Code: https://github.com/YangtaoWANG/Tokencut

- **MaskDistill** — Van Gansbeke et al., "Discovering Object Masks with Transformers for Unsupervised Semantic Segmentation", 2022.
  Per-image clustering of the patch tokens into foreground/background; the seed selection uses the [CLS] attention, the clustering is the point. Implemented as the `kmeans` readout.
  Paper: https://arxiv.org/abs/2112.06266
  Code: https://github.com/valeoai/MaskDistill

- **PatchCore** — Roth et al., "Towards Total Recall in Industrial Anomaly Detection", CVPR 2022.
  Anomaly = distance of a patch feature to its nearest neighbour in a memory of "ordinary" features, built without labels. Implemented as the `memory` readout (saccades are rare, fixations are the bulk).
  Paper: https://arxiv.org/abs/2106.05225
  Code: https://github.com/amazon-science/patchcore-ml-metrics

## The distillation-of-pseudo-masks family (the proposed next step: HMM as the "SAM of eye traces")

- **CutLER** — Wang et al., "Cut and Learn for Unsupervised Object Detection and Instance Segmentation", CVPR 2023.
  MaskCut (recursive NCut) generates coarse pseudo-masks, then a detector is trained on them with a loss-dropping strategy robust to misses. The pattern proposed for distilling the universal HMM into the ViT.
  Paper: https://arxiv.org/abs/2301.11320
  Code: https://github.com/facebookresearch/CutLER

- **FreeSOLO** — Wang et al., "FreeSOLO: Learning to Segment Objects without Annotations", CVPR 2022.
  Another instance of masks-from-features-then-self-training.
  Paper: https://arxiv.org/abs/2202.12181
  Code: https://github.com/NVlabs/FreeSOLO

- **CuVLER** — Sol'nin et al., "CuVLER: Enhanced Unsupervised Object Discoveries through Exhaustive Self-Supervised Transformers", 2024.
  Votes over multiple ViTs and NCut variants; useful if several encoders (JEPA, MAE, DINO) are available for voting.
  Paper: https://arxiv.org/abs/2403.07700

- **STEGO** — Hamilton et al., "Unsupervised Semantic Segmentation by Distilling Feature Correspondences", ICLR 2022.
  Dense feature-space clustering and correspondences instead of attention at all.
  Paper: https://arxiv.org/abs/2203.08414
  Code: https://github.com/mhamilton723/STEGO

## Masked-prediction pretexts (why I-JEPA is the right pretext for rare events)

- **I-JEPA** — Assran et al., "Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture", CVPR 2023.
  Prediction of the representation of large masked target blocks from an informative context; the masking itself replaces augmentations. Key lesson for eye traces: the target blocks must be at the SEMANTIC scale (a saccade is ~10 ms, not 336 samples), otherwise the model predicts the drift and ignores the saccades. Tried in `foundation/ssl_vit.py` (I-JEPA a-e rows of the benchmark).
  Paper: https://arxiv.org/abs/2301.08243
  Code: https://github.com/facebookresearch/ijepa

- **Object-centric LeJEPA** — Haeusser et al., "Object-centric LeJEPA", 2026.
  Decouples the scene partition (off-the-shelf masks) from the representation learning; unsupervised masks are enough. The argument for using the universal HMM as the partition instead of asking the ViT to discover saccades from scratch.
  Paper: https://arxiv.org/abs/2607.02404

- **MAE** — He et al., "Masked Autoencoders Are Scalable Vision Learners", CVPR 2022.
  The pixel-reconstruction control: attention spreads over textures and noise, which is why our MAE rows equal the untrained control.
  Paper: https://arxiv.org/abs/2111.06377

## Time-series / 1-D analogues

- **AnomalyBERT** — Joo et al., "AnomalyBERT: Self-Supervised Transformer for Time Series Anomaly Detection using Data Degradation Scheme", ICLR 2024.
  A degradation pretext (part of the input replaced by synthetic outliers) teaches a 1-D transformer to point at unnatural segments; the closest time-series analogue of the masking objective proposed here.
  Paper: https://arxiv.org/abs/2305.04468
  Code: https://github.com/DAMO-EDU/AnomalyBERT

- **TiViT** — Zhang et al., "Time Series Representations for Classification Lie Hidden in Pretrained Vision Transformers", NeurIPS 2024.
  Time series fed to pretrained 2-D ViTs beat dedicated time-series models; supports reusing vision recipes on eye traces.
  Paper: https://arxiv.org/abs/2506.08641

## Known failure modes and fixes, if DINO-style attention is retried

- **DINOv2** — Oquab et al., "DINOv2: Learning Robust Visual Features without Supervision", 2023.
  Curated data, longer schedules, registers; the recipe that stabilises the training that collapsed in our first run.
  Paper: https://arxiv.org/abs/2304.07193
  Code: https://github.com/facebookresearch/dinov2

- **Registers** — Darcet et al., "Vision Transformers Need Registers", ICLR 2024.
  High-norm outlier tokens steal the attention and pollute the maps; adding register tokens fixes it. The likely fix for the attention that went to the quiet parts of the trace.
  Paper: https://arxiv.org/abs/2309.07908

- **SEAM** — Wang et al., "Self-supervised Equivariant Attention Mechanism for Weakly Supervised Semantic Segmentation", CVPR 2020.
  Equivariance/consistency of the map under augmentations as an additional self-supervision; directly applicable to the rotation/gain augmentations already used in `train_lodo.py::augment`.
  Paper: https://arxiv.org/abs/2004.04581
  Code: https://github.com/YudeWang/SEAM
