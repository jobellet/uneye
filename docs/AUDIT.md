# Independent audit of the scripts that produced the results (2026-10-10)

One Antigravity `agy` agent (model `gemini-3.8-flash-medium`, read-only: file reading only, no shell) audited each of the 51 scripts that had already been run (`foundation/`, `free_saccade/`,
`cpp/scripts/`, `online/`). Every finding was then checked against the code by hand before being accepted; accepted ones were fixed and the affected experiments re-run (old results kept in
`foundation/runs/night/before_audit/`). Raw reports are not kept in the repository; `foundation/conventions.py` (the newest script) was reviewed by `agy` before its first run.

**Reading guide.** 38 of the 51 agents answered "NO ERRORS FOUND" in a few lines: with this fast model that is weak evidence, not a proof. The findings below are the ones that were kept after checking.

## Accepted and fixed
| script | finding (verified) | what it changed | action |
|---|---|---|---|
| `foundation/compare.py` `to_1khz` | native 500 Hz predictions were repeated with `np.repeat` while the human labels were upsampled with `np.round` indices (different grid, one-sample shifts on half of the transitions) | only native-rate predictors on d3 and Andersson: U'n'Eye on Andersson with its own weights 0.890 / 0.815 -> 0.881 / 0.822, general weights 0.550 / 0.332 -> 0.535 / 0.337, d3 0.929 / 0.815 -> 0.930 / 0.821; BiTCN rows (scored at 1 kHz) unaffected | fixed, reference rows recomputed |
| `foundation/unet_ssl.py` `build_pool` | the pre-training pool contained unlabeled windows of dataset 1 set A, which is the few-label TEST set | pre-trained rows of the few-label table may have been slightly optimistic | pool now uses set B for d1; re-run of pre-training, fine-tuning, multi-scale, layer statistics, freezing and PEFT (`foundation/rerun_audit.py`) |
| `foundation/sup_bitcn.py` | the EMA weights were checkpointed at the step where the plain weights peaked (EMA still immature): the row `sup_bitcn_b_ema_tta` (0.74 / 0.71) was an artifact; the headline row `sup_bitcn_b_tta` uses the plain weights and was not affected | EMA rows of the 8-channel run only; docs wrongly called the headline "EMA+TTA" | checkpoints decoupled, wording corrected; EMA row re-run: `sup_bitcn_b_ema_tta` 0.74 / 0.71 -> 0.84 / 0.78 (plain weights with TTA unchanged, 0.89 / 0.81) |
| `foundation/make_summary.py` | the novelty-curve row of table 2 used the threshold readout (F1 0.53 / kappa 0.00) instead of the only meaningful one (HMM, F1 0.33 / kappa 0.20); row 15 of table 1 hid that it is the 8-pass TTA run | table 2 and table 1 labels | fixed |
| `foundation/hmm_improvements.py` | the boundary search took the first / last sample above the fraction in an extended window (noise stretched it) | kappa of the "boundary at a fraction of the peak" rule: d1 0.740 -> 0.758, d3 0.608 -> 0.647, d4 0.772 -> 0.802, Andersson 0.332 -> 0.589; still below the constant shift on d1-d4 | fixed and re-run, roadmap updated |
| `foundation/dino1d.py` `attention_maps` and its copy `foundation/dino_eval.py` `sliding` | features computed on the NaN-padded window (noise scale and first samples polluted). The fix was first applied to `dino1d.py` only and a re-run of `dino_eval.py` was reported as "after the fix" although `dino_eval.py` has its own copy of the function (found later with the branch review); both are now fixed and re-run | CLS attention AUC 0.138 / 0.258 / 0.191 / 0.109 / 0.052 (d1 / d2 / d3 / d4 / Andersson) against 0.138 / 0.245 / 0.192 / 0.110 / 0.052 before the real fix: no measurable effect (< 0.015), the attention is still anti-correlated with saccades | fixed (both files) and re-run |
| `foundation/ssl_encoder.py` `--hard` | the hard negatives of all references were added to one shared pool instead of one per reference | the "hard negative did not help" test of the CEBRA-style encoder | fixed and re-run: event F1 0.56 / 0.50 / 0.81 / 0.54 / 0.54 -> 0.57 / 0.48 / 0.84 / 0.63 / 0.61, still below the plain input features (same conclusion) |
| `foundation/hmm_score.py` | `"sup" in name` is also true for `emb-unsup`: a meaningless 0.5-threshold row | one extra row of an exploratory table | fixed |
| `free_saccade/detectors.py` `HMM.fit` | stay probability of the last chain state used the wrong denominator | universal HMM F1 0.881 -> 0.882, kappa 0.656 -> 0.656: negligible (the label-free student was not regenerated) | fixed |
| `cpp/scripts/calibrate_ood.py` | "current" run compared with the calibrated file, so the script could not reproduce its own table | none on reported numbers (the numbers in `docs/safety/hazards.md` come from the original run) | fixed |
| `foundation/uneye_curves.py` | a run that never improves would be scored with the weights of the previous draw | none: the logs show every one of the 18 + 21 runs saved its weights | guard added |
| docs | the DINO section said "trained ONLY on archive/" while the pool also had the unlabeled benchmark train splits | wording | fixed |

## Review of the code added on the branch `vibe/lost-ncut-memory-readouts-d0df29` (`foundation/lost1d.py`, written by another agent)
Reviewed by `agy` before running (first review truncated, second one complete), every finding checked in the code. Fixed: PatchCore took the MINIMUM over the memory chunks of the best similarity (a nearest neighbour needs the maximum); TokenCut used `D - A_n` instead of the normalised Laplacian `I - A_n`; the arbitrary sign of the Fiedler vector was not oriented (overlapping windows cancelled: now TokenCut's own rule, largest |value| positive, no label); keys were averaged over the heads instead of concatenated (LOST / TokenCut); features were computed on the NaN-padded window (same flaw as `dino_eval.py`, also fixed there); k-means could start from two identical tokens. Not changed (design limits, documented): k-means always splits a fixation window in two, LOST always picks a seed. Result: no readout beats the speed channel (docs/BENCHMARK_SUMMARY.md section 6).

## Rejected (checked, not errors, or no effect)
- `hmm_improvements.py:27` "peak index adds `(s - lo)` twice": wrong algebra, `lo + (s - lo) = s`, the index is right.
- `hmm_improvements.py` semi-Markov transition remark: not demonstrated.
- `dino1d.py`: missing `--eval` flag (stale docstring, evaluation is `dino_eval.py`), positional table of 256 tokens, padding index remarks: not used by any reported number.
- `data.py`: `n` argument ordering (never used), 2T-1 samples after upsampling a 500 Hz trial (1 ms at the end, handled by `to_1khz` and consistent for all models).
- `compare.py`: the argmax-versus-threshold question on Andersson (0.890 / 0.815 against 0.883 / 0.817: negligible).

## Earlier review (`cpp/`, `foundation/`): see docs/BENCHMARK_SUMMARY.md section 7.
