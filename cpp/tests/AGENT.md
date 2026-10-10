# cpp/tests/ — ctest executables and golden files
`test_parity` (native/ONNX vs PyTorch U-Net, `golden_*.txt`), `test_equivalence` (causal TCN, `golden_equivalence_causal.bin`), `test_zero_heap` (no allocation in steady state), `test_gaze_engine` (determinism, hostile input), `test_safety_guard` (one test per check), `test_bitcn` (`golden_bitcn_2ch.txt`).
Regenerate goldens with `cpp/scripts/make_golden.py`, `cpp/scripts/equivalence.py`, `foundation/export_bitcn.py`; never edit them by hand.
