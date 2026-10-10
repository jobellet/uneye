# cpp/scripts/ — Python helpers for the C++ side
`export_model.py` PyTorch U-Net -> `.onnx` + `.bin`; `make_golden.py` reference outputs for `test_parity`; `equivalence.py` causal TCN Python-vs-C++; `fit_landing_tables.py` -> `include/uneye/gaze_tables.hpp`;
`training_range.py` and `calibrate_ood.py` input range and out-of-distribution noise limit of the guard (calibrated without the test set). The BiTCN export lives in `foundation/export_bitcn.py`.
