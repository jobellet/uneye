# cpp/src/ — implementations
`bitcn.cpp` BiTCN + `features2` (mirrors `foundation/dino1d.py::feats2` and `foundation/bitcn.py`); `causal_engine.cpp` streaming TCN (BatchNorm folded); `native_engine.cpp` pure C++ U-Net (mirrors `uneye/functions.py::UNet`);
`onnx_engine.cpp` optional ONNX Runtime back-end (`-DONNXRUNTIME_ROOT`); `detector.cpp` post-processing as in `uneye/classifier.py::predict`; `gaze_engine.cpp`, `safety_guard.cpp` safety-relevant, zero allocation, must stay warning-free.
Any change to a network or feature computation must keep the matching test in `../tests/` green (golden values come from PyTorch).
