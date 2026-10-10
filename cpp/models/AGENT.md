# cpp/models/ — exported networks (binary; do not open)
`<name>.bin` weights for the native engine, `<name>.onnx` for ONNX Runtime, `<name>.json` metadata. `bitcn_sup_bitcn_v2.*` best detector (2 channels vx, vy, offline, receptive field 515 samples, threshold 0 on the logit, optional TTA = 4 rotations x mirror); `causal.*` streaming; `combined` = datasets 1+2+3; `andersson` 5-class.
Re-export with `cpp/scripts/export_model.py` or `foundation/export_bitcn.py`.
