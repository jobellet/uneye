# foundation/runs/ — outputs, not source
Checkpoints `*.pt` (gitignored), caches (`pool.npz` training pool, `pseudo_hmm.npy` HMM pseudo-labels, `hubert_km_p4.npz`), small JSON results (`ssl_*.json`, `*_f1.json`), logs.
`night/<id>.json`: one result per candidate (schema in `foundation/night_eval.py::save`), `night/logs/<id>.log`: training log, `night/status*.json`: queue state, `label_eff.json`.
Do not read `.pt`/`.npz` files; read the JSON or the log. Names: `sup_bitcn_v2` = 2-channel BiTCN (best), `sup_bitcn_b` = 8-channel, `selftrain_b` = label-free student, `ssl_<name>` = self-supervised encoders.
