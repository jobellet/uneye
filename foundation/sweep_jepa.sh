#!/bin/sh
# sequential sweep of I-JEPA variants (one GPU job at a time); each stops early on the validation event F1 (probe + HMM, labeled TRAIN splits only)
cd "$(dirname "$0")/.."
PY=.venv/bin/python
$PY -u foundation/ssl_vit.py --name jepa_a                                  > foundation/runs/ssl_jepa_a.log 2>&1   # lr 3e-4, blocks of 12 tokens, patch 4, EMA 0.996
$PY -u foundation/ssl_vit.py --name jepa_b --lr 1e-4                         > foundation/runs/ssl_jepa_b.log 2>&1   # lower learning rate
$PY -u foundation/ssl_vit.py --name jepa_c --patch 2 --block 12              > foundation/runs/ssl_jepa_c.log 2>&1   # finer patches (2 ms), blocks of 24 ms
$PY -u foundation/ssl_vit.py --name jepa_d --block 6 --ema 0.99              > foundation/runs/ssl_jepa_d.log 2>&1   # blocks of 24 ms, faster EMA
