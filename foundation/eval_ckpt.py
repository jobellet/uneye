"""Load a self-supervised checkpoint foundation/runs/ssl_<name>.pt (I-JEPA / MAE / HuBERT transformer or TS2Vec) and evaluate it on the common test subsets.
Used for a run made before the final evaluation was added to the training script, and by label_eff.py.   python foundation/eval_ckpt.py --name jepa_a
"""
import argparse, os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "online"))
import numpy as np, torch
from foundation import dino1d as DN, ssl_probe as SP, night_eval as NE
from foundation.train_lodo import DEV


def load_feature_fn(name):
    """returns (feature_fn(pos) -> (n, T, C), args dict)"""
    ck = torch.load(os.path.join(ROOT, "foundation", "runs", f"ssl_{name}.pt"), weights_only=False); a = ck["args"]
    if "hid" in a:                                                          # TS2Vec
        from foundation.ts2vec import TSEncoder
        net = TSEncoder(hid=a["hid"], dim=a["dim"], depth=a["depth"]).to(DEV); net.load_state_dict(ck["state"]); net.eval()
        return (lambda pos: SP.sliding_tokens(lambda f: net(f, mask="none"), pos, 1)), a
    from foundation.ssl_vit import TokViT                                   # transformer encoder
    enc = TokViT(a["patch"], depth=a["depth"]).to(DEV); enc.load_state_dict(ck["state"]); enc.eval()
    return (lambda pos: SP.sliding_tokens(lambda f: enc.encode(enc.tokens(f)), pos, a["patch"])), a


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--name", required=True); ap.add_argument("--paper", default="Assran et al. 2023 (I-JEPA), Bardes et al. 2024 (V-JEPA)"); a = ap.parse_args()
    t0 = time.time(); fn, args = load_feature_fn(a.name); res = NE.eval_features(fn)
    NE.save(a.name, "self-supervised representation + linear probe + HMM", "labels of the other 4 datasets (linear probe only); the encoder itself saw no label", a.paper, args, res, (time.time() - t0) / 60)
