"""Linear readout of the self-supervised embedding vs the plain input features: sample AUC and best event F1 (single threshold chosen
WITH the labels of set B among 6 quantiles, plus a 3 ms clean-up: an optimistic bound, not a label-free result).
Run from the repository root:  python foundation/diagnose_embedding.py [encoder.pt]"""
import sys, os; ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'online'))
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from foundation import data as FD, ssl_encoder as SE, compare as CP
from foundation.prior_hyperplane import Rec
from free_saccade import detectors as D, cebra_seed as C
net = SE.load_encoder(sys.argv[1] if len(sys.argv) > 1 else None)
def clean(lab): return D.cleanup(lab, 1000.0, min_ms=3.0, merge_ms=3.0)
for nm in FD.ALL:
    A0, B = FD.load(nm,"train"), FD.load(nm,"test")
    sel = np.random.RandomState(1).permutation(len(A0))[:300 if nm!="andersson" else 80]
    A = FD.Set(nm, A0.pos[sel], A0.lab[sel], A0.coarse)
    RA, RB = Rec(A, net), Rec(B, net)
    FA, FB = C.feats(A.pos,1000.0).reshape(-1,6), C.feats(B.pos,1000.0).reshape(-1,6)
    mA = ((A.lab>=0)&~RA.bad).ravel(); mB = ((B.lab>=0)&~RB.bad).ravel()
    yA, yB = (A.lab==1).ravel()[mA], (B.lab==1).ravel()[mB]
    out=[]
    for name,XA,XB in (("embedding",RA.Zf,RB.Zf),("input features",FA,FB),("detrended speed only",FA[:,5:6],FB[:,5:6]),("speed only",FA[:,0:1],FB[:,0:1])):
        lr = LogisticRegression(max_iter=500, class_weight="balanced").fit(XA[mA][::3], yA[::3])
        s = XB@lr.coef_[0]+lr.intercept_[0]; auc = roc_auc_score(yB, s[mB])
        best=0
        for q in (0.9,0.93,0.95,0.97,0.98,0.99):
            thr=np.quantile(s[mB],q); lab=clean((s>thr).reshape(B.lab.shape)&~RB.bad); best=max(best,CP.event_f1(lab,B)["ev_f1"])
        out.append(f"{name}: sample AUC {auc:.3f}, best event F1 {best:.3f}")
    print(f"[{nm}] "+" | ".join(out), flush=True)
