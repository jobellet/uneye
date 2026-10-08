import numpy as np
import torch

DATA = "../data/"
SETS = {"1": ("dataset1", "dataset1_1000hz", 1000), "2": ("dataset2", "dataset2_1000hz", 1000),
        "3": ("dataset3", "dataset3_500hz", 500)}


def load(setname, which):
    d, f, fs = SETS[setname]
    p = lambda k: np.loadtxt(f"{DATA}{d}/{f}_{k}_set{which}.csv", delimiter=",")
    return p("X"), p("Y"), p("Labels"), fs


def velocity(X, Y, inf_correction=1.5):
    """differentiated signal exactly as in uneye/classifier.py (first sample 0, NaN->0, Inf->1.5)"""
    n = X.shape[0]
    dx = np.concatenate((np.zeros((n, 1)), np.diff(X, axis=-1)), 1)
    dy = np.concatenate((np.zeros((n, 1)), np.diff(Y, axis=-1)), 1)
    for d in (dx, dy):
        d[np.isinf(d)] = inf_correction
        d[np.isnan(d)] = 0
    return np.stack([dx, dy], 1).astype(np.float32)  # (n, 2, T)


def mc_loss(pred, target):
    """MCLoss of uneye/functions.py"""
    eps = 1e-7
    pred = torch.clamp(pred, min=eps, max=1 - eps)
    return -torch.mean(target * torch.log(pred))
