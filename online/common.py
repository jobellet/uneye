import numpy as np
import torch

DATA = "../data/"
SETS = {"1": ("dataset1", "dataset1_1000hz", 1000), "2": ("dataset2", "dataset2_1000hz", 1000),
        "3": ("dataset3", "dataset3_500hz", 500)}


def load(setname, which):
    d, f, fs = SETS[setname]
    p = lambda k: np.loadtxt(f"{DATA}{d}/{f}_{k}_set{which}.csv", delimiter=",")
    return p("X"), p("Y"), p("Labels"), fs


MAX_SPEED_DEG_S = 1300.0   # speed limit of the network input (deg/s), see uneye.functions.clip_speed


def velocity(X, Y, inf_correction=1.5, fs=1000.0, max_speed=MAX_SPEED_DEG_S):
    """differentiated signal exactly as in uneye/classifier.py (first sample 0, NaN->0, Inf->1.5, then the speed limited to
    max_speed deg/s with the direction kept; fs = sampling rate of X, Y; max_speed None or 0: no limit)"""
    n = X.shape[0]
    dx = np.concatenate((np.zeros((n, 1)), np.diff(X, axis=-1)), 1)
    dy = np.concatenate((np.zeros((n, 1)), np.diff(Y, axis=-1)), 1)
    for d in (dx, dy):
        d[np.isinf(d)] = inf_correction
        d[np.isnan(d)] = 0
    from uneye.functions import clip_speed
    dx, dy = clip_speed(dx, dy, max_speed, fs)
    return np.stack([dx, dy], 1).astype(np.float32)  # (n, 2, T)


def mc_loss(pred, target, class_weight=None):
    """MCLoss of uneye/functions.py"""
    eps = 1e-7
    pred = torch.clamp(pred, min=eps, max=1 - eps)
    t = target if class_weight is None else target * class_weight
    return -torch.mean(t * torch.log(pred))
