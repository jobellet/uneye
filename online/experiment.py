"""Label-efficiency experiments: data, supervised training / fine-tuning, SSL pretraining, evaluation.

Protocol (as in uneye.DNN): train data = set A of datasets 1,2,3; test = set B (never used for any choice).
With a budget of N labeled trials, 20 % (at least 3) of them are used for early stopping / threshold tuning.
SSL and weak-label pretraining see ALL set-A recordings but no human labels.
"""
import copy, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import archs
from common import load, velocity, mc_loss
from jepa import JEPA, random_rotate
from metrics import evaluate_probs, tune_threshold, pool
from weak_labels import engbert_kliegl

SETS = ("1", "2", "3")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = True      # fixed shapes: let cuDNN pick the fastest kernels


# ----------------------------------------------------------------------------- data
class Data:
    """all trials as velocity tensors. trials[(split, set)] = (V (n,2,T), L (n,T), fs)"""

    def __init__(self, sets=SETS, n_test=300, extra_unlabeled=False, data_dir=None):
        self.kwargs = dict(sets=sets, n_test=n_test, extra_unlabeled=extra_unlabeled, data_dir=data_dir)   # to rebuild the data in worker processes
        if data_dir:
            import common; common.DATA = data_dir
        self.sets, self.tr, self.te, self.raw = sets, {}, {}, {}
        rng = np.random.RandomState(10)
        for s in sets:
            X, Y, L, fs = load(s, "A")
            self.tr[s] = (velocity(X, Y), (L > 0).astype(np.float32), fs)
            self.raw[s] = (X, Y, fs)
            X, Y, L, fs = load(s, "B")
            ind = rng.permutation(X.shape[0])[:n_test]
            self.te[s] = (velocity(X[ind], Y[ind]), (L[ind] > 0).astype(np.float32), fs)
        self.extra = []
        if extra_unlabeled:  # dataset 4 (1 kHz, 750 samples): used ONLY as unlabeled recordings
            d = os.path.join(common.DATA if data_dir else "../data/", "dataset4")
            X = np.loadtxt(f"{d}/dataset4_1000hz_X_setA.csv", delimiter=","); Y = np.loadtxt(f"{d}/dataset4_1000hz_Y_setA.csv", delimiter=",")
            self.extra = [velocity(X, Y)]

    def unlabeled(self):
        return [self.tr[s][0] for s in self.sets] + self.extra

    def weak(self):
        return {s: (self.tr[s][0], engbert_kliegl(self.raw[s][0], self.raw[s][1], self.raw[s][2])) for s in self.sets}

    def labeled_budget(self, n, seed):
        """n labeled trials drawn at random from the pooled set-A trials -> (train, val) = lists of (V, L)"""
        rng = np.random.RandomState(seed)
        pool_idx = [(s, i) for s in self.sets for i in range(self.tr[s][0].shape[0])]
        sel = [pool_idx[j] for j in rng.permutation(len(pool_idx))[:n]]
        nv = max(3, n // 5)
        def group(items):
            out = {}
            for s, i in items: out.setdefault(s, []).append(i)
            return [(self.tr[s][0][ix], self.tr[s][1][ix]) for s, ix in out.items()]
        return group(sel[nv:]), group(sel[:nv])


class Bank:
    """All trials of a list of (V (n,2,T), L (n,T) or None) groups, kept ON THE COMPUTE DEVICE.
    A batch is drawn with a few gather operations: no Python loop over trials, no host->device copy per step."""

    def __init__(self, groups, device=None, rotate=True):
        device = device or DEVICE
        self.V = [torch.as_tensor(V, device=device) for V, _ in groups]
        self.L = [None if L is None else torch.as_tensor(L, device=device) for _, L in groups]
        n = np.array([v.shape[0] for v in self.V], float)
        self.p, self.rotate, self.device = n / n.sum(), rotate, device
        self.ch = torch.arange(2, device=device)[None, :, None]

    def sample(self, batch, crop):
        T = min(crop, min(v.shape[2] for v in self.V))
        counts = np.random.multinomial(batch, self.p)          # how many trials from each group (CPU, no GPU sync)
        pos0 = torch.arange(T, device=self.device)
        vs, ls = [], []
        for k, c in enumerate(counts):
            if c == 0: continue
            V, L = self.V[k], self.L[k]
            i = torch.randint(V.shape[0], (c,), device=self.device)
            s = torch.randint(V.shape[2] - T + 1, (c,), device=self.device)
            pos = s[:, None] + pos0                                # (c,T)
            vs.append(V[i[:, None, None], self.ch, pos[:, None, :]])
            if L is not None: ls.append(L[i[:, None], pos])
        v = torch.cat(vs)
        if self.rotate: v = random_rotate(v)
        return v, (torch.cat(ls) if ls else None)


def onehot(L):
    return torch.stack([1 - L, L], 1)


# ----------------------------------------------------------------------------- supervised training / fine-tuning
def train_supervised(model, train, val, pos_weight=3.0, lr=1e-3, l2=1e-4, max_epochs=60, steps=20, batch=32,
                     crop=700, patience=4, min_epochs=15, freeze_backbone=False, log=None):
    """original uneye logic: Adam, lr halved when validation gets worse (best weights restored), stop after > patience bad epochs
    (not before `min_epochs`, so that every strategy, including training from scratch, gets the same minimum training).
    One 'epoch' = `steps` random minibatches (so that epochs are comparable for any number of labeled trials).
    No lr decay / early stopping before `min_epochs`: otherwise a from-scratch net, which first sits on the
    'predict no saccade' plateau, is stopped before it learns anything and the baseline is unfairly weak.
    Speed: data lives on the device (Bank), no per-step host sync."""
    model.to(DEVICE)
    if freeze_backbone:
        for p in model.backbone.parameters(): p.requires_grad = False
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)
    cw = torch.tensor([1.0, pos_weight]).view(1, 2, 1).to(DEVICE)
    bank = train if isinstance(train, Bank) else Bank(train, DEVICE, rotate=True)
    vt = [(torch.from_numpy(V).to(DEVICE), onehot(torch.from_numpy(L).to(DEVICE))) for V, L in val]

    def val_loss():
        model.eval()
        with torch.no_grad():
            n = sum(v.shape[0] for v, _ in vt)
            return float(sum(mc_loss(model(v), t, cw) * v.shape[0] for v, t in vt) / n)

    best, best_w, bad = None, None, 0
    for ep in range(1, max_epochs + 1):
        model.train()
        if freeze_backbone: model.backbone.eval()
        for _ in range(steps):
            v, l = bank.sample(batch, crop)
            loss = mc_loss(model(v), onehot(l), cw)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        vl = val_loss()
        if best is None or vl < best:
            best, bad, best_w = vl, 0, copy.deepcopy(model.state_dict())
        elif ep <= min_epochs:
            pass  # initial plateau (a from-scratch net first predicts 'no saccade'): no lr decay / early stopping yet
        else:
            bad += 1; model.load_state_dict(best_w)
            for g in opt.param_groups: g["lr"] *= 0.5
        if log: log(f"  ep {ep:3d} val {vl:.4f} best {best:.4f} bad {bad}")
        if bad > patience: break
    model.load_state_dict(best_w)
    return model


def pretrain_jepa(backbone, data, epochs=40, steps=30, batch=48, crop=500, lr=2e-3, target="latent", horizons=(0, 5, 10, 20),
                  mask_frac=0.3, log=None):
    """label-free pretraining on all set-A recordings (and dataset 4 if loaded). Returns the context backbone + loss history."""
    bank = Bank([(V, None) for V in data.unlabeled()], DEVICE, rotate=False)
    jepa = JEPA(backbone, horizons, target=target).to(DEVICE)
    opt = torch.optim.AdamW([p for p in jepa.parameters() if p.requires_grad], lr=lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * steps)
    hist = []
    for ep in range(epochs):
        jepa.train(); acc = []
        for _ in range(steps):
            v, _ = bank.sample(batch, crop)
            loss, stats = jepa.loss(v, mask_frac)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
            if target == "latent": jepa.update_target()
            acc.append(stats)
        m = torch.stack(acc).mean(0).tolist()                      # one host sync per epoch
        hist.append(dict(zip(("pred", "var", "emb_std"), m)))
        if log: log(f"  jepa ep {ep+1:3d} " + " ".join(f"{k} {v:.4f}" for k, v in hist[-1].items()))
    return backbone, hist


def pretrain_weak(model, data, **kw):
    """supervised pretraining on Engbert-Kliegl pseudo-labels of ALL set-A recordings (no human labels)"""
    w = data.weak()
    nv = 30
    train = [(V[nv:], L[nv:]) for V, L in w.values()]; val = [(V[:nv], L[:nv]) for V, L in w.values()]
    return train_supervised(model, train, val, **kw)


# ----------------------------------------------------------------------------- prediction / evaluation
@torch.no_grad()
def predict(model, V, bs=64):
    model.eval(); dev = next(model.parameters()).device
    return np.concatenate([model(torch.from_numpy(V[i:i + bs]).to(dev))[:, 1].cpu().numpy() for i in range(0, V.shape[0], bs)])


def evaluate(model, data, val=None, thr=None):
    """per-dataset + pooled metrics on set B. If `val` (list of (V,L)) is given also the kappa-tuned threshold result."""
    out = {}
    tuned = None
    if val is not None and thr is None:
        Pv = np.concatenate([predict(model, V).ravel() for V, _ in val]); Lv = np.concatenate([L.ravel() for _, L in val])
        tuned = tune_threshold(Pv, Lv, "kappa")
    for tag, t in (("0.5", 0.5), ("tuned", tuned)):
        if t is None: continue
        res = []
        for s in data.sets:
            V, L, fs = data.te[s]
            res.append(evaluate_probs(predict(model, V), L, fs, t, with_ap=(tag == "0.5")) | {})
        out[tag] = {"per_set": dict(zip(data.sets, res)), "pooled": pool(res), "thr": t}
    return out


def unet_lastbin(data, weights="../training/weights_1+2+3"):
    """original U'n'Eye evaluated on the last sample of a 200 ms bin (reference)"""
    from uneye.functions import UNet
    net = UNet(2, 5, 5); net.load_state_dict(torch.load(weights, weights_only=False, map_location=DEVICE)); net.to(DEVICE).eval()
    res = []
    for s in data.sets:
        V, L, fs = data.te[s]
        W = (int(round(0.2 * fs)) + 24) // 25 * 25
        n, _, T = V.shape
        P = np.zeros((n, T), np.float32)
        pad = np.concatenate([np.zeros((n, 2, W - 1), np.float32), V], 2)
        with torch.no_grad():
            for i in range(n):
                x = torch.from_numpy(pad[i]).to(DEVICE).unfold(1, W, 1).permute(1, 2, 0).unsqueeze(1).contiguous()
                for j in range(0, T, 500):
                    P[i, j:j + 500] = net(x[j:j + 500], ["out"])[0][:, 1, -1].cpu().numpy()
        res.append(evaluate_probs(P, L, fs, 0.5))
    return pool(res), res


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def latency_ms(model, window=200, reps=50):
    """CPU time to label one 200 ms window (batch 1, one thread) in PyTorch eager mode"""
    model = copy.deepcopy(model).cpu().eval(); torch.set_num_threads(1); x = torch.randn(1, 2, window) * 0.05
    with torch.no_grad():
        for _ in range(5): model(x)
        t = time.perf_counter()
        for _ in range(reps): model(x)
    return (time.perf_counter() - t) / reps * 1000


# ----------------------------------------------------------------------------- the grid
STRATEGIES = ("scratch", "jepa_ft", "jepa_probe", "recon_ft", "weak_ft")
FULL = dict(pre_epochs=40, pre_steps=30, ft_epochs=60, ft_steps=20)
QUICK = dict(pre_epochs=2, pre_steps=5, ft_epochs=3, ft_steps=5)
_PRE_KIND = {"jepa_ft": "jepa", "jepa_probe": "jepa", "recon_ft": "recon", "weak_ft": "weak"}


def _seed(s):
    np.random.seed(s); torch.manual_seed(s)


def default_devices(workers_per_gpu=2):
    """one worker slot per GPU x workers_per_gpu (tiny models cannot fill a GPU alone); on CPU one slot per 2 cores"""
    cores = os.cpu_count() or 2
    if torch.cuda.is_available():
        devs = [f"cuda:{i}" for i in range(torch.cuda.device_count())] * workers_per_gpu
        return devs[:max(cores, torch.cuda.device_count())]       # never more worker processes than CPU cores (each needs a core)
    return ["cpu"] * max(1, cores // 2)


# --- jobs: module-level functions so that they can run in worker processes (they use the process-global _DATA / DEVICE)
_DATA = None


def _init_worker(dev_queue, data_kwargs):
    global DEVICE, _DATA
    DEVICE = dev_queue.get()
    torch.set_num_threads(1)                      # several processes: avoid CPU oversubscription
    if DEVICE.startswith("cuda"): torch.cuda.set_device(DEVICE)
    _DATA = Data(**data_kwargs)


def _job_pretrain(kind, bb, channels, cfg):
    _seed(123); t = time.time()
    net = archs.build(bb, channels)
    if kind == "weak":
        if bb.startswith("tcn"): archs.init_like_uneye(net)
        pretrain_weak(net, _DATA, max_epochs=cfg["ft_epochs"], steps=cfg["ft_steps"])
        state, h = net.state_dict(), None
    else:
        b, h = pretrain_jepa(net.backbone, _DATA, cfg["pre_epochs"], cfg["pre_steps"], target="latent" if kind == "jepa" else "raw")
        state = b.state_dict()
    return (bb, kind), {k: v.cpu() for k, v in state.items()}, h, time.time() - t


def _job_finetune(bb, st, n, sd, channels, cfg, state):
    _seed(1000 + sd); t = time.time()
    train, val = _DATA.labeled_budget(n, sd)
    net = archs.build(bb, channels)
    if st == "scratch":
        if bb.startswith("tcn"): archs.init_like_uneye(net)
    elif st in ("jepa_ft", "jepa_probe", "recon_ft"): net.backbone.load_state_dict(state)
    elif st == "weak_ft": net.load_state_dict(state)
    train_supervised(net, train, val, max_epochs=cfg["ft_epochs"], steps=cfg["ft_steps"],
                     freeze_backbone=(st == "jepa_probe"), lr=(1e-3 if st in ("scratch", "jepa_probe") else 5e-4))
    r = evaluate(net, _DATA, val)
    row = dict(backbone=bb, strategy=st, n_labels=n, seed=sd, params=count_params(net), seconds=time.time() - t)
    for tag in r:
        for k, v in r[tag]["pooled"].items(): row[f"{k}@{tag}"] = v
        row[f"thr@{tag}"] = r[tag]["thr"]
        for s_, m in r[tag]["per_set"].items(): row[f"kappa_set{s_}@{tag}"] = m["kappa"]
    return row


class JobRunner:
    """Runs independent jobs, one worker process per entry of `devices` (spawned once, data loaded once per worker).
    len(devices) <= 1 -> jobs run in this process."""

    def __init__(self, data, devices, log=print):
        global _DATA
        self.parallel, self.ex = len(devices) > 1, None
        if not self.parallel:
            _DATA = data
            return
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor
        ctx = mp.get_context("spawn")
        q = ctx.Queue()
        for d in devices: q.put(d)
        log(f"[parallel] {len(devices)} worker processes on {sorted(set(devices))}")
        self.ex = ProcessPoolExecutor(len(devices), mp_context=ctx, initializer=_init_worker, initargs=(q, data.kwargs))

    def run(self, jobs):  # yields results as they finish
        if not self.parallel:
            for fn, args in jobs: yield fn(*args)
            return
        from concurrent.futures import as_completed
        for f in as_completed([self.ex.submit(fn, *args) for fn, args in jobs]):
            yield f.result()

    def close(self):
        if self.ex: self.ex.shutdown()


def run_grid(data, backbones=("tcn",), strategies=("scratch", "jepa_ft"), n_labels=(5, 20, 100), seeds=(0,), cfg=None,
             channels=48, log=print, save=None, devices=None, resume=True):
    """Returns (rows, hist): one row per backbone x strategy x N x seed.
    devices=None -> this process only; devices=default_devices() -> independent jobs run in parallel on all GPUs
    (pretraining of every backbone/kind first, then all fine-tuning runs, longest first).
    resume=True: runs already stored in `save` (json written after every finished run) are kept and skipped, so an interrupted
    or slow run can be restarted without losing finished work."""
    import json
    cfg = dict(FULL, **(cfg or {}))
    devices = devices or [DEVICE]
    rows, pre, hist = [], {}, {}
    if resume and save and os.path.exists(save):
        rows = json.load(open(save))
    done = {(r["backbone"], r["strategy"], r["n_labels"], r["seed"]) for r in rows}
    todo = [(bb, st, n, sd) for bb in backbones for st in strategies for n in n_labels for sd in seeds if (bb, st, n, sd) not in done]
    if done: log(f"[resume] {len(done)} runs already finished, {len(todo)} left")
    kinds = sorted({(bb, _PRE_KIND[st]) for bb, st, _, _ in todo if st in _PRE_KIND})
    t0 = time.time()
    runner = JobRunner(data, devices, log)
    for (bb, kind), state, h, secs in runner.run([(_job_pretrain, (kind, bb, channels, cfg)) for bb, kind in kinds]):
        pre[(bb, kind)], hist[(bb, kind)] = state, h
        log(f"[pretrain] {bb} {kind}: {secs:.0f}s" + (f", loss {h[0]['pred']:.3f} -> {h[-1]['pred']:.3f}, emb std {h[-1]['emb_std']:.2f}" if h else ""))
    cost = lambda bb, n: (n + 20) * (4 if bb in ("s4d", "gru") else 1)
    jobs = [(_job_finetune, (bb, st, n, sd, channels, cfg, pre.get((bb, _PRE_KIND.get(st))))) for bb, st, n, sd in todo]
    jobs.sort(key=lambda j: -cost(j[1][0], j[1][2]))
    for row in runner.run(jobs):
        rows.append(row)
        log(f"{row['backbone']:9s} {row['strategy']:10s} N={row['n_labels']:4d} seed {row['seed']}: kappa {row['kappa@0.5']:.3f} "
            f"(tuned {row['kappa@tuned']:.3f}) mcc {row['mcc@0.5']:.3f} f1 {row['f1@0.5']:.3f} ev_f1 {row['ev_f1@0.5']:.3f}  "
            f"[{row['seconds']:.0f}s, {len(rows) - len(done)}/{len(jobs)}, elapsed {(time.time() - t0) / 60:.1f} min]")
        if save: json.dump(rows, open(save, "w"))
    runner.close()
    return rows, hist
