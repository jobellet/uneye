"""Which causal network, trained how, for my project? (v2: finalists only)

Protocol (as in uneye.DNN): train data = set A of datasets 1,2,3; test = set B (never used for any choice).
With a budget of N labeled trials, 20 % (at least 3) of them are used for early stopping / threshold tuning.

Strategies for the causal student (all end with supervised training on the N labeled trials):
  scratch   random initialisation
  weak_ft   first trained on free Engbert-Kliegl pseudo-labels of ALL set-A recordings, then fine-tuned
  distill   teacher-student: the original (non-causal) U'n'Eye is trained on the N labels, labels ALL set-A recordings
            (soft probabilities), and the causal student learns from these + the true labels
Delay: `lookahead_ms` > 0 trains the network to output the label of sample t-L at time t (L = lookahead): still real time,
but L ms late, and the network gets L ms of 'future' context.
References: the original U'n'Eye trained on the same N trials, offline and on the last sample of a 200 ms bin.
"""
import copy, json, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import archs
from common import load, velocity, mc_loss
from metrics import evaluate_probs, tune_threshold, pool
from weak_labels import engbert_kliegl

SETS = ("1", "2", "3")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = True      # fixed shapes: let cuDNN pick the fastest kernels
NAN_METRICS = ("kappa@tuned", "mcc@tuned", "f1@tuned")


def random_rotate(v):
    """rotate the velocity vector by a random angle per trial (= rotating the position trace: data augmentation of U'n'Eye)"""
    a = torch.rand(v.shape[0], device=v.device) * 2 * np.pi
    c, s = torch.cos(a)[:, None], torch.sin(a)[:, None]
    return torch.stack([v[:, 0] * c + v[:, 1] * s, -v[:, 0] * s + v[:, 1] * c], 1)


def shift_samples(lookahead_ms, fs):
    return int(round(lookahead_ms * fs / 1000.0))


def shift_labels(L, k):
    """label of sample t-k placed at t: the network output at time t is trained to say what happened k samples ago"""
    if k <= 0: return L
    return np.concatenate([np.zeros((L.shape[0], k), L.dtype), L[:, :-k]], 1)


# ----------------------------------------------------------------------------- data
class Data:
    """tr[s] = (V (n,2,T) velocities, L (n,T) human labels, fs) for set A; te[s] likewise for set B (300 random trials);
    te_pos[s] = (X, Y) test positions (for plots)"""

    def __init__(self, sets=SETS, n_test=300, data_dir=None):
        self.kwargs = dict(sets=sets, n_test=n_test, data_dir=data_dir)   # to rebuild the data in worker processes
        if data_dir:
            import common; common.DATA = data_dir
        self.sets, self.tr, self.te, self.te_pos, self.raw = sets, {}, {}, {}, {}
        rng = np.random.RandomState(10)
        for s in sets:
            X, Y, L, fs = load(s, "A")
            self.tr[s] = (velocity(X, Y), (L > 0).astype(np.float32), fs)
            self.raw[s] = (X, Y, fs)
            X, Y, L, fs = load(s, "B")
            ind = rng.permutation(X.shape[0])[:n_test]
            self.te[s] = (velocity(X[ind], Y[ind]), (L[ind] > 0).astype(np.float32), fs)
            self.te_pos[s] = (X[ind], Y[ind])

    def weak(self):
        return {s: (self.tr[s][0], engbert_kliegl(self.raw[s][0], self.raw[s][1], self.raw[s][2])) for s in self.sets}

    def split(self, n, seed):
        """n labeled trials drawn at random from the pooled set-A trials -> (train_ids, val_ids), ids = (set, index)"""
        rng = np.random.RandomState(seed)
        pool_idx = [(s, i) for s in self.sets for i in range(self.tr[s][0].shape[0])]
        sel = [pool_idx[j] for j in rng.permutation(len(pool_idx))[:n]]
        nv = max(3, n // 5)
        return sel[nv:], sel[:nv]

    def groups(self, ids, lookahead_ms=0.0):
        """list of (V, L) per dataset for the trials `ids`; labels shifted by the lookahead"""
        out = {}
        for s, i in ids: out.setdefault(s, []).append(i)
        return [(self.tr[s][0][ix], shift_labels(self.tr[s][1][ix], shift_samples(lookahead_ms, self.tr[s][2]))) for s, ix in out.items()]

    def groups_distill(self, train_ids, val_ids, soft, lookahead_ms=0.0):
        """ALL set-A recordings (minus validation trials): teacher soft labels, replaced by the true labels for the labeled trials"""
        out = []
        for s in self.sets:
            V, L, fs = self.tr[s]
            lab = soft[s].astype(np.float32).copy()
            ti = [i for t, i in train_ids if t == s]; lab[ti] = L[ti]
            keep = np.ones(V.shape[0], bool); keep[[i for t, i in val_ids if t == s]] = False
            out.append((V[keep], shift_labels(lab[keep], shift_samples(lookahead_ms, fs))))
        return out


class Bank:
    """All trials of a list of (V (n,2,T), L (n,T)) groups, kept ON THE COMPUTE DEVICE.
    A batch is drawn with a few gather operations: no Python loop over trials, no host->device copy per step."""

    def __init__(self, groups, device=None, rotate=True):
        device = device or DEVICE
        self.V = [torch.as_tensor(V, device=device) for V, _ in groups]
        self.L = [torch.as_tensor(L, device=device) for _, L in groups]
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
            ls.append(L[i[:, None], pos])
        v = torch.cat(vs)
        return (random_rotate(v) if self.rotate else v), torch.cat(ls)


def onehot(L):
    return torch.stack([1 - L, L], 1)


# ----------------------------------------------------------------------------- models
class UNetTeacher(nn.Module):
    """the original U'n'Eye (non-causal) with the (B,2,T)->(B,2,T) interface of SaccadeNet"""

    def __init__(self):
        super().__init__()
        from uneye.functions import UNet
        self.net = UNet(2, 5, 5)
        for m in self.net.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.ConvTranspose1d)): m.weight.data.normal_(0.0, 0.02)

    def forward(self, v):
        T = v.shape[2]
        x = F.pad(v, (0, (-T) % 25)).transpose(1, 2).unsqueeze(1)   # T must be a multiple of 25 (two max-poolings of 5)
        return self.net(x, ["out"])[0][:, :, :T]


def build_student(bb, channels=48):
    net = archs.build(bb, channels)
    if bb.startswith("tcn"): archs.init_like_uneye(net)
    return net


# ----------------------------------------------------------------------------- training
def train_supervised(model, train, val, pos_weight=3.0, lr=1e-3, l2=1e-4, max_epochs=60, steps=20, batch=32,
                     crop=700, patience=4, min_epochs=15, log=None):
    """original uneye logic: Adam, lr halved when validation gets worse (best weights restored), stop after > patience bad epochs.
    One 'epoch' = `steps` random minibatches (so that epochs are comparable for any number of labeled trials).
    No lr decay / early stopping before `min_epochs`: a from-scratch net first sits on the 'predict no saccade' plateau and
    would otherwise be stopped before it learns anything. Model selection uses the validation LOSS (with only 3 validation
    trials at N=10 a validation kappa is too noisy). Data lives on the device (Bank); no per-step host sync."""
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
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
        for _ in range(steps):
            v, l = bank.sample(batch, crop)
            loss = mc_loss(model(v), onehot(l), cw)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        vl = val_loss()
        if best is None or vl < best:
            best, bad, best_w = vl, 0, copy.deepcopy(model.state_dict())
        elif ep <= min_epochs:
            pass
        else:
            bad += 1; model.load_state_dict(best_w)
            for g in opt.param_groups: g["lr"] *= 0.5
        if log: log(f"  ep {ep:3d} val {vl:.4f} best {best:.4f} bad {bad}")
        if bad > patience: break
    model.load_state_dict(best_w)
    model.best_val = best
    return model


def pretrain_weak(model, data, **kw):
    """supervised training on Engbert-Kliegl pseudo-labels of ALL set-A recordings (no human labels)"""
    w = data.weak(); nv = 30
    return train_supervised(model, [(V[nv:], L[nv:]) for V, L in w.values()], [(V[:nv], L[:nv]) for V, L in w.values()], **kw)


# ----------------------------------------------------------------------------- prediction / evaluation
@torch.no_grad()
def predict(model, V, bs=64):
    model.eval(); dev = next(model.parameters()).device
    return np.concatenate([model(torch.from_numpy(V[i:i + bs]).to(dev))[:, 1].cpu().numpy() for i in range(0, V.shape[0], bs)])


@torch.no_grad()
def predict_lastbin(teacher, V, fs, bin_ms=200.0, chunk=1500):
    """for every t: the U-Net on the bin [t-W+1, t] (zeros before the start), output at the newest sample"""
    W = (int(round(bin_ms * fs / 1000)) + 24) // 25 * 25
    teacher.eval(); dev = next(teacher.parameters()).device
    n, _, T = V.shape
    P = np.zeros((n, T), np.float32)
    pad = np.concatenate([np.zeros((n, 2, W - 1), np.float32), V], 2)
    for i in range(n):
        x = torch.from_numpy(pad[i]).to(dev).unfold(1, W, 1).permute(1, 2, 0).unsqueeze(1).contiguous()   # (T,1,W,2)
        for j in range(0, T, chunk):
            P[i, j:j + chunk] = teacher.net(x[j:j + chunk], ["out"])[0][:, 1, -1].cpu().numpy()
    return P


def metrics_from_probs(Ps, data, thr=0.5, lookahead_ms=0.0, with_ap=True):
    """Ps[s] = probabilities (n,T) on the test trials of set s. The label of sample t-L is compared with the output at t."""
    res = []
    for s in data.sets:
        V, L, fs = data.te[s]
        P, k = Ps[s], shift_samples(lookahead_ms, fs)
        if k > 0: P, L = P[:, k:], L[:, :L.shape[1] - k]
        res.append(evaluate_probs(P, L, fs, thr, with_ap=with_ap, extra_delay_ms=k * 1000.0 / fs))
    return dict(per_set=dict(zip(data.sets, res)), pooled=pool(res))


def evaluate(model, data, val=None, lookahead_ms=0.0):
    """per-dataset + pooled metrics on set B at threshold 0.5 and at the kappa-tuned threshold (tuned on `val`, aligned labels)"""
    Ps = {s: predict(model, data.te[s][0]) for s in data.sets}
    out = {"0.5": dict(metrics_from_probs(Ps, data, 0.5, lookahead_ms), thr=0.5)}
    if val is not None:
        Pv = np.concatenate([predict(model, V).ravel() for V, _ in val]); Lv = np.concatenate([L.ravel() for _, L in val])
        t = tune_threshold(Pv, Lv, "kappa")
        out["tuned"] = dict(metrics_from_probs(Ps, data, t, lookahead_ms, with_ap=False), thr=t)
    return out


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def latency_ms(model, window=200, reps=50):
    """CPU time to label one 200 ms window (batch 1, one thread) in PyTorch eager mode"""
    import time as _t
    model = copy.deepcopy(model).cpu().eval(); torch.set_num_threads(1); x = torch.randn(1, 2, window) * 0.05
    with torch.no_grad():
        for _ in range(5): model(x)
        t = _t.perf_counter()
        for _ in range(reps): model(x)
    return (_t.perf_counter() - t) / reps * 1000


# ----------------------------------------------------------------------------- jobs (module level: they run in worker processes)
FULL = dict(ft_epochs=60, ft_steps=20)
QUICK = dict(ft_epochs=3, ft_steps=5)
STUDENTS = ("scratch", "weak_ft", "distill")
_DATA = None


def _seed(s):
    np.random.seed(s); torch.manual_seed(s)


def default_devices(workers_per_gpu=2):
    """one worker slot per GPU x workers_per_gpu (tiny models cannot fill a GPU alone); on CPU one slot per 2 cores"""
    cores = os.cpu_count() or 2
    if torch.cuda.is_available():
        devs = [f"cuda:{i}" for i in range(torch.cuda.device_count())] * workers_per_gpu
        return devs[:max(cores, torch.cuda.device_count())]       # never more worker processes than CPU cores
    return ["cpu"] * max(1, cores // 2)


def _init_worker(dev_queue, data_kwargs):
    global DEVICE, _DATA
    DEVICE = dev_queue.get()
    torch.set_num_threads(1)                      # several processes: avoid CPU oversubscription
    if DEVICE.startswith("cuda"): torch.cuda.set_device(DEVICE)
    _DATA = Data(**data_kwargs)


def _row(base, res):
    row = dict(base)
    for tag in res:
        for k, v in res[tag]["pooled"].items(): row[f"{k}@{tag}"] = v
        row[f"thr@{tag}"] = res[tag]["thr"]
        for s_, m in res[tag]["per_set"].items(): row[f"kappa_set{s_}@{tag}"] = m["kappa"]
    return row


def _job_weak(bb, channels, cfg):
    _seed(123); t = time.time()
    net = build_student(bb, channels)
    pretrain_weak(net, _DATA, max_epochs=cfg["ft_epochs"], steps=cfg["ft_steps"])
    return bb, {k: v.cpu() for k, v in net.state_dict().items()}, time.time() - t


def _job_teacher(n, sd, cfg):
    """original U'n'Eye trained on the same N labeled trials: reference rows (offline / last-bin) + soft labels for distillation"""
    _seed(2000 + sd); t = time.time()
    train_ids, val_ids = _DATA.split(n, sd)
    teacher = UNetTeacher()
    train_supervised(teacher, _DATA.groups(train_ids), _DATA.groups(val_ids), pos_weight=1.0, max_epochs=cfg["ft_epochs"], steps=cfg["ft_steps"])
    base = dict(backbone="unet", n_labels=n, seed=sd, lookahead_ms=0.0, params=count_params(teacher))
    off = metrics_from_probs({s: predict(teacher, _DATA.te[s][0]) for s in _DATA.sets}, _DATA)
    last = metrics_from_probs({s: predict_lastbin(teacher, _DATA.te[s][0], _DATA.te[s][2]) for s in _DATA.sets}, _DATA)
    rows = [_row(dict(base, strategy="offline"), {"0.5": dict(off, thr=0.5)}), _row(dict(base, strategy="lastbin"), {"0.5": dict(last, thr=0.5)})]
    soft = {s: predict(teacher, _DATA.tr[s][0]).astype(np.float16) for s in _DATA.sets}
    return (n, sd), rows, soft, time.time() - t


def _job_student(bb, st, n, sd, lookahead_ms, channels, cfg, aux):
    _seed(1000 + sd); t = time.time()
    train_ids, val_ids = _DATA.split(n, sd)
    val = _DATA.groups(val_ids, lookahead_ms)
    net = build_student(bb, channels)
    if st == "distill": train = _DATA.groups_distill(train_ids, val_ids, aux, lookahead_ms)
    else: train = _DATA.groups(train_ids, lookahead_ms)
    lr = 1e-3
    if st == "weak_ft": net.load_state_dict(aux); lr = 5e-4
    train_supervised(net, train, val, max_epochs=cfg["ft_epochs"], steps=cfg["ft_steps"], lr=lr)
    r = evaluate(net, _DATA, val, lookahead_ms)
    return _row(dict(backbone=bb, strategy=st, n_labels=n, seed=sd, lookahead_ms=float(lookahead_ms), params=count_params(net), seconds=time.time() - t), r)


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


def make_plan(backbones, strategies, n_labels, seeds, lookaheads=(0.0,)):
    return [dict(backbone=bb, strategy=st, n_labels=n, seed=sd, lookahead_ms=float(la))
            for bb in backbones for st in strategies for n in n_labels for sd in seeds for la in lookaheads]


def _key(r):
    return (r["backbone"], r["strategy"], int(r["n_labels"]), int(r["seed"]), float(r["lookahead_ms"]))


def run_plan(data, plan, cfg=None, channels=48, log=print, save=None, devices=None, resume=True, unet_baselines=True):
    """plan: list of dicts (backbone, strategy, n_labels, seed, lookahead_ms) from make_plan (concatenate several plans).
    Returns a list of result rows. The original U'n'Eye trained on the same N labels (backbone='unet', strategy='offline'|'lastbin')
    is added for every (N, seed) of the plan. devices=None -> this process only; devices=default_devices() -> parallel.
    resume=True: runs already stored in `save` are kept and skipped (json written after every finished run)."""
    cfg = dict(FULL, **(cfg or {}))
    devices = devices or [DEVICE]
    rows = json.load(open(save)) if (resume and save and os.path.exists(save)) else []
    done = {_key(r) for r in rows}
    todo = [p for p in plan if _key(p) not in done]
    ns = sorted({(p["n_labels"], p["seed"]) for p in plan})
    need_teacher = [(n, sd) for n, sd in ns if (("unet", "lastbin", n, sd, 0.0) not in done and unet_baselines)
                    or any(p["strategy"] == "distill" and p["n_labels"] == n and p["seed"] == sd for p in todo)]
    need_weak = sorted({p["backbone"] for p in todo if p["strategy"] == "weak_ft"})
    if done: log(f"[resume] {len(done)} runs already finished, {len(todo)} student runs left")
    t0 = time.time()
    runner = JobRunner(data, devices, log)
    pre, soft = {}, {}
    pre_jobs = [(_job_weak, (bb, channels, cfg)) for bb in need_weak] + [(_job_teacher, (n, sd, cfg)) for n, sd in need_teacher]
    for res in runner.run(pre_jobs):
        if isinstance(res[0], str):
            pre[res[0]] = res[1]; log(f"[weak-label pretraining] {res[0]}: {res[2]:.0f}s")
        else:
            (n, sd), trows, soft[(n, sd)], secs = res
            if unet_baselines:
                rows += [r for r in trows if _key(r) not in done]
                if save: json.dump(rows, open(save, "w"))
            log(f"[U'n'Eye teacher/baseline] N={n} seed {sd}: offline kappa {trows[0]['kappa@0.5']:.3f}, last-bin kappa {trows[1]['kappa@0.5']:.3f} [{secs:.0f}s]")
    cost = lambda p: (p["n_labels"] + 20) * (3 if p["backbone"] == "gru" else 1) * (4 if p["strategy"] == "distill" else 1)
    todo.sort(key=lambda p: -cost(p))
    jobs = [(_job_student, (p["backbone"], p["strategy"], p["n_labels"], p["seed"], p["lookahead_ms"], channels, cfg,
                            pre.get(p["backbone"]) if p["strategy"] == "weak_ft" else soft.get((p["n_labels"], p["seed"]))
                            if p["strategy"] == "distill" else None)) for p in todo]
    n_before = len(rows)
    for row in runner.run(jobs):
        rows.append(row)
        log(f"{row['backbone']:9s} {row['strategy']:8s} N={row['n_labels']:4d} L={row['lookahead_ms']:4.0f}ms seed {row['seed']}: kappa {row['kappa@0.5']:.3f} "
            f"(tuned {row['kappa@tuned']:.3f}) mcc {row['mcc@0.5']:.3f} ev_f1 {row['ev_f1@0.5']:.3f}  "
            f"[{row['seconds']:.0f}s, {len(rows) - n_before}/{len(jobs)}, elapsed {(time.time() - t0) / 60:.1f} min]")
        if save: json.dump(rows, open(save, "w"))
    runner.close()
    return rows


# ----------------------------------------------------------------------------- final model
def train_final(data, backbone="tcn", strategy="scratch", lookahead_ms=0.0, n="all", seeds=(0, 1, 2, 3, 4), channels=48, cfg=None, log=print):
    """Train the chosen design on ALL human labels of set A (20 % of them for early stopping, the same split for every seed) with several
    training seeds and keep the model with the lowest VALIDATION loss. A single training run is a gamble: on the small dataset 3 the
    same design gave kappa 0.61 in one run and 0.78 in another. Returns (model, evaluation of the chosen model, table with one row per seed;
    the test kappa in the table is for information only and is never used for the choice)."""
    cfg = dict(FULL, **(cfg or {}))
    n = sum(data.tr[s][0].shape[0] for s in data.sets) if n == "all" else n
    train_ids, val_ids = data.split(n, 0)
    val = data.groups(val_ids, lookahead_ms)
    best, table = None, []
    for sd in seeds:
        _seed(sd)
        net = build_student(backbone, channels)
        lr = 1e-3
        if strategy == "weak_ft":
            pretrain_weak(net, data, max_epochs=cfg["ft_epochs"], steps=cfg["ft_steps"]); lr = 5e-4
        train_supervised(net, data.groups(train_ids, lookahead_ms), val, max_epochs=cfg["ft_epochs"], steps=cfg["ft_steps"], lr=lr)
        res = evaluate(net, data, val, lookahead_ms)
        row = dict(seed=sd, val_loss=net.best_val, kappa_test=res["0.5"]["pooled"]["kappa"], ev_f1_test=res["0.5"]["pooled"]["ev_f1"],
                   **{f"kappa_set{s}": res["0.5"]["per_set"][s]["kappa"] for s in data.sets})
        table.append(row)
        if best is None or net.best_val < best[0]: best = (net.best_val, net, res, sd)
        if log: log(f"  seed {sd}: validation loss {net.best_val:.4f}")
    for r in table: r["chosen"] = (r["seed"] == best[3])
    return best[1], best[2], table
