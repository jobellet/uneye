"""Data access for label-free experiments.

Two sources, same output (`Windows`):
  * H5Source    your Kaggle layout: files `x` of shape (n_sequences, L, 3) = (x, y, dt) per sequence, listed in a metadata csv
                with at least `filename` and `num_sequences` (all further columns are optional).
  * labeled repo recordings (datasets 1-4 of U'n'Eye, with human labels): used ONLY to check that the label-free scores predict real accuracy.
Everything is resampled to one common rate (FS) so that "20 ms" means the same number of samples everywhere.
"""
import os
from dataclasses import dataclass, field
import numpy as np

FS = 500.0                                     # common sampling rate (Hz) of all windows


@dataclass
class Windows:
    pos: np.ndarray                            # (n, T, 2) float32, degrees (x unit_scale), NaN = invalid sample (blink, dropout)
    fs: float                                  # Hz
    group: np.ndarray                          # (n,) dataset / source name of every window
    labels: np.ndarray = None                  # (n, T) bool human labels (labeled repo data only)
    info: dict = field(default_factory=dict)

    def __len__(self): return self.pos.shape[0]

    def subset(self, idx):
        return Windows(self.pos[idx], self.fs, self.group[idx], None if self.labels is None else self.labels[idx], self.info)


# ----------------------------------------------------------------------------- cleaning and resampling
def clean_positions(xy, limit=200.0):
    """inf / absurd values -> NaN (they are blinks or dropouts; they are never filled with fake data)"""
    xy = np.asarray(xy, np.float64).copy()
    xy[~np.isfinite(xy) | (np.abs(np.nan_to_num(xy, nan=0.0, posinf=1e30, neginf=-1e30)) > limit)] = np.nan
    return xy


def resample(xy, dt, fs_out=FS, unit_scale=1.0):
    """xy (L,2) with NaN, dt (L,) seconds (or None). Returns (L',2) at fs_out (linear interpolation over valid samples; a sample of the new
    grid is invalid when its nearest original sample is invalid)."""
    xy = clean_positions(xy) * unit_scale
    L = xy.shape[0]
    if dt is None or not np.isfinite(dt).any():
        t = np.arange(L, dtype=np.float64) / fs_out; fs_in = fs_out
    else:
        dt = np.asarray(dt, np.float64)
        med = np.nanmedian(dt[dt > 0]) if (dt > 0).any() else 1.0 / fs_out
        if med > 0.5: dt = dt / 1000.0; med /= 1000.0           # dt given in ms
        dt = np.where(np.isfinite(dt) & (dt > 0) & (dt < 10 * med), dt, med)
        t = np.cumsum(dt) - dt[0]; fs_in = 1.0 / med
    if abs(fs_in - fs_out) / fs_out < 0.02 and np.allclose(np.diff(t), 1.0 / fs_in, rtol=0.2):
        return xy.astype(np.float32), fs_in
    ok = np.isfinite(xy).all(1)
    if ok.sum() < 4: return np.full((max(int(t[-1] * fs_out), 1), 2), np.nan, np.float32), fs_in
    tg = np.arange(0.0, t[-1], 1.0 / fs_out)
    out = np.stack([np.interp(tg, t[ok], xy[ok, a]) for a in range(2)], 1)
    nearest = np.clip(np.searchsorted(t, tg), 0, L - 1)
    out[~np.isfinite(xy[nearest]).all(1)] = np.nan
    return out.astype(np.float32), fs_in


def resample_labels(lab, fs_in, fs_out=FS):
    if abs(fs_in - fs_out) / fs_out < 0.02: return lab.astype(bool)
    t_in = np.arange(lab.shape[-1]) / fs_in; tg = np.arange(0.0, t_in[-1] + 1e-9, 1.0 / fs_out)
    return lab[..., np.clip(np.searchsorted(t_in, tg), 0, lab.shape[-1] - 1)].astype(bool)


# ----------------------------------------------------------------------------- your Kaggle layout
def infer_groups(meta, hint=None):
    """one source name per file: a column called dataset / source / study ..., else the file name before the first digit or '_'"""
    import re
    cols = [c for c in meta.columns]
    for c in ([hint] if hint else []) + [c for c in cols if any(k in c.lower() for k in ("dataset", "source", "study", "experiment"))]:
        if c in meta.columns: return meta[c].astype(str).values, c
    names = meta["filename"].astype(str).values
    return np.array([re.split(r"[_\d]", n, maxsplit=1)[0] or "all" for n in names]), "filename prefix"


class H5Source:
    def __init__(self, src_dir, meta_path, group_col=None, unit_scale=1.0):
        import pandas as pd
        self.dir, self.unit_scale = src_dir, unit_scale
        self.meta = pd.read_csv(meta_path)
        self.meta["group"], self.group_from = infer_groups(self.meta, group_col)
        self.meta = self.meta.reset_index(drop=True)

    def audit(self, per_group_files=3, seqs=20, rng=None):
        """what is really in the data: sampling rate, position range, noise, missing fraction, speed percentiles, per source"""
        import h5py, pandas as pd
        rng = rng or np.random.RandomState(0); rows = []
        for g, d in self.meta.groupby("group"):
            files = d.sample(min(per_group_files, len(d)), random_state=0)["filename"].tolist()
            fs, rng_pos, miss, noise, p99, L = [], [], [], [], [], []
            for f in files:
                try:
                    with h5py.File(os.path.join(self.dir, f), "r") as h:
                        x = h["x"]; n = x.shape[0]
                        for i in rng.choice(n, min(seqs, n), replace=False):
                            raw = np.asarray(x[int(i)]); L.append(raw.shape[0])
                            dt = raw[:, 2] if raw.shape[1] > 2 else None
                            xy = clean_positions(raw[:, :2]) * self.unit_scale
                            miss.append(float(np.isnan(xy).any(1).mean()))
                            if dt is not None and np.isfinite(dt).any():
                                m = np.nanmedian(dt[dt > 0]) if (dt > 0).any() else np.nan
                                fs.append(1000.0 / m if m > 0.5 else 1.0 / m)
                            ok = np.isfinite(xy).all(1)
                            if ok.sum() > 10:
                                rng_pos.append(np.ptp(xy[ok, 0])); v = np.hypot(*np.diff(xy[ok], axis=0).T) * (fs[-1] if fs else 1000.0)
                                noise.append(1.4826 * np.median(np.abs(v - np.median(v)))); p99.append(np.percentile(v, 99))
                except Exception as e:
                    pass
            rows.append(dict(source=g, files=len(d), sequences=int(d["num_sequences"].sum()) if "num_sequences" in d else None,
                             fs_hz=np.nanmedian(fs) if fs else np.nan, seq_len=np.median(L) if L else np.nan, missing=np.mean(miss) if miss else np.nan,
                             x_range=np.median(rng_pos) if rng_pos else np.nan, speed_noise=np.median(noise) if noise else np.nan, speed_p99=np.median(p99) if p99 else np.nan))
        return pd.DataFrame(rows)

    def sample(self, n_per_group, T, rng, fs=FS, groups=None, max_missing=0.2):
        """n_per_group random windows of T samples (at fs) from every source; windows with too many invalid samples are skipped"""
        import h5py
        pos, grp = [], []
        for g, d in self.meta.groupby("group"):
            if groups and g not in groups: continue
            d = d.reset_index(drop=True); got = 0; tries = 0
            while got < n_per_group and tries < n_per_group * 20:
                tries += 1
                r = d.iloc[rng.randint(len(d))]
                try:
                    with h5py.File(os.path.join(self.dir, r["filename"]), "r") as h:
                        x = h["x"]; raw = np.asarray(x[rng.randint(x.shape[0])])
                except Exception:
                    continue
                xy, fs_in = resample(raw[:, :2], raw[:, 2] if raw.shape[1] > 2 else None, fs, self.unit_scale)
                if xy.shape[0] < T: continue
                s = rng.randint(0, xy.shape[0] - T + 1); w = xy[s:s + T]
                if np.isnan(w).any(1).mean() > max_missing: continue
                pos.append(w); grp.append(g); got += 1
            if got < n_per_group:
                print(f"WARNING source '{g}': only {got}/{n_per_group} windows of {T} samples at {fs:.0f} Hz found (sequences too short or too many invalid samples)")
        return Windows(np.stack(pos).astype(np.float32), fs, np.array(grp))


# ----------------------------------------------------------------------------- labeled recordings of the repository (validation only)
REPO_SETS = {"u1": ("dataset1", "dataset1_1000hz", 1000.0), "u2": ("dataset2", "dataset2_1000hz", 1000.0),
             "u3": ("dataset3", "dataset3_500hz", 500.0), "u4": ("dataset4", "dataset4_1000hz", 1000.0)}


def load_labeled(repo_data_dir, names=("u1", "u2", "u3", "u4"), which="B", n=300, seed=0, fs=FS):
    """{name: Windows with human labels} (whole trials, resampled to fs). Used only to validate label-free scores against real accuracy."""
    out = {}; rng = np.random.RandomState(seed)
    for nm in names:
        d, f, fs_in = REPO_SETS[nm]
        ld = lambda k: np.loadtxt(os.path.join(repo_data_dir, d, f"{f}_{k}_set{which}.csv"), delimiter=",")
        X, Y, L = ld("X"), ld("Y"), ld("Labels")
        ind = rng.permutation(X.shape[0])[:n]
        pos, lab = [], []
        for i in ind:
            xy, _ = resample(np.stack([X[i], Y[i]], 1), np.full(X.shape[1], 1.0 / fs_in), fs)
            pos.append(xy); lab.append(resample_labels(L[i] > 0, fs_in, fs))
        T = min(p.shape[0] for p in pos)
        out[nm] = Windows(np.stack([p[:T] for p in pos]), fs, np.array([nm] * len(pos)), np.stack([l[:T] for l in lab]), {"native_fs": fs_in})
    return out


# ----------------------------------------------------------------------------- a fake dataset in YOUR layout (for tests and demos)
def write_demo_h5(repo_data_dir, dst_dir, per_file=40, files_per_set=4, seed=0):
    """Writes h5 files with x of shape (n_sequences, L, 3) = (x, y, dt) and subject_h5_metadata.csv from the repository recordings (set A, labels NOT
    included) so that the unlabeled pipeline can be tried anywhere. Blinks stay as NaN/inf, like in real recordings."""
    import h5py, pandas as pd
    os.makedirs(dst_dir, exist_ok=True); rng = np.random.RandomState(seed); rows = []
    for nm, (d, f, fs_in) in REPO_SETS.items():
        X = np.loadtxt(os.path.join(repo_data_dir, d, f"{f}_X_setA.csv"), delimiter=","); Y = np.loadtxt(os.path.join(repo_data_dir, d, f"{f}_Y_setA.csv"), delimiter=",")
        for k in range(files_per_set):
            idx = rng.choice(X.shape[0], per_file, replace=False)
            arr = np.stack([X[idx], Y[idx], np.full_like(X[idx], 1.0 / fs_in)], axis=2).astype(np.float32)
            fn = f"{nm}_subject{k:02d}.h5"
            with h5py.File(os.path.join(dst_dir, fn), "w") as h: h.create_dataset("x", data=arr)
            rows.append(dict(filename=fn, num_sequences=per_file, dataset=nm))
    pd.DataFrame(rows).to_csv(os.path.join(dst_dir, "subject_h5_metadata.csv"), index=False)
    return os.path.join(dst_dir, "subject_h5_metadata.csv")
