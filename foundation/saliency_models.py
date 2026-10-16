#!/usr/bin/env python3
"""Low-cost saliency models for frames (numpy / scipy), all returning a map normalised to max 1 on a coarse grid (row 0 = top):
  pqft      phase spectrum of the quaternion Fourier transform (Guo et al. 2008), motion = frame difference      (same as cpp/app/engine/include/eyelive/saliency.hpp)
  sr        spectral residual (Hou & Zhang 2007), static
  itti      Itti, Koch & Niebur 1998: intensity, colour opponency (R-G, B-Y), 4 Gabor orientations, centre-surround c = 2,3,4, s = c + 3, c + 4, normalisation N(.)
  itti_dyn  the same + flicker (|I_t - I_t-1|) and motion (Reichardt detectors in 4 directions), as in Itti, Dhavale & Pighin 2003
  motion    smoothed |I_t - I_t-1| only
  center    Gaussian centre bias (sigma = 0.25 of the width / height)
"""
import numpy as np
from scipy.ndimage import gaussian_filter, zoom, maximum_filter, uniform_filter

def _pyr(img, levels=9):
    p = [img]
    for _ in range(levels - 1): p.append(gaussian_filter(p[-1], 1.0)[::2, ::2])
    return p
def _up(a, shape): return zoom(a, (shape[0] / a.shape[0], shape[1] / a.shape[1]), order=1)
def _N(m):
    m = m - m.min(); mx = m.max()
    if mx <= 0: return m
    m = m / mx; lm = maximum_filter(m, size=3); peaks = m[(m == lm) & (m > 0.05)]
    mbar = (peaks.sum() - 1.0) / max(len(peaks) - 1, 1) if len(peaks) > 1 else 0.0
    return m * (1 - mbar) ** 2
def _cs(pyr, out_level=4):
    shape = pyr[out_level].shape; acc = np.zeros(shape)
    for c in (2, 3, 4):
        for d in (3, 4):
            s = c + d
            if s >= len(pyr): continue
            acc += _up(_N(np.abs(pyr[c] - _up(pyr[s], pyr[c].shape))), shape)
    return acc
def _gabor(img, theta, lam=7.0, sigma=2.5):
    r = 7; y, x = np.mgrid[-r:r + 1, -r:r + 1]; xr = x * np.cos(theta) + y * np.sin(theta)
    k = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2)) * np.cos(2 * np.pi * xr / lam); k -= k.mean()
    from scipy.signal import fftconvolve
    return np.abs(fftconvolve(img, k, mode="same"))

class Model:
    def __init__(self, name): self.name = name; self.prev = None; self.prev_pyr = None
    def __call__(self, rgb):                                        # rgb: uint8 H x W x 3 (any size); returns (map, ...)
        f = rgb.astype(np.float64) / 255.0; r, g, b = f[..., 0], f[..., 1], f[..., 2]; I = (r + g + b) / 3; n = self.name
        if n == "center":
            h, w = 36, 64; y, x = np.mgrid[0:h, 0:w]; m = np.exp(-(((x + 0.5) / w - 0.5) ** 2 / (2 * 0.25 ** 2) + ((y + 0.5) / h - 0.5) ** 2 / (2 * 0.25 ** 2))); return m / m.max()
        if n in ("pqft", "sr", "motion"):
            small = zoom(f, (64 / f.shape[0], 64 / f.shape[1], 1), order=1); r, g, b = small[..., 0], small[..., 1], small[..., 2]; Is = (r + g + b) / 3
            M = np.zeros_like(Is) if self.prev is None else np.abs(Is - self.prev); self.prev = Is
            if n == "motion": m = gaussian_filter(M, 3.0); return m / m.max() if m.max() > 0 else np.ones_like(m)
            if n == "sr":
                F = np.fft.fft2(Is); L = np.log(np.abs(F) + 1e-9); R = L - uniform_filter(L, 3); m = gaussian_filter(np.abs(np.fft.ifft2(np.exp(R + 1j * np.angle(F)))) ** 2, 3.0); return m / m.max()
            F1 = np.fft.fft2(M + 1j * (r - g)); F2 = np.fft.fft2((b - 0.5 * (r + g)) + 1j * Is); mod = np.sqrt(np.abs(F1) ** 2 + np.abs(F2) ** 2) + 1e-9
            m = gaussian_filter(np.abs(np.fft.ifft2(F1 / mod)) ** 2 + np.abs(np.fft.ifft2(F2 / mod)) ** 2, 3.0); return m / m.max()
        # Itti-Koch on a 640 x 360 base
        base = zoom(f, (360 / f.shape[0], 640 / f.shape[1], 1), order=1); r, g, b = base[..., 0], base[..., 1], base[..., 2]; I = (r + g + b) / 3
        norm = np.where(I > 0.1 * I.max(), I, 1.0); rn, gn, bn = r / norm, g / norm, b / norm; rn[I <= 0.1 * I.max()] = gn[I <= 0.1 * I.max()] = bn[I <= 0.1 * I.max()] = 0
        R = rn - (gn + bn) / 2; G = gn - (rn + bn) / 2; B = bn - (rn + gn) / 2; Y = np.clip((rn + gn) / 2 - np.abs(rn - gn) / 2 - bn, 0, None)
        PI = _pyr(I); PRG = _pyr(R - G); PBY = _pyr(B - Y)
        Ib = _N(_cs(PI)); Cb = _N(_cs(PRG) + _cs(PBY)); Ob = np.zeros_like(Ib)
        for th in (0, np.pi / 4, np.pi / 2, 3 * np.pi / 4): Ob += _N(_cs([_gabor(l, th) if k >= 2 else l for k, l in enumerate(PI)]))
        Ob = _N(Ob); chans = [Ib, Cb, Ob]
        if n == "itti_dyn":
            if self.prev_pyr is not None:
                Fl = _N(_cs([np.abs(a - c) for a, c in zip(PI, self.prev_pyr)])); Mo = np.zeros_like(Ib)
                for dy, dx in ((0, 1), (1, 0), (0, -1), (-1, 0)):            # Reichardt: I_t * shift(I_t-1) - I_t-1 * shift(I_t)
                    Mo += _N(_cs([np.abs(a * np.roll(c, (dy, dx), (0, 1)) - c * np.roll(a, (dy, dx), (0, 1))) for a, c in zip(PI, self.prev_pyr)]))
                chans += [Fl, _N(Mo)]
            self.prev_pyr = PI
        m = sum(chans) / len(chans); m = gaussian_filter(m, 1.0); return m / m.max() if m.max() > 0 else np.ones_like(m)
