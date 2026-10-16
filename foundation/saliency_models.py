#!/usr/bin/env python3
"""Low-cost saliency models for frames (numpy / scipy), all returning a map normalised to max 1 on a coarse grid (row 0 = top):
  pqft      phase spectrum of the quaternion Fourier transform (Guo et al. 2008), motion = frame difference      (same as cpp/app/engine/include/eyelive/saliency.hpp)
  sr        spectral residual (Hou & Zhang 2007), static
  itti      Itti, Koch & Niebur 1998: intensity, colour opponency (R-G, B-Y), 4 Gabor orientations, centre-surround c = 2,3,4, s = c + 3, c + 4, normalisation N(.)
  itti_dyn  the same + flicker (|I_t - I_t-1|) and motion (Reichardt detectors in 4 directions), as in Itti, Dhavale & Pighin 2003
  motion    smoothed |I_t - I_t-1| only
  center    Gaussian centre bias (sigma = 0.25 of the width / height)
  *_c       camera-motion compensated versions (motion_c, pqft_c, itti_dyn_c): the global affine motion between two frames (drone ego-motion) is estimated (direct method, 2-level
            pyramid, Huber re-weighting so that independently moving objects do not drag the estimate) and the previous frame is warped before every temporal difference, so that
            only what moves relative to the background (not the background flow caused by the camera) feeds the motion / flicker channels.
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

def _affine_warp(img, A):                                       # img(x) <- img(A x), A = 2 x 3 (pixel units, x = column, y = row)
    from scipy.ndimage import map_coordinates
    h, w = img.shape; yy, xx = np.mgrid[0:h, 0:w].astype(np.float64); X = A[0, 0] * xx + A[0, 1] * yy + A[0, 2]; Y = A[1, 0] * xx + A[1, 1] * yy + A[1, 2]
    return map_coordinates(img, [Y, X], order=1, mode="nearest")
def global_motion(prev, cur, levels=3, iters=8):
    """affine A (2 x 3) such that cur(x) ~ prev(A x), by Gauss-Newton on the brightness constancy with Huber weights; prev, cur: gray float images of the same size (>= 90 x 160)"""
    A = np.array([[1.0, 0, 0], [0, 1.0, 0]])
    for lv in range(levels - 1, -1, -1):
        f = 2 ** lv; c = cur[::f, ::f] if f > 1 else cur; p = prev[::f, ::f] if f > 1 else prev; c = gaussian_filter(c, 0.8 * f / 2 + 0.01); p = gaussian_filter(p, 0.8 * f / 2 + 0.01)
        A[:, 2] /= f if lv == levels - 1 else 1; h, w = c.shape; yy, xx = np.mgrid[0:h, 0:w].astype(np.float64); gy, gx = np.gradient(c); xc, yc = xx - w / 2, yy - h / 2
        if lv < levels - 1: A[:, 2] = A[:, 2] / 2.0
        for _ in range(iters):
            Aw = A.copy(); Aw[:, 2] = A[:, 2] + (A[:, :2] - np.eye(2)) @ np.array([w / 2, h / 2]) * 0                                                   # (the centred parametrisation below keeps it simple)
            wp = _affine_warp(p, A); r = c - wp; sig = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-6; wt = np.where(np.abs(r) < 1.5 * sig, 1.0, 1.5 * sig / np.maximum(np.abs(r), 1e-12))
            J = np.stack([gx * xc, gx * yc, gx, gy * xc, gy * yc, gy], -1).reshape(-1, 6); sw = np.sqrt(wt.ravel())[:, None]
            dp = np.linalg.lstsq(J * sw, (r.ravel()[:, None] * sw)[:, 0], rcond=None)[0]
            # update in the centred frame: x' = A x -> compose with the increment (small-motion approximation)
            d = np.array([[dp[0], dp[1], dp[2] - dp[0] * w / 2 - dp[1] * h / 2], [dp[3], dp[4], dp[5] - dp[3] * w / 2 - dp[4] * h / 2]]); A = A + d
        A[:, 2] = A[:, 2] * (2 if lv > 0 else 1)
    return A
def compensated_diff(prev, cur):
    """|cur - warp(prev)| after removing the global affine motion; also returns the estimated image translation (pixels) for diagnostics"""
    A = global_motion(prev, cur); return np.abs(cur - _affine_warp(prev, A)), A

class Model:
    def __init__(self, name): self.name = name; self.prev = None; self.prev_pyr = None
    def __call__(self, rgb):                                        # rgb: uint8 H x W x 3 (any size); returns (map, ...)
        f = rgb.astype(np.float64) / 255.0; r, g, b = f[..., 0], f[..., 1], f[..., 2]; I = (r + g + b) / 3; n = self.name
        if n == "center":
            h, w = 36, 64; y, x = np.mgrid[0:h, 0:w]; m = np.exp(-(((x + 0.5) / w - 0.5) ** 2 / (2 * 0.25 ** 2) + ((y + 0.5) / h - 0.5) ** 2 / (2 * 0.25 ** 2))); return m / m.max()
        if n in ("pqft_c", "motion_c"):
            gray = zoom(I, (180 / I.shape[0], 320 / I.shape[1]), order=1); small = zoom(f, (64 / f.shape[0], 64 / f.shape[1], 1), order=1); r, g, b = small[..., 0], small[..., 1], small[..., 2]; Is = (r + g + b) / 3
            if self.prev is None: M = np.zeros_like(Is)
            else: Mb, self.last_A = compensated_diff(self.prev, gray); M = zoom(Mb, (64 / 180, 64 / 320), order=1)
            self.prev = gray
            if n == "motion_c": m = gaussian_filter(M, 3.0); return m / m.max() if m.max() > 0 else np.ones_like(m)
            F1 = np.fft.fft2(M + 1j * (r - g)); F2 = np.fft.fft2((b - 0.5 * (r + g)) + 1j * Is); mod = np.sqrt(np.abs(F1) ** 2 + np.abs(F2) ** 2) + 1e-9
            m = gaussian_filter(np.abs(np.fft.ifft2(F1 / mod)) ** 2 + np.abs(np.fft.ifft2(F2 / mod)) ** 2, 3.0); return m / m.max()
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
        PI = _pyr(I); PRG = _pyr(R - G); PBY = _pyr(B - Y); prev_I = self.prev_pyr                  # (the previous pyramid, warped onto this frame for itti_dyn_c)
        if n == "itti_dyn_c" and prev_I is not None:
            A = global_motion(zoom(prev_I[0], 0.5, order=1), zoom(I, 0.5, order=1)); A[:, 2] *= 2; self.last_A = A; prev_I = _pyr(_affine_warp(prev_I[0], A))
        Ib = _N(_cs(PI)); Cb = _N(_cs(PRG) + _cs(PBY)); Ob = np.zeros_like(Ib)
        for th in (0, np.pi / 4, np.pi / 2, 3 * np.pi / 4): Ob += _N(_cs([_gabor(l, th) if k >= 2 else l for k, l in enumerate(PI)]))
        Ob = _N(Ob); chans = [Ib, Cb, Ob]
        if n in ("itti_dyn", "itti_dyn_c"):
            if prev_I is not None:
                Fl = _N(_cs([np.abs(a - c) for a, c in zip(PI, prev_I)])); Mo = np.zeros_like(Ib)
                for dy, dx in ((0, 1), (1, 0), (0, -1), (-1, 0)):            # Reichardt: I_t * shift(I_t-1) - I_t-1 * shift(I_t)
                    Mo += _N(_cs([np.abs(a * np.roll(c, (dy, dx), (0, 1)) - c * np.roll(a, (dy, dx), (0, 1))) for a, c in zip(PI, prev_I)]))
                chans += [Fl, _N(Mo)]
            self.prev_pyr = PI
        m = sum(chans) / len(chans); m = gaussian_filter(m, 1.0); return m / m.max() if m.max() > 0 else np.ones_like(m)
