// Label-free saccade proposals for the labelers: the universal minimum-duration HMM + acceleration-zero boundaries ("hmm_accel"), a line-by-line
// JavaScript port of the batch detector of the C++ engine (cpp/app/include/eyelive/dsp.hpp: score_trace, viterbi, drop_invalid_zone_and_short,
// refine_events; cpp/app/detect_hmm/src/hmm_detector.cpp: detect_all with its default options). Parameters: cpp/models/universal_hmm.txt.
// Parity with the C++ detector: test_hmm_prelabel.js. Positions in any unit (the score and the boundaries are unit-free); invalid samples = NaN or valid[i] == 0.
(function (root) {
    'use strict';

    const PARAMS = {
        1000: { d: 6, mu0: 0.416573763, mu1: 1.32039225, sd0: 0.201105669, sd1: 0.766239583, p_on: 0.00554218027, p_stay: 0.949469511 },
        500: { d: 3, mu0: 0.406975567, mu1: 1.3010565, sd0: 0.20460619, sd1: 0.573560297, p_on: 0.00969478964, p_stay: 0.928820044 },
    };

    // the parameters exist for 1000 and 500 Hz (as in the C++ detector); other rates use the closest set, with the minimum duration rescaled
    function paramsFor(fs) {
        if (Math.abs(fs - 1000) < 50) return Object.assign({ fs, exact: true }, PARAMS[1000]);
        if (Math.abs(fs - 500) < 25) return Object.assign({ fs, exact: true }, PARAMS[500]);
        const base = Math.abs(fs - 1000) < Math.abs(fs - 500) ? PARAMS[1000] : PARAMS[500];
        return Object.assign({}, base, { fs, exact: false, d: Math.max(2, Math.round(6 * fs / 1000)) });
    }

    function nearbyint(v) {                                         // C++ std::nearbyint (round half to even), so that boundaries match at every rate
        const r = Math.round(v); return (Math.abs(v % 1) === 0.5 && r % 2 !== 0) ? r - 1 : r;
    }

    function median(arr) {                                          // numpy.median (copies)
        const n = arr.length; if (!n) return 0;
        const v = Float64Array.from(arr).sort(); const h = n >> 1;
        return n % 2 ? v[h] : 0.5 * (v[h - 1] + v[h]);
    }

    function fillPositions(p, valid) {                              // linear interpolation over invalid samples, constant extrapolation, zeros if none valid
        const T = p.length, f = new Float64Array(T);
        let first = -1, last = -1;
        for (let t = 0; t < T; t++) if (valid[t]) { if (first < 0) first = t; last = t; }
        if (first < 0) return f;
        let prev = -1;
        for (let t = 0; t < T; t++) {
            if (valid[t]) { f[t] = p[t]; prev = t; continue; }
            if (t < first) { f[t] = p[first]; continue; }
            if (t > last) { f[t] = p[last]; continue; }
            let nx = t; while (!valid[nx]) nx++;
            const w = (t - prev) / (nx - prev); f[t] = p[prev] * (1 - w) + p[nx] * w;
        }
        return f;
    }

    function velocity5(p, fs) {
        const T = p.length, v = new Float64Array(T);
        for (let t = 2; t + 2 < T; t++) v[t] = (p[t + 2] + p[t + 1] - p[t - 1] - p[t - 2]) * fs / 6.0;
        return v;
    }

    // scipy.ndimage.median_filter(v, size=w, mode="nearest"), w odd; sorted sliding window
    function runningMedian(v, w) {
        const T = v.length, h = w >> 1, out = new Float64Array(T);
        if (!T) return out;
        const at = i => v[Math.min(Math.max(i, 0), T - 1)];
        const win = [];
        for (let k = -h; k <= h; k++) win.push(at(k));
        win.sort((a, b) => a - b);
        const lowerBound = (x) => { let lo = 0, hi = win.length; while (lo < hi) { const m = (lo + hi) >> 1; if (win[m] < x) lo = m + 1; else hi = m; } return lo; };
        for (let t = 0; t < T; t++) {
            out[t] = win[h];
            if (t + 1 < T) {
                win.splice(lowerBound(at(t - h)), 1);
                const x = at(t + 1 + h); win.splice(lowerBound(x), 0, x);
            }
        }
        return out;
    }

    function robustSigma(v) {
        const sq = new Float64Array(v.length); for (let i = 0; i < v.length; i++) sq[i] = v[i] * v[i];
        const m2 = median(sq), m1 = median(v); return Math.sqrt(Math.max(m2 - m1 * m1, 1e-12));
    }

    function scoreTrace(x, y, valid, fs) {
        const T = x.length, fx = fillPositions(x, valid), fy = fillPositions(y, valid);
        const vx = velocity5(fx, fs), vy = velocity5(fy, fs);
        const w = Math.trunc(0.1 * fs) | 1, mx = runningMedian(vx, w), my = runningMedian(vy, w);
        const dvx = new Float64Array(T), dvy = new Float64Array(T);
        for (let t = 0; t < T; t++) { dvx[t] = vx[t] - mx[t]; dvy[t] = vy[t] - my[t]; }
        const noise = Math.max(0.5 * (robustSigma(dvx) + robustSigma(dvy)), 1e-9);
        const score = new Float32Array(T);
        for (let t = 0; t < T; t++) score[t] = Math.fround(Math.log10(1.0 + Math.hypot(dvx[t], dvy[t]) / noise));
        return { fx, fy, vx, vy, score, noise };
    }

    function viterbi(f, valid, h) {
        const T = f.length, K = h.d + 1, lab = new Uint8Array(T); if (!T) return lab;
        const A = new Float64Array(K * K).fill(-1e9);
        A[0] = Math.log(1 - h.p_on); A[1] = Math.log(h.p_on);
        for (let i = 1; i < h.d; i++) A[i * K + i + 1] = 0.0;
        A[h.d * K + h.d] = Math.log(h.p_stay); A[h.d * K + 0] = Math.log(1 - h.p_stay);
        const back = new Int8Array(T * K); let delta = new Float64Array(K).fill(-1e9); const ll = new Float64Array(K);
        const lsd0 = Math.log(h.sd0), lsd1 = Math.log(h.sd1);
        const emit = (t) => {
            if (!valid[t]) { ll.fill(0); return; }
            const z0 = (f[t] - h.mu0) / h.sd0, z1 = (f[t] - h.mu1) / h.sd1; ll[0] = -0.5 * z0 * z0 - lsd0;
            const s = -0.5 * z1 * z1 - lsd1; for (let k = 1; k < K; k++) ll[k] = s;
        };
        emit(0); delta[0] = ll[0];
        let nd = new Float64Array(K);
        for (let t = 1; t < T; t++) {
            emit(t);
            for (let j = 0; j < K; j++) {
                let best = -1e300, arg = 0;
                for (let i = 0; i < K; i++) { const s = delta[i] + A[i * K + j]; if (s > best) { best = s; arg = i; } }
                nd[j] = best + ll[j]; back[t * K + j] = arg;
            }
            const tmp = delta; delta = nd; nd = tmp;
        }
        let st = 0; for (let k = 1; k < K; k++) if (delta[k] > delta[st]) st = k;
        for (let t = T - 1; ; t--) { lab[t] = st > 0 ? 1 : 0; if (t === 0) break; st = back[t * K + st]; }
        return lab;
    }

    function runsOf(m) {
        const ev = [], T = m.length;
        for (let t = 0; t < T; t++) if (m[t] && (t === 0 || !m[t - 1])) { let e = t; while (e + 1 < T && m[e + 1]) e++; ev.push([t, e]); t = e; }
        return ev;
    }

    function dropInvalidZoneAndShort(lab, valid, fs, minSamples, marginMs = 20.0) {
        const T = lab.length, m = nearbyint(marginMs * fs / 1000.0), bad = new Uint8Array(T);
        for (let t = 0; t < T; t++) if (!valid[t]) for (let k = Math.max(0, t - m); k <= Math.min(T - 1, t + m); k++) bad[k] = 1;
        for (let t = 0; t < T; t++) if (bad[t]) lab[t] = 0;
        for (const [s, e] of runsOf(lab)) if (e - s + 1 < minSamples) for (let t = s; t <= e; t++) lab[t] = 0;
    }

    function makeSG(w) {                                            // order-3 Savitzky-Golay: 4 x w pseudo-inverse of the Vandermonde matrix over -h..h
        const h = (w - 1) / 2, n = 4, V = [];
        for (let i = 0; i < w; i++) { const x = i - h; V.push([1, x, x * x, x * x * x]); }
        const M = [];
        for (let a = 0; a < n; a++) { M.push(new Float64Array(2 * n)); for (let b = 0; b < n; b++) { let s = 0; for (let i = 0; i < w; i++) s += V[i][a] * V[i][b]; M[a][b] = s; M[a][n + b] = a === b ? 1 : 0; } }
        for (let c = 0; c < n; c++) {
            let p = c; for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
            const tmp = M[c]; M[c] = M[p]; M[p] = tmp;
            const d = M[c][c]; for (let k = 0; k < 2 * n; k++) M[c][k] /= d;
            for (let r = 0; r < n; r++) if (r !== c) { const fct = M[r][c]; for (let k = 0; k < 2 * n; k++) M[r][k] -= fct * M[c][k]; }
        }
        const P = new Float64Array(n * w);
        for (let a = 0; a < n; a++) for (let i = 0; i < w; i++) { let acc = 0; for (let b = 0; b < n; b++) acc += M[a][n + b] * V[i][b]; P[a * w + i] = acc; }
        return { w, P };
    }

    function sgDeriv(y, sg, fs) {
        const T = y.length, w = sg.w, h = (w - 1) / 2, d1 = new Float64Array(T), d2 = new Float64Array(T);
        for (let i = 0; i < T; i++) {
            const s0 = Math.min(Math.max(i - h, 0), T - w), c = [0, 0, 0, 0];
            for (let a = 0; a < 4; a++) { let acc = 0; for (let j = 0; j < w; j++) acc += sg.P[a * w + j] * y[s0 + j]; c[a] = acc; }
            const u = i - (s0 + h); d1[i] = (c[1] + 2 * c[2] * u + 3 * c[3] * u * u) * fs; d2[i] = (2 * c[2] + 6 * c[3] * u) * fs * fs;
        }
        return [d1, d2];
    }

    // acceleration-zero boundaries; defaults of the C++ detector: 15 ms window, 25 ms margin, offset after the post-saccadic oscillation ("last")
    function refineEvents(ev, fx, fy, valid, fs, r = { winMs: 15.0, marginMs: 25.0 }) {
        const T = fx.length, out = new Uint8Array(T); if (!ev.length) return out;
        let w = nearbyint(r.winMs * fs / 1000.0) | 1; w = Math.max(w, 5); w = Math.max(Math.min(w, Math.trunc(T / 2) * 2 - 1), 5);
        if (T < w) return out;
        const sg = makeSG(w), [vx, ax] = sgDeriv(fx, sg, fs), [vy, ay] = sgDeriv(fy, sg, fs);
        const sp = new Float64Array(T), at = new Float64Array(T);
        for (let t = 0; t < T; t++) { sp[t] = Math.hypot(vx[t], vy[t]); at[t] = (vx[t] * ax[t] + vy[t] * ay[t]) / Math.max(sp[t], 1e-9); }
        let nv = 0; for (let t = 0; t < T; t++) nv += valid[t] ? 1 : 0;
        const sx = [], sy = []; for (let t = 0; t < T; t++) if (nv < 20 || valid[t]) { sx.push(vx[t]); sy.push(vy[t]); }
        const floor = 3.0 * 0.5 * (robustSigma(sx) + robustSigma(sy)), m = Math.max(nearbyint(r.marginMs * fs / 1000.0), 2);
        for (let q = 0; q < ev.length; q++) {
            const s = ev[q][0], e = ev[q][1];
            const lo = Math.max(s - m, q > 0 ? ev[q - 1][1] + 1 : 0), hi = Math.min(e + m, q + 1 < ev.length ? ev[q + 1][0] - 1 : T - 1);
            let pk = s; for (let t = s; t <= e; t++) if (sp[t] > sp[pk]) pk = t;
            let j = pk - 1; while (j >= lo && at[j] > 0) j--;
            const on = Math.min(j + 1, pk); let off = e;
            let last = -1; for (let t = pk; t <= hi; t++) if (sp[t] >= floor) last = t;
            if (last >= 0) { j = last; while (j <= hi && at[j] < 0) j++; off = Math.min(j, hi); }
            if (off >= on) for (let t = on; t <= off; t++) out[t] = 1;
        }
        return out;
    }

    // whole trial -> saccade proposals [{onset, offset}] (inclusive sample indices)
    function detect(x, y, validIn, fs) {
        const T = x.length, valid = new Uint8Array(T);
        for (let i = 0; i < T; i++) valid[i] = (!validIn || validIn[i]) && Number.isFinite(x[i]) && Number.isFinite(y[i]) ? 1 : 0;
        if (T < 10) return [];
        const h = paramsFor(fs), k = scoreTrace(x, y, valid, fs);
        const lab = viterbi(k.score, valid, h); dropInvalidZoneAndShort(lab, valid, fs, h.d);
        const ev = runsOf(lab);
        return runsOf(refineEvents(ev, k.fx, k.fy, valid, fs)).map(([s, e]) => ({ onset: s, offset: e }));
    }

    const api = { detect, paramsFor, scoreTrace, viterbi, refineEvents, runningMedian, runsOf };
    if (typeof module !== 'undefined' && module.exports) module.exports = api;
    root.UnEyeHMM = api;
})(typeof self !== 'undefined' ? self : this);
