// Layers with forward AND backward for the U'n'Eye U-Net (task T09a). Header-only templates on the scalar type R: the trainer uses float, the tests use double (gradient checks to 1e-6).
// Tensor layout: [B][C][T] row-major. backward() OVERWRITES the input gradient and ACCUMULATES parameter gradients (call zero_grad() before a batch).
#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <random>
#include <thread>
#include <vector>

namespace eyelive { namespace train {

// runs f(i) for i in [0, n) on up to `threads` threads (inline when threads <= 1)
template <class F> void par_for(int n, int threads, F f) {
    if (threads <= 1 || n <= 1) { for (int i = 0; i < n; ++i) f(i); return; }
    const int nt = std::min(threads, n);
    std::vector<std::thread> th;
    for (int t = 1; t < nt; ++t) th.emplace_back([&, t] { for (int i = t; i < n; i += nt) f(i); });
    for (int i = 0; i < n; i += nt) f(i);
    for (auto& x : th) x.join();
}

template <class R> struct Param { std::vector<R> w, g; void init(size_t n) { w.assign(n, R(0)); g.assign(n, R(0)); } void zero_grad() { std::fill(g.begin(), g.end(), R(0)); } };

// 1-D convolution, stride 1, 'same' zero padding (k odd). w[o][i][k]
template <class R> struct Conv1d {
    int cin = 0, cout = 0, k = 1; Param<R> w, b; int threads = 1;
    void init(int ci, int co, int kk) { cin = ci; cout = co; k = kk; w.init(size_t(co) * ci * kk); b.init(co); }
    void forward(const R* in, R* out, int B, int T) const {
        const int pd = (k - 1) / 2;
        par_for(B, threads, [&](int bi) {
            for (int o = 0; o < cout; ++o) {
                R* po = out + (size_t(bi) * cout + o) * T; std::fill(po, po + T, b.w[o]);
                for (int i = 0; i < cin; ++i) {
                    const R* pi = in + (size_t(bi) * cin + i) * T; const R* wk = &w.w[(size_t(o) * cin + i) * k];
                    for (int kk = 0; kk < k; ++kk) {
                        const int sh = kk - pd, t0 = std::max(0, -sh), t1 = std::min(T, T - sh); const R wv = wk[kk];
                        for (int t = t0; t < t1; ++t) po[t] += wv * pi[t + sh];
                    }
                }
            }
        });
    }
    void backward(const R* in, const R* gout, R* gin, int B, int T) {
        const int pd = (k - 1) / 2; std::mutex mu;
        std::fill(gin, gin + size_t(B) * cin * T, R(0));
        par_for(B, threads, [&](int bi) {
            std::vector<R> gw(w.g.size(), R(0)), gb(cout, R(0));
            for (int o = 0; o < cout; ++o) {
                const R* go = gout + (size_t(bi) * cout + o) * T; R s = 0; for (int t = 0; t < T; ++t) s += go[t]; gb[o] = s;
                for (int i = 0; i < cin; ++i) {
                    const R* pi = in + (size_t(bi) * cin + i) * T; R* gi = gin + (size_t(bi) * cin + i) * T;
                    const R* wk = &w.w[(size_t(o) * cin + i) * k]; R* gwk = &gw[(size_t(o) * cin + i) * k];
                    for (int kk = 0; kk < k; ++kk) {
                        const int sh = kk - pd, t0 = std::max(0, -sh), t1 = std::min(T, T - sh); const R wv = wk[kk]; R acc = 0;
                        for (int t = t0; t < t1; ++t) { acc += go[t] * pi[t + sh]; gi[t + sh] += wv * go[t]; }
                        gwk[kk] += acc;
                    }
                }
            }
            std::lock_guard<std::mutex> lk(mu);
            for (size_t j = 0; j < gw.size(); ++j) w.g[j] += gw[j];
            for (int o = 0; o < cout; ++o) b.g[o] += gb[o];
        });
    }
};

// ConvTranspose1d with kernel = stride = mp, no padding. w[i][o][j]; T -> T*mp
template <class R> struct UpConv1d {
    int cin = 0, cout = 0, mp = 1; Param<R> w, b; int threads = 1;
    void init(int ci, int co, int m) { cin = ci; cout = co; mp = m; w.init(size_t(ci) * co * m); b.init(co); }
    void forward(const R* in, R* out, int B, int T) const {
        const int To = T * mp;
        par_for(B, threads, [&](int bi) {
            for (int o = 0; o < cout; ++o) {
                R* po = out + (size_t(bi) * cout + o) * To; std::fill(po, po + To, b.w[o]);
                for (int i = 0; i < cin; ++i) {
                    const R* pi = in + (size_t(bi) * cin + i) * T; const R* wk = &w.w[(size_t(i) * cout + o) * mp];
                    for (int t = 0; t < T; ++t) for (int j = 0; j < mp; ++j) po[t * mp + j] += pi[t] * wk[j];
                }
            }
        });
    }
    void backward(const R* in, const R* gout, R* gin, int B, int T) {
        const int To = T * mp; std::mutex mu;
        std::fill(gin, gin + size_t(B) * cin * T, R(0));
        par_for(B, threads, [&](int bi) {
            std::vector<R> gw(w.g.size(), R(0)), gb(cout, R(0));
            for (int o = 0; o < cout; ++o) {
                const R* go = gout + (size_t(bi) * cout + o) * To; R s = 0; for (int t = 0; t < To; ++t) s += go[t]; gb[o] = s;
                for (int i = 0; i < cin; ++i) {
                    const R* pi = in + (size_t(bi) * cin + i) * T; R* gi = gin + (size_t(bi) * cin + i) * T;
                    const R* wk = &w.w[(size_t(i) * cout + o) * mp]; R* gwk = &gw[(size_t(i) * cout + o) * mp];
                    for (int t = 0; t < T; ++t) { R a = 0; for (int j = 0; j < mp; ++j) { a += go[t * mp + j] * wk[j]; gwk[j] += pi[t] * go[t * mp + j]; } gi[t] += a; }
                }
            }
            std::lock_guard<std::mutex> lk(mu);
            for (size_t j = 0; j < gw.size(); ++j) w.g[j] += gw[j];
            for (int o = 0; o < cout; ++o) b.g[o] += gb[o];
        });
    }
};

// max pooling, kernel = stride = mp (T a multiple of mp). The first maximum receives the gradient (as PyTorch)
template <class R> struct MaxPool1d {
    int mp = 5; int C = 0; std::vector<int32_t> arg;
    void forward(const R* in, R* out, int B, int Cc, int T) {
        C = Cc; const int To = T / mp; arg.assign(size_t(B) * C * To, 0);
        for (int bc = 0; bc < B * C; ++bc)
            for (int t = 0; t < To; ++t) {
                const R* p = in + size_t(bc) * T + size_t(t) * mp; int bj = 0; R m = p[0];
                for (int j = 1; j < mp; ++j) if (p[j] > m) { m = p[j]; bj = j; }
                out[size_t(bc) * To + t] = m; arg[size_t(bc) * To + t] = bj;
            }
    }
    void backward(const R* gout, R* gin, int B, int Cc, int T) const {
        const int To = T / mp; std::fill(gin, gin + size_t(B) * Cc * T, R(0));
        for (int bc = 0; bc < B * Cc; ++bc)
            for (int t = 0; t < To; ++t) gin[size_t(bc) * T + size_t(t) * mp + arg[size_t(bc) * To + t]] = gout[size_t(bc) * To + t];
    }
};

template <class R> void relu_forward(const R* in, R* out, size_t n) { for (size_t i = 0; i < n; ++i) out[i] = in[i] > R(0) ? in[i] : R(0); }
template <class R> void relu_backward(const R* in, const R* gout, R* gin, size_t n) { for (size_t i = 0; i < n; ++i) gin[i] = in[i] > R(0) ? gout[i] : R(0); }

// batch normalisation over (B, T) per channel; training uses batch statistics (biased variance) and updates the running statistics (momentum 0.1, unbiased variance), as PyTorch
template <class R> struct BatchNorm1d {
    int C = 0; Param<R> gamma, beta; std::vector<R> rmean, rvar, mean, inv; R eps = R(1e-5), momentum = R(0.1);
    void init(int c) { C = c; gamma.init(c); beta.init(c); std::fill(gamma.w.begin(), gamma.w.end(), R(1)); rmean.assign(c, R(0)); rvar.assign(c, R(1)); mean.assign(c, R(0)); inv.assign(c, R(1)); }
    void forward(const R* in, R* out, int B, int T, bool train) {
        const double N = double(B) * T;
        for (int c = 0; c < C; ++c) {
            R mu, iv;
            if (train) {
                double s = 0, s2 = 0;
                for (int b = 0; b < B; ++b) { const R* p = in + (size_t(b) * C + c) * T; for (int t = 0; t < T; ++t) s += p[t]; }
                const double m = s / N;
                for (int b = 0; b < B; ++b) { const R* p = in + (size_t(b) * C + c) * T; for (int t = 0; t < T; ++t) { const double d = p[t] - m; s2 += d * d; } }
                const double var = s2 / N; mu = R(m); iv = R(1.0 / std::sqrt(var + double(eps)));
                rmean[c] = (R(1) - momentum) * rmean[c] + momentum * R(m);
                rvar[c] = (R(1) - momentum) * rvar[c] + momentum * R(N > 1 ? var * N / (N - 1) : var);
                mean[c] = mu; inv[c] = iv;
            } else { mu = rmean[c]; iv = R(1.0 / std::sqrt(double(rvar[c]) + double(eps))); }
            const R g = gamma.w[c], be = beta.w[c];
            for (int b = 0; b < B; ++b) { const R* p = in + (size_t(b) * C + c) * T; R* q = out + (size_t(b) * C + c) * T; for (int t = 0; t < T; ++t) q[t] = (p[t] - mu) * iv * g + be; }
        }
    }
    void backward(const R* in, const R* gout, R* gin, int B, int T) {      // after a training-mode forward on the same `in`
        const double N = double(B) * T;
        for (int c = 0; c < C; ++c) {
            double sg = 0, sgx = 0; const R mu = mean[c], iv = inv[c];
            for (int b = 0; b < B; ++b) { const R* p = in + (size_t(b) * C + c) * T; const R* go = gout + (size_t(b) * C + c) * T;
                for (int t = 0; t < T; ++t) { const double xh = (p[t] - mu) * iv; sg += go[t]; sgx += go[t] * xh; } }
            gamma.g[c] += R(sgx); beta.g[c] += R(sg);
            const double k = double(gamma.w[c]) * iv / N;
            for (int b = 0; b < B; ++b) { const R* p = in + (size_t(b) * C + c) * T; const R* go = gout + (size_t(b) * C + c) * T; R* gi = gin + (size_t(b) * C + c) * T;
                for (int t = 0; t < T; ++t) { const double xh = (p[t] - mu) * iv; gi[t] = R(k * (N * go[t] - sg - xh * sgx)); } }
        }
    }
};

// U'n'Eye loss (uneye/functions.py::MCLoss): E = -mean(target * log(clamp(softmax(z), 1e-7, 1 - 1e-7))), the mean over ALL B*K*T elements. target[B][K][T] (all zeros for ignored samples).
// Returns E and writes dE/dz into gz.
template <class R> double softmax_mcloss(const R* z, const R* target, R* gz, int B, int K, int T) {
    const double eps = 1e-7, N = double(B) * K * T; double E = 0; std::vector<double> p(K), g(K);
    for (int b = 0; b < B; ++b)
        for (int t = 0; t < T; ++t) {
            double m = -1e300; for (int k = 0; k < K; ++k) m = std::max(m, double(z[(size_t(b) * K + k) * T + t]));
            double s = 0; for (int k = 0; k < K; ++k) s += (p[k] = std::exp(double(z[(size_t(b) * K + k) * T + t]) - m)); for (int k = 0; k < K; ++k) p[k] /= s;
            double dot = 0;
            for (int k = 0; k < K; ++k) {
                const double tg = target[(size_t(b) * K + k) * T + t], pc = std::min(std::max(p[k], eps), 1 - eps); E -= tg * std::log(pc);
                g[k] = (p[k] >= eps && p[k] <= 1 - eps) ? -tg / pc / N : 0.0; dot += g[k] * p[k];
            }
            for (int k = 0; k < K; ++k) gz[(size_t(b) * K + k) * T + t] = R(p[k] * (g[k] - dot));
        }
    return E / N;
}

}}  // namespace eyelive::train
