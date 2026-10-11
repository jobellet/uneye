// The U'n'Eye U-Net (uneye/functions.py::UNet) with hand-written backpropagation. Input [B][2][T] = (dX, dY) per sample, T a multiple of mp^2; output logits [B][K][T] (softmax is in the loss).
// state() / load_state() use the order and layout of the cpp/models/*.bin files (conv: weight, bias; batch norm: gamma, beta, running mean, running var; c0 weights as [o][k][c]).
#pragma once
#include "eyelive/train/layers.hpp"

namespace eyelive { namespace train {

template <class R> struct Block {                       // conv (or transposed conv) -> ReLU -> batch norm
    Conv1d<R> conv; UpConv1d<R> up; bool is_up = false; BatchNorm1d<R> bn; std::vector<R> co, ro, bo;
    int cin = 0, cout = 0, tmul = 1;                    // output time = Tin * tmul
    void init_conv(int ci, int c, int k) { is_up = false; cin = ci; cout = c; tmul = 1; conv.init(ci, c, k); bn.init(c); }
    void init_up(int ci, int c, int mp) { is_up = true; cin = ci; cout = c; tmul = mp; up.init(ci, c, mp); bn.init(c); }
    void set_threads(int n) { conv.threads = n; up.threads = n; }
    const std::vector<R>& forward(const R* in, int B, int Tin, bool train) {
        const int To = Tin * tmul; const size_t n = size_t(B) * cout * To; co.resize(n); ro.resize(n); bo.resize(n);
        if (is_up) up.forward(in, co.data(), B, Tin); else conv.forward(in, co.data(), B, Tin);
        relu_forward(co.data(), ro.data(), n); bn.forward(ro.data(), bo.data(), B, To, train); return bo;
    }
    void backward(const R* in, const R* gout, R* gin, int B, int Tin) {
        const int To = Tin * tmul; const size_t n = size_t(B) * cout * To; std::vector<R> g1(n), g2(n);
        bn.backward(ro.data(), gout, g1.data(), B, To); relu_backward(co.data(), g1.data(), g2.data(), n);
        if (is_up) up.backward(in, g2.data(), gin, B, Tin); else conv.backward(in, g2.data(), gin, B, Tin);
    }
};

template <class R> class UNet {
 public:
    int K = 2, ks = 5, mp = 5;
    explicit UNet(int classes = 2, int kernel = 5, int pool = 5, int threads = 1) : K(classes), ks(kernel), mp(pool) {
        c0.init_conv(2, 10, ks); c1.init_conv(10, 20, ks); c2.init_conv(20, 20, ks); c3.init_conv(20, 20, ks); up1.init_up(20, 20, mp); c4.init_conv(40, 20, ks);
        up2.init_up(20, 20, mp); c5.init_conv(40, 20, ks); c6.init_conv(20, 10, ks); c7.init(10, K, 1); set_threads(threads);
    }
    void set_threads(int n) { for (auto* b : blocks()) b->set_threads(n); c7.threads = n; }

    // PyTorch-like initialisation: conv weights N(0, 0.02) (weights_init), biases U(-1/sqrt(fan_in), 1/sqrt(fan_in)), batch norm gamma 1, beta 0
    void init_random(uint64_t seed) {
        std::mt19937_64 rng(seed); std::normal_distribution<double> nd(0.0, 0.02); std::uniform_real_distribution<double> ud(-1.0, 1.0);
        for (auto* b : blocks()) {
            auto& w = b->is_up ? b->up.w : b->conv.w; auto& bi = b->is_up ? b->up.b : b->conv.b;
            const double fan = b->is_up ? double(b->cout) * mp : double(b->cin) * (b == &c0 ? ks : ks);      // c0: PyTorch's Conv2d fan_in = 1 * ks * 2 = cin(2) * ks
            for (auto& x : w.w) x = R(nd(rng)); for (auto& x : bi.w) x = R(ud(rng) / std::sqrt(fan));
        }
        for (auto& x : c7.w.w) x = R(nd(rng)); for (auto& x : c7.b.w) x = R(ud(rng) / std::sqrt(double(10)));
    }

    // x[B][2][T] -> logits [B][K][T] (valid until the next forward)
    const std::vector<R>& forward(const R* x, int B, int T, bool train) {
        B_ = B; T_ = T; x_ = x; const int T1 = T / mp, T2 = T1 / mp;
        a0 = &c0.forward(x, B, T, train); a1 = &c1.forward(a0->data(), B, T, train);
        p1.resize(size_t(B) * 20 * T1); pool1.forward(a1->data(), p1.data(), B, 20, T);
        a2 = &c2.forward(p1.data(), B, T1, train);
        p2.resize(size_t(B) * 20 * T2); pool2.forward(a2->data(), p2.data(), B, 20, T1);
        a3 = &c3.forward(p2.data(), B, T2, train);
        u1 = &up1.forward(a3->data(), B, T2, train);
        cat1 = concat(p1.data(), 20, u1->data(), 20, B, T1);
        a4 = &c4.forward(cat1.data(), B, T1, train);
        u2 = &up2.forward(a4->data(), B, T1, train);
        cat2 = concat(a1->data(), 20, u2->data(), 20, B, T);
        a5 = &c5.forward(cat2.data(), B, T, train); a6 = &c6.forward(a5->data(), B, T, train);
        logits.resize(size_t(B) * K * T); c7.forward(a6->data(), logits.data(), B, T);
        return logits;
    }
    // gz = dLoss/dlogits [B][K][T]; accumulates parameter gradients (loss gradient wrt the input is not needed)
    void backward(const R* gz) {
        const int B = B_, T = T_, T1 = T / mp, T2 = T1 / mp; std::vector<R> g6(size_t(B) * 10 * T), g5(size_t(B) * 20 * T), gcat2(size_t(B) * 40 * T), gu2(size_t(B) * 20 * T), ga1b(size_t(B) * 20 * T);
        c7.backward(a6->data(), gz, g6.data(), B, T);
        c6.backward(a5->data(), g6.data(), g5.data(), B, T);
        c5.backward(cat2.data(), g5.data(), gcat2.data(), B, T);
        split(gcat2.data(), ga1b.data(), 20, gu2.data(), 20, B, T);                   // ga1b: gradient of a1 through the skip connection
        std::vector<R> g4(size_t(B) * 20 * T1), gcat1(size_t(B) * 40 * T1), gp1b(size_t(B) * 20 * T1), gu1(size_t(B) * 20 * T1);
        up2.backward(a4->data(), gu2.data(), g4.data(), B, T1);
        c4.backward(cat1.data(), g4.data(), gcat1.data(), B, T1);
        split(gcat1.data(), gp1b.data(), 20, gu1.data(), 20, B, T1);
        std::vector<R> g3(size_t(B) * 20 * T2), gp2(size_t(B) * 20 * T2), ga2(size_t(B) * 20 * T1), gp1a(size_t(B) * 20 * T1), gp1(size_t(B) * 20 * T1), ga1a(size_t(B) * 20 * T), ga1(size_t(B) * 20 * T), ga0(size_t(B) * 10 * T), gx(size_t(B) * 2 * T);
        up1.backward(a3->data(), gu1.data(), g3.data(), B, T2);
        c3.backward(p2.data(), g3.data(), gp2.data(), B, T2);
        pool2.backward(gp2.data(), ga2.data(), B, 20, T1);
        c2.backward(p1.data(), ga2.data(), gp1a.data(), B, T1);
        for (size_t i = 0; i < gp1.size(); ++i) gp1[i] = gp1a[i] + gp1b[i];
        pool1.backward(gp1.data(), ga1a.data(), B, 20, T);
        for (size_t i = 0; i < ga1.size(); ++i) ga1[i] = ga1a[i] + ga1b[i];
        c1.backward(a0->data(), ga1.data(), ga0.data(), B, T);
        c0.backward(x_, ga0.data(), gx.data(), B, T);
    }

    std::vector<Param<R>*> params() { std::vector<Param<R>*> p; for (auto* b : blocks()) { if (b->is_up) { p.push_back(&b->up.w); p.push_back(&b->up.b); } else { p.push_back(&b->conv.w); p.push_back(&b->conv.b); } p.push_back(&b->bn.gamma); p.push_back(&b->bn.beta); } p.push_back(&c7.w); p.push_back(&c7.b); return p; }
    size_t n_params() { size_t n = 0; for (auto* p : params()) n += p->w.size(); return n; }
    void zero_grad() { for (auto* p : params()) p->zero_grad(); }
    double l2() { double s = 0; for (auto* p : params()) for (R v : p->w) s += double(v) * v; return s; }       // sum of squared parameters (the original adds 0.001 * l2 to the loss)
    void add_l2_grad(double lambda) { for (auto* p : params()) for (size_t i = 0; i < p->w.size(); ++i) p->g[i] += R(2 * lambda * p->w[i]); }

    // .bin order tensors (see header comment)
    std::vector<std::vector<R>> state() {
        std::vector<std::vector<R>> s;
        for (auto* b : blocks()) {
            if (b->is_up) { s.push_back(b->up.w.w); s.push_back(b->up.b.w); }
            else if (b == &c0) s.push_back(c0_to_file(b->conv.w.w)), s.push_back(b->conv.b.w);
            else { s.push_back(b->conv.w.w); s.push_back(b->conv.b.w); }
            s.push_back(b->bn.gamma.w); s.push_back(b->bn.beta.w); s.push_back(b->bn.rmean); s.push_back(b->bn.rvar);
        }
        s.push_back(c7.w.w); s.push_back(c7.b.w); return s;
    }
    std::vector<std::vector<R>> grads() {            // same order; batch norm entries: gamma grad, beta grad only
        std::vector<std::vector<R>> s;
        for (auto* b : blocks()) {
            if (b->is_up) { s.push_back(b->up.w.g); s.push_back(b->up.b.g); }
            else if (b == &c0) s.push_back(c0_to_file(b->conv.w.g)), s.push_back(b->conv.b.g);
            else { s.push_back(b->conv.w.g); s.push_back(b->conv.b.g); }
            s.push_back(b->bn.gamma.g); s.push_back(b->bn.beta.g);
        }
        s.push_back(c7.w.g); s.push_back(c7.b.g); return s;
    }
    bool load_state(const std::vector<std::vector<R>>& s, bool keep_head = false) {   // keep_head: do not load c7 (class count differs)
        size_t j = 0;
        for (auto* b : blocks()) {
            auto& w = b->is_up ? b->up.w.w : b->conv.w.w; auto& bi = b->is_up ? b->up.b.w : b->conv.b.w;
            if (s[j].size() != w.size() || s[j + 1].size() != bi.size()) return false;
            w = (b == &c0) ? c0_from_file(s[j]) : s[j]; bi = s[j + 1];
            if (s[j + 2].size() != b->bn.gamma.w.size()) return false;
            b->bn.gamma.w = s[j + 2]; b->bn.beta.w = s[j + 3]; b->bn.rmean = s[j + 4]; b->bn.rvar = s[j + 5]; j += 6;
        }
        if (!keep_head) { if (s[j].size() != c7.w.w.size()) return false; c7.w.w = s[j]; c7.b.w = s[j + 1]; }
        return true;
    }
    std::vector<R> param_snapshot() { std::vector<R> v; for (auto* p : params()) v.insert(v.end(), p->w.begin(), p->w.end()); for (auto* b : blocks()) { v.insert(v.end(), b->bn.rmean.begin(), b->bn.rmean.end()); v.insert(v.end(), b->bn.rvar.begin(), b->bn.rvar.end()); } return v; }
    void restore_snapshot(const std::vector<R>& v) { size_t j = 0; for (auto* p : params()) { std::copy(v.begin() + j, v.begin() + j + p->w.size(), p->w.begin()); j += p->w.size(); }
        for (auto* b : blocks()) { std::copy(v.begin() + j, v.begin() + j + b->bn.C, b->bn.rmean.begin()); j += b->bn.C; std::copy(v.begin() + j, v.begin() + j + b->bn.C, b->bn.rvar.begin()); j += b->bn.C; } }

    Block<R> c0, c1, c2, c3, up1, c4, up2, c5, c6; Conv1d<R> c7;
 private:
    std::vector<Block<R>*> blocks() { return {&c0, &c1, &c2, &c3, &up1, &c4, &up2, &c5, &c6}; }
    std::vector<R> c0_to_file(const std::vector<R>& w) const { std::vector<R> f(w.size()); for (int o = 0; o < 10; ++o) for (int c = 0; c < 2; ++c) for (int k = 0; k < ks; ++k) f[(size_t(o) * ks + k) * 2 + c] = w[(size_t(o) * 2 + c) * ks + k]; return f; }
    std::vector<R> c0_from_file(const std::vector<R>& f) const { std::vector<R> w(f.size()); for (int o = 0; o < 10; ++o) for (int c = 0; c < 2; ++c) for (int k = 0; k < ks; ++k) w[(size_t(o) * 2 + c) * ks + k] = f[(size_t(o) * ks + k) * 2 + c]; return w; }
    static std::vector<R> concat(const R* a, int ca, const R* b, int cb, int B, int T) { std::vector<R> o(size_t(B) * (ca + cb) * T);
        for (int i = 0; i < B; ++i) { std::copy(a + size_t(i) * ca * T, a + size_t(i + 1) * ca * T, o.begin() + size_t(i) * (ca + cb) * T); std::copy(b + size_t(i) * cb * T, b + size_t(i + 1) * cb * T, o.begin() + (size_t(i) * (ca + cb) + ca) * T); } return o; }
    static void split(const R* g, R* ga, int ca, R* gb, int cb, int B, int T) { for (int i = 0; i < B; ++i) { std::copy(g + size_t(i) * (ca + cb) * T, g + (size_t(i) * (ca + cb) + ca) * T, ga + size_t(i) * ca * T); std::copy(g + (size_t(i) * (ca + cb) + ca) * T, g + size_t(i + 1) * (ca + cb) * T, gb + size_t(i) * cb * T); } }
    int B_ = 0, T_ = 0; const R* x_ = nullptr;
    const std::vector<R> *a0 = nullptr, *a1 = nullptr, *a2 = nullptr, *a3 = nullptr, *u1 = nullptr, *a4 = nullptr, *u2 = nullptr, *a5 = nullptr, *a6 = nullptr;
    std::vector<R> p1, p2, cat1, cat2, logits; MaxPool1d<R> pool1, pool2;
};

}}  // namespace eyelive::train
