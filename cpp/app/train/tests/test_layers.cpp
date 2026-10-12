// T09a: analytic gradients of every layer (and of the whole U-Net) against central finite differences, in double precision (relative error < 1e-6).
#include <cstdio>
#include <functional>
#include <random>
#include <algorithm>
#include <cmath>
#include "eyelive/train/unet.hpp"
using namespace eyelive::train;
using D = double;
static int fails = 0;
static std::mt19937_64 rng(5);
static std::vector<D> rnd(size_t n, D s = 1.0) { std::normal_distribution<D> nd(0, s); std::vector<D> v(n); for (auto& x : v) x = nd(rng); return v; }
static void report(const char* name, D err, D tol = 1e-6) { std::printf("%-30s max rel err %.2e %s\n", name, err, err < tol ? "ok" : "FAIL"); if (!(err < tol)) ++fails; }
static D relerr(D a, D n) { return std::fabs(a - n) / std::max(1e-3, std::max(std::fabs(a), std::fabs(n))); }
// central finite differences of loss() with respect to every `stride`-th element of x, compared with `analytic`
static D fd(std::vector<D>& x, const std::vector<D>& analytic, const std::function<D()>& loss, size_t stride = 1) {
    D worst = 0; const D h = 1e-5;
    for (size_t i = 0; i < x.size(); i += stride) { const D o = x[i]; x[i] = o + h; const D lp = loss(); x[i] = o - h; const D lm = loss(); x[i] = o; worst = std::max(worst, relerr(analytic[i], (lp - lm) / (2 * h))); }
    return worst;
}
static D dot(const std::vector<D>& a, const std::vector<D>& b) { D s = 0; for (size_t i = 0; i < a.size(); ++i) s += a[i] * b[i]; return s; }

template <class Layer> void check_param_layer(const char* name, Layer& l, int cin, int cout_t_mul, int cout, int B, int T) {
    auto in = rnd(size_t(B) * cin * T); const int To = T * cout_t_mul; auto r = rnd(size_t(B) * cout * To); std::vector<D> out(r.size()), gin(in.size());
    auto L = [&] { l.forward(in.data(), out.data(), B, T); return dot(r, out); };
    l.w.zero_grad(); l.b.zero_grad(); l.backward(in.data(), r.data(), gin.data(), B, T);
    report((std::string(name) + " d input").c_str(), fd(in, gin, L));
    report((std::string(name) + " d weight").c_str(), fd(l.w.w, l.w.g, L));
    report((std::string(name) + " d bias").c_str(), fd(l.b.w, l.b.g, L));
}

int main() {
    const int B = 3, T = 25;
    { Conv1d<D> c; c.init(3, 4, 5); c.w.w = rnd(c.w.w.size(), 0.5); c.b.w = rnd(4); check_param_layer("conv1d k5", c, 3, 1, 4, B, T); }
    { Conv1d<D> c; c.init(4, 3, 1); c.w.w = rnd(c.w.w.size(), 0.5); c.b.w = rnd(3); check_param_layer("conv1d k1", c, 4, 1, 3, B, T); }
    { Conv1d<D> c; c.init(3, 4, 5); c.threads = 3; c.w.w = rnd(c.w.w.size(), 0.5); c.b.w = rnd(4); check_param_layer("conv1d k5 3 threads", c, 3, 1, 4, B, T); }
    { UpConv1d<D> u; u.init(3, 4, 5); u.w.w = rnd(u.w.w.size(), 0.5); u.b.w = rnd(4); check_param_layer("upconv1d", u, 3, 5, 4, B, 5); }
    {   // max pooling: distinct values so the argmax is stable under the finite-difference step
        MaxPool1d<D> p; p.mp = 5; const int C = 4; auto in = rnd(size_t(B) * C * T); auto r = rnd(size_t(B) * C * (T / 5)); std::vector<D> out(r.size()), gin(in.size());
        auto L = [&] { p.forward(in.data(), out.data(), B, C, T); return dot(r, out); }; L(); p.backward(r.data(), gin.data(), B, C, T); report("maxpool1d d input", fd(in, gin, L));
    }
    { auto in = rnd(200), r = rnd(200); std::vector<D> out(200), gin(200); auto L = [&] { relu_forward(in.data(), out.data(), 200); return dot(r, out); }; relu_backward(in.data(), r.data(), gin.data(), 200); report("relu d input", fd(in, gin, L)); }
    {   // batch norm, training mode (the loss depends on the batch statistics, so the full formula is checked)
        const int C = 4; BatchNorm1d<D> bn; bn.init(C); bn.gamma.w = rnd(C, 0.5); for (auto& g : bn.gamma.w) g += 1; bn.beta.w = rnd(C);
        auto in = rnd(size_t(B) * C * T, 2.0), r = rnd(in.size()); std::vector<D> out(in.size()), gin(in.size());
        auto L = [&] { bn.forward(in.data(), out.data(), B, T, true); return dot(r, out); };
        L(); bn.gamma.zero_grad(); bn.beta.zero_grad(); bn.backward(in.data(), r.data(), gin.data(), B, T);
        report("batchnorm d input", fd(in, gin, L)); report("batchnorm d gamma", fd(bn.gamma.w, bn.gamma.g, L)); report("batchnorm d beta", fd(bn.beta.w, bn.beta.g, L));
        // running statistics: momentum 0.1, unbiased variance
        BatchNorm1d<D> b2; b2.init(1); std::vector<D> x{1, 2, 3, 4}, o(4); b2.forward(x.data(), o.data(), 1, 4, true);
        report("batchnorm running mean", std::fabs(b2.rmean[0] - 0.25), 1e-12); report("batchnorm running var", std::fabs(b2.rvar[0] - (0.9 + 0.1 * (5.0 / 3.0))), 1e-12);
    }
    {   // loss: gradient w.r.t. the logits (with an ignored sample), and the value against a direct computation
        const int K = 3; auto z = rnd(size_t(B) * K * T); std::vector<D> tg(z.size(), 0.0), gz(z.size());
        for (int b = 0; b < B; ++b) for (int t = 0; t < T; ++t) if (t != 3) tg[(size_t(b) * K + (t % K)) * T + t] = 1;
        auto L = [&] { std::vector<D> g(z.size()); return softmax_mcloss(z.data(), tg.data(), g.data(), B, K, T); };
        softmax_mcloss(z.data(), tg.data(), gz.data(), B, K, T); report("softmax MCLoss d logits", fd(z, gz, L));
    }
    {   // whole network, tiny: all parameter gradients (every 7th) and the L2 term, in training mode
        const int Tn = 50, Bn = 2, K = 3; UNet<D> net(K, 5, 5); net.init_random(11);
        for (auto* p : net.params()) for (auto& v : p->w) v += 0.05 * std::normal_distribution<D>()(rng);
        auto x = rnd(size_t(Bn) * 2 * Tn, 0.3); std::vector<D> tg(size_t(Bn) * K * Tn, 0.0), gz(tg.size());
        for (int b = 0; b < Bn; ++b) for (int t = 0; t < Tn; ++t) tg[(size_t(b) * K + ((t / 7) % K)) * Tn + t] = 1;
        auto L = [&] { net.zero_grad(); auto& z = net.forward(x.data(), Bn, Tn, true); std::vector<D> g(z.size()); return softmax_mcloss(z.data(), tg.data(), g.data(), Bn, K, Tn) + 0.001 * net.l2(); };
        net.zero_grad(); auto& z = net.forward(x.data(), Bn, Tn, true); softmax_mcloss(z.data(), tg.data(), gz.data(), Bn, K, Tn); net.backward(gz.data()); net.add_l2_grad(0.001);
        D worst = 0; const D h = 1e-5; int n = 0; std::vector<std::vector<D>> all_g; for (auto* p : net.params()) all_g.push_back(p->g); size_t pi = 0;      // L() zeroes the gradients: keep a copy
        for (auto* p : net.params()) { const auto& an = all_g[pi++]; for (size_t i = 0; i < p->w.size(); i += 7) { const D o = p->w[i]; p->w[i] = o + h; const D lp = L(); p->w[i] = o - h; const D lm = L(); p->w[i] = o; worst = std::max(worst, relerr(an[i], (lp - lm) / (2 * h))); ++n; } }
        std::printf("whole U-Net: %zu parameters, %d gradient entries checked\n", net.n_params(), n); report("U-Net all parameter gradients", worst, 1e-4);   // finite differences through ReLU kinks and 18 613 parameters are noisier than a single layer (1.6e-5 on libstdc++, whose normal_distribution gives other weights than libc++); a real error gives ~1
    }
    std::printf(fails ? "FAILED\n" : "all gradient checks passed\n");
    return fails ? 1 : 0;
}
