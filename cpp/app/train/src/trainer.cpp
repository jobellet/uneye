// ITrainer: U'n'Eye training in plain C++ (task T09b). Recipe of uneye/classifier.py::train: Adam, 10 minibatches per epoch, L2 0.001, validation loss each epoch, learning-rate halving + restoring the best weights when the validation loss does not improve, early stopping.
#include "eyelive/trainer.hpp"
#include "eyelive/types.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <fstream>
#include <mutex>
#include <numeric>
#include <random>
#include <sstream>
#include <thread>
#include "eyelive/train/dataset.hpp"
#include "eyelive/train/model_io.hpp"

namespace eyelive {
namespace {
using clk = std::chrono::steady_clock;
using namespace eyelive::train;

struct Adam {
    double lr = 1e-3, b1 = 0.9, b2 = 0.999, eps = 1e-8; long t = 0; std::vector<std::vector<float>> m, v;
    void step(const std::vector<Param<float>*>& ps) {
        if (m.empty()) for (auto* p : ps) { m.emplace_back(p->w.size(), 0.f); v.emplace_back(p->w.size(), 0.f); }
        ++t; const double c1 = 1 - std::pow(b1, double(t)), c2 = 1 - std::pow(b2, double(t));
        for (size_t k = 0; k < ps.size(); ++k) { auto& w = ps[k]->w; auto& g = ps[k]->g; auto& mk = m[k]; auto& vk = v[k];
            for (size_t i = 0; i < w.size(); ++i) { mk[i] = float(b1 * mk[i] + (1 - b1) * g[i]); vk[i] = float(b2 * vk[i] + (1 - b2) * double(g[i]) * g[i]);
                w[i] -= float(lr * (mk[i] / c1) / (std::sqrt(vk[i] / c2) + eps)); } }
    }
};

class Trainer : public ITrainer {
 public:
    ~Trainer() override { cancel(); join(); }
    void start(std::vector<TrainTrace> traces, const TrainConfig& cfg, ProgressFn progress) override {
        cancel_ = true; join(); cancel_ = false; running_ = true; has_model_ = false;
        worker_ = std::thread([this, tr = std::move(traces), cfg, progress]() mutable { run(tr, cfg, progress); running_ = false; });
    }
    void cancel() override { cancel_ = true; }
    bool running() const override { return running_; }
    bool save_model(const std::string& path, std::string* error) override {
        join();                                                           // wait for the worker (the model is complete only after it finished)
        std::lock_guard<std::mutex> lk(mu_);
        if (!has_model_) { if (error) *error = "no trained model available (training not finished, cancelled or failed)"; return false; }
        UNet<float> net(K_, 5, 5); net.init_random(1); net.restore_snapshot(best_); if (!save_bin(path, net, error)) return false;
        std::ofstream j(path.substr(0, path.size() >= 4 && path.compare(path.size() - 4, 4, ".bin") == 0 ? path.size() - 4 : path.size()) + ".json");
        if (!j) { if (error) *error = "cannot write the model sidecar"; return false; }
        j << "{\"window\": " << window_ << ", \"classes\": " << K_ << ", \"ks\": 5, \"mp\": 5, \"deg_per_unit\": " << deg_per_unit_ << ", \"rate_hz\": " << rate_ << ", \"epochs\": " << epochs_
          << ", \"val_loss\": " << val_loss_ << ", \"val_event_f1\": " << val_f1_ << ", \"val_kappa\": " << val_kappa_ << ", \"rotation_aug\": " << (rot_ ? "true" : "false")
          << ", \"train_windows\": " << n_train_ << ", \"source\": \"eyelive trainer\"}\n";
        return true;
    }

 private:
    void join() { if (worker_.joinable()) worker_.join(); }

    void run(std::vector<TrainTrace>& traces, const TrainConfig& cfg, const ProgressFn& progress) {
        const auto t0 = clk::now(); TrainProgress pg; auto elapsed = [&] { return std::chrono::duration<double>(clk::now() - t0).count(); };
        auto fail = [&](const std::string& msg) { pg.failed = true; pg.finished = true; pg.message = msg; pg.elapsed_s = elapsed(); if (progress) progress(pg); };
        if (cfg.n_classes < 2 || cfg.n_classes > kMaxClasses) return fail("the number of classes must be between 2 and 5");
        Dataset ds = make_dataset(traces, cfg); if (!ds.error.empty()) return fail(ds.error);
        traces.clear(); traces.shrink_to_fit();
        const int K = cfg.n_classes, W = ds.train.W, nthr = cfg.n_threads > 0 ? cfg.n_threads : std::max(1u, std::thread::hardware_concurrency() > 1 ? std::thread::hardware_concurrency() - 1 : 1u);
        UNet<float> net(K, 5, 5, nthr); net.init_random(cfg.seed);
        if (!cfg.init_model.empty()) { std::string err; if (!load_bin(cfg.init_model, net, &err)) return fail("init_model: " + err); }
        int ntr = ds.train.n, iters = 1, batch = 1; const int nval = ds.val.n;
        auto plan = [&] {                                                 // minibatches: cfg.batch_size windows, or the original 10 per epoch
            ntr = ds.train.n;
            if (cfg.batch_size > 0) { batch = cfg.batch_size; iters = std::max(1, (ntr + batch - 1) / batch); }
            else { iters = std::min(10, ntr); batch = std::max(1, ntr / iters); }
        };
        plan();
        std::mt19937_64 rng(cfg.seed + 17), rrng(cfg.seed + 101); std::vector<int> order(ntr); std::iota(order.begin(), order.end(), 0);
        Adam adam; adam.lr = cfg.learning_rate; auto params = net.params();
        const double l2 = 0.001;
        auto make_batch = [&](const Windows& w, const int* idx, int B, std::vector<float>& x, std::vector<float>& tg) {
            x.assign(size_t(B) * 2 * W, 0.f); tg.assign(size_t(B) * K * W, 0.f);
            for (int b = 0; b < B; ++b) { const int i = idx[b]; std::copy(&w.vel[size_t(i) * 2 * W], &w.vel[size_t(i) * 2 * W] + 2 * W, &x[size_t(b) * 2 * W]);
                for (int t = 0; t < W; ++t) { const uint8_t l = w.lab[size_t(i) * W + t]; if (l < K) tg[(size_t(b) * K + l) * W + t] = 1.f; } }
        };
        std::vector<float> xb, tb, gz;
        auto eval_val = [&](Scores* sc) {                                    // validation loss (eval-mode batch norm) + scores
            std::vector<int> idx(nval); std::iota(idx.begin(), idx.end(), 0); double loss = 0; std::vector<uint8_t> pred(size_t(nval) * W), truth = ds.val.lab; size_t done = 0;
            for (int s = 0; s < nval; s += 32) { const int B = std::min(32, nval - s); make_batch(ds.val, &idx[s], B, xb, tb); auto& z = net.forward(xb.data(), B, W, false); gz.resize(z.size());
                loss += softmax_mcloss(z.data(), tb.data(), gz.data(), B, K, W) * B;
                for (int b = 0; b < B; ++b) for (int t = 0; t < W; ++t) { int best = 0; float m = z[(size_t(b) * K) * W + t]; for (int k = 1; k < K; ++k) if (z[(size_t(b) * K + k) * W + t] > m) { m = z[(size_t(b) * K + k) * W + t]; best = k; } pred[done + size_t(b) * W + t] = uint8_t(best); }
                done += size_t(B) * W; }
            if (sc) *sc = score_predictions(pred, truth, W, K);
            return loss / nval + l2 * net.l2();
        };
        std::vector<double> Lv; double best_loss = 1e30; std::vector<float> snapshot = net.param_snapshot(); int worse = 0; Scores best_sc; double best_sc_loss = 0;
        int epoch = 0;
        while (epoch < cfg.max_epochs && !cancel_) {
            ++epoch;
            if (cfg.resample_each_epoch && epoch > 1) { resample_train(ds, cfg, rrng); plan(); order.resize(size_t(ntr)); std::iota(order.begin(), order.end(), 0); }
            std::shuffle(order.begin(), order.end(), rng); double ltr = 0; int cnt = 0;
            for (int it = 0; it < iters && !cancel_; ++it) {
                const int s = it * batch, B = (it == iters - 1) ? ntr - s : batch; if (B <= 0) continue;
                make_batch(ds.train, &order[s], B, xb, tb); net.zero_grad(); auto& z = net.forward(xb.data(), B, W, true); gz.resize(z.size());
                const double e = softmax_mcloss(z.data(), tb.data(), gz.data(), B, K, W); net.backward(gz.data()); net.add_l2_grad(l2); adam.step(params); ltr += e + l2 * net.l2(); ++cnt;
            }
            if (cancel_) break;
            Scores sc; const double lv = eval_val(&sc); Lv.push_back(lv);
            if (Lv.size() > 3) {
                const double mean3 = (Lv[Lv.size() - 2] + Lv[Lv.size() - 3] + Lv[Lv.size() - 4]) / 3.0;
                if (lv < mean3) { worse = 0; if (lv < best_loss) { best_loss = lv; snapshot = net.param_snapshot(); best_sc = sc; best_sc_loss = lv; } else net.restore_snapshot(snapshot); }
                else { adam.lr *= 0.5; ++worse; net.restore_snapshot(snapshot); }
            } else if (lv <= best_loss) { best_loss = lv; snapshot = net.param_snapshot(); best_sc = sc; best_sc_loss = lv; }
            pg.epoch = epoch; pg.train_loss = cnt ? ltr / cnt : 0; pg.val_loss = lv; pg.val_event_f1 = sc.event_f1; pg.val_kappa = sc.kappa; pg.elapsed_s = elapsed(); pg.finished = false; pg.message.clear();
            if (progress) progress(pg);
            if (worse > cfg.patience - 1) { pg.message = "early stopping"; break; }
        }
        if (cancel_) { pg.cancelled = true; pg.finished = true; pg.elapsed_s = elapsed(); pg.message = "cancelled"; if (progress) progress(pg); return; }
        net.restore_snapshot(snapshot);
        { std::lock_guard<std::mutex> lk(mu_); best_ = snapshot; K_ = K; window_ = W; deg_per_unit_ = cfg.deg_per_unit; rate_ = ds.rate_hz; epochs_ = epoch; val_loss_ = best_sc_loss; val_f1_ = best_sc.event_f1; val_kappa_ = best_sc.kappa; rot_ = cfg.rotation_aug; n_train_ = ntr; has_model_ = true; }
        pg.finished = true; pg.val_loss = best_sc_loss; pg.val_event_f1 = best_sc.event_f1; pg.val_kappa = best_sc.kappa; pg.elapsed_s = elapsed(); if (pg.message.empty()) pg.message = "max epochs reached"; if (progress) progress(pg);
    }

    std::thread worker_; std::atomic<bool> cancel_{false}, running_{false}; std::mutex mu_;
    bool has_model_ = false; std::vector<float> best_; int K_ = 2, window_ = 1000, epochs_ = 0, n_train_ = 0; double deg_per_unit_ = 1, rate_ = 1000, val_loss_ = 0, val_f1_ = 0, val_kappa_ = 0; bool rot_ = true;
};

}  // namespace

std::unique_ptr<ITrainer> make_trainer() { return std::make_unique<Trainer>(); }

}  // namespace eyelive
