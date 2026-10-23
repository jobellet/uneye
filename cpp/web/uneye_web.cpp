#include <emscripten/bind.h>
#include <emscripten/val.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "eyelive/trainer.hpp"
#include "eyelive/train/dataset.hpp"
#include "eyelive/train/layers.hpp"
#include "eyelive/train/model_io.hpp"
#include "eyelive/train/unet.hpp"

using namespace eyelive;
using namespace eyelive::train;
using namespace emscripten;

namespace {

struct ClassMetric {
    double f1 = 0;
    double kappa = 0;
    int n_pred = 0;
    int n_true = 0;
};

ClassMetric compute_class_metric(const uint8_t* pred, const uint8_t* truth, size_t n, int target_cls, int minlen = 3) {
    ClassMetric cm;
    int tp = 0, fp = 0, fn = 0, tn = 0;
    size_t valid = 0;
    for (size_t i = 0; i < n; ++i) {
        if (truth[i] >= 250) continue; // ignore unreviewed
        valid++;
        bool p = (pred[i] == target_cls);
        bool t = (truth[i] == target_cls);
        if (p && t) tp++;
        else if (p && !t) fp++;
        else if (!p && t) fn++;
        else tn++;
    }
    if (valid > 0) {
        double po = double(tp + tn) / valid;
        double pe = (double(tp + fp) * (tp + fn) + double(tn + fp) * (tn + fn)) / (double(valid) * valid);
        cm.kappa = pe < 1.0 ? (po - pe) / (1.0 - pe) : 1.0;
    }

    struct Segment { size_t s, e; };
    std::vector<Segment> p_runs, t_runs;
    for (size_t i = 0; i < n;) {
        if (pred[i] == target_cls) {
            size_t j = i;
            while (j < n && pred[j] == target_cls) ++j;
            if (j - i >= size_t(minlen)) p_runs.push_back({i, j - 1});
            i = j;
        } else {
            ++i;
        }
    }
    for (size_t i = 0; i < n;) {
        if (truth[i] == target_cls) {
            size_t j = i;
            while (j < n && truth[j] == target_cls) ++j;
            t_runs.push_back({i, j - 1});
            i = j;
        } else {
            ++i;
        }
    }

    cm.n_pred = int(p_runs.size());
    cm.n_true = int(t_runs.size());
    int tp_p = 0, tp_t = 0;
    for (const auto& pr : p_runs) {
        for (const auto& tr : t_runs) {
            if (pr.s <= tr.e && tr.s <= pr.e) { tp_p++; break; }
        }
    }
    for (const auto& tr : t_runs) {
        for (const auto& pr : p_runs) {
            if (pr.s <= tr.e && tr.s <= pr.e) { tp_t++; break; }
        }
    }

    double prec = p_runs.empty() ? 0.0 : double(tp_p) / p_runs.size();
    double rec = t_runs.empty() ? 0.0 : double(tp_t) / t_runs.size();
    cm.f1 = (prec + rec > 0.0) ? (2.0 * prec * rec / (prec + rec)) : 0.0;
    return cm;
}

struct AdamOpt {
    double lr = 1e-3, b1 = 0.9, b2 = 0.999, eps = 1e-8;
    long t = 0;
    std::vector<std::vector<float>> m, v;
    void step(const std::vector<Param<float>*>& ps) {
        if (m.empty()) {
            for (auto* p : ps) {
                m.emplace_back(p->w.size(), 0.f);
                v.emplace_back(p->w.size(), 0.f);
            }
        }
        ++t;
        const double c1 = 1 - std::pow(b1, double(t)), c2 = 1 - std::pow(b2, double(t));
        for (size_t k = 0; k < ps.size(); ++k) {
            auto& w = ps[k]->w; auto& g = ps[k]->g; auto& mk = m[k]; auto& vk = v[k];
            for (size_t i = 0; i < w.size(); ++i) {
                mk[i] = float(b1 * mk[i] + (1 - b1) * g[i]);
                vk[i] = float(b2 * vk[i] + (1 - b2) * double(g[i]) * g[i]);
                w[i] -= float(lr * (mk[i] / c1) / (std::sqrt(vk[i] / c2) + eps));
            }
        }
    }
};

} // namespace

class WebTrainer {
public:
    WebTrainer() = default;

    void reset() {
        traces_.clear();
        net_.reset();
        ds_ = Dataset();
        cancel_ = false;
        finished_ = false;
        epoch_ = 0;
        worse_ = 0;
        best_loss_ = 1e30;
        Lv_.clear();
        best_snapshot_.clear();
        best_scores_ = Scores();
        best_sc_loss_ = 0.0;
        init_model_bytes_.clear();
    }

    void set_config(int n_classes, double max_seconds_of_data, bool rotation_aug,
                    int rotations, double val_fraction, double max_speed_deg_s,
                    int max_epochs, int patience, double learning_rate,
                    uint64_t seed, double deg_per_unit) {
        cfg_.n_classes = n_classes;
        cfg_.max_seconds_of_data = max_seconds_of_data;
        cfg_.rotation_aug = rotation_aug;
        cfg_.rotations = rotations;
        cfg_.val_fraction = val_fraction;
        cfg_.max_speed_deg_s = max_speed_deg_s;
        cfg_.max_epochs = max_epochs;
        cfg_.patience = patience;
        cfg_.learning_rate = learning_rate;
        cfg_.seed = seed;
        cfg_.deg_per_unit = deg_per_unit;
        cfg_.n_threads = 1;
    }

    void add_trace(val x_val, val y_val, val labels_val, double rate_hz) {
        TrainTrace tr;
        tr.rate_hz = rate_hz;
        const size_t n = x_val["length"].as<size_t>();
        tr.x.resize(n);
        tr.y.resize(n);
        tr.labels.resize(n);
        for (size_t i = 0; i < n; ++i) {
            tr.x[i] = x_val[i].as<float>();
            tr.y[i] = y_val[i].as<float>();
            tr.labels[i] = labels_val[i].as<uint8_t>();
        }
        traces_.push_back(std::move(tr));
    }

    void set_init_model_bytes(val uint8_bytes) {
        const size_t len = uint8_bytes["length"].as<size_t>();
        init_model_bytes_.resize(len);
        for (size_t i = 0; i < len; ++i) {
            init_model_bytes_[i] = uint8_bytes[i].as<uint8_t>();
        }
    }

    std::string init_training() {
        if (cfg_.n_classes < 2 || cfg_.n_classes > 5) {
            return "n_classes must be between 2 and 5";
        }
        ds_ = make_dataset(traces_, cfg_);
        if (!ds_.error.empty()) {
            return ds_.error;
        }

        K_ = cfg_.n_classes;
        W_ = ds_.train.W;
        net_ = std::make_unique<UNet<float>>(K_, 5, 5, 1);
        net_->init_random(cfg_.seed);

        if (!init_model_bytes_.empty()) {
            std::string lerr;
            if (!load_bin_from_memory(init_model_bytes_.data(), init_model_bytes_.size(), *net_, &lerr)) {
                return "init_model: " + lerr;
            }
        }

        ntr_ = ds_.train.n;
        nval_ = ds_.val.n;
        iters_ = std::min(10, ntr_);
        batch_ = std::max(1, ntr_ / iters_);
        rng_.seed(cfg_.seed + 17);
        order_.resize(ntr_);
        std::iota(order_.begin(), order_.end(), 0);

        adam_ = AdamOpt();
        adam_.lr = cfg_.learning_rate;
        l2_ = 0.001;

        epoch_ = 0;
        worse_ = 0;
        best_loss_ = 1e30;
        best_sc_loss_ = 0;
        Lv_.clear();
        best_snapshot_ = net_->param_snapshot();
        finished_ = false;
        cancel_ = false;
        t0_ = std::chrono::steady_clock::now();
        return "";
    }

    std::string step() {
        if (finished_ || cancel_ || !net_) {
            return progress_json(epoch_, 0, 0, 0, 0, true, cancel_, cancel_ ? "cancelled" : "already finished");
        }

        ++epoch_;
        std::shuffle(order_.begin(), order_.end(), rng_);
        double ltr = 0;
        int cnt = 0;

        auto make_batch = [&](const Windows& w, const int* idx, int B, std::vector<float>& x, std::vector<float>& tg) {
            x.assign(size_t(B) * 2 * W_, 0.f);
            tg.assign(size_t(B) * K_ * W_, 0.f);
            for (int b = 0; b < B; ++b) {
                const int i = idx[b];
                std::copy(&w.vel[size_t(i) * 2 * W_], &w.vel[size_t(i) * 2 * W_] + 2 * W_, &x[size_t(b) * 2 * W_]);
                for (int t = 0; t < W_; ++t) {
                    const uint8_t l = w.lab[size_t(i) * W_ + t];
                    if (l < K_) tg[(size_t(b) * K_ + l) * W_ + t] = 1.f;
                }
            }
        };

        std::vector<float> xb, tb, gz;
        auto params = net_->params();

        for (int it = 0; it < iters_ && !cancel_; ++it) {
            const int s = it * batch_, B = (it == iters_ - 1) ? ntr_ - s : batch_;
            if (B <= 0) continue;
            make_batch(ds_.train, &order_[s], B, xb, tb);
            net_->zero_grad();
            auto& z = net_->forward(xb.data(), B, W_, true);
            gz.resize(z.size());
            const double e = softmax_mcloss(z.data(), tb.data(), gz.data(), B, K_, W_);
            net_->backward(gz.data());
            net_->add_l2_grad(l2_);
            adam_.step(params);
            ltr += e + l2_ * net_->l2();
            ++cnt;
        }

        if (cancel_) {
            finished_ = true;
            return progress_json(epoch_, ltr / std::max(1, cnt), 0, 0, 0, true, true, "cancelled");
        }

        // Eval validation
        Scores sc;
        double loss = 0;
        std::vector<int> idx_val(nval_);
        std::iota(idx_val.begin(), idx_val.end(), 0);
        std::vector<uint8_t> pred(size_t(nval_) * W_), truth = ds_.val.lab;
        size_t done = 0;
        for (int s = 0; s < nval_; s += 32) {
            const int B = std::min(32, nval_ - s);
            make_batch(ds_.val, &idx_val[s], B, xb, tb);
            auto& z = net_->forward(xb.data(), B, W_, false);
            gz.resize(z.size());
            loss += softmax_mcloss(z.data(), tb.data(), gz.data(), B, K_, W_) * B;
            for (int b = 0; b < B; ++b) {
                for (int t = 0; t < W_; ++t) {
                    int best = 0;
                    float m = z[(size_t(b) * K_) * W_ + t];
                    for (int k = 1; k < K_; ++k) {
                        if (z[(size_t(b) * K_ + k) * W_ + t] > m) {
                            m = z[(size_t(b) * K_ + k) * W_ + t];
                            best = k;
                        }
                    }
                    pred[done + size_t(b) * W_ + t] = uint8_t(best);
                }
            }
            done += size_t(B) * W_;
        }
        sc = score_predictions(pred, truth, W_, K_);
        const double lv = loss / nval_ + l2_ * net_->l2();
        Lv_.push_back(lv);

        std::string stop_msg = "";
        if (Lv_.size() > 3) {
            const double mean3 = (Lv_[Lv_.size() - 2] + Lv_[Lv_.size() - 3] + Lv_[Lv_.size() - 4]) / 3.0;
            if (lv < mean3) {
                worse_ = 0;
                if (lv < best_loss_) {
                    best_loss_ = lv;
                    best_snapshot_ = net_->param_snapshot();
                    best_scores_ = sc;
                    best_sc_loss_ = lv;
                } else {
                    net_->restore_snapshot(best_snapshot_);
                }
            } else {
                adam_.lr *= 0.5;
                ++worse_;
                net_->restore_snapshot(best_snapshot_);
            }
        } else if (lv <= best_loss_) {
            best_loss_ = lv;
            best_snapshot_ = net_->param_snapshot();
            best_scores_ = sc;
            best_sc_loss_ = lv;
        }

        if (worse_ > cfg_.patience - 1) {
            finished_ = true;
            stop_msg = "early stopping";
            net_->restore_snapshot(best_snapshot_);
        } else if (epoch_ >= cfg_.max_epochs) {
            finished_ = true;
            stop_msg = "max epochs reached";
            net_->restore_snapshot(best_snapshot_);
        }

        return progress_json(epoch_, cnt ? ltr / cnt : 0, lv, sc.event_f1, sc.kappa, finished_, false, stop_msg);
    }

    void cancel() {
        cancel_ = true;
    }

    bool is_finished() const { return finished_; }
    bool is_cancelled() const { return cancel_; }

    val predict_trial(val x_val, val y_val, double rate_hz, double max_speed_deg_s) {
        const size_t n = x_val["length"].as<size_t>();
        std::vector<float> x(n), y(n);
        for (size_t i = 0; i < n; ++i) {
            x[i] = x_val[i].as<float>();
            y[i] = y_val[i].as<float>();
        }

        std::vector<uint8_t> pred = predict_internal(x, y, rate_hz, max_speed_deg_s);
        val result = val::global("Uint8Array").new_(pred.size());
        for (size_t i = 0; i < pred.size(); ++i) {
            result.set(i, pred[i]);
        }
        return result;
    }

    val predict_probs(val x_val, val y_val, double rate_hz, double max_speed_deg_s) {
        const size_t n = x_val["length"].as<size_t>();
        std::vector<float> x(n), y(n);
        for (size_t i = 0; i < n; ++i) {
            x[i] = x_val[i].as<float>();
            y[i] = y_val[i].as<float>();
        }

        std::vector<float> probs = predict_probs_internal(x, y, rate_hz, max_speed_deg_s);
        val result = val::global("Float32Array").new_(probs.size());
        for (size_t i = 0; i < probs.size(); ++i) {
            result.set(i, probs[i]);
        }
        return result;
    }

    val export_bin() {
        if (!net_) return val::null();
        std::vector<uint8_t> bytes = export_bin_internal();
        val arr = val::global("Uint8Array").new_(bytes.size());
        for (size_t i = 0; i < bytes.size(); ++i) arr.set(i, bytes[i]);
        return arr;
    }

    std::string export_json() {
        std::ostringstream j;
        j << "{\"window\": " << W_ << ", \"classes\": " << K_ << ", \"ks\": 5, \"mp\": 5, \"deg_per_unit\": "
          << cfg_.deg_per_unit << ", \"rate_hz\": " << ds_.rate_hz << ", \"epochs\": " << epoch_
          << ", \"val_loss\": " << best_sc_loss_ << ", \"val_event_f1\": " << best_scores_.event_f1
          << ", \"val_kappa\": " << best_scores_.kappa << ", \"rotation_aug\": "
          << (cfg_.rotation_aug ? "true" : "false") << ", \"train_windows\": " << ntr_
          << ", \"source\": \"uneye web wasm\"}";
        return j.str();
    }

    bool load_model_from_bytes(val uint8_bytes) {
        const size_t len = uint8_bytes["length"].as<size_t>();
        std::vector<uint8_t> data(len);
        for (size_t i = 0; i < len; ++i) data[i] = uint8_bytes[i].as<uint8_t>();
        if (data.size() < 20) return false;
        int32_t hdr[4];
        std::memcpy(hdr, data.data() + 4, 16);
        K_ = hdr[1];
        net_ = std::make_unique<UNet<float>>(K_, hdr[2], hdr[3], 1);
        std::string err;
        return load_bin_from_memory(data.data(), data.size(), *net_, &err);
    }

    static std::string evaluate_scores_json(val pred_val, val truth_val, int n_classes) {
        const size_t n = pred_val["length"].as<size_t>();
        std::vector<uint8_t> p(n), t(n);
        for (size_t i = 0; i < n; ++i) {
            p[i] = pred_val[i].as<uint8_t>();
            t[i] = truth_val[i].as<uint8_t>();
        }

        ClassMetric sac = compute_class_metric(p.data(), t.data(), n, 1, 3);
        ClassMetric blk = compute_class_metric(p.data(), t.data(), n, 2, 3);

        std::ostringstream os;
        os << "{"
           << "\"testF1Saccade\": " << sac.f1 << ","
           << "\"testKappaSaccade\": " << sac.kappa << ","
           << "\"testF1Blink\": " << blk.f1 << ","
           << "\"testKappaBlink\": " << blk.kappa << ","
           << "\"nPredSaccades\": " << sac.n_pred << ","
           << "\"nTrueSaccades\": " << sac.n_true << ","
           << "\"nPredBlinks\": " << blk.n_pred << ","
           << "\"nTrueBlinks\": " << blk.n_true
           << "}";
        return os.str();
    }

private:
    std::string progress_json(int ep, double tr_loss, double v_loss, double f1, double kap, bool fin, bool canc, const std::string& msg) {
        double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0_).count();
        double eta = (ep > 0 && !fin) ? (el / ep) * (cfg_.max_epochs - ep) : 0;
        std::ostringstream os;
        os << "{"
           << "\"epoch\": " << ep << ","
           << "\"train_loss\": " << tr_loss << ","
           << "\"val_loss\": " << v_loss << ","
           << "\"val_event_f1\": " << f1 << ","
           << "\"val_kappa\": " << kap << ","
           << "\"elapsed_s\": " << el << ","
           << "\"eta_s\": " << eta << ","
           << "\"finished\": " << (fin ? "true" : "false") << ","
           << "\"cancelled\": " << (canc ? "true" : "false") << ","
           << "\"message\": \"" << msg << "\""
           << "}";
        return os.str();
    }

    std::vector<uint8_t> predict_internal(const std::vector<float>& x, const std::vector<float>& y, double rate_hz, double max_speed_deg_s) {
        const size_t n = x.size();
        if (n == 0 || !net_) return {};
        std::vector<float> probs = predict_probs_internal(x, y, rate_hz, max_speed_deg_s);
        std::vector<uint8_t> pred(n, 0);
        if (K_ == 2) {
            for (size_t t = 0; t < n; ++t) {
                pred[t] = (probs[n + t] > 0.5f) ? 1 : 0;
            }
        } else {
            for (size_t t = 0; t < n; ++t) {
                int best = 0;
                float bv = probs[t];
                for (int k = 1; k < K_; ++k) {
                    if (probs[size_t(k) * n + t] > bv) {
                        bv = probs[size_t(k) * n + t];
                        best = k;
                    }
                }
                pred[t] = uint8_t(best);
            }
        }
        return pred;
    }

    std::vector<float> predict_probs_internal(const std::vector<float>& x, const std::vector<float>& y, double rate_hz, double max_speed_deg_s) {
        const size_t n = x.size();
        if (n == 0 || !net_) return {};

        std::vector<float> dxy(2 * n, 0.f);
        const double lim = max_speed_deg_s > 0 ? max_speed_deg_s / rate_hz : 0.0;
        for (size_t t = 0; t < n; ++t) {
            float dx = 0.f, dy = 0.f;
            if (t > 0) { dx = x[t] - x[t - 1]; dy = y[t] - y[t - 1]; }
            if (!std::isfinite(dx) || !std::isfinite(dy)) dx = dy = 0.f;
            if (lim > 0) {
                const double norm = std::hypot(double(dx), double(dy));
                if (norm > lim) { dx = float(dx * lim / norm); dy = float(dy * lim / norm); }
            }
            dxy[t] = dx;
            dxy[n + t] = dy;
        }

        std::vector<float> logits(size_t(K_) * n, 0.f);
        if (n >= 25) {
            const int rem = int(n % 25);
            const int t1 = int(n - rem);
            if (rem == 0) {
                auto& z = net_->forward(dxy.data(), 1, int(n), false);
                std::copy(z.begin(), z.end(), logits.begin());
            } else {
                std::vector<float> dxy1(2 * t1);
                std::copy(dxy.begin(), dxy.begin() + t1, dxy1.begin());
                std::copy(dxy.begin() + n, dxy.begin() + n + t1, dxy1.begin() + t1);
                auto& z1 = net_->forward(dxy1.data(), 1, t1, false);
                for (int k = 0; k < K_; ++k) {
                    std::copy(z1.begin() + size_t(k) * t1, z1.begin() + size_t(k + 1) * t1, logits.begin() + size_t(k) * n);
                }

                std::vector<float> dxy2(2 * 25);
                std::copy(dxy.begin() + (n - 25), dxy.begin() + n, dxy2.begin());
                std::copy(dxy.begin() + (2 * n - 25), dxy.end(), dxy2.begin() + 25);
                auto& z2 = net_->forward(dxy2.data(), 1, 25, false);
                for (int k = 0; k < K_; ++k) {
                    std::copy(z2.begin() + size_t(k) * 25 + (25 - rem), z2.begin() + size_t(k + 1) * 25, logits.begin() + size_t(k) * n + t1);
                }
            }
        } else {
            std::vector<float> dxy_pad(2 * 25, 0.f);
            std::copy(dxy.begin(), dxy.begin() + n, dxy_pad.begin());
            std::copy(dxy.begin() + n, dxy.end(), dxy_pad.begin() + 25);
            auto& z = net_->forward(dxy_pad.data(), 1, 25, false);
            for (int k = 0; k < K_; ++k) {
                std::copy(z.begin() + size_t(k) * 25, z.begin() + size_t(k) * 25 + n, logits.begin() + size_t(k) * n);
            }
        }

        // Softmax
        std::vector<float> probs(size_t(K_) * n);
        for (size_t t = 0; t < n; ++t) {
            float max_l = logits[t];
            for (int k = 1; k < K_; ++k) {
                if (logits[size_t(k) * n + t] > max_l) max_l = logits[size_t(k) * n + t];
            }
            float sum = 0.f;
            for (int k = 0; k < K_; ++k) {
                float e = std::exp(logits[size_t(k) * n + t] - max_l);
                probs[size_t(k) * n + t] = e;
                sum += e;
            }
            if (sum > 0) {
                for (int k = 0; k < K_; ++k) probs[size_t(k) * n + t] /= sum;
            }
        }
        return probs;
    }

    std::vector<uint8_t> export_bin_internal() {
        std::vector<uint8_t> out;
        out.resize(20);
        std::memcpy(out.data(), "UNEY", 4);
        int32_t hdr[4] = {1, net_->K, net_->ks, net_->mp};
        std::memcpy(out.data() + 4, hdr, 16);
        for (const auto& t : net_->state()) {
            const size_t bsize = t.size() * sizeof(float);
            const size_t prev = out.size();
            out.resize(prev + bsize);
            std::memcpy(out.data() + prev, t.data(), bsize);
        }
        return out;
    }

    bool load_bin_from_memory(const uint8_t* data, size_t size, UNet<float>& net, std::string* error) {
        if (size < 20) { if (error) *error = "file too short"; return false; }
        if (std::memcmp(data, "UNEY", 4) != 0) { if (error) *error = "not a U'n'Eye model file"; return false; }
        int32_t hdr[4];
        std::memcpy(hdr, data + 4, 16);
        const int classes = hdr[1];
        if (hdr[2] != net.ks || hdr[3] != net.mp) { if (error) *error = "kernel / pool mismatch"; return false; }

        UNet<float> src(classes, net.ks, net.mp);
        auto layout = src.state();
        const bool keep_head = classes != net.K;
        size_t offset = 20;
        for (auto& t : layout) {
            const size_t bsize = t.size() * sizeof(float);
            if (offset + bsize > size) { if (error) *error = "unexpected end of file"; return false; }
            std::memcpy(t.data(), data + offset, bsize);
            offset += bsize;
        }
        std::vector<std::vector<float>> mine = layout;
        if (keep_head) {
            auto own = net.state();
            mine[mine.size() - 2] = own[own.size() - 2];
            mine[mine.size() - 1] = own[own.size() - 1];
        }
        return net.load_state(mine);
    }

    std::vector<TrainTrace> traces_;
    TrainConfig cfg_;
    Dataset ds_;
    std::unique_ptr<UNet<float>> net_;
    int K_ = 3, W_ = 1000;
    int ntr_ = 0, nval_ = 0, iters_ = 0, batch_ = 0;
    std::mt19937_64 rng_;
    std::vector<int> order_;
    AdamOpt adam_;
    double l2_ = 0.001;
    int epoch_ = 0, worse_ = 0;
    double best_loss_ = 1e30, best_sc_loss_ = 0;
    std::vector<double> Lv_;
    std::vector<float> best_snapshot_;
    Scores best_scores_;
    bool finished_ = false, cancel_ = false;
    std::chrono::steady_clock::time_point t0_;
    std::vector<uint8_t> init_model_bytes_;
};

EMSCRIPTEN_BINDINGS(uneye_web_module) {
    class_<WebTrainer>("WebTrainer")
        .constructor<>()
        .function("reset", &WebTrainer::reset)
        .function("setConfig", &WebTrainer::set_config)
        .function("addTrace", &WebTrainer::add_trace)
        .function("setInitModelBytes", &WebTrainer::set_init_model_bytes)
        .function("initTraining", &WebTrainer::init_training)
        .function("step", &WebTrainer::step)
        .function("cancel", &WebTrainer::cancel)
        .function("isFinished", &WebTrainer::is_finished)
        .function("isCancelled", &WebTrainer::is_cancelled)
        .function("predictTrial", &WebTrainer::predict_trial)
        .function("predictProbs", &WebTrainer::predict_probs)
        .function("exportBin", &WebTrainer::export_bin)
        .function("exportJson", &WebTrainer::export_json)
        .function("loadModelFromBytes", &WebTrainer::load_model_from_bytes)
        .class_function("evaluateScoresJson", &WebTrainer::evaluate_scores_json);
}
