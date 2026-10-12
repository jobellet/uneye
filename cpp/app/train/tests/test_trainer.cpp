// T09b: training on a synthetic 50 s dataset: validation F1 >= 0.9 with early stopping, the saved model gives the same output in the existing native engine, transfer converges faster, rotation switch, cancel(), progress callbacks. Timing is printed.
#include <chrono>
#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <thread>
#include <cmath>
#include "eyelive/train/dataset.hpp"
#include "eyelive/train/model_io.hpp"
#include "eyelive/trainer.hpp"
#include "synth.hpp"
#include "uneye/engine.hpp"
using namespace eyelive;
static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { ++fails; std::printf("FAIL line %d: ", __LINE__); std::printf(__VA_ARGS__); std::printf("\n"); } } while (0)
struct Run { std::vector<TrainProgress> log; int first_ok_epoch = -1; TrainProgress last; double wall = 0; };
static Run run_training(ITrainer& t, std::vector<TrainTrace> tr, TrainConfig cfg) {
    Run r; const auto t0 = std::chrono::steady_clock::now();
    t.start(std::move(tr), cfg, [&](const TrainProgress& p) { r.log.push_back(p); if (!p.finished && r.first_ok_epoch < 0 && p.val_event_f1 >= 0.9) r.first_ok_epoch = p.epoch; if (p.finished) r.last = p; });
    while (t.running()) std::this_thread::sleep_for(std::chrono::milliseconds(20));
    r.wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); return r;
}
int main(int, char** argv) {
    (void)argv; const std::string dir = (std::filesystem::temp_directory_path() / "eyelive_train_test").string(); std::filesystem::create_directories(dir);
    TrainConfig cfg; cfg.n_classes = 2; cfg.max_epochs = 150; cfg.n_threads = 4; cfg.seed = 1;
    auto data = [&](uint64_t seed, double rot) { return std::vector<TrainTrace>{synth_trace(50.0, 1000.0, seed, false, rot)}; };

    // 1. train on 50 s, 2 classes
    auto trainer = make_trainer(); Run a = run_training(*trainer, data(1, 0.0), cfg);
    std::printf("run 1: %zu epochs, %.1f s wall (%.2f s/epoch, 4 threads), val F1 %.3f kappa %.3f, message '%s'\n", a.log.size() - 1, a.wall, a.wall / std::max<size_t>(1, a.log.size() - 1), a.last.val_event_f1, a.last.val_kappa, a.last.message.c_str());
    CHECK(!a.last.failed, "training failed: %s", a.last.message.c_str());
    CHECK(a.last.val_event_f1 >= 0.9, "validation event F1 %.3f < 0.9", a.last.val_event_f1);
    CHECK(a.last.message == "early stopping", "did not stop by early stopping: '%s'", a.last.message.c_str());
    { int per_epoch = 0; double prev = -1; bool mono = true; for (auto& p : a.log) { if (!p.finished) ++per_epoch; if (p.elapsed_s < prev) mono = false; prev = p.elapsed_s; }
      CHECK(mono, "elapsed_s not monotonic"); CHECK(per_epoch == a.last.epoch, "progress callbacks (%d) != epochs (%d)", per_epoch, a.last.epoch); }

    // 2. save, then load with the EXISTING native engine; compare with the trainer's own forward pass
    const std::string path = dir + "/trainer_test_model.bin"; std::string err;
    CHECK(trainer->save_model(path, &err), "save: %s", err.c_str());
    {
        const int W = 1000; auto eng = uneye::make_native_engine(path, W); CHECK(eng->classes() == 2, "classes %d", eng->classes());
        train::UNet<float> net(2, 5, 5); std::string e2; CHECK(train::load_bin(path, net, &e2), "load_bin: %s", e2.c_str());
        auto tr = synth_trace(3.0, 1000.0, 99); std::vector<float> dxy(2 * W), prob(2 * W), x(2 * W);
        for (int t = 0; t < W; ++t) { const float vx = t ? tr.x[500 + t] - tr.x[499 + t] : 0.f, vy = t ? tr.y[500 + t] - tr.y[499 + t] : 0.f; dxy[2 * t] = vx; dxy[2 * t + 1] = vy; x[t] = vx; x[W + t] = vy; }
        eng->infer(dxy.data(), prob.data()); auto& z = net.forward(x.data(), 1, W, false); double worst = 0;
        for (int t = 0; t < W; ++t) { const float m = std::max(z[t], z[W + t]); const float e0 = std::exp(z[t] - m), e1 = std::exp(z[W + t] - m); worst = std::max({worst, double(std::fabs(e0 / (e0 + e1) - prob[t])), double(std::fabs(e1 / (e0 + e1) - prob[W + t]))}); }
        std::printf("native engine vs trainer forward: max |difference| %.2e\n", worst); CHECK(worst < 1e-5, "native engine output differs by %.2e", worst);
        std::ifstream js(path.substr(0, path.size() - 4) + ".json"); CHECK(bool(js), "sidecar json missing");
    }

    // 3. transfer: a model trained on one orientation, retrained on a rotated copy of the data, against random initialisation
    {
        TrainConfig c2 = cfg; c2.seed = 2; c2.max_epochs = 60; c2.rotation_aug = false;                          // no rotation augmentation: the pretrained model has not seen the new orientation
        auto t_rand = make_trainer(); Run r = run_training(*t_rand, data(7, 0.9), c2);
        c2.init_model = path; auto t_init = make_trainer(); Run x = run_training(*t_init, data(7, 0.9), c2);
        std::printf("transfer: epochs to val F1 >= 0.9: from the pretrained model %d, random initialisation %d (final F1 %.3f vs %.3f)\n", x.first_ok_epoch, r.first_ok_epoch, x.last.val_event_f1, r.last.val_event_f1);
        CHECK(x.first_ok_epoch > 0 && (r.first_ok_epoch < 0 || x.first_ok_epoch < r.first_ok_epoch), "transfer did not converge faster");
    }

    // 4. rotation_aug switch: count distinct orientations in the prepared training windows
    {
        TrainConfig c3 = cfg; c3.rotation_aug = true; auto d1 = train::make_dataset(data(3, 0.0), c3); c3.rotation_aug = false; auto d0 = train::make_dataset(data(3, 0.0), c3);
        std::vector<int> o1 = d1.train.orient, o0 = d0.train.orient; std::sort(o1.begin(), o1.end()); o1.erase(std::unique(o1.begin(), o1.end()), o1.end()); std::sort(o0.begin(), o0.end()); o0.erase(std::unique(o0.begin(), o0.end()), o0.end());
        std::printf("distinct orientations in the training windows: with rotation_aug %zu (%d windows), without %zu (%d windows)\n", o1.size(), d1.train.n, o0.size(), d0.train.n);
        CHECK(o1.size() == 5 && o0.size() == 1, "orientation counts %zu / %zu", o1.size(), o0.size()); CHECK(d1.train.n == 5 * d0.train.n, "window counts %d vs %d", d1.train.n, d0.train.n);
    }

    // 5. cancel() returns within 1 s
    {
        TrainConfig c4 = cfg; c4.max_epochs = 100000; c4.patience = 100000; auto tc = make_trainer(); bool cancelled = false;
        tc->start(data(5, 0.0), c4, [&](const TrainProgress& p) { if (p.cancelled) cancelled = true; });
        std::this_thread::sleep_for(std::chrono::milliseconds(1500)); const auto t0 = std::chrono::steady_clock::now(); tc->cancel(); while (tc->running()) std::this_thread::sleep_for(std::chrono::milliseconds(5));
        const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); std::printf("cancel() to worker stopped: %.3f s\n", dt);
        CHECK(dt < 1.0, "cancel took %.2f s", dt); CHECK(cancelled, "final callback without cancelled = true");
    }

    // 6. three classes (fixation, saccade, blink) trains without error
    {
        TrainConfig c5 = cfg; c5.n_classes = 3; c5.max_epochs = 3; auto t3 = make_trainer(); Run r = run_training(*t3, std::vector<TrainTrace>{synth_trace(50.0, 1000.0, 4, true)}, c5);
        std::printf("3 classes: %d epochs, val loss %.4f kappa %.3f\n", r.last.epoch, r.last.val_loss, r.last.val_kappa); CHECK(!r.last.failed && r.last.epoch == 3, "3-class run failed or wrong epoch count: %s", r.last.message.c_str());
        std::string e; CHECK(t3->save_model(dir + "/trainer_test_model3.bin", &e), "save 3 classes: %s", e.c_str());
    }
    std::printf(fails ? "FAILED\n" : "all trainer tests passed\n"); return fails ? 1 : 0;
}
