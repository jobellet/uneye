// Phase 1 test for the OfflineSession facade: load (cut + resample + scale) ->
// HMM pre-label -> pass-1 point marks -> pass-2 refinement -> reviewed ranges
// -> training traces -> train a few epochs -> save model -> score the test set.
// Synthetic data (no dataset files needed), same generator style as
// cpp/app/train/tests/synth.hpp.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <thread>
#include "offline_session/offline_session.hpp"

static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { ++fails; std::printf("FAIL line %d: ", __LINE__); std::printf(__VA_ARGS__); std::printf("\n"); } } while (0)

// one 1 s trial @ 1 kHz with two saccades and one blink, positions in degrees
static void synth_trial(std::vector<double>& x, std::vector<double>& y, std::vector<double>& lab, uint64_t seed) {
    std::mt19937_64 rng(seed); std::normal_distribution<double> nd(0, 1); std::uniform_real_distribution<double> ud(0, 1);
    const size_t n = 1000; x.assign(n, 0); y.assign(n, 0); lab.assign(n, 0);
    double px = 0, py = 0; size_t i = 0;
    auto fixation = [&](size_t len) { for (size_t k = 0; k < len && i < n; ++k, ++i) { x[i] = px + 0.01 * nd(rng); y[i] = py + 0.01 * nd(rng); } };
    auto saccade = [&](double amp) {
        const double th = 6.2831853 * ud(rng); const size_t len = size_t((21 + 2.2 * amp));
        const double x0 = px, y0 = py;
        for (size_t k = 0; k < len && i < n; ++k, ++i) {
            const double u = double(k + 1) / len, s = u * u * u * (10 - 15 * u + 6 * u * u);
            x[i] = x0 + amp * std::cos(th) * s + 0.01 * nd(rng); y[i] = y0 + amp * std::sin(th) * s + 0.01 * nd(rng); lab[i] = 1;
        }
        px += amp * std::cos(th); py += amp * std::sin(th);
    };
    fixation(300); saccade(4.0); fixation(150); saccade(7.0); fixation(150);
    // blink: 200 ms of NaN
    for (size_t k = 0; k < 200 && i < n; ++k, ++i) { x[i] = std::nan(""); y[i] = std::nan(""); lab[i] = 2; }
    fixation(n - i);
}

int main() {
    using namespace offline;
    OfflineSession s;
    CHECK(!s.is_loaded(), "not loaded before load()");

    // 8 trials of 1 s, one row (single long column of 8000 samples @ 1 kHz)
    std::vector<double> X, Y, L;
    for (uint64_t k = 0; k < 8; ++k) {
        std::vector<double> x, y, l; synth_trial(x, y, l, 100 + k);
        X.insert(X.end(), x.begin(), x.end()); Y.insert(Y.end(), y.begin(), y.end()); L.insert(L.end(), l.begin(), l.end());
    }
    LoadOptions opt; opt.rate_hz = 1000.0; opt.deg_per_unit = 1.0; opt.prelabeler = Prelabeler::HMM;
    auto res = s.load({X}, {Y}, opt);
    CHECK(res.ok, "load failed: %s", res.error.c_str());
    CHECK(s.n_trials() == 8, "expected 8 trials, got %lld", (long long)s.n_trials());
    CHECK(s.n_samples() == 8000, "expected 8000 samples, got %lld", (long long)s.n_samples());
    CHECK(s.rate_hz() == 1000.0, "rate");
    CHECK(!res.prelabels.empty(), "HMM should pre-label saccades");
    std::printf("pre-labels: %zu over %lld trials\n", res.prelabels.size(), (long long)s.n_trials());
    int n_sacc = 0, n_blink = 0;
    for (auto& m : res.prelabels) { if (m.cls == 1) ++n_sacc; if (m.cls == 2) ++n_blink; }
    CHECK(n_sacc >= 8, "expected >= 8 saccade pre-labels (2 per trial), got %d", n_sacc);
    CHECK(n_blink >= 8, "expected >= 8 blink pre-labels (1 per trial), got %d", n_blink);

    // trace access
    std::vector<float> x, y;
    s.trace(0, 0, 999, &x, &y);
    CHECK(x.size() == 1000 && y.size() == 1000, "trace size %zu", x.size());
    CHECK(std::isfinite(x[0]) && std::isfinite(y[0]), "trace values");

    // pass 1: point marks; snap to a pre-label; pass 2: refine -> ExpertLabels
    s.add_point_mark(0, 310, 1);
    s.add_point_mark(0, 500, 1);
    CHECK(s.point_marks(0).size() == 2, "2 point marks");
    const int64_t snapped = s.snap_to_prelabel(0, 310, 30);
    CHECK(snapped >= 0, "snap should find the pre-label near 310");
    // pre-labels are loaded into ExpertLabels as DETECTOR events (provenance);
    // pass 2 turns the mark into an EXPERT event. The contract resolves the
    // overlap in favour of the new (expert) event, so the count stays stable.
    const size_t before = s.events(0).size();
    CHECK(s.refine_mark(0, 310, snapped >= 0 ? snapped : 310, (snapped >= 0 ? snapped : 310) + 30, 1), "refine mark");
    CHECK(s.point_marks(0).size() == 1, "refinement consumes the point mark");
    CHECK(s.events(0).size() >= 1, "expert event after refinement (had %zu, now %zu)", before, s.events(0).size());
    bool has_expert = false;
    for (auto& e : s.events(0)) if (!e.from_prelabel) has_expert = true;
    CHECK(has_expert, "the refined event is an expert event");
    s.mark_epoch_reviewed(0, 0, 999);   // not reviewed: a point mark remains in the epoch
    CHECK(s.reviewed_seconds() == 0.0, "epoch with an unrefined mark must not be reviewed");
    s.refine_mark(0, 500, 500, 540, 1);
    s.mark_epoch_reviewed(0, 0, 999);
    CHECK(s.reviewed_seconds() > 0.9, "epoch reviewed after all marks refined, got %.2f s", s.reviewed_seconds());
    // undo reverts the last edit
    CHECK(s.expert().undo(), "undo");

    // mark all trials reviewed (the app does this as the expert confirms)
    for (int64_t k = 0; k < 8; ++k) {
        s.clear_point_marks(k);
        // confirm every pre-label as an expert event at its own boundaries
        for (auto& m : s.prelabels(k)) s.refine_mark(k, m.onset, m.onset, m.offset, m.cls);
        s.mark_epoch_reviewed(k, 0, 999);
    }
    // trial 0 was left with an undo above: re-confirm its second mark so the
    // epoch is fully refined and can be reviewed
    if (s.point_marks(0).size() == 1) s.refine_mark(0, 500, 500, 540, 1);
    s.mark_epoch_reviewed(0, 0, 999);
    CHECK(s.reviewed_seconds() > 7.0, "8 trials reviewed, got %.2f s", s.reviewed_seconds());
    {
        // the rendered reviewed labels must contain the confirmed saccades
        std::vector<uint8_t> lab;
        s.labels(0, 0, 999, true, &lab);
        int n_sacc = 0, n_blink = 0, n_ign = 0;
        for (auto l : lab) { if (l == 1) ++n_sacc; else if (l == 2) ++n_blink; else if (l == 255) ++n_ign; }
        CHECK(n_sacc > 0, "rendered reviewed labels contain saccades (%d)", n_sacc);
        CHECK(n_ign == 0, "a reviewed epoch renders no ignored samples (%d)", n_ign);
        std::printf("trial 0 rendered: %d saccade, %d blink, %d ignored samples\n", n_sacc, n_blink, n_ign);
    }

    // convergence test set: fixed, never trained on
    s.choose_test_set(4, 2);
    auto test = s.test_trials();
    CHECK(test.size() >= 2, "test set size %zu", test.size());
    for (auto k : test) CHECK(k % 4 == 0, "test trial %lld not on the grid", (long long)k);

    // training on the reviewed data (train trials only). The synthetic saccades
    // are identical in every trial, so the model can actually learn them and
    // the loop closes with a non-zero test F1.
    TrainRequest req; req.max_seconds = 6.0; req.n_threads = 1; req.max_epochs = 15; req.use_last_model_as_init = false;
    int epochs = 0; TrainEvent last;
    std::string terr;
    CHECK(s.start_training(req, [&](const TrainEvent& e) { if (!e.finished) ++epochs; last = e; }, &terr), "start_training: %s", terr.c_str());
    while (s.training_running()) std::this_thread::sleep_for(std::chrono::milliseconds(5));
    CHECK(last.finished && !last.failed, "training finished (failed=%d msg=%s)", (int)last.failed, last.message.c_str());
    CHECK(epochs >= 5, "the model learns the repeated saccade pattern (got %d epochs)", epochs);
    std::printf("training: %d epochs, last val F1 %.3f kappa %.3f\n", epochs, last.val_event_f1, last.val_kappa);

    std::string merr;
    CHECK(s.save_trained_model("/tmp/offline_session_test_model.bin", &merr), "save model: %s", merr.c_str());
    CHECK(!s.last_trained_model().empty(), "last trained model set");

    // score the test set with the trained model (phase 4 path): the saved .bin
    // must load through detect_net and produce the confirmed saccades
    std::string serr;
    auto sc = s.score_test_set(s.last_trained_model(), &serr);
    CHECK(sc.ok, "score_test_set: %s", serr.c_str());
    CHECK(sc.n_samples > 0, "test samples");
    CHECK(sc.event_f1[1] > 0.5, "trained model detects the saccades on the held-out test set (F1 %.3f)", sc.event_f1[1]);
    CHECK(sc.event_f1[2] > 0.5, "blink detection on the test set (F1 %.3f)", sc.event_f1[2]);
    std::printf("test set: %lld samples, saccade event F1 %.3f kappa %.3f, blink F1 %.3f kappa %.3f (%d/%d events)\n",
                (long long)sc.n_samples, sc.event_f1[1], sc.kappa[1], sc.event_f1[2], sc.kappa[2], sc.n_events[1], sc.n_events[2]);

    // labels save/load round trip (exact)
    std::string lerr;
    CHECK(s.save_labels("/tmp/offline_session_test_labels.bin", &lerr), "save labels: %s", lerr.c_str());
    CHECK(s.load_labels("/tmp/offline_session_test_labels.bin", &lerr), "load labels: %s", lerr.c_str());

    // phase 3 cycle: swap the pre-labeler to the trained model, re-prelabel all
    // trials, and check that (a) expert events survive (apply_auto_labels never
    // touches them), (b) the rendered reviewed labels are unchanged, (c) the
    // new model pre-labels the unreviewed parts.
    s.set_prelabeler(Prelabeler::LastTrained, "");
    CHECK(s.prelabeler() == Prelabeler::LastTrained, "prelabeler swapped to LastTrained");
    const size_t n_expert_before = s.events(0).size();
    std::vector<uint8_t> lab_before;
    s.labels(0, 0, 999, true, &lab_before);
    std::string rerr;
    CHECK(s.re_prelabel_all(&rerr), "re_prelabel_all: %s", rerr.c_str());
    CHECK(s.events(0).size() == n_expert_before, "expert events survive the re-prelabeling (%zu -> %zu)",
          n_expert_before, s.events(0).size());
    std::vector<uint8_t> lab_after;
    s.labels(0, 0, 999, true, &lab_after);
    CHECK(lab_before == lab_after, "the reviewed rendered labels are unchanged by the re-prelabeling");
    // the new pre-labels come from the trained model: prelabels() (provenance)
    // still holds the originals, the detector events now come from the model
    bool any_expert = false;
    for (auto& e : s.events(0)) if (!e.from_prelabel) any_expert = true;
    CHECK(any_expert, "trial 0 still has expert events after the swap");

    // a second training round starting from the last model (init_model) works
    TrainRequest req2; req2.max_seconds = 6.0; req2.n_threads = 1; req2.max_epochs = 5; req2.use_last_model_as_init = true;
    int epochs2 = 0; TrainEvent last2;
    std::string terr2;
    CHECK(s.start_training(req2, [&](const TrainEvent& e) { if (!e.finished) ++epochs2; last2 = e; }, &terr2), "second training round: %s", terr2.c_str());
    while (s.training_running()) std::this_thread::sleep_for(std::chrono::milliseconds(5));
    CHECK(last2.finished && !last2.failed, "second round finished (failed=%d msg=%s)", (int)last2.failed, last2.message.c_str());
    std::string merr2;
    CHECK(s.save_trained_model("/tmp/offline_session_test_model2.bin", &merr2), "save model 2: %s", merr2.c_str());
    std::string serr2;
    auto sc2 = s.score_test_set(s.last_trained_model(), &serr2);
    CHECK(sc2.ok && sc2.event_f1[1] > 0.5, "the warm-started model still detects the saccades (F1 %.3f)", sc2.event_f1[1]);
    std::printf("phase 3 cycle: re-prelabel OK, expert events preserved, warm start F1 %.3f\n", sc2.event_f1[1]);

    // resample path: 500 Hz input -> 1 kHz internal
    OfflineSession s2;
    std::vector<double> x5, y5;
    for (int k = 0; k < 4; ++k) { std::vector<double> x, y, l; synth_trial(x, y, l, 200 + k);
        for (size_t i = 0; i < x.size(); i += 2) { x5.push_back(x[i]); y5.push_back(y[i]); } }   // decimate to 500 Hz
    LoadOptions opt2; opt2.rate_hz = 500.0; opt2.prelabeler = Prelabeler::None;
    auto res2 = s2.load({x5}, {y5}, opt2);
    CHECK(res2.ok, "load 500 Hz failed: %s", res2.error.c_str());
    CHECK(s2.n_trials() == 4, "4 trials from 500 Hz, got %lld", (long long)s2.n_trials());
    CHECK(s2.n_samples() == 4000, "resampled to 1 kHz: %lld samples", (long long)s2.n_samples());

    if (fails == 0) std::printf("ALL OFFLINE_SESSION TESTS PASSED\n");
    else std::printf("%d FAILURES\n", fails);
    return fails ? 1 : 0;
}
