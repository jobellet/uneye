// OfflineSession: the single C++ facade of the Android offline-labeling app
// (plan phase 1). Owns one loaded recording (resampled to 1 kHz, scaled to
// degrees), its ExpertLabels, the pre-labeler and the trainer, and exposes
// everything the Kotlin UI needs through plain C functions (see jni/).
//
// Reuses the frozen contracts of cpp/app/include/eyelive/ and the engine glue
// of cpp/app/engine/ (blink events, reviewed-range training traces). No
// contract is modified; this header is new code in android/.
#pragma once
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include "eyelive/annotation.hpp"
#include "eyelive/trainer.hpp"
#include "eyelive/types.hpp"

namespace offline {

// One labelled movement, in samples of the loaded (1 kHz) trace, plus the
// pre-label it came from (provenance) and a confidence for the review queue.
struct Mark {
    int64_t onset = 0, offset = 0;     // inclusive sample indices
    uint8_t cls = 1;                   // 1 saccade, 2 blink
    bool from_prelabel = false;        // true: snapped from a pre-label (provenance kept in ExpertLabels)
    float confidence = 1.0f;
};

// One point mark of pass 1 (never used for training, never in ExpertLabels).
struct PointMark {
    int64_t at = 0;                    // sample index
    uint8_t cls = 1;
};

enum class Prelabeler { None = 0, HMM = 1, ShippedModel = 2, LastTrained = 3 };

struct LoadOptions {
    double rate_hz = 0;                // declared sampling rate of the file; <= 0 = 1000
    double deg_per_unit = 1.0;         // units -> degrees; <= 0 = ask select_scale later
    bool estimate_scale = false;       // run select_scale against the HMM reference
    Prelabeler prelabeler = Prelabeler::HMM;
    std::string model_bin;             // ShippedModel / LastTrained: path of the .bin (+ .json sidecar)
};

struct LoadResult {
    bool ok = false;
    std::string error;
    int64_t n_samples = 0;             // after resampling to 1 kHz
    double rate_hz = 1000.0;
    double deg_per_unit = 1.0;         // the scale actually applied
    bool scale_estimated = false;
    bool hmm_rate_warning = false;     // the file rate is not 500/1000 Hz and was resampled
    std::string prelabeler_error;      // non-empty when the chosen pre-labeler failed (events = none)
    std::vector<Mark> prelabels;
};

struct TrainRequest {
    double max_seconds = 50.0;         // reviewed seconds used, most recent first
    int n_threads = 2;
    int max_epochs = 500;
    int patience = 4;
    bool rotation_aug = true;
    std::string init_model;            // empty = random init; default = last trained model
    bool use_last_model_as_init = true;
};

struct TrainEvent {
    int epoch = 0;
    double train_loss = 0, val_loss = 0, val_event_f1 = 0, val_kappa = 0, elapsed_s = 0;
    bool finished = false, cancelled = false, failed = false;
    std::string message;
};

// Scores of the current model on the fixed convergence test set
// (phase 4): event F1 + kappa per class, against the expert events.
struct TestScores {
    bool ok = false;
    std::string error;
    int64_t n_samples = 0;
    double event_f1[eyelive::kMaxClasses] = {0, 0, 0, 0, 0};
    double kappa[eyelive::kMaxClasses] = {0, 0, 0, 0, 0};
    int n_events[eyelive::kMaxClasses] = {0, 0, 0, 0, 0};   // expert events in the test set
};

class OfflineSession {
 public:
    OfflineSession();
    ~OfflineSession();
    OfflineSession(const OfflineSession&) = delete;
    OfflineSession& operator=(const OfflineSession&) = delete;

    // --- lifecycle ---------------------------------------------------------
    // X, Y: trials x time or a single column (cut into 1 s trials at rate_hz).
    // The trace is cut into 1 s trials, resampled to 1 kHz (linear), scaled to
    // degrees, and pre-labeled in one call.
    LoadResult load(const std::vector<std::vector<double>>& X,
                    const std::vector<std::vector<double>>& Y,
                    const LoadOptions& opt);
    bool is_loaded() const;
    int64_t n_samples() const;                 // total samples (all trials, 1 kHz)
    int64_t n_trials() const;
    double rate_hz() const;                    // always 1000 after load
    double deg_per_unit() const;
    bool save_labels(const std::string& path, std::string* error) const;   // ExpertLabels::save (exact round trip)
    bool load_labels(const std::string& path, std::string* error);

    // --- trace access (for drawing; downsample for display in the UI) ------
    // Trial k, samples [from, to]: positions in degrees. Thread-safe.
    void trace(int64_t trial, int64_t from, int64_t to, std::vector<float>* x, std::vector<float>* y) const;
    // Rendered per-sample labels of trial k (0 background, 1.. class,
    // 255 = ignored: outside reviewed ranges when only_reviewed).
    void labels(int64_t trial, int64_t from, int64_t to, bool only_reviewed, std::vector<uint8_t>* out) const;

    // --- pass 1: point marks (kept here, never trained on) -----------------
    void add_point_mark(int64_t trial, int64_t at, uint8_t cls);
    bool remove_point_mark(int64_t trial, int64_t at);       // nearest mark of any class within 5 samples
    void clear_point_marks(int64_t trial);
    std::vector<PointMark> point_marks(int64_t trial) const;
    // Snap: nearest pre-label onset within `tolerance` samples -> returns its index or -1.
    int64_t snap_to_prelabel(int64_t trial, int64_t at, int64_t tolerance) const;

    // --- pass 2: refinement (the only path into ExpertLabels) --------------
    // Refine the mark at `at` (or the snapped pre-label): creates / replaces
    // the expert event [onset, offset]. Returns false on invalid input.
    bool refine_mark(int64_t trial, int64_t at, int64_t onset, int64_t offset, uint8_t cls);
    bool delete_mark(int64_t trial, int64_t at);             // remove the expert event covering `at`
    // The epoch [from, to] of trial k was scanned and all its marks refined:
    void mark_epoch_reviewed(int64_t trial, int64_t from, int64_t to);

    // --- ExpertLabels passthrough (undo/redo, edits, reviewed/excluded) ----
    eyelive::ExpertLabels& expert();
    const eyelive::ExpertLabels& expert() const;
    std::vector<Mark> events(int64_t trial) const;           // expert events of one trial
    std::vector<Mark> prelabels(int64_t trial) const;
    double reviewed_seconds() const;

    // --- training (phase 3) -------------------------------------------------
    // Starts the trainer on the reviewed ranges (render(..., only_reviewed=true),
    // stretches >= 0.25 s, positions scaled to degrees). Runs on worker threads;
    // `progress` is called from the worker thread (post to the UI thread in JNI).
    bool start_training(const TrainRequest& req, std::function<void(const TrainEvent&)> progress, std::string* error);
    void cancel_training();
    bool training_running() const;
    // Saves the trained model (.bin + .json sidecar, read by detect_net) and
    // makes it the pre-labeler for the next loads.
    bool save_trained_model(const std::string& path, std::string* error);
    std::string last_trained_model() const;

    // --- convergence test set (phase 4) --------------------------------------
    // Chooses the test trials before pass 1 (fixed between rounds): every
    // `every_nth` trial, at least `min_trials`, never trained on.
    void choose_test_set(int64_t every_nth, int64_t min_trials);
    std::vector<int64_t> test_trials() const;
    // Scores the current model on the test set with eyelive::agreement().
    TestScores score_test_set(const std::string& model_bin, std::string* error);

    // --- pre-labeling ---------------------------------------------------------
    // Re-runs the chosen pre-labeler on one trial (after the model changed).
    bool prelabel_trial(int64_t trial, std::string* error);

 private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace offline
