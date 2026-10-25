// Network trainer interface (CONTRACT). Implemented by task T09 in C++ without PyTorch: the original U'n'Eye topology (cpp/src/native_engine.cpp + uneye/functions.py::UNet) with K output classes.
// The GUI (Train tab, T16) and the engine glue only use this interface; a mock trainer is provided in gui/ for development.
#pragma once
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace eyelive {

struct TrainTrace {                    // one reviewed stretch of the recording
    std::vector<float> x, y;           // positions in the units the network will see at inference (the engine has already multiplied by the scale)
    std::vector<uint8_t> labels;       // per-sample class 0..K-1, kIgnoreLabel (255) = not used
    double rate_hz = 1000.0;
};

struct TrainConfig {
    int n_classes = 3;                 // 2..5; default 3 = fixation, saccade, blink
    double max_seconds_of_data = 50.0; // default 50 s; the GUI may raise it (the trainer uses at most this much reviewed data, most recent first)
    bool rotation_aug = true;          // random rotations of the (x, y) trace, ON by default; false disables it
    int rotations = 4;                 // number of rotated copies per trial (as uneye.DNN: angles 0.25, 0.75, 1.25, 1.75 x pi)
    bool noise_aug = false;            // added white noise (std 0..0.005 deg), off by default
    bool gain_aug = false;             // random gain 0.7..1.4, off by default
    double val_fraction = 0.2;         // held out for early stopping and the displayed validation score
    double max_speed_deg_s = 1300.0;   // limit of the speed of the network input (deg/s, direction kept; 0 = off), as uneye/functions.py::clip_speed and the detector's Config::max_speed_deg_s: blinks and tracker losses would otherwise reach the network at up to a thousand times the size of a saccade. Use the same value at inference
    int max_epochs = 500;              // upper bound; early stopping normally ends before
    int patience = 4;                  // epochs without improvement of the validation loss before stopping (the original uses 4 with learning-rate decay)
    double learning_rate = 1e-3;
    int n_threads = 0;                 // 0 = automatic (hardware threads - 1, at least 1)
    uint64_t seed = 1;
    std::string init_model;            // path of a model to start from (transfer learning); empty = random initialisation
    double deg_per_unit = 1.0;         // stored in the saved model (the scale the training data was brought to)
    // Recipe of the PyTorch cross-validation (MS_CD_repo exploratory_analysis/mocol_uneye_crossval.py), off by default (the original recipe):
    // validation = whole traces held out (val_fraction of the traces), so that early stopping sees data the network was not trained on; the training
    // windows are cut again at every epoch around the labelled samples with a random offset, and each window gets `rotations` extra copies rotated by a
    // random angle and mirrored with probability 0.5; minibatches of batch_size windows.
    bool resample_each_epoch = false;
    int batch_size = 0;                // > 0: minibatches of this many windows (all windows every epoch); 0 = the original 10 minibatches per epoch
};

struct TrainProgress {
    int epoch = 0; double train_loss = 0, val_loss = 0, val_event_f1 = 0, val_kappa = 0, elapsed_s = 0, eta_s = 0;
    bool finished = false; bool cancelled = false; bool failed = false; std::string message;
};
using ProgressFn = std::function<void(const TrainProgress&)>;      // called from the worker thread, at least once per epoch; must be cheap

class ITrainer {
 public:
    virtual ~ITrainer() = default;
    virtual void start(std::vector<TrainTrace> traces, const TrainConfig& cfg, ProgressFn progress) = 0;   // returns immediately; training runs on worker threads
    virtual void cancel() = 0;                                      // cooperative, returns at once; the final callback has cancelled = true
    virtual bool running() const = 0;
    virtual bool save_model(const std::string& path, std::string* error) = 0;   // model file readable by the detector adapter (T06): weights in the cpp/models/*.bin layout + a metadata sidecar (classes, scale, training summary)
};

std::unique_ptr<ITrainer> make_trainer();                           // defined by T09

}  // namespace eyelive
