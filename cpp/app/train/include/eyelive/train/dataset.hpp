// Training data preparation for the trainer (T09b): windows of ~1 s, held-out validation windows, rotation / noise / gain augmentation, velocity input (dX, dY per sample).
#pragma once
#include <cstdint>
#include <random>
#include <vector>
#include "eyelive/trainer.hpp"
#include "eyelive/types.hpp"

namespace eyelive { namespace train {

// Robust per-axis noise scale of the valid samples: median absolute first
// difference / 0.6745 (MAD estimator, immune to the large saccade steps).
double noise_std(const std::vector<float>& v);

// The velocity input maps any non-finite difference to 0, and a constant or
// interpolated segment reads as a zero-velocity block: a blink labelled 2
// with such positions collapses 3-class training (validation event F1 stays
// 0.000 whether the blink is labelled 2 or ignored). A real tracker returns
// noise during a blink: filling the invalid stretches with Gaussian noise at
// 3x the local fixation noise makes the blink distinguishable from fixation
// again. The label of the filled samples is kept (blink = class 2); callers
// that display or run detection keep their own raw copy of the trace.
void fill_invalid(std::vector<float>& v, double std_, std::mt19937_64& rng);

struct Windows {                         // n windows of W samples
    int W = 0, n = 0;
    std::vector<float> vel;              // [n][2][W] velocity (dX, dY) in the units of the positions per sample
    std::vector<uint8_t> lab;            // [n][W], kIgnoreLabel = not used
    std::vector<int> orient;             // [n] index of the rotation applied (0 = original), for tests and statistics
};
struct Dataset {
    Windows train, val; int n_classes = 2; double rate_hz = 1000.0; std::string error;
    // resample_each_epoch: the training traces (invalid stretches filled, labels >= K ignored), from which resample_train cuts new windows every epoch
    std::vector<std::vector<float>> tx, ty; std::vector<std::vector<uint8_t>> tl; double lim = 0.0;
};

int window_for_rate(double rate_hz);     // about 1 s, a multiple of 25 (mp^2)
Dataset make_dataset(const std::vector<TrainTrace>& traces, const TrainConfig& cfg);   // error non-empty on failure
// cfg.resample_each_epoch: new training windows from d.tx / d.ty / d.tl (random offset around the labelled samples, random rotation + mirror copies)
void resample_train(Dataset& d, const TrainConfig& cfg, std::mt19937_64& rng);

struct Scores { double event_f1 = 0, kappa = 0; };
Scores score_predictions(const std::vector<uint8_t>& pred, const std::vector<uint8_t>& truth, int W, int n_classes);   // event F1 of class 1 (saccade) and multi-class kappa on the labelled samples

}}  // namespace eyelive::train
