// Training data preparation for the trainer (T09b): windows of ~1 s, held-out validation windows, rotation / noise / gain augmentation, velocity input (dX, dY per sample).
#pragma once
#include <cstdint>
#include <vector>
#include "eyelive/trainer.hpp"
#include "eyelive/types.hpp"

namespace eyelive { namespace train {

struct Windows {                         // n windows of W samples
    int W = 0, n = 0;
    std::vector<float> vel;              // [n][2][W] velocity (dX, dY) in the units of the positions per sample
    std::vector<uint8_t> lab;            // [n][W], kIgnoreLabel = not used
    std::vector<int> orient;             // [n] index of the rotation applied (0 = original), for tests and statistics
};
struct Dataset { Windows train, val; int n_classes = 2; double rate_hz = 1000.0; std::string error; };

int window_for_rate(double rate_hz);     // about 1 s, a multiple of 25 (mp^2)
Dataset make_dataset(const std::vector<TrainTrace>& traces, const TrainConfig& cfg);   // error non-empty on failure

struct Scores { double event_f1 = 0, kappa = 0; };
Scores score_predictions(const std::vector<uint8_t>& pred, const std::vector<uint8_t>& truth, int W, int n_classes);   // event F1 of class 1 (saccade) and multi-class kappa on the labelled samples

}}  // namespace eyelive::train
