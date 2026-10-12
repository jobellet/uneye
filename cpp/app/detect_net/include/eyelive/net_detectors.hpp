// Neural detectors behind IDetector (task T06): U'n'Eye (2 or 5 classes), the causal TCN, the BiTCN and user models written by the trainer (T09). All wrap the engines of cpp/src/ (native C++, no ONNX).
// Registered ids: unet_1+2+3, unet_andersson5, tcn_causal, bitcn_v2. Models are looked for in the directory given by set_models_dir() (default: $EYELIVE_MODELS, then ./models, ../models, cpp/models, ../cpp/models, ../../cpp/models).
#pragma once
#include <functional>
#include <memory>
#include <string>
#include "eyelive/detector.hpp"
#include "uneye/engine.hpp"

namespace eyelive {

void set_models_dir(const std::string& dir);                  // process-wide default used by the registered detectors and by make_user_model_detector for relative paths
std::string models_dir();

// A model written by the trainer: <path> = the .bin file, <path without .bin>.json = metadata (classes, window, deg_per_unit). nullptr + message when the file is missing or invalid.
std::unique_ptr<IDetector> make_user_model_detector(const std::string& path, std::string* error);

using EngineFactory = std::function<std::unique_ptr<uneye::Engine>()>;
// A window detector over any uneye::Engine (tests with a recording double, other back-ends such as ONNX): the factory is called whenever the detector needs an engine instance.
std::unique_ptr<IDetector> make_detector_with_engine(EngineFactory factory, const std::string& id, std::string* error);

// Options (set_option): "class_map" = "default3" (default: PSO -> saccade, pursuit -> fixation, blink -> blink, i.e. 0->0, 1->1, 2->1, 3->0, 4->2 for a 5-class model) | "identity" (the model's own numbering);
// "threshold" (2-class window models, default 0.5); "min_duration_ms" (default 6 for 2-class models, 3 otherwise).
}  // namespace eyelive
