// Test hooks of the engine module: an engine whose trainer is supplied by the caller (a test double implementing ITrainer).
#pragma once
#include <memory>
#include "eyelive/engine.hpp"
namespace eyelive {
std::unique_ptr<IEngine> make_engine_with_trainer(std::unique_ptr<ITrainer> trainer);
}
