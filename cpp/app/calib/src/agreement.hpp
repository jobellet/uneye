#pragma once
#include "eyelive/annotation.hpp"

namespace eyelive {
AgreementStats agreement(const std::vector<Event>& detected, const std::vector<Event>& expert, int64_t n_samples, uint8_t cls);
}
