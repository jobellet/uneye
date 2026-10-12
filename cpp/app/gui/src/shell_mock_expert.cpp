#include "eyelive/annotation.hpp"

#ifndef EYELIVE_HAVE_ANNOT

namespace eyelive {

// Dummy implementations for ExpertLabels if 'annot' module is not linked
ExpertLabels::ExpertLabels(int n_classes) : impl_(nullptr) {}

int ExpertLabels::n_classes() const { return 3; }
void ExpertLabels::set_n_classes(int n) {}

const std::vector<Event>& ExpertLabels::events() const {
    static std::vector<Event> empty_events;
    return empty_events;
}

const std::vector<Event>& ExpertLabels::prelabels() const {
    static std::vector<Event> empty_prelabels;
    return empty_prelabels;
}

void ExpertLabels::load_prelabels(const std::vector<Event>& events, int64_t from, int64_t to) {}
bool ExpertLabels::add(int64_t onset, int64_t offset, uint8_t cls) { return false; }
bool ExpertLabels::remove(size_t index) { return false; }
bool ExpertLabels::move_onset(size_t index, int64_t new_onset) { return false; }
bool ExpertLabels::move_offset(size_t index, int64_t new_offset) { return false; }
bool ExpertLabels::set_class(size_t index, uint8_t cls) { return false; }
bool ExpertLabels::split(size_t index, int64_t at) { return false; }
bool ExpertLabels::merge(size_t first_index, size_t second_index) { return false; }
bool ExpertLabels::undo() { return false; }
bool ExpertLabels::redo() { return false; }
void ExpertLabels::clear_history() {}

void ExpertLabels::exclude_range(int64_t from, int64_t to, bool excluded) {}
const std::vector<std::pair<int64_t, int64_t>>& ExpertLabels::excluded() const {
    static std::vector<std::pair<int64_t, int64_t>> empty_excluded;
    return empty_excluded;
}

void ExpertLabels::mark_reviewed(int64_t from, int64_t to, bool reviewed) {}
double ExpertLabels::reviewed_seconds(double rate_hz) const { return 0.0; }

void ExpertLabels::render(int64_t from, int64_t to, bool only_reviewed, std::vector<uint8_t>& out) const {}
bool ExpertLabels::save(const std::string& path, std::string* error) const { return false; }
bool ExpertLabels::load(const std::string& path, std::string* error) { return false; }

} // namespace eyelive

#endif // EYELIVE_HAVE_ANNOT
