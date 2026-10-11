#include "eyelive/annotation.hpp"
#include <algorithm>
#include <deque>
#include <fstream>
#include <iostream>

// Use a simple JSON parsing approach since we don't have a third-party library specified,
// or we can just write custom minimal parsing since "persistence format of your choice"
// We will write a tiny JSON writer/reader for our required format.

namespace eyelive {

// Helper to check valid range
static bool valid_idx(size_t index, const std::vector<Event>& v) {
    return index < v.size();
}

struct State {
    std::vector<Event> events;
    std::vector<Event> prelabels;
    std::vector<std::pair<int64_t, int64_t>> excluded;
    std::vector<std::pair<int64_t, int64_t>> reviewed;
};

struct ExpertLabels::Impl {
    int n_classes;
    State current;
    std::deque<State> undo_stack;
    std::deque<State> redo_stack;

    Impl(int n) : n_classes(n) {
        if (n_classes < 2) n_classes = 2;
        if (n_classes > kMaxClasses) n_classes = kMaxClasses;
    }

    void save_state() {
        undo_stack.push_back(current);
        redo_stack.clear();
        // optionally limit undo stack size
    }

    void trim_overlapping(const Event& new_event, std::vector<Event>& out) {
        std::vector<Event> temp;
        for (const auto& ev : current.events) {
            // No overlap
            if (ev.offset < new_event.onset || ev.onset > new_event.offset) {
                temp.push_back(ev);
            } else {
                // Overlap exists. The new_event wins.
                // Trimming the existing event
                if (ev.onset < new_event.onset && ev.offset > new_event.offset) {
                    // Split in two
                    Event ev1 = ev;
                    ev1.offset = new_event.onset - 1;
                    Event ev2 = ev;
                    ev2.onset = new_event.offset + 1;
                    if (ev1.onset <= ev1.offset) temp.push_back(ev1);
                    if (ev2.onset <= ev2.offset) temp.push_back(ev2);
                } else if (ev.onset < new_event.onset) {
                    // Trim end
                    Event ev1 = ev;
                    ev1.offset = new_event.onset - 1;
                    if (ev1.onset <= ev1.offset) temp.push_back(ev1);
                } else if (ev.offset > new_event.offset) {
                    // Trim start
                    Event ev1 = ev;
                    ev1.onset = new_event.offset + 1;
                    if (ev1.onset <= ev1.offset) temp.push_back(ev1);
                }
                // If completely inside, it's removed
            }
        }
        out = temp;
    }

    void merge_ranges(std::vector<std::pair<int64_t, int64_t>>& ranges) {
        if (ranges.empty()) return;
        std::sort(ranges.begin(), ranges.end(), [](const auto& a, const auto& b) {
            return a.first < b.first;
        });
        std::vector<std::pair<int64_t, int64_t>> merged;
        merged.push_back(ranges[0]);
        for (size_t i = 1; i < ranges.size(); ++i) {
            auto& last = merged.back();
            if (ranges[i].first <= last.second + 1) { // adjacent or overlapping
                last.second = std::max(last.second, ranges[i].second);
            } else {
                merged.push_back(ranges[i]);
            }
        }
        ranges = merged;
    }

    void remove_range(std::vector<std::pair<int64_t, int64_t>>& ranges, int64_t from, int64_t to) {
        std::vector<std::pair<int64_t, int64_t>> res;
        for (const auto& r : ranges) {
            if (r.second < from || r.first > to) {
                res.push_back(r);
            } else {
                if (r.first < from) res.push_back({r.first, from - 1});
                if (r.second > to) res.push_back({to + 1, r.second});
            }
        }
        ranges = res;
    }
};

ExpertLabels::ExpertLabels(int n_classes) : impl_(new Impl(n_classes)) {}

int ExpertLabels::n_classes() const { return impl_->n_classes; }

void ExpertLabels::set_n_classes(int n) {
    if (n < 2) n = 2;
    if (n > kMaxClasses) n = kMaxClasses;
    impl_->save_state();
    impl_->n_classes = n;

    std::vector<Event> valid_events;
    for (const auto& ev : impl_->current.events) {
        if (ev.cls < n) {
            valid_events.push_back(ev);
        }
    }
    impl_->current.events = valid_events;
}

const std::vector<Event>& ExpertLabels::events() const { return impl_->current.events; }

const std::vector<Event>& ExpertLabels::prelabels() const { return impl_->current.prelabels; }

void ExpertLabels::load_prelabels(const std::vector<Event>& events, int64_t from, int64_t to) {
    if (from > to) return;
    impl_->save_state();

    // update current.prelabels: remove any in [from, to] and add new ones
    std::vector<Event> new_prelabels;
    for (const auto& ev : impl_->current.prelabels) {
        if (ev.onset < from || ev.offset > to) {
            new_prelabels.push_back(ev); // Keep strictly outside
            // Note: the requirements don't explicitly say trim prelabels, but says "replaces the events inside [from, to]".
            // Let's assume complete replacement of events starting/ending in that range, or overlapping.
            // A simple implementation is to remove any prelabel that overlaps [from, to].
        }
    }
    // Correct way: "replaces the events inside [from, to]"
    std::vector<Event> trimmed_prelabels;
    for(const auto& ev: impl_->current.prelabels) {
         if (ev.offset < from || ev.onset > to) {
             trimmed_prelabels.push_back(ev);
         } else {
             if (ev.onset < from) {
                 Event ev1 = ev; ev1.offset = from - 1;
                 if (ev1.onset <= ev1.offset) trimmed_prelabels.push_back(ev1);
             }
             if (ev.offset > to) {
                 Event ev1 = ev; ev1.onset = to + 1;
                 if (ev1.onset <= ev1.offset) trimmed_prelabels.push_back(ev1);
             }
         }
    }
    for (const auto& ev : events) {
        trimmed_prelabels.push_back(ev);
    }

    // Sort prelabels
    std::sort(trimmed_prelabels.begin(), trimmed_prelabels.end(), [](const Event& a, const Event& b) {
        return a.onset < b.onset;
    });
    impl_->current.prelabels = trimmed_prelabels;

    // update current.events
    std::vector<Event> trimmed_events;
    for(const auto& ev: impl_->current.events) {
         if (ev.offset < from || ev.onset > to) {
             trimmed_events.push_back(ev);
         } else {
             if (ev.onset < from) {
                 Event ev1 = ev; ev1.offset = from - 1;
                 if (ev1.onset <= ev1.offset) trimmed_events.push_back(ev1);
             }
             if (ev.offset > to) {
                 Event ev1 = ev; ev1.onset = to + 1;
                 if (ev1.onset <= ev1.offset) trimmed_events.push_back(ev1);
             }
         }
    }
    for (const auto& ev : events) {
        trimmed_events.push_back(ev);
    }
    std::sort(trimmed_events.begin(), trimmed_events.end(), [](const Event& a, const Event& b) {
        return a.onset < b.onset;
    });
    impl_->current.events = trimmed_events;
}

bool ExpertLabels::add(int64_t onset, int64_t offset, uint8_t cls) {
    if (onset > offset || cls == 0 || cls >= impl_->n_classes || onset < 0) return false;

    impl_->save_state();

    Event new_ev;
    new_ev.onset = onset;
    new_ev.offset = offset;
    new_ev.cls = cls;
    new_ev.origin = EventOrigin::Expert;

    std::vector<Event> new_events;
    impl_->trim_overlapping(new_ev, new_events);
    new_events.push_back(new_ev);

    std::sort(new_events.begin(), new_events.end(), [](const Event& a, const Event& b) {
        return a.onset < b.onset;
    });

    impl_->current.events = new_events;
    return true;
}

bool ExpertLabels::remove(size_t index) {
    if (!valid_idx(index, impl_->current.events)) return false;

    impl_->save_state();
    impl_->current.events.erase(impl_->current.events.begin() + index);
    return true;
}

bool ExpertLabels::move_onset(size_t index, int64_t new_onset) {
    if (!valid_idx(index, impl_->current.events) || new_onset < 0) return false;
    if (new_onset > impl_->current.events[index].offset) return false;

    impl_->save_state();
    Event ev = impl_->current.events[index];
    impl_->current.events.erase(impl_->current.events.begin() + index);

    ev.onset = new_onset;
    ev.origin = EventOrigin::Expert;

    std::vector<Event> new_events;
    impl_->trim_overlapping(ev, new_events);
    new_events.push_back(ev);

    std::sort(new_events.begin(), new_events.end(), [](const Event& a, const Event& b) {
        return a.onset < b.onset;
    });
    impl_->current.events = new_events;
    return true;
}

bool ExpertLabels::move_offset(size_t index, int64_t new_offset) {
    if (!valid_idx(index, impl_->current.events) || new_offset < 0) return false;
    if (new_offset < impl_->current.events[index].onset) return false;

    impl_->save_state();
    Event ev = impl_->current.events[index];
    impl_->current.events.erase(impl_->current.events.begin() + index);

    ev.offset = new_offset;
    ev.origin = EventOrigin::Expert;

    std::vector<Event> new_events;
    impl_->trim_overlapping(ev, new_events);
    new_events.push_back(ev);

    std::sort(new_events.begin(), new_events.end(), [](const Event& a, const Event& b) {
        return a.onset < b.onset;
    });
    impl_->current.events = new_events;
    return true;
}

bool ExpertLabels::set_class(size_t index, uint8_t cls) {
    if (!valid_idx(index, impl_->current.events) || cls == 0 || cls >= impl_->n_classes) return false;

    impl_->save_state();
    impl_->current.events[index].cls = cls;
    impl_->current.events[index].origin = EventOrigin::Expert;
    return true;
}

bool ExpertLabels::split(size_t index, int64_t at) {
    if (!valid_idx(index, impl_->current.events)) return false;
    Event ev = impl_->current.events[index];
    if (at <= ev.onset || at > ev.offset) return false;

    impl_->save_state();

    Event ev1 = ev;
    ev1.offset = at - 1;
    ev1.origin = EventOrigin::Expert;

    Event ev2 = ev;
    ev2.onset = at;
    ev2.origin = EventOrigin::Expert;

    impl_->current.events.erase(impl_->current.events.begin() + index);
    impl_->current.events.insert(impl_->current.events.begin() + index, ev2);
    impl_->current.events.insert(impl_->current.events.begin() + index, ev1);

    return true;
}

bool ExpertLabels::merge(size_t first_index, size_t second_index) {
    if (!valid_idx(first_index, impl_->current.events) || !valid_idx(second_index, impl_->current.events)) return false;
    if (first_index > second_index) std::swap(first_index, second_index);
    if (first_index + 1 != second_index) return false;

    const Event& ev1 = impl_->current.events[first_index];
    const Event& ev2 = impl_->current.events[second_index];

    if (ev1.cls != ev2.cls) return false;

    impl_->save_state();

    Event merged = ev1;
    merged.offset = ev2.offset;
    merged.origin = EventOrigin::Expert;

    impl_->current.events.erase(impl_->current.events.begin() + second_index);
    impl_->current.events.erase(impl_->current.events.begin() + first_index);

    impl_->current.events.insert(impl_->current.events.begin() + first_index, merged);

    return true;
}

bool ExpertLabels::undo() {
    if (impl_->undo_stack.empty()) return false;
    impl_->redo_stack.push_back(impl_->current);
    impl_->current = impl_->undo_stack.back();
    impl_->undo_stack.pop_back();
    return true;
}

bool ExpertLabels::redo() {
    if (impl_->redo_stack.empty()) return false;
    impl_->undo_stack.push_back(impl_->current);
    impl_->current = impl_->redo_stack.back();
    impl_->redo_stack.pop_back();
    return true;
}

void ExpertLabels::clear_history() {
    impl_->undo_stack.clear();
    impl_->redo_stack.clear();
}

void ExpertLabels::exclude_range(int64_t from, int64_t to, bool excluded) {
    if (from > to) return;
    impl_->save_state();
    if (excluded) {
        impl_->current.excluded.push_back({from, to});
        impl_->merge_ranges(impl_->current.excluded);
    } else {
        impl_->remove_range(impl_->current.excluded, from, to);
    }
}

const std::vector<std::pair<int64_t, int64_t>>& ExpertLabels::excluded() const {
    return impl_->current.excluded;
}

void ExpertLabels::mark_reviewed(int64_t from, int64_t to, bool reviewed) {
    if (from > to) return;
    impl_->save_state();
    if (reviewed) {
        impl_->current.reviewed.push_back({from, to});
        impl_->merge_ranges(impl_->current.reviewed);
    } else {
        impl_->remove_range(impl_->current.reviewed, from, to);
    }
}

double ExpertLabels::reviewed_seconds(double rate_hz) const {
    if (rate_hz <= 0) return 0.0;
    double total = 0;
    for (const auto& r : impl_->current.reviewed) {
        total += (r.second - r.first + 1);
    }
    return total / rate_hz;
}

void ExpertLabels::render(int64_t from, int64_t to, bool only_reviewed, std::vector<uint8_t>& out) const {
    if (from > to) {
        out.clear();
        return;
    }
    size_t sz = to - from + 1;
    out.assign(sz, 0); // 0 = background

    // Fill classes
    for (const auto& ev : impl_->current.events) {
        if (ev.offset >= from && ev.onset <= to) {
            int64_t start = std::max(from, ev.onset) - from;
            int64_t end = std::min(to, ev.offset) - from;
            for (int64_t i = start; i <= end; ++i) {
                out[i] = ev.cls;
            }
        }
    }

    // Apply excluded
    for (const auto& ex : impl_->current.excluded) {
        if (ex.second >= from && ex.first <= to) {
            int64_t start = std::max(from, ex.first) - from;
            int64_t end = std::min(to, ex.second) - from;
            for (int64_t i = start; i <= end; ++i) {
                out[i] = kIgnoreLabel;
            }
        }
    }

    // Apply only_reviewed
    if (only_reviewed) {
        // Create mask of reviewed
        std::vector<bool> rev(sz, false);
        for (const auto& r : impl_->current.reviewed) {
            if (r.second >= from && r.first <= to) {
                int64_t start = std::max(from, r.first) - from;
                int64_t end = std::min(to, r.second) - from;
                for (int64_t i = start; i <= end; ++i) {
                    rev[i] = true;
                }
            }
        }
        for (size_t i = 0; i < sz; ++i) {
            if (!rev[i] && out[i] != kIgnoreLabel) {
                out[i] = kIgnoreLabel;
            }
        }
    }
}

// Simple JSON serialization without dependencies
static std::string escape_json(const std::string& s) {
    return "\"" + s + "\""; // Very simple, sufficient for this task
}

static void serialize_events(std::ostream& os, const std::vector<Event>& events) {
    os << "[\n";
    for (size_t i = 0; i < events.size(); ++i) {
        const auto& ev = events[i];
        os << "    {\n";
        os << "      \"onset\": " << ev.onset << ",\n";
        os << "      \"offset\": " << ev.offset << ",\n";
        os << "      \"cls\": " << (int)ev.cls << ",\n";
        os << "      \"origin\": " << (int)ev.origin << ",\n";
        os << "      \"detector_id\": " << ev.detector_id << ",\n";
        os << "      \"model_version\": " << ev.model_version << ",\n";
        os << "      \"provisional\": " << (ev.provisional ? "true" : "false") << ",\n";
        os << "      \"confidence\": " << ev.confidence << ",\n";
        os << "      \"t_onset\": " << ev.t_onset << ",\n";
        os << "      \"t_offset\": " << ev.t_offset << ",\n";
        os << "      \"amplitude_deg\": " << ev.amplitude_deg << ",\n";
        os << "      \"peak_speed_deg_s\": " << ev.peak_speed_deg_s << ",\n";
        os << "      \"direction_rad\": " << ev.direction_rad << "\n";
        os << "    }";
        if (i + 1 < events.size()) os << ",";
        os << "\n";
    }
    os << "  ]";
}

static void serialize_ranges(std::ostream& os, const std::vector<std::pair<int64_t, int64_t>>& ranges) {
    os << "[\n";
    for (size_t i = 0; i < ranges.size(); ++i) {
        os << "    [" << ranges[i].first << ", " << ranges[i].second << "]";
        if (i + 1 < ranges.size()) os << ",";
        os << "\n";
    }
    os << "  ]";
}

bool ExpertLabels::save(const std::string& path, std::string* error) const {
    std::ofstream out(path);
    if (!out) {
        if (error) *error = "Failed to open file for writing.";
        return false;
    }

    out << "{\n";
    out << "  \"version\": 1,\n";
    out << "  \"n_classes\": " << impl_->n_classes << ",\n";

    out << "  \"events\": ";
    serialize_events(out, impl_->current.events);
    out << ",\n";

    out << "  \"prelabels\": ";
    serialize_events(out, impl_->current.prelabels);
    out << ",\n";

    out << "  \"excluded\": ";
    serialize_ranges(out, impl_->current.excluded);
    out << ",\n";

    out << "  \"reviewed\": ";
    serialize_ranges(out, impl_->current.reviewed);
    out << "\n";

    out << "}\n";
    return true;
}

// A crude JSON parser for the exact format we produce.
// Since standard C++ lacks json, and we can't use 3rd party without permission...
static void skip_whitespace(const std::string& json, size_t& pos) {
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\n' || json[pos] == '\r' || json[pos] == '\t' || json[pos] == ',')) {
        pos++;
    }
}

static bool expect(const std::string& json, size_t& pos, const std::string& str) {
    skip_whitespace(json, pos);
    if (json.compare(pos, str.size(), str) == 0) {
        pos += str.size();
        return true;
    }
    return false;
}

static std::string read_string(const std::string& json, size_t& pos) {
    skip_whitespace(json, pos);
    if (pos >= json.size() || json[pos] != '"') return "";
    pos++; // skip "
    size_t end = pos;
    while (end < json.size() && json[end] != '"') end++;
    std::string res = json.substr(pos, end - pos);
    pos = end + 1;
    return res;
}

static double read_number(const std::string& json, size_t& pos) {
    skip_whitespace(json, pos);
    size_t end = pos;
    while (end < json.size() && (std::isdigit(json[end]) || json[end] == '.' || json[end] == '-' || json[end] == '+' || json[end] == 'e' || json[end] == 'E')) {
        end++;
    }
    std::string num_str = json.substr(pos, end - pos);
    pos = end;
    if (num_str.empty()) return 0;
    try {
        return std::stod(num_str);
    } catch (...) {
        return 0;
    }
}

static bool read_bool(const std::string& json, size_t& pos) {
    skip_whitespace(json, pos);
    if (json.compare(pos, 4, "true") == 0) {
        pos += 4;
        return true;
    }
    if (json.compare(pos, 5, "false") == 0) {
        pos += 5;
        return false;
    }
    return false;
}

bool ExpertLabels::load(const std::string& path, std::string* error) {
    std::ifstream in(path);
    if (!in) {
        if (error) *error = "Failed to open file for reading.";
        return false;
    }

    std::string json((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    size_t pos = 0;

    if (!expect(json, pos, "{")) {
        if (error) *error = "Invalid JSON format: missing {";
        return false;
    }

    State new_state;
    int parsed_n_classes = impl_->n_classes;

    while (pos < json.size() && json[pos] != '}') {
        std::string key = read_string(json, pos);
        if (key.empty() && json[pos] == '}') break; // handle trailing commas/spaces
        expect(json, pos, ":");

        if (key == "version") {
            read_number(json, pos);
        } else if (key == "n_classes") {
            parsed_n_classes = (int)read_number(json, pos);
        } else if (key == "events" || key == "prelabels") {
            std::vector<Event>* target = (key == "events") ? &new_state.events : &new_state.prelabels;
            expect(json, pos, "[");
            while (pos < json.size() && json[pos] != ']') {
                skip_whitespace(json, pos);
                if (json[pos] == ']') break;
                expect(json, pos, "{");
                Event ev;
                while (pos < json.size() && json[pos] != '}') {
                    std::string ev_key = read_string(json, pos);
                    if (ev_key.empty() && json[pos] == '}') break;
                    expect(json, pos, ":");

                    if (ev_key == "onset") ev.onset = (int64_t)read_number(json, pos);
                    else if (ev_key == "offset") ev.offset = (int64_t)read_number(json, pos);
                    else if (ev_key == "cls") ev.cls = (uint8_t)read_number(json, pos);
                    else if (ev_key == "origin") ev.origin = (EventOrigin)(int)read_number(json, pos);
                    else if (ev_key == "detector_id") ev.detector_id = (uint16_t)read_number(json, pos);
                    else if (ev_key == "model_version") ev.model_version = (uint32_t)read_number(json, pos);
                    else if (ev_key == "provisional") ev.provisional = read_bool(json, pos);
                    else if (ev_key == "confidence") ev.confidence = (float)read_number(json, pos);
                    else if (ev_key == "t_onset") ev.t_onset = read_number(json, pos);
                    else if (ev_key == "t_offset") ev.t_offset = read_number(json, pos);
                    else if (ev_key == "amplitude_deg") ev.amplitude_deg = (float)read_number(json, pos);
                    else if (ev_key == "peak_speed_deg_s") ev.peak_speed_deg_s = (float)read_number(json, pos);
                    else if (ev_key == "direction_rad") ev.direction_rad = (float)read_number(json, pos);
                    else {
                        // ignore unknown
                        skip_whitespace(json, pos);
                        if (json[pos] == '"') read_string(json, pos);
                        else if (std::isalpha(json[pos])) read_bool(json, pos);
                        else read_number(json, pos);
                    }
                }
                expect(json, pos, "}");
                target->push_back(ev);
            }
            expect(json, pos, "]");
        } else if (key == "excluded" || key == "reviewed") {
            std::vector<std::pair<int64_t, int64_t>>* target = (key == "excluded") ? &new_state.excluded : &new_state.reviewed;
            expect(json, pos, "[");
            while (pos < json.size() && json[pos] != ']') {
                skip_whitespace(json, pos);
                if (json[pos] == ']') break;
                expect(json, pos, "[");
                int64_t a = (int64_t)read_number(json, pos);
                expect(json, pos, ",");
                int64_t b = (int64_t)read_number(json, pos);
                expect(json, pos, "]");
                target->push_back({a, b});
            }
            expect(json, pos, "]");
        } else {
             // skip unknown
             skip_whitespace(json, pos);
             if (json[pos] == '"') read_string(json, pos);
             else if (json[pos] == '[') {
                 // nested skip not fully supported but fine
             }
             else read_number(json, pos);
        }
    }

    impl_->n_classes = parsed_n_classes;
    impl_->current = new_state;
    clear_history();
    return true;
}

// ---------------- Agreement ----------------

static std::vector<std::pair<int64_t, int64_t>> find_runs(const std::vector<bool>& b) {
    std::vector<std::pair<int64_t, int64_t>> runs;
    bool in_run = false;
    int64_t start = 0;
    for (size_t i = 0; i < b.size(); ++i) {
        if (b[i] && !in_run) {
            in_run = true;
            start = i;
        } else if (!b[i] && in_run) {
            in_run = false;
            runs.push_back({start, i - 1});
        }
    }
    if (in_run) runs.push_back({start, b.size() - 1});
    return runs;
}

AgreementStats agreement(const std::vector<Event>& detected, const std::vector<Event>& expert, int64_t n_samples, uint8_t cls) {
    AgreementStats stats;
    if (n_samples <= 0) return stats;

    std::vector<bool> P(n_samples, false);
    for (const auto& ev : detected) {
        if (ev.cls == cls && ev.onset < n_samples) {
            int64_t s = std::max((int64_t)0, ev.onset);
            int64_t e = std::min(n_samples - 1, ev.offset);
            for (int64_t i = s; i <= e; ++i) P[i] = true;
        }
    }

    std::vector<bool> L(n_samples, false);
    for (const auto& ev : expert) {
        if (ev.cls == cls && ev.onset < n_samples) {
            int64_t s = std::max((int64_t)0, ev.onset);
            int64_t e = std::min(n_samples - 1, ev.offset);
            for (int64_t i = s; i <= e; ++i) L[i] = true;
        }
    }

    int64_t tp_s = 0, fp_s = 0, fn_s = 0, tn_s = 0;
    for (int64_t i = 0; i < n_samples; ++i) {
        if (L[i] && P[i]) tp_s++;
        else if (!L[i] && P[i]) fp_s++;
        else if (L[i] && !P[i]) fn_s++;
        else tn_s++;
    }

    double n = n_samples;
    double po = (tp_s + tn_s) / n;
    double pe = ((tp_s + fp_s) * (tp_s + fn_s) + (fn_s + tn_s) * (fp_s + tn_s)) / (n * n);
    stats.kappa = n > 0 ? (po - pe) / std::max(1.0 - pe, 1e-12) : 0.0;

    auto te = find_runs(L);
    auto all_pe = find_runs(P);
    std::vector<std::pair<int64_t, int64_t>> pe_vec;
    for (const auto& r : all_pe) {
        if (r.second - r.first + 1 >= 3) {
            pe_vec.push_back(r);
        }
    }

    int hits = 0;
    std::vector<bool> used(pe_vec.size(), false);
    for (const auto& truth_ev : te) {
        for (size_t j = 0; j < pe_vec.size(); ++j) {
            if (!used[j] && pe_vec[j].first <= truth_ev.second && pe_vec[j].second >= truth_ev.first) {
                used[j] = true;
                hits++;
                break;
            }
        }
    }

    int nt = te.size();
    int npred = pe_vec.size();
    int nok = 0;
    for (bool u : used) if (u) nok++;

    double ev_recall = hits / std::max(1.0, (double)nt);
    double ev_precision = nok / std::max(1.0, (double)npred);
    stats.event_f1 = 2 * ev_recall * ev_precision / std::max(ev_recall + ev_precision, 1e-12);

    stats.tp = hits;
    stats.fn = nt - hits;
    stats.fp = npred - nok;

    return stats;
}

} // namespace eyelive
