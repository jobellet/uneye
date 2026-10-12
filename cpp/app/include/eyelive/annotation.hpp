// Expert annotation model (CONTRACT). Implemented by task T07, consumed by the GUI Review tab (T15) and the trainer glue (engine).
// Rules: events are sorted by onset; classes are mutually exclusive per sample (class 0 = background is implicit); an edit that makes an event overlap another one WINS: the overlapped part of the other event is trimmed
// (an event that would become empty is removed, an event cut in the middle is split in two). Every mutating operation is one undo step. Invalid requests (index out of range, onset > offset, negative indices,
// class 0 or >= n_classes) return false and leave the model unchanged.
#pragma once
#include <cstdint>
#include <string>
#include <utility>
#include <vector>
#include "eyelive/types.hpp"

namespace eyelive {

class ExpertLabels {
 public:
    explicit ExpertLabels(int n_classes = 3);              // default classes: 0 fixation (background), 1 saccade, 2 blink
    ExpertLabels(const ExpertLabels& o);                          // deep copies (the state is owned through a pointer): added by the integrator after T07 delivered a leaking, shallow-copy version
    ExpertLabels& operator=(const ExpertLabels& o);
    ExpertLabels(ExpertLabels&& o) noexcept;
    ExpertLabels& operator=(ExpertLabels&& o) noexcept;
    ~ExpertLabels();
    int n_classes() const;
    void set_n_classes(int n);                                    // 2..kMaxClasses; events of removed classes are deleted

    const std::vector<Event>& events() const;                     // current (corrected) events, sorted by onset
    const std::vector<Event>& prelabels() const;                  // the pre-labeler's events as first loaded (provenance, never modified by edits)

    // pre-labeling: replaces the events inside [from, to] (inclusive) by `events` (origin Detector, with the detector id / model version they carry); one undo step; the original copy is kept in prelabels()
    void load_prelabels(const std::vector<Event>& events, int64_t from, int64_t to);

    bool add(int64_t onset, int64_t offset, uint8_t cls);         // origin Expert
    bool remove(size_t index);
    bool move_onset(size_t index, int64_t new_onset);
    bool move_offset(size_t index, int64_t new_offset);
    bool set_class(size_t index, uint8_t cls);
    bool split(size_t index, int64_t at);                         // [onset, at-1] and [at, offset]
    bool merge(size_t first_index, size_t second_index);          // adjacent events of the same class

    bool undo();
    bool redo();
    void clear_history();

    // ranges the expert marks as "do not train on" and ranges marked as reviewed (training-set counter)
    void exclude_range(int64_t from, int64_t to, bool excluded);
    const std::vector<std::pair<int64_t, int64_t>>& excluded() const;
    void mark_reviewed(int64_t from, int64_t to, bool reviewed, bool undoable = true);   // undoable = false: the range changes without an undo step (used right after an edit, so that one Undo reverts the edit AND its range)
    const std::vector<std::pair<int64_t, int64_t>>& reviewed() const;
    // Automatic labels (the current detector or model) for [from, to]: replaces the DETECTOR events and pre-labels inside the parts of [from, to] that are not reviewed; reviewed parts and every expert event are never touched.
    // Not an undo step: the same replacement is applied to the undo / redo history, so Undo never removes or resurrects automatic labels. `events` must lie inside [from, to].
    void apply_auto_labels(const std::vector<Event>& events, int64_t from, int64_t to);
    double reviewed_seconds(double rate_hz) const;                // total reviewed length, ranges merged, minus nothing

    // per-sample labels for training / display: 0 background, 1.. class, kIgnoreLabel inside excluded ranges and outside reviewed ranges when only_reviewed is true
    void render(int64_t from, int64_t to, bool only_reviewed, std::vector<uint8_t>& out) const;

    bool save(const std::string& path, std::string* error) const; // binary or JSON, documented by the implementation; round trip must be exact
    bool load(const std::string& path, std::string* error);
 private:
    struct Impl; Impl* impl_;                                     // pimpl: the implementation owns it (copy / move to be implemented by T07)
};

// Agreement between detected and expert events of one class (same scoring as foundation/compare.py::event_f1): a detected run of >= 3 samples overlapping an expert event counts as a hit; kappa is
// sample-wise, class-versus-rest, over n_samples. Used for the live "agreement with the expert" display and for the calibrator.
struct AgreementStats { int tp = 0, fp = 0, fn = 0; double event_f1 = 0, kappa = 0; };
AgreementStats agreement(const std::vector<Event>& detected, const std::vector<Event>& expert, int64_t n_samples, uint8_t cls);

}  // namespace eyelive
