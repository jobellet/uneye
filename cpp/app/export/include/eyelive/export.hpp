// Export of a session file (recorder.hpp) to the formats laboratories use, a one-page report, and the reproducibility journal. Everything works on a finished or crashed session opened read-only.
//   events CSV      one row per final event: index, class, onsets / offsets (sample and seconds), amplitude, peak speed, direction, origin
//   BIDS-style      <base>_events.tsv (onset, duration, trial_type, amplitude, peak_velocity, direction, origin) + <base>_events.json, <base>_eyetrack.tsv (timestamp x y valid label) + <base>_eyetrack.json
//   MAT-file v5     variables t, x, y, valid, label, expert (n x 1 doubles), events (m x 9), rate_hz, deg_per_unit; readable by MATLAB >= 5, Octave, scipy.io.loadmat
//   report          statistics of the session (SessionSummary), as text and as a self-contained HTML page
//   provenance      <session>.provenance.json: engine version, parameters, SHA-256 of the data and of the models, written when a session ends
#pragma once
#include <cstdint>
#include <map>
#include <string>
#include <vector>
#include "eyelive/recorder.hpp"

namespace eyelive {

// ---- SHA-256 (FIPS 180-4), used for the data / model fingerprints
class Sha256 {
 public:
    Sha256();
    void update(const void* data, size_t n);
    std::string hex();                          // finishes the hash; the object can not be used afterwards
 private:
    void block(const uint8_t* p);
    uint32_t h_[8]; uint8_t buf_[64]; size_t nbuf_ = 0; uint64_t total_ = 0;
};
std::string sha256_hex(const void* data, size_t n);
std::string sha256_file(const std::string& path, std::string* error);       // "" on error

// ---- events: the final events of the table (provisional ones are never written)
std::vector<Event> final_events(const Recorder& rec);
bool export_events_csv(const Recorder& rec, const std::string& path, std::string* error);
bool write_events_csv(const std::vector<Event>& events, const SessionInfo& info, const std::string& path, std::string* error);       // any list of events (a re-detection that is not stored in the session file)
bool export_bids(const Recorder& rec, const std::string& base_path, std::string* error);      // writes <base>_events.tsv / .json and <base>_eyetrack.tsv / .json
bool export_mat(const Recorder& rec, const std::string& path, std::string* error);

// ---- report
struct SessionSummary {
    std::string path;
    uint64_t samples = 0; double duration_s = 0, rate_hz = 0, deg_per_unit = 1.0; bool scale_estimated = false;
    double invalid_fraction = 0; uint64_t blink_samples = 0, gaps = 0; double max_gap_ms = 0;
    std::map<std::string, uint64_t> events_by_class;
    double saccade_rate_hz = 0, amp_median = 0, amp_q1 = 0, amp_q3 = 0, peak_median = 0, isi_median_ms = 0;
    double main_sequence_slope = 0, main_sequence_r2 = 0;        // log10(peak speed) = a + slope * log10(amplitude), saccades >= 0.3 deg
    std::vector<std::pair<float, float>> main_sequence;          // (amplitude, peak speed) of at most 2000 saccades, for the plot
    std::vector<std::string> warnings;
};
SessionSummary summarize(const Recorder& rec);
SessionSummary summarize(const ITraceView& trace, const std::vector<Event>& final_events, const std::string& label);
std::string report_text(const SessionSummary& s);
std::string report_html(const SessionSummary& s);
bool write_report(const SessionSummary& s, const std::string& path, std::string* error);       // .html or anything else (text)

// ---- provenance journal
struct ProvenanceInfo {
    std::string engine_version, build, profile, detector_id, landing_model, user_model, command_line;
    std::map<std::string, std::string> extra;
};
std::string engine_version();                                  // git description at build time, "unknown" otherwise
std::string provenance_json(const Recorder& rec, const ProvenanceInfo& info, double wall_clock_unix_s);
bool write_provenance(const Recorder& rec, const ProvenanceInfo& info, const std::string& path, std::string* error);      // path "" = <session>.provenance.json
std::string json_escape(const std::string& s);

}  // namespace eyelive
