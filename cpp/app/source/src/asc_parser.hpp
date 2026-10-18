// EyeLink ASC (the text file written by edf2asc from an EDF recording): samples and the parser's own events. Written from the documented format (EyeLink Data Viewer / EyeLink 1000 Plus user manual); it has NOT been
// run on real EyeLink files yet (docs/eyelive/EYELINK_DATA.md says how to try one).
//   header      SAMPLES GAZE LEFT|RIGHT|LEFT RIGHT RATE 1000.00 TRACKING CR FILTER 2 INPUT [VEL] [RES]      (positions in screen pixels; HREF / PUPIL-only recordings are refused)
//   sample      <time ms> <x> <y> <pupil> [<x> <y> <pupil> for the second eye] [velocity] [resolution] [flags]      '.' = missing (blink, lost pupil)
//   events      SFIX / EFIX, SSACC / ESACC <eye> t1 t2 dur x1 y1 x2 y2 amp pv, SBLINK / EBLINK; MSG, INPUT, BUTTON, START, END, ... are ignored
#pragma once
#include <string>
#include <vector>
#include "eyelive/types.hpp"

namespace eyelive {

struct AscEvent { char kind = 'S';                // 'S' saccade, 'F' fixation, 'B' blink
                  char eye = 'R'; double t_start = 0, t_end = 0, amplitude_deg = 0, peak_speed_deg_s = 0; };

class AscParser {
 public:
    explicit AscParser(const std::string& eye = "auto") : eye_(eye) {}      // "auto": the only eye, or the mean of both; "left" / "right"
    // one line of the file; true when it was a sample (stored in `s`, time in seconds)
    bool feed(const std::string& line, Sample& s);
    const std::string& error() const { return error_; }                      // set when the file cannot be used (HREF data, no SAMPLES header before the first sample)
    bool header_seen() const { return header_seen_; }
    double declared_rate_hz() const { return rate_hz_; }
    bool binocular() const { return left_ && right_; }
    const std::vector<AscEvent>& events() const { return events_; }
    uint64_t samples() const { return n_samples_; }
    uint64_t missing_samples() const { return n_missing_; }
    static bool looks_like_asc(const std::string& first_lines);             // the first lines of a file: '**' comments, MSG, START, ...
 private:
    std::string eye_, error_; bool header_seen_ = false, left_ = false, right_ = false, gaze_ = true; double rate_hz_ = 0; uint64_t n_samples_ = 0, n_missing_ = 0; std::vector<AscEvent> events_;
};

}  // namespace eyelive
