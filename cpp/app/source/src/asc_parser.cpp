#include "asc_parser.hpp"
#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <sstream>

namespace eyelive {
namespace {
std::vector<std::string> split(const std::string& line) { std::vector<std::string> v; std::istringstream in(line); std::string t; while (in >> t) v.push_back(t); return v; }
bool number(const std::string& t, double* d) { if (t.empty() || t == ".") return false; char* e = nullptr; errno = 0; const double v = std::strtod(t.c_str(), &e); if (e == t.c_str() || *e != 0 || errno == ERANGE || !std::isfinite(v)) return false; *d = v; return true; }
bool is_digit_start(const std::string& s) { return !s.empty() && (std::isdigit(static_cast<unsigned char>(s[0])) || s[0] == '-' || s[0] == '+'); }
}  // namespace

bool AscParser::looks_like_asc(const std::string& head) {
    std::istringstream in(head); std::string line; int checked = 0;
    while (std::getline(in, line) && checked < 40) { ++checked; if (line.empty()) continue; if (line.rfind("**", 0) == 0 || line.rfind("MSG", 0) == 0 || line.rfind("START", 0) == 0 || line.rfind("SAMPLES", 0) == 0 || line.rfind("EVENTS", 0) == 0 || line.rfind("PRESCALER", 0) == 0) return true; if (is_digit_start(line)) return false; }
    return false;
}

bool AscParser::feed(const std::string& line, Sample& s) {
    if (line.empty()) return false;
    const unsigned char c0 = static_cast<unsigned char>(line[0]);
    if (!std::isdigit(c0) && c0 != '-' && c0 != '+') {                                   // not a sample: header, event or message
        const std::vector<std::string> tok = split(line); if (tok.empty()) return false; const std::string& k = tok[0];
        if (k == "SAMPLES") { header_seen_ = true; left_ = right_ = false; gaze_ = true; for (size_t i = 1; i < tok.size(); ++i) { if (tok[i] == "LEFT") left_ = true; else if (tok[i] == "RIGHT") right_ = true; else if (tok[i] == "HREF" || tok[i] == "PUPIL") gaze_ = false; else if (tok[i] == "RATE" && i + 1 < tok.size()) number(tok[i + 1], &rate_hz_); }
            if (!gaze_) error_ = "this ASC file holds HREF data, not screen coordinates: convert the EDF again with edf2asc and the option that writes GAZE samples (screen pixels)"; if (!left_ && !right_) error_ = "no eye (LEFT / RIGHT) in the SAMPLES line"; }
        else if (k == "START" && !header_seen_) { for (size_t i = 1; i < tok.size(); ++i) { if (tok[i] == "LEFT") left_ = true; else if (tok[i] == "RIGHT") right_ = true; } }
        else if ((k == "ESACC" || k == "EFIX" || k == "EBLINK") && tok.size() >= 5) {
            AscEvent e; e.kind = k == "ESACC" ? 'S' : (k == "EFIX" ? 'F' : 'B'); e.eye = tok[1].empty() ? 'R' : tok[1][0]; double a = 0, b = 0; if (number(tok[2], &a) && number(tok[3], &b)) { e.t_start = a / 1000.0; e.t_end = b / 1000.0;
                if (e.kind == 'S' && tok.size() >= 11) { double amp = 0, pv = 0; if (number(tok[9], &amp)) e.amplitude_deg = amp; if (number(tok[10], &pv)) e.peak_speed_deg_s = pv; } events_.push_back(e); } }
        return false;
    }
    if (!error_.empty()) return false;
    if (!header_seen_) { error_ = "a sample line comes before the SAMPLES header: not an EyeLink ASC file with samples (was it converted with edf2asc -e only?)"; return false; }
    const std::vector<std::string> tok = split(line); double t = 0; if (tok.size() < 3 || !number(tok[0], &t)) return false;
    const bool bino = left_ && right_; if (bino && tok.size() < 7) return false;
    auto eye = [&](size_t first, double* x, double* y) { return first + 1 < tok.size() && number(tok[first], x) && number(tok[first + 1], y); };
    double xl = 0, yl = 0, xr = 0, yr = 0; bool vl = false, vr = false;
    if (bino) { vl = eye(1, &xl, &yl); vr = eye(4, &xr, &yr); } else if (left_) vl = eye(1, &xl, &yl); else vr = eye(1, &xr, &yr);
    bool want_l = false, want_r = false; if (eye_ == "left") want_l = true; else if (eye_ == "right") want_r = true; else { want_l = vl; want_r = vr; }      // auto: every valid eye
    if (!bino) { want_l = want_l && left_; want_r = want_r && right_; }
    double x = 0, y = 0; int n = 0; if (want_l && vl) { x += xl; y += yl; ++n; } if (want_r && vr) { x += xr; y += yr; ++n; }
    s = Sample(); s.t = t / 1000.0; ++n_samples_;
    if (n == 0) { s.x = s.y = 0; s.flags = 0; ++n_missing_; } else { s.x = float(x / n); s.y = float(y / n); s.flags = kFlagValid; }
    return true;
}

}  // namespace eyelive
