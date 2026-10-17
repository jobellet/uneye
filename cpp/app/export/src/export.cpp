#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <numeric>
#include "eyelive/export.hpp"

namespace eyelive {
namespace {

struct File {                                                    // fclose on every path
    std::FILE* f = nullptr; explicit File(const std::string& p, const char* mode) { f = std::fopen(p.c_str(), mode); } ~File() { if (f) std::fclose(f); }
    bool close() { const bool ok = f && std::fclose(f) == 0; f = nullptr; return ok; }
};
std::string class_name(const SessionInfo& in, int c) { return c >= 0 && c < int(in.class_names.size()) ? in.class_names[size_t(c)] : "class" + std::to_string(c); }
double quantile(std::vector<double> v, double q) { if (v.empty()) return 0; std::sort(v.begin(), v.end()); const double p = q * double(v.size() - 1); const size_t i = size_t(p); const double f = p - double(i); return i + 1 < v.size() ? v[i] * (1 - f) + v[i + 1] * f : v[i]; }
std::string num(double v) { char b[40]; std::snprintf(b, sizeof b, "%.6g", v); return b; }

}  // namespace

std::string json_escape(const std::string& s) {
    std::string o; o.reserve(s.size() + 2);
    for (unsigned char c : s) { switch (c) { case '"': o += "\\\""; break; case '\\': o += "\\\\"; break; case '\n': o += "\\n"; break; case '\r': o += "\\r"; break; case '\t': o += "\\t"; break;
        default: if (c < 0x20) { char b[8]; std::snprintf(b, sizeof b, "\\u%04x", c); o += b; } else o += char(c); } }
    return o;
}

std::vector<Event> final_events(const Recorder& rec) {
    std::vector<Event> v; const int64_t n = rec.event_count(); v.reserve(size_t(std::max<int64_t>(n, 0)));
    for (int64_t k = 0; k < n; ++k) { Event e = rec.event(k); if (!e.provisional) v.push_back(e); }
    std::stable_sort(v.begin(), v.end(), [](const Event& a, const Event& b) { return a.onset < b.onset; });
    return v;
}

bool write_events_csv(const std::vector<Event>& events, const SessionInfo& in, const std::string& path, std::string* error) {
    File f(path, "w"); if (!f.f) { if (error) *error = "cannot write " + path; return false; }
    std::fprintf(f.f, "index,class,class_name,onset_sample,offset_sample,onset_s,offset_s,duration_ms,amplitude_deg,peak_speed_deg_s,direction_deg,origin,confidence\n"); int64_t k = 0;
    for (const Event& e : events) { if (e.provisional) continue; std::fprintf(f.f, "%lld,%d,%s,%lld,%lld,%.6f,%.6f,%.3f,%.4f,%.2f,%.2f,%s,%.3f\n", (long long)k++, int(e.cls), class_name(in, e.cls).c_str(), (long long)e.onset, (long long)e.offset, e.t_onset, e.t_offset,
        (e.t_offset - e.t_onset) * 1000.0 + 1000.0 / std::max(in.rate_hz, 1.0), double(e.amplitude_deg), double(e.peak_speed_deg_s), double(e.direction_rad) * 180.0 / M_PI, e.origin == EventOrigin::Expert ? "expert" : "detector", double(e.confidence)); }
    if (!f.close()) { if (error) *error = "write failed: " + path; return false; } return true;
}
bool export_events_csv(const Recorder& rec, const std::string& path, std::string* error) { return write_events_csv(final_events(rec), rec.info(), path, error); }

bool export_bids(const Recorder& rec, const std::string& base, std::string* error) {
    const SessionInfo& in = rec.info(); const int64_t n = rec.size(); const double t0 = n > 0 ? rec.t(0) : 0.0;
    { File f(base + "_events.tsv", "w"); if (!f.f) { if (error) *error = "cannot write " + base + "_events.tsv"; return false; }
      std::fprintf(f.f, "onset\tduration\ttrial_type\tamplitude\tpeak_velocity\tdirection\torigin\n");
      for (const Event& e : final_events(rec)) std::fprintf(f.f, "%.6f\t%.6f\t%s\t%.4f\t%.2f\t%.2f\t%s\n", e.t_onset - t0, e.t_offset - e.t_onset + 1.0 / std::max(in.rate_hz, 1.0), class_name(in, e.cls).c_str(), double(e.amplitude_deg), double(e.peak_speed_deg_s), double(e.direction_rad) * 180.0 / M_PI, e.origin == EventOrigin::Expert ? "expert" : "detector");
      if (!f.close()) { if (error) *error = "write failed: " + base + "_events.tsv"; return false; } }
    { File f(base + "_events.json", "w"); if (!f.f) { if (error) *error = "cannot write " + base + "_events.json"; return false; }
      std::fprintf(f.f, "{\n \"onset\": {\"Description\": \"Event onset relative to the first sample\", \"Units\": \"s\"},\n \"duration\": {\"Description\": \"Event duration\", \"Units\": \"s\"},\n \"trial_type\": {\"Description\": \"Class of the movement\", \"Levels\": {");
      for (size_t c = 1; c < in.class_names.size(); ++c) std::fprintf(f.f, "%s\"%s\": \"%s\"", c > 1 ? ", " : "", json_escape(in.class_names[c]).c_str(), json_escape(in.class_names[c]).c_str());
      std::fprintf(f.f, "}},\n \"amplitude\": {\"Description\": \"Amplitude of the movement\", \"Units\": \"deg\"},\n \"peak_velocity\": {\"Description\": \"Peak speed\", \"Units\": \"deg/s\"},\n \"direction\": {\"Description\": \"Direction, 0 = rightwards, counter-clockwise\", \"Units\": \"deg\"},\n \"origin\": {\"Description\": \"detector or expert (human review)\"}\n}\n");
      if (!f.close()) { if (error) *error = "write failed: " + base + "_events.json"; return false; } }
    { File f(base + "_eyetrack.tsv", "w"); if (!f.f) { if (error) *error = "cannot write " + base + "_eyetrack.tsv"; return false; }
      std::fprintf(f.f, "timestamp\tx\ty\tvalid\tlabel\n");
      for (int64_t i = 0; i < n; ++i) std::fprintf(f.f, "%.6f\t%.6g\t%.6g\t%d\t%d\n", rec.t(i) - t0, double(rec.x(i)), double(rec.y(i)), rec.valid(i) ? 1 : 0, int(rec.expert(i) ? rec.expert(i) : rec.label(i)));
      if (!f.close()) { if (error) *error = "write failed: " + base + "_eyetrack.tsv"; return false; } }
    { File f(base + "_eyetrack.json", "w"); if (!f.f) { if (error) *error = "cannot write " + base + "_eyetrack.json"; return false; }
      std::fprintf(f.f, "{\n \"SamplingFrequency\": %.6g,\n \"StartTime\": 0,\n \"Columns\": [\"timestamp\", \"x\", \"y\", \"valid\", \"label\"],\n \"timestamp\": {\"Units\": \"s\"},\n \"x\": {\"Description\": \"Horizontal gaze position in stream units\", \"DegreesPerUnit\": %.9g},\n \"y\": {\"Description\": \"Vertical gaze position in stream units\", \"DegreesPerUnit\": %.9g},\n \"valid\": {\"Description\": \"1 = sample usable, 0 = missing / blink\"},\n \"label\": {\"Description\": \"class of the sample (expert label where reviewed, else detector)\", \"Levels\": {", in.rate_hz, in.deg_per_unit, in.deg_per_unit);
      for (size_t c = 0; c < in.class_names.size(); ++c) std::fprintf(f.f, "%s\"%zu\": \"%s\"", c ? ", " : "", c, json_escape(in.class_names[c]).c_str());
      std::fprintf(f.f, "}}\n}\n"); if (!f.close()) { if (error) *error = "write failed: " + base + "_eyetrack.json"; return false; } }
    return true;
}

// ---- MAT-file level 5 (uncompressed, little endian): only real double matrices
namespace {
void put32(std::FILE* f, uint32_t v) { std::fwrite(&v, 4, 1, f); }
bool mat_var(std::FILE* f, const char* name, const std::vector<double>& data, uint32_t rows, uint32_t cols) {
    if (data.size() != size_t(rows) * cols) return false; const uint32_t nl = uint32_t(std::strlen(name)), nl_pad = (nl + 7) / 8 * 8; const uint64_t dbytes = uint64_t(data.size()) * 8;
    const uint64_t body = 16 + 16 + 8 + nl_pad + 8 + dbytes; if (body > 0xFFFFFFFFull) return false;                                   // (a single element is limited to 4 GB)
    put32(f, 14); put32(f, uint32_t(body)); put32(f, 6); put32(f, 8); put32(f, 6); put32(f, 0); put32(f, 5); put32(f, 8); put32(f, rows); put32(f, cols);   // miMATRIX, flags (mxDOUBLE_CLASS), dimensions
    put32(f, 1); put32(f, nl); std::fwrite(name, 1, nl, f); for (uint32_t i = nl; i < nl_pad; ++i) std::fputc(0, f);                          // name (miINT8)
    put32(f, 9); put32(f, uint32_t(dbytes)); if (!data.empty()) std::fwrite(data.data(), 8, data.size(), f); return true;               // real part (miDOUBLE)
}
}  // namespace

bool export_mat(const Recorder& rec, const std::string& path, std::string* error) {
    File f(path, "wb"); if (!f.f) { if (error) *error = "cannot write " + path; return false; }
    char head[128]; std::memset(head, ' ', 116); std::memset(head + 116, 0, 12); const char* txt = "MATLAB 5.0 MAT-file, created by EyeLive"; std::memcpy(head, txt, std::strlen(txt)); head[124] = 0x00; head[125] = 0x01; head[126] = 'I'; head[127] = 'M';
    std::fwrite(head, 1, 128, f.f); const int64_t n = rec.size(); const uint32_t N = uint32_t(n); std::vector<double> v(size_t(std::max<int64_t>(n, 0))); bool ok = true;
    auto col = [&](const char* name, auto get) { for (int64_t i = 0; i < n; ++i) v[size_t(i)] = get(i); ok = ok && mat_var(f.f, name, v, N, 1); };
    col("t", [&](int64_t i) { return rec.t(i); }); col("x", [&](int64_t i) { return double(rec.x(i)); }); col("y", [&](int64_t i) { return double(rec.y(i)); }); col("valid", [&](int64_t i) { return rec.valid(i) ? 1.0 : 0.0; });
    col("label", [&](int64_t i) { return double(rec.label(i)); }); col("expert", [&](int64_t i) { return double(rec.expert(i)); });
    const std::vector<Event> ev = final_events(rec); const uint32_t M = uint32_t(ev.size()); std::vector<double> m(size_t(M) * 9);                                  // column-major: onset, offset (1-based samples), t_onset, t_offset, class, amplitude, peak speed, direction, origin
    for (uint32_t k = 0; k < M; ++k) { const Event& e = ev[k]; const double r[9] = {double(e.onset + 1), double(e.offset + 1), e.t_onset, e.t_offset, double(e.cls), double(e.amplitude_deg), double(e.peak_speed_deg_s), double(e.direction_rad), e.origin == EventOrigin::Expert ? 1.0 : 0.0}; for (int c = 0; c < 9; ++c) m[size_t(c) * M + k] = r[c]; }
    ok = ok && mat_var(f.f, "events", m, M, 9) && mat_var(f.f, "rate_hz", {rec.info().rate_hz}, 1, 1) && mat_var(f.f, "deg_per_unit", {rec.info().deg_per_unit}, 1, 1);
    if (!ok || !f.close()) { if (error) *error = "write failed (or a variable above 4 GB): " + path; return false; } return true;
}

// ---- report
SessionSummary summarize(const Recorder& rec) { return summarize(rec, final_events(rec), rec.path()); }
SessionSummary summarize(const ITraceView& rec, const std::vector<Event>& ev, const std::string& label) {
    SessionSummary s; const SessionInfo& in = rec.info(); const int64_t n = rec.size(); s.path = label; s.samples = uint64_t(n); s.rate_hz = in.rate_hz; s.deg_per_unit = in.deg_per_unit; s.scale_estimated = in.scale_estimated;
    if (n > 1) s.duration_s = rec.t(n - 1) - rec.t(0);
    uint64_t invalid = 0; std::vector<double> dts; const int64_t stride = std::max<int64_t>(1, n / 20000);
    for (int64_t i = 0; i < n; ++i) { if (!rec.valid(i)) ++invalid; if (rec.flags(i) & kFlagBlink) ++s.blink_samples; if (i > 0 && i % stride == 0) dts.push_back(rec.t(i) - rec.t(i - 1)); }
    s.invalid_fraction = n ? double(invalid) / double(n) : 0; const double dt = dts.empty() ? 0 : quantile(dts, 0.5);
    if (dt > 0) for (int64_t i = 1; i < n; ++i) { const double d = rec.t(i) - rec.t(i - 1); if (d > 1.5 * dt) { ++s.gaps; s.max_gap_ms = std::max(s.max_gap_ms, d * 1000.0); } }
    std::vector<double> amp, peak, isi; double prev_onset = -1; std::vector<double> lx, ly;
    for (const Event& e : ev) { if (e.provisional) continue; ++s.events_by_class[class_name(in, e.cls)];
        if (e.cls == 1) { amp.push_back(e.amplitude_deg); peak.push_back(e.peak_speed_deg_s); if (prev_onset >= 0) isi.push_back((e.t_onset - prev_onset) * 1000.0); prev_onset = e.t_onset;
            if (e.amplitude_deg >= 0.3f && e.peak_speed_deg_s > 0) { lx.push_back(std::log10(double(e.amplitude_deg))); ly.push_back(std::log10(double(e.peak_speed_deg_s))); if (s.main_sequence.size() < 2000) s.main_sequence.emplace_back(e.amplitude_deg, e.peak_speed_deg_s); } } }
    if (s.duration_s > 0) s.saccade_rate_hz = double(amp.size()) / s.duration_s;
    s.amp_median = quantile(amp, 0.5); s.amp_q1 = quantile(amp, 0.25); s.amp_q3 = quantile(amp, 0.75); s.peak_median = quantile(peak, 0.5); s.isi_median_ms = quantile(isi, 0.5);
    if (lx.size() >= 5) { const double mx = std::accumulate(lx.begin(), lx.end(), 0.0) / double(lx.size()), my = std::accumulate(ly.begin(), ly.end(), 0.0) / double(ly.size()); double sxx = 0, sxy = 0, syy = 0;
        for (size_t i = 0; i < lx.size(); ++i) { sxx += (lx[i] - mx) * (lx[i] - mx); sxy += (lx[i] - mx) * (ly[i] - my); syy += (ly[i] - my) * (ly[i] - my); }
        if (sxx > 1e-12 && syy > 1e-12) { s.main_sequence_slope = sxy / sxx; s.main_sequence_r2 = sxy * sxy / (sxx * syy); } }
    if (n == 0) s.warnings.push_back("the session contains no sample");
    if (s.invalid_fraction > 0.10) s.warnings.push_back("more than 10 % of the samples are missing or blinks (" + num(100 * s.invalid_fraction) + " %): check the tracker, the pupil detection and the head position");
    if (s.gaps > 0) s.warnings.push_back(std::to_string(s.gaps) + " gap(s) in the time stamps, the longest " + num(s.max_gap_ms) + " ms: packets were lost or the tracker paused");
    if (n > 0 && s.duration_s > 5 && ev.empty()) s.warnings.push_back("no event: the detector was not run, or the scale is wrong");
    if (amp.size() >= 30 && s.main_sequence_r2 < 0.4) s.warnings.push_back("the main sequence (peak speed against amplitude) is weak (R2 = " + num(s.main_sequence_r2) + "): the detected 'saccades' may include noise, or the scale is wrong");
    if (!amp.empty() && s.deg_per_unit == 1.0 && s.amp_median > 40) s.warnings.push_back("the median saccade amplitude is " + num(s.amp_median) + " deg: the stream is probably in pixels, set the screen geometry (degrees per unit)");
    return s;
}

std::string report_text(const SessionSummary& s) {
    std::string o = "EyeLive session report\n======================\nfile: " + s.path + "\n";
    o += "samples: " + std::to_string(s.samples) + "   duration: " + num(s.duration_s) + " s   rate: " + num(s.rate_hz) + " Hz   degrees per unit: " + num(s.deg_per_unit) + (s.scale_estimated ? " (estimated)" : " (set)") + "\n";
    o += "missing / invalid samples: " + num(100 * s.invalid_fraction) + " %   blink samples: " + std::to_string(s.blink_samples) + "   gaps: " + std::to_string(s.gaps) + " (longest " + num(s.max_gap_ms) + " ms)\n";
    o += "events:"; for (const auto& kv : s.events_by_class) o += " " + kv.first + "=" + std::to_string(kv.second); if (s.events_by_class.empty()) o += " none"; o += "\n";
    o += "saccades: " + num(s.saccade_rate_hz) + " /s, amplitude median " + num(s.amp_median) + " deg (IQR " + num(s.amp_q1) + "-" + num(s.amp_q3) + "), peak speed median " + num(s.peak_median) + " deg/s, inter-saccadic interval median " + num(s.isi_median_ms) + " ms\n";
    o += "main sequence: log10(peak speed) = a + " + num(s.main_sequence_slope) + " * log10(amplitude), R2 = " + num(s.main_sequence_r2) + "\n";
    if (s.warnings.empty()) o += "no warning\n"; else for (const auto& w : s.warnings) o += "WARNING: " + w + "\n";
    return o;
}

namespace {
std::string html_escape(const std::string& s) { std::string o; for (char c : s) { switch (c) { case '&': o += "&amp;"; break; case '<': o += "&lt;"; break; case '>': o += "&gt;"; break; case '"': o += "&quot;"; break; default: o += c; } } return o; }
}
std::string report_html(const SessionSummary& s) {
    std::string o = "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width, initial-scale=1\"><title>EyeLive session report</title><style>body{font:15px/1.45 system-ui,sans-serif;max-width:780px;margin:2em auto;padding:0 1em;color:#222}table{border-collapse:collapse;width:100%}td,th{border-bottom:1px solid #ddd;padding:.35em .5em;text-align:left}.warn{background:#fff3cd;border-left:4px solid #e0a800;padding:.5em .8em;margin:.4em 0}.ok{background:#e6f4ea;border-left:4px solid #2e7d32;padding:.5em .8em}svg{background:#fafafa;border:1px solid #ddd}@media(prefers-color-scheme:dark){body{background:#181818;color:#e6e6e6}td,th{border-color:#444}svg{background:#202020;border-color:#444}.warn{background:#3b3200}.ok{background:#12301a}}</style></head><body>";
    o += "<h1>Session report</h1><p><code>" + html_escape(s.path) + "</code></p><table>";
    auto row = [&](const std::string& k, const std::string& v) { o += "<tr><th>" + k + "</th><td>" + html_escape(v) + "</td></tr>"; };
    row("Samples", std::to_string(s.samples)); row("Duration", num(s.duration_s) + " s"); row("Sampling rate", num(s.rate_hz) + " Hz"); row("Degrees per unit", num(s.deg_per_unit) + (s.scale_estimated ? " (estimated)" : " (set)"));
    row("Missing or invalid samples", num(100 * s.invalid_fraction) + " %"); row("Gaps in the time stamps", std::to_string(s.gaps) + " (longest " + num(s.max_gap_ms) + " ms)");
    std::string ev; for (const auto& kv : s.events_by_class) ev += kv.first + ": " + std::to_string(kv.second) + "  "; row("Events", ev.empty() ? "none" : ev);
    row("Saccade rate", num(s.saccade_rate_hz) + " /s"); row("Amplitude (median, IQR)", num(s.amp_median) + " deg (" + num(s.amp_q1) + " - " + num(s.amp_q3) + ")"); row("Peak speed (median)", num(s.peak_median) + " deg/s"); row("Inter-saccadic interval (median)", num(s.isi_median_ms) + " ms");
    row("Main sequence", "slope " + num(s.main_sequence_slope) + " (log-log), R2 " + num(s.main_sequence_r2)); o += "</table>";
    if (s.warnings.empty()) o += "<p class=\"ok\">No warning.</p>"; else for (const auto& w : s.warnings) o += "<p class=\"warn\">" + html_escape(w) + "</p>";
    if (!s.main_sequence.empty()) {                                       // log-log scatter of the main sequence
        const double W = 520, H = 300, L = 48, B = 34; double xmin = 0.3, xmax = 40, ymin = 20, ymax = 1200; o += "<h2>Main sequence</h2><svg viewBox=\"0 0 " + num(W) + " " + num(H) + "\" width=\"100%\" role=\"img\" aria-label=\"peak speed against amplitude\">";
        auto px = [&](double a) { return L + (std::log10(std::clamp(a, xmin, xmax)) - std::log10(xmin)) / (std::log10(xmax) - std::log10(xmin)) * (W - L - 10); }; auto py = [&](double v) { return H - B - (std::log10(std::clamp(v, ymin, ymax)) - std::log10(ymin)) / (std::log10(ymax) - std::log10(ymin)) * (H - B - 10); };
        for (double a : {0.3, 1.0, 3.0, 10.0, 30.0}) o += "<text x=\"" + num(px(a)) + "\" y=\"" + num(H - 12) + "\" font-size=\"11\" text-anchor=\"middle\" fill=\"currentColor\">" + num(a) + "</text>";
        for (double v : {30.0, 100.0, 300.0, 1000.0}) o += "<text x=\"" + num(L - 6) + "\" y=\"" + num(py(v) + 4) + "\" font-size=\"11\" text-anchor=\"end\" fill=\"currentColor\">" + num(v) + "</text>";
        for (const auto& p : s.main_sequence) o += "<circle cx=\"" + num(px(p.first)) + "\" cy=\"" + num(py(p.second)) + "\" r=\"2\" fill=\"#1f77b4\" fill-opacity=\"0.45\"/>";
        o += "<text x=\"" + num(L + (W - L) / 2) + "\" y=\"" + num(H - 1) + "\" font-size=\"11\" text-anchor=\"middle\" fill=\"currentColor\">amplitude (deg)</text><text x=\"10\" y=\"" + num(H / 2) + "\" font-size=\"11\" fill=\"currentColor\" transform=\"rotate(-90 10 " + num(H / 2) + ")\" text-anchor=\"middle\">peak speed (deg/s)</text></svg>";
    }
    return o + "</body></html>\n";
}

bool write_report(const SessionSummary& s, const std::string& path, std::string* error) {
    const bool html = path.size() > 5 && path.compare(path.size() - 5, 5, ".html") == 0; const std::string body = html ? report_html(s) : report_text(s);
    File f(path, "wb"); if (!f.f) { if (error) *error = "cannot write " + path; return false; } std::fwrite(body.data(), 1, body.size(), f.f); if (!f.close()) { if (error) *error = "write failed: " + path; return false; } return true;
}

// ---- provenance
std::string engine_version() {
#ifdef EYELIVE_VERSION
    return EYELIVE_VERSION;
#else
    return "unknown";
#endif
}

std::string provenance_json(const Recorder& rec, const ProvenanceInfo& info, double wall_clock_unix_s) {
    const SessionInfo& in = rec.info(); const int64_t n = rec.size(); Sha256 h;
    for (int64_t i = 0; i < n; ++i) { const double t = rec.t(i); const float x = rec.x(i), y = rec.y(i); const uint32_t fl = rec.flags(i); h.update(&t, 8); h.update(&x, 4); h.update(&y, 4); h.update(&fl, 4); }
    char when[40] = "unknown"; const std::time_t tt = std::time_t(wall_clock_unix_s); std::tm tmv{}; 
#ifdef _WIN32
    gmtime_s(&tmv, &tt);
#else
    gmtime_r(&tt, &tmv);
#endif
    if (wall_clock_unix_s > 0) std::strftime(when, sizeof when, "%Y-%m-%dT%H:%M:%SZ", &tmv);
    auto str = [](const std::string& k, const std::string& v, bool last = false) { return "  \"" + k + "\": \"" + json_escape(v) + "\"" + (last ? "\n" : ",\n"); };
    auto file_entry = [&](const std::string& k, const std::string& path) { std::string err; const std::string d = path.empty() ? "" : sha256_file(path, &err); return "  \"" + k + "\": {\"path\": \"" + json_escape(path) + "\", \"sha256\": \"" + d + "\"},\n"; };
    std::string o = "{\n  \"schema\": \"eyelive-provenance-1\",\n"; o += str("engine_version", info.engine_version.empty() ? engine_version() : info.engine_version); o += str("build", info.build); o += str("written_utc", when);
    o += str("session_path", rec.path()); o += "  \"samples\": " + std::to_string(n) + ",\n  \"rate_hz\": " + num(in.rate_hz) + ",\n  \"deg_per_unit\": " + num(in.deg_per_unit) + ",\n  \"scale_estimated\": " + (in.scale_estimated ? "true" : "false") + ",\n  \"n_classes\": " + std::to_string(in.n_classes) + ",\n";
    o += "  \"events_in_table\": " + std::to_string(rec.event_count()) + ",\n"; o += str("data_sha256", h.hex()); o += str("detector_id", info.detector_id); o += str("profile", info.profile); o += file_entry("landing_model", info.landing_model); o += file_entry("user_model", info.user_model);
    o += str("command_line", info.command_line); o += "  \"extra\": {"; bool first = true; for (const auto& kv : info.extra) { o += std::string(first ? "" : ", ") + "\"" + json_escape(kv.first) + "\": \"" + json_escape(kv.second) + "\""; first = false; } o += "}\n}\n"; return o;
}

bool write_provenance(const Recorder& rec, const ProvenanceInfo& info, const std::string& path, std::string* error) {
    const std::string p = path.empty() ? rec.path() + ".provenance.json" : path; const std::string body = provenance_json(rec, info, double(std::time(nullptr)));
    File f(p, "wb"); if (!f.f) { if (error) *error = "cannot write " + p; return false; } std::fwrite(body.data(), 1, body.size(), f.f); if (!f.close()) { if (error) *error = "write failed: " + p; return false; } return true;
}

}  // namespace eyelive
