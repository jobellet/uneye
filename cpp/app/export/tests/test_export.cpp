// Export, report, provenance: round trips on a synthetic session; SHA-256 vectors; MAT-file structure (and scipy when available).
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include "eyelive/export.hpp"
using namespace eyelive;
namespace fs = std::filesystem;
static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { ++fails; std::printf("FAIL %s:%d: ", __FILE__, __LINE__); std::printf(__VA_ARGS__); std::printf("\n"); } } while (0)
static std::string slurp(const std::string& p) { std::ifstream f(p, std::ios::binary); std::stringstream s; s << f.rdbuf(); return s.str(); }
static size_t lines(const std::string& s) { size_t n = 0; for (char c : s) n += c == '\n'; return n; }

static std::string make_session(const std::string& path, int n, bool many_invalid = false) {
    RecorderConfig rc; rc.path = path; rc.rate_hz = 1000; rc.duration_s = 3; rc.reserve_disk = false; rc.lock_memory = false; rc.session.rate_hz = 1000; rc.session.deg_per_unit = 0.5; rc.session.scale_estimated = false; rc.session.n_classes = 3;
    std::string err; auto rec = Recorder::create(rc, &err); if (!rec) { std::printf("cannot create %s: %s\n", path.c_str(), err.c_str()); std::exit(2); }
    for (int i = 0; i < n; ++i) { Sample s; s.t = 10.0 + i * 0.001; s.x = float(i * 0.01); s.y = float(-i * 0.02); s.flags = (i == 5 || (many_invalid && i % 3 == 0)) ? 0u : kFlagValid; rec->push(s); rec->set_label(i, uint8_t(i >= 100 && i < 120 ? 1 : 0)); }
    auto ev = [&](int64_t on, int64_t off, uint8_t cls, float amp, EventOrigin o, bool prov) { Event e; e.onset = on; e.offset = off; e.t_onset = 10.0 + on * 0.001; e.t_offset = 10.0 + off * 0.001; e.cls = cls; e.amplitude_deg = amp; e.peak_speed_deg_s = float(500 * std::pow(amp, 0.8)); e.direction_rad = 0.5f; e.origin = o; e.provisional = prov; rec->append_event(e); };
    ev(100, 119, 1, 4.0f, EventOrigin::Detector, false); ev(300, 330, 1, 9.0f, EventOrigin::Expert, false); ev(600, 640, 2, 0.f, EventOrigin::Detector, false); ev(900, 905, 1, 2.0f, EventOrigin::Detector, true);   // the last one is provisional
    for (int k = 0; k < 40; ++k) ev(1000 + 20 * k, 1010 + 20 * k, 1, 0.5f + 0.2f * float(k % 15), EventOrigin::Detector, false);
    rec->close(&err); return path;
}

int main(int argc, char** argv) {
    const std::string tools = std::string(argc > 1 ? argv[1] : ".") + "/app/export/tests"; const std::string dir = (fs::temp_directory_path() / "eyelive_export_test").string(); fs::remove_all(dir); fs::create_directories(dir);
    // 1. SHA-256 known answers
    CHECK(sha256_hex("", 0) == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", "sha256 of the empty string");
    CHECK(sha256_hex("abc", 3) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad", "sha256 of abc");
    { std::string m(1000000, 'a'); CHECK(sha256_hex(m.data(), m.size()) == "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0", "sha256 of a million a");
      Sha256 s; for (size_t i = 0; i < m.size(); i += 777) s.update(m.data() + i, std::min<size_t>(777, m.size() - i)); CHECK(s.hex() == "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0", "chunked sha256");
      std::ofstream(dir + "/m.bin", std::ios::binary) << m; std::string err; CHECK(sha256_file(dir + "/m.bin", &err) == "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0", "sha256_file"); CHECK(sha256_file(dir + "/missing", &err).empty() && !err.empty(), "missing file"); }
    CHECK(json_escape("a\"b\\c\n\t\x01") == "a\\\"b\\\\c\\n\\t\\u0001", "json_escape");
    // 2. session round trips
    const std::string sp = make_session(dir + "/s.eyelive", 2000); std::string err; auto rec = Recorder::open_readonly(sp, &err); CHECK(bool(rec), "reopen: %s", err.c_str()); if (!rec) return 1;
    CHECK(final_events(*rec).size() == 43, "final events %zu (the provisional one must be left out)", final_events(*rec).size());
    CHECK(export_events_csv(*rec, dir + "/e.csv", &err), "csv: %s", err.c_str()); { const std::string c = slurp(dir + "/e.csv"); CHECK(lines(c) == 44, "csv lines %zu", lines(c)); CHECK(c.find("saccade") != std::string::npos && c.find("expert") != std::string::npos && c.find("blink") != std::string::npos, "csv content"); CHECK(c.find(",100,119,10.100000,10.119000,20.000,4.0000,") != std::string::npos, "csv row of the first event: %s", c.substr(0, 400).c_str()); }
    CHECK(export_bids(*rec, dir + "/sub-01_task-x", &err), "bids: %s", err.c_str());
    { const std::string t = slurp(dir + "/sub-01_task-x_eyetrack.tsv"), e = slurp(dir + "/sub-01_task-x_events.tsv"), j = slurp(dir + "/sub-01_task-x_eyetrack.json"), ej = slurp(dir + "/sub-01_task-x_events.json");
      CHECK(lines(t) == 2001 && lines(e) == 44, "tsv lines %zu %zu", lines(t), lines(e)); CHECK(t.find("\n0.100000\t") != std::string::npos, "timestamps relative to the first sample"); CHECK(j.find("\"SamplingFrequency\": 1000") != std::string::npos && j.find("\"DegreesPerUnit\": 0.5") != std::string::npos, "eyetrack json: %s", j.c_str());
      CHECK(ej.find("\"saccade\"") != std::string::npos && e.find("\t20.000000\t") == std::string::npos && e.find("0.020000\tsaccade\t4.0000") != std::string::npos, "events tsv / json"); }
    // 3. MAT-file: walk the elements
    CHECK(export_mat(*rec, dir + "/x.mat", &err), "mat: %s", err.c_str());
    { const std::string m = slurp(dir + "/x.mat"); CHECK(m.size() > 128 && m.compare(0, 20, "MATLAB 5.0 MAT-file,") == 0 && m[126] == 'I' && m[127] == 'M' && m[124] == 0 && m[125] == 1, "mat header");
      size_t pos = 128; std::vector<std::string> names; auto u32 = [&](size_t p) { uint32_t v; std::memcpy(&v, m.data() + p, 4); return v; }; bool structure_ok = true; double t0 = 0, ev_amp = 0;
      while (pos + 8 <= m.size()) { if (u32(pos) != 14) { structure_ok = false; break; } const uint32_t body = u32(pos + 4); if (pos + 8 + body > m.size()) { structure_ok = false; break; }
          const size_t q = pos + 8; const uint32_t rows = u32(q + 24), cols = u32(q + 28), nl = u32(q + 36); const std::string name = m.substr(q + 40, nl); const size_t dpos = q + 40 + (nl + 7) / 8 * 8; if (u32(dpos) != 9 || u32(dpos + 4) != rows * cols * 8) { structure_ok = false; break; }
          names.push_back(name); double v0; std::memcpy(&v0, m.data() + dpos + 8, 8); if (name == "t") t0 = v0; if (name == "events") { std::memcpy(&ev_amp, m.data() + dpos + 8 + 8 * (5 * rows + 0), 8); }
          pos = q + body; }
      CHECK(structure_ok && pos == m.size(), "mat elements do not tile the file (%zu of %zu)", pos, m.size()); CHECK(names.size() == 9 && names[0] == "t" && names[6] == "events" && names[8] == "deg_per_unit", "mat variables (%zu)", names.size()); CHECK(std::fabs(t0 - 10.0) < 1e-12 && std::fabs(ev_amp - 4.0) < 1e-5, "mat values %g %g", t0, ev_amp); }
    if (std::system("python3 -c \"import scipy.io\" > /dev/null 2>&1") == 0) { const std::string cmd = "python3 \"" + tools + "/check_mat.py\" \"" + dir + "/x.mat\" > \"" + dir + "/mat.log\" 2>&1"; const int rc = std::system(cmd.c_str()); CHECK(rc == 0, "scipy cannot read the MAT-file: %s", slurp(dir + "/mat.log").c_str()); if (rc == 0) std::printf("export: scipy reads the MAT-file\n"); }
    else std::printf("export: scipy not available, MAT check by structure only\n");
    // 4. summary and report
    { SessionSummary s = summarize(*rec); CHECK(s.samples == 2000 && std::fabs(s.duration_s - 1.999) < 1e-9 && std::fabs(s.rate_hz - 1000) < 1e-9, "summary basics"); CHECK(s.events_by_class["saccade"] == 42 && s.events_by_class["blink"] == 1, "events by class: %llu saccades", (unsigned long long)s.events_by_class["saccade"]);
      CHECK(std::fabs(s.main_sequence_slope - 0.8) < 1e-6 && s.main_sequence_r2 > 0.999999, "main sequence slope %.6f R2 %.6f", s.main_sequence_slope, s.main_sequence_r2); CHECK(s.invalid_fraction > 0 && s.invalid_fraction < 0.01 && s.gaps == 0, "invalid %.4f gaps %llu", s.invalid_fraction, (unsigned long long)s.gaps);
      CHECK(s.warnings.empty(), "unexpected warning: %s", s.warnings.empty() ? "" : s.warnings[0].c_str()); const std::string txt = report_text(s), html = report_html(s); CHECK(txt.find("no warning") != std::string::npos && txt.find("saccades: ") != std::string::npos, "text report");
      CHECK(html.find("<svg") != std::string::npos && html.find("</html>") != std::string::npos, "html report"); CHECK(write_report(s, dir + "/r.html", &err) && write_report(s, dir + "/r.txt", &err) && slurp(dir + "/r.html") == html, "write_report");
      SessionSummary hostile = s; hostile.path = "<script>alert(1)</script>"; hostile.warnings.push_back("<b>x</b> & y"); const std::string h2 = report_html(hostile); CHECK(h2.find("<script>") == std::string::npos && h2.find("&lt;script&gt;") != std::string::npos && h2.find("&lt;b&gt;x&lt;/b&gt; &amp; y") != std::string::npos, "html escaping"); }
    { const std::string sp2 = make_session(dir + "/bad.eyelive", 2000, true); auto r2 = Recorder::open_readonly(sp2, &err); SessionSummary s = summarize(*r2); bool inv = false; for (auto& w : s.warnings) inv |= w.find("missing or blinks") != std::string::npos; CHECK(inv, "no warning for 33 %% invalid samples"); }
    // 5. provenance
    { ProvenanceInfo pi; pi.detector_id = "hmm_accel"; pi.profile = "lab \"A\""; pi.landing_model = dir + "/m.bin"; pi.extra["note"] = "x\ny"; const std::string js = provenance_json(*rec, pi, 1760000000.0);
      CHECK(js.find("\"schema\": \"eyelive-provenance-1\"") != std::string::npos && js.find("\"detector_id\": \"hmm_accel\"") != std::string::npos && js.find("\"profile\": \"lab \\\"A\\\"\"") != std::string::npos && js.find("2025-10-09T08:53:20Z") != std::string::npos, "provenance fields: %s", js.c_str());
      CHECK(js.find("\"sha256\": \"cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0\"") != std::string::npos, "model hash in the journal"); CHECK(js.find("\"note\": \"x\\ny\"") != std::string::npos, "extra escaped");
      const std::string other = make_session(dir + "/o.eyelive", 2001); auto r3 = Recorder::open_readonly(other, &err); const std::string js2 = provenance_json(*r3, pi, 1760000000.0); auto h = [](const std::string& j) { const size_t p = j.find("\"data_sha256\": \""); return j.substr(p + 16, 64); };
      CHECK(h(js) != h(js2) && h(js).size() == 64, "the data hash does not depend on the data"); CHECK(write_provenance(*rec, pi, "", &err) && fs::exists(sp + ".provenance.json"), "write_provenance: %s", err.c_str()); CHECK(!write_provenance(*rec, pi, dir + "/nodir/x.json", &err) && !err.empty(), "unwritable path accepted"); }
    fs::remove_all(dir); if (fails == 0) std::printf("export: OK\n"); return fails ? 1 : 0;
}
