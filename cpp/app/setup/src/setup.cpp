#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include "eyelive/setup.hpp"
#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
#else
#include <arpa/inet.h>
#include <sys/socket.h>
#include <sys/statvfs.h>
#include <unistd.h>
#endif

namespace eyelive {
namespace fs = std::filesystem;
namespace {
std::string trim(const std::string& s) { size_t a = 0, b = s.size(); while (a < b && std::isspace((unsigned char)s[a])) ++a; while (b > a && std::isspace((unsigned char)s[b - 1])) --b; return s.substr(a, b - a); }
bool to_double(const std::string& v, double* out) { if (v.empty()) return false; char* e = nullptr; errno = 0; const double d = std::strtod(v.c_str(), &e); if (e == v.c_str() || *e != 0 || errno == ERANGE || !std::isfinite(d)) return false; *out = d; return true; }
std::string num(double v) { char b[40]; std::snprintf(b, sizeof b, "%.10g", v); return b; }
std::string human_bytes(double b) { char o[40]; if (b >= 1e9) std::snprintf(o, sizeof o, "%.1f GB", b / 1e9); else if (b >= 1e6) std::snprintf(o, sizeof o, "%.0f MB", b / 1e6); else std::snprintf(o, sizeof o, "%.0f kB", b / 1e3); return o; }
}  // namespace

double pixels_per_degree(const ScreenGeometry& g) {
    if (!g.complete()) return 0; const double half_deg = std::atan2(g.width_cm / 2.0, g.distance_cm) * 180.0 / M_PI; return half_deg > 0 ? double(g.res_x) / (2.0 * half_deg) : 0;
}
double degrees_per_pixel(const ScreenGeometry& g) { const double p = pixels_per_degree(g); return p > 0 ? 1.0 / p : 0; }

std::string profile_file_name(const std::string& name) {
    std::string o; for (unsigned char c : name) { if (std::isalnum(c) || c == '.' || c == '_' || c == '-') o += char(c); else if (c == ' ' || c == '/' || c == '\\') { if (!o.empty() && o.back() != '_') o += '_'; } }
    for (size_t i; (i = o.find("..")) != std::string::npos;) o.erase(i, 1);        // no ".."
    while (!o.empty() && (o.front() == '.' || o.front() == '_')) o.erase(o.begin());   // no hidden file
    if (o.empty()) o = "profile"; if (o.size() > 80) o.resize(80); return o + ".profile";
}

std::string serialize_profile(const DeviceProfile& p) {
    auto clean = [](std::string s) { for (char& c : s) if (c == '\n' || c == '\r') c = ' '; return s; };
    std::string o = "# EyeLive device profile\n"; auto kv = [&](const std::string& k, const std::string& v) { o += k + "=" + clean(v) + "\n"; };
    kv("name", p.name); kv("tracker", p.tracker); kv("source", p.source.kind); kv("bind", p.source.bind_address); kv("port", std::to_string(p.source.port)); kv("file", p.source.path); kv("file_speed", num(p.source.speed)); kv("sim_rate_hz", num(p.source.rate_hz));
    kv("rate_hz", num(p.rate_hz)); kv("units", p.units); kv("screen_width_cm", num(p.screen.width_cm)); kv("screen_height_cm", num(p.screen.height_cm)); kv("screen_distance_cm", num(p.screen.distance_cm));
    kv("screen_res_x", std::to_string(p.screen.res_x)); kv("screen_res_y", std::to_string(p.screen.res_y)); kv("detector", p.detector_id); kv("landing_model", p.landing_model); kv("triggers_file", p.triggers_file); kv("duration_min", num(p.duration_min)); kv("notes", p.notes);
    for (const auto& e : p.extra) kv(e.first, e.second); return o;
}

bool parse_profile(const std::string& text, DeviceProfile& out, std::string* error) {
    DeviceProfile p; std::istringstream in(text); std::string line; int ln = 0;
    auto fail = [&](const std::string& m) { if (error) *error = "line " + std::to_string(ln) + ": " + m; return false; };
    while (std::getline(in, line)) { ++ln; line = trim(line); if (line.empty() || line[0] == '#') continue; const size_t e = line.find('='); if (e == std::string::npos) return fail("expected key=value");
        const std::string k = trim(line.substr(0, e)), v = trim(line.substr(e + 1)); double d = 0;
        auto number = [&](double* dst) { if (!to_double(v, &d)) return false; *dst = d; return true; }; auto integer = [&](int* dst) { if (!to_double(v, &d) || d != std::floor(d) || std::fabs(d) > 2e9) return false; *dst = int(d); return true; };
        if (k == "name") p.name = v; else if (k == "tracker") p.tracker = v; else if (k == "source") p.source.kind = v; else if (k == "bind") p.source.bind_address = v; else if (k == "file") p.source.path = v;
        else if (k == "port") { if (!integer(&p.source.port)) return fail("bad number for port"); } else if (k == "file_speed") { if (!number(&p.source.speed)) return fail("bad number for file_speed"); }
        else if (k == "sim_rate_hz") { if (!number(&p.source.rate_hz)) return fail("bad number for sim_rate_hz"); } else if (k == "rate_hz") { if (!number(&p.rate_hz)) return fail("bad number for rate_hz"); }
        else if (k == "units") { if (v != "degrees" && v != "pixels" && v != "auto") return fail("units must be degrees, pixels or auto"); p.units = v; } else if (k == "screen_width_cm") { if (!number(&p.screen.width_cm)) return fail("bad number for screen_width_cm"); }
        else if (k == "screen_height_cm") { if (!number(&p.screen.height_cm)) return fail("bad number for screen_height_cm"); } else if (k == "screen_distance_cm") { if (!number(&p.screen.distance_cm)) return fail("bad number for screen_distance_cm"); }
        else if (k == "screen_res_x") { if (!integer(&p.screen.res_x)) return fail("bad number for screen_res_x"); } else if (k == "screen_res_y") { if (!integer(&p.screen.res_y)) return fail("bad number for screen_res_y"); }
        else if (k == "detector") p.detector_id = v; else if (k == "landing_model") p.landing_model = v; else if (k == "triggers_file") p.triggers_file = v; else if (k == "duration_min") { if (!number(&p.duration_min)) return fail("bad number for duration_min"); }
        else if (k == "notes") p.notes = v; else p.extra[k] = v; }
    out = p; return true;
}

bool profile_problems(const DeviceProfile& p, std::vector<std::string>* problems) {
    std::vector<std::string> bad, warn; const std::string& k = p.source.kind;
    if (k != "udp" && k != "file" && k != "sim") bad.push_back("source must be udp, file or sim (not '" + k + "')");
    if (k == "udp" && (p.source.port < 1 || p.source.port > 65535)) bad.push_back("port must be between 1 and 65535");
    if (k == "udp" && p.source.bind_address.empty()) bad.push_back("bind address is empty");
    if (k == "file" && p.source.path.empty()) bad.push_back("no file to replay");
    if (p.rate_hz < 0 || p.rate_hz > 10000) bad.push_back("rate_hz must be 0 (measured) or up to 10000");
    if (k == "sim" && (p.source.rate_hz < 50 || p.source.rate_hz > 10000)) bad.push_back("simulation rate must be between 50 and 10000 Hz");
    if (p.duration_min <= 0 || p.duration_min > 24 * 60) bad.push_back("planned duration must be between 1 minute and 24 hours");
    if (p.units != "degrees" && p.units != "pixels" && p.units != "auto") bad.push_back("units must be degrees, pixels or auto");
    if (p.units == "pixels" && !p.screen.complete()) bad.push_back("the tracker sends pixels: give the screen width, the viewing distance and the horizontal resolution");
    if (p.screen.width_cm < 0 || p.screen.distance_cm < 0 || p.screen.res_x < 0) bad.push_back("screen values cannot be negative");
    if (p.screen.complete()) { if (p.screen.distance_cm < 20 || p.screen.distance_cm > 300) warn.push_back("warning: viewing distance " + num(p.screen.distance_cm) + " cm is unusual");
        if (p.screen.width_cm < 10 || p.screen.width_cm > 300) warn.push_back("warning: screen width " + num(p.screen.width_cm) + " cm is unusual");
        if (p.screen.height_cm > 0 && p.screen.res_y > 0) { const double a = (p.screen.width_cm / p.screen.res_x) / (p.screen.height_cm / p.screen.res_y); if (std::fabs(a - 1.0) > 0.05) warn.push_back("warning: the pixels are not square (" + num(100 * (a - 1)) + " % difference between the horizontal and vertical size): check the screen size"); } }
    if (p.detector_id.empty()) bad.push_back("no detector");
    if (problems) { problems->clear(); for (auto& s : bad) problems->push_back(s); for (auto& s : warn) problems->push_back(s); } return bad.empty();
}

bool save_profile(const std::string& dir, const DeviceProfile& p, std::string* error) {
    std::error_code ec; fs::create_directories(dir, ec); if (ec) { if (error) *error = "cannot create " + dir + ": " + ec.message(); return false; }
    const std::string path = (fs::path(dir) / profile_file_name(p.name)).string(), tmp = path + ".tmp"; { std::ofstream f(tmp, std::ios::binary); if (!f) { if (error) *error = "cannot write " + tmp; return false; } f << serialize_profile(p); if (!f) { if (error) *error = "write failed: " + tmp; return false; } }
    fs::rename(tmp, path, ec); if (ec) { fs::remove(tmp, ec); if (error) *error = "cannot replace " + path; return false; } return true;      // (written to a temporary file first: a crash never leaves half a profile)
}
bool load_profile(const std::string& path, DeviceProfile& out, std::string* error) {
    std::ifstream f(path, std::ios::binary); if (!f) { if (error) *error = "cannot read " + path; return false; } std::stringstream s; s << f.rdbuf(); if (s.str().size() > 1 << 20) { if (error) *error = "profile file too large: " + path; return false; } return parse_profile(s.str(), out, error);
}
std::vector<std::string> list_profiles(const std::string& dir) {
    std::vector<std::string> v; std::error_code ec; for (const auto& e : fs::directory_iterator(dir, ec)) { if (ec) break; const auto& p = e.path(); if (p.extension() == ".profile") v.push_back(p.stem().string()); } std::sort(v.begin(), v.end()); return v;
}
std::string default_profile_dir() { const char* h = std::getenv("HOME"); if (!h) h = std::getenv("USERPROFILE"); return std::string(h ? h : ".") + "/.eyelive/profiles"; }

std::vector<DeviceProfile> builtin_profiles() {
    std::vector<DeviceProfile> v; DeviceProfile sim; sim.name = "Simulation (demo)"; sim.tracker = "simulated eye"; sim.source.kind = "sim"; sim.source.rate_hz = 1000; sim.duration_min = 10; v.push_back(sim);
    DeviceProfile udp; udp.name = "Tracker over UDP, degrees"; udp.tracker = "generic"; udp.source.kind = "udp"; udp.source.port = 5005; udp.source.bind_address = "127.0.0.1"; udp.duration_min = 60; v.push_back(udp);
    DeviceProfile px = udp; px.name = "Tracker over UDP, screen pixels"; px.units = "pixels"; px.screen.width_cm = 53; px.screen.height_cm = 30; px.screen.distance_cm = 60; px.screen.res_x = 1920; px.screen.res_y = 1080; v.push_back(px);
    DeviceProfile file; file.name = "Replay a recorded file"; file.tracker = "recording"; file.units = "auto"; file.source.kind = "file"; file.source.speed = 1.0; file.duration_min = 30; v.push_back(file); return v;
}

SessionConfig session_config(const DeviceProfile& p, const std::string& session_path) {
    SessionConfig c; c.source = p.source; c.session_path = session_path; c.duration_s = p.duration_min * 60.0; c.rate_hz = p.rate_hz; c.detector_id = p.detector_id; c.deg_per_unit = p.units == "degrees" ? 1.0 : (p.units == "pixels" ? degrees_per_pixel(p.screen) : 0.0);
    c.profile_name = p.name; c.landing_model = p.landing_model; return c;
}

bool parse_trigger_rule(const std::string& line, TriggerRule& out, std::string* error) {
    TriggerRule r; std::stringstream ss(line); std::string kv; auto fail = [&](const std::string& m) { if (error) *error = m; return false; }; auto num_of = [&](const std::string& k, const std::string& v, double* d) { if (!to_double(v, d)) { fail("bad number for " + k); return false; } return true; };
    while (std::getline(ss, kv, ';')) { kv = trim(kv); if (kv.empty()) continue; const size_t e = kv.find('='); if (e == std::string::npos) return fail("expected key=value in '" + kv + "'"); const std::string k = trim(kv.substr(0, e)), v = trim(kv.substr(e + 1)); double d = 0;
        if (k == "name") r.name = v; else if (k == "cls") { if (!num_of(k, v, &d)) return false; if (d < 1 || d >= kMaxClasses || d != std::floor(d)) return fail("cls must be 1 to " + std::to_string(kMaxClasses - 1)); r.cls = uint8_t(d); }
        else if (k == "any") r.any_class = v == "1" || v == "true"; else if (k == "amp_min") { if (!num_of(k, v, &d)) return false; r.amp_min_deg = float(d); } else if (k == "amp_max") { if (!num_of(k, v, &d)) return false; r.amp_max_deg = float(d); }
        else if (k == "anchor") { if (v == "provisional") r.anchor = TriggerAnchor::OnsetProvisional; else if (v == "final") r.anchor = TriggerAnchor::OnsetFinal; else if (v == "offset") r.anchor = TriggerAnchor::Offset; else return fail("anchor must be provisional, final or offset"); }
        else if (k == "delay_ms") { if (!num_of(k, v, &d)) return false; if (d < 0 || d > 5000) return fail("delay_ms must be between 0 and 5000"); r.delay_ms = d; } else if (k == "refractory_ms") { if (!num_of(k, v, &d)) return false; if (d < 0 || d > 600000) return fail("refractory_ms must be between 0 and 600000"); r.refractory_ms = d; }
        else if (k == "sink") r.sink = v; else if (k == "payload") r.payload = v; else if (k == "enabled") r.enabled = v != "0" && v != "false"; else return fail("unknown key '" + k + "'"); }
    if (r.name.empty()) return fail("a rule needs name=..."); if (r.sink.empty()) return fail("a rule needs sink=udp:HOST:PORT, log:FILE or stdout");
    if (r.amp_min_deg > r.amp_max_deg) return fail("amp_min is larger than amp_max");
    if (r.sink.rfind("udp:", 0) == 0) { const size_t c = r.sink.rfind(':'); double port = 0; if (c <= 4 || !to_double(r.sink.substr(c + 1), &port) || port < 1 || port > 65535 || port != std::floor(port)) return fail("sink 'udp:HOST:PORT' needs a port between 1 and 65535"); }
    else if (r.sink.rfind("log:", 0) == 0) { if (r.sink.size() == 4) return fail("sink 'log:' needs a file name"); } else if (r.sink != "stdout") return fail("sink must be udp:HOST:PORT, log:FILE or stdout");
    out = r; return true;
}
bool load_trigger_rules(const std::string& path, std::vector<TriggerRule>& out, std::string* error) {
    std::ifstream f(path); if (!f) { if (error) *error = "cannot read " + path; return false; } std::vector<TriggerRule> v; std::string line; int ln = 0;
    while (std::getline(f, line)) { ++ln; line = trim(line); if (line.empty() || line[0] == '#') continue; TriggerRule r; std::string e; if (!parse_trigger_rule(line, r, &e)) { if (error) *error = path + ":" + std::to_string(ln) + ": " + e; return false; } v.push_back(r); }
    out = v; return true;
}

bool udp_port_free(const std::string& bind_address, int port, std::string* error) {
#ifdef _WIN32
    WSADATA w; if (WSAStartup(MAKEWORD(2, 2), &w) != 0) { if (error) *error = "network stack not available"; return false; }
    SOCKET s = socket(AF_INET, SOCK_DGRAM, 0); if (s == INVALID_SOCKET) { if (error) *error = "cannot create a socket"; return false; }
#else
    int s = socket(AF_INET, SOCK_DGRAM, 0); if (s < 0) { if (error) *error = "cannot create a socket"; return false; }
#endif
    sockaddr_in a{}; a.sin_family = AF_INET; a.sin_port = htons(uint16_t(port)); bool ok = inet_pton(AF_INET, bind_address.c_str(), &a.sin_addr) == 1; if (!ok && error) *error = "'" + bind_address + "' is not an IPv4 address";
    if (ok) { ok = bind(s, reinterpret_cast<sockaddr*>(&a), sizeof a) == 0; if (!ok && error) *error = "cannot listen on " + bind_address + ":" + std::to_string(port) + " (already used by another program?)"; }
#ifdef _WIN32
    closesocket(s);
#else
    close(s);
#endif
    return ok;
}

double free_disk_bytes(const std::string& dir) {
#ifdef _WIN32
    std::error_code ec; const auto si = fs::space(dir, ec); return ec ? -1.0 : double(si.available);
#else
    struct statvfs st; if (statvfs(dir.c_str(), &st) != 0) return -1.0; return double(st.f_bavail) * double(st.f_frsize);
#endif
}

std::vector<CheckResult> run_checks(const DeviceProfile& p, const std::string& session_dir) {
    std::vector<CheckResult> r; auto add = [&](const std::string& what, CheckLevel l, const std::string& m, const std::string& hint = "") { r.push_back({what, l, m, hint}); };
    std::vector<std::string> problems; const bool usable = profile_problems(p, &problems);
    if (!usable) for (auto& s : problems) add("profile", CheckLevel::Fail, s, "open the profile and correct it"); else if (!problems.empty()) for (auto& s : problems) add("profile", CheckLevel::Warn, s); else add("profile", CheckLevel::Ok, "'" + p.name + "' is complete");
    if (p.source.kind == "udp") { std::string e; if (udp_port_free(p.source.bind_address, p.source.port, &e)) add("network port", CheckLevel::Ok, "UDP port " + std::to_string(p.source.port) + " is free"); else add("network port", CheckLevel::Fail, e, "close the program that uses the port, or change the port in the tracker settings and in the profile"); }
    if (p.source.kind == "file") { std::error_code ec; if (fs::exists(p.source.path, ec)) add("replay file", CheckLevel::Ok, p.source.path); else add("replay file", CheckLevel::Fail, "file not found: " + p.source.path, "choose the CSV (t,x,y[,valid]) to replay"); }
    { std::error_code ec; fs::create_directories(session_dir, ec); const std::string probe = (fs::path(session_dir) / ".eyelive_write_test").string(); std::ofstream f(probe); const bool w = bool(f); f.close(); fs::remove(probe, ec);
      if (!w) add("session folder", CheckLevel::Fail, "cannot write in " + session_dir, "choose a folder you own");
      else { const double free_b = free_disk_bytes(session_dir); const double need = (p.rate_hz > 0 ? p.rate_hz : (p.source.kind == "sim" ? p.source.rate_hz : 1000.0)) * p.duration_min * 60.0 * 19.0 * 1.05;
          if (free_b >= 0 && free_b < need * 1.5) add("disk space", CheckLevel::Fail, "needs about " + human_bytes(need) + ", only " + human_bytes(free_b) + " free", "free some space or shorten the planned duration"); else add("disk space", CheckLevel::Ok, "about " + human_bytes(need) + " needed, " + (free_b >= 0 ? human_bytes(free_b) + " free" : "free space unknown")); } }
    if (p.units == "pixels" && p.screen.complete()) add("scale", CheckLevel::Ok, num(pixels_per_degree(p.screen)) + " pixels per degree at the screen centre"); else if (p.units == "degrees") add("scale", CheckLevel::Ok, "the stream is in degrees"); else if (p.units == "auto") add("scale", CheckLevel::Warn, "the scale will be estimated from the data (needs about a minute of saccades)", "give the screen geometry for an exact scale");
    if (!p.landing_model.empty()) add("landing network", CheckLevel::Ok, "'" + p.landing_model + "' will be loaded at the start (a failure is reported, the tables are used instead)");
    return r;
}

Health assess_health(const EngineStats& s, double since_start) {
    Health h; if (!s.running) { h.level = HealthLevel::Idle; h.message = "not recording"; return h; }
    if (s.samples == 0) { if (since_start > 2.0) { h.level = HealthLevel::Bad; h.message = "no sample received for " + num(since_start) + " s: is the tracker sending to this port?"; } else { h.level = HealthLevel::Warn; h.message = "waiting for the first sample"; } return h; }
    if (!s.warning.empty()) { h.level = HealthLevel::Warn; h.message = s.warning; }
    if (s.dropped_packets > 0 && s.samples > 0 && double(s.dropped_packets) > 0.01 * double(s.samples)) { h.level = HealthLevel::Bad; h.message = std::to_string(s.dropped_packets) + " packets lost: the network or the tracker is overloaded"; return h; }
    if (s.gaps > 0 && h.level == HealthLevel::Idle) { h.level = HealthLevel::Warn; h.message = std::to_string(s.gaps) + " gap(s) in the time stamps"; }
    if (s.detect_latency_ms_p95 > 25.0) { h.level = HealthLevel::Warn; h.message = "detection latency is high (" + num(s.detect_latency_ms_p95) + " ms at the 95th percentile): the computer is busy"; }
    if (s.capacity_s > 0 && s.recorded_s > 0.95 * s.capacity_s) { h.level = HealthLevel::Bad; h.message = "the session file is almost full (" + num(s.recorded_s) + " of " + num(s.capacity_s) + " s)"; return h; }
    if (h.level == HealthLevel::Idle) { h.level = HealthLevel::Ok; h.message = "recording, " + num(s.rate_hz_measured) + " Hz"; } return h;
}

}  // namespace eyelive
