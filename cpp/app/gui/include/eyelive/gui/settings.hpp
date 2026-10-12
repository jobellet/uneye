// Settings persistence for the GUI (task T12): a plain "key=value" file in the user's home directory (~/.eyelive/settings.txt), read at start and rewritten whenever a value changes. Never throws; a missing file is empty settings.
#pragma once
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <cstdio>

namespace eyelive { namespace gui {

class Settings {
 public:
    static Settings& instance() { static Settings s; return s; }
    std::string get(const std::string& key, const std::string& def = "") const { auto it = kv_.find(key); return it == kv_.end() ? def : it->second; }
    double get_number(const std::string& key, double def) const { auto it = kv_.find(key); if (it == kv_.end()) return def; char* e = nullptr; const double v = std::strtod(it->second.c_str(), &e); return e == it->second.c_str() ? def : v; }
    void set(const std::string& key, const std::string& value) { auto it = kv_.find(key); if (it != kv_.end() && it->second == value) return; kv_[key] = value; dirty_ = true; last_change_ = std::chrono::steady_clock::now(); }      // in memory; written by tick() / flush()
    void set_number(const std::string& key, double v) { char b[64]; std::snprintf(b, sizeof b, "%.10g", v); set(key, b); }
    void set_path(const std::string& path) { flush(); path_ = path; kv_.clear(); dirty_ = false; load(); }
    void tick() { if (dirty_ && std::chrono::steady_clock::now() - last_change_ > std::chrono::seconds(1)) flush(); }      // called by the shell loop: at most one write per second, never while typing
    void flush() { if (dirty_) { save(); dirty_ = false; } }
    ~Settings() { flush(); }          // tests / --settings
 private:
    Settings() { const char* h = std::getenv("HOME"); if (!h) h = std::getenv("USERPROFILE"); path_ = std::string(h ? h : ".") + "/.eyelive/settings.txt"; load(); }
    void load() { std::ifstream f(path_); std::string line; while (std::getline(f, line)) { const size_t e = line.find('='); if (e != std::string::npos) kv_[line.substr(0, e)] = line.substr(e + 1); } }
    void save() const { std::error_code ec; std::filesystem::create_directories(std::filesystem::path(path_).parent_path(), ec); std::ofstream f(path_); for (const auto& kv : kv_) f << kv.first << '=' << kv.second << '\n'; }
    std::string path_; std::map<std::string, std::string> kv_; bool dirty_ = false; std::chrono::steady_clock::time_point last_change_{};
};

}}  // namespace eyelive::gui
