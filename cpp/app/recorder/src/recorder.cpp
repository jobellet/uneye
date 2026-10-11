// Recorder: preallocated, memory-mapped session file (task T01). Layout in include/eyelive/recorder.hpp; platform code in os_mmap.cpp.
#include "eyelive/recorder.hpp"
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <sstream>
#include "os_mmap.hpp"

namespace eyelive {
namespace {

constexpr uint64_t kHeaderBytes = 256, kStateBytes = 256, kArraysOffset = 512, kMaxEvents = 100000;
constexpr uint32_t kFlagClosed = 1u, kFlagScaleEstimated = 2u;
constexpr uint64_t kMaxFileBytes = uint64_t(1) << 40;     // 1 TiB: anything larger is an input error, refused before touching the disk

#pragma pack(push, 1)
struct FileHeader {
    char magic[8];                  // "EYELIVE1"
    uint32_t version, header_bytes;
    uint64_t capacity;
    uint64_t count;                 // published with release semantics after the sample is written
    double rate_hz, deg_per_unit;
    uint32_t n_classes, flags;      // flags bit0 = cleanly closed, bit1 = scale estimated
    uint64_t created_unix_ns;
    uint64_t max_events;
    char class_names[128];          // comma separated
    char reserved[256 - 8 - 4 - 4 - 8 - 8 - 8 - 8 - 4 - 4 - 8 - 8 - 128];
};
#pragma pack(pop)
static_assert(sizeof(FileHeader) == 256, "FileHeader must be 256 bytes");
static_assert(std::atomic<uint64_t>::is_always_lock_free, "the published counters must be lock-free");

constexpr uint64_t align64(uint64_t v) { return (v + 63) & ~uint64_t(63); }
struct Layout { uint64_t t, x, y, f, l, e, ev, total; };
Layout layout_for(uint64_t cap) {
    Layout L; L.t = kArraysOffset; L.x = align64(L.t + cap * 8); L.y = align64(L.x + cap * 4); L.f = align64(L.y + cap * 4); L.l = align64(L.f + cap); L.e = align64(L.l + cap);
    L.ev = align64(L.e + cap); L.total = L.ev + 8 + kMaxEvents * sizeof(Event); return L;
}
uint64_t capacity_for(double rate_hz, double duration_s, double margin) { return uint64_t(std::ceil(rate_hz * duration_s * margin)) + 1024; }

inline std::atomic<uint64_t>& A64(void* p) { return *reinterpret_cast<std::atomic<uint64_t>*>(p); }
inline std::atomic<uint8_t>& A8(void* p) { return *reinterpret_cast<std::atomic<uint8_t>*>(p); }
inline std::atomic<uint32_t>& A32(void* p) { return *reinterpret_cast<std::atomic<uint32_t>*>(p); }

class MappedRecorder : public Recorder {
 public:
    MappedRecorder(os::MappedFile m, std::string path, bool writable) : m_(m), path_(std::move(path)), writable_(writable) {}
    ~MappedRecorder() override { os::unmap(m_); }

    bool init_from_header(std::string* error) {
        if (m_.size < kArraysOffset) { if (error) *error = "file too short for a session header"; return false; }
        h_ = static_cast<FileHeader*>(m_.base);
        if (std::memcmp(h_->magic, "EYELIVE1", 8) != 0) { if (error) *error = "not an EyeLive session file (bad magic)"; return false; }
        if (h_->version != 1 || h_->header_bytes != kHeaderBytes) { if (error) *error = "unsupported session file version"; return false; }
        cap_ = h_->capacity; L_ = layout_for(cap_);
        if (cap_ == 0 || cap_ > kMaxFileBytes / 19 || m_.size < L_.total) { if (error) *error = "session file truncated or inconsistent (capacity does not fit the file size)"; return false; }
        char* b = static_cast<char*>(m_.base); t_ = reinterpret_cast<double*>(b + L_.t); x_ = reinterpret_cast<float*>(b + L_.x); y_ = reinterpret_cast<float*>(b + L_.y);
        f_ = reinterpret_cast<uint8_t*>(b + L_.f); l_ = reinterpret_cast<uint8_t*>(b + L_.l); e_ = reinterpret_cast<uint8_t*>(b + L_.e); ev_n_ = b + L_.ev; ev_ = b + L_.ev + 8;
        info_.rate_hz = h_->rate_hz; info_.deg_per_unit = h_->deg_per_unit; info_.n_classes = int(h_->n_classes); info_.scale_estimated = (h_->flags & kFlagScaleEstimated) != 0;
        info_.class_names.clear(); { std::string s(h_->class_names, strnlen(h_->class_names, sizeof h_->class_names)); std::stringstream ss(s); std::string tok; while (std::getline(ss, tok, ',')) info_.class_names.push_back(tok); }
        return true;
    }
    void init_new(const RecorderConfig& cfg, uint64_t cap) {
        h_ = static_cast<FileHeader*>(m_.base); std::memset(h_, 0, sizeof(FileHeader)); std::memcpy(h_->magic, "EYELIVE1", 8); h_->version = 1; h_->header_bytes = uint32_t(kHeaderBytes); h_->capacity = cap;
        h_->rate_hz = cfg.rate_hz; h_->deg_per_unit = cfg.session.deg_per_unit; h_->n_classes = uint32_t(cfg.session.n_classes); h_->flags = cfg.session.scale_estimated ? kFlagScaleEstimated : 0;
        h_->created_unix_ns = uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count()); h_->max_events = kMaxEvents;
        std::string names; for (size_t i = 0; i < cfg.session.class_names.size(); ++i) names += (i ? "," : "") + cfg.session.class_names[i];
        std::strncpy(h_->class_names, names.c_str(), sizeof h_->class_names - 1);
        std::string e; init_from_header(&e); info_ = cfg.session; info_.rate_hz = cfg.rate_hz;
        std::memset(static_cast<char*>(m_.base) + kHeaderBytes, 0, kStateBytes);
    }

    int64_t size() const override { return int64_t(std::min<uint64_t>(A64(&h_->count).load(std::memory_order_acquire), cap_)); }
    double t(int64_t i) const override { return t_[i]; }
    float x(int64_t i) const override { return x_[i]; }
    float y(int64_t i) const override { return y_[i]; }
    uint32_t flags(int64_t i) const override { return f_[i]; }
    uint8_t label(int64_t i) const override { return A8(&l_[i]).load(std::memory_order_relaxed); }
    const SessionInfo& info() const override { return info_; }

    bool push(const Sample& s) override {
        if (!writable_ || closed_) return false;
        const uint64_t n = A64(&h_->count).load(std::memory_order_relaxed); if (n >= cap_) return false;
        t_[n] = s.t; x_[n] = s.x; y_[n] = s.y; f_[n] = uint8_t(s.flags & 0xFFu); A8(&l_[n]).store(0, std::memory_order_relaxed); A8(&e_[n]).store(255, std::memory_order_relaxed);
        A64(&h_->count).store(n + 1, std::memory_order_release); return true;
    }
    int64_t capacity() const override { return int64_t(cap_); }
    void set_label(int64_t i, uint8_t c) override { if (writable_ && i >= 0 && i < size()) A8(&l_[i]).store(c, std::memory_order_relaxed); }
    void set_expert(int64_t i, uint8_t c) override { if (writable_ && i >= 0 && i < size()) A8(&e_[i]).store(c, std::memory_order_relaxed); }
    uint8_t expert(int64_t i) const override { return (i >= 0 && i < size()) ? A8(&e_[i]).load(std::memory_order_relaxed) : uint8_t(255); }
    bool append_event(const Event& e) override {
        if (!writable_ || closed_) return false;
        const uint64_t n = A64(ev_n_).load(std::memory_order_relaxed); if (n >= kMaxEvents) return false;
        std::memcpy(ev_ + n * sizeof(Event), &e, sizeof(Event)); A64(ev_n_).store(n + 1, std::memory_order_release); return true;
    }
    int64_t event_count() const override { return int64_t(std::min<uint64_t>(A64(ev_n_).load(std::memory_order_acquire), kMaxEvents)); }
    Event event(int64_t k) const override { Event e; if (k >= 0 && k < event_count()) std::memcpy(&e, ev_ + uint64_t(k) * sizeof(Event), sizeof(Event)); return e; }
    void* state_block() override { return static_cast<char*>(m_.base) + kHeaderBytes; }
    bool close(std::string* error) override {
        if (!writable_ || closed_) return true;
        closed_ = true; A32(&h_->flags).fetch_or(kFlagClosed, std::memory_order_release); return os::sync(m_, error);
    }
    bool export_csv(const std::string& path, std::string* error) const override {
        std::FILE* f = std::fopen(path.c_str(), "w"); if (!f) { if (error) *error = "cannot write " + path; return false; }
        std::fprintf(f, "index,t,x,y,flags,label,expert\n"); const int64_t n = size();
        for (int64_t i = 0; i < n; ++i) std::fprintf(f, "%lld,%.17g,%.9g,%.9g,%u,%u,%u\n", (long long)i, t_[i], x_[i], y_[i], unsigned(f_[i]), unsigned(label(i)), unsigned(expert(i)));
        const bool ok = std::fclose(f) == 0; if (!ok && error) *error = "write failed: " + path; return ok;
    }
    const std::string& path() const override { return path_; }

 private:
    os::MappedFile m_; std::string path_; bool writable_ = false, closed_ = false; FileHeader* h_ = nullptr; uint64_t cap_ = 0; Layout L_{};
    double* t_ = nullptr; float *x_ = nullptr, *y_ = nullptr; uint8_t *f_ = nullptr, *l_ = nullptr, *e_ = nullptr; char *ev_n_ = nullptr, *ev_ = nullptr; SessionInfo info_;
};

}  // namespace

RecorderSizing recorder_sizing(double rate_hz, double duration_s, double margin) {
    const uint64_t cap = capacity_for(rate_hz, duration_s, margin); return {cap, layout_for(cap).total};
}

std::unique_ptr<Recorder> Recorder::create(const RecorderConfig& cfg, std::string* error) {
    if (!(cfg.rate_hz > 0) || !(cfg.duration_s > 0) || !(cfg.margin >= 1.0) || !std::isfinite(cfg.rate_hz * cfg.duration_s * cfg.margin)) { if (error) *error = "invalid rate, duration or margin"; return nullptr; }
    if (cfg.path.empty()) { if (error) *error = "no session file path"; return nullptr; }
    const double est = cfg.rate_hz * cfg.duration_s * cfg.margin * 19.0;
    if (est > double(kMaxFileBytes)) { if (error) *error = "requested session is too large (" + std::to_string(est / 1e9) + " GB): check the rate and the duration"; return nullptr; }
    const uint64_t cap = capacity_for(cfg.rate_hz, cfg.duration_s, cfg.margin); const Layout L = layout_for(cap);
    os::MappedFile m; bool locked = false;
    if (!os::create_map(cfg.path, L.total, cfg.reserve_disk, cfg.lock_memory, m, &locked, error)) return nullptr;
    auto r = std::make_unique<MappedRecorder>(m, cfg.path, true); r->init_new(cfg, cap); return r;
}

std::unique_ptr<Recorder> Recorder::open_readonly(const std::string& path, std::string* error) {
    os::MappedFile m; if (!os::open_map_readonly(path, m, error)) return nullptr;
    auto r = std::make_unique<MappedRecorder>(m, path, false); if (!r->init_from_header(error)) return nullptr; return r;
}

}  // namespace eyelive
