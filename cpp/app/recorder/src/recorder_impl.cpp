#include "eyelive/recorder.hpp"
#include "mmap_wrapper.hpp"
#include <atomic>
#include <cmath>
#include <cstring>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>

namespace eyelive {

// Ensure matching struct definitions for the layout
constexpr uint64_t MAX_EVENTS = 100000;
constexpr uint32_t HEADER_BYTES = 256;
constexpr uint32_t STATE_BLOCK_BYTES = 256;

// We use __atomic intrinsics on the standard uint64_t field instead of std::atomic
// to guarantee safety within the packed struct (even if poorly aligned).
#pragma pack(push, 1)
struct FileHeader {
    char magic[8]; // "EYELIVE1"
    uint32_t version;
    uint32_t header_bytes;
    uint64_t capacity;
    uint64_t count;
    double rate_hz;
    double deg_per_unit;
    uint32_t n_classes;
    uint32_t flags; // bit0 = cleanly closed
    uint64_t created_unix_ns;
    char reserved[256 - 8 - 4 - 4 - 8 - 8 - 8 - 8 - 4 - 4 - 8];
};
#pragma pack(pop)
static_assert(sizeof(FileHeader) == 256, "FileHeader must be exactly 256 bytes");

#pragma pack(push, 1)
struct EventTable {
    uint64_t n_events;
    Event events[MAX_EVENTS];
};
#pragma pack(pop)

RecorderSizing recorder_sizing(double rate_hz, double duration_s, double margin) {
    RecorderSizing sizing;
    sizing.capacity_samples = std::ceil(rate_hz * duration_s * margin) + 1024;

    // Header (256) + StateBlock (256) + arrays (t, x, y, flags, label, expert) + event table
    uint64_t array_bytes = sizing.capacity_samples * (sizeof(double) + sizeof(float) + sizeof(float) + sizeof(uint8_t) + sizeof(uint8_t) + sizeof(uint8_t));
    uint64_t events_bytes = sizeof(uint64_t) + MAX_EVENTS * sizeof(Event);

    sizing.bytes = HEADER_BYTES + STATE_BLOCK_BYTES + array_bytes + events_bytes;
    return sizing;
}

class RecorderImpl : public Recorder {
public:
    RecorderImpl(recorder::MmapResult map, bool read_only) : map_(map), read_only_(read_only) {
        char* base = static_cast<char*>(map_.ptr);
        header_ = reinterpret_cast<FileHeader*>(base);
        state_block_ptr_ = base + HEADER_BYTES;

        char* array_base = base + HEADER_BYTES + STATE_BLOCK_BYTES;
        uint64_t cap = header_->capacity;

        t_ptr_ = reinterpret_cast<double*>(array_base);
        x_ptr_ = reinterpret_cast<float*>(array_base + cap * sizeof(double));
        y_ptr_ = reinterpret_cast<float*>(array_base + cap * sizeof(double) + cap * sizeof(float));
        flags_ptr_ = reinterpret_cast<uint8_t*>(array_base + cap * (sizeof(double) + 2 * sizeof(float)));
        label_ptr_ = reinterpret_cast<uint8_t*>(array_base + cap * (sizeof(double) + 2 * sizeof(float) + sizeof(uint8_t)));
        expert_ptr_ = reinterpret_cast<uint8_t*>(array_base + cap * (sizeof(double) + 2 * sizeof(float) + 2 * sizeof(uint8_t)));

        events_ptr_ = reinterpret_cast<EventTable*>(array_base + cap * (sizeof(double) + 2 * sizeof(float) + 3 * sizeof(uint8_t)));

        info_.rate_hz = header_->rate_hz;
        info_.deg_per_unit = header_->deg_per_unit;
        info_.n_classes = header_->n_classes;
        info_.scale_estimated = false; // By default
        path_ = "unknown";
    }

    ~RecorderImpl() override {
        std::string err;
        close(&err);
    }

    bool push(const Sample& s) override {
        if (read_only_) return false;

        uint64_t idx;
        __atomic_load(&header_->count, &idx, __ATOMIC_RELAXED);

        if (idx >= header_->capacity) {
            return false;
        }

        t_ptr_[idx] = s.t;
        x_ptr_[idx] = s.x;
        y_ptr_[idx] = s.y;
        flags_ptr_[idx] = s.flags;
        label_ptr_[idx] = 0;
        expert_ptr_[idx] = kIgnoreLabel;

        uint64_t next = idx + 1;
        __atomic_store(&header_->count, &next, __ATOMIC_RELEASE);
        return true;
    }

    int64_t size() const override {
        uint64_t sz;
        __atomic_load(&header_->count, &sz, __ATOMIC_ACQUIRE);
        return sz;
    }

    double t(int64_t i) const override { return t_ptr_[i]; }
    float x(int64_t i) const override { return x_ptr_[i]; }
    float y(int64_t i) const override { return y_ptr_[i]; }
    uint32_t flags(int64_t i) const override { return flags_ptr_[i]; }
    uint8_t label(int64_t i) const override { return label_ptr_[i]; }
    const SessionInfo& info() const override { return info_; }

    int64_t capacity() const override {
        return header_->capacity;
    }

    void set_label(int64_t i, uint8_t cls) override {
        if (!read_only_) label_ptr_[i] = cls;
    }

    void set_expert(int64_t i, uint8_t cls) override {
        if (!read_only_) expert_ptr_[i] = cls;
    }

    uint8_t expert(int64_t i) const override {
        return expert_ptr_[i];
    }

    bool append_event(const Event& e) override {
        if (read_only_) return false;
        uint64_t n = events_ptr_->n_events;
        if (n >= MAX_EVENTS) return false;
        events_ptr_->events[n] = e;
        events_ptr_->n_events = n + 1; // Single writer
        return true;
    }

    int64_t event_count() const override {
        return events_ptr_->n_events;
    }

    Event event(int64_t k) const override {
        return events_ptr_->events[k];
    }

    void* state_block() override {
        return state_block_ptr_;
    }

    bool close(std::string* error) override {
        (void)error;
        if (map_.success) {
            if (!read_only_) {
                header_->flags |= 1; // Mark cleanly closed
                recorder::unmap_file(map_, true); // Sync
            } else {
                recorder::unmap_file(map_, false);
            }
        }
        return true;
    }

    bool export_csv(const std::string& path, std::string* error) const override {
        std::ofstream f(path);
        if (!f.is_open()) {
            if (error) *error = "Failed to open CSV for writing";
            return false;
        }
        f << "i,t,x,y,flags,label,expert\n";
        int64_t n = size();
        for (int64_t i = 0; i < n; ++i) {
            f << i << "," << std::fixed << std::setprecision(6) << t(i) << ","
              << std::setprecision(3) << x(i) << "," << y(i) << ","
              << flags(i) << "," << (int)label(i) << "," << (int)expert(i) << "\n";
        }
        return true;
    }

    const std::string& path() const override {
        return path_;
    }

    void set_path(const std::string& p) { path_ = p; }

private:
    recorder::MmapResult map_;
    bool read_only_;
    std::string path_;

    FileHeader* header_ = nullptr;
    void* state_block_ptr_ = nullptr;
    double* t_ptr_ = nullptr;
    float* x_ptr_ = nullptr;
    float* y_ptr_ = nullptr;
    uint8_t* flags_ptr_ = nullptr;
    uint8_t* label_ptr_ = nullptr;
    uint8_t* expert_ptr_ = nullptr;
    EventTable* events_ptr_ = nullptr;

    SessionInfo info_;
};

std::unique_ptr<Recorder> Recorder::create(const RecorderConfig& cfg, std::string* error) {
    RecorderSizing sizing = recorder_sizing(cfg.rate_hz, cfg.duration_s, cfg.margin);

    recorder::MmapResult map = recorder::map_file(cfg.path, sizing.bytes, true, cfg.reserve_disk, cfg.lock_memory, error);
    if (!map.success) {
        return nullptr;
    }

    // Initialize header
    FileHeader* header = reinterpret_cast<FileHeader*>(map.ptr);
    std::memcpy(header->magic, "EYELIVE1", 8);
    header->version = 1;
    header->header_bytes = HEADER_BYTES;
    header->capacity = sizing.capacity_samples;
    header->count = 0;
    header->rate_hz = cfg.rate_hz;
    header->deg_per_unit = cfg.session.deg_per_unit;
    header->n_classes = cfg.session.n_classes;
    header->flags = 0;

    auto now = std::chrono::system_clock::now().time_since_epoch();
    header->created_unix_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(now).count();

    // Initialize events table count to 0
    char* array_base = static_cast<char*>(map.ptr) + HEADER_BYTES + STATE_BLOCK_BYTES;
    EventTable* events = reinterpret_cast<EventTable*>(array_base + sizing.capacity_samples * (sizeof(double) + 2 * sizeof(float) + 3 * sizeof(uint8_t)));
    events->n_events = 0;

    auto rec = std::make_unique<RecorderImpl>(map, false);
    rec->set_path(cfg.path);
    return rec;
}

std::unique_ptr<Recorder> Recorder::open_readonly(const std::string& path, std::string* error) {
    recorder::MmapResult map = recorder::map_file(path, 0, false, false, false, error);
    if (!map.success) {
        return nullptr;
    }

    FileHeader* header = reinterpret_cast<FileHeader*>(map.ptr);
    if (std::memcmp(header->magic, "EYELIVE1", 8) != 0 || header->version != 1) {
        if (error) *error = "Invalid file format";
        recorder::unmap_file(map, false);
        return nullptr;
    }

    auto rec = std::make_unique<RecorderImpl>(map, true);
    rec->set_path(path);
    return rec;
}

} // namespace eyelive
