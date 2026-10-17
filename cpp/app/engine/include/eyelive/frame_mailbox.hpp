// Frame mailbox: a stimulus script (Psychtoolbox, PsychoPy, a C++ program) shares small frames and/or candidate targets with the engine through one memory-mapped file.
// The script owns and creates the file; the engine maps it read-only and polls it. No sockets, no copy through the GPU: the script sends the image it already has (or a 64 x 64 copy of
// the screen drawn in an off-screen window). Layout (little endian, all offsets in bytes):
//   0   char[8]  magic "EYLFRM01"            8  u32 version = 1          12  u32 header_bytes = 256
//   16  u32 width       20 u32 height        24 u32 channels (1 gray | 3 RGB8, row 0 = top)       28 u32 slots (frame ring, >= 2)
//   32  u32 slot_header_bytes = 32           36 u32 max_candidates = 64
//   40  u64 frame_count (frames published so far; the newest is in slot (frame_count - 1) % slots)  -- written last
//   48  f64 gaze_x_left   56 f64 gaze_y_top   64 f64 gaze_x_right   72 f64 gaze_y_bottom   (gaze-unit coordinates of the frame edges)
//   80  u64 cand_seq (seqlock: odd while the script writes the candidates)   88 u32 cand_count   96 f64 cand_until_ms
//   104 f64 cand_t_writer_s (time stamp of the candidate list in the writer's clock, 0 = none: the engine then uses the time it saw the list)
//   256 candidates: max_candidates x { f32 x, f32 y, f32 weight, f32 reserved }  (gaze units)
//   256 + 16 * max_candidates = first slot; slot = { u64 seq (seqlock), f64 t_writer_s, u64 frame_no, u32 flags, u32 reserved } (32 bytes) + width * height * channels pixel bytes
// Writer protocol: slot.seq += 1 (odd); write t, frame_no, pixels; slot.seq += 1 (even); then frame_count = frame_no (release). Reader: copy the slot, accept it only if seq is even and unchanged and frame_no matches.
// Python (standard library only) and MATLAB writers: cpp/app/tools/eyelive_frames.py, eyelive_frames.m.
#pragma once
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif
#include "eyelive/engine.hpp"

namespace eyelive {
namespace fm {
constexpr uint32_t kVersion = 1, kHeaderBytes = 256, kSlotHeader = 32, kMaxCand = 64;
constexpr size_t kOffMagic = 0, kOffVersion = 8, kOffHeader = 12, kOffW = 16, kOffH = 20, kOffC = 24, kOffSlots = 28, kOffSlotHdr = 32, kOffMaxCand = 36, kOffCount = 40, kOffGx0 = 48, kOffGy0 = 56, kOffGx1 = 64, kOffGy1 = 72, kOffCandSeq = 80, kOffCandN = 88, kOffCandUntil = 96, kOffCandT = 104;
inline size_t slot_bytes(uint32_t w, uint32_t h, uint32_t c) { return kSlotHeader + size_t(w) * h * c; }
inline size_t file_bytes(uint32_t w, uint32_t h, uint32_t c, uint32_t slots) { return kHeaderBytes + 16 * size_t(kMaxCand) + slot_bytes(w, h, c) * slots; }

// a memory-mapped file, read-only or read-write
class Map {
 public:
    Map() = default; Map(const Map&) = delete; Map& operator=(const Map&) = delete; ~Map() { close(); }
    bool create(const std::string& path, size_t bytes, std::string* err) { return open_(path, bytes, true, err); }
    bool open_readonly(const std::string& path, std::string* err) { return open_(path, 0, false, err); }
    void close() {
#ifdef _WIN32
        if (p_) UnmapViewOfFile(p_); if (map_) CloseHandle(map_); if (file_ != INVALID_HANDLE_VALUE) CloseHandle(file_); file_ = INVALID_HANDLE_VALUE; map_ = nullptr;
#else
        if (p_) munmap(p_, n_); if (fd_ >= 0) ::close(fd_); fd_ = -1;
#endif
        p_ = nullptr; n_ = 0;
    }
    uint8_t* data() const { return static_cast<uint8_t*>(p_); } size_t size() const { return n_; } explicit operator bool() const { return p_ != nullptr; }
 private:
    bool open_(const std::string& path, size_t bytes, bool create, std::string* err) {
        close();
#ifdef _WIN32
        file_ = CreateFileA(path.c_str(), create ? (GENERIC_READ | GENERIC_WRITE) : GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, create ? CREATE_ALWAYS : OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (file_ == INVALID_HANDLE_VALUE) { if (err) *err = "cannot open " + path; return false; }
        LARGE_INTEGER sz; if (create) { sz.QuadPart = LONGLONG(bytes); SetFilePointerEx(file_, sz, nullptr, FILE_BEGIN); SetEndOfFile(file_); } else GetFileSizeEx(file_, &sz);
        n_ = size_t(create ? bytes : size_t(sz.QuadPart)); map_ = CreateFileMappingA(file_, nullptr, create ? PAGE_READWRITE : PAGE_READONLY, 0, 0, nullptr);
        p_ = map_ ? MapViewOfFile(map_, create ? FILE_MAP_ALL_ACCESS : FILE_MAP_READ, 0, 0, 0) : nullptr;
#else
        fd_ = ::open(path.c_str(), create ? (O_RDWR | O_CREAT | O_TRUNC) : O_RDONLY, 0644);
        if (fd_ < 0) { if (err) *err = "cannot open " + path; return false; }
        if (create) { if (ftruncate(fd_, off_t(bytes)) != 0) { if (err) *err = "cannot size " + path; close(); return false; } n_ = bytes; }
        else { struct stat st; if (fstat(fd_, &st) != 0) { close(); return false; } n_ = size_t(st.st_size); }
        void* m = n_ ? mmap(nullptr, n_, create ? (PROT_READ | PROT_WRITE) : PROT_READ, MAP_SHARED, fd_, 0) : MAP_FAILED; p_ = (m == MAP_FAILED) ? nullptr : m;
#endif
        if (!p_) { if (err) *err = "cannot map " + path; close(); return false; }
        return true;
    }
    void* p_ = nullptr; size_t n_ = 0;
#ifdef _WIN32
    HANDLE file_ = INVALID_HANDLE_VALUE, map_ = nullptr;
#else
    int fd_ = -1;
#endif
};
inline std::atomic<uint64_t>* a64(uint8_t* base, size_t off) { return reinterpret_cast<std::atomic<uint64_t>*>(base + off); }
inline const std::atomic<uint64_t>* a64(const uint8_t* base, size_t off) { return reinterpret_cast<const std::atomic<uint64_t>*>(base + off); }
template <class T> inline T get(const uint8_t* b, size_t off) { T v; std::memcpy(&v, b + off, sizeof v); return v; }
template <class T> inline void put(uint8_t* b, size_t off, T v) { std::memcpy(b + off, &v, sizeof v); }
}  // namespace fm

// C++ writer (tests, C++ stimulus programs); the same protocol is implemented in Python and MATLAB.
class FrameMailboxWriter {
 public:
    bool create(const std::string& path, uint32_t w, uint32_t h, uint32_t channels, uint32_t slots, double gx0, double gy0, double gx1, double gy1, std::string* err) {
        if (channels != 1 && channels != 3) { if (err) *err = "channels must be 1 or 3"; return false; }
        if (slots < 2) slots = 2; if (!map_.create(path, fm::file_bytes(w, h, channels, slots), err)) return false;
        uint8_t* b = map_.data(); std::memcpy(b, "EYLFRM01", 8); fm::put<uint32_t>(b, fm::kOffVersion, fm::kVersion); fm::put<uint32_t>(b, fm::kOffHeader, fm::kHeaderBytes); fm::put<uint32_t>(b, fm::kOffW, w); fm::put<uint32_t>(b, fm::kOffH, h);
        fm::put<uint32_t>(b, fm::kOffC, channels); fm::put<uint32_t>(b, fm::kOffSlots, slots); fm::put<uint32_t>(b, fm::kOffSlotHdr, fm::kSlotHeader); fm::put<uint32_t>(b, fm::kOffMaxCand, fm::kMaxCand);
        fm::put<double>(b, fm::kOffGx0, gx0); fm::put<double>(b, fm::kOffGy0, gy0); fm::put<double>(b, fm::kOffGx1, gx1); fm::put<double>(b, fm::kOffGy1, gy1); w_ = w; h_ = h; c_ = channels; slots_ = slots; n_ = 0; return true;
    }
    void push_frame(const uint8_t* pixels, double t_writer_s) {                               // pixels: w * h * channels bytes, row 0 = top
        uint8_t* b = map_.data(); const uint64_t no = n_ + 1; uint8_t* s = b + fm::kHeaderBytes + 16 * size_t(fm::kMaxCand) + fm::slot_bytes(w_, h_, c_) * (size_t((no - 1) % slots_));
        auto* seq = fm::a64(s, 0); seq->fetch_add(1, std::memory_order_acq_rel); fm::put<double>(s, 8, t_writer_s); fm::put<uint64_t>(s, 16, no); fm::put<uint32_t>(s, 24, 0);
        std::memcpy(s + fm::kSlotHeader, pixels, size_t(w_) * h_ * c_); seq->fetch_add(1, std::memory_order_release); n_ = no; fm::a64(b, fm::kOffCount)->store(no, std::memory_order_release);
    }
    void set_candidates(const std::vector<Candidate>& c, double until_ms = 1e9, double t_writer_s = 0) {
        uint8_t* b = map_.data(); auto* seq = fm::a64(b, fm::kOffCandSeq); seq->fetch_add(1, std::memory_order_acq_rel); const uint32_t n = uint32_t(std::min<size_t>(c.size(), fm::kMaxCand));
        fm::put<uint32_t>(b, fm::kOffCandN, n); fm::put<double>(b, fm::kOffCandUntil, until_ms); fm::put<double>(b, fm::kOffCandT, t_writer_s);
        for (uint32_t i = 0; i < n; ++i) { uint8_t* p = b + fm::kHeaderBytes + 16 * size_t(i); fm::put<float>(p, 0, c[i].x); fm::put<float>(p, 4, c[i].y); fm::put<float>(p, 8, c[i].weight); fm::put<float>(p, 12, 0.f); }
        seq->fetch_add(1, std::memory_order_release);
    }
    uint64_t frames() const { return n_; }
 private:
    fm::Map map_; uint32_t w_ = 0, h_ = 0, c_ = 0, slots_ = 0; uint64_t n_ = 0;
};

struct MailboxFrame { uint32_t width = 0, height = 0, channels = 0; uint64_t frame_no = 0; double t_writer_s = 0; double gx0 = 0, gy0 = 0, gx1 = 0, gy1 = 0; std::vector<uint8_t> pixels; };
struct MailboxCandidates { uint64_t seq = 0; double until_ms = 1e9, t_writer_s = 0; std::vector<Candidate> list; };

class FrameMailboxReader {
 public:
    bool open(const std::string& path, std::string* err) {
        if (!map_.open_readonly(path, err)) return false; const uint8_t* b = map_.data();
        if (map_.size() < fm::kHeaderBytes || std::memcmp(b, "EYLFRM01", 8) != 0) { if (err) *err = "not a frame mailbox (bad magic)"; map_.close(); return false; }
        if (fm::get<uint32_t>(b, fm::kOffVersion) != fm::kVersion) { if (err) *err = "unsupported frame mailbox version"; map_.close(); return false; }
        w_ = fm::get<uint32_t>(b, fm::kOffW); h_ = fm::get<uint32_t>(b, fm::kOffH); c_ = fm::get<uint32_t>(b, fm::kOffC); slots_ = fm::get<uint32_t>(b, fm::kOffSlots);
        if ((c_ != 1 && c_ != 3) || slots_ < 2 || w_ == 0 || h_ == 0 || map_.size() < fm::file_bytes(w_, h_, c_, slots_)) { if (err) *err = "inconsistent frame mailbox header"; map_.close(); return false; }
        return true;
    }
    bool is_open() const { return bool(map_); }
    uint32_t width() const { return w_; } uint32_t height() const { return h_; } uint32_t channels() const { return c_; }
    uint64_t frame_count() const { return fm::a64(static_cast<const uint8_t*>(map_.data()), fm::kOffCount)->load(std::memory_order_acquire); }
    // the newest frame, if it is newer than `after`; false when there is none or when it was overwritten during the copy (try again later)
    bool read_frame(uint64_t after, MailboxFrame& out) const {
        const uint8_t* b = map_.data(); const uint64_t fc = frame_count(); if (fc == 0 || fc <= after) return false;
        const uint8_t* s = b + fm::kHeaderBytes + 16 * size_t(fm::kMaxCand) + fm::slot_bytes(w_, h_, c_) * size_t((fc - 1) % slots_);
        for (int attempt = 0; attempt < 4; ++attempt) {
            const uint64_t s1 = fm::a64(s, 0)->load(std::memory_order_acquire); if (s1 & 1) continue;
            out.t_writer_s = fm::get<double>(s, 8); out.frame_no = fm::get<uint64_t>(s, 16); out.pixels.resize(size_t(w_) * h_ * c_); std::memcpy(out.pixels.data(), s + fm::kSlotHeader, out.pixels.size());
            std::atomic_thread_fence(std::memory_order_acquire); if (fm::a64(s, 0)->load(std::memory_order_acquire) != s1 || out.frame_no != fc) continue;
            out.width = w_; out.height = h_; out.channels = c_; out.gx0 = fm::get<double>(b, fm::kOffGx0); out.gy0 = fm::get<double>(b, fm::kOffGy0); out.gx1 = fm::get<double>(b, fm::kOffGx1); out.gy1 = fm::get<double>(b, fm::kOffGy1); return true;
        }
        return false;
    }
    // the candidates, if their sequence number differs from `after_seq`
    bool read_candidates(uint64_t after_seq, MailboxCandidates& out) const {
        const uint8_t* b = map_.data(); const uint64_t cs = fm::a64(b, fm::kOffCandSeq)->load(std::memory_order_acquire); if (cs == after_seq) return false;
        for (int attempt = 0; attempt < 4; ++attempt) {
            const uint64_t s1 = fm::a64(b, fm::kOffCandSeq)->load(std::memory_order_acquire); if (s1 & 1) continue;
            const uint32_t n = std::min<uint32_t>(fm::get<uint32_t>(b, fm::kOffCandN), fm::kMaxCand); out.until_ms = fm::get<double>(b, fm::kOffCandUntil); out.t_writer_s = fm::get<double>(b, fm::kOffCandT); out.list.resize(n);
            for (uint32_t i = 0; i < n; ++i) { const uint8_t* p = b + fm::kHeaderBytes + 16 * size_t(i); out.list[i].x = fm::get<float>(p, 0); out.list[i].y = fm::get<float>(p, 4); out.list[i].weight = fm::get<float>(p, 8); }
            std::atomic_thread_fence(std::memory_order_acquire); if (fm::a64(b, fm::kOffCandSeq)->load(std::memory_order_acquire) != s1) continue; out.seq = s1; return true;
        }
        return false;
    }
 private:
    fm::Map map_; uint32_t w_ = 0, h_ = 0, c_ = 0, slots_ = 0;
};

}  // namespace eyelive
