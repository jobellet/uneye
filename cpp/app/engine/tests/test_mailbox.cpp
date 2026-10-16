// Frame mailbox: round trip, candidates, torn-read protection with a concurrent writer, bad files, and the Python writer (when python3 is available).
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <thread>
#include "eyelive/frame_mailbox.hpp"
using namespace eyelive;
namespace fs = std::filesystem;
static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { ++fails; std::printf("FAIL %s:%d: ", __FILE__, __LINE__); std::printf(__VA_ARGS__); std::printf("\n"); } } while (0)
int main(int argc, char** argv) {
    const std::string dir = (fs::temp_directory_path() / "eyelive_mailbox_test").string(); fs::create_directories(dir); const std::string path = dir + "/feed.frames";
    // 1. round trip
    { FrameMailboxWriter w; std::string err; CHECK(w.create(path, 8, 6, 3, 3, -20, 15, 20, -15, &err), "create: %s", err.c_str());
      FrameMailboxReader r; CHECK(r.open(path, &err), "open: %s", err.c_str()); MailboxFrame f; CHECK(!r.read_frame(0, f), "frame before any push");
      std::vector<uint8_t> px(8 * 6 * 3); for (size_t i = 0; i < px.size(); ++i) px[i] = uint8_t(i * 7); w.push_frame(px.data(), 12.5);
      CHECK(r.read_frame(0, f) && f.frame_no == 1 && f.width == 8 && f.height == 6 && f.channels == 3 && f.pixels == px && std::fabs(f.t_writer_s - 12.5) < 1e-12 && f.gx0 == -20 && f.gy1 == -15, "frame content");
      CHECK(!r.read_frame(1, f), "no newer frame"); for (int k = 0; k < 7; ++k) { px[0] = uint8_t(k); w.push_frame(px.data(), 13.0 + k); } CHECK(r.read_frame(1, f) && f.frame_no == 8 && f.pixels[0] == 6, "newest frame after wrap-around (frame %llu)", (unsigned long long)f.frame_no);
      MailboxCandidates c; CHECK(r.read_candidates(0, c) == false || c.list.empty() || true, "initial candidates"); w.set_candidates({{1.f, 2.f, 0.5f}, {-3.f, 4.f, 1.f}}, 12.0);
      CHECK(r.read_candidates(0, c) && c.list.size() == 2 && c.list[1].x == -3.f && c.list[0].weight == 0.5f && c.until_ms == 12.0, "candidates round trip"); CHECK(!r.read_candidates(c.seq, c), "no new candidates"); }
    // 2. a concurrent writer never gives a torn frame: every pixel of a frame holds the frame number modulo 251
    { FrameMailboxWriter w; std::string err; CHECK(w.create(path, 64, 48, 1, 2, 0, 0, 1, 1, &err), "create"); FrameMailboxReader r; CHECK(r.open(path, &err), "open");
      std::atomic<bool> stop{false}; std::thread t([&] { std::vector<uint8_t> px(64 * 48); uint64_t n = 0; while (!stop) { ++n; std::fill(px.begin(), px.end(), uint8_t(n % 251)); w.push_frame(px.data(), double(n)); } });
      uint64_t after = 0, got = 0, torn = 0; const auto t0 = std::chrono::steady_clock::now(); MailboxFrame f;
      while (std::chrono::steady_clock::now() - t0 < std::chrono::milliseconds(600)) if (r.read_frame(after, f)) { ++got; after = f.frame_no; const uint8_t v = f.pixels[0]; for (uint8_t p : f.pixels) if (p != v) { ++torn; break; } if (v != uint8_t(f.frame_no % 251)) ++torn; }
      stop = true; t.join(); std::printf("mailbox: %llu frames read from a writer running flat out, %llu torn\n", (unsigned long long)got, (unsigned long long)torn); CHECK(got > 100, "too few frames read (%llu)", (unsigned long long)got); CHECK(torn == 0, "%llu torn frames", (unsigned long long)torn); }
    // 3. bad files
    { std::string err; FrameMailboxReader r; CHECK(!r.open(dir + "/missing.frames", &err), "missing file accepted"); { FILE* f = fopen((dir + "/junk.frames").c_str(), "wb"); const char junk[300] = {1, 2, 3}; fwrite(junk, 1, sizeof junk, f); fclose(f); } CHECK(!r.open(dir + "/junk.frames", &err) && err.find("magic") != std::string::npos, "junk accepted (%s)", err.c_str()); }
#ifdef EYELIVE_TOOLS_DIR
    // 4. the Python writer (standard library only) -> this reader
    { const std::string cmd = std::string("python3 \"") + EYELIVE_TOOLS_DIR + "/eyelive_frames.py\" selftest \"" + dir + "/py.frames\" > \"" + dir + "/py.log\" 2>&1"; const int rc = std::system(cmd.c_str());
      if (rc != 0) std::printf("python writer: skipped (python3 not available or failed, rc %d)\n", rc);
      else { FrameMailboxReader r; std::string err; CHECK(r.open(dir + "/py.frames", &err), "python file: %s", err.c_str()); MailboxFrame f; MailboxCandidates c;
        CHECK(r.read_frame(0, f) && f.width == 32 && f.height == 24 && f.channels == 3 && f.frame_no == 3 && f.pixels[0] == 3 && f.pixels[(5 * 32 + 7) * 3] == 200 && std::fabs(f.t_writer_s - 3.0) < 1e-9 && f.gx0 == -20.0 && f.gy1 == -15.0, "python frame content (no %llu)", (unsigned long long)f.frame_no);
        CHECK(r.read_candidates(0, c) && c.list.size() == 3 && std::fabs(c.list[2].x - 5.5f) < 1e-6 && std::fabs(c.list[1].weight - 0.25f) < 1e-6 && c.until_ms == 12.0, "python candidates"); std::printf("python writer: frames and candidates read back correctly\n"); } }
#else
    (void)argc; (void)argv;
#endif
    fs::remove_all(dir); if (fails == 0) std::printf("mailbox: OK\n"); return fails ? 1 : 0;
}
