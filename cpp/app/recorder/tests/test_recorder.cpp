#include "eyelive/recorder.hpp"
#include <iostream>
#include <thread>
#include <vector>
#include <cmath>
#include <chrono>
#include <unistd.h>
#include <sys/wait.h>
#include <fstream>
#include <filesystem>
#include <cassert>
#include <cstdlib>
#include <atomic>

using namespace eyelive;
namespace fs = std::filesystem;

int fails = 0;

#define EXPECT_EQ(a, b) do { if ((a) != (b)) { std::cerr << __FILE__ << ":" << __LINE__ << ": EXPECT_EQ failed\n"; fails++; } } while(0)
#define EXPECT_TRUE(a) do { if (!(a)) { std::cerr << __FILE__ << ":" << __LINE__ << ": EXPECT_TRUE failed\n"; fails++; } } while(0)
#define EXPECT_FALSE(a) do { if (a) { std::cerr << __FILE__ << ":" << __LINE__ << ": EXPECT_FALSE failed\n"; fails++; } } while(0)
#define ASSERT_NE(a, b) do { if ((a) == (b)) { std::cerr << __FILE__ << ":" << __LINE__ << ": ASSERT_NE failed\n"; exit(1); } } while(0)
#define ASSERT_TRUE(a) do { if (!(a)) { std::cerr << __FILE__ << ":" << __LINE__ << ": ASSERT_TRUE failed\n"; exit(1); } } while(0)

void Sizing() {
    RecorderSizing sizing = recorder_sizing(1000.0, 3600.0, 1.05);

    // capacity: 1000 * 3600 * 1.05 + 1024 = 3781024
    uint64_t expected_cap = 3781024;
    EXPECT_EQ(sizing.capacity_samples, expected_cap);

    uint64_t samples_bytes = expected_cap * 19;
    uint64_t expected_bytes = 256 + 256 + samples_bytes + 8 + 100000 * sizeof(Event);

    EXPECT_EQ(sizing.bytes, expected_bytes);

    // Check if within 1% of 68MB + table
    uint64_t mb68 = 68ULL * 1024 * 1024;
    uint64_t base_expected = mb68 + 8 + 100000 * sizeof(Event);
    double ratio = (double)sizing.bytes / base_expected;
    EXPECT_TRUE(ratio >= 0.99 && ratio <= 1.01);
}

void CreateErrors(fs::path test_dir) {
    RecorderConfig cfg;
    cfg.path = (test_dir / "nonexistent_dir" / "test.rec").string();
    std::string err;
    auto rec = Recorder::create(cfg, &err);
    EXPECT_EQ(rec.get(), nullptr);
    EXPECT_FALSE(err.empty());

    cfg.path = (test_dir / "test.rec").string();
    cfg.duration_s = 1e15; // Impossible size
    rec = Recorder::create(cfg, &err);
    EXPECT_EQ(rec.get(), nullptr);
    EXPECT_FALSE(err.empty());
}

void PushLimit(fs::path test_dir) {
    RecorderConfig cfg;
    cfg.path = (test_dir / "test_limit.rec").string();
    cfg.rate_hz = 100.0;
    cfg.duration_s = 0.0; // capacity = 1024
    cfg.margin = 1.0;
    std::string err;
    auto rec = Recorder::create(cfg, &err);
    ASSERT_NE(rec.get(), nullptr);

    EXPECT_EQ(rec->capacity(), 1024);
    for (int i = 0; i < 1024; ++i) {
        EXPECT_TRUE(rec->push({(double)i, (float)i, (float)i, 0}));
    }
    EXPECT_FALSE(rec->push({1024.0, 1024.0f, 1024.0f, 0}));
}

void RoundTrips(fs::path test_dir) {
    RecorderConfig cfg;
    cfg.path = (test_dir / "test_roundtrips.rec").string();
    std::string err;
    auto rec = Recorder::create(cfg, &err);
    ASSERT_NE(rec.get(), nullptr);

    rec->push({1.0, 2.0f, 3.0f, 4});
    rec->push({5.0, 6.0f, 7.0f, 8});

    rec->set_label(0, 1);
    rec->set_expert(1, 2);

    EXPECT_EQ(rec->label(0), 1);
    EXPECT_EQ(rec->expert(1), 2);

    Event e;
    e.onset = 0;
    e.offset = 1;
    e.cls = 2;
    EXPECT_TRUE(rec->append_event(e));
    EXPECT_EQ(rec->event_count(), 1);
    EXPECT_EQ(rec->event(0).onset, 0);

    std::string csv_path = (test_dir / "test.csv").string();
    EXPECT_TRUE(rec->export_csv(csv_path, &err));

    std::ifstream f(csv_path);
    ASSERT_TRUE(f.is_open());
    std::string line;
    std::getline(f, line); // header
    std::getline(f, line);
    EXPECT_TRUE(line.find("1.000000,2.000,3.000,4,1,255") != std::string::npos);
    std::getline(f, line);
    EXPECT_TRUE(line.find("5.000000,6.000,7.000,8,0,2") != std::string::npos);
}

#ifndef _WIN32
void CrashReload(fs::path test_dir) {
    std::string path = (test_dir / "test_crash.rec").string();

    pid_t pid = fork();
    if (pid == 0) {
        RecorderConfig cfg;
        cfg.path = path;
        cfg.rate_hz = 1000.0;
        cfg.duration_s = 1000.0;
        std::string err;
        auto rec = Recorder::create(cfg, &err);
        if (!rec) exit(1);

        for (int i = 0; i < 1000000; ++i) {
            rec->push({(double)i, (float)i, (float)i, 0});
        }
        // "Crash" by exiting without closing
        exit(0);
    } else {
        int status;
        waitpid(pid, &status, 0);
        EXPECT_EQ(WEXITSTATUS(status), 0);

        std::string err;
        auto rec = Recorder::open_readonly(path, &err);
        ASSERT_NE(rec.get(), nullptr);

        EXPECT_EQ(rec->size(), 1000000);
        EXPECT_EQ(rec->x(999999), 999999.0f);
    }
}
#endif

void Concurrency(fs::path test_dir) {
    RecorderConfig cfg;
    cfg.path = (test_dir / "test_conc.rec").string();
    cfg.rate_hz = 1000.0;
    cfg.duration_s = 2000.0;
    std::string err;
    auto rec = Recorder::create(cfg, &err);
    ASSERT_NE(rec.get(), nullptr);

    std::atomic<bool> writer_done{false};
    std::atomic<int64_t> read_errors{0};

    std::thread writer([&]() {
        for (int i = 0; i < 2000000; ++i) {
            rec->push({(double)i, (float)i, (float)i, 0});
        }
        writer_done = true;
    });

    std::thread reader([&]() {
        int64_t last_i = 0;
        while (!writer_done) {
            int64_t size = rec->size();
            for (int64_t i = last_i; i < size; ++i) {
                if (rec->x(i) != (float)i) {
                    read_errors++;
                }
            }
            last_i = size;
        }
        // Final check
        int64_t size = rec->size();
        for (int64_t i = last_i; i < size; ++i) {
            if (rec->x(i) != (float)i) {
                read_errors++;
            }
        }
    });

    writer.join();
    reader.join();

    EXPECT_EQ(read_errors.load(), 0);
    EXPECT_EQ(rec->size(), 2000000);
}

void Throughput(fs::path test_dir) {
    RecorderConfig cfg;
    cfg.path = (test_dir / "test_tp.rec").string();
    cfg.rate_hz = 1000.0;
    cfg.duration_s = 5000.0;
    std::string err;
    auto rec = Recorder::create(cfg, &err);
    ASSERT_NE(rec.get(), nullptr);

    int num_samples = 5000000;

    auto t1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_samples; ++i) {
        rec->push({(double)i, (float)i, (float)i, 0});
    }
    auto t2 = std::chrono::high_resolution_clock::now();

    double duration = std::chrono::duration<double>(t2 - t1).count();
    double rate = num_samples / duration;

    std::cout << "Throughput: " << rate << " samples/s" << std::endl;

    // Fail only below 1e6
    EXPECT_TRUE(rate > 1000000.0);
}

int main(int argc, char** argv) {
    (void)argc;
    (void)argv;
    fs::path test_dir = fs::temp_directory_path() / "eyelive_recorder_test";
    fs::create_directories(test_dir);

    Sizing();
    CreateErrors(test_dir);
    PushLimit(test_dir);
    RoundTrips(test_dir);
#ifndef _WIN32
    CrashReload(test_dir);
#endif
    Concurrency(test_dir);
    Throughput(test_dir);

    fs::remove_all(test_dir);

    if (fails > 0) {
        std::cerr << fails << " tests failed.\n";
        return 1;
    }
    std::cout << "All tests passed.\n";
    return 0;
}
