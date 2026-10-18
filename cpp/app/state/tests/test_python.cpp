#include "eyelive/state_io.hpp"
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cstring>
#ifndef _WIN32
#include <unistd.h>
#endif
#include <string>

using namespace eyelive;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <cpp_dir>\n";
        return 1;
    }
    std::string cpp_dir = argv[1];

    int ret = std::system("python3 -c 'import sys'");
    if (ret != 0) {
        std::cout << "Skipped: python3 is missing\n";
        return 0;
    }

    const char* temp_file = "test_state_block.bin";

    // Create a 256-byte block
    alignas(64) uint8_t raw_block[256];
    std::memset(raw_block, 0, sizeof(raw_block));

    StateBlockWriter writer(raw_block);
    writer.init(120.0, 1.5, 3);

    Sample s;
    s.t = 1.234;
    s.x = 5.0f;
    s.y = 10.0f;
    writer.publish_sample(s, 2.5f, 1, true, 42);

    Event ev;
    ev.t_onset = 1.0;
    ev.t_offset = 1.2;
    ev.amplitude_deg = 3.14f;
    ev.peak_speed_deg_s = 50.0f;
    ev.direction_rad = 1.57f;
    ev.cls = 2;
    ev.provisional = false;
    writer.publish_event(ev, 7);

    // We must flush or ensure memory order
    std::atomic_thread_fence(std::memory_order_release);

    // Write to temp file
    std::ofstream out(temp_file, std::ios::binary);
    out.write(reinterpret_cast<const char*>(raw_block), 256);
    out.close();

    // Call python script
    std::string cmd = "python3 " + cpp_dir + "/app/readers/python/test_reader.py " + std::string(temp_file);
    int p_ret = std::system(cmd.c_str());

    std::remove(temp_file);

    if (p_ret != 0) {
        std::cerr << "Python integration test failed.\n";
        return 1;
    }

    std::cout << "Python integration test passed.\n";
    return 0;
}
