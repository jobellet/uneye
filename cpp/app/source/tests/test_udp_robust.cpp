#ifdef _WIN32
int main() { return 0; }          // POSIX sockets only: the Windows build is not tested here
#else
// UDP source robustness: garbage and valid packets mixed, packets with more samples than the reader asks for, floods, bad numbers. Valid samples arrive once, in order; nothing crashes; garbage is counted.
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <vector>
#include "eyelive/source.hpp"
using namespace eyelive;
static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { ++fails; std::printf("FAIL %s:%d: ", __FILE__, __LINE__); std::printf(__VA_ARGS__); std::printf("\n"); } } while (0)
struct Sender {
    int s; sockaddr_in a{}; explicit Sender(int port) { s = socket(AF_INET, SOCK_DGRAM, 0); a.sin_family = AF_INET; a.sin_port = htons(uint16_t(port)); inet_pton(AF_INET, "127.0.0.1", &a.sin_addr); }
    ~Sender() { close(s); }
    void send(const void* d, size_t n) { sendto(s, d, n, 0, reinterpret_cast<sockaddr*>(&a), sizeof a); }
    void send(const std::string& t) { send(t.data(), t.size()); }
};
static std::vector<char> binary_packet(const std::vector<Sample>& v) {
    std::vector<char> p(8 + 20 * v.size()); const uint32_t magic = 0x31455945u /* "EYE1" little endian */, n = uint32_t(v.size()); std::memcpy(p.data(), &magic, 4); std::memcpy(p.data() + 4, &n, 4);
    for (size_t i = 0; i < v.size(); ++i) { char* q = p.data() + 8 + 20 * i; std::memcpy(q, &v[i].t, 8); std::memcpy(q + 8, &v[i].x, 4); std::memcpy(q + 12, &v[i].y, 4); std::memcpy(q + 16, &v[i].flags, 4); } return p;
}
int main() {
    SourceSpec spec; spec.kind = "udp"; spec.port = 5071; std::string err; auto src = make_source(spec, &err); if (!src) { std::printf("cannot open: %s\n", err.c_str()); return 2; }
    Sender tx(spec.port); std::mt19937 rng(7); std::vector<Sample> expected; std::atomic<bool> sent_all{false};   // (expected is read only after the sender has finished)
    std::thread sender([&] {
        double t = 0; for (int round = 0; round < 300; ++round) {
            // garbage: random bytes of random length, a wrong magic, a huge claimed count, truncated binary, text that is not numbers, empty datagram
            { std::vector<char> g(size_t(rng() % 1200) + 1); for (auto& c : g) c = char(rng()); if (g[0] == '-' || g[0] == '+' || g[0] == '.' || (g[0] >= '0' && g[0] <= '9')) g[0] = 'x'; uint32_t m; if (g.size() >= 4) { std::memcpy(&m, g.data(), 4); if (m == 0x31455945u) g[0] = 'q'; } tx.send(g.data(), g.size()); }
            { const uint32_t magic = 0x31455945u, n = 0xFFFFFFFFu; char p[8]; std::memcpy(p, &magic, 4); std::memcpy(p + 4, &n, 4); tx.send(p, 8); }
            { std::vector<Sample> v(10); auto p = binary_packet(v); tx.send(p.data(), p.size() - 7); }
            tx.send("1.0 abc def\n2.0\n\n\r\n3 4"); tx.send("nan nan nan\n"); tx.send("1e999 1 1\n");
            // valid: a binary packet of 1..50 samples, then a text packet of 1..200 lines
            { std::vector<Sample> v(size_t(rng() % 50) + 1); for (auto& s : v) { s.t = t; t += 0.001; s.x = float(std::sin(t) * 5); s.y = float(std::cos(t) * 5); s.flags = 1; expected.push_back(s); } auto p = binary_packet(v); tx.send(p.data(), p.size()); }
            { const int n = int(rng() % 200) + 1;       /* at most 6 kB: macOS refuses datagrams above 9216 bytes */ std::string txt; for (int i = 0; i < n; ++i) { char b[96]; Sample s; s.t = t; t += 0.001; s.x = float(i % 7); s.y = float(-i % 5); std::snprintf(b, sizeof b, "%.6f %.6f %.6f\n", s.t, double(s.x), double(s.y)); txt += b; expected.push_back(s); } tx.send(txt); }
            std::this_thread::sleep_for(std::chrono::microseconds(500));
        }
        sent_all = true;
    });
    std::vector<Sample> got; const auto t0 = std::chrono::steady_clock::now(); size_t last = 0; auto last_change = t0;
    while (std::chrono::steady_clock::now() - t0 < std::chrono::seconds(20)) {
        std::vector<Sample> out; src->read(out, 7, 20); for (auto& s : out) got.push_back(s);          // 7 per call: far fewer than a packet holds
        if (got.size() != last) { last = got.size(); last_change = std::chrono::steady_clock::now(); } else if (sent_all && std::chrono::steady_clock::now() - last_change > std::chrono::milliseconds(600)) break;
        else if (std::chrono::steady_clock::now() - last_change > std::chrono::seconds(3)) break;
    }
    sender.join();
    for (int k = 0; k < 50; ++k) { std::vector<Sample> out; if (src->read(out, 7, 5) == 0) break; for (auto& s : out) got.push_back(s); }    // what was left
    std::printf("udp robust: %zu valid samples sent, %zu received, %llu datagrams counted as dropped\n", expected.size(), got.size(), (unsigned long long)src->dropped_packets());
    CHECK(got.size() == expected.size(), "received %zu of %zu valid samples", got.size(), expected.size());
    bool order = true, exact = true; for (size_t i = 0; i < std::min(got.size(), expected.size()); ++i) { exact &= std::fabs(got[i].t - expected[i].t) < 1e-5 && std::fabs(double(got[i].x) - double(expected[i].x)) < 1e-4; if (i && got[i].t < got[i - 1].t) order = false; }
    CHECK(order && exact, "samples are out of order or altered");
    { size_t k = 0; while (k < std::min(got.size(), expected.size()) && std::fabs(got[k].t - expected[k].t) < 1e-5) ++k; if (k < std::min(got.size(), expected.size())) std::printf("first difference at %zu: got t=%.6f x=%.4f, expected t=%.6f x=%.4f (next expected %.6f)\n", k, got[k].t, double(got[k].x), expected[k].t, double(expected[k].x), k + 1 < expected.size() ? expected[k + 1].t : -1.0); }
    CHECK(src->dropped_packets() >= 600, "garbage was not counted (%llu)", (unsigned long long)src->dropped_packets());
    for (const auto& s : got) if (!std::isfinite(s.t) || !std::isfinite(s.x) || !std::isfinite(s.y)) { CHECK(false, "a non-finite sample was delivered"); break; }
    // a flood without a reader: memory stays bounded and the source keeps working afterwards
    { for (int i = 0; i < 3000; ++i) { std::string txt; for (int k = 0; k < 40; ++k) txt += "1.0 2.0 3.0\n"; tx.send(txt); } std::this_thread::sleep_for(std::chrono::milliseconds(100)); size_t n = 0; for (int k = 0; k < 20000 && k < 1000; ++k) { std::vector<Sample> out; const size_t r = src->read(out, 5000, 1); n += r; if (r == 0) break; } CHECK(n > 0, "nothing read after the flood"); tx.send("5.0 6.0 7.0\n"); std::vector<Sample> out; bool seen = false; for (int k = 0; k < 8000 && !seen; ++k) { out.clear(); src->read(out, 5000, 10); for (auto& s : out) seen |= s.t == 5.0; } CHECK(seen, "the source does not recover after a flood"); }
    if (fails == 0) std::printf("udp robust: OK\n"); return fails ? 1 : 0;
}
#endif
