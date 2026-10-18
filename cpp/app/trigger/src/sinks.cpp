#include "sinks.hpp"
#include <cstdio>
#include <cstring>
#include <mutex>

#ifdef _WIN32
// Windows branch: Winsock written against the documentation, NOT built or run in CI yet (see docs/tasks/status/T19.md).
#define WIN32_LEAN_AND_MEAN
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
#include <ws2tcpip.h>
using sock_t = SOCKET; static const sock_t kBadSock = INVALID_SOCKET;
static void close_sock(sock_t s) { closesocket(s); }
static void set_nonblocking(sock_t s) { u_long on = 1; ioctlsocket(s, FIONBIO, &on); }
struct WsaInit { WsaInit() { WSADATA d; WSAStartup(MAKEWORD(2, 2), &d); } ~WsaInit() { WSACleanup(); } };
#else
#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#include <cstdlib>
using sock_t = int; static const sock_t kBadSock = -1;
static void close_sock(sock_t s) { ::close(s); }
static void set_nonblocking(sock_t s) { const int fl = fcntl(s, F_GETFL, 0); fcntl(s, F_SETFL, fl | O_NONBLOCK); }
#endif

namespace eyelive {
namespace {

class UdpSink final : public Sink {
 public:
    UdpSink(std::string host, int port) : host_(std::move(host)), port_(port) {}
    ~UdpSink() override { if (sock_ != kBadSock) close_sock(sock_); }
    bool open(std::string* error) override {
        if (sock_ != kBadSock) return true;
#ifdef _WIN32
        static WsaInit wsa; (void)wsa;
#endif
        std::memset(&addr_, 0, sizeof addr_); addr_.sin_family = AF_INET; addr_.sin_port = htons(uint16_t(port_));
        if (inet_pton(AF_INET, host_ == "localhost" ? "127.0.0.1" : host_.c_str(), &addr_.sin_addr) != 1) { if (error) *error = "udp sink: '" + host_ + "' is not an IPv4 address (use a dotted address or localhost)"; return false; }
        sock_ = ::socket(AF_INET, SOCK_DGRAM, 0); if (sock_ == kBadSock) { if (error) *error = "udp sink: cannot create a socket"; return false; }
        set_nonblocking(sock_); return true;
    }
    void send(const char* data, size_t n) override {
        if (sock_ == kBadSock) { ++errors; return; }
        const auto r = ::sendto(sock_, data, int(n), 0, reinterpret_cast<const sockaddr*>(&addr_), sizeof addr_);
        if (r < 0 || size_t(r) != n) ++errors; else ++sent;
    }
 private:
    std::string host_; int port_; sock_t sock_ = kBadSock; sockaddr_in addr_{};
};

class LogSink final : public Sink {
 public:
    explicit LogSink(std::string path) : path_(std::move(path)) {}
    ~LogSink() override { if (f_) std::fclose(f_); }
    bool open(std::string* error) override { if (f_) return true; f_ = std::fopen(path_.c_str(), "a"); if (!f_) { if (error) *error = "log sink: cannot open " + path_; return false; } return true; }
    void send(const char* data, size_t n) override {
        std::lock_guard<std::mutex> l(mu_); if (!f_) { ++errors; return; }
        const bool ok = std::fwrite(data, 1, n, f_) == n && std::fputc('\n', f_) != EOF && std::fflush(f_) == 0; if (ok) ++sent; else ++errors;
    }
 private:
    std::string path_; std::FILE* f_ = nullptr; std::mutex mu_;
};

class StdoutSink final : public Sink {
 public:
    bool open(std::string*) override { return true; }
    void send(const char* data, size_t n) override { std::fwrite(data, 1, n, stdout); std::fputc('\n', stdout); std::fflush(stdout); ++sent; }
};

class NullSink final : public Sink {                          // counts firings, sends nothing (tests, dry runs)
 public:
    bool open(std::string*) override { return true; }
    void send(const char*, size_t) override { ++sent; }
};

}  // namespace

std::shared_ptr<Sink> make_sink(const std::string& spec, std::string* error) {
    if (spec == "stdout") return std::make_shared<StdoutSink>();
    if (spec == "null") return std::make_shared<NullSink>();
    if (spec.rfind("log:", 0) == 0) { if (spec.size() == 4) { if (error) *error = "log sink: no path"; return nullptr; } return std::make_shared<LogSink>(spec.substr(4)); }
    if (spec.rfind("udp:", 0) == 0) {
        const std::string rest = spec.substr(4); const size_t c = rest.rfind(':'); if (c == std::string::npos || c == 0) { if (error) *error = "udp sink: expected udp:<host>:<port>"; return nullptr; }
        char* end = nullptr; const long port = std::strtol(rest.c_str() + c + 1, &end, 10); if (*end != '\0' || port < 1 || port > 65535) { if (error) *error = "udp sink: bad port in '" + spec + "'"; return nullptr; }
        return std::make_shared<UdpSink>(rest.substr(0, c), int(port));
    }
    if (error) *error = "unknown sink '" + spec + "' (use udp:<host>:<port>, log:<path> or stdout)"; return nullptr;
}

}  // namespace eyelive
