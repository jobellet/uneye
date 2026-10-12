#include "udp_source.hpp"
#include <cstring>
#include <sstream>

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
#else
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <poll.h>
#include <fcntl.h>
#endif

namespace eyelive {

UdpSource::UdpSource(const SourceSpec& spec) : spec_(spec) {}

UdpSource::~UdpSource() {
    close();
}

bool UdpSource::open(std::string* error) {
#ifdef _WIN32
    WSADATA wsaData;
    if (WSAStartup(MAKEWORD(2, 2), &wsaData) != 0) {
        if (error) *error = "WSAStartup failed";
        return false;
    }
    socket_ = ::socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (socket_ == INVALID_SOCKET) {
        if (error) *error = "socket() failed";
        return false;
    }
    u_long mode = 1;
    ioctlsocket(socket_, FIONBIO, &mode);
#else
    socket_ = ::socket(AF_INET, SOCK_DGRAM, 0);
    if (socket_ < 0) {
        if (error) *error = "socket() failed";
        return false;
    }
    int flags = fcntl(socket_, F_GETFL, 0);
    fcntl(socket_, F_SETFL, flags | O_NONBLOCK);
#endif

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(spec_.port);
    if (inet_pton(AF_INET, spec_.bind_address.c_str(), &addr.sin_addr) <= 0) {
        if (error) *error = "Invalid bind_address";
        return false;
    }

    if (::bind(socket_, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        if (error) *error = "bind() failed";
        return false;
    }

    return true;
}

void UdpSource::close() {
#ifdef _WIN32
    if (socket_ != ~0ull) {
        ::closesocket(socket_);
        socket_ = ~0ull;
        WSACleanup();
    }
#else
    if (socket_ >= 0) {
        ::close(socket_);
        socket_ = -1;
    }
#endif
}

std::string UdpSource::description() const {
    return "udp://" + spec_.bind_address + ":" + std::to_string(spec_.port);
}

size_t UdpSource::read(std::vector<Sample>& out, size_t max_samples, int timeout_ms) {
#ifdef _WIN32
    if (socket_ == ~0ull) return 0;
#else
    if (socket_ < 0) return 0;
#endif

    size_t initial_size = out.size();

#ifdef _WIN32
    fd_set readfds;
    FD_ZERO(&readfds);
    FD_SET(socket_, &readfds);
    struct timeval tv;
    tv.tv_sec = timeout_ms / 1000;
    tv.tv_usec = (timeout_ms % 1000) * 1000;
    int res = select(0, &readfds, nullptr, nullptr, &tv);
#else
    struct pollfd pfd;
    pfd.fd = socket_;
    pfd.events = POLLIN;
    int res = poll(&pfd, 1, timeout_ms);
#endif

    if (res > 0) {
        char buf[65536];
        sockaddr_in src_addr;
        socklen_t addr_len = sizeof(src_addr);
#ifdef _WIN32
        int n = ::recvfrom(socket_, buf, sizeof(buf), 0, (struct sockaddr*)&src_addr, &addr_len);
#else
        int n = ::recvfrom(socket_, buf, sizeof(buf), 0, (struct sockaddr*)&src_addr, &addr_len);
#endif
        if (n > 0) {
            // Text mode check
            if (n > 0 && (buf[0] == '-' || buf[0] == '+' || buf[0] == '.' || (buf[0] >= '0' && buf[0] <= '9'))) {
                std::string line(buf, n);
                std::stringstream ss(line);
                std::string token;
                while (std::getline(ss, token, '\n')) {
                    if (token.empty() || token[0] == '\r') continue;
                    std::stringstream ts(token);
                    Sample s;
                    if (ts >> s.t >> s.x >> s.y) {
                        uint32_t valid = 1;
                        if (ts >> valid) {
                            if (!valid) s.flags = 0;
                        }
                        if (spec_.pixels_per_degree > 0) {
                            s.x /= spec_.pixels_per_degree;
                            s.y /= spec_.pixels_per_degree;
                        }
                        out.push_back(s);
                        if (out.size() - initial_size >= max_samples) break;
                    }
                }
            } else if (n >= 8) {
                // Binary mode
                uint32_t magic;
                uint32_t count;
                std::memcpy(&magic, buf, 4);
                std::memcpy(&count, buf + 4, 4);

                if (magic == kUdpMagic) {
                    if (count > 50) {
                        dropped_++;
                    } else if (n < (int)(8 + count * 20)) {
                        dropped_++;
                    } else {
                        const char* ptr = buf + 8;
                        for (uint32_t i = 0; i < count && out.size() - initial_size < max_samples; ++i) {
                            Sample s;
                            std::memcpy(&s.t, ptr, 8);
                            std::memcpy(&s.x, ptr + 8, 4);
                            std::memcpy(&s.y, ptr + 12, 4);
                            std::memcpy(&s.flags, ptr + 16, 4);
                            if (spec_.pixels_per_degree > 0) {
                                s.x /= spec_.pixels_per_degree;
                                s.y /= spec_.pixels_per_degree;
                            }
                            out.push_back(s);
                            ptr += 20;
                        }
                    }
                } else {
                    dropped_++;
                }
            } else {
                dropped_++;
            }
        }
    }
    return out.size() - initial_size;
}

} // namespace eyelive
