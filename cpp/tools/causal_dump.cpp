// causal_dump: Step 2 helper. Streams sequences of (dx, dy) through the C++ causal TCN (StepEngine, one step per sample, reset per
// sequence) and writes the saccade probability of every step, so that cpp/scripts/equivalence.py can compare it with PyTorch.
// Input file : int32 n_seq, int32 T, then n_seq * T * 2 float32 (dx0, dy0, dx1, dy1, ...), little endian.
// Output file: int32 n_seq, int32 T, then n_seq * T float32 p(saccade).
// Usage: causal_dump models/causal.bin in.bin out.bin
#include <cstdint>
#include <cstdio>
#include <vector>

#include "uneye/causal_engine.hpp"

int main(int argc, char** argv) {
    if (argc != 4) { std::fprintf(stderr, "usage: causal_dump model.bin in.bin out.bin\n"); return 2; }
    auto net = uneye::make_causal_engine(argv[1]);
    std::FILE* f = std::fopen(argv[2], "rb");
    if (!f) { std::fprintf(stderr, "cannot open %s\n", argv[2]); return 1; }
    int32_t hdr[2];
    if (std::fread(hdr, sizeof(int32_t), 2, f) != 2) return 1;
    const size_t n = static_cast<size_t>(hdr[0]), T = static_cast<size_t>(hdr[1]);
    std::vector<float> in(n * T * 2), out(n * T);
    if (std::fread(in.data(), sizeof(float), in.size(), f) != in.size()) { std::fprintf(stderr, "short input\n"); return 1; }
    std::fclose(f);
    float p[2];
    for (size_t s = 0; s < n; ++s) {
        net->reset();
        for (size_t t = 0; t < T; ++t) {
            net->step(in[(s * T + t) * 2], in[(s * T + t) * 2 + 1], p);
            out[s * T + t] = p[1];
        }
    }
    std::FILE* g = std::fopen(argv[3], "wb");
    if (!g) return 1;
    std::fwrite(hdr, sizeof(int32_t), 2, g);
    std::fwrite(out.data(), sizeof(float), out.size(), g);
    std::fclose(g);
    return 0;
}
