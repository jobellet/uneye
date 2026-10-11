// T09b: seconds per epoch for 50 s of data with 4 rotations (200 training windows + 10 validation windows of 1000 samples), 1 thread and 4 threads; and the number of parameters. Limits are loose (CI machines differ); the numbers go to docs/tasks/status/T09.md.
#include <chrono>
#include <cstdio>
#include "eyelive/train/unet.hpp"
#include "eyelive/trainer.hpp"
#include "synth.hpp"
#include <thread>
using namespace eyelive;
static double per_epoch(int threads) {
    TrainConfig c; c.n_classes = 3; c.max_epochs = 3; c.n_threads = threads; auto t = make_trainer(); int epochs = 0; double t_first = 0, t_last = 0;
    t->start({synth_trace(50.0, 1000.0, 11, true)}, c, [&](const TrainProgress& p) { if (!p.finished) { ++epochs; if (epochs == 1) t_first = p.elapsed_s; t_last = p.elapsed_s; } });
    while (t->running()) std::this_thread::sleep_for(std::chrono::milliseconds(10));
    return epochs > 1 ? (t_last - t_first) / (epochs - 1) : t_first;
}
int main() {
    train::UNet<float> n2(2), n3(3), n5(5); std::printf("parameters: 2 classes %zu, 3 classes %zu, 5 classes %zu\n", n2.n_params(), n3.n_params(), n5.n_params());
    const double s1 = per_epoch(1), s4 = per_epoch(4); std::printf("seconds per epoch (50 s of data, 4 rotations): 1 thread %.2f, 4 threads %.2f\n", s1, s4);
    if (s1 > 10.0 || s4 > 3.0) { std::printf("FAIL: too slow\n"); return 1; } return 0;
}
