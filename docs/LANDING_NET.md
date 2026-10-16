# Landing-point network in the engine

`engine->landing_model_load(path, &error)` loads a small network (`cpp/models/landing_mlp_gazebase.bin` or `landing_mlp_drone.bin`, 1.2 MB each; `""` unloads) whose output replaces the tables' landing estimate of an ongoing saccade
(`ForecastSample.lx / ly`, flag `landing_net`; candidates, if any, use it for the amplitude consistency). The tables stay the fallback (no model, no direction yet).

**Network** (cpp/app/engine/include/eyelive/landing_net.hpp, foundation/landing_train.py variant `ctx`): MLP, window of the last 64 samples at 1 kHz x 15 channels (velocity, validity, displacement since the onset, in-saccade flag, time since the onset,
absolute position, position before the onset, cos / sin of the saccade direction and of twice that angle) -> 256 -> 256 -> (dx, dy, remaining duration). **No rotation augmentation**: the dynamics of the eye depend on the starting point and on the direction (muscle anatomy), the
absolute position and the direction are inputs. 274 us per prediction in plain C++ (one core of an M1), parity with PyTorch 3e-6 deg (test_landing_net.cpp, 24 golden windows per model).

**Results** (foundation/landing_pooled.py, foundation/runs/landing_pooled.json; mean landing error in degrees over 0-60 ms after the onset, test participants never seen in training):

| trained on | GazeBase RAN (target steps, large saccades) | drone videos (free viewing, small saccades) |
|---|---|---|
| tables (refit) | 3.89 | 1.75 |
| MLP, GazeBase | **3.12** | 3.82 |
| MLP, drone videos | 4.92 | **1.55** |
| MLP, both | 3.13 | 1.72 |
| GRU, GazeBase (reference, not in C++) | 3.02 | 2.78 |
| GRU, drone videos | 4.46 | 1.39 |
| GRU, both | 3.04 | 1.51 |

In the first 10 ms (drone videos) the GRU gains 22-26 % on the tables and the MLP 14-16 %; on GazeBase the MLP gains 25 % in the first 5 ms (11.07 -> 8.27).
* **A network is specific to its setting** (screen, tracker, saccade amplitudes): a model trained on the other setting is much worse than the tables (3.82 vs 1.75 on the drones, 4.92 vs 3.89 on GazeBase). Pick the file that matches the experiment, or train one on your own data (foundation/landing_gazebase.py, landing_uav_nets.py show how).
* Pooling both settings in one network keeps the GazeBase result but loses most of the gain on the drones (1.72 for the MLP).
* The MLP is 10-12 % behind the GRU; the GRU is not in C++ (11 M multiply-adds per call if the last 128 samples are recomputed at every sample).
* The shipped files are trained on one dataset each with one seed; the drone model on 2 169 saccades of 15 participants. The network is only valid at 1 kHz.
