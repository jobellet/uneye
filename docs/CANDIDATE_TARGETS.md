# Candidate targets and saliency peaks for the landing estimate

The application can tell the engine where the eye may go; the engine then picks the most plausible candidate while the saccade is under way.

```cpp
std::vector<eyelive::Candidate> c = {{x1, y1, 1.0f}, {x2, y2, 1.0f}};   // stream units, same frame as the gaze; weight = prior (> 0)
engine->candidates_set(c);                                               // any time; empty list = off
engine->forecast_enable(true);
// ForecastSample: landing (lx, ly) = chosen candidate (or the tables' estimate when none), candidate = index, candidate_confidence = share of the total score
```

**Saliency map.** `eyelive::peaks_from_map(map, w, h, x0, y0, x1, y1, max_peaks, rel_threshold)` (cpp/app/engine/include/eyelive/candidates.hpp) turns a row-major map into weighted candidates (local maxima above `rel_threshold` of the maximum); pass them to `candidates_set`.
The weight multiplies the score, so a salient peak is preferred when the direction and the amplitude fit equally well.
**Not measured:** no saliency map was tested against real fixation data in this repository; the benefit of saliency weights over a flat list is unknown. The measured results below are for a flat list.

**Selector.** For each candidate: angle between its direction (from the position before the onset) and the chord of the saccade so far, tolerance `0.12 rad / (1 + d / 0.5 deg)`; distance against the amplitude predicted by the fixed tables, tolerance 1.0 on the logarithm. Score = weight x both Gaussians; the best candidate wins, confidence = its share of the total.
Parameters tuned on the train split of GazeBase RAN (foundation/landing_candidates_heuristic.py).

**Measured** (GazeBase RAN, 8 000 saccades of 33 held-out participants, distractors drawn from the same screen distribution, mean landing error in degrees over the first 60 ms):

| candidates (1 true) | tables alone | selector, landing = candidate | + amplitude rule (shipped) | true chosen | network trained for it (GRU) |
|---|---|---|---|---|---|
| 1 | 5.24 | 1.88 | – | 100 % | 1.35 |
| 2 | 5.24 | 2.38 | 2.09 | 91.8 % | 1.93 |
| 4 | 5.24 | 3.01 | 2.68 | 80.1 % | 2.42 |
| 8 | 5.24 | 3.69 | 3.29 | 65.4 % | 2.82 |

The engine's shipped tables are worse on GazeBase (large saccades) than the tables refit on the same data (3.89). The selector picks the right candidate as often as the network. The shipped amplitude rule (`landing_from_candidate`: landing = tables + w * (p0 + 0.95 * (candidate - p0) - tables), w = 1 before 30 ms, 0.65 for 30-45 ms, 0.43 after) recovers about half of the gap to the network at no cost (a handful of multiplications per sample).
Test in the engine: `cpp/app/engine/tests/test_engine.cpp` (8b, simulated saccades) and `test_candidates.cpp` (selector, peaks).

## Dynamic saliency from timestamped frames (PQFT)

`eyelive::SaliencyPQFT` (cpp/app/engine/include/eyelive/saliency.hpp, header-only, no dependency): push frames (gray or RGB8, any size, with their time in seconds); `map()` is a 64 x 64 saliency map (maximum 1) that `peaks_from_map` turns into weighted candidates.

```cpp
eyelive::SaliencyPQFT sal;                                   // 64 x 64, blur 3 px, no centre prior (SaliencyPQFT(64, 3.0, 0.3) adds a centre prior)
sal.push(rgb, width, height, 3, t_seconds);                  // at the video rate (25-60 Hz), not at the gaze rate
auto c = eyelive::peaks_from_map(sal.map().data(), 64, 64, x_left, y_top, x_right, y_bottom, 8, 0.4);   // gaze units; y_top = value of the first row
engine->candidates_set(c);
```

Method: phase spectrum of the quaternion Fourier transform (Guo, Ma & Zhang, CVPR 2008) of (motion, red-green, blue-yellow, intensity). The motion channel is the absolute difference with the previous frame, scaled so that 33 ms is the reference interval (so 60 Hz and 24 Hz videos give comparable maps); a gap above 0.5 s starts a new stream.
Cost: 0.7 ms per 640 x 480 RGB frame on the development machine (most of it is the resampling to 64 x 64), i.e. 2 % of one core at 30 Hz. Tests: `cpp/app/engine/tests/test_saliency.cpp` (static pop-out, motion pop-out, timestamps, speed, link with the candidates).

**Not measured:** whether saliency weights improve the landing estimate on real videos. Plan (needs a video database with gaze at 1 kHz, e.g. DIEM; not downloaded): for each saccade, candidates = saliency peaks of the frame shown at the onset; compare with the real landing point (tables alone, flat candidate list, weighted by saliency).
