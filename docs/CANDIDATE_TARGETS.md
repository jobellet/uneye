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

| candidates (1 true) | tables alone | selector | true chosen | network trained for it (GRU) |
|---|---|---|---|---|
| 1 | 5.24 | 1.88 | 100 % | 1.35 |
| 2 | 5.24 | 2.38 | 91.8 % | 1.93 |
| 4 | 5.24 | 3.01 | 80.1 % | 2.42 |
| 8 | 5.24 | 3.69 | 65.4 % | 2.82 |

The engine's shipped tables are worse on GazeBase (large saccades) than the tables refit on the same data (3.89). The selector picks the right candidate as often as the network but lands exactly on the candidate, which loses the network's amplitude correction (about 0.5 deg).
Test in the engine: `cpp/app/engine/tests/test_engine.cpp` (8b, simulated saccades) and `test_candidates.cpp` (selector, peaks).
