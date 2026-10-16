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

## Reading (GazeBase TEX, one poem page, 1 kHz): what works and what does not

Saccades are small to the right (median +3.7 deg) and large to the left (median -10.2 deg, return sweeps); the tables alone are already good at 20 ms (2.07 deg) because the saccades are short.
Test on 42 000 saccades of held-out participants (foundation/landing_tex_saliency.py, landing_tex_prior.py), error in degrees at 10 / 20 ms after the onset:

| estimate | 10 ms | 20 ms |
|---|---|---|
| tables alone | 4.09 | **2.07** |
| PQFT saliency peaks of the page (N = 16), forced choice | 5.84 | 8.15 |
| word centres (all words of the page), flat weights | 3.92 | 3.88 |
| word centres + prior on the saccade vector (2-D histogram of the train participants) | 3.34 | 3.46 |
| same, weight w toward the candidate = 0.75 (0.5) | **3.30** (3.41) | - |
| same, w = 0.25 | 3.66 | 2.22 |

* A saliency map of a text page does not find words (8 % of the landings within 1 deg of a peak, 77 % for word centres); forced choice among such candidates is worse than the tables.
* With word candidates and a prior the first 10-15 ms improve by about 18 %; later the tables are better (the candidate loses the precision of the eye position). **Use `candidates_set(c, use_until_ms)` with about 12 ms for reading.**
* The prior depends on the position before the saccade: the application re-weights the candidates at each fixation (`weight` = prior of the vector candidate - current fixation) and calls `candidates_set` again.
* A precision-weighted blend (weight from the tables' uncertainty proportional to the amplitude, candidate noise in degrees) was tried and was worse than the time-based rule on GazeBase RAN (2.64 / 3.05 / 3.57 deg for 2 / 4 / 8 candidates against 2.09 / 2.68 / 3.29). Not kept.
* A gate that refuses a candidate far from the tables' landing keeps the tables' error on reading (4.01 / 2.02 deg) but removes the gain on RAN (2.0 -> 3.9+ deg), because the tables are poor for large saccades there. Not kept; the engine always chooses unless `min_confidence` is set.

## Real videos (EyeTrackUAV2 + DTB70, 2026-10-16): saliency candidates make things worse

9 DTB70 drone videos (1280 x 720 shown at native size, 20.1 x 11.3 deg), 30 participants in free viewing, gaze at 1 kHz; 4 826 saccades detected by a speed threshold (no human labels).
Mean landing error (deg) at 10 / 20 ms after the onset (foundation/landing_uav_saliency.py, foundation/runs/landing_uav_saliency.json):

| estimate | 10 ms | 20 ms | candidate within 1 / 2 deg of the landing |
|---|---|---|---|
| tables alone | **1.67** | **0.90** | - |
| PQFT saliency peaks (16), forced choice | 7.33 | 7.50 | 9 % / 27 % |
| tracked object's centre (DTB70 ground truth) | 5.35 | 5.67 | 12 % / 29 % |
| object + saliency peaks | 5.41 | 5.42 | - |
| 16 random points in the frame (control) | 5.05 | 5.33 | 19 % / 57 % |

* The saliency peaks land near the eye less often than random points (they cluster on a few salient spots); in free viewing of these videos the eye does not go to them.
* Even the object being tracked in the video is reached by only 12 % of the saccades within 1 deg.
* The saccades are small here, and the tables alone are already good (0.9 deg at 20 ms); any forced choice among candidates that do not contain the landing point is much worse.
* **Conclusion: do not enable `saliency` for natural scenes / videos.** Candidates are useful only when the possible targets are known and few (target steps, buttons, words with a prior and `use_until_ms` about 12 ms). The saliency path stays in the engine as an option, off by default.
