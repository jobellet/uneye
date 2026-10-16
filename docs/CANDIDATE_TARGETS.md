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

### Other saliency models on the same videos (foundation/saliency_models.py, landing_uav_saliency2.py, runs/landing_uav_saliency2.json)

NSS = saliency at the landing point in standard deviations of the map (0 = chance). Errors in degrees at 10 / 20 ms; tables alone: **1.67 / 0.90**; dense 32 x 18 grid of candidates with flat weights (no saliency): 2.23 / 2.17.

| model | Python ms / frame | NSS | a peak within 1 / 2 deg | 16 peaks as candidates | grid weighted by the map |
|---|---|---|---|---|---|
| PQFT (the engine's) | 7.3 | **0.73** | 23 / 57 % | 5.52 / 5.79 | 2.39 / 2.60 |
| spectral residual | 7.6 | 0.55 | 23 / 57 % | 5.64 / 5.95 | 2.49 / 2.67 |
| Itti-Koch-Niebur 1998 (static) | 55.9 | 0.42 | 24 / 66 % | 4.92 / 5.11 | 2.31 / 2.37 |
| Itti dynamic (+ flicker, Reichardt motion) | 66.1 | 0.65 | 27 / 69 % | 4.96 / 5.12 | 2.28 / 2.37 |
| frame difference only | 7.1 | 0.28 | 24 / 60 % | 5.58 / 5.90 | 2.27 / 2.39 |
| centre bias (Gaussian) | 6.5 | 0.61 | 6 / 17 % | 4.83 / 4.92 | **2.19 / 2.18** |

* Every map predicts the landing point a little better than chance (NSS 0.3-0.7), but a plain centre bias does almost as well as the best (0.61 against 0.73); the dynamic Itti model is not better than PQFT and is 9 times slower.
* As a prior on a dense grid, no map beats the flat grid, and nothing beats the tables alone: in this viewing (small saccades, 20 x 11 deg frames) the kinematics already say more than any of these maps.
* The PQFT peak coverage depends on the resampling (box filter in landing_uav_saliency.py: 9 % within 1 deg; bilinear here: 23 %); 16 random points reach 19 %.

### Camera-motion compensation (owner's remark: the drone's ego-motion creates optical-flow contrast that the subjects ignore)

`global_motion` / `compensated_diff` (foundation/saliency_models.py): global affine motion between two frames (direct method, 3-level pyramid, Huber weights; exact for shifts up to about 5 px at 320 px width), the previous frame is warped before every temporal difference. Models `motion_c`, `pqft_c`, `itti_dyn_c`.
Same 4 826 saccades (runs/landing_uav_saliency2_comp.json): NSS motion 0.28 -> 0.33, PQFT 0.73 -> 0.72, Itti dynamic 0.65 -> 0.62; landing error with the 16 peaks or with a weighted grid unchanged (about 5.0-5.6 and 2.3-2.4 deg, tables alone 1.67 / 0.90).
NSS of the frame-difference map split by the size of the camera motion at the onset frame (foundation/uav_camera_motion_nss.py): small (mean 0.17 px / frame) 0.35 -> 0.41; medium (0.7 px) 0.17 -> 0.26; **large (10.9 px) 0.34 -> 0.31**. In most of the frames the camera hardly moves (terciles 0.3 and 1.4 px at full resolution),
and where it moves a lot the compensated map is not better at predicting the landing point. The figure docs/figs_saliency/uav_saliency_Soccer1_subj10_cam.png shows the 1 s window with the largest camera motion (16.7 px per frame): the compensation removes part of the response of the building edges and keeps the players, where the subject looks, but it is a single example.

### PQFT saliency x Gaussian centred on the pre-saccadic eye position (foundation/landing_uav_proximity.py, runs/landing_uav_proximity.json)

Weight of each point of a dense 32 x 18 candidate grid = saliency^g * exp(-|c - p0|^2 / (2 s^2)), p0 = eye position before the saccade; s (1-12 deg or infinite) tuned on the even-numbered saccades, errors in degrees (10 / 20 ms) on the odd ones (2 413 saccades).
Tables alone 1.67 / 0.90.

| use of the weights | g = 0 (proximity only) | g = 1 (saliency x proximity) | g = 2 |
|---|---|---|---|
| prior of the engine's selector (direction / amplitude consistency) | 2.19 / 2.19 (s = infinite) | 2.18 / 2.28 (s = 8 deg) | 2.25 / 2.30 (s = 5 deg) |
| arg-max of the weighted map alone | 4.12 / 4.96 (s = 3 deg) | 4.15 / 5.03 (s = 1 deg) | 4.17 / 5.05 (s = 1 deg) |

* The proximity Gaussian improves the saliency map used alone (2.43 / 2.57 -> 2.29 / 2.15 on the train half for g = 1, s = 8 deg instead of infinite) but never beats a flat grid (2.27 / 2.15), and nothing beats the tables.
* With the saliency, the tuned width is 8 deg, comparable with the saccade amplitudes in these videos; with proximity alone the best width is infinite: once the kinematics have given the direction and part of the amplitude, a prior around the starting point adds nothing.
* The arg-max of the weighted map without kinematics is much worse (4.1 / 5.0 deg): the map locates a possible target, not the next fixation.
