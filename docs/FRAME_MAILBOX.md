# Frame mailbox: sharing the stimulus with the engine

A stimulus script (Psychtoolbox, PsychoPy, any program) writes small frames and / or candidate targets into one memory-mapped file; the engine polls it and uses it for the landing estimate (docs/CANDIDATE_TARGETS.md).
Why not read the screen textures: they live in another process (own OpenGL context); an OS-level screen capture needs permissions, adds 5-15 ms and does not know the flip time; `Screen('GetImage')` of a large window stalls the GPU.
The script already knows what it draws: it sends that array (or a 64 x 64 copy of the screen drawn in an off-screen window, read back at a negligible cost).

**File format**: see the header of `cpp/app/engine/include/eyelive/frame_mailbox.hpp` (little endian, seqlock per slot and for the candidates, a ring of >= 2 frame slots, 64 candidates).

**Engine side**
```cpp
eyelive::StimulusFeedConfig cfg; cfg.path = "/tmp/stim.frames"; cfg.saliency = true; cfg.max_peaks = 16; cfg.use_until_ms = 12;   // saliency peaks -> candidates
engine->stimulus_feed_open(cfg, &error);          // 5 ms polling thread, independent of start / stop
auto st = engine->stimulus_feed_status();         // frames, candidate_updates, candidates, explicit_candidates, last_frame_age_s
```
Explicit candidates written by the script have priority over the saliency peaks; an empty explicit list gives the control back to the saliency. The frame edges are given in gaze units in the file header (`gaze_rect`), so the peaks come out in the frame of the gaze.

**Writers** (all write the same file; only the C++ and Python ones are tested)
* C++: `eyelive::FrameMailboxWriter` (frame_mailbox.hpp).
* Python, standard library only: `cpp/app/tools/eyelive_frames.py` (`FrameWriter`); PsychoPy example: `cpp/app/tools/psychopy_feed_demo.py` (run with PsychoPy 2026.2.4 on macOS arm64, see below).
* MATLAB / Octave / Psychtoolbox: `cpp/app/tools/eyelive_frames.m` (NOT TESTED; MATLAB is not available here).

**Tests** (`cpp/app/engine/tests/test_mailbox.cpp`, test 8c of `test_engine.cpp`): round trip; ring wrap-around; a writer running flat out against a reader (714 000 frames read, 0 torn); bad files; the Python writer read by the C++ reader; the engine turning a frame into saliency candidates and explicit candidates taking priority.

**Limits of this first version**
* The engine applies the newest list at the moment it polls (about 5 ms after the script wrote it); the frame timestamp `t_writer_s` is stored but not yet matched to the gaze clock. For a video, the list lags the display by the polling delay plus the time between the flip and the push. A clock handshake (the engine publishing its host time) is the next step.
* Python / MATLAB writers cannot issue memory barriers; the seqlock makes a torn read detectable, but on weakly ordered CPUs (ARM) a rare inconsistent frame cannot be excluded for them. The effect would be one wrong saliency map.
* Windows: the mapping code is written but never compiled or run.

## Run with PsychoPy (2026-10-16, macOS arm64, Python 3.12)
`eyelive_feed_probe /tmp/stim.frames 14 --saliency` (the engine side, cpp/app/cli/tools/feed_probe.cpp) in one terminal, `python cpp/app/tools/psychopy_feed_demo.py` in another. Result: 120 frames pushed after each `win.flip()` and received by the engine (about 40 per second, the demo window's own rate),
4 saliency peaks computed from the first frame (first peak at (14.7, 11.0) deg for a square centred at (11.3, 8.4): within 4 deg, the map is 64 x 64 over 40 x 30 deg), then at frame 60 the explicit candidates of the script (2) replaced them (`explicit=1`).
Installing PsychoPy where there is no compiler: `uv pip install psychopy` fails (it builds `psychtoolbox`); use `--no-deps` and add `pyglet==1.5.27 pyserial pyobjc-framework-Cocoa pyobjc-framework-Quartz python-bidi arabic-reshaper freetype-py json-tricks i18next` (plus numpy / pandas / scipy / pillow / pyyaml / requests as asked by the import errors). `units='deg'` needs a `monitors.Monitor` with width, distance and `setSizePix`.
The timing of the flip (`t_writer_s`) is stored in the file but not yet matched to the gaze clock.
