# Labeler behaviour spec (web labeler = reference; the Android app should match)

Implemented in `app.js` / `index.html` (web) and checked by `test_labeler_flow.mjs` (headless Chrome).

1. **Refractory period after a pinch zoom.** Once a second finger touched the screen during a gesture, nothing is marked when the
   fingers lift, and any new touch within `TOUCH_REFRACTORY_MS` (600 ms) of the last multi-touch release is ignored (no tap, no drag).
2. **Pass 1 windows.** The visible window has a user-chosen width in ms (`winMs`, default 500, remembered, also asked in the load dialog).
   "Next window" moves to the next window of the same trial with a 10 % overlap (step = 0.9 x width, last window clipped to the trial end);
   after the last window of a trial it goes to the first window of the next trial. "Prev window" is the inverse.
3. **Only windows that were seen can be trained on.** A window is "seen" when it was left with Next (or when switching to pass 2 with it on
   screen). Reviewed data = seen windows that contain no point mark still waiting for its refinement in pass 2. Nothing else is ever
   passed to the trainer, the test-set scoring or the exports (`-1` = not reviewed).
4. **Training / export labels are the human's refined events only.** Automatic pre-labels in a reviewed window do NOT count as saccades
   (a reviewed window without an event is a true "no event" window).
5. **Pass 2 has no Next Trial button.** Next / Prev event walks through the events of all trials in order; after the last event of a trial the
   next one is the first event of the next trial that has events (and the other way round). Only marks lying in seen windows are offered.
6. **Eraser tool.** With the eraser on, dragging between two times (or tapping) deletes every refined event (saccade or blink), point mark
   and automatic pre-label overlapping that interval. Undoable.
7. **Save / load labels.** `uneye_labels_YYYYMMDD-HHMM.json` (`format: "uneye-labels-v1"`): per trial `scanWins`, `expertEvents`, `pointMarks`,
   plus `trialNames`, `trialLengths`, `rateHz`, `winMs`. Loading matches trials by name (else by index, lengths must be equal).
8. **1-degree scale bar.** A black 1-degree bar at the right edge of the X and Y lanes; the automatic vertical zoom never goes below
   1.25 degrees of range so the bar always fits.
9. **Trainer memory.** The batch is capped at 32 windows (`cpp/web/uneye_web.cpp`) and `max_seconds_of_data` is a total budget over all traces
   (`cpp/app/train/src/dataset.cpp`); the page trains on all reviewed labels.
10. **Screen space.** The top bar keeps only the annotation tools (class / eraser / undo, and the pass switch on Android); files, labels, filter,
    training, metrics, export and the window width are behind a menu button (hidden by default). One status line.
11. **One position lane.** X (blue) and Y (green) share the same axes, scaled on the extreme values of the visible window
    (4 % margin, never tighter than 1.25 degrees); the speed lane gets the remaining third.
12. **No simulated data at start.** The page opens on a file window (`startModal`), or resumes the autosaved session; `generateDemoDataset()`
    exists for the automated tests only (no button).
13. **Suggestions.** Default: the label-free universal HMM + acceleration-zero boundaries (`hmm_prelabel.js`, identical to the C++ `hmm_accel`
    batch detector, `test_hmm_prelabel.js`), blinks = NaN runs. Never the pretrained network. Once a network has been trained, the menu toggle
    *Network suggestions* (default On, remembered) shows its proposals; Off = HMM. The load dialog offers HMM or None.
14. **Automatic training only after a pass 2.** Training starts by itself when pass 2 is left (or its last event is reached) and at least
    `minNewSecondsForAutoTrain` (2 s) were reviewed since the last training; never during pass 1 (point marks are refined in pass 2, so only then are
    the labels complete). *Train* in the menu still starts it at any time.
15. **Machine detection in the file** (`pred_labels`, `prob`; `formats.js`, `nearmiss.js`). `pred_labels` replace the HMM as the proposal source
    `'file'` (blinks still = NaN runs); `prob` alone is thresholded with the U'n'Eye post-processing (> 0.5, gaps < 5 ms merged, events < 5 ms
    dropped). Neither ever counts as labels or as reviewed. `prob` is drawn in every pass in a band under the speed (0..1, dashed line at the
    threshold). It is autosaved with the traces as Uint16 (error < 2e-5); `pred_labels` as their events.
16. **Near misses** (button in pass 2 and in the menu, only with `prob`). Queue of all trials, sorted by decreasing peak: a run of `prob >= lower`
    (default 0.1, menu, remembered; NaN breaks a run) whose peak is below the threshold and that overlaps no expert event, saccade proposal or
    rejected near miss; bounds where `prob` falls below half the peak (inside the run), centre = the peak. Shown like a pass-2 event (±50 ms,
    Wider, Prev / Next, "Near miss k/N (p = 0.43)"). Dragging a bound handle moves it; *Accept*, a double tap or a press-drag store an expert
    saccade (as a pass-2 refinement); *Not an Event* keeps it in `rejectedNearMisses` (saved), and the window on screen joins the seen windows
    (reviewed as fixation). Leaving (button again) returns to the normal pass 2; Switch Pass leaves it too.
17. **Viewed periods** (menu). Every window drawn on screen, in any mode, is recorded per trial (`viewedRanges`, merged; autosaved, in the labels
    file, old files without it still load). The window shows one row per trial (time from the trial start, all trials on the scale of the longest):
    not viewed / viewed / reviewed / saccade / blink / rejected near miss, the totals (% of samples viewed, trials fully viewed), the trial on hover;
    a click opens that trial in pass 1 at the window holding that time. Drawn from a one-pixel-per-trial offscreen image (1289 trials: ~60 ms).
