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
