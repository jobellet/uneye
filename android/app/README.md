# U'n'Eye offline labeler — Android app (phase 2)

Touch-optimized, offline, manual saccade labeling on the phone. The labels are
owned by the C++ `ExpertLabels` (frozen contract); Kotlin only draws and
handles gestures. Room stores only the file list.

## Build

```
cd android
./gradlew :app:assembleRelease          # needs the Android SDK + NDK (arm64-v8a)
```

The native side (`../CMakeLists.txt`) builds the frozen `cpp/` tree plus
`offline_session/` and the JNI bridge (`liboffline_session.so`).

Host-side checks (no SDK needed):

```
cmake -S android -B build-host -DCMAKE_BUILD_TYPE=Release
cmake --build build-host --target test_offline_session -j
ctest --test-dir build-host -R test_offline_session
```

## Structure

```
app/src/main/java/com/uneye/offline/
  Native.kt          JNI mirror of the OfflineSession facade (37 functions)
  Session.kt         labeling state machine (pass 1 / pass 2), UI state only
  TraceView.kt       touch Canvas: 3 lanes (X, Y, speed), marks, loupe, zoom/pan
  TraceFiles.kt      CSV pair reader (UnEye convention) + MATLAB v5 reader
  FileListDb.kt      Room: the file list only
  MainActivity.kt    file list + load dialog (rate, units, pre-labeler)
  LabelingScreen.kt  the two-pass labeling UX
  UneyeApp.kt        Application: owns the Session, app file layout
app/src/main/assets/models/
  combined.bin/.json  shipped pre-trained model (U-Net 2 classes, window 200)
```

## The two passes

**Pass 1 — mark (tap + auto-scroll).** A class toggle (saccade / blink) stays
visible. One tap places a point mark; a tap on a mark removes it; a tap near a
pre-label snaps to it. After each mark the view auto-scrolls so the next epoch
starts just after the last marked event (+30 ms saccade, +300 ms blink). A
large "Next" button moves to the next epoch. Undo is always visible.

**Pass 2 — refine (press-drag-lift).** For each mark, in order, the view
centers a window around it (±50 ms saccade, ±300 ms blink, widened when the
pre-label is longer). Press = onset, drag, lift = offset; the app moves to the
next mark at once. A double tap accepts the pre-label boundaries. Buttons:
"not an event" (deletes), "back", "wider". The velocity lane is always shown
and a loupe follows the finger.

Data rules (enforced by the facade): point marks never enter `ExpertLabels`
and never train; `ExpertLabels::add` happens only at pass-2 refinement; an
epoch is reviewed only when scanned AND all its marks are refined.

## Formats

- CSV: two files (X, Y), each a trials × time matrix or a single column
  (cut into 1 s trials by the facade), comma or whitespace separated,
  `nan`/`inf`/empty = lost sample.
- `.mat`: MATLAB v5, variables `X` and `Y` (what `io.loadmat` reads).
- The sampling rate and the units (deg per unit) are asked at load; the HMM
  has parameters for 500/1000 Hz, everything else is resampled to 1 kHz.
- Pre-labeler: HMM (default), the shipped `combined.bin`, the last trained
  model, or none. After the first training the user's model becomes the
  default for new data (phase 3).

## Phase 3+4: background training & convergence

- `TrainingWorker` (WorkManager): runs the C++ training cycle via the facade
  on a background worker (foreground notification, dataSync type), saves the
  model, swaps the pre-labeler to it, re-prelabels the unreviewed parts
  (`apply_auto_labels`: expert events and reviewed ranges are never touched),
  appends a round to the convergence history. Triggered by
  `TrainingScheduler` after every 20 newly labeled seconds (charging
  constraint only above 120 s reviewed) or manually ("Train now").
- `ConvergenceStore`: JSON history of rounds (labeled seconds, epochs, biased
  val F1/kappa for reference, test F1/kappa per class, expert event counts).
- `ConvergenceFlag`: the flag rule — SILENT until the fixed test set has
  >= 100 saccades and >= 20 blinks, then CONVERGED when the mean gain of the
  saccade test F1 over the last 3 rounds is inside its bootstrap 95% CI
  (2000 resamples). The trainer's val F1 is biased and never used.
- `MetricsView`: learning curves — test F1 and kappa per class vs reviewed
  seconds, with a marker where the flag fired. Shown in a dialog ("Metrics").
