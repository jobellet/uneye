# Laboratory test of 2026-10-12: record real EyeLink data, analyse them here

Goal: check on REAL pixel data (a) that the detection works, (b) that the automatic unit-scale selection finds the right scale, (c) what the saccades / microsaccades look like at your tracker's rate and noise. Nothing has to be installed on the laboratory computer: you record with your usual MATLAB / Psychtoolbox script, convert the EDF to text, and bring (or send) the text file. About 6 minutes of recording.

## 1. Write down (needed to check the automatic scale against the truth)
* tracker model and sampling rate (e.g. EyeLink 1000 Plus, 1000 Hz), which eye is tracked, head fixed or free
* screen resolution in pixels (width x height) — the one the Psychtoolbox window uses
* **width of the visible screen area in cm** (measure the image, not the casing) and **eye-to-screen distance in cm**
* the stimulus used for each block below, with its size in degrees

## 2. What to record (one EDF with markers, or one EDF per block)
Send a message at the start of each block with `Eyelink('Message', 'BLOCK_<name>_START')` so that the blocks can be told apart (optional: the analysis does not need it). The recording settings your script already uses are fine as long as the file contains **samples and events with gaze in pixels**. In the Psychtoolbox `EyelinkExample` these are the standard lines (keep what your script has):
```
Eyelink('command', 'file_sample_data  = LEFT,RIGHT,GAZE,GAZERES,AREA,STATUS,INPUT');
Eyelink('command', 'file_event_filter = LEFT,RIGHT,FIXATION,SACCADE,BLINK,MESSAGE,BUTTON,INPUT');
Eyelink('command', 'file_event_data   = GAZE,GAZERES,AREA,VELOCITY');
Eyelink('Message', 'DISPLAY_COORDS %ld %ld %ld %ld', 0, 0, width-1, height-1);     % screen size in the file
```
| block | what | duration | what it tells us |
|---|---|---|---|
| A | steady fixation of a small central dot | 60 s | noise level, microsaccades |
| B | free viewing of natural images or a video (nothing to do) | 120 s | saccades of 2-15 degrees, the main sequence |
| C | fixation with about 10 deliberate blinks | 30 s | blink handling |
| D | **target jumps** of a known size: a dot alternating between two positions 10 degrees apart horizontally, 1.5 s at each, 20 jumps (and if possible a second series of 5 degrees) | 40-60 s | saccades of KNOWN amplitude: checks the scale independently of any detector |
| E (optional) | smooth pursuit of a dot moving sinusoidally (about 0.3 Hz, +-10 degrees) | 30 s | pursuit stays in the fixation class: do we mislabel it? |
Calibrate the tracker as usual before the recording.

## 3. Convert and bring
1. Convert the EDF to text with the EyeLink Developers Kit command `edf2asc yourfile.edf` (or the "Visual EDF2ASC" application): this writes `yourfile.asc` with samples and events. Do NOT use options that remove the samples.
2. Put the `.asc` file(s) in `data/lab/` on the analysis computer (that folder is ignored by git: the recordings stay private), together with the numbers of section 1 in a text file `data/lab/NOTES.txt`.
3. Analysis (done here, nothing for you to run):
```
.venv/bin/python foundation/lab_analyze.py data/lab/yourfile.asc --width-cm <W> --distance-cm <D> --res-x <pixels>
```
The report (`data/lab/report_yourfile/report.md` and figures) gives: sampling rate and valid-sample rate, the automatic scale in degrees per pixel and in pixels per degree against the geometry, the number and the statistics of saccades (amplitude, duration, peak speed, microsaccade fraction), and the agreement with EyeLink's own saccade events (`ESACC` lines).

## 4. What would count as a success
* the automatic scale within 40 % of the geometry value (the benchmark tests did within 40 %, but only with data in degrees multiplied by a factor, never with real pixel data: this is the first real test);
* in block D, the detected saccade amplitudes cluster around the jump size (10 and 5 degrees) with the scale of the geometry;
* in block A, a plausible microsaccade rate (about 0.5-2 per second) at amplitudes below 1 degree, and no burst of events from noise alone;
* clear disagreements with EyeLink's events are expected (different thresholds) and are not a failure by themselves: we look at what each detector misses and why.
If the automatic scale is wrong, the report still gives the geometry-based result and the agreement curve, which shows whether the criterion was ambiguous.

## 5. A live test (UDP, the application) is NOT ready for tomorrow
The application's modules are being written; a live demonstration needs the engine, the recorder and the interface to be merged. Recording and analysing a file tomorrow does not depend on any of it.
