# Reading EyeLink recordings

EyeLive's file source reads EyeLink **ASC** text files (what `edf2asc` writes from an `.edf`), in addition to `t,x,y[,valid]` CSV.
Detection is automatic (`.asc` extension or `**` / `MSG` / `START` / `SAMPLES` lines at the top).

```
eyelive_cli --replay recording.asc --eye right --ppd 35.0 ...
```

* Samples need **GAZE** coordinates (screen pixels). A file converted with HREF-only samples is refused with an explanatory message.
* `eye`: `auto` (the only eye, or the mean of both when both are valid), `left`, `right`.
* `.` in a sample = missing (blink); it becomes an invalid sample.
* Time jumps between recording blocks (a `START` after an `END`) are not waited for in real-time replay.
* Pixels are converted to degrees with `pixels_per_degree` (from the screen geometry in the profile).
* `ESACC` / `EFIX` / `EBLINK` lines are parsed (`AscParser::events()`) for comparing EyeLink's own events with EyeLive's; they are not yet used by the engine.

**Status**: written from the documented format and tested on synthetic files only (`eyelive_source_test_asc`). Please try a real file and report any parse problem (send the first 60 lines).
