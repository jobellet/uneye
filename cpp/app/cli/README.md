# eyelive_cli

Headless front-end of the EyeLive engine: records a stream (or a replayed file, or a simulation) into a session file, labels it with a detector, fires triggers, and can pre-label and train. Everything the GUI does at acquisition time, without a window.

```sh
# 1. record 3 s of a simulated eye as fast as possible and print the events found by the label-free HMM
eyelive_cli --sim --duration 3 --session /tmp/a.eyelive

# 2. replay a recorded trace (CSV "t,x,y[,valid]") at real time, with the fast detector, and write the batch events to a file
eyelive_cli --replay trace.csv --detector fast_threshold --session /tmp/b.eyelive --events-out events.csv

# 3. listen to a tracker streaming UDP packets (protocol v1: binary "EYE1" or text lines "t x y [valid]") on port 5005 for one hour
eyelive_cli --udp 5005 --duration 3600 --session /data/subject1.eyelive --scale auto

# 4. fire a UDP message 10 ms after every saccade onset, from a rules file
eyelive_cli --udp 5005 --triggers rules.txt --duration 600

# 5. pre-label the first 50 s with the detector and train a U'n'Eye on them (no review: for tests and benchmarks)
eyelive_cli --sim --duration 60 --prelabel-seconds 50 --train --session /tmp/c.eyelive
```

A rules file has one rule per line (`#` starts a comment):

```
name=sacc;cls=1;anchor=provisional;delay_ms=10;sink=udp:127.0.0.1:6000;payload=SACCADE {t} {amp}
name=micro;cls=1;anchor=final;amp_max=1.0;sink=log:/tmp/micro.log;payload=MICRO {t} {amp}
```

Output: one line per final event (`event <onset s> <offset s> class=<c> amp=<deg> peak=<deg/s>`), then a summary line (samples, rate, dropped packets, gaps, events, latency, scale). Exit code 0 on success, 1 when the session cannot start or no data was recorded, 2 for a bad command line. The session file can be reopened with `Recorder::open_readonly` and read live from MATLAB / Python (see `cpp/app/readers/README.md`).
