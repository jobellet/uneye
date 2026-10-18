# Testing EyeLive on Windows with MATLAB

What the continuous integration cannot show: the program running on a real Windows computer, MATLAB writing and reading its files, and the timing of the triggers with the Windows timer. About 15 minutes, nothing is installed, nothing leaves the computer.

## What you need
* Windows 10 or 11, MATLAB R2020b or later (`udpport`, `memmapfile`; no toolbox is needed: the script does not use the Statistics or Signal Processing toolboxes).
* The folder `eyelive-windows` built by GitHub Actions: open https://github.com/jobellet/uneye/actions, the last green run of the workflow **eyelive**, section *Artifacts*, download `eyelive-windows` (a zip; GitHub asks you to sign in). Unzip it into a folder you own, for example `C:\eyelive`.
  It holds `bin\` (eyelive_cli.exe, eyelive_tool.exe, eyelive_feed_probe.exe, models), and the MATLAB files in the same folder tree as the repository.
* If Windows or the antivirus blocks the programs: right-click the zip before unzipping, Properties, tick *Unblock*; see INSTALL_UNSIGNED.md. Windows Firewall may ask about network access: choose *Private networks* (the test only uses 127.0.0.1).

## Run it
In MATLAB:
```matlab
cd C:\eyelive\cpp\app\readers\matlab\windows_test
eyelive_windows_selftest('C:\eyelive\bin')
```
It takes about two minutes (20 s of simulated eye tracker plus waiting) and prints `[ ok ]`, `[warn]` and `[FAIL]` lines. It writes `eyelive_windows_selftest_report.txt` in the current folder: **send me that file** (or paste it).

## What each part checks
* **A. environment:** MATLAB version and the real duration of `pause(0.001)` on your Windows (the default timer step is 15.6 ms; that is why the sender busy-waits).
* **B. frame mailbox:** `eyelive_frames.m` (MATLAB branch, `memmapfile`; never run before) writes five frames and three candidates into a file; the script reads every field back byte by byte and the C++ reader (`eyelive_feed_probe.exe`) reads the same file.
* **C. a tracker-like stream:** MATLAB sends a synthetic eye (20 s, 1000 Hz, 33 saccades) in protocol-v1 UDP packets to `eyelive_cli.exe`, which records it and sends one UDP trigger per saccade onset to MATLAB. Measured: the latency of the triggers, the share of saccades that fired one, the pace of the sender, and the state block (`eyelive_open.m`) read **while the other process writes the session file** (memory mapping of a growing file on Windows).
* **D. after the recording:** `eyelive_tool.exe info` and `export`; the MAT-file (`load`), the events CSV (`readtable`) and the provenance journal (`jsondecode`) are read back and compared with what was sent.

## Reading the result
* `[FAIL]` = something does not work: send the report.
* `[warn]` on latency or on the sender's pace = it works but not as precisely as expected (the Windows timer, an antivirus scanning the session file, power saving): a laptop on battery power gives worse numbers than one plugged in.
* Typical values measured on a Mac for the same chain: median 5-10 ms between the *send* of the first detected sample and the arrival of the trigger (the 5 ms packets, 3-6 ms detection, the UDP trip).
* Options: `eyelive_windows_selftest('C:\eyelive\bin', 'Seconds', 40, 'Keep', true)` keeps the temporary folder (session file, exports) for a closer look.

## If the build is missing
The Windows build has not been run on a real machine by the authors: if the artifact is not there or the programs do not start, the log of the job *windows* in GitHub Actions (annotations at the top of the run) says what failed; tell me and I fix it.
