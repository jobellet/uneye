#!/usr/bin/env python3
"""PsychoPy example: share the stimulus with the EyeLive engine. Run the engine side with  engine->stimulus_feed_open({path: "/tmp/stim.frames", saliency: true}).
Needs psychopy (pip install psychopy; on a machine without compilers install it with --no-deps and add pyglet==1.5.27, pyserial, pyobjc-framework-Cocoa/Quartz, python-bidi, arabic-reshaper, freetype-py, json-tricks, i18next ...). The frames are sent from the array that built the stimulus (no read-back from the GPU)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from psychopy import visual, core, monitors
from eyelive_frames import FrameWriter

W, H = 64, 64
fw = FrameWriter("/tmp/stim.frames", W, H, 3, gaze_rect=(-20.0, 15.0, 20.0, -15.0))          # the picture spans +-20 x +-15 deg
mon = monitors.Monitor("eyelive_demo", width=40.0, distance=57.0); mon.setSizePix((800, 600))      # 40 cm wide screen at 57 cm
win = visual.Window(size=(800, 600), units="deg", monitor=mon, color=(-1, -1, -1), fullscr=False)
img = np.full((H, W, 3), 110, np.uint8); img[8:20, 44:56] = (230, 40, 40)                       # a red square on gray (uint8, row 0 = top)
stim = visual.ImageStim(win, image=(img.astype(np.float32) / 127.5 - 1.0)[::-1], size=(40, 30), units="deg")   # PsychoPy: row 0 = bottom, values in [-1, 1]
for frame in range(120):
    stim.draw(); win.flip()
    fw.push_frame(img.tobytes(), t=core.getTime())                                                # once is enough for a static picture; every flip for a video
    if frame == 60: fw.set_candidates([(10.0, 5.0, 1.0), (-8.0, -2.0, 0.5)], until_ms=12.0)        # candidates known by the script (e.g. buttons) take priority
win.close(); fw.close()
