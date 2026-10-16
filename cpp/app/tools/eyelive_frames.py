#!/usr/bin/env python3
"""Writer of the EyeLive frame mailbox (see cpp/app/engine/include/eyelive/frame_mailbox.hpp). Python standard library only (mmap, struct), usable from PsychoPy or any script.

    from eyelive_frames import FrameWriter
    fw = FrameWriter("/tmp/stim.frames", 64, 64, 3, gaze_rect=(x_left, y_top, x_right, y_bottom))   # gaze units of the frame edges
    fw.push_frame(rgb_bytes, t=core.getTime())          # w * h * 3 bytes, row 0 = top (numpy: arr.tobytes())
    fw.set_candidates([(x, y, weight), ...], until_ms=12)   # up to 64, gaze units; weight > 0 (prior)

PsychoPy: draw the stimulus into a small off-screen buffer (or use the array you built for the ImageStim), downscale to 64 x 64 and push it once per frame (or once for a static picture).
    python3 eyelive_frames.py selftest <path>        writes a small test file (used by the C++ test)
"""
import mmap, os, struct, sys, time

MAGIC, VERSION, HEADER, SLOT_HEADER, MAX_CAND = b"EYLFRM01", 1, 256, 32, 64


class FrameWriter:
    def __init__(self, path, width, height, channels=3, slots=3, gaze_rect=(0.0, 0.0, 1.0, 1.0)):
        if channels not in (1, 3): raise ValueError("channels must be 1 or 3")
        self.w, self.h, self.c, self.slots, self.n = int(width), int(height), int(channels), max(2, int(slots)), 0
        self.slot_bytes = SLOT_HEADER + self.w * self.h * self.c; size = HEADER + 16 * MAX_CAND + self.slot_bytes * self.slots
        self.f = open(path, "w+b"); self.f.truncate(size); self.m = mmap.mmap(self.f.fileno(), size)
        self.m[0:8] = MAGIC; struct.pack_into("<IIIIIIII", self.m, 8, VERSION, HEADER, self.w, self.h, self.c, self.slots, SLOT_HEADER, MAX_CAND)
        struct.pack_into("<dddd", self.m, 48, *[float(v) for v in gaze_rect])

    def push_frame(self, pixels, t=None):
        if len(pixels) != self.w * self.h * self.c: raise ValueError("expected %d bytes, got %d" % (self.w * self.h * self.c, len(pixels)))
        self.n += 1; off = HEADER + 16 * MAX_CAND + self.slot_bytes * ((self.n - 1) % self.slots)
        seq, = struct.unpack_from("<Q", self.m, off); struct.pack_into("<Q", self.m, off, seq + 1)                  # odd: being written
        struct.pack_into("<dQII", self.m, off + 8, time.monotonic() if t is None else float(t), self.n, 0, 0)
        self.m[off + SLOT_HEADER: off + SLOT_HEADER + len(pixels)] = bytes(pixels); struct.pack_into("<Q", self.m, off, seq + 2)    # even: done
        struct.pack_into("<Q", self.m, 40, self.n)                                                                  # published last

    def set_candidates(self, candidates, until_ms=1e9):
        seq, = struct.unpack_from("<Q", self.m, 80); struct.pack_into("<Q", self.m, 80, seq + 1); c = list(candidates)[:MAX_CAND]
        struct.pack_into("<I", self.m, 88, len(c)); struct.pack_into("<d", self.m, 96, float(until_ms))
        for i, cand in enumerate(c):
            x, y = cand[0], cand[1]; w = cand[2] if len(cand) > 2 else 1.0; struct.pack_into("<ffff", self.m, HEADER + 16 * i, float(x), float(y), float(w), 0.0)
        struct.pack_into("<Q", self.m, 80, seq + 2)

    def close(self): self.m.flush(); self.m.close(); self.f.close()


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "selftest":
        fw = FrameWriter(sys.argv[2], 32, 24, 3, slots=2, gaze_rect=(-20.0, 15.0, 20.0, -15.0))
        for k in range(1, 4):
            px = bytearray(32 * 24 * 3); px[0] = k; px[(5 * 32 + 7) * 3] = 200; fw.push_frame(bytes(px), t=float(k))
        fw.set_candidates([(1.0, 2.0, 1.0), (-3.0, 4.0, 0.25), (5.5, -6.5, 1.0)], until_ms=12.0); fw.close(); print("selftest written", sys.argv[2])
    else:
        print(__doc__)
