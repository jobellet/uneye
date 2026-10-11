# EyeLive Readers

This directory contains reference implementations to read real-time state blocks produced by EyeLive without a driver and without compilation. It uses memory-mapped files via `memmapfile` in MATLAB and `mmap` in Python.

## Python / PsychoPy Example

```python
from eyelive_reader import StateReader
import time

def main():
    reader = StateReader("session.bin")

    # Read the latest state:
    state = reader.latest()
    if state:
        print(f"Latest position: ({state['x']:.2f}, {state['y']:.2f})")
        print(f"Sample count: {state['n_samples']}")

    # Continuously follow events (generator)
    print("Following events...")
    for ev in reader.follow_events():
        print(f"Event {ev['id']} (cls {ev['cls']}) amp={ev['amplitude_deg']:.2f}")

if __name__ == "__main__":
    main()
```
