# ONNX Runtime as SOUP (software of unknown provenance)

**Illustrative only**, inspired by how IEC 62304 treats SOUP; not a compliance claim.

## What it is used for here
`cpp/src/onnx_engine.cpp` can run the window network (the original U'n'Eye U-Net) through ONNX Runtime (ORT) instead of the
native C++ engine. It is **optional**: it is only compiled with `-DONNXRUNTIME_ROOT=...`, and nothing in the safety evidence
(`ctest`, `fault_injection`, `bench_latency`) uses it. Every number in `docs/safety/hazards.md` comes from the native engine.

## Why the native engine exists next to it
| Concern | ONNX Runtime | Native engine (`native_engine.cpp`, `causal_engine.cpp`) |
|---|---|---|
| Memory at run time | ORT is a general runtime with its own allocators (the session here is created with an arena allocator); whether `Run()` allocates was **not measured** in this repository (no ORT build on the test machines) | all buffers sized in the constructor; 0 allocations measured (`test_zero_heap`) |
| Size we must understand | large third-party code base, many execution providers and graph optimisations | ~300 lines we wrote, readable line by line |
| Numerical behaviour | graph optimisations (`ORT_ENABLE_ALL`) may reorder / fuse operations | same operation order as the export; parity with PyTorch within 1e-4 on probabilities (`test_parity`) |
| Speed (earlier cloud measurements, `cpp/README.md`) | ~0.12 ms per window | ~0.27-0.28 ms per window |

The trade-off: ORT is about twice as fast, the native engine is the one whose behaviour we can show and test completely.

## What qualifying ORT would need (not done)
1. Pin one exact version and build (record its hash), keep it under configuration management; no automatic updates.
2. List the functions used (`Ort::Env`, `Ort::Session`, `Session::Run`, CPU execution provider only) and disable everything else
   (single thread, fixed graph optimisation level, no other providers).
3. Requirements for those functions (correct output within a tolerance, bounded time, no allocation or a documented one) and
   tests for each: parity against the PyTorch reference and the native engine on many windows, allocation counting around
   `Run()`, latency distribution on the target machine.
4. Review the published issues / security advisories of that version and decide which apply.
5. A fallback when ORT fails: in this design the window network is only a third stage; if it is removed or fails, the engine keeps
   the labels of the causal networks and the guard still checks every output.
