# AGENTS.md
Router for coding agents. The repository's per-folder guidance is in the `AGENT.md` files (start with `AGENT.md` at the root).
If you were asked to "execute the instructions in docs/tasks/Txx.md": read `docs/tasks/README.md` first, then your task file, and obey its ownership list (CI enforces it). Contracts are in `cpp/app/include/eyelive/` and are frozen.
Environment: `sh cpp/app/ci/jules_setup.sh`; build and test with `cmake -S cpp -B build -G Ninja && cmake --build build && ctest --test-dir build -L eyelive --output-on-failure`.
