---
icon: lucide/zap
---

# Quickstart cheatsheet

Day-to-day `make` targets for an existing checkout. Times below are typical
"cached" wall clocks on an RTX 3090 / WSL2 — first run is slower because
Mojo has to compile.

## Build

| Target                  | What it does                                                      |
| ----------------------- | ----------------------------------------------------------------- |
| `make all`              | Build every driver, bench, and test. ~2 min cold.                |
| `make gpu`              | Just the GPU drivers under `examples/`.                           |
| `make test`             | Just the unit-test binaries.                                      |
| `make <driver>`         | Build one specific binary, e.g. `make euler_vortex`.              |
| `make clean`            | Remove `build/` + compiled binaries.                              |

## Test

| Target                   | What it gates                                                  | Time   |
| ------------------------ | -------------------------------------------------------------- | ------ |
| `make test-quick`        | 8-test smoke: host operators, 2D + 3D constant-state           | ~13 s  |
| `make test-limiter`      | 8 BJ-limiter unit tests, P=2/3/4/5 in 2D + 3D                  | ~14 s  |
| `make test-utils`        | 5 helper-module tests (perf, VTU, SSPRK3, partition, Sod)      | ~7 s   |
| `make test-all`          | The full 33-test inventory                                     | ~4 min |

## Bench

| Target                   | What it gates                                                  | Time   |
| ------------------------ | -------------------------------------------------------------- | ------ |
| `make bench-quick`       | 13 representative gates                                        | ~30 s  |
| `make bench-p5`          | 19 P=5 high-order gates                                        | ~65 s  |
| `make bench-shocks`      | 13 Sod / dam-break / Brio–Wu gates                             | ~40 s  |
| `make bench-rates`       | 10 explicit log₂ convergence-rate gates                        | ~30 s  |
| `make bench-bcs`         | 27 BC + source-term gates                                      | ~45 s  |
| `make bench-all`         | All 125 analytic-solution gates                                | ~5 min |

## Per-physics aggregators

```bash
make bench-mhd          # 33 gates touching IdealMHD / GLM
make bench-euler        # 33 gates touching Euler (smooth + shocked)
make bench-maxwell      # 25 vacuum-EM + J/M source gates
make bench-sw           # 14 ShallowWater gates
make bench-advection    # 12 scalar-advection gates
make bench-two-fluid    #  8 FiveMomentTwoFluid gates
```

## Smoke gates (the ones you'll actually run)

```bash
make smoke         # test-quick + bench-quick      ~45 s cached
make pre-push      # format-check + smoke +
                   # profile-summary-test + test-vtu-meshio    ~50 s cached
```

`make pre-push` is the gate to run before pushing — it covers everything
the CI hooks will trip on.

## Profile

```bash
make profile-bench-<name>     # Single bench under nsys --stats=true
make profile-summary          # Cross-bench ranking by dominant kernel
make profile-summary --by-physics
```

Reports land in `benchmarks/profile_reports/`. Baselines are checked in.

## Visualise

```bash
./euler_rising_bubble
pixi run python scripts/animate_dashboard.py     # output/dashboard.gif

./euler_vortex_2d_gpu
pixi run python scripts/animate_2d.py output/solution_euler2d_gpu.pvd \
  -f rho,p,'|v|'                                  # 3-panel MP4
```

## Docs

```bash
make docs-serve     # live-reload preview, http://localhost:8000
make docs-build     # production build to ./site/
```

!!! tip "Don't pass `-j`"

    `make -j4` is *slower* than serial because Mojo's compiler is already
    parallel per file. Just `make`.
