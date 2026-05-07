---
icon: lucide/gauge
---

# Benchmarks

mojoxm has a 125-gate benchmark suite under `benchmarks/` that ties every
scheme to closed-form analytic reference states. Every BC dispatch arm in
every physics has at least one direct gate, every supported polynomial
order has parity coverage across all 5 smooth physics in 2D + 3D, and
every shocked configuration has its own gates with carefully sized
tolerances.

This page is the user-facing tour. For the per-bench inventory see the
[main repository README](https://github.com/EvanBluhm/mojoxm#performance).

## When to run what

| Want to…                                | Use                          | Time   |
| --------------------------------------- | ---------------------------- | ------ |
| Quick green-bar before pushing          | `make smoke`                 | ~45 s  |
| Full pre-push gate (matches CI)         | `make pre-push`              | ~50 s  |
| 13 representative gates                 | `make bench-quick`           | ~30 s  |
| Just the high-order P=5 gates           | `make bench-p5`              | ~65 s  |
| Just shocked flow                       | `make bench-shocks`          | ~40 s  |
| Convergence-rate gates only             | `make bench-rates`           | ~30 s  |
| Boundary-condition + source-term gates  | `make bench-bcs`             | ~45 s  |
| One specific physics                    | `make bench-<physics>`       | varies |
| The full suite (125 gates)              | `make bench-all`             | ~5 min |

Times are *cached* wall clocks — first run is slower because Mojo has to
compile every bench binary.

## Per-physics aggregators

```bash
make bench-mhd          # 33 gates, IdealMHD (plain + GLM, 2D + 3D)
make bench-euler        # 33 gates, Euler smooth + shocked
make bench-maxwell      # 25 gates, vacuum EM + J/M sources
make bench-sw           # 14 gates, ShallowWater
make bench-advection    # 12 gates, scalar advection
make bench-two-fluid    #  8 gates, FiveMomentTwoFluid
```

Use these when you've changed one physics module and want the focused
slice, not the whole 5-minute run.

## Convergence-rate gates

10 gates verify explicit \(\log_2(e_N / e_{2N})\) rates:

| Bench                                        | P  | Reported rate     | Floor   |
| -------------------------------------------- | -- | ----------------- | ------- |
| `bench_advection_translation_2d`             | 2  | ~2.77             | ≥ 2.0   |
| `bench_advection_translation_2d_p3`          | 3  | ~3.92 (P+1=4)     | ≥ 2.5   |
| `bench_advection_translation_2d_p4`          | 4  | ~4.67 (P+1=5)     | ≥ 2.5   |
| `bench_advection_translation_2d_p5`          | 5  | ~5.83 (P+1=6)     | ≥ 2.5   |
| `bench_advection_3d`                         | 2  | ~2.60             | ≥ 2.0   |
| `bench_advection_3d_p3`                      | 3  | ~3.69             | ≥ 2.5   |
| `bench_advection_3d_p4`                      | 4  | ~4.68 (NP=35)     | ≥ 2.5   |
| `bench_advection_3d_p5`                      | 5  | ~5.33 (NP=56)     | ≥ 2.5   |
| `bench_euler_smooth_wave_3d`                 | 2  | ~2.58             | ≥ 2.0   |
| `bench_euler_smooth_wave_3d_p3`              | 3  | ~3.69             | ≥ 2.5   |

The 2D Euler smooth-wave bench saturates at the Float32 round-off floor,
so it gates **absolute L2** instead of a rate.

## How a gate is structured

A typical bench source file:

```mojo
def main():
    # --- Setup ---
    var solver = Solver[Euler, 2](
        mesh=Mesh[2].build_periodic_3d(NX, NY, NZ, LX, LY, LZ),
        phys=Euler(gamma=1.4, eflux_kind=EFLUX_HLLEC),
    )
    launch_smooth_wave_ic(solver)

    # --- Run ---
    var dt = 0.4 / max_speed * h_min
    run_ssprk3_loop[Euler](solver, dt, T_FINAL=0.1, ...)

    # --- Compute analytic solution at T_FINAL on the same mesh ---
    var q_exact = compute_exact_at_t_final(...)

    # --- Compute relative L2 error ---
    var err = solver.rel_l2_error(q_exact)

    # --- Gate ---
    expect[Float32](err < 4e-5,
                    "Euler smooth-wave 3D rel L2 above floor")
    print(...)
```

Each gate's threshold is **sized to ~1.4–3× the empirical error floor**.
Single-physics bugs that move L2 by more than a small constant trip the
gate.

## What CI runs

`make pre-push` chains:

1. `format-check` (~3 s, mojo-format gate over the working tree)
2. `profile-summary-test` (~0.2 s, in-script self-test on the
   cross-bench summary parser)
3. `smoke` (~45 s = `test-utils + test-quick + bench-quick`)
4. `test-vtu-meshio` (~1 s, meshio-roundtrip + VTK_LAGRANGE_* spec
   compliance on 3D P=2..5 fixtures)

Total ~50 s cached. Run this before every push.
