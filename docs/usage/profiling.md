---
icon: lucide/activity
---

# Profiling

mojoxm has two layers of perf introspection:

1. **Built-in.** Every example driver and every benchmark prints a
   WARPXM-Kokkos-style memory breakdown at startup and a synced
   throughput report at shutdown. No external tool needed.
2. **Nsight Systems baselines.** `make profile-bench-<n>` wraps a bench
   in `nsys profile --stats=true` and saves a per-kernel time summary
   to `benchmarks/profile_reports/<bench>.kern.txt`. Baselines are
   checked into git.

## Built-in introspection

Sample output from `./euler_vortex` at 32³ (RTX 3090):

```text
=== Device memory usage ===
  Temporal solver (RK stages): 112.5 MB
  Spatial solvers (DG ops):    2.1 KB
  Cell limiter scratch:        4.5 MB
  Mesh connectivity:           84.8 MB
  Ghost cell sync (device):    0 B
  Ghost cell sync (pinned):    0 B
  ----------------------------------
  Total device memory:         201.8 MB
===========================

=== Step-loop throughput ===
  steps:                50
  DOF / step:           9830400
  wall time:            429.0 ms
  per-step wall:        8.6 ms
  throughput:           1.1 GDOF/s
  state bytes / step:   300.0 MB
  state bandwidth:      34.1 GB/s  (lower bound; excludes operator + mesh reads)
============================
```

The memory breakdown is **exact** — computed from each allocation's known
shape, no probe / sampling. The throughput report comes from
`Solver.bench_step_loop()`, which runs a 5-step warmup + 50-step
measurement loop with `ctx.synchronize()` at both endpoints so the elapsed
wall covers GPU compute (not async enqueue).

The reported state-bandwidth is a true **lower bound** on achieved DRAM
bandwidth: it counts only the bytes the SSPRK3 kernel sequence MUST move
between launches (8 × total_q_len × Float32 per step, derived from the
2-3-3 read/write pattern of the three RK stages), and **excludes operator
+ connectivity reads** (small, cache-friendly). Achieved bandwidth divided
by device peak gives a quick "how memory-bound is this run" intuition.

| NC | Workload      | Typical state bandwidth on RTX 3090 |
| -- | ------------- | ----------------------------------- |
| 1  | Advection     | >100 GB/s (~10% of peak)            |
| 5  | Euler         | ~40 GB/s (~3.5% of peak)            |
| 9  | IdealMHD-GLM  | ~30 GB/s                            |
| 17 | TwoFluid      | ~20 GB/s                            |

NC=1 advection routinely hits the highest state-bandwidth share because
each loaded byte fuels less arithmetic per element. NC=17 fuels much
more arithmetic per byte and is closer to compute-bound.

## Per-bench Nsight profiles

```bash
make profile-bench-euler-vortex
# Wraps ./bench_euler_vortex in:
#   nsys profile --stats=true ./bench_euler_vortex
# Saves the kernel summary to:
#   benchmarks/profile_reports/bench_euler_vortex_2d.kern.txt
```

The `.kern.txt` files are diff-able across runs. They look like:

```text
Time (%)   Total Time (ns)   Instances   Avg (ns)   ...
   76.0     12,034,128         150        80,228   euler_vol_lift_combine_rk_kernel_2d
   17.4      2,757,401         150        18,383   euler_face_flux_kernel_2d
    6.6      1,043,852         150         6,959   cell_mean_kernel_2d
```

Run the same bench again after a code change and `git diff` the report.
A 10–20% kernel time delta is meaningful; <5% is usually noise.

## Cross-bench summary

```bash
make profile-summary
# Or directly:
pixi run python scripts/profile_summary.py
```

Reads the cached `.kern.txt` reports for every bench and ranks each by
**dominant-kernel avg μs / launch**. Quick visual scan of where the
per-step cost sits across the suite.

Sample top-5 (RTX 3090):

```text
top 5 benches, sorted by avg us/launch:
  bench                            kernel        avg us    inst   total ms
  -------------------------------- ----------    ------ ------- ----------
  bench_mhd_alfven_3d_p5           rk_stage       659.1    4953     3264.5
  bench_maxwell_plane_wave_3d_p5   rk_stage       621.3    3303     2052.1
  bench_euler_sod_3d_p5            rk_stage       608.8    4998     3042.8
  bench_mhd_brio_wu_3d_p3          rk_stage       408.9    3807     1556.8
  bench_euler_vortex_3d_p3         rk_stage       392.3   11403     4473.2
  -> shown 5 benches: 14.39 s of dominant-kernel time;
     full suite (67.46 s across all benches dominant-kernel-only)
```

Useful flags:

| Flag             | What it does                                                |
| ---------------- | ----------------------------------------------------------- |
| `--by-physics`   | Roll up per-physics totals                                  |
| `--show-cv`      | Show coefficient of variation (run-to-run noise estimate)   |
| `--csv`          | Emit CSV (for spreadsheets / scripts)                       |
| `--markdown`     | Emit Markdown table (for PR descriptions)                   |
| `--filter=<re>`  | Regex filter on bench names                                 |
| `--sort=total`   | Sort by total time (not avg/launch) — find suite-wide hotspots |

On the current baseline:

| Physics       | % of suite total |
| ------------- | ---------------- |
| Euler         | 56%              |
| Two-Fluid     | 16%              |
| MHD           | 14%              |
| Advection     |  6%              |
| Maxwell       |  4%              |
| ShallowWater  |  3%              |

3D dominates 2D by ~18× of total suite time.

## Where to optimise

A common pitfall: the top of the avg-µs/launch list isn't always the
top of the total-time list. Use `--sort=total` to find where the
suite-wide budget actually lives.

For 2D limiter optimisation, the ROI is roughly **3–4× the equivalent
3D effort** because the 3D cooperative `rk_stage_kernel` already
absorbs the runtime. See [Architecture · 2D vs 3D](../architecture/2d-vs-3d.md#cost-story-limiter-share)
for the cost-share math.
