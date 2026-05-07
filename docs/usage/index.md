---
icon: lucide/wrench
---

# Usage

Day-to-day operation of mojoxm is mediated by a single `Makefile` with
~300 targets organised into a few aggregator groups. This section walks
through the main workflows.

<div class="grid cards" markdown>

-   :material-hammer-wrench:{ .lg .middle } **[Building](building.md)**

    ---

    `make` targets, drivers, build products, common build-time
    troubleshooting.

-   :material-folder-multiple:{ .lg .middle } **[Examples](examples.md)**

    ---

    The 23 reference drivers under `examples/`. What each one shows.

-   :material-speedometer:{ .lg .middle } **[Benchmarks](benchmarks.md)**

    ---

    The 124-gate benchmark suite. Aggregators, per-physics slices, what
    each gate actually checks.

-   :material-chart-areaspline:{ .lg .middle } **[Profiling](profiling.md)**

    ---

    `nsys` profile baselines, `make profile-summary` cross-bench
    ranking, where to look first.

-   :material-image-multiple:{ .lg .middle } **[Visualization](visualization.md)**

    ---

    VTU output formats, ParaView, the `animate_2d.py` and
    `animate_dashboard.py` scripts.

</div>

## Top-level Makefile vocabulary

| Verb         | Meaning                                                              |
| ------------ | -------------------------------------------------------------------- |
| `make`       | Compile every binary the suite needs. ~2 min cold.                  |
| `make test*` | Run unit tests (33 of them, gating compile-time invariants).         |
| `make bench*`| Run analytic-solution gates (124 of them).                           |
| `make profile-bench-<n>` | Wrap one bench in `nsys profile --stats=true`.           |
| `make smoke` | `test-quick + bench-quick`, ~45 s. The day-to-day green-bar.        |
| `make pre-push` | Everything CI will run on push. ~50 s.                            |

The top of the `Makefile` is heavily commented; `make help` (if the
target exists for your branch) prints a quick orientation.
