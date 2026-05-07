---
icon: lucide/atom
hide:
  - navigation
---

# mojoxm

**A GPU-accelerated discontinuous Galerkin solver for hyperbolic PDEs, written
in [Mojo](https://docs.modular.com/mojo/manual).**

mojoxm parameterises the entire numerical pipeline by a `Physics` trait. Each
simulation is a single-file Mojo driver that composes a mesh, a physics type,
an initial condition, and an SSPRK3 time integrator. Six physics modules ship
today — from scalar advection to a 17-component electron/ion two-fluid
plasma model coupled to full Maxwell.

[Get started](getting-started/index.md){ .md-button .md-button--primary }
[Browse the architecture](architecture/index.md){ .md-button }
[See the physics](physics/index.md){ .md-button }

---

## Capabilities at a glance

<div class="grid cards" markdown>

-   :material-flash:{ .lg .middle } **GPU-only, fused kernels**

    ---

    Single fused `rk_stage_kernel` per SSPRK3 stage in 3D; two-launch
    flux + vol+lift+RK pipeline in 2D. No CPU fallbacks, no host-side
    arrays past startup.

-   :material-chart-line:{ .lg .middle } **Arbitrary order P=1..5**

    ---

    Lagrange DG validated end-to-end at P=2/3/4/5 across all 5 smooth
    physics in 2D + 3D, plus shocked Euler and Brio–Wu MHD with a
    Barth–Jespersen limiter.

-   :material-atom:{ .lg .middle } **Six physics modules**

    ---

    Advection, Euler, ShallowWater, Maxwell, IdealMHD (with Dedner GLM
    cleaning), and a 17-component FiveMomentTwoFluid plasma model.

-   :material-test-tube:{ .lg .middle } **Validated**

    ---

    33 unit tests + 125 analytic-solution benchmark gates, including
    10 explicit log₂ convergence-rate gates. Every BC dispatch arm in
    every physics has at least one direct gate.

-   :material-microscope:{ .lg .middle } **Built-in introspection**

    ---

    Every driver prints a WARPXM-style device-memory breakdown at startup
    and a synced throughput report at shutdown. `make profile-summary`
    ranks every benchmark by dominant-kernel cost.

-   :material-image-multiple:{ .lg .middle } **Streaming VTU output**

    ---

    Async `writev()`-based per-frame VTU writer with overlap; multi-field
    final-state snapshots; an animated 2×2 dashboard script for quick
    sanity checks.

</div>

## What kind of problems does it solve?

mojoxm handles **systems of hyperbolic conservation laws** of the form

\[
\frac{\partial \mathbf{q}}{\partial t} + \nabla \cdot \mathbf{F}(\mathbf{q})
  = \mathbf{S}(\mathbf{q}, \mathbf{x})
\]

on Cartesian-derived simplex meshes — Kuhn 6-tetrahedra in 3D, Kuhn 2-triangles
in 2D — with periodic, slip-wall, transmissive-outflow, or prescribed-inflow
boundaries. Every physics module plugs into the same DG machinery by providing
four hooks: `internal_flux`, `numerical_flux`, `boundary_flux`, and
`source_term`.

## Where to next?

| If you want to…                                | Go here                                                               |
| ---------------------------------------------- | --------------------------------------------------------------------- |
| Install the toolchain and run an example       | [Getting started](getting-started/index.md)                           |
| Understand the DG scheme and Kuhn decomposition | [Architecture · Numerical scheme](architecture/numerical-scheme.md)   |
| See how to add a new physics module            | [Architecture · Physics trait](architecture/physics-trait.md)         |
| Read the per-physics formulation and BCs       | [Physics](physics/index.md)                                           |
| Run the test + benchmark suite                 | [Usage · Benchmarks](usage/benchmarks.md)                             |
| Profile a kernel with Nsight Systems           | [Usage · Profiling](usage/profiling.md)                               |
| Visualise the output                           | [Usage · Visualization](usage/visualization.md)                       |

## Project status

mojoxm is a research codebase — fast moving, tightly scoped, and
single-precision throughout. The current release is **single-rank
GPU-only** for the 2D pipeline; the 3D pipeline supports MPI multi-rank
runs with periodic and non-periodic BCs (`mpi_advection_test` and
`mpi_bc_test` gate np=1 ↔ np=4 bit-identicality).

Current limitations and open directions are catalogued in the
[architecture overview](architecture/index.md#limitations-and-roadmap).
