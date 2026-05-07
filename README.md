# mojoxm — GPU DG hyperbolic solver in Mojo

A GPU-accelerated [discontinuous
Galerkin](https://en.wikipedia.org/wiki/Discontinuous_Galerkin_method)
finite-element solver in [Mojo](https://docs.modular.com/mojo/manual),
inspired by [WARPXM](https://doi.org/10.1016/j.cpc.2010.12.048).

The solver is parameterized by a `Physics` trait; each simulation is
its own single-file Mojo driver that composes a mesh, a physics type,
an initial condition, and a time integrator. Six physics implementations
ship today:

- **Advection** — scalar linear advection, upwind flux. 1 component.
- **Euler** — 5-moment compressible gas dynamics with four selectable
  numerical fluxes (Rusanov, Roe, HLLE, HLLEC), optional Harten-Hyman
  entropy fix, and optional uniform gravity source.  5 components.
- **ShallowWater** — 2D shallow water (h, h·u, h·v) embedded in 3D
  with `F^z = 0`. Rusanov flux, slip-wall BC. 3 components.
- **Maxwell** — vacuum Maxwell (E, B) with Rusanov flux, full BC menu
  (PEC / outflow / prescribed-inflow), and optional uniform current
  (J) + magnetization (M) sources.  6 components.
- **IdealMHD** — single-fluid ideal MHD with Dedner GLM div(B)
  cleaning. 8 conservation components + 1 GLM scalar = 9 total.
- **FiveMomentTwoFluid** — electron + ion fluids + full Maxwell + GLM.
  17 components. Lorentz force + current coupling through the
  source-term hook.

Nine 3D reference drivers under `examples/`:

- `advection_gaussian` — Gaussian pulse on `[0, 1]³` with `v = (1, 1, 1)`,
  triply periodic. After `T = 1` the exact solution returns to the IC.
- `euler_vortex` — classical isentropic vortex (Shu form) on `[0, 10]³`
  with background velocity `(1, 1, 0)` and HLLEC Riemann solver.
- `euler_taylor_green` — compressible Taylor-Green vortex on a 2π cube
  at Ma ≈ 0.3. Kinetic-energy cascade toward turbulence.
- `euler_sod` — smoothed Sod shock tube with non-periodic BCs
  (transmissive outflow on x, slip walls on y/z).
- `euler_rising_bubble` — buoyant thermal bubble in a hydrostatic
  atmosphere, closed reflecting box. Demonstrates the gravity source
  term.
- `shallow_water_drop` — radial Gaussian perturbation in a closed
  basin, slip walls.
- `maxwell_cavity` — PEC-bounded standing wave, analytic round-trip
  verification.
- `mhd_alfven` — linearly polarised Alfvén wave in a periodic box.
- `two_fluid_langmuir` — electron plasma oscillation at the plasma
  frequency, analytic two-fluid period match.

Plus ten 2D-triangle GPU drivers under `examples/*_2d_gpu.mojo`:
`advection_gaussian_2d_gpu`, `advection_outflow_2d_gpu`,
`euler_vortex_2d_gpu`, `euler_sod_2d_gpu`, `euler_channel_2d_gpu`,
`shallow_water_drop_2d_gpu`, `shallow_water_dam_break_2d_gpu`,
`maxwell_cavity_2d_gpu`, `mhd_alfven_2d_gpu`, plus the NC=7 GLM
variant `mhd_alfven_glm_2d_gpu` -- covering periodic / non-periodic /
shock-tube / GLM cleaning configurations on Kuhn-decomposition
triangle meshes.  See [2D triangulated DG (GPU-only)](#system-architecture)
below for the full list with measured periodic / non-periodic / shock
benchmark numbers.

Plus four MPI utility drivers (`mpi_hello`, `mpi_partition`,
`mpi_patch_mesh`, `mpi_halo_pingpong`) for hands-on inspection of the
multi-rank decomposition + halo-exchange machinery.

## Contents

- [Capabilities at a glance](#capabilities-at-a-glance)
- [Numerical scheme](#numerical-scheme)
- [System architecture](#system-architecture)
- [Diagnostics & visualization](#diagnostics--visualization)
- [Performance](#performance)
- [Build & run](#build--run)
- [Profiling](#profiling)
- [Design decisions worth knowing](#design-decisions-worth-knowing)
- [Limitations](#limitations)
- [References](#references)

## Capabilities at a glance

- **Boundary conditions**: periodic, slip wall (`BC_WALL`), transmissive
  outflow (`BC_OUTFLOW`), prescribed inflow (`BC_INFLOW`). Single- and
  multi-rank supported. Drivers declare BCs via a `BoundaryConditions`
  struct; the mesh builder allocates BC-side faces and routes them to
  a physics-provided `boundary_flux(q, bc_type, n)` hook.  Every BC
  dispatch arm in every physics has at least one direct gate in the
  test/bench harness.
- **Source-term hook**: physics types provide a `source_term(q, x, s_out)`
  method evaluated per-node and added pointwise to the SSPRK3 RHS.
  Used for gravity (Euler), current / charge coupling (Maxwell,
  two-fluid), GLM damping (MHD).
- **Diagnostics**: every driver can register linear, squared, and
  max-norm conserved-quantity integrals via `DiagnosticsWriter`. One
  CSV row per frame, with MPI allreduce at np>1.
- **Perf introspection**: `Solver.memory_report()` returns a WARPXM-
  Kokkos-style categorised device-memory breakdown; `Solver.bench_
  step_loop()` runs a sync'd warmup + measurement loop and returns a
  `ThroughputReport` with per-step wall time, DOF/s, and a state-
  bandwidth lower bound.  Every 2D-GPU and 3D example driver prints
  both at startup / shutdown; see [Performance](#performance) for a
  sample.
- **Animated dashboard**: `scripts/animate_dashboard.py` reads the VTU
  frames + diagnostics CSV and emits a 2×2 animated GIF (density
  field + auto-grouped time-series panels). Lazy frame loader scales
  to large meshes.
- **Higher-order reference element**: `ReferenceElement[P]` computes
  Lagrange basis, mass, stiffness, and lift operators at arbitrary
  order via Vandermonde inverse + analytic integration. Node ordering
  matches VTK_LAGRANGE_TETRAHEDRON and is back-compatible with
  VTK_QUADRATIC_TETRA at P=2. `P` is a comptime parameter on every
  3D struct (`Mesh[P]`, `LocalMesh[P]`, `Solver[PhysT, P]`,
  `FrameWriter[PhysT, P]`, `DiagnosticsWriter[PhysT, P]`).  Validated
  end-to-end at P=2, 3, 4, and 5 across all 5 smooth physics in 2D
  (NP=21) and 3D (NP=56), plus shocked Euler at P=2/3/4/5 in both
  dimensions, plus the BJ limiter at P=2/3/4/5 in both dimensions
  (`make bench-p5`, `make test-quick`).
- **Shock limiter (3D)**: `Solver.enable_cell_limiter(eps)` turns on a
  Venkatakrishnan-smoothed Barth-Jespersen slope limiter that runs
  after every RK stage.  Brings classical Sod to T=0.20 with left
  rho = 0.998, right rho = 0.119 (within ~0.2%/5% of exact).
- **2D triangulated DG (GPU-only)**: `src/reference_2d.mojo` +
  `local_mesh_2d.mojo` build the host-side mesh / reference element
  (validated by `make test-reference-2d` / `test-local-mesh-2d`) and
  are uploaded to the device through `LocalMesh2DGpu[P]` /
  `ReferenceElement2DGpu[P]`.  All physics run on device in Float32
  via per-physics modules under `src/`:
  `local_mesh_2d_gpu_advection.mojo` /
  `local_mesh_2d_gpu_euler.mojo` /
  `local_mesh_2d_gpu_sw.mojo` /
  `local_mesh_2d_gpu_mhd.mojo` /
  `local_mesh_2d_gpu_mhd_glm.mojo` /
  `local_mesh_2d_gpu_maxwell.mojo`, each running 2 launches per
  SSPRK3 stage (face-flux kernel + per-(elem, node) fused
  vol+lift+RK kernel) orchestrated by `<name>_rk_stage_2d[P]`.
  The Shu-Osher 3-stage SSPRK3 buffer-routing dance is factored
  into `src/ssprk3.ssprk3_stage_plans` -- a non-templated helper
  shared by all 10 example drivers + 60 benches (every 2D bench); one compiled
  binary regardless of how many physics modules consume it.
  The Venkat-smoothed Barth-Jespersen slope limiter lives in its own
  `local_mesh_2d_gpu_limiter.mojo`.  The parent `src/local_mesh_2d_gpu.mojo`
  holds the `LocalMesh2DGpu` struct + generic NC-templated helpers
  (cell-avg / cell-mean / rk-update).  Five physics
  (Advection / Euler / ShallowWater / IdealMHD / Maxwell), with the
  full periodic / wall / outflow / inflow BC menu in every 2D
  physics (Euler / SW / Maxwell / MHD plain (NC=6) / GLM-MHD (NC=7));
  two Euler Riemann solvers
  (Rusanov and HLLC) and two SW Riemann solvers (Rusanov and HLL), a
  Venkat-smoothed Barth-Jespersen cell-level limiter
  (`bj_limit_full_2d`) for shock stability.  Ten end-to-end
  drivers under `examples/*_2d_gpu.mojo` emit 21-frame VTU
  sequences + `.pvd` collections.  Multi-component drivers ship
  multiple physically meaningful scalar fields per VTU frame
  (Euler: rho + p + |v|; SW: h + |v|; MHD plain: rho + |v| + |B| + By;
  MHD-GLM: rho + |v| + |B| + psi; Maxwell: Ez + |E| + |B|), so
  users can flip between fields in ParaView or render side-by-side
  multi-panel animations via
  [`scripts/animate_2d.py`](scripts/animate_2d.py) -- e.g.
  `scripts/animate_2d.py output/solution_euler2d_gpu.pvd -f rho,p,'|v|'`
  produces a 3-panel MP4.

  Every multi-component 3D driver under `examples/*.mojo` (Euler
  vortex / Taylor-Green / Sod / rising-bubble, Maxwell cavity, MHD
  Alfvén, SW drop, Two-Fluid Langmuir) additionally writes a final-
  state multi-field VTU at `output/snapshot_t_final.vtu` (rho+p+|v|
  for Euler, Ex+|E|+|B| for Maxwell, By+|B|+psi for MHD, h+|u| for
  SW, n_e+n_i+Ex+charge for Two-Fluid).  ParaView opens this
  directly without a PVD.  The per-frame async pipeline still emits
  one component per frame for performance.

  Per-driver smoke / rate verifications:
  * **Periodic:** `advection_gaussian_2d_gpu` (~7700 compute
    steps/sec, 0.07 % rel L2 over one period), `euler_vortex_2d_gpu`
    (~7600, isentropic vortex, 6.8 % rel L2 at P=2 / 32x32),
    `shallow_water_drop_2d_gpu` (~6300, mean-h conservation ~4e-5),
    `mhd_alfven_2d_gpu` (~9100, 0.36 % rel L2 on a one-period
    linear Alfven wave at P=2 / 64x4),
    `mhd_alfven_glm_2d_gpu` (~12600, NC=7 GLM-MHD with c_h=1.5 +
    alpha_d=0.5; psi cleaning visible in animation -- |psi|
    advects at +-c_h while damping).
  * **Non-periodic (BC_WALL / BC_INFLOW / BC_OUTFLOW):**
    `advection_outflow_2d_gpu` (Gaussian drains out, mass -> 5e-7
    of IC by t=1), `euler_channel_2d_gpu` (Mach-2 wind tunnel
    with inflow + outflow + walls; rho_max_drift = 1.2e-7 --
    analytic steady answer preserved to Float32 epsilon),
    `shallow_water_dam_break_2d_gpu` (h_L=2 / h_R=1 Riemann in a
    closed basin; ~8700 steps/sec),
    `maxwell_cavity_2d_gpu` (~12600 steps/sec, TM(1,1) PEC standing
    wave with rel L2 ~4e-5 at P=2 / 32x32 over one period; BC_WALL
    on all four sides).
  * **Shocks (HLLC + BJ limiter):** `euler_sod_2d_gpu` (classical
    Sod lifted to 2D at P=2 / 128x16: rho overshoot 0.27 % above
    rho_L, boundary states within 0.2 % of expected; 5100 steps/sec
    with three limiter passes per SSPRK3 step).

  **Tests:** GPU kernels are validated by self-consistent invariants
  rather than CPU reference code.  33 tests run under `make test-all`,
  each gating a hard regression.  Quick aggregators: `make test-quick`
  (8-test ~15s cached smoke covering host operator construction, 2D + 3D
  physics constant-state, the limiter pipelines, and the 3D multi-
  field VTU writer); `make test-limiter` (8 BJ-limiter unit tests
  at P=2/3/4/5 in 2D + 3D); `make test-utils` (5 helper-module
  tests covering perf-introspection, VTU multi-field, SSPRK3
  stage-plan, MPI partition factorisation, Sod Riemann solver --
  3 host-only + 2 small GPU-using; ~7s cached).  The full inventory:
  * 2D pipeline -- `local_mesh_2d_gpu_test` (upload round-trips +
    constant-state at P=1..5), `euler_2d_gpu_test` /
    `sw_2d_gpu_test` / `mhd_2d_gpu_test` / `mhd_glm_2d_gpu_test` /
    `maxwell_2d_gpu_test` (constant-state per physics, P=2..5,
    including the NC=7 GLM-MHD path), `limiter_2d_gpu_test` + `_p3`
    + `_p4` + `_p5` (smooth passthrough + within-cell spike
    monotonicity at every supported order).
  * 3D pipeline -- `euler_3d_test` / `maxwell_3d_test` /
    `sw_3d_test` / `mhd_3d_test` / `two_fluid_3d_test` (constant-
    state preservation per physics through Solver[PhysT, P], P=2..5),
    `limiter_3d_test` + `_p3` + `_p4` + `_p5` (BJ slope-limiter cell-
    mean conservation invariant, drift = 0 exactly at every supported
    order), `p3_smoke_test` (Mesh[3] + Solver[Advection, 3] round-trip).
  * MPI -- `mpi_advection_test` / `mpi_bc_test` (np=1 vs np=4
    bit-identical periodic / non-periodic).
  * Misc -- `reference_element_test`, `reference_element_2d_test`,
    `local_mesh_2d_test`, `diagnostics_test`.

- **Benchmark harness** (`benchmarks/`, run via `make bench-all`):
  121 analytic-solution gates tying schemes to closed-form reference
  states.  Coverage is parity across dimensions for every core
  physics, plus shocked-flow gates wherever a stable scheme exists,
  10 explicit log2(e_N / e_2N) rate gates -- advection 2D + 3D at
  P=2/3/4/5 (8) and Euler smooth-wave 3D at P=2 + P=3 (2; the 2D
  Euler smooth-wave bench saturates at the Float32 round-off floor
  so it gates absolute L2 instead),
  P=4 absolute-L2 sentinels for every core physics in 2D + 3D
  (Euler / Maxwell / SW / MHD-GLM), and full P=5 P-parity coverage
  across all 5 smooth physics in both dimensions (advection / Euler
  / Maxwell / SW / IdealMHD-GLM at 2D NP=21 and 3D NP=56) -- the
  largest comptime configuration the suite gates end-to-end.  When
  iterating, pick the slice that matches what's changing:
  `make bench-quick` (13 representative gates, ~30s cached),
  `make bench-p5` (19 P=5 high-order gates, ~65s cached),
  `make bench-shocks` (13 Sod / dam-break / Brio-Wu gates, ~40s cached),
  `make bench-rates` (10 explicit convergence-rate gates, ~30s cached),
  `make bench-bcs` (26 BC + source-term gates, ~45s cached).  When iterating
  on a single physics module, the per-physics aggregators run only
  the gates exercising that module: `make bench-mhd` (33), `bench-euler`
  (33), `bench-maxwell` (23), `bench-sw` (14), `bench-advection` (12),
  `bench-two-fluid` (6).  Or `make smoke` for a one-command
  `test-quick + bench-quick` combined sanity check (~45s cached);
  `make pre-push` adds the working-tree `format-check` (~3s) and
  the VTK_LAGRANGE_TETRAHEDRON spec validation in front + after
  so the standard pre-push gate is one command (~50s cached).
  * **2D smooth (31):** `bench_advection_translation_2d` (rate >= 2.0)
    + `_p3` (rate ~3.92, P+1=4) + `_p4` (rate ~4.67, P+1=5) + `_p5`
    (rate ~5.83, P+1=6), `bench_advection_outflow_2d` (BC_OUTFLOW drainage gate),
    `bench_advection_inflow_2d` (BC_INFLOW preservation gate with
    non-trivial inflow_q),
    `bench_euler_vortex_2d` + `_p3` (HLLC NP=10 long-time vortex
    sentinel; rel L2 ~6.8%% at T=10, comparable to the P=2 / Rusanov
    floor since the long-horizon dissipation is roughly P- and
    flux-type-independent on this setup),
    `bench_euler_smooth_wave_2d` + `_p3` + `_p4` + `_p5` (NP=15 / NP=21
    multi-component HLLC absolute-L2 sentinels at the Float32 floor;
    `_p5` is the highest-NP multi-component bench in the suite),
    `bench_mhd_alfven_2d`, `bench_mhd_alfven_glm_2d` + `_p3` + `_p4`
    + `_p5` (NP=10 / NP=15 / NP=21 GLM-MHD gate; the `_p5` variant
    closes the 2D MHD-GLM P-parity sweep P=2/3/4/5 at the same
    A^2~0.01 nonlinear floor as the lower P -- highest-NC 2D physics
    gate at NC=7 / NP=21 = 147 q values per element),
    `bench_mhd_glm_psi_transport_2d` + `_p3` (c_h>0 psi/Bx wave
    coupling at NP=6 and NP=10),
    `bench_mhd_glm_psi_damp_2d` + `_p3` (alpha_d>0 decay matches A0/e
    via the 2D operator-splitting damp kernel; gated at NP=6 and
    NP=10),
    `bench_shallow_water_wave_2d` + `_p3` + `_p4` + `_p5` (NP=10 /
    NP=15 / NP=21 SW HLL gate; closes the 2D SW P-parity sweep
    P=2/3/4/5 at the same A/H=0.01 nonlinear floor across all P) +
    `_rusanov` (gates the Rusanov-flux SW path used by
    `examples/shallow_water_dam_break_2d_gpu`),
    `bench_shallow_water_inflow_2d` (BC_INFLOW + BC_OUTFLOW preservation
    gate; uniform subcritical state matched to inflow ghost stays
    unchanged to ~Float32 epsilon),
    `bench_euler_channel_steady_2d`,
    `bench_euler_hydrostatic_2d` + `_p3` (closes a 2D gravity
    source-term parity gap with the 3D path: 3D Euler had
    `Euler.source_term` plumbing for (gx, gy, gz) and was gated
    by bench_euler_hydrostatic_3d, but the 2D Euler vol+lift
    kernel had no source-term path at all.  Now both dimensions
    exercise the same physics at NP=6 and NP=10.  Hydrostatic
    equilibrium preserves the rest state to ~Float32 epsilon
    over T=1).
  * **2D shocks + EM (17):** `bench_euler_sod_2d`,
    `bench_euler_sod_limited_2d` + `_p3` + `_p4` + `_p5` (HLLC + BJ
    limiter at P=2 / P=3 / P=4 / P=5; the P=2 variant lands the
    shock within 0.2 cells of Rankine-Hugoniot, the P=3 variant
    within 0.3, the P=4 variant within 0.4 with NP=15, and the
    P=5 variant within 0.4 cells with NP=21.  Higher-order capture
    keeps the shock sharp even with limiter dispersion; the trade-
    off is that the BJ stencil at NP=21 reaches deeper into the
    rarefaction tail, producing ~8% under-shoot on the left
    plateau vs ~0.04% at NP=15.  The P=5 gate exercises the split-
    kernel BJ limiter at the highest 2D-Euler NP shocked-flow
    config in the suite),
    `bench_shallow_water_dam_break_2d` (closed-pool conservation
    invariants), `bench_maxwell_cavity_2d` (TM(1,1) standing wave in
    PEC cavity, period sqrt(2), rel L2 ~3e-4),
    `bench_maxwell_plane_wave_2d` (TM plane wave traveling in +x
    on a periodic box, one full period; gates actual wave
    propagation + zero-component leakage),
    `bench_maxwell_te_plane_wave_2d` (TE-polarization dual: gates
    the previously-untested Bz / Ey flux paths in the same kernel),
    `bench_maxwell_plane_wave_2d_p3` + `_p4` + `_p5` (NP=10/15/21
    Maxwell gates -- close the P-parity gap up to P=5; rel L2 at
    Float32 floor (~5e-5 / 2e-5 / 1.8e-5 respectively) vs the P=2
    bench's 6e-4 floor; the `_p5` variant is the highest-order
    vacuum-Maxwell gate in the suite),
    `bench_maxwell_outflow_2d` + `_inflow` (uniform-state
    preservation under BC_OUTFLOW / BC_INFLOW on all four faces;
    closes the BC_INFLOW dispatch arm of
    `maxwell_face_flux_kernel_2d`, which previously fell through
    to the BC_OUTFLOW branch),
    `bench_maxwell_uniform_j_2d` + `_p3` + `_m` + `_m_p3` (uniform
    J / M on a periodic box with q=0 IC: dEx/dt = -c^2*Jx and
    dBz/dt = -Mz give exact linear ramps.  Closes a 2D feature-
    parity gap -- 3D Maxwell had `Maxwell.source_term` coupling
    J / M into the RK rhs, but the 2D vol+lift kernel had no
    source-term plumbing at all.  Now both paths exercise the same
    physics, with full P-parity on both J and M arms at NP=6 and
    NP=10).
  * **3D smooth (43):** `bench_advection_3d` + `_p3` (rate ~3.7) +
    `_p4` (rate ~4.65, NP=35) + `_p5` (rate ~5.33, NP=56),
    `bench_advection_outflow_3d` (BC_OUTFLOW x6 drainage),
    `bench_advection_inflow_3d` (BC_INFLOW + BC_OUTFLOW + BC_WALL
    steady-state),
    `bench_euler_smooth_wave_3d` + `_p3` + `_p4` + `_p5` (NP=20 /
    NP=35 / NP=56 multi-component HLLEC gates at the Float32 floor;
    the `_p5` variant is the highest-order multi-component 3D gate
    in the suite),
    `bench_euler_flux_coverage_3d` (gates the Rusanov / Roe / HLLE
    flux paths in `src/euler.mojo`; HLLEC is gated by the other
    Euler benches),
    `bench_euler_hydrostatic_3d` + `_p3` (gates the Euler gravity source --
    constant-density rest state with linear-in-z pressure remains at
    rest to ~1e-5 in v, drho/rho0, dp/p0 over T=1),
    `bench_euler_inflow_3d` (gates the BC_INFLOW arm of
    `Euler.boundary_flux`: subcritical inflow + outflow with the
    inflow ghost matched to the IC preserves uniform state to
    Float32-epsilon * step accumulation over 674 SSPRK3 steps),
    `bench_euler_vortex_3d` + `_p3` (3D Shu-Erlebacher isentropic
    vortex, z-uniform extrusion of the 2D vortex IC; rel L2 ~6.8 %%
    over one period at both P=2 / NP=10 and P=3 / NP=20, matching
    the 2D bench since F^z = 0 on z-uniform fields and the long-
    horizon dissipation floor is roughly P- and flux-type-
    independent on this setup),
    `bench_mhd_alfven_3d` + `_p3` + `_p4` + `_p5` (NP=20 / NP=35 /
    NP=56 IdealMHD gate; the `_p5` variant exercises the cooperative
    shared-memory rk_stage_kernel at NC=9 / NP=56 = 504 q values per
    element -- the largest per-element working set in the suite --
    and closes the 3D MHD P-parity sweep P=2/3/4/5 at the same
    A^2~0.01 nonlinear floor as the lower P),
    `bench_mhd_glm_psi_damp_3d` + `_p3` (3D GLM psi-damping via
    the source_term hook in rk_stage_kernel; analytic decay match
    to Float32 epsilon at both NP=10 and NP=20),
    `bench_mhd_glm_psi_transport_3d` + `_p3` (3D GLM psi/Bx
    linear-wave coupling via the regular flux kernel; rel L2 ~4e-5
    over one period at c_h=1, at NP=10 and NP=20).  Together the
    `_p3` damp + transport pair completes 3D GLM P-parity at P=3
    with the existing 2D _p3 analogs,
    `bench_maxwell_cavity_3d`,
    `bench_maxwell_plane_wave_3d` + `_p3` + `_p4` + `_p5` (TM plane
    wave on triply-periodic cube; NP=10 / NP=20 / NP=35 / NP=56;
    the `_p5` variant is the highest-order vacuum-Maxwell gate in
    the 3D suite, rel L2 ~2.3e-5 + zero-component leakage ~6e-6
    under the Kuhn-tet asymmetry floor),
    `bench_maxwell_uniform_j_3d` + `_p3` + `_m` + `_m_p3` (uniform-J /
    uniform-M source-term gates: Ex / Bz grow linearly under the
    source while flux divergences stay zero on uniform fields; both
    match analytic to ~Float32 epsilon.  Both J and M arms have full
    4-corner P/dimension parity at NP=6/10/10/20 across the 2D and
    3D variants),
    `bench_maxwell_outflow_3d` (uniform-state preservation under
    BC_OUTFLOW on all six faces; the third dispatch arm in
    `Maxwell.boundary_flux` that the cavity / plane-wave gates
    didn't exercise),
    `bench_maxwell_inflow_3d` (final BC dispatch arm: uniform
    constant state matched to the inflow ghost on all six faces;
    after this gate every BC arm in every physics has a direct
    test),
    `bench_shallow_water_wave_3d` + `_p3` + `_p4` + `_p5` (NP=20 /
    NP=35 / NP=56 SW gates; the `_p5` variant closes 3D SW P-parity
    sweep at NP=56 and sits at the same A/H=0.01 nonlinear floor as
    P=3/P=4) + `_inflow` (BC_INFLOW + BC_OUTFLOW preservation),
    `bench_two_fluid_outflow_3d` + `_walls` (charge-balanced
    17-component rest state preserved under BC_OUTFLOW / BC_WALL
    on all 6 faces),
    `bench_two_fluid_langmuir_3d`.
  * **3D shocks (7):** `bench_euler_sod_3d` + `_p3` + `_p4` + `_p5`
    (BJ-limited, bounds + mass conservation; the `_p5` / NP=56
    variant is the highest-NP shocked-flow gate in the suite --
    exercises the cooperative shared-memory rk_stage_kernel + the
    split-kernel BJ limiter at NC=5 / NP=56 / 280 q-values per tet,
    the largest shocked-flow comptime config the suite covers, and
    closes the 3D shocked-Euler P-parity sweep P=2/3/4/5),
    `bench_mhd_brio_wu_3d` + `_p3` (canonical 1D MHD Riemann embedded
    in 3D, GLM + BJ limiter; the P=3 variant is the first analytic-
    Riemann-style 3D MHD shocked gate at NP=20 / NC=9, ~408 us per
    rk_stage launch -- previously the heaviest 3D bench, since
    overtaken by `bench_mhd_alfven_3d_p5` at ~659 us / launch on
    NC=9 NP=56),
    `bench_shallow_water_dam_break_3d` (closed-pool h_L=2 / h_R=1
    Riemann in a 3D box with BC_WALL on all 6 faces and the BJ
    limiter -- first 3D SW shocked-flow gate, mirrors the 2D
    dam-break invariants at NP=10).

  Tight tolerances where the problem admits them, with each gate's
  threshold sized to ~1.4-3x the empirical error floor (catches any
  meaningful regression away from current accuracy).  Single-physics
  bugs that move L2 by more than a small constant trip the gate.

- **GLM-enabled 2D MHD** (`mhd_glm_*` kernels in
  `src/local_mesh_2d_gpu_mhd_glm.mojo`):  Dedner divergence-cleaning ported from
  3D as a parallel NC=7 path; the existing NC=6 `mhd_rk_stage_2d`
  remains the smooth-flow workhorse.  Validated end-to-end through
  `bench_mhd_alfven_glm_2d` and `bench_mhd_glm_psi_transport_2d`.
  Note: 2D shocked MHD is not gated.  A 2D Brio-Wu retry with the
  current BJ-limited GLM stack (2026-05-05) confirmed that HLLD is
  genuinely needed -- the 2D Kuhn-triangle Rusanov flux produces
  ~16x faster transverse Bx error growth than the 3D split-tet
  pattern (Bx drift ~0.6 in 2D at T=0.005 vs ~0.3 in 3D at T=0.08).
  See `project_2d_mhd_no_glm` memory for the full retry findings.

- **Profiling**: `make profile-bench-<name>` runs a benchmark under
  `nsys profile --stats=true` and saves a per-kernel time summary
  to `benchmarks/profile_reports/<name>.kern.txt` for diff-ability
  across runs.  Baseline reports live in git for every benchmark.
  `make profile-summary` (or `scripts/profile_summary.py`) reads the
  cached reports and ranks every bench by dominant-kernel avg
  us/launch -- a quick visual scan of where the per-step cost sits
  across the suite (heaviest = best per-step ROI for kernel-level
  optimisation).

- **2D kernel fusion (phase 2):** Each 2D physics path now
  runs its flux pipeline in **2 kernel launches per SSPRK3 stage**
  (down from 3): the per-face flux kernel writes `fstar` to global,
  then a single per-(elem, node) fused vol+lift+RK kernel computes
  the volume RHS contribution locally (no global vol_c round-trip),
  applies the face-lift, and does the RK update.  Implemented for
  all five 2D physics:
  * `advection_vol_lift_combine_rk_kernel_2d`     (NC=1)
  * `euler_vol_lift_combine_rk_kernel_2d`         (NC=4, Rusanov + HLLC)
  * `sw_vol_lift_combine_rk_kernel_2d`            (NC=3, Rusanov + HLL)
  * `mhd_vol_lift_combine_rk_kernel_2d`           (NC=6)
  * `mhd_glm_vol_lift_combine_rk_kernel_2d`       (NC=7, with c_h^2 B / psi flux additions)
  * `maxwell_vol_lift_combine_rk_kernel_2d`       (NC=6, EM)

  Profile measurements (smooth-flow benchmarks, NX=32-64 mesh):
  per-stage compute is **20-30%% smaller** depending on NC; launches
  per stage **3 -> 2 (-33%%)**.  All 121 analytic-solution gates remain
  bit-identical to the pre-fusion path.

  The 3D pipeline's `rk_stage_kernel` is already a single fused
  kernel that spends > 99 %% of GPU time in one launch; that pattern
  is the reference target for any further 2D consolidation.

## Numerical scheme

- **Lagrange DG at arbitrary P (validated P=1..5)** on tetrahedra.
  Node count is `(P+1)(P+2)(P+3)/6` (10 at P=2, 20 at P=3, 35 at P=4,
  56 at P=5); operators are constructed at compile time via Vandermonde
  inverse + analytic Dirichlet-formula integration in
  `src/reference.mojo`.  P=2 emits `VTK_QUADRATIC_TETRA` (cell type 24)
  for direct ParaView compatibility; P>=3 emits `VTK_LAGRANGE_TETRAHEDRON`
  (cell type 71).  `make test-vtu-meshio` (or `scripts/validate_vtu.py`)
  validates VTK-spec node ordering at every P: corners distinct,
  edge interiors at parametric `(j+1)/P` spacing, face interiors on
  face planes, volume interiors strictly inside.  The same script
  also handles 2D triangle VTUs (`triangle6` at P=2, cell type 69
  `VTK_LAGRANGE_TRIANGLE` at P>=3) when invoked directly on
  `output/*.vtu` from a 2D-GPU driver.
- **Kuhn 6-tet decomposition** of a Cartesian cell grid. Each cube
  owns 12 uniquely numbered faces (6 interior diagonal + 6 external on
  its +x/+y/+z boundaries). Face IDs are `owner_cell * 12 + face_type`,
  giving a Dict-free mesh build.
- **Canonical face-node ordering by owner-cell cube-corner index**
  (ascending). Because Kuhn tets are translation-invariant, this makes
  the `(tet, local_face, side) → element-local node` mapping a set
  of small precomputed tables — no per-cell orientation bookkeeping
  and no special cases for periodic-wrap cells.
- **Multi-component conserved state**: `q[(e*N_P + i)*NC + c]` where
  `NC = PhysT.NUM_COMPONENTS`.
- **Upwind / wave-based numerical flux**: the physics type's
  `numerical_flux(q_l, q_r, n, flux)` hook writes the NC-vector flux at
  a face given the outward normal. Advection uses plain upwind; Euler
  rotates into the face-normal frame, solves the chosen 1D Riemann
  problem, and rotates back.
- **Fused RK-stage kernel**: one GPU kernel computes the numerical
  flux, volume DG term, and SSPRK3 linear combination per thread, per
  stage. No intermediate `rhs` / `face_flux` scratch buffers. Launched
  three times per timestep. Generic over NC and physics type, so the
  compiler emits one specialized kernel per physics module.
- **Affine tets**: inverse Jacobian and `1/(6V)` are constants per
  tet-type; the kernel reads them from per-element arrays filled by
  the build-time tet-type tables.

## System architecture

Roughly 16k lines of Mojo + a thin MPI shim in C. Core components live
under `src/`; problem-specific drivers live under `examples/`.

| file                                | lines | role                                                                                   |
|-------------------------------------|-------|----------------------------------------------------------------------------------------|
| `src/reference.mojo`                |   826 | Reference element: arbitrary-order equispaced Lagrange (Vandermonde inverse + analytic integration), `D_ref`, `Lift_ref`, `face_to_elem` table |
| `src/local_mesh.mojo`               |  1649 | Raw periodic Kuhn-tet mesh builder + BC overlay kernels, **GPU-resident**              |
| `src/mesh.mojo`                     |  1015 | Patch-aware `Mesh`: `LocalMesh` + partition / ghost ring / permutation + BC filtering  |
| `src/boundary.mojo`                 |    93 | `BoundaryConditions` + BC_WALL / BC_OUTFLOW / BC_INFLOW constants + 2D variant         |
| `src/partition.mojo`                |   226 | `(PX, PY, PZ)` factorisation of nprocs, minimising ghost-exchange surface              |
| `src/halo_exchange.mojo`            |   517 | MPI pack / Isend / Irecv / unpack + per-direction BC skip flag                         |
| `src/solver.mojo`                   |  1291 | `Physics` trait, cooperative `rk_stage_kernel`, `Solver[PhysT, P]`, SSPRK3 stepper, BJ limiter pipeline |
| `src/advection.mojo`                |   151 | `Advection`: scalar upwind flux + wall / outflow / inflow BC                           |
| `src/euler.mojo`                    |   989 | `Euler`: 5-moment, 4 Riemann solvers, entropy fix, face rotation, gravity source       |
| `src/maxwell.mojo`                  |   281 | `Maxwell`: vacuum E+B, Rusanov, full BC menu, uniform J / M source                     |
| `src/shallow_water.mojo`            |   200 | `ShallowWater`: 2D shallow water embedded in 3D                                        |
| `src/mhd.mojo`                      |   461 | `IdealMHD`: single-fluid MHD + Dedner GLM div(B) cleaning                              |
| `src/two_fluid.mojo`                |   627 | `FiveMomentTwoFluid`: electron + ion + Maxwell + GLM, 17 components                    |
| `src/vtu.mojo`                      |   599 | Zero-copy binary-appended VTU writer (one scalar field per frame)                      |
| `src/async_writer.mojo`             |   182 | pthread-based `writev()` scatter-gather file writer                                    |
| `src/frame_writer.mojo`             |   286 | Per-rank frame output: `FrameWriter[PhysT, P]`, auto rank-subdir + `mkdir -p` at init  |
| `src/diagnostics.mojo`              |   202 | `DiagnosticsWriter[PhysT, P]`: domain-integrated linear / squared / max-abs per-frame with allreduce |
| `src/time_integrator.mojo`          |   180 | `run_ssprk3_loop` and `run_ssprk3_loop_with_diagnostics`                               |
| `src/nvtx.mojo`                     |    81 | Runtime-loaded NVTX shim for Nsight Systems timelines                                  |
| `src/mpi.mojo` + `src/mpi_shim.c`   |   ~300| Mojo / C-shim bindings for OpenMPI (init, point-to-point, allreduce, request handling) |

The 2D GPU stack lives in a separate set of modules (single-rank, no
HaloExchange):

| file                                          | lines | role                                                                            |
|-----------------------------------------------|-------|---------------------------------------------------------------------------------|
| `src/reference_2d.mojo`                       |   324 | 2D reference triangle: equispaced Lagrange, `D_ref`, `Lift_ref`, edge node maps |
| `src/reference_2d_gpu.mojo`                   |    80 | Float32 device mirror of `ReferenceElement2D` + `device_bytes()` helper         |
| `src/local_mesh_2d.mojo`                      |   433 | Periodic Kuhn-2-tri (per cube halved on diagonal) mesh + BC overlay             |
| `src/local_mesh_2d_gpu.mojo`                  |   231 | `LocalMesh2DGpu[P]` upload + mass-matrix-weighted `cell_mean_kernel_2d` + `device_bytes()` helper |
| `src/local_mesh_2d_gpu_advection.mojo`        |   422 | 2D scalar advection (NC=1)                                                      |
| `src/local_mesh_2d_gpu_euler.mojo`            |   777 | 2D Euler (NC=4): Rusanov + HLLC, gravity (gx,gy) source                         |
| `src/local_mesh_2d_gpu_sw.mojo`               |   636 | 2D Shallow Water (NC=3): Rusanov + HLL                                          |
| `src/local_mesh_2d_gpu_mhd.mojo`              |   518 | 2D plain ideal MHD (NC=6, no GLM)                                               |
| `src/local_mesh_2d_gpu_mhd_glm.mojo`          |   611 | 2D MHD + Dedner GLM divB cleaning (NC=7)                                        |
| `src/local_mesh_2d_gpu_maxwell.mojo`          |   470 | 2D Maxwell (NC=6 EM): Rusanov + PEC reflection + J/M source                     |
| `src/local_mesh_2d_gpu_limiter.mojo`          |   259 | 2D Barth-Jespersen slope limiter (Venkat-smoothed): split into `cell_mean` + `compute_theta` + coalesced `apply` kernels |
| `src/vtu_2d.mojo`                             |   475 | 2D VTU frame writer (incl. `dump_vtu_2d_frame_multi`) + `dump_pvd_collection` + `vtu_frame_name` helpers |
| `src/sod_exact_riemann.mojo`                  |   197 | Toro-style analytic Riemann solver for the rare+contact+shock structure (zero initial velocities); used by the 5 Sod benches and unit-tested against canonical Sod (Toro 2009 Test 1) and strong-shock (Toro 2009 Test 3) reference values |
| `src/ssprk3.mojo`                             |   112 | Non-templated SSPRK3 stage-plan helper shared by every 2D-GPU example + bench   |
| `src/memory_report.mojo`                      |   253 | `MemoryReport` + `ThroughputReport` for WARPXM-style perf-introspection output  |

Example drivers exercise various combinations of physics, BC kind,
and diagnostics; see the list at the top of this README.

### The `Physics` trait

Each physics module implements a tiny interface:

```mojo
trait Physics(Copyable, Movable, ImplicitlyDestructible, DevicePassable):
    comptime NUM_COMPONENTS: Int

    def internal_flux(
        self, q, flux,
    ) -> Float32: ...      # writes flux[d * NC + c] = F_d_c(q)

    def numerical_flux(
        self, q_l, q_r, nx, ny, nz, flux,
    ) -> Float32: ...      # writes NC-vector two-sided Riemann flux

    def boundary_flux(
        self, q_int, bc_type, nx, ny, nz, flux,
    ) -> Float32: ...      # Riemann flux against a BC-synthesised ghost

    def source_term(
        self, q, x, y, z, source_out,
    ): ...                 # pointwise S(q, x) added to RHS at every node
```

`DevicePassable` is required because the physics instance is passed
*by value* to the RK-stage kernel; the compiler copies the struct into
kernel launch parameters, inlines every method call into the kernel,
and emits one specialized kernel per `(NC, PhysT)` pair.

### Everything that lives on the GPU

After construction, no host-side arrays indexed by element, face, or
DOF survive past startup. All of the following are device-resident:

- **Mesh** (`Mesh` struct, populated by two build kernels):
  - 10 node coordinates per element (`elem_node_xyz`)
  - inverse Jacobian + `1/(6V)` per element
  - 4 face indices, sides, and canonical→reference permutations per element
  - per-face: 2 elements, 12 per-side node indices, normal, area
- **DG operators** (uploaded once from host, ~540 Float32s total):
  - `D_ref[3][10][10]` = `M_ref⁻¹ · S_ref^k` (volume)
  - `Lift_ref[4][10][6]` = `M_ref⁻¹ · L_ref^f` (face)
- **Solution state**: three `q` buffers for SSPRK3
  (each `num_elements * N_P * NC` Float32s)
- **Initial condition**: each driver owns an IC kernel that writes
  `d_q` directly, reading the already-resident node coordinates.

The only host-side array of note is a single pointer to `elem_node_xyz`
that `VtuWriter` references by pointer (zero-copy) when streaming VTU
frames out to disk.

### VTU output

The VTU writer emits one scalar (named `"density"`) per frame. Drivers
use `solver.download_component(c, buf)` to extract a single component
from the multi-component `q` for visualization. For Advection, `c = 0`
is the whole solution; for Euler, `c = 0` is the mass density ρ.

### The I/O pipeline

Per-frame writes use a pthread-based `AsyncWriter` that issues
`writev()` with 6 scatter-gather segments per frame:
`[xml_header][density(owned)][pts_count][elem_node_xyz(ref)][conn+offsets+types][xml_tail]`.
Only the density section is copied per frame; the mesh coord segment
is a pointer into `Mesh`'s host-side download. Up to
`max_concurrent=8` writer threads in flight, joined lazily in the
next `submit()` or at `wait_all()`.

## Diagnostics & visualization

### Per-frame CSV

Drivers register conservation-law integrals via `DiagnosticsWriter`:

```mojo
var linear = List[NamedComponent]()
linear.append(NamedComponent("mass",         0))
linear.append(NamedComponent("momentum_x",   1))
linear.append(NamedComponent("total_energy", 4))

var squared = List[NamedComponent]()
squared.append(NamedComponent("Bx_sq", 5))     # magnetic energy tracker

var max_abs = List[NamedComponent]()
max_abs.append(NamedComponent("max_abs_psi", 8))   # GLM div(B) noise

var diag = DiagnosticsWriter[IdealMHD](
    solver, "output/diagnostics.csv",
    linear, squared, max_abs, LX, LY, LZ,
)
var result = run_ssprk3_loop_with_diagnostics[IdealMHD](
    solver, writer, diag, dt, T_FINAL, NUM_FRAMES, nvtx,
)
```

At np>1 each value is gathered across ranks via `MPI_Allreduce` and
rank 0 appends one CSV row per frame. The writer supports three
reduction kinds:
- `linear` — `∫q[c] dV` (conservation-law integrals: mass, momentum,
  total energy).
- `squared` — `∫q[c]² dV` (L²² norms, EM / magnetic / kinetic energy
  components).
- `max_abs` — `max_x |q[c]|` (peak-value tracker for shocks, div(B)
  noise, overshoots).

### Animated dashboard

`scripts/animate_dashboard.py` reads the VTU frame series and the
CSV, and emits `output/dashboard.gif` with a 2×2 layout:
- field panel: `tricontourf` of the leading scalar on a thin-z slice
- three time-series panels, auto-grouped from the CSV column names
  (mass / momentum / everything-else)

Frames are loaded lazily (one VTU at a time), so RAM usage scales
with a single frame, not with the run length. A 32³ Taylor-Green
dashboard with 81 frames peaks at under 1 GB of resident memory.

```bash
./euler_rising_bubble
pixi run python scripts/animate_dashboard.py
# ... writes output/dashboard.gif
```

## Performance

### Built-in perf introspection

Every 2D-GPU and 3D example driver prints two reports per run -- a
WARPXM-Kokkos-style device-memory table at startup and a sync'd
throughput summary at the end -- so users can see where GPU memory
goes and how fast the compute kernels are running without reaching
for an external profiler.

Sample output from `./euler_vortex` at 32³ (RTX 3090):

```
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
...
=== Step-loop throughput ===
  steps:                50
  DOF / step:           9830400
  wall time:            388.8 ms
  per-step wall:        7.8 ms
  throughput:           1.3 GDOF/s
  state bytes / step:   300.0 MB
  state bandwidth:      37.7 GB/s  (lower bound; excludes operator + mesh reads)
============================
```

The memory breakdown is exact (computed from each allocation's
known shape -- no probe / sampling).  The throughput report comes
from `Solver.bench_step_loop()`, which runs a 5-step warmup + 50-
step measurement loop with `ctx.synchronize()` at both endpoints
so the elapsed wall covers GPU compute (not async enqueue).  See
`src/memory_report.mojo` for the structs (`MemoryReport`,
`ThroughputReport`) and `src/solver.mojo` for the helper methods
(`Solver.memory_report()`, `Solver.dof_count()`,
`Solver.state_bytes_per_step()`, `Solver.bench_step_loop()`).

The reported state-bandwidth is a true LOWER BOUND on achieved
DRAM bandwidth: it counts only the bytes the SSPRK3 kernel sequence
MUST move between launches (8 × total_q_len × Float32 per step,
derived from the 2-3-3 read/write pattern of the three RK stages),
and excludes operator + connectivity reads (small, cache-friendly).
Achieved bandwidth divided by device peak (e.g. ~1 TB/s on RTX
3090) gives a quick "how memory-bound is this run" intuition.
NC=1 advection routinely hits >100 GB/s state bandwidth at this
mesh size; NC=5 Euler runs at ~40 GB/s on the same hardware
because each loaded byte fuels more arithmetic per element.

### advection_gaussian at 48³ (RTX 3090, WSL2)

End-to-end wall clock for a full 20-frame run at 48³ (663 K tets,
6.6 M DOF, 2080 SSPRK3 steps, 26 MB density per VTU file):
**~5.6 s**, of which ~5.4 s is download + VTU frame I/O.

Sync'd post-run measurement (`Solver.bench_step_loop`) reports
**3.3 GDOF/s** throughput and **~97 GB/s state bandwidth** (lower
bound; ~10 % of the RTX 3090's ~1 TB/s peak DRAM).  The fused
RK-stage kernel additionally achieves **98% of peak L1-cache
throughput** according to Nsight Compute at 48³ (scalar advection
specialization).

### euler_vortex at 32³ (RTX 3090, WSL2)

End-to-end wall clock for a 10-frame run at 32³ (196 K tets, 2.0 M DOF
× 5 components = 9.8 M unknowns, 280 SSPRK3 steps):
**~2.8 s** total, of which ~2.5 s is download + VTU frame I/O,
~0.2 s is the final sync, and ~0.01 s is async kernel enqueue.

Sync'd post-run measurement (`Solver.bench_step_loop`, 50 steps
after warmup) reports **1.2 GDOF/s** throughput and **~36 GB/s
state bandwidth** (lower bound; ~3.5 % of the RTX 3090's ~1 TB/s
peak DRAM).  The mean density drifts by ~2·10⁻⁶ over the 280-step
run — well within single-precision round-off expectations.

Euler's RK-stage kernel is substantially heavier than advection's
(HLLEC Riemann solver + 3×3 face rotation matrix construction per
face node, 5-component accumulators everywhere), so its bandwidth
share is lower despite NC=5: each byte loaded fuels significantly
more arithmetic per element.

### Limiter cost share scales differently in 2D vs 3D

Per-kernel breakdown on the limited Sod path at P=3 / NP=10 in 2D
(per nsys, ~5300 SSPRK3 steps × 3 launches per stage), after the
compute_theta + apply split (commit `cdc3210`):

| kernel                     | % of GPU time |
|----------------------------|---------------|
| `vol+lift+rk` (Euler)      | 36.9%         |
| `bj_limit_compute_theta_2d`| 34.8%         |
| `face_flux` (HLLC)         | 12.5%         |
| `bj_limit_apply_2d`        |  9.2%         |
| `cell_mean_kernel_2d`      |  6.6%         |

The 2D limiter pipeline tracks ~44 / 48 / 50% of GPU time at
P=3 / 4 / 5 -- it grows with NP because the inner per-element
`compute_theta` loop is O(NP).  The apply-pass uncoalesced
pattern that previously dominated has been neutralized:
adjacent threads now share `elem` and stride-1 writes to `q[]`.

The same split was mirrored to 3D in commit `0435ad6`, but the
cost story flips: in 3D the cooperative `rk_stage_kernel` grows
~quadratic with NP, so at the highest P the limiter is
proportionally tiny:

| kernel                  | % at 3D P=3 | % at 3D P=5 |
|-------------------------|-------------|-------------|
| `rk_stage_kernel`       |  82.1%      |  87.7%      |
| `bj_limiter compute_theta` |  14.2%   |  10.4%      |
| `bj_limiter apply`      |   2.0%      |   1.0%      |
| `compute_cell_average`  |   1.7%      |   0.8%      |

This means: 2D limiter optimization has ~3-4x the ROI of the
same effort in 3D, where the cooperative kernel is the real
bottleneck.  Possible 2D paths (see `project_2d_limiter_perf`):
pre-pack density for coalesced compute_theta reads (TRIED, was a
regression because L2 absorbs the uncoalesced cost); fuse
cell_mean into the rk_stage kernel.  See
`benchmarks/profile_reports/` for per-kernel measurements on every
benchmark.

### Cross-bench profile summary

`make profile-summary` (or `scripts/profile_summary.py`) parses the
121 cached profile reports and ranks every bench by dominant-kernel
cost.  Use the default `avg us/launch` view to spot kernel-level
optimisation targets; switch to `--sort total` to see where the
suite-wide kernel budget actually sits (top of the avg list isn't
always top of the total list).

Sample top-5 (RTX 3090, current baselines):

```
top 5 benches, sorted by avg us/launch:
  bench                            kernel        avg us    inst   total ms
  -------------------------------- ----------    ------ ------- ----------
  bench_mhd_alfven_3d_p5           rk_stage       659.1    4953     3264.5
  bench_maxwell_plane_wave_3d_p5   rk_stage       621.3    3303     2052.1
  bench_euler_sod_3d_p5            rk_stage       608.8    4998     3042.8
  bench_mhd_brio_wu_3d_p3          rk_stage       408.9    3807     1556.8
  bench_euler_vortex_3d_p3         rk_stage       392.3   11403     4473.2
  -> shown 5 benches: 14.39 s of dominant-kernel time;
     full suite (62.28 s across all benches dominant-kernel-only)
```

The full-suite footer (`62.28 s`) is the irreducible compute budget
for `make profile-bench-all` (excluding nsys profile + launch
overhead which dominates wall time).  3D dominates (58.7 s) over 2D
(3.6 s) by ~16x.

Other useful flags:

* `--by-physics` rolls up per-physics totals (Euler 60% / MHD 15% /
  Two-Fluid 9% / Advection 7% / Maxwell 5% / SW 3% on the current
  baselines) -- size where the suite-wide compute budget actually
  sits across the 121 benches.
* `--show-cv` adds a CV% column (stddev / avg).  Distinguishes
  refinement-sweep benches (CV ~50-66%, expected) from single-
  resolution benches (CV <1%).  Combined with `--by-physics`, the
  rollup gains a `max CV %` column that surfaces refinement-sweep
  modules at a glance (e.g. euler / advection at >60% on current
  baselines, vs mhd / two-fluid / maxwell <14%).
* `--csv` emits machine-readable output for downstream tooling
  (`bench,kernel,avg_us,instances,total_ms,cv_pct`).  Mutually
  exclusive with `--markdown`.
* `--markdown` emits a markdown table -- handy for pasting into PR
  descriptions / issues / docs.  Composes with `--show-cv` and
  `--by-physics`; mutually exclusive with `--csv`.
* `--filter <substr>` narrows by bench-name substring; `--kernel
  <substr>` narrows by kernel-name substring; `--sort
  {avg,total,inst}` picks the ranking key.
* `--self-test` (also `make profile-summary-test`) runs an in-script
  parser + classifier check (24 assertions, sub-second) -- catches a
  future nsys-output-format change.  Wired into `make pre-push`.

For refresh-then-rank workflows, `make profile-bench-quick`
re-baselines just the 13 bench-quick gates (~10-15 min) -- much
faster than `profile-bench-all` (~90 min) when the goal is a
sanity-check re-rank after a kernel-level change.

## Build & run

### Prerequisites

- NVIDIA GPU with compute capability 7.5+ (tested on RTX 3090).
- CUDA 12.x installed (for `libnvtx3interop`, `ncu`, `nsys`).
- [`pixi`](https://pixi.sh/) to install Mojo + the Python helpers
  (`meshio`, `scipy`).
- A C toolchain (linker). On Linux, `libm` and `libpthread` via the
  system `glibc`.
- For multi-rank runs (`mpi_hello`, `mpi_partition`,
  `mpi_patch_mesh`, `mpi_halo_pingpong`, plus every 3D physics
  driver under `examples/*.mojo` -- they all support np>=2 via
  the 3D Mesh + HaloExchange + Solver path): OpenMPI 4.x
  (`apt install openmpi-bin libopenmpi-dev` on Ubuntu) and
  `mpicc` on PATH.

### Install Mojo

The project is managed by [pixi](https://pixi.sh/); the Mojo nightly
channel + the Python helpers are wired into `pyproject.toml`.

```bash
cd /path/to/mojoxm
pixi install                    # materializes .pixi/envs/default
pixi run mojo --version         # sanity check
```

The compiler binary ends up at `.pixi/envs/default/bin/mojo`, but
that bare path won't find its stdlib without `MODULAR_HOME` —
go through `pixi run mojo` (what the Makefile defaults to) or
drop into a subshell with `pixi shell`.

### Compile a driver

`src/` is a Mojo package (it contains `__init__.mojo`), so drivers
import the core modules as `from src.solver import Solver`, etc.
Everything builds through a `Makefile` at the project root:

```bash
make               # build every driver (CPU + GPU, MPI + non-MPI)
make cpu           # just the MPI drivers that don't touch the GPU
make <driver>      # e.g. `make mpi_hello`
make test          # periodic MPI correctness: np=1 vs np=4 (bit-identical)
make test-bc       # non-periodic BC correctness: np=1 vs np=4 (bit-identical)
make test-reference  # host-side reference-element unit test (P=1..5)
make test-klone    # test dispatched through scripts/klone-run on the cluster
make clean
make help
```

Key build flags baked into the Makefile:

- `-O3` — full optimization.
- **`-g0` is critical** — the default Mojo debug info inflates register
  pressure from 40 → 114 per thread in the advection RK kernel and drops
  ncu's reported memory throughput from 98% to 8%.
- `-I .` — with `src/__init__.mojo` in place, lets drivers import as
  `from src.solver import Solver`.

Every driver links `build/mpi_shim.o` (built once by the `shim`
target) against the system's `libmpi`.  There is no "MPI vs non-MPI"
driver distinction any more -- np=1 is just a one-rank MPI
communicator with no ghost ring and no halo exchange.  Run any driver
as:

```bash
./advection_gaussian                          # single rank
mpirun -np 8 ./advection_gaussian             # eight ranks
mpirun -np 4 ./euler_taylor_green             # four ranks
mpirun -np 8 ./mpi_hello                      # pure MPI smoke test
mpirun -np 8 ./mpi_patch_mesh                 # per-rank Mesh build
mpirun -np 8 ./mpi_halo_pingpong              # end-to-end halo exchange
```

### Module stack, bottom-up

| Module | Role |
|---|---|
| `src/local_mesh.mojo` | Raw periodic Kuhn-tet mesh builder over an `(nx, ny, nz)` cube grid. Knows nothing about partitions; used internally by `Mesh`. |
| `src/partition.mojo` | Picks the `(PX, PY, PZ)` factorisation of `nprocs` that minimises per-patch surface area. 6 face-neighbours form a 3D torus under triply periodic BCs. |
| `src/mesh.mojo` | User-facing `Mesh`: wraps `LocalMesh` with patch / ghost-ring / owned-element metadata. At np=1 the ghost ring is omitted entirely (`ghost_width=0`), making single-rank runs a no-op on anything MPI-related. At np>1 each rank builds `(nx, ny, nz)` owned cubes + a 1-cube ghost ring, and owned elements are classified as **interior** or **halo** then reordered so both subsets are contiguous in element-id space. |
| `src/halo_exchange.mojo` | Pack / `MPI_Isend` + `MPI_Irecv` / unpack on the 6 face rings. Host-staged (OpenMPI 4.1.6 on this system isn't CUDA-aware; pinned `HostBuffer` is used as the staging layer). Short-circuits at np=1 where there are no ghost elements. |
| `src/solver.mojo` | `Solver[PhysT]` built on `Mesh` + `HaloExchange`. `rk_stage_kernel` is the cooperative-shared-memory kernel; thanks to the element reordering in `Mesh`, each RK stage dispatches over a contiguous `[elem_base, elem_base + num_elems)` range with no scatter indirection. `step_ssprk3` runs one kernel per stage at np=1, and the classical split-kernel / MPI-overlap pattern at np>1. |

End-to-end correctness is verified by three tests:

- `make test` — periodic Gaussian advection at np=1 vs np=4 must be
  **bit-identical** after 50 SSPRK3 steps (`max |a - b| = 0` over
  196,608 elements × 10 DOFs).
- `make test-bc` — same structure but with `BC_OUTFLOW` on all six
  domain faces, so every rank sees a different mix of periodic peer
  boundaries vs non-periodic global boundaries. Also **bit-identical**.
- `make test-reference` — host-side unit test of the arbitrary-order
  Lagrange reference element at P=1..5.  Six invariants per P: size
  closed-forms, mass matrix SPD, node positions inside the reference
  simplex with pairwise separation >= 1/(2P), face-to-element lookup
  coverage, `node_weights` partition-of-unity (load-bearing for the
  BJ limiter's cell-mean conservation; closed-form spot check at
  P=2), and a direct math check on the `mat_inv` Gauss-Jordan helper
  that underpins the Vandermonde inversion.

At np>1 each SSPRK3 stage runs in **split-kernel / MPI-overlap mode**:

```
halo.submit_pack(q)                # pack + D→H + MPI_Isend/Irecv (non-blocking)
rk_stage_kernel(interior=[0, NI))  # runs on default stream while MPI progresses
halo.complete_exchange(q)          # MPI_Waitall + H→D + unpack
rk_stage_kernel(halo=[NI, NI+NH))  # reads ghost q on default stream
```

so interior compute on the device overlaps with MPI progress on the
host.  At np=1 the whole sequence collapses to a single kernel launch
over `[0, num_owned)` (no pack, no MPI, no split).  On a single-GPU
box wall time grows with rank count because all ranks time-share one
GPU and MPI is host-staged; on a multi-GPU cluster with CUDA-aware
MPI that relation inverts.

### Running on a cluster: Apptainer image for Klone

Klone (UW Hyak) is Rocky Linux 8.9 with glibc 2.28 on both login and
compute nodes, but the Modular nightly Mojo wheel requires glibc
≥ 2.34 (`manylinux_2_34_x86_64`).  `mojo.def` at the project root is
an Apptainer definition that wraps Mojo in an Ubuntu 24.04 base
(glibc 2.39) and layers just enough compiler + OpenMPI-dev to
cross-compile mojoxm drivers against the host's `cuda/12.9.1` +
`ompi/4.1.6-2` modules.

**One-time build** (takes ~3 min on the login node; no GPU required
for this step):

```bash
apptainer build --fakeroot mojo.sif mojo.def
```

**Build and run in one command** via `scripts/klone-run`.  The
wrapper requests an srun allocation, compiles the driver inside that
allocation (so Mojo's generated GPU code matches the actual GPU
assigned), and then launches the binary under `mpirun` — all in a
single command.  Compiling every run trades ~15 s of startup for
immunity to the `CUDA_ERROR_NO_BINARY_FOR_GPU` you otherwise hit when
SLURM moves you between GPU generations.

```bash
# CPU-only MPI smoke test (no GPU requested, runs fast):
NP=4 scripts/klone-run mpi_hello

# GPU + MPI, pinned to A40:
NP=4 GPU=a40 scripts/klone-run advection_gaussian

# GPU + MPI, any GPU matching a SLURM constraint expression:
NP=4 CONSTRAINT='a100|a40|l40|l40s' scripts/klone-run advection_gaussian

# Single-rank GPU driver (no MPI):
scripts/klone-run euler_taylor_green
```

Env vars recognised by the script: `NP`, `GPU`, `CONSTRAINT`, `TIME`,
`ACCOUNT`, `PARTITION`, `MEM`, `SIF`, `NO_BUILD`.  `CONSTRAINT` wins
over `GPU` when both are set.  Run `scripts/klone-run` with no
arguments to see usage.

Verified on a Klone A40 at `NP=4`: wall time ≈ 40 s (≈ 18 s build +
20 s sim), conservation drift 3.56·10⁻⁵, overshoot 1.003426 —
identical to the WSL2 RTX 3090 run.

**Known limitations and future work:**

- Mojo 0.26.2 doesn't expose stream-targeted `enqueue_copy` or
  cross-stream `wait_event`, so the device-side D↔H transfers still
  serialise with the RK kernel on the default stream. The overlap
  today is specifically *host MPI* ↔ *device interior compute*.
- OpenMPI on this system isn't CUDA-aware, so `HaloExchange`
  host-stages through pinned `HostBuffer`s. With CUDA-aware MPI, the
  D↔H stage can be dropped.
- **Element reordering** (WARPXM-style) would make each face-ring
  contiguous in `q` so pack/unpack becomes a `memcpy` — and under
  CUDA-aware MPI could skip pack entirely. Worth ~10–20 µs/stage
  here; a big deal on a multi-GPU cluster. Not implemented yet.

### Run

```bash
./advection_gaussian        # writes output/frame_NNNNN.vtu + solution.pvd
./euler_vortex              # same output layout, density field
./euler_taylor_green        # density tracks the vortex pressure field
```

Open `output/solution.pvd` in ParaView.

### Writing a new physics / driver

1. Create `src/my_physics.mojo` with a struct conforming to `Physics`
   (`NUM_COMPONENTS`, `internal_flux`, `numerical_flux`) plus the
   three `DevicePassable` plumbing items (`device_type`,
   `_to_device_type`, `get_type_name`). See `src/advection.mojo` for
   the minimal example.
2. Create `examples/my_sim.mojo` with a `main()` that builds a `Mesh`,
   an instance of your physics, and a `Solver[MyPhysics]`, plus an
   initial-condition kernel you launch once. Copy the structure of
   `examples/advection_gaussian.mojo`.
3. Compile with `-I .` from the project root — no changes to
   `solver.mojo`, `mesh.mojo`, or `vtu.mojo` are needed.

## Profiling

### Nsight Systems (CPU timeline + GPU kernels)

```bash
nsys profile --trace=nvtx,cuda --output=trace ./advection_gaussian
nsys stats --report nvtx_pushpop_sum --report cuda_gpu_kern_sum trace.nsys-rep
nsight-sys trace.nsys-rep          # or open in the GUI
```

The code is instrumented with NVTX ranges at every interesting phase
(`build_mesh`, `solver_setup`, `initial_condition`, `ssprk3_step`,
`rk_stage_{1,2,3}`, `frame_boundary_sync`, `write_frame`,
`vtu_build_segments`, `vtu_submit`, `download_q` / `download_component`,
`wait_async_writes`, …) so the timeline view tells a readable story.

NVTX support is **runtime-optional** — `src/nvtx.mojo` does a `dlopen`
of `libnvtx3interop.so.1` (shipped with CUDA 12), and if the library
isn't present every call becomes a silent no-op, the same model NVTX
itself uses when no profiler is attached.

### Nsight Compute (kernel analysis)

```bash
ncu --kernel-name regex:rk_stage --launch-count 3 --launch-skip 10 \
    --set detailed ./advection_gaussian
```

The fused RK-stage kernel dominates execution time in both drivers.

### Cached cross-bench summaries

`make profile-bench-<name>` and `make profile-bench-all` save
nsys's `cuda_gpu_kern_sum` output into `benchmarks/profile_reports/
<name>.kern.txt` (committed to git as a baseline).  `make
profile-summary` (or `scripts/profile_summary.py`) ranks the cached
reports by avg us/launch / total ms / instance count.  See the
"Cross-bench profile summary" subsection above under Performance
for sample output.

## Design decisions worth knowing

### Physics as a trait, not a virtual-call interface

`Solver[PhysT: Physics]` is parametric on the physics type. Every RK
kernel launch is a *specialization* on `PhysT`, so the compiler can
inline `physics.internal_flux` / `physics.numerical_flux` into the
kernel body. No vtable, no runtime branch on physics — the HLLEC
solver compiles down to the same PTX it would if it were written
inline as a scalar advection specialization.

This is why the physics struct has to be `DevicePassable` — the
instance travels to the device as part of the kernel launch
parameters, so every field read in `internal_flux` / `numerical_flux`
is a register load, not a memory fetch.

### The Mojo 0.26.2 `ByteBuf` `__del__` workaround

`ByteBuf` in `src/vtu.mojo` deliberately has **no `__del__`**. The
Mojo 0.26.2 compiler was observed to emit spurious destructor calls
on struct values that were still reachable through another path,
which caused the 109 MB static mesh buffer to be freed while a writer
thread was mid-`writev()`. Since the three `ByteBuf`s in `VtuWriter`
have process-lifetime scope, the one-time leak at exit is harmless.

### Cube-corner canonical ordering (not sorted-global-ID)

Earlier versions of the mesh builder canonicalized shared faces by
sorting the three global vertex IDs. At periodic boundaries the
wrap flips the sort order, which silently broke the
canonical-to-element-node permutations — constant fields still
propagated (no information in the node-identity mapping), but any
non-constant field blew up. The current mesh builder uses the **owner
cell's cube-corner indices** as the canonical ordering, which is
position-independent and therefore works everywhere.

### Async VTU writing via pthread + `writev`

Frame writes were originally synchronous and a 48³ simulation spent
more time in disk I/O than in GPU compute. `src/async_writer.mojo`
issues `pthread_create` with a Mojo callback function pointer (which
works on x86-64 because thin Mojo function types match the C ABI for
our pointer-only signatures), and the writer threads call `writev()`
on scatter-gather segment lists so the zero-copy mesh-coords segment
can be referenced in place without ever going through a staging
buffer. `cuMemAllocHost` is avoided on the device→host path too.

## Limitations

- **Cartesian block mesh only.** Unstructured tet meshes from GMSH /
  etc. would need a different `Mesh` that loads from file and computes
  face-node mappings via the sorted-global-ID scheme (with care at
  periodic boundaries).
- **Float32 everywhere.** RTX 3090 FP64 is 1/64 of FP32, so FP32 is
  the right call for performance, but some applications may need FP64
  mass-matrix inversion or accumulation.
- **2D MPI not implemented.** The 2D triangular GPU stack runs at
  np=1 only; the 3D Mesh + HaloExchange + Solver path supports np>=2
  via `make test`/`test-bc`.
- **2D MHD shocked-Riemann problems are unsupported.** The
  `mhd_glm_*` kernels in `src/local_mesh_2d_gpu_mhd_glm.mojo` add
  GLM as an opt-in NC=7 path (gated by alfven-via-GLM at P=2/3/4/5,
  psi-transport at P=2/3/4/5, and psi-damp at P=2/3/4/5 in both 2D
  and 3D).  Shocked 2D MHD
  (Brio-Wu, OT vortex, ...) is not gated.  A 2D Brio-Wu retry
  2026-05-05 with the current BJ-limited GLM stack confirmed that
  HLLD is genuinely needed: the run no longer NaNs immediately
  (BJ keeps integration alive at very short T) but Bx drifts ~0.6
  even at T=0.005 vs ~0.3 in 3D at T=0.08.  The 2D Kuhn-triangle
  Rusanov flux produces ~16x faster transverse Bx error than the
  3D split-tet pattern, so the Rusanov-only path can't reproduce
  the seven-wave Brio-Wu structure cleanly enough.  HLLD is the
  right next step if 2D shocked-MHD coverage is wanted.
- **2D pipeline still uses 2 kernel launches per RK stage** vs 1 in
  the 3D `rk_stage_kernel`.  The vol+lift+RK fusion in commits
  f7bff49..fa4257a brought 2D from 3 launches/stage down to 2
  (per-face flux + per-element fused vol+lift); collapsing into a
  single launch would require unifying the per-face and per-element
  parallelism (e.g. via cooperative shared-memory phases like the
  3D kernel uses).
- **3D async-writev `FrameWriter` is single-field.** The async
  scatter-gather pipeline (`VtuWriter` + `AsyncWriter`) writes a
  single rho field per frame for performance.  Synchronous
  multi-field output is available via `FrameWriter.write_frame_multi`
  (in addition to the underlying `dump_vtu_3d_frame_multi` helper
  in `src/vtu.mojo`); both accept N named scalar fields and the
  `FrameWriter` method correctly handles per-rank output dirs at
  np>1.  3D drivers that want multi-field output can opt into the
  sync path at the cost of one synchronous write per frame; folding
  multi-field into the async pipeline would be the natural next
  step.

## References

- **DG formulation**: Hesthaven & Warburton, *Nodal Discontinuous
  Galerkin Methods*, Springer 2008.
- **Kuhn tetrahedra**: Moore, *Simplicial Mesh Generation with
  Applications* (thesis), Cornell 1992.
- **Isentropic vortex test**: Shu, "Essentially Non-Oscillatory and
  Weighted Essentially Non-Oscillatory Schemes for Hyperbolic
  Conservation Laws", ICASE 97-65.
- **Roe / HLLE / HLLEC**: LeVeque, *Finite Volume Methods for
  Hyperbolic Problems*, Cambridge 2002.
- **Riemann problems / analytic Sod solver**: Toro,
  *Riemann Solvers and Numerical Methods for Fluid Dynamics*
  (3rd ed.), Springer 2009.  Sec 4.3 / Table 4.1 + 4.3 give the
  canonical Sod (Test 1) and strong-shock (Test 3) reference
  values gated by `test/sod_exact_riemann_test.mojo`.
- **WARPXM**: the reference implementation whose `advection_t` and
  `euler_t` numerical formulas were the starting point here.
- **VTK appended binary format**: Kitware's VTK File Formats
  documentation, section "UnstructuredGrid".

