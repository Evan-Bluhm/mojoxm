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
- **Maxwell** — vacuum Maxwell (E, B) with Rusanov flux, PEC / outflow
  boundary conditions, and an optional uniform current source.
  6 components.
- **IdealMHD** — single-fluid ideal MHD with Dedner GLM div(B)
  cleaning. 8 conservation components + 1 GLM scalar = 9 total.
- **FiveMomentTwoFluid** — electron + ion fluids + full Maxwell + GLM.
  17 components. Lorentz force + current coupling through the
  source-term hook.

Nine reference drivers under `examples/`:

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

## Capabilities at a glance

- **Boundary conditions**: periodic, slip wall (`BC_WALL`), transmissive
  outflow (`BC_OUTFLOW`). Single- and multi-rank supported. Drivers
  declare BCs via a `BoundaryConditions` struct; the mesh builder
  allocates BC-side faces and routes them to a physics-provided
  `boundary_flux(q, bc_type, n)` hook.
- **Source-term hook**: physics types provide a `source_term(q, x, s_out)`
  method evaluated per-node and added pointwise to the SSPRK3 RHS.
  Used for gravity (Euler), current / charge coupling (Maxwell,
  two-fluid), GLM damping (MHD).
- **Diagnostics**: every driver can register linear, squared, and
  max-norm conserved-quantity integrals via `DiagnosticsWriter`. One
  CSV row per frame, with MPI allreduce at np>1.
- **Animated dashboard**: `scripts/animate_dashboard.py` reads the VTU
  frames + diagnostics CSV and emits a 2×2 animated GIF (density
  field + auto-grouped time-series panels). Lazy frame loader scales
  to large meshes.
- **Higher-order reference element**: `ReferenceElement[P]` computes
  Lagrange basis, mass, stiffness, and lift operators at arbitrary
  order via Vandermonde inverse + analytic integration. Node ordering
  matches VTK_LAGRANGE_TETRAHEDRON and is back-compatible with
  VTK_QUADRATIC_TETRA at P=2. `P` is now a comptime parameter on
  every 3D struct (`Mesh[P]`, `LocalMesh[P]`, `Solver[PhysT, P]`,
  `FrameWriter[PhysT, P]`, `DiagnosticsWriter[PhysT, P]`); `make test-p3`
  round-trips a scalar field through the full Mesh[3] + Solver pipeline.
- **Shock limiter (3D)**: `Solver.enable_cell_limiter(eps)` turns on a
  Venkatakrishnan-smoothed Barth-Jespersen slope limiter that runs
  after every RK stage.  Brings classical Sod to T=0.20 with left
  rho = 0.998, right rho = 0.119 (within ~0.2%/5% of exact).
- **2D triangular elements (CPU)**: parallel stack at `src/reference_2d.mojo`
  / `local_mesh_2d.mojo` / `dg_rhs_2d.mojo` / `vtu_2d.mojo`.
  `Physics2D` trait + **Advection2D / Euler2D / ShallowWater2D /
  IdealMHD2D** (4 physics);
  full BC menu: periodic / wall / outflow / **inflow** (each physics
  carries its own user-set inflow ghost state); SSPRK2, SSPRK3, RK4
  time integrators.  Spatial convergence rates verified to (P+1)
  on smooth advection (P=1: 1.99, P=2: 3.28, P=3: 3.96).
  7 demo drivers under `examples/*_2d_cpu.mojo` cover advection (periodic
  + outflow), Euler (vortex / Sod / Mach-2 channel with inflow),
  shallow water (drop / dam break).  `dump_vtu_2d_frame` writes
  ParaView-visualisable VTU; `scripts/animate_2d.py` renders them
  to MP4 via matplotlib + ffmpeg.
  **GPU port (task #19, complete):** all four 2D physics
  (Advection / Euler / ShallowWater / IdealMHD) run on device via
  Float32 kernels orchestrated as volume-rhs + Rusanov face-flux +
  multi-component lift-combine + rk-update.  Each physics's SSPRK3
  step matches the CPU Float64 reference to ~1e-7 at P=1..3.  Four
  end-to-end periodic drivers under `examples/*_2d_gpu.mojo` emit
  21-frame VTU sequences (+ .pvd) matching the CPU format so
  `scripts/animate_2d.py` works on either family:
  `advection_gaussian_2d_gpu` (~7700 compute steps/sec, 0.07 % rel
  L2 over one period), `euler_vortex_2d_gpu` (~7600, isentropic
  vortex, 6.8 % rel L2 at P=2 / 32x32), `shallow_water_drop_2d_gpu`
  (~6300, mean-h conservation ~4e-5), `mhd_alfven_2d_gpu` (~9100,
  0.36 % rel L2 on a one-period linear Alfven wave at P=2 / 64x4).

## Numerical scheme

- **P2 Lagrange DG** on tetrahedra (10 nodes per element: 4 vertices +
  6 edge midpoints). Node ordering matches `VTK_QUADRATIC_TETRA` (cell
  type 24) so ParaView opens the output with no re-indexing.
- **Kuhn 6-tet decomposition** of a Cartesian cell grid. Each cube
  owns 12 uniquely numbered faces (6 interior diagonal + 6 external on
  its +x/+y/+z boundaries). Face IDs are `owner_cell * 12 + face_type`,
  giving a Dict-free mesh build.
- **Canonical face-node ordering by owner-cell cube-corner index**
  (ascending). Because Kuhn tets are translation-invariant, this makes
  the `(tet, local_face, side) → element-local P2 node` mapping a set
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

Roughly 10k lines of Mojo + a thin MPI shim in C. Core components live
under `src/`; problem-specific drivers live under `examples/`.

| file                                | lines | role                                                                                   |
|-------------------------------------|-------|----------------------------------------------------------------------------------------|
| `src/reference.mojo`                |   729 | Reference element: arbitrary-order equispaced Lagrange (Vandermonde inverse + analytic integration), `D_ref`, `Lift_ref`, `face_to_elem` table |
| `src/local_mesh.mojo`               |  1189 | Raw periodic Kuhn-tet mesh builder + BC overlay kernels, **GPU-resident**              |
| `src/mesh.mojo`                     |   872 | Patch-aware `Mesh`: `LocalMesh` + partition / ghost ring / permutation + BC filtering  |
| `src/boundary.mojo`                 |    52 | `BoundaryConditions` + `BC_WALL` / `BC_OUTFLOW` constants                              |
| `src/partition.mojo`                |   192 | `(PX, PY, PZ)` factorisation of nprocs, minimising ghost-exchange surface              |
| `src/halo_exchange.mojo`            |   476 | MPI pack / Isend / Irecv / unpack + per-direction BC skip flag                         |
| `src/solver.mojo`                   |   667 | `Physics` trait, cooperative `rk_stage_kernel`, `Solver[PhysT]`, SSPRK3 stepper, per-NC shared-memory budget |
| `src/advection.mojo`                |   102 | `Advection`: scalar upwind flux + wall / outflow BC                                    |
| `src/euler.mojo`                    |   746 | `Euler`: 5-moment, 4 Riemann solvers, entropy fix, face rotation, gravity source       |
| `src/maxwell.mojo`                  |   213 | `Maxwell`: vacuum E+B, Rusanov, PEC / outflow BC, uniform J / M source                 |
| `src/shallow_water.mojo`            |   167 | `ShallowWater`: 2D shallow water embedded in 3D                                        |
| `src/mhd.mojo`                      |   326 | `IdealMHD`: single-fluid MHD + Dedner GLM div(B) cleaning                              |
| `src/two_fluid.mojo`                |   440 | `FiveMomentTwoFluid`: electron + ion + Maxwell + GLM, 17 components                    |
| `src/vtu.mojo`                      |   372 | Zero-copy binary-appended VTU writer (one scalar field per frame)                      |
| `src/async_writer.mojo`             |   167 | pthread-based `writev()` scatter-gather file writer                                    |
| `src/frame_writer.mojo`             |   173 | Per-rank frame output: `FrameWriter[PhysT]`, auto rank-subdir + `mkdir -p` at init     |
| `src/diagnostics.mojo`              |   ~250| `DiagnosticsWriter[PhysT]`: domain-integrated linear / squared / max-abs per-frame with allreduce |
| `src/time_integrator.mojo`          |   177 | `run_ssprk3_loop` and `run_ssprk3_loop_with_diagnostics`                               |
| `src/nvtx.mojo`                     |    84 | Runtime-loaded NVTX shim for Nsight Systems timelines                                  |
| `src/mpi.mojo` + `src/mpi_shim.c`   |   ~300| Mojo / C-shim bindings for OpenMPI (init, point-to-point, allreduce, request handling) |

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
.venv/bin/python scripts/animate_dashboard.py
# ... writes output/dashboard.gif
```

## Performance

### advection_gaussian at 48³ (RTX 3090, WSL2)

End-to-end wall clock for a full 20-frame run at 48³ (663 K tets,
6.6 M DOF, 2080 SSPRK3 steps, 26 MB density per VTU file):
**~4.7 s**.

The fused RK-stage kernel achieves **98% of peak L1-cache throughput**
according to Nsight Compute at 48³ (scalar advection specialization).

### euler_vortex at 32³ (RTX 3090, WSL2)

End-to-end wall clock for a 10-frame run at 32³ (196 K tets, 2.0 M DOF
× 5 components = 9.8 M unknowns, 280 SSPRK3 steps):
**~21 s**.

Euler's RK-stage kernel is substantially heavier than advection's
(HLLEC Riemann solver + 3×3 face rotation matrix construction per
face node, 5-component accumulators everywhere), so the per-step
wall time is roughly an order of magnitude larger. The mean density
drifts by ~2·10⁻⁶ over the 280-step run — well within single-precision
round-off expectations.

## Build & run

### Prerequisites

- NVIDIA GPU with compute capability 7.5+ (tested on RTX 3090).
- CUDA 12.x installed (for `libnvtx3interop`, `ncu`, `nsys`).
- `uv` (or `pip`) to install Mojo.
- A C toolchain (linker). On Linux, `libm` and `libpthread` via the
  system `glibc`.
- For the MPI examples (`mpi_hello`, `mpi_partition`, future
  multi-rank drivers): OpenMPI 4.x (`apt install openmpi-bin
  libopenmpi-dev` on Ubuntu) and `mpicc` on PATH.

### Install Mojo

```bash
cd /path/to/mojoxm
uv venv
uv pip install mojo        # installs Mojo 0.26.2.0 (or newer)
```

The compiler binary ends up at `.venv/bin/mojo`.

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
make test-reference  # host-side reference-element unit test (P=1..4)
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
  Lagrange reference element at P=1, 2, 3, 4: mass matrix SPD, node
  positions inside the reference simplex with correct pairwise
  separation, face-to-element lookup covers every face-local index.

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

- Only periodic BCs. No wall, Dirichlet, inflow/outflow.
- Only P2 elements. The reference-element module could be generalized
  to higher orders; the kernel ABI is order-independent but the
  on-face orientation handling assumes the current 6-face-node
  triangle layout.
- Cartesian block mesh only. Unstructured tet meshes from GMSH/etc.
  would need a different `Mesh` that loads from file and computes
  face-node mappings via the sorted-global-ID scheme (with care at
  periodic boundaries).
- Float32 everywhere. RTX 3090 FP64 is 1/64 of FP32, so FP32 is the
  right call, but some applications may need FP64 mass-matrix inversion.
- No limiter (Moe-Rossmanith, etc.) — smooth solutions only. Gibbs
  oscillations on discontinuous ICs grow without bound.
- VTU writer emits one scalar per frame; visualizing multiple Euler
  components (momentum, pressure) requires extending the writer.

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
- **WARPXM**: the reference implementation whose `advection_t` and
  `euler_t` numerical formulas were the starting point here.
- **VTK appended binary format**: Kitware's VTK File Formats
  documentation, section "UnstructuredGrid".

