# mojoxm — GPU DG advection solver in Mojo

Testing out the use of [Mojo](https://docs.modular.com/mojo/manual) to apply the discontinuous Galerkin method (DG) 
over tetrahedral meshes to solve hyperbolic differential equations, 
and inspired by [WARPXM](https://github.com/orgs/iws-hyperion/repositories).
Intended as a minimum viable test bed for exploring Mojo's GPU programming
model on a real finite-element kernel.

```
              ∂q/∂t + v · ∇q = 0,   on  [0, 1]³,  triply periodic
              initial:  q(x, 0) = exp(-|x - x₀|² / 2σ²)
              velocity: v = (1, 1, 1)
```

## Numerical scheme

- **P2 Lagrange DG** on tetrahedra (10 nodes per element: 4 vertices +
  6 edge midpoints), with node ordering matching `VTK_QUADRATIC_TETRA`
  (cell type 24) so ParaView opens the output with no re-indexing.
- **Kuhn 6-tet decomposition** of a Cartesian cell grid.  Each cube
  owns 12 uniquely numbered faces (6 interior diagonal + 6 external on
  its +x/+y/+z boundaries).  Face IDs are computed as
  `owner_cell * 12 + face_type`, giving a Dict-free mesh build.
- **Canonical face-node ordering by owner-cell cube-corner index**
  (ascending).  Because Kuhn tets are translation-invariant, this makes
  the `(tet, local_face, side) → element-local P2 node` mapping a set
  of small precomputed tables — no per-cell orientation bookkeeping
  and no special cases for periodic-wrap cells.
- **Upwind numerical flux** for the face term,
  `F* = 0.5·((v·n + |v·n|) q_L + (v·n - |v·n|) q_R)`, matching the
  `advection_t::numerical_flux_impl` formulation in WARPXM.
- **Fused RK-stage kernel**: one GPU kernel computes the upwind flux,
  volume DG term, and SSPRK3 linear combination per thread, avoiding
  any intermediate `rhs` / `face_flux` scratch buffers.  Launched
  **three times per timestep** for SSPRK3.
- **Affine tets**: inverse Jacobian and `1/(6V)` are constants per
  tet-type; the kernel reads them from per-element arrays filled by
  the build-time tet-type tables.

Validated at 48³ (663 552 tets, 6.6 M DOF) against the analytical
solution: after 2080 SSPRK3 steps at `dt ≈ 4.8·10⁻⁴`, integral
conservation holds to ~10⁻⁵ and the overshoot is ~0.2 % (inherent DG
Gibbs behaviour with no limiter).

## System architecture

The project is ~2400 lines of Mojo across seven files:

| file                    | size  | role                                                  |
|-------------------------|-------|-------------------------------------------------------|
| `src/main.mojo`         |  234  | entry point, simulation driver, NVTX instrumentation  |
| `src/reference.mojo`    |  433  | P2 reference element: analytic monomial integration of `M_ref`, `S_ref^k`, face mass/lift matrices |
| `src/mesh.mojo`         |  756  | periodic Kuhn-tet mesh, **GPU-resident build**        |
| `src/solver.mojo`       |  366  | fused RK-stage kernel, Gaussian IC kernel, SSPRK3 stepper |
| `src/vtu.mojo`          |  372  | zero-copy binary-appended VTU writer                  |
| `src/async_writer.mojo` |  167  | pthread-based `writev()` scatter-gather file writer   |
| `src/nvtx.mojo`         |   84  | runtime-loaded NVTX shim for Nsight Systems timelines |

### Everything that lives on the GPU

After construction, no host-side arrays indexed by element, face, or
DOF survive past startup.  All of the following are device-resident:

- **Mesh** (`Mesh` struct, populated by two build kernels):
  - 10 node coordinates per element (`elem_node_xyz`)
  - inverse Jacobian + `1/(6V)` per element
  - 4 face indices, sides, and canonical→reference permutations per element
  - per-face: 2 elements, 12 per-side node indices, normal, area
- **DG operators** (uploaded once from host, ~540 Float32s total):
  - `D_ref[3][10][10]` = `M_ref⁻¹ · S_ref^k` (volume)
  - `Lift_ref[4][10][6]` = `M_ref⁻¹ · L_ref^f` (face)
- **Solution state**: three `q` buffers for SSPRK3 (`d_q`, `d_q1`, `d_q2`)
- **Initial condition**: evaluated on-device by `gaussian_ic_kernel`
  reading the already-resident node coordinates

The only host-side array of note is a single pointer to `elem_node_xyz`
that `VtuWriter` references by pointer (zero-copy) when streaming VTU
frames out to disk.

### The I/O pipeline

- Per-frame writes use a pthread-based `AsyncWriter` that issues
  `writev()` with 6 scatter-gather segments per frame:
  `[xml_header][density(owned)][pts_count][elem_node_xyz(ref)][conn+offsets+types][xml_tail]`.
  Only the 26 MB density section is copied per frame; the 80 MB mesh
  coord segment is a pointer into `Mesh`'s host-side download.
- Up to `max_concurrent=8` writer threads in flight, joined lazily in
  the next `submit()` or at `wait_all()`.  Disk I/O is fully hidden
  behind GPU compute in the steady state (typical `wait_async_writes`
  at shutdown is tens of ms).

## Performance at 48³ (RTX 3090, WSL2)

End-to-end wall clock for a full 20-frame run at 48³ (663 K tets,
6.6 M DOF, 2080 SSPRK3 steps, 136 MB per VTU file, 2.7 GB total on disk):
**~7 s**.

NVTX breakdown (host timeline; GPU compute is `frame_boundary_sync`):

```
frame_boundary_sync    ~4.3 s    (2080 steps × 3 stages × ~730 µs = GPU compute)
build_mesh             ~0.68 s   (GPU kernels + 80 MB device->host coord download)
device_context_create  ~0.73 s   (CUDA runtime init, one-time)
init_vtu_writer         ~25 ms   (connectivity iota + offsets + memset types)
write_frame (×20)      ~8 ms ea  (async; overlapping with subsequent compute)
initial_condition       ~1 ms    (one GPU kernel launch)
solver_setup            ~0.2 ms  (D_ref + Lift_ref upload, ~540 floats)
reference_element       ~0.2 ms
```

The fused RK-stage kernel achieves **98 % of peak L1-cache throughput**
according to Nsight Compute at 48³.  For the ncu drill-down, see the
notes at the end of this README.

## Build & run

### Prerequisites

- NVIDIA GPU with compute capability 7.5+ (tested on RTX 3090).
- CUDA 12.x installed (for `libnvtx3interop`, `ncu`, `nsys`).
- `uv` (or `pip`) to install Mojo.
- A C toolchain (linker).  On Linux, `libm` and `libpthread` via the
  system `glibc`.

### Install Mojo

```bash
cd /path/to/mojoxm
uv venv
uv pip install mojo        # installs Mojo 0.26.2.0 (or newer)
```

The compiler binary ends up at `.venv/bin/mojo`.

### Compile

```bash
.venv/bin/mojo build -O3 -g0 src/main.mojo -o mojoxm \
    -Xlinker -lm -Xlinker -lpthread
```

- `-O3` — full optimization (default already, but explicit).
- **`-g0` is critical** — the default Mojo debug info inflates register
  pressure from 40 → 114 per thread in the RK kernel and drops ncu's
  reported memory throughput from 98 % to 8 %.  See the ncu analysis
  notes at the end.

### Run

```bash
./mojoxm
```

20 binary VTU frames and one PVD collection file land in `output/`.
Open `output/solution.pvd` in ParaView to see the Gaussian pulse
advecting across the periodic cube along the diagonal.

### Change the problem

All simulation parameters are `comptime` constants at the top of
`src/main.mojo` — edit and rebuild:

```mojo
comptime NX = 48                     # cells per axis (mesh is NX³ × 6 tets)
comptime LX = 1.0                    # domain size
comptime VX: Float32 = 1.0           # advection velocity components
comptime T_FINAL: Float32 = 1.0      # simulation end time
comptime NUM_FRAMES = 20             # output frames (evenly spaced)
comptime CFL = Float32(0.2)          # SSPRK3 safety factor

comptime GAUSS_CX: Float32 = 0.5     # pulse center
comptime GAUSS_SIGMA: Float32 = 0.12 # pulse width
```

### Expected scaling

| NX  | tets     | DOF       | steps | wall (s) |
|-----|----------|-----------|-------|----------|
| 16  | 24 576   | 245 k     | 700   | ~1.0     |
| 32  | 196 608  | 1.97 M    | 1400  | ~2.5     |
| 48  | 663 552  | 6.64 M    | 2080  | ~7       |
| 64  | 1 572 864| 15.7 M    | 2780  | ~15      |

Wall time scales near-linearly with DOF count × step count at these
sizes (frame I/O is hidden; `device_context_create` is a one-time cost
that becomes a smaller fraction on longer runs).

## Profiling

### Nsight Systems (CPU timeline + GPU kernels)

```bash
nsys profile --trace=nvtx,cuda --output=trace ./mojoxm
nsys stats --report nvtx_pushpop_sum --report cuda_gpu_kern_sum trace.nsys-rep
nsight-sys trace.nsys-rep          # or open in the GUI
```

The code is instrumented with NVTX ranges at every interesting phase
(`build_mesh`, `solver_setup`, `initial_condition`, `ssprk3_step`,
`rk_stage_{1,2,3}`, `frame_boundary_sync`, `write_frame`,
`vtu_build_segments`, `vtu_submit`, `download_q`, `wait_async_writes`,
…) so the timeline view tells a readable story.

NVTX support is **runtime-optional** — `src/nvtx.mojo` does a `dlopen`
of `libnvtx3interop.so.1` (shipped with CUDA 12), and if the library
isn't present every call becomes a silent no-op, the same model NVTX
itself uses when no profiler is attached.

### Nsight Compute (kernel analysis)

```bash
ncu --kernel-name regex:rk_stage --launch-count 3 --launch-skip 10 \
    --set detailed ./mojoxm
```

The fused RK-stage kernel (`solver_rk_stage_kernel`) dominates
execution time; the three mesh-build kernels and the IC kernel each
run once.

## Design decisions worth knowing

### The Mojo 0.26.2 `ByteBuf` `__del__` workaround

`ByteBuf` in `src/vtu.mojo` deliberately has **no `__del__`**.  The
Mojo 0.26.2 compiler was observed to emit spurious destructor calls
on struct values that were still reachable through another path,
which caused the 109 MB static mesh buffer to be freed while a writer
thread was mid-`writev()`.  Since the three `ByteBuf`s in `VtuWriter`
have process-lifetime scope, the one-time leak at exit is harmless.
Diagnosing this is the single longest debugging anecdote in the
codebase (the writer thread printed a valid pointer whose contents
had been zeroed the moment `__del__` fired).

### Cube-corner canonical ordering (not sorted-global-ID)

Earlier versions of the mesh builder canonicalized shared faces by
sorting the three global vertex IDs.  At periodic boundaries the
wrap flips the sort order, which silently broke the
canonical-to-element-node permutations — constant fields still
propagated (no information in the node-identity mapping), but any
non-constant field blew up.  The current mesh builder uses the **owner
cell's cube-corner indices** as the canonical ordering, which is
position-independent and therefore works everywhere.

### Async VTU writing via pthread + `writev`

Frame writes were originally synchronous and a 48³ simulation spent
more time in disk I/O than in GPU compute.  `src/async_writer.mojo`
issues `pthread_create` with a Mojo callback function pointer (which
works on x86-64 because thin Mojo function types match the C ABI for
our pointer-only signatures), and the writer threads call `writev()`
on scatter-gather segment lists so the zero-copy mesh-coords segment
can be referenced in place without ever going through a staging
buffer.  `cuMemAllocHost` is avoided on the device→host path too.

## File index

```
src/main.mojo         Driver, NVTX scopes, hard-coded problem parameters
src/reference.mojo    P2 basis, analytic ∫ λ₀^α₀ … λ₃^α₃ dV = ∏αᵢ!/(|α|+3)!,
                      reference operators D_ref and Lift_ref
src/mesh.mojo         Kuhn-tet mesh, build_elements_kernel, build_faces_kernel,
                      Mesh struct owns all device mesh buffers
src/solver.mojo       rk_stage_kernel (fused), gaussian_ic_kernel, Solver
                      owns Mesh + DeviceContext + 3 RK buffers + D_ref/Lift_ref
src/vtu.mojo          ByteBuf raw-buffer helper, VtuWriter, writes binary-
                      appended XML with zero-copy points via scatter-gather
src/async_writer.mojo _writer_entry pthread callback, AsyncWriter with up to
                      8 concurrent writev()-based write threads
src/nvtx.mojo         OwnedDLHandle-based NVTX wrapper, silent no-op when
                      libnvtx3interop isn't present

output/               Generated: frame_00000.vtu … frame_00019.vtu + solution.pvd
.venv/                uv-managed Python venv containing mojo 0.26.2
.claude/              Skills registry (mojo-gpu-fundamentals, mojo-syntax,
                      mojo-python-interop)
```

## Limitations

This is an MVP and nowhere near as capable as WARPXM:
- Only scalar advection (1 component).  Vector-field apps (Euler,
  Maxwell, MHD) would need per-component kernel generalization.
- Only periodic BCs, triply.  No wall, Dirichlet, inflow/outflow.
- Only P2.  The reference-element module could be generalized to
  higher orders; the kernel ABI is order-independent but the on-face
  orientation handling assumes the current 6-face-node triangle
  layout.
- Cartesian block mesh only.  Unstructured tet meshes from GMSH/etc.
  would need a different `Mesh` that loads from file and computes
  face-node mappings via the sorted-global-ID scheme (with care at
  periodic boundaries).
- Float32 everywhere.  RTX 3090 FP64 is 1/64 of FP32, so FP32 is the
  right call, but some applications may need FP64 mass-matrix inversion.
- No limiter (Moe-Rossmanith, etc.) — smooth solutions only.  Gibbs
  oscillations on discontinuous ICs grow without bound.

## References

- **DG formulation**: Hesthaven & Warburton, *Nodal Discontinuous
  Galerkin Methods*, Springer 2008.
- **Kuhn tetrahedra**: Moore, *Simplicial Mesh Generation with
  Applications* (thesis), Cornell 1992; or any introductory
  computational-geometry text.
- **WARPXM**: the reference implementation whose `advection_t::internal_flux_impl`
  and `advection_t::numerical_flux_impl` were the starting point for
  the numerical formulas here.
- **VTK appended binary format**: Kitware's VTK File Formats
  documentation, section "UnstructuredGrid".

## License

Same as the parent `mojo-playground` project (if any).  This is
exploratory / educational code; not intended for production use.
