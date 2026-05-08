---
icon: lucide/folder-tree
---

# Examples

23 reference drivers ship under `examples/`. Each is a single Mojo file
exercising a specific physics + BC + diagnostic configuration. They serve
as both runnable tutorials and as the "shape" you should copy when
writing your own driver.

## 3D drivers (with full pipeline + MPI)

| Driver                       | Physics            | What it shows                                         |
| ---------------------------- | ------------------ | ----------------------------------------------------- |
| `advection_gaussian`         | Advection          | Triply-periodic Gaussian, exact return after T=1      |
| `euler_vortex`               | Euler (HLLEC)      | Shu isentropic vortex, smooth-flow gate               |
| `euler_taylor_green`         | Euler              | Taylor–Green vortex on 2π cube, Ma ≈ 0.3              |
| `euler_sod`                  | Euler (HLLC + BJ)  | Smoothed Sod, transmissive x / slip y/z               |
| `euler_rising_bubble`        | Euler + gravity    | Buoyant thermal in hydrostatic atmosphere             |
| `shallow_water_drop`         | ShallowWater       | Radial Gaussian drop, slip walls                      |
| `maxwell_cavity`             | Maxwell            | PEC standing wave, analytic round-trip                |
| `mhd_alfven`                 | IdealMHD           | Linearly-polarised Alfvén wave, periodic              |
| `two_fluid_langmuir`         | FiveMomentTwoFluid | Electron plasma oscillation at \(\omega_p\)           |

## 2D-GPU drivers (single-rank)

| Driver                                | Physics            | Pattern                                       |
| ------------------------------------- | ------------------ | --------------------------------------------- |
| `advection_gaussian_2d_gpu`           | Advection          | Periodic                                      |
| `advection_outflow_2d_gpu`            | Advection          | BC_OUTFLOW; mass drains to 5e-7 of IC by t=1  |
| `euler_vortex_2d_gpu`                 | Euler (HLLC)       | Periodic isentropic vortex                    |
| `euler_sod_2d_gpu`                    | Euler (HLLC + BJ)  | Classical Sod, BJ limiter, 128×16             |
| `euler_channel_2d_gpu`                | Euler              | Mach-2 wind tunnel: inflow + outflow + walls  |
| `shallow_water_drop_2d_gpu`           | ShallowWater       | Radial drop, slip walls                       |
| `shallow_water_dam_break_2d_gpu`      | ShallowWater       | h_L=2 / h_R=1 Riemann, closed basin           |
| `maxwell_cavity_2d_gpu`               | Maxwell            | TM(1,1) PEC standing wave, period \(\sqrt{2}\) |
| `mhd_alfven_2d_gpu`                   | IdealMHD (NC=6)    | Plain MHD periodic Alfvén                     |
| `mhd_alfven_glm_2d_gpu`               | IdealMHD-GLM (NC=7)| GLM-MHD with c_h=1.5, α_d=0.5                 |

## MPI utility drivers

| Driver                | What it shows                                            |
| --------------------- | -------------------------------------------------------- |
| `mpi_hello`           | Bare-bones MPI init / rank / size                        |
| `mpi_partition`       | Domain partitioning by `(PX, PY, PZ)` factorisation      |
| `mpi_patch_mesh`      | Patch-aware mesh subdivision + ghost ring                |
| `mpi_halo_pingpong`   | Pack / Isend / Irecv / unpack of one ghost dimension     |

These are diagnostic drivers — they print the partition layout and halo
exchange pattern rather than producing VTU output.

## Reading a driver

Every driver follows the same shape — a useful template if you're writing
a new one:

```mojo
def main():
    # 1. Build mesh on host
    var mesh = Mesh[2].build_periodic_3d(NX, NY, NZ, LX, LY, LZ)

    # 2. Construct physics
    var phys = Euler(gamma=1.4, eflux_kind=EFLUX_HLLEC)

    # 3. Construct solver (implicitly uploads mesh + operators to GPU)
    var solver = Solver[Euler, 2](mesh, phys)

    # 4. Run an IC kernel that writes solver.q[] directly on device
    launch_initial_condition_kernel(solver)

    # 5. Set up writer + diagnostics
    var writer = FrameWriter[Euler, 2](solver, "output/solution", num_frames=20)
    var diag   = DiagnosticsWriter[Euler, 2](solver, "output/diagnostics.csv", ...)

    # 6. Print perf introspection
    print(solver.memory_report())

    # 7. Run the SSPRK3 loop
    var result = run_ssprk3_loop_with_diagnostics[Euler](
        solver, writer, diag, dt, T_FINAL, NUM_FRAMES, nvtx,
    )

    # 8. Final-state multi-field snapshot
    dump_vtu_3d_frame_multi(solver, "output/snapshot_t_final.vtu",
                            ["rho", "p", "|v|"], ...)

    # 9. Print throughput
    print(solver.bench_step_loop(warmup=5, measure=50))
```

Every step except the IC kernel is shared boilerplate. The IC kernel is
the one piece you actually own per problem.
