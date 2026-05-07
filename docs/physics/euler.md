---
icon: lucide/wind
---

# Euler

The 5-moment compressible-gas dynamics module, with four selectable
numerical-flux families and an optional gravity source.

## Equations

The 3D compressible Euler equations are the conservation laws for mass,
momentum (3 components), and total energy:

\[
\frac{\partial}{\partial t}
\begin{pmatrix} \rho \\ \rho \mathbf{u} \\ E \end{pmatrix}
+ \nabla \cdot
\begin{pmatrix} \rho \mathbf{u} \\ \rho \mathbf{u} \otimes \mathbf{u} + p \mathbf{I} \\ (E + p) \mathbf{u} \end{pmatrix}
=
\begin{pmatrix} 0 \\ \rho \mathbf{g} \\ \rho \mathbf{g} \cdot \mathbf{u} \end{pmatrix}
\]

with closure \(p = (\gamma - 1)\bigl(E - \tfrac{1}{2}\rho |\mathbf{u}|^2\bigr)\)
for a calorically-perfect ideal gas. `NC = 5`. Source: `src/euler.mojo`
(989 lines).

## Riemann solvers

The 3D path offers four solvers, dispatched by an `eflux_kind` enum on
the `Euler` struct:

| Solver                       | When to use                                           |
| ---------------------------- | ----------------------------------------------------- |
| **Rusanov**                  | Fast, monotone, dissipative. Default for smooth flow. |
| **Roe** (with entropy fix)   | Lower dissipation than Rusanov. Harten–Hyman fix prevents expansion-shock failure. |
| **HLLE**                     | Robust two-wave estimator. Tends to smear contacts.  |
| **HLLEC**                    | Three-wave (rarefaction + contact + shock). Sharp at contact discontinuities; the default for production smooth-flow.  |

The 2D path offers Rusanov and HLLC. HLLC is preferred for the 2D Sod
gates because it captures the contact discontinuity sharply.

## Boundary conditions

| Constant       | Behaviour                                                                            |
| -------------- | ------------------------------------------------------------------------------------ |
| `BC_WALL`      | Slip wall (no-penetration): mirror the normal velocity component, identical pressure |
| `BC_OUTFLOW`   | Transmissive: flux from interior state directly                                      |
| `BC_INFLOW`    | Riemann against the prescribed inflow ghost state                                    |

The `boundary_flux` hook **rotates** \(q_\mathrm{int}\) into the face-normal
frame, computes the 1D Riemann flux, and rotates back. Same code path
as `numerical_flux` interior.

## Source term

Optional uniform gravity \(\mathbf{g}\) added to the momentum and energy
equations as

\[
\mathbf{S}(q) = (0,\ \rho \mathbf{g},\ \rho \mathbf{g} \cdot \mathbf{u})^\top.
\]

Used by `examples/euler_rising_bubble.mojo` (buoyant thermal in a
hydrostatic atmosphere) and gated by `bench_euler_hydrostatic_{2d,3d}` at
P=2/3/4/5.

## Reference drivers

### 3D

| Driver                              | What it shows                                            |
| ----------------------------------- | -------------------------------------------------------- |
| `examples/euler_vortex.mojo`        | Shu isentropic vortex, periodic, HLLEC, smooth-flow gate |
| `examples/euler_taylor_green.mojo`  | Compressible Taylor–Green vortex on a 2π cube, Ma ≈ 0.3  |
| `examples/euler_sod.mojo`           | Smoothed Sod with non-periodic BCs (transmissive on x, slip y/z) |
| `examples/euler_rising_bubble.mojo` | Buoyant thermal bubble + hydrostatic atmosphere          |

### 2D (GPU-only, single-rank)

| Driver                                      | What it shows                                                  |
| ------------------------------------------- | -------------------------------------------------------------- |
| `examples/euler_vortex_2d_gpu.mojo`         | 2D isentropic vortex, periodic                                 |
| `examples/euler_sod_2d_gpu.mojo`            | Classical Sod in 2D, P=2 / 128×16, HLLC + BJ limiter           |
| `examples/euler_channel_2d_gpu.mojo`        | Mach-2 wind tunnel, BC_INFLOW + BC_OUTFLOW + walls             |

## Validation gates (selected)

| Bench                                         | What it gates                                                |
| --------------------------------------------- | ------------------------------------------------------------ |
| `bench_euler_smooth_wave_{2d,3d}_p{2..5}`     | Absolute-L2 sentinels at the Float32 floor across 2D NP=6/10/15/21 + 3D NP=10/20/35/56 |
| `bench_euler_vortex_{2d,3d}_p{2,3}`           | Long-time isentropic-vortex dissipation floor (~6.8% rel L2 at T=10) |
| `bench_euler_sod_2d`                          | Classical Sod in 2D, BJ limiter at P=2                      |
| `bench_euler_sod_limited_2d_p{3,4,5}`         | High-order shocked Sod with BJ limiter                       |
| `bench_euler_sod_3d_p{2,3,4,5}`               | 3D shocked Sod, BJ limiter, P=2..5; \(P=5/NP=56\) is the heaviest shocked gate in the suite |
| `bench_euler_hydrostatic_{2d,3d}_p{2..5}`     | Gravity source-term parity gate (constant-density rest state at rest to ~1e-5 over T=1) |
| `bench_euler_inflow_{2d,3d}`                  | BC_INFLOW arm of `Euler.boundary_flux`                       |
| `bench_euler_channel_steady_2d`               | Mach-2 wind tunnel, `rho_max_drift = 1.2e-7`                 |
| `bench_euler_flux_coverage_3d`                | Rusanov / Roe / HLLE flux paths in 3D                        |

## Notes

- **Long-time vortex dissipation floor**: at T=10 the Shu–Erlebacher
  vortex sits at ~6.8% rel L2 across `{2D, 3D} × {Rusanov, HLLC, HLLEC}
  × {NP=6, 10, 20}`. Higher P doesn't help. Float32 step-count
  accumulation likely dominates; use short-T entropy-wave benches for
  rate gates.
- **Unlimited DG on Sod blows up past t≈0.15.** Gibbs oscillations catch
  a NaN. Reproduces with periodic BCs — not a BC bug. The full classical
  Sod gates use the BJ limiter; see `bench_euler_sod_limited_*`.
