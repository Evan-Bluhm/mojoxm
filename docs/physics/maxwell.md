---
icon: lucide/zap
---

# Maxwell

Vacuum Maxwell — electric field \(\mathbf{E}\) and magnetic field
\(\mathbf{B}\) — with optional uniform current \(\mathbf{J}\) and
magnetisation \(\mathbf{M}\) sources.

## Equations

\[
\frac{\partial \mathbf{E}}{\partial t} =
  c^2 \nabla \times \mathbf{B} - \frac{\mathbf{J}}{\varepsilon_0}
\qquad
\frac{\partial \mathbf{B}}{\partial t} = -\nabla \times \mathbf{E} - \mathbf{M}
\]

`NC = 6` packed as `[Ex, Ey, Ez, Bx, By, Bz]`. Source:
`src/maxwell.mojo` (281 lines).

The hyperbolic structure is the standard Maxwell wave system. The
divergence constraints \(\nabla \cdot \mathbf{E} = \rho/\varepsilon_0\)
and \(\nabla \cdot \mathbf{B} = 0\) are not enforced via cleaning in the
plain Maxwell module (they are in IdealMHD via Dedner GLM, and in
FiveMomentTwoFluid for the full multiphysics).

## Riemann solver

Rusanov only. The wave-speed estimate is \(\lambda_\max = c\) — the
speed of light is the only characteristic speed.

## Boundary conditions

The full BC menu is implemented:

| Constant       | Behaviour                                                          |
| -------------- | ------------------------------------------------------------------ |
| `BC_WALL` (PEC)| Perfect electric conductor: tangential \(\mathbf{E} = 0\), normal \(\mathbf{B} = 0\) |
| `BC_OUTFLOW`   | Transmissive (one-way absorber via Rusanov + extrapolated ghost)   |
| `BC_INFLOW`    | Riemann against prescribed plane-wave ghost (used for plane-wave gates) |

## Source term

Optional uniform \(\mathbf{J}, \mathbf{M}\):

\[
\mathbf{S}(q) = \begin{pmatrix} -\mathbf{J} / \varepsilon_0 \\ -\mathbf{M} \end{pmatrix}
\]

On a uniform background with \(q = 0\), the analytic solution is a linear
ramp: \(E_x(t) = -J_x t / \varepsilon_0\) and \(B_z(t) = -M_z t\). This
is the basis of the `bench_maxwell_uniform_{j,m}_*` gates.

## Reference drivers

| Driver                                       | What it shows                                              |
| -------------------------------------------- | ---------------------------------------------------------- |
| `examples/maxwell_cavity.mojo`               | 3D PEC-bounded standing wave, analytic round-trip          |
| `examples/maxwell_cavity_2d_gpu.mojo`        | 2D TM(1,1) PEC standing wave, period \(\sqrt{2}\)          |

## Validation gates

| Bench                                                  | What it gates                                                  |
| ------------------------------------------------------ | -------------------------------------------------------------- |
| `bench_maxwell_cavity_{2d,3d}`                         | TM(1,1) PEC standing wave, rel L2 ~2.8e-4 (2D P=2), ~3.5e-5 (3D P=2) |
| `bench_maxwell_cavity_3d_p3`                           | Same TM(1,1) PEC cavity at P=3 (NP=20); closes the only PEC-at-higher-P coverage gap, rel L2 ~4.5e-5 |
| `bench_maxwell_plane_wave_{2d,3d}_p{2..5}`             | TM plane wave on periodic box; 2D rel L2 floor ~5.8e-4 / 4.7e-5 / 1.8e-5 / 1.8e-5 at P=2/3/4/5 |
| `bench_maxwell_te_plane_wave_{2d,3d}`                  | TE polarisation dual: gates Bz / Ey flux paths                 |
| `bench_maxwell_outflow_{2d,3d}`                        | BC_OUTFLOW preservation on all faces                           |
| `bench_maxwell_inflow_{2d,3d}`                         | BC_INFLOW preservation on all faces                            |
| `bench_maxwell_uniform_{j,m}_{2d,3d}{,_p3}`            | Uniform J / M source on q=0 IC: linear ramp matches analytic   |

The `bench_maxwell_uniform_j_2d` / `_m` gates closed a 2D feature-parity
gap — 3D Maxwell already had `Maxwell.source_term` coupling J / M into
the RK rhs, but the 2D vol+lift kernel had no source-term plumbing
at all. Now both paths exercise the same physics.

## Notes on plane-wave gates

The 2D and 3D plane-wave benches **gate actual wave propagation** — they
initialise a TM (or TE) plane wave traveling in +x, integrate one full
period, and check both the rel L2 of the active components and the
zero-component leakage. The leakage check catches accidental coupling of
unintended components through the kernel (e.g., a sign error in the
normal-rotation matrix that would leak Bz into Ey).
