---
icon: lucide/layers-3
---

# FiveMomentTwoFluid

The most complex physics module in the suite — a 17-component plasma
model coupling two independent five-moment fluids (electrons + ions) to
the full Maxwell system, with Dedner GLM cleaning on \(\mathbf{B}\).

## Equations

Two five-moment Euler equations, one per species \(s \in \{e, i\}\):

\[
\frac{\partial}{\partial t}
\begin{pmatrix} \rho_s \\ \rho_s \mathbf{u}_s \\ E_s \end{pmatrix}
+ \nabla \cdot (\ldots) =
\begin{pmatrix}
0 \\
\rho_s\, q_s\, (\mathbf{E} + \mathbf{u}_s \times \mathbf{B}) \\
\rho_s\, q_s\, \mathbf{u}_s \cdot \mathbf{E}
\end{pmatrix}
\]

coupled to Maxwell with the species currents collected as the source:

\[
\frac{\partial \mathbf{E}}{\partial t} = c^2 \nabla \times \mathbf{B} - \mathbf{J}/\varepsilon_0,
\qquad
\mathbf{J} = \sum_{s \in \{e, i\}} q_s \rho_s \mathbf{u}_s.
\]

Plus the GLM ψ-cleaning equation on B as in [IdealMHD](mhd.md).

`NC = 17`, packed as:

```
q[0..4]    — electron (rho_e, rho_e u_e, rho_e v_e, rho_e w_e, E_e)
q[5..9]    — ion      (rho_i, rho_i u_i, rho_i v_i, rho_i w_i, E_i)
q[10..15]  — Maxwell  (Ex, Ey, Ez, Bx, By, Bz)
q[16]      — psi      (GLM scalar)
```

Source: `src/two_fluid.mojo` (627 lines).

!!! info "3D pipeline only"

    `FiveMomentTwoFluid` runs through the 3D `Solver[FiveMomentTwoFluid, P]`
    only. The 2D pipeline ships the other 5 smooth physics + GLM-MHD,
    but a `local_mesh_2d_gpu_two_fluid.mojo` module isn't implemented
    yet — it's tracked on the
    [near-term roadmap](../architecture/index.md#limitations-and-roadmap).

## Composition pattern

Internally `FiveMomentTwoFluid` **composes** `Euler` and `Maxwell`
rather than re-implementing them. The flux of the electron block goes
through `Euler.numerical_flux` over `q[0..4]`; same for ions over
`q[5..9]`; same for the EM block through `Maxwell.numerical_flux` over
`q[10..15]`. The GLM correction is applied in-place. This is one of the
clearest demonstrations of the trait-driven design: two physics modules
plug together at the source level.

## Source term (the heavy lifting)

Per-node, per-step:

1. Lorentz force on electrons: \(\rho_e q_e (\mathbf{E} + \mathbf{u}_e \times \mathbf{B})\)
2. Lorentz force on ions:      \(\rho_i q_i (\mathbf{E} + \mathbf{u}_i \times \mathbf{B})\)
3. Joule heating in each energy equation: \(\rho_s q_s \mathbf{u}_s \cdot \mathbf{E}\)
4. Current source on Maxwell: \(-\mathbf{J}/\varepsilon_0\) where \(\mathbf{J} = q_e \rho_e \mathbf{u}_e + q_i \rho_i \mathbf{u}_i\)
5. GLM hyperbolic damping on \(\psi\)

All evaluated pointwise at every node and added to the SSPRK3 RHS by the
fused `rk_stage_kernel`.

## Boundary conditions

The full BC menu is supported on every component block. Source: each
sub-physics dispatches its own `boundary_flux` arm; the FiveMomentTwoFluid
struct just routes per-block.

## Reference driver

| Driver                                       | What it shows                                                |
| -------------------------------------------- | ------------------------------------------------------------ |
| `examples/two_fluid_langmuir.mojo`           | Electron plasma oscillation at the plasma frequency \(\omega_p\); analytic two-fluid period match |

## Validation gates

| Bench                                                  | What it gates                                                  |
| ------------------------------------------------------ | -------------------------------------------------------------- |
| `bench_two_fluid_walls_3d_p{2..5}`                     | Charge-balanced 17-component rest state preserved under BC_WALL on all 6 faces, full P-parity at NP=10/20/35/56. Highest-NC physics × every supported P. |
| `bench_two_fluid_outflow_3d`                           | BC_OUTFLOW preservation                                        |
| `bench_two_fluid_inflow_3d`                            | BC_INFLOW preservation; closes the FiveMomentTwoFluid BC_INFLOW dispatch arm |
| `bench_two_fluid_langmuir_3d{,_p3}`                    | Electron plasma oscillation at \(\omega_p = \sqrt{26/25}\); the `_p3` variant exercises the Lorentz + Ampère source-term hook at NP=20 under non-trivial state evolution. |

The `_walls_p5` config at NC=17 / NP=56 = **952 q-values per element** is
the largest comptime configuration the suite covers end-to-end.

## Notes

- This module is the most expensive per DOF of any physics in the suite,
  by a factor of ~2× over even MHD-GLM. The Lorentz + Ampère source
  evaluation alone is ~40 floating-point ops per node per stage.
- Rest-state walls benches preserve a charge-balanced plasma exactly —
  i.e. `J = 0` on the IC, so the source term is identically zero.
  These don't exercise the Lorentz or Ampère arms. The Langmuir bench
  fills that coverage gap.
