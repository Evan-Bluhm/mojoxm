---
icon: lucide/waves
---

# ShallowWater

2D shallow-water equations, embedded in 3D with \(F_z = 0\).

## Equations

\[
\frac{\partial}{\partial t}
\begin{pmatrix} h \\ h u \\ h v \end{pmatrix}
+ \nabla \cdot
\begin{pmatrix} h \mathbf{u} \\ h u \mathbf{u} + \tfrac{1}{2} g h^2 \hat{\mathbf{x}} \\ h v \mathbf{u} + \tfrac{1}{2} g h^2 \hat{\mathbf{y}} \end{pmatrix}
= 0
\]

with \(\mathbf{u} = (u, v)\) the depth-averaged horizontal velocity.

`NC = 3`. Source: `src/shallow_water.mojo` (200 lines).

The 3D embedding stores all elements with \(F_z = 0\), so a z-uniform
initial condition stays z-uniform exactly. This means the 3D path *can*
run shallow-water problems if the mesh has trivial z-extrusion — useful
for testing the 3D pipeline against a 2D-known answer.

## Riemann solvers

| Solver       | Pipeline | Notes                                                       |
| ------------ | -------- | ----------------------------------------------------------- |
| **Rusanov**  | 2D + 3D  | Dissipative, robust                                         |
| **HLL**      | 2D       | Lower dissipation, two-wave; default in `bench_shallow_water_wave_2d` |

## Boundary conditions

| Constant       | Behaviour                                                  |
| -------------- | ---------------------------------------------------------- |
| `BC_WALL`      | Slip: mirror normal velocity component, identical \(h\)    |
| `BC_OUTFLOW`   | Transmissive                                               |
| `BC_INFLOW`    | Riemann against prescribed ghost                           |

## Source term

Zero. (Bottom topography, Coriolis, friction — not implemented.)

## Reference drivers

| Driver                                              | What it shows                                              |
| --------------------------------------------------- | ---------------------------------------------------------- |
| `examples/shallow_water_drop.mojo`                  | 3D, radial Gaussian perturbation in a closed basin (slip walls) |
| `examples/shallow_water_drop_2d_gpu.mojo`           | 2D version                                                 |
| `examples/shallow_water_dam_break_2d_gpu.mojo`      | 2D, h_L=2 / h_R=1 Riemann in a closed basin                |

## Validation gates

| Bench                                              | What it gates                                              |
| -------------------------------------------------- | ---------------------------------------------------------- |
| `bench_shallow_water_wave_{2d,3d}_p{2..5}`         | SW HLL gate at \(A/H=0.01\) nonlinear floor across 2D NP=6/10/15/21 + 3D NP=10/20/35/56 |
| `bench_shallow_water_wave_2d_rusanov`              | Rusanov-flux path used by the 2D dam break                 |
| `bench_shallow_water_dam_break_{2d,3d}`            | Closed-pool conservation invariants                        |
| `bench_shallow_water_inflow_{2d,3d}`               | BC_INFLOW + BC_OUTFLOW preservation                        |
| `bench_shallow_water_inflow_2d_rusanov`            | Same on the Rusanov path                                   |
