---
icon: lucide/wind
---

# Advection

The simplest physics in the suite — scalar linear advection.

## Equation

\[
\frac{\partial q}{\partial t} + \mathbf{v} \cdot \nabla q = 0
\]

with \(\mathbf{v}\) a constant velocity vector specified per driver.

`NC = 1`. Source: `src/advection.mojo` (151 lines).

## Numerical flux

Plain upwind: at a face with outward normal \(\mathbf{n}\),

\[
\hat{F}(q_L, q_R, \mathbf{n}) =
\begin{cases}
(\mathbf{v} \cdot \mathbf{n})\, q_L & \text{if } \mathbf{v} \cdot \mathbf{n} \geq 0 \\
(\mathbf{v} \cdot \mathbf{n})\, q_R & \text{otherwise}.
\end{cases}
\]

No Riemann solver dispatch — there is only one wave family.

## Boundary conditions

| Constant       | Behaviour                                                   |
| -------------- | ----------------------------------------------------------- |
| `BC_WALL`      | Reflecting: \(\hat{F} = 0\) regardless of \(\mathbf{v}\)    |
| `BC_OUTFLOW`   | Pure upwind from interior; ghost = interior                 |
| `BC_INFLOW`    | Use the prescribed `inflow_q` as the upstream state         |

## Source term

Zero by default. The trait still requires `source_term`; it writes 0.

## Reference drivers

| Driver                                     | Setup                                                   |
| ------------------------------------------ | ------------------------------------------------------- |
| `examples/advection_gaussian.mojo`         | 3D, triply-periodic, Gaussian pulse, \(\mathbf{v}=(1,1,1)\). After \(T=1\), exact return to IC. |
| `examples/advection_gaussian_2d_gpu.mojo`  | 2D, periodic, lighter version of the above              |
| `examples/advection_outflow_2d_gpu.mojo`   | 2D, outflow box, Gaussian drains out (mass → 5e-7 of IC)|

## Validation gates

| Bench                                          | What it gates                                           |
| ---------------------------------------------- | ------------------------------------------------------- |
| `bench_advection_translation_2d{,_p3,_p4,_p5}` | log₂ rate gate. Measured rates: 2.77 / 3.92 / 4.67 / 5.83 |
| `bench_advection_3d{,_p3,_p4,_p5}`             | 3D analogue. Rates 2.60 / 3.69 / 4.68 / 5.33            |
| `bench_advection_outflow_{2d,3d}`              | BC_OUTFLOW drainage gate                                |
| `bench_advection_inflow_{2d,3d}`               | BC_INFLOW preservation with non-trivial `inflow_q`      |

Convergence rates approach the optimal \(P+1\) order. The small
sub-optimal rate at \(P=3\) in 3D (3.7 instead of 4) is consistent
with operator-construction round-off at NP=20 in Float32 — not a
scheme bug.

## Quickstart

```bash
make advection_gaussian
./advection_gaussian
ls output/
# solution_0000.vtu ... solution.pvd  diagnostics.csv
```
