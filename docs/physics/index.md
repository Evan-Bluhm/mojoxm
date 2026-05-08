---
icon: lucide/atom
---

# Physics

Six physics modules ship with mojoxm. Each is implemented as a single Mojo
struct satisfying the [`Physics` trait](../architecture/physics-trait.md);
the rest of the pipeline picks up the new module by compile-time
specialisation.

## Comparison

| Module                  | NC | Riemann solvers                           | BCs                                         | Source term used for                          |
| ----------------------- | -- | ----------------------------------------- | ------------------------------------------- | --------------------------------------------- |
| **[Advection](advection.md)**     | 1  | Plain upwind                              | Periodic, wall, outflow, inflow             | —                                             |
| **[Euler](euler.md)**             | 5  | Rusanov / Roe / HLLE / HLLEC + entropy fix; HLLC in 2D | Periodic, slip wall, outflow, inflow | Uniform gravity                              |
| **[ShallowWater](shallow-water.md)** | 3  | Rusanov; HLL in 2D                       | Periodic, slip wall, outflow, inflow        | —                                             |
| **[Maxwell](maxwell.md)**         | 6  | Rusanov                                   | Periodic, PEC, outflow, inflow              | Uniform J / M                                 |
| **[IdealMHD](mhd.md)**            | 9 in 3D; 6 plain / 7 GLM in 2D | Rusanov                          | Periodic, slip wall, outflow, inflow        | Dedner GLM hyperbolic damping                 |
| **[FiveMomentTwoFluid](two-fluid.md)** | 17 | (per-fluid) Rusanov + Maxwell Rusanov | Periodic, wall, outflow, inflow             | Lorentz force + currents + GLM damping        |

## Conserved-state layouts

```text
Advection:    [q]
Euler:        [rho, rho·u, rho·v, rho·w, E]
ShallowWater: [h, h·u, h·v]
Maxwell:      [Ex, Ey, Ez, Bx, By, Bz]
IdealMHD 3D:  [rho, rho·u, rho·v, rho·w, Bx, By, Bz, E, psi]               (NC=9, GLM-capable)
IdealMHD 2D:  [rho, rho·u, rho·v, Bx, By, E]                               (NC=6, no GLM)
IdealMHD-GLM 2D: [rho, rho·u, rho·v, Bx, By, E, psi]                       (NC=7, GLM)
TwoFluid:     [electron 5][ion 5][E,B 6][psi 1]                            (NC=17)
```

## How to pick

- **Smooth scalar transport** → Advection.
- **Compressible gas, smooth or shocked** → Euler. Pick HLLC / HLLEC for
  shocks, Rusanov for fastest smooth-flow when the small-amplitude
  dissipation is acceptable.
- **Free-surface flow** → ShallowWater.
- **Vacuum electromagnetics** → Maxwell.
- **Single-fluid plasma / MHD** → IdealMHD. Enable GLM cleaning for
  div(B)-sensitive shocked flow (Brio–Wu, etc.).
- **Plasma physics with finite-temperature electrons + ions** →
  FiveMomentTwoFluid. The most expensive module per DOF; only use it
  when the two-fluid coupling matters.

## Validation status

Every physics is validated by **constant-state unit tests** (preserve
uniform fields to Float32 epsilon), **smooth-wave benchmark gates**
(eigenmode periods match analytic to closed-form), and **boundary-
condition gates** (every BC arm has at least one direct test).

| Physics                  | Pipelines | P-parity sweep covered           | Shocked gates                                |
| ------------------------ | --------- | -------------------------------- | -------------------------------------------- |
| Advection                | 2D + 3D   | P=2/3/4/5                        | n/a (smooth)                                 |
| Euler                    | 2D + 3D   | P=2/3/4/5                        | Sod 2D + 3D, P=2/3/4/5                       |
| ShallowWater             | 2D + 3D   | P=2/3/4/5                        | Dam break 2D + 3D                            |
| Maxwell                  | 2D + 3D   | P=2/3/4/5                        | n/a (smooth)                                 |
| IdealMHD (GLM)           | 2D + 3D   | P=2/3/4/5                        | 3D Brio–Wu P=2/3 (2D not gated; needs HLLD)  |
| FiveMomentTwoFluid       | **3D only**   | P=2/3/4/5 (walls); P=2/3 (Langmuir) | n/a; gated via Langmuir oscillation + BC preservation |
