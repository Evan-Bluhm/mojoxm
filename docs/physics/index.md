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
| **[IdealMHD](mhd.md)**            | 8 (plain) / 9 (GLM) | Rusanov                          | Periodic, slip wall, outflow, inflow        | Dedner GLM hyperbolic damping                 |
| **[FiveMomentTwoFluid](two-fluid.md)** | 17 | (per-fluid) Rusanov + Maxwell Rusanov | Periodic, wall, outflow, inflow             | Lorentz force + currents + GLM damping        |

## Conserved-state layouts

```text
Advection:    [q]
Euler:        [rho, rho·u, rho·v, rho·w, E]
ShallowWater: [h, h·u, h·v]
Maxwell:      [Ex, Ey, Ez, Bx, By, Bz]
IdealMHD:     [rho, rho·u, rho·v, rho·w, Bx, By, Bz, E]                     (NC=8)
IdealMHDGLM:  [rho, rho·u, rho·v, rho·w, Bx, By, Bz, E, psi]               (NC=9)
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

| Physics                  | P-parity sweep covered (2D + 3D) | Shocked gates                                |
| ------------------------ | -------------------------------- | -------------------------------------------- |
| Advection                | P=2/3/4/5                        | n/a (smooth)                                 |
| Euler                    | P=2/3/4/5                        | Sod 2D + 3D, P=2/3/4/5                       |
| ShallowWater             | P=2/3/4/5                        | Dam break 2D + 3D                            |
| Maxwell                  | P=2/3/4/5                        | n/a (smooth)                                 |
| IdealMHD (GLM)           | P=2/3/4/5                        | 3D Brio–Wu P=2/3 (2D not gated; needs HLLD)  |
| FiveMomentTwoFluid       | P=2/3/4/5 (walls); P=2/3 (Langmuir) | n/a; gated via Langmuir oscillation + BC preservation |
