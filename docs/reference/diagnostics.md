---
icon: lucide/clipboard-list
---

# Diagnostics

Every driver can register conservation-law integrals via `DiagnosticsWriter`.
The writer emits one CSV row per frame, with `MPI_Allreduce` at np>1.

## Registering integrals

```mojo
from src.diagnostics import DiagnosticsWriter, NamedComponent
from src.time_integrator import run_ssprk3_loop_with_diagnostics

# Linear: ∫ q[c] dV
var linear = List[NamedComponent]()
linear.append(NamedComponent("mass",         0))
linear.append(NamedComponent("momentum_x",   1))
linear.append(NamedComponent("total_energy", 4))

# Squared: ∫ q[c]² dV  (L²² norms)
var squared = List[NamedComponent]()
squared.append(NamedComponent("Bx_sq", 5))     # magnetic energy contribution

# max_abs: max_x |q[c]|  (peak trackers)
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

## The three reduction kinds

| Kind        | Per-element                  | MPI reduction      | Used for                                  |
| ----------- | ---------------------------- | ------------------ | ----------------------------------------- |
| `linear`    | \(\sum_i M_{ii}\, q[c]_i\)   | `MPI_SUM`          | Mass, momentum, total energy              |
| `squared`   | \(\sum_i M_{ii}\, q[c]_i^2\) | `MPI_SUM`          | L²² norms, magnetic energy components     |
| `max_abs`   | \(\max_i |q[c]_i|\)          | `MPI_MAX`          | Shock peaks, ψ noise, overshoot tracking  |

`linear` and `squared` use the same mass-matrix-weighted node weights as
the BJ limiter — they're true volume integrals, not unweighted nodal
sums. This is correct for any integral diagnostic at P ≥ 2.

## CSV output

```
time,         mass,         momentum_x,   total_energy, Bx_sq,        max_abs_psi
0.0000000,    1.000e+00,   -2.456e-21,    8.500e+00,   2.500e-01,    0.000e+00
0.0500000,    1.000e+00,    1.234e-08,    8.500e+00,   2.499e-01,    1.234e-04
...
```

Conservation-of-mass and conservation-of-energy bugs jump out
immediately if you plot any `linear` column over time. Charge-balance
bugs in two-fluid show up as a non-zero `linear` integral on the charge
density.

## When MPI is in the picture

At `np > 1`:

1. Each rank computes its local sum / max over locally-owned elements.
2. `MPI_Allreduce` gathers across ranks.
3. **Rank 0 only** appends the CSV row.

The reduce kind (`MPI_SUM` for linear and squared, `MPI_MAX` for
max_abs) is fixed by the `NamedComponent` kind. There is no
sum-of-max bug because the wiring is matched at the type level.

`mpi_advection_test` and `mpi_bc_test` check that the diagnostics
output at np=1 matches np=4 to bit-identical precision (modulo
floating-point reduction order, which is small for these workloads).

## Plotting

Pandas works directly:

```python
import pandas as pd
df = pd.read_csv("output/diagnostics.csv")
print(df["mass"].iloc[0] - df["mass"].iloc[-1])     # mass drift over the run
df.plot(x="time", y=["mass", "total_energy"])
```

The bundled `scripts/animate_dashboard.py` auto-groups the columns by
prefix into three time-series panels (mass / momentum / everything-else)
without needing per-driver configuration.
