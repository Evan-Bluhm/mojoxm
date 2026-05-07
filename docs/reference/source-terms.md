---
icon: lucide/zap
---

# Source terms

The `Physics` trait's `source_term` hook is evaluated **pointwise per
node** and added to the SSPRK3 RHS by the fused `rk_stage_kernel`. It is
the mechanism by which non-flux physics enters the equations.

## The hook

```mojo
def source_term(
    self, q, x, y, z, source_out,
):
    # source_out is an NC-vector to fill in (initially zero).
    # q is the local NC-vector state.
    # (x, y, z) are the node coordinates in physical space.
    ...
```

All inputs are by value (or device-passable references). The output is an
in-place fill.

The kernel that calls this hook walks every `(elem, node)` pair in
parallel, evaluates `source_term`, and adds the result to the RHS that
already contains the volume + lift contributions. There is no separate
"source kernel" — it's fused into the same launch.

## What each physics uses it for

| Physics             | Source term                                                     |
| ------------------- | --------------------------------------------------------------- |
| Advection           | None (writes zero)                                              |
| Euler               | Optional uniform gravity \(\mathbf{g}\)                         |
| ShallowWater        | None                                                            |
| Maxwell             | Optional uniform \(\mathbf{J}, \mathbf{M}\)                     |
| IdealMHD            | None at NC=8; GLM damping \(-\alpha_d \psi\) at NC=9             |
| FiveMomentTwoFluid  | Lorentz on each fluid + Joule heating + Ampère current + GLM    |

## Worked examples

### Euler gravity

For a uniform gravity vector \(\mathbf{g}\):

```mojo
def source_term(self, q, x, y, z, source_out):
    var rho = q[0]
    var u   = q[1] / rho
    var v   = q[2] / rho
    var w   = q[3] / rho
    source_out[0] = 0.0
    source_out[1] = rho * self.gx
    source_out[2] = rho * self.gy
    source_out[3] = rho * self.gz
    source_out[4] = rho * (self.gx * u + self.gy * v + self.gz * w)
```

Used by `examples/euler_rising_bubble.mojo` and gated by
`bench_euler_hydrostatic_{2d,3d}` at every supported P.

### Maxwell J / M

```mojo
def source_term(self, q, x, y, z, source_out):
    source_out[0] = -self.Jx / self.eps0
    source_out[1] = -self.Jy / self.eps0
    source_out[2] = -self.Jz / self.eps0
    source_out[3] = -self.Mx
    source_out[4] = -self.My
    source_out[5] = -self.Mz
```

On a uniform `q = 0` IC the analytic solution is a linear ramp:
\(E_x(t) = -J_x t / \varepsilon_0\) and \(B_z(t) = -M_z t\). Gated
by `bench_maxwell_uniform_{j,m}_*` at multiple P in both 2D and 3D.

### IdealMHD GLM damping

```mojo
def source_term(self, q, x, y, z, source_out):
    var psi = q[8]
    source_out[0..7] = 0
    source_out[8] = -self.alpha_d * psi
```

Hyperbolic damping decays \(\psi \to 0\) on a timescale
\(1/\alpha_d\). Validated by
`bench_mhd_glm_psi_damp_{2d,3d}_p{2..5}` — analytic decay matches
\(A_0/e\) via the operator-splitting damp kernel.

### FiveMomentTwoFluid (the heavy one)

Per-node, per-step, the source term computes:

1. **Lorentz force on electrons**:
   \(\rho_e q_e (\mathbf{E} + \mathbf{u}_e \times \mathbf{B})\) added to
   `source_out[1..3]`
2. **Lorentz force on ions**: same shape, `source_out[6..8]`
3. **Joule heating** in each energy equation:
   \(\rho_s q_s \mathbf{u}_s \cdot \mathbf{E}\) added to `source_out[4]`
   and `source_out[9]`
4. **Ampère current source on Maxwell**:
   \(-\mathbf{J}/\varepsilon_0\) where
   \(\mathbf{J} = q_e \rho_e \mathbf{u}_e + q_i \rho_i \mathbf{u}_i\),
   added to `source_out[10..12]`
5. **GLM ψ damping**, added to `source_out[16]`

About 40 floating-point operations per node per stage. This is why the
two-fluid module is the most expensive per DOF in the suite. Validated
by `bench_two_fluid_langmuir_3d{,_p3}` — the Langmuir oscillation period
matches the analytic two-fluid value to closed-form, exercising every
arm of the source term.

## Implementation note: pointwise, not integral

The hook is **pointwise** — it does not integrate over the element.
Integration happens implicitly through the DG weights when the RHS is
applied. So `source_out[c]` should be \(S_c(q(\mathbf{x}))\), not
\(\int S_c \, dV\).

If your problem needs an integral source (e.g. radiation transport),
that's a different mechanism — not currently implemented.

## Default implementation

A physics that doesn't use `source_term` should still implement it,
writing zero. There's no "no-op" optimisation needed — the compiler
inlines the call into the kernel and dead-code-eliminates the zero
writes.
