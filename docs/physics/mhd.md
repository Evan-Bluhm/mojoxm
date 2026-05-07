---
icon: lucide/orbit
---

# IdealMHD

Single-fluid ideal magnetohydrodynamics, with optional Dedner GLM
divergence cleaning.

## Equations

\[
\frac{\partial}{\partial t}
\begin{pmatrix} \rho \\ \rho \mathbf{u} \\ \mathbf{B} \\ E \end{pmatrix}
+ \nabla \cdot
\begin{pmatrix}
\rho \mathbf{u} \\
\rho \mathbf{u} \otimes \mathbf{u} + \bigl(p + \tfrac{1}{2}|\mathbf{B}|^2\bigr)\mathbf{I} - \mathbf{B}\otimes\mathbf{B} \\
\mathbf{u} \otimes \mathbf{B} - \mathbf{B} \otimes \mathbf{u} \\
(E + p + \tfrac{1}{2}|\mathbf{B}|^2)\mathbf{u} - (\mathbf{u}\cdot\mathbf{B})\mathbf{B}
\end{pmatrix}
= \mathbf{S}_\mathrm{GLM}(q)
\]

with closure
\(p = (\gamma - 1)(E - \tfrac{1}{2}\rho|\mathbf{u}|^2 - \tfrac{1}{2}|\mathbf{B}|^2)\).

| Variant       | NC | State layout                                          |
| ------------- | -- | ----------------------------------------------------- |
| `IdealMHD`    | 8  | `[rho, rho·u, rho·v, rho·w, Bx, By, Bz, E]`          |
| `IdealMHDGLM` | 9  | `[rho, rho·u, rho·v, rho·w, Bx, By, Bz, E, psi]`     |

Source: `src/mhd.mojo` (461 lines, 3D); `src/local_mesh_2d_gpu_mhd.mojo`
(NC=6 plain) and `src/local_mesh_2d_gpu_mhd_glm.mojo` (NC=7 GLM) for the
2D path.

## Dedner GLM divergence cleaning

The `IdealMHDGLM` variant adds a hyperbolic scalar \(\psi\) coupled to
\(\nabla \cdot \mathbf{B}\) via

\[
\frac{\partial \mathbf{B}}{\partial t} + \nabla \cdot (\ldots) + \nabla \psi = 0
\qquad
\frac{\partial \psi}{\partial t} + c_h^2 \nabla \cdot \mathbf{B} = -\alpha_d \psi
\]

The first equation propagates \(\nabla \cdot \mathbf{B}\) errors at speed
\(c_h\) (a free parameter, typically a small multiple of the fastest
fluid wave speed). The second damps \(\psi\) on a timescale
\(1/\alpha_d\).

The result is that any divergence error introduced by truncation
**propagates away as a wave** rather than accumulating, and is
dissipated by the damping term. Without cleaning, divergence error
accumulates monotonically in shocked MHD and ultimately produces
unphysical states.

!!! warning "GLM leaks ~1% drift through non-periodic BCs"

    Sub-Alfvénic matched-state ICs with `BC_INFLOW`/`BC_OUTFLOW` and GLM
    enabled drift ~1.6% on E over T=1, even from psi=0. The same setup
    with GLM **disabled** sits at Float32 noise. Disable GLM for
    non-periodic MHD BC gates. `bench_mhd_inflow_3d` is the sentinel —
    it gates the plain (NC=8) path specifically.

## Riemann solver

**Rusanov only**, both 2D and 3D. HLLD is **not** implemented anywhere.

A 2D Brio–Wu retry on 2026-05-05 with the current BJ-limited GLM stack
NaN'd at T=0.04 and busted tolerances at T=0.005. The 2D Kuhn-triangle
Rusanov flux produces ~16× faster transverse Bx error growth than the
3D split-tet pattern — flux-function quality, not limiter. **HLLD really
is needed for 2D MHD shocks.** It's tracked on the roadmap.

The 3D path passes Brio–Wu via Rusanov + GLM + BJ on the split-tet
pattern, so the gap is geometric (transverse-flux quality at the
tessellation), not algorithmic.

## Boundary conditions

| Constant       | Behaviour                                                          |
| -------------- | ------------------------------------------------------------------ |
| `BC_WALL`      | Slip wall: mirror normal velocity, identical pressure / B         |
| `BC_OUTFLOW`   | Transmissive                                                       |
| `BC_INFLOW`    | Riemann against prescribed inflow ghost (NC=8 plain only — GLM disabled here, see warning above) |

## Reference drivers

| Driver                                       | What it shows                                                  |
| -------------------------------------------- | -------------------------------------------------------------- |
| `examples/mhd_alfven.mojo`                   | 3D linearly-polarised Alfvén wave on a periodic box            |
| `examples/mhd_alfven_2d_gpu.mojo`            | 2D analogue (NC=6 plain MHD, no GLM)                           |
| `examples/mhd_alfven_glm_2d_gpu.mojo`        | 2D NC=7 GLM-MHD with c_h=1.5, α_d=0.5; visualises ψ cleaning   |

## Validation gates (selected)

| Bench                                                  | What it gates                                                  |
| ------------------------------------------------------ | -------------------------------------------------------------- |
| `bench_mhd_alfven_3d_p{2..5}`                          | NP=10/20/35/56 IdealMHD gate; `_p5` exercises the cooperative kernel at NC=9 / 504 q-values per element (largest in suite) |
| `bench_mhd_alfven_glm_2d_p{2..5}`                      | NC=7 GLM-MHD gate; closes 2D MHD-GLM P-parity sweep            |
| `bench_mhd_glm_psi_transport_{2d,3d}_p{2..5}`          | \(c_h>0\) ψ/Bx wave coupling                                   |
| `bench_mhd_glm_psi_damp_{2d,3d}_p{2..5}`               | \(\alpha_d>0\) ψ decay matches \(A_0/e\) via operator splitting |
| `bench_mhd_brio_wu_3d{,_p3}`                           | Canonical 1D MHD Riemann embedded in 3D, GLM + BJ; first analytic-Riemann 3D MHD shocked gate |
| `bench_mhd_inflow_{2d,3d}`                             | BC_INFLOW + BC_OUTFLOW preservation, plain NC=8                |
| `bench_mhd_inflow_2d_glm` / `_wall_2d_glm`             | Same on GLM NC=7 path (periodic-equivalent IC)                 |
