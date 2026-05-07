---
icon: lucide/chart-bar-decreasing
---

# Slope limiter (Barth–Jespersen)

For shock-capturing problems, mojoxm runs a **Venkatakrishnan-smoothed
Barth–Jespersen** slope limiter after every RK stage. It's the only
limiter in the suite, and it ships in both 2D and 3D pipelines with
identical math but different kernel layouts.

## What the limiter does

For each element \(K\) and each component \(c\), find a scalar
\(\theta_K^c \in [0, 1]\) such that the limited solution

\[
q^c_\text{lim}(\mathbf{x}) = \bar{q}^c_K + \theta_K^c \, \bigl(q^c(\mathbf{x}) - \bar{q}^c_K\bigr)
\]

stays bounded by neighbour cell means:

\[
\min_{K' \in N(K)} \bar{q}^c_{K'}
\;\;\leq\;\; q^c_\text{lim}(\mathbf{x})
\;\;\leq\;\; \max_{K' \in N(K)} \bar{q}^c_{K'}
\quad \forall\, \mathbf{x} \in K.
\]

Here \(\bar{q}^c_K\) is the **mass-matrix-weighted cell mean** of
component \(c\) over element \(K\):

\[
\bar{q}^c_K = \frac{\int_K q^c \, dV}{\int_K 1 \, dV}
            = \frac{\sum_i M_{ii} \, q^c_i}{\sum_i M_{ii}}.
\]

\(M_{ii}\) is the diagonal of the diagonal-mass-matrix-weighted
quadrature on the reference element. **Critically, this is *not* the
unweighted nodal average** at \(P \geq 2\) — tet vertex nodes carry
weight \(-1/20\) while interior nodes carry positive weights, so the
unweighted \((1/N_P) \sum_i q^c_i\) is wrong.

!!! warning "Reuse `node_weights` for any new cell-mean diagnostic"

    The mass-matrix-weighted weights are stored in `node_weights` on the
    `LocalMesh`/`LocalMesh2DGpu` struct. Any new diagnostic that
    needs a cell mean should reuse this buffer.

## Venkatakrishnan smoothing

The raw Barth–Jespersen limiter clips \(\theta\) abruptly, which kills
convergence in smooth regions. Venkatakrishnan smoothing replaces

\[
\theta_K^c = \min(1, \, \alpha)
\quad \to \quad
\theta_K^c = \frac{\alpha^2 + 2\alpha}{\alpha^2 + \alpha + 2}
\]

where \(\alpha\) is the bound-violation ratio. Smooth in \(\alpha\), so
the limiter is "off" (\(\theta \approx 1\)) in smooth regions and
"on" (\(\theta \to 0\)) only at genuine extrema.

A small parameter \(\varepsilon\) (passed to `Solver.enable_cell_limiter(eps)`)
pushes the activation threshold off the round-off floor; \(\varepsilon = 0\)
turns the limiter into pure Barth–Jespersen.

## Three-kernel split

Both pipelines split the limiter into three kernels:

```mermaid
flowchart LR
    Q[q buffer] --> A[cell_mean_kernel<br/>per-element scan over nodes,<br/>mass-matrix-weighted]
    A --> B[compute_theta_kernel<br/>per-element scan over neighbours,<br/>over nodes, over components]
    B --> C[apply_kernel<br/>per-(elem, node, comp)<br/>coalesced write]
    C --> Q
```

| Kernel                  | Parallelism            | Memory pattern                           |
| ----------------------- | ---------------------- | ---------------------------------------- |
| `cell_mean_kernel`      | One thread per element | Reads `q`, writes one Float32 per (e, c) |
| `compute_theta`         | One thread per element | Reads neighbour means, writes one Float32 per (e, c) |
| `apply`                 | One thread per (e, i, c) | Reads `q`, `mean`, `theta`; writes `q` (stride-1 coalesced) |

Splitting `apply` out is the key perf move — it lets adjacent threads
share `elem` and stride-1 the write to `q[]`, eliminating the uncoalesced
pattern that previously dominated. **Measured 9.3× speedup on the apply
pass alone** (commit `cdc3210` on the 2D side).

## Cost share by P

The 2D pipeline has the limiter (compute_theta + apply) at roughly
40–48% of GPU time because `compute_theta` is O(NP) and the
rk_stage_kernel doesn't grow fast enough to absorb it.  At P=5 the
share comes back down because the Euler vol+lift+rk kernel scales
as NP × NC × arithmetic and pulls ahead at NP=21 / NC=4 = 84
q-values per thread block.

In 3D the cooperative `rk_stage_kernel` grows roughly **quadratically**
with NP, so the limiter share *shrinks* with P:

| P | 2D limiter share | 3D limiter share |
| - | ---------------- | ---------------- |
| 2 | ~40%             | ~15%             |
| 3 | ~44%             | ~14%             |
| 4 | ~48%             | ~12%             |
| 5 | ~45%             | ~11%             |

**2D limiter optimisation has ~3–4× the ROI of the same effort in 3D.**
A pre-pack of density for coalesced compute_theta reads was tried on
2026-04-26 — it was a regression, because the L2 cache was absorbing
the nominally-uncoalesced cost; the experiment was reverted.

See [Architecture · 2D vs 3D](../architecture/2d-vs-3d.md#cost-story-limiter-share)
for the cost story.

## Validation

| Test                                           | What it gates                                   |
| ---------------------------------------------- | ----------------------------------------------- |
| `limiter_2d_gpu_test{,_p3,_p4,_p5}`            | Smooth passthrough + within-cell spike monotonicity |
| `limiter_3d_test{,_p3,_p4,_p5}`                | Cell-mean conservation invariant: drift = 0 exactly |
| `bench_euler_sod_limited_2d_p{2,3,4,5}`        | End-to-end shocked Sod in 2D                    |
| `bench_euler_sod_3d_p{2,3,4,5}`                | End-to-end shocked Sod in 3D                    |
| `bench_mhd_brio_wu_3d{,_p3}`                   | 3D MHD Riemann with GLM + BJ                    |

The cell-mean conservation invariant test is the load-bearing one for
the cell-mean kernel itself: applying the limiter to a smooth field
should not change the cell mean at all. Drift = 0 *exactly* at every
supported P confirms the mass-matrix weighting is correct.

## Enabling the limiter

```mojo
var solver = Solver[Euler, 2](mesh, phys)
solver.enable_cell_limiter(eps=Float32(1e-6))    # smoothed BJ
# or
solver.enable_cell_limiter(eps=Float32(0.0))     # pure BJ
```

The 2D pipeline calls `bj_limit_full_2d` (in
`src/local_mesh_2d_gpu_limiter.mojo`) with the same three-kernel split
as the 3D path.
