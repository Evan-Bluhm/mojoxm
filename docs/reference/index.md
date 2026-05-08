---
icon: lucide/book-open
---

# Reference

Reference material for the cross-cutting machinery — boundary conditions,
the slope limiter, diagnostics, and source terms. These pages are meant
to be looked up rather than read straight through.

<div class="grid cards" markdown>

-   :material-fence:{ .lg .middle } **[Boundary conditions](boundary-conditions.md)**

    ---

    `BC_WALL`, `BC_OUTFLOW`, `BC_INFLOW` — what each constant means in
    each physics, and how the dispatch is structured.

-   :material-chart-bell-curve:{ .lg .middle } **[Slope limiter](limiter.md)**

    ---

    The Venkatakrishnan-smoothed Barth–Jespersen limiter, the
    mass-matrix-weighted cell mean, and the three-kernel split.

-   :material-chart-line:{ .lg .middle } **[Diagnostics](diagnostics.md)**

    ---

    `DiagnosticsWriter`, the three reduction kinds (linear, squared,
    max_abs), and how MPI allreduce is wired in.

-   :material-flash-outline:{ .lg .middle } **[Source terms](source-terms.md)**

    ---

    The pointwise `source_term` hook: gravity, currents, GLM damping,
    and the Lorentz / Ampère coupling in two-fluid.

</div>
