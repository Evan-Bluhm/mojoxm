---
icon: lucide/rocket
---

# Getting started

This section walks you from a fresh clone to a running simulation in three
short pages.

<div class="grid cards" markdown>

-   :material-package-variant:{ .lg .middle } **[Installation](installation.md)**

    ---

    Install Pixi, materialise the conda environment, and verify the Mojo
    toolchain works end-to-end.

-   :material-rocket-launch:{ .lg .middle } **[Your first simulation](your-first-simulation.md)**

    ---

    Build and run `advection_gaussian` — a triply-periodic Gaussian pulse
    that returns to its initial condition exactly. Inspect the printed
    memory and throughput reports.

-   :material-fast-forward:{ .lg .middle } **[Quickstart cheatsheet](quickstart.md)**

    ---

    The 30-second tour: the `make` targets you'll reach for during day-to-day
    development.

</div>

## What you need

- A Linux box (or WSL2) with a CUDA-capable NVIDIA GPU. The repo is
  developed and tested on RTX 3090, but anything Maxwell-class or newer
  with up-to-date drivers should work.
- Either Pixi (recommended — it pulls Mojo from the Modular nightly conda
  channel automatically) or a manual `mojo` install on `$PATH`.
- ~5 GB free for the conda environment + build artefacts.

!!! note "Why GPU-only?"

    mojoxm is intentionally GPU-only. There is no CPU reference physics
    pipeline and no GPU-vs-CPU diff harness — instead, every kernel is
    validated by self-consistent invariants (constant-state preservation,
    cell-mean conservation, analytic Riemann gates) that run entirely on
    device. See [Architecture · 2D vs 3D](../architecture/2d-vs-3d.md) for
    why, and how the validation pyramid is structured.
