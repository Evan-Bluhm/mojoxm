---
icon: lucide/image
---

# Visualization

Every driver writes its output to `output/`. Three flavours of file are
produced, each with its own consumer.

## Output files

| File pattern                          | What it is                                                 | Open with         |
| ------------------------------------- | ---------------------------------------------------------- | ----------------- |
| `output/frame_*.vtu`                  | Per-frame VTU (async path; 5-digit zero-padded)            | ParaView via PVD  |
| `output/solution.pvd` (3D)            | ParaView collection (XML index of the VTU sequence)        | ParaView          |
| `output/solution_<driver>.pvd` (2D)   | Per-driver PVD (each 2D driver writes a uniquely-named one)| ParaView          |
| `output/snapshot_t_final.vtu`         | Multi-field final snapshot (rho + p + \|v\|, etc.)         | ParaView directly |
| `output/diagnostics.csv`              | One row per frame: time + integrated quantities            | Pandas, gnuplot   |
| `output/dashboard.gif`                | 2×2 animated overview (after running `animate_dashboard.py`)| Any viewer       |

3D drivers write a single global `solution.pvd`; 2D-GPU drivers write
driver-specific `solution_<driver>.pvd` files (e.g. `solution_adv2d_gpu.pvd`,
`solution_sod2d_gpu.pvd`) so multiple 2D runs can share `output/`
without clobbering each other.

For 2D-GPU drivers, `frame_*.vtu` already contain multiple physical
fields per frame (e.g. rho + p + \|v\| for Euler) because the 2D pipeline
uses the multi-field path exclusively. `snapshot_t_final.vtu` is a
3D-only artefact emitted by the 8 multi-component 3D drivers.

## ParaView

Open `output/solution.pvd` directly. ParaView reads the PVD as a
time series and lets you scrub through frames. For multi-field 3D drivers,
also open `output/snapshot_t_final.vtu` to inspect the final state with
all derived fields.

VTU compatibility:

| P    | VTK cell type             | ParaView default warpings           |
| ---- | ------------------------- | ----------------------------------- |
| 2    | `VTK_QUADRATIC_TETRA` (24)| Yes — supported by every ParaView ≥5 |
| ≥3   | `VTK_LAGRANGE_TETRAHEDRON` (71) | Yes — requires ParaView ≥5.5 |

Same in 2D: `triangle6` (P=2), `VTK_LAGRANGE_TRIANGLE` (P≥3, cell type 69).

## The animated dashboard (3D)

```bash
./euler_rising_bubble                                # produces output/
pixi run python scripts/animate_dashboard.py        # writes output/dashboard.gif
```

The dashboard is a 2×2 GIF:

- **Top-left**: `tricontourf` of the per-frame scalar on a thin-z slice
  through z = LZ/2 (the field whichever component the driver wrote;
  hardcoded to the XML name `"density"` regardless of physics).
- **Top-right, bottom-left, bottom-right**: three time-series panels
  with auto-grouped CSV column names.  The actual panel titles are
  `"linear conserved"` (any column matching `mass`), `"momentum"`
  (any column matching `momentum` or `mom_`), and `"energy / squared
  / max"` (everything else).  Empty groups collapse, so a CSV with
  only mass + total_energy still renders cleanly.

Frames are loaded **lazily** (one VTU at a time), so RAM usage scales
with a single frame, not with the run length. A 32³ Taylor-Green
dashboard with 81 frames peaks under 1 GB resident.

## The 2D animator

```bash
./euler_vortex_2d_gpu
pixi run python scripts/animate_2d.py output/solution_euler2d_gpu.pvd \
  -f rho,p,'|v|'                                    # 3-panel MP4
```

Multi-component 2D-GPU drivers ship multiple physically meaningful scalar
fields per VTU frame:

| Driver                      | Fields written                       |
| --------------------------- | ------------------------------------ |
| `euler_*_2d_gpu`            | rho + p + \|v\|                      |
| `shallow_water_*_2d_gpu`    | h + \|v\|                            |
| `mhd_alfven_2d_gpu`         | rho + \|v\| + \|B\| + By             |
| `mhd_alfven_glm_2d_gpu`     | rho + \|v\| + \|B\| + ψ              |
| `maxwell_cavity_2d_gpu`     | Ez + \|E\| + \|B\|                   |

Pass `-f field1,field2,...` to render side-by-side animated panels.

## Validating VTU output

```bash
make test-vtu-meshio
# or directly:
pixi run python scripts/validate_vtu.py output/frame_00010.vtu
```

The validator checks **VTK-spec node ordering** at every supported P:

- Corner nodes are distinct
- Edge interiors are at parametric \((j+1)/P\) spacing along their edges
- Face interiors lie on the correct face plane
- Volume interiors are strictly inside the element

The same script handles both 2D triangle and 3D tetrahedral VTUs. It's
gated by `make pre-push` so any node-ordering regression trips CI before
it lands.

## Diagnostics CSV

The CSV is one row per frame with these column kinds:

| Column prefix      | Source                                            |
| ------------------ | ------------------------------------------------- |
| `time`             | Simulation time at the frame                      |
| `mass_*`, `momentum_*`, `total_energy_*` | `linear`-kind reductions   |
| `<name>_sq`        | `squared`-kind L²² reductions                     |
| `max_abs_<name>`   | `max_abs`-kind peak trackers                      |

At `np > 1`, every value is `MPI_Allreduce`d before rank 0 appends.
Plot in pandas:

```python
import pandas as pd
df = pd.read_csv("output/diagnostics.csv")
df.plot(x="time", y=["mass", "total_energy"])
```
