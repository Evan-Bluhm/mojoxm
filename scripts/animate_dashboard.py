#!/usr/bin/env python3
"""
animate_dashboard.py -- render a 2x2 animated dashboard of a mojoxm run.

Inputs:
    output/diagnostics.csv       conservation-law time series (written
                                 by src/diagnostics.mojo)
    output/frame_NNNNN.vtu       per-frame VTU files (optionally under
                                 output/rank_NNN/ for multi-rank runs)

Output:
    output/dashboard.gif         animated GIF, one frame per VTU

Layout:
    [ density slice ]  [ mass vs t ]
    [ momentum xyz ]   [ total energy ]

Dependencies: meshio, matplotlib, numpy.

Usage:
    scripts/animate_dashboard.py                        # np=1 layout
    scripts/animate_dashboard.py --rank-dirs            # np>1 glob
    scripts/animate_dashboard.py --fps 15 --dpi 80      # tweak quality
"""

import argparse
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import meshio
import numpy as np
from PIL import Image


def load_diagnostics(csv_path: str) -> dict[str, np.ndarray]:
    """Parse the diagnostics CSV into a dict of column arrays.  First
    row is a comma-separated list of column names; the first column
    is always `time`."""
    with open(csv_path) as f:
        header = f.readline().strip().split(",")
        rows = [line.strip().split(",") for line in f if line.strip()]
    data = np.array(rows, dtype=np.float64)
    return {name: data[:, i] for i, name in enumerate(header)}


def find_vtu_files(output_dir: str, multi_rank: bool) -> list[list[str]]:
    """Return per-frame lists of VTU files.  np=1: one VTU per frame;
    np>1: one per rank per frame, concatenated at load time."""
    if multi_rank:
        rank_dirs = sorted(glob.glob(os.path.join(output_dir, "rank_*")))
        if not rank_dirs:
            raise SystemExit(f"no rank_* subdirs in {output_dir}")
        per_rank_files = [sorted(glob.glob(os.path.join(d, "frame_*.vtu")))
                          for d in rank_dirs]
        n_frames = min(len(files) for files in per_rank_files)
        return [[files[i] for files in per_rank_files]
                for i in range(n_frames)]
    files = sorted(glob.glob(os.path.join(output_dir, "frame_*.vtu")))
    if not files:
        raise SystemExit(f"no frame_*.vtu in {output_dir}")
    return [[f] for f in files]


def load_frame(vtu_paths: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Load one frame's points + density scalar.  Multi-rank frames
    get concatenated (each rank's subdomain becomes a contiguous slab
    of points)."""
    pts_list, dens_list = [], []
    for p in vtu_paths:
        m = meshio.read(p)
        pts_list.append(m.points)
        dens_list.append(m.point_data["density"])
    return np.concatenate(pts_list, axis=0), np.concatenate(dens_list)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--output-dir", default="output")
    ap.add_argument("--rank-dirs", action="store_true",
                    help="glob output/rank_*/ for multi-rank runs")
    ap.add_argument("--out", default="output/dashboard.gif")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--dpi", type=int, default=100)
    args = ap.parse_args()

    diag = load_diagnostics(os.path.join(args.output_dir, "diagnostics.csv"))
    vtu_frames = find_vtu_files(args.output_dir, args.rank_dirs)
    n_frames = min(len(vtu_frames), len(diag["time"]))
    if n_frames == 0:
        raise SystemExit("no frames to animate")

    # Lazy frame loader.  Instead of preloading every VTU into memory
    # (at 2M points/frame, a 32^3 mesh with 20 frames hits ~4 GB), we
    # load frame-by-frame at render time, holding only the current
    # frame + the z-slice mask precomputed once from frame 0.
    # Requires two passes over the VTU files:
    #   1. Frame 0 only -- get the (x, y) slice coordinates.
    #   2. All frames -- do a fast min/max sweep of the density slice
    #      to fix the global color scale.
    # ... then a third pass at render time.  For huge meshes this is
    # slower than preloading but orders-of-magnitude lighter on RAM.
    print(f"frame 0 preview...", flush=True)
    pts0, dens0 = load_frame(vtu_frames[0])
    z_mid = 0.5 * (pts0[:, 2].min() + pts0[:, 2].max())
    z_tol = 0.6 * (pts0[:, 2].max() - pts0[:, 2].min() + 1e-12)
    slice_mask = np.abs(pts0[:, 2] - z_mid) < z_tol
    x_sl = pts0[slice_mask, 0].copy()
    y_sl = pts0[slice_mask, 1].copy()

    # Fix a global color scale by streaming through every frame once.
    # Density-only scan -- no point allocations kept around.
    print(f"color-scale sweep ({n_frames} frames)...", flush=True)
    vmin = float(dens0[slice_mask].min())
    vmax = float(dens0[slice_mask].max())
    for i in range(1, n_frames):
        _, di = load_frame(vtu_frames[i])
        ds = di[slice_mask]
        vmin = min(vmin, float(ds.min()))
        vmax = max(vmax, float(ds.max()))
        del di

    # Fetch a single frame's density slice on demand.
    def load_slice(frame: int) -> np.ndarray:
        _, d = load_frame(vtu_frames[frame])
        return d[slice_mask]
    dens0_sl = dens0[slice_mask].copy()
    del pts0, dens0

    ts = diag["time"]
    levels = np.linspace(vmin, vmax, 32)

    # Auto-group CSV columns into the three time-series panels by
    # name pattern.  Panel 2 = "linear invariants" (mass, per-species
    # masses).  Panel 3 = momenta (anything with 'momentum' or 'mom_'
    # in the name).  Panel 4 = everything else -- typically energies,
    # squared components, and max-norms.  Drivers can override with a
    # --groups CLI arg; the auto-detection gives a sane default.
    cols = [c for c in diag.keys() if c != "time"]
    def matches(name, needles):
        return any(w in name for w in needles)
    mass_cols   = [c for c in cols if matches(c, ["mass"])]
    mom_cols    = [c for c in cols if matches(c, ["momentum", "mom_"])]
    other_cols  = [c for c in cols if c not in mass_cols + mom_cols]
    # If any panel is empty, shift the remaining ones so we don't
    # render an empty subplot.
    panel_specs = [
        (mass_cols, "linear conserved"),
        (mom_cols, "momentum"),
        (other_cols, "energy / squared / max"),
    ]
    panel_specs = [p for p in panel_specs if p[0]]
    while len(panel_specs) < 3:
        panel_specs.append(([], ""))

    def render_one(frame: int):
        """Render one dashboard frame to an in-memory PIL image.  The
        density slice is loaded lazily (one VTU per call) so peak RAM
        stays proportional to a single frame, not the whole run."""
        fig, axes = plt.subplots(2, 2, figsize=(11, 8), dpi=args.dpi)
        (ax_field, ax_p1), (ax_p2, ax_p3) = axes
        fig.suptitle("mojoxm dashboard", fontsize=14)

        # Frame 0's slice is already in hand; every later frame triggers
        # one VTU load.
        ds = dens0_sl if frame == 0 else load_slice(frame)
        cf = ax_field.tricontourf(
            x_sl, y_sl, ds, levels=levels, cmap="viridis",
        )
        ax_field.set_xlim(x_sl.min(), x_sl.max())
        ax_field.set_ylim(y_sl.min(), y_sl.max())
        ax_field.set_aspect("equal")
        ax_field.set_xlabel("x"); ax_field.set_ylabel("y")
        ax_field.set_title(f"field @ t={ts[frame]:.3f}")
        fig.colorbar(cf, ax=ax_field, shrink=0.8)

        def plot_series(ax, names, title):
            if not names:
                ax.set_axis_off()
                return
            for i, n in enumerate(names):
                ax.plot(ts, diag[n], label=n, linewidth=1.5, color="C%d" % i)
                ax.plot([ts[frame]], [diag[n][frame]], "o",
                        color="C%d" % i, markersize=8)
            ax.legend(loc="best", fontsize=8)
            ax.set_title(title)
            ax.set_xlabel("time")
            ax.grid(True, alpha=0.3)

        plot_series(ax_p1, panel_specs[0][0], panel_specs[0][1])
        plot_series(ax_p2, panel_specs[1][0], panel_specs[1][1])
        plot_series(ax_p3, panel_specs[2][0], panel_specs[2][1])
        fig.tight_layout()
        fig.canvas.draw()
        arr = np.asarray(fig.canvas.buffer_rgba())
        plt.close(fig)
        return Image.fromarray(arr).convert("P", palette=Image.ADAPTIVE)

    images = []
    for i in range(n_frames):
        print(f"  rendering frame {i+1}/{n_frames}", flush=True)
        images.append(render_one(i))

    print(f"writing {args.out}...", flush=True)
    images[0].save(
        args.out,
        save_all=True,
        append_images=images[1:],
        duration=int(1000 / args.fps),
        loop=0,
        optimize=False,
    )
    print("done")


if __name__ == "__main__":
    main()
