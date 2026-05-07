#!/usr/bin/env python3
"""Animate a 2D VTU frame sequence (produced by the 2D GPU drivers) as an
MP4.  Intended as a lightweight alternative to ParaView for the
triangulated 2D output.

Usage
-----
    scripts/animate_2d.py <pvd_file>                     (infer settings)
    scripts/animate_2d.py <pvd_file> -o movie.mp4 -f rho
    scripts/animate_2d.py <pvd_file> -f rho,p,'|v|'      (multi-panel)
    scripts/animate_2d.py <pvd_file> --list-fields       (no rendering, just print)

Examples
--------
    ./euler_vortex_2d_gpu                                # produces output/solution_euler2d_gpu.pvd
    scripts/animate_2d.py output/solution_euler2d_gpu.pvd \
        -f rho,p,'|v|' -o vortex.mp4
                                                         # 1x3 panel layout

If `-f <name>` doesn't match a field in frame 0 the script lists the
available fields and exits cleanly.  Use `--list-fields` to inspect
without re-running the simulation.  Comma-separate names to render
multiple side-by-side panels with per-panel colourbars (per-panel
range, so disparate-magnitude fields each show useful detail).

The PVD file contains the frame list + per-frame time; if the pvd is
missing the script falls back to sorting the VTU files in the directory
lexicographically and guessing frame intervals from the first two
timesteps (or 0, dt, 2dt, ... if there's only one frame).

Requires: meshio, numpy, matplotlib (plus ffmpeg on PATH for MP4).
"""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import meshio
import numpy as np
from matplotlib.animation import FFMpegWriter, PillowWriter
from matplotlib.tri import Triangulation


# ----------------------------------------------------------------------
# .pvd parsing -- small enough that we do it directly.
# ----------------------------------------------------------------------


def parse_pvd(pvd_path: Path) -> list[tuple[Path, float]]:
    """Return (path, time) pairs in order."""
    tree = ET.parse(pvd_path)
    root = tree.getroot()
    out: list[tuple[Path, float]] = []
    for ds in root.iter("DataSet"):
        rel = ds.attrib["file"]
        t = float(ds.attrib["timestep"])
        out.append((pvd_path.parent / rel, t))
    out.sort(key=lambda pt: pt[1])
    return out


def infer_from_dir(dir_path: Path) -> list[tuple[Path, float]]:
    vtus = sorted(dir_path.glob("*.vtu"))
    if not vtus:
        raise SystemExit(f"no .vtu files in {dir_path}")
    # Evenly-spaced synthetic times; good enough for a plain animation.
    return [(p, i) for i, p in enumerate(vtus)]


def build_triangulation(mesh: meshio.Mesh) -> Triangulation:
    """meshio loads our quadratic triangles as 'triangle6'; matplotlib's
    Triangulation only knows about 3-vertex triangles, so we subdivide
    each triangle6 into 4 linear sub-triangles for plotting
    (standard Lagrange P2 -> P1 refinement)."""
    pts_xy = mesh.points[:, :2]
    for cblock in mesh.cells:
        if cblock.type == "triangle":
            return Triangulation(pts_xy[:, 0], pts_xy[:, 1], cblock.data)
        if cblock.type == "triangle6":
            # Node ordering: 0, 1, 2 are vertices; 3, 4, 5 are the
            # midpoints of edges (0,1), (1,2), (2,0).
            cells = cblock.data
            sub = np.empty((cells.shape[0] * 4, 3), dtype=cells.dtype)
            sub[0::4] = cells[:, [0, 3, 5]]   # corner 0 + two adjacent mids
            sub[1::4] = cells[:, [1, 4, 3]]   # corner 1 + two adjacent mids
            sub[2::4] = cells[:, [2, 5, 4]]   # corner 2 + two adjacent mids
            sub[3::4] = cells[:, [3, 4, 5]]   # middle triangle of the 3 mids
            return Triangulation(pts_xy[:, 0], pts_xy[:, 1], sub)
        if cblock.type == "triangle10":
            # P3 subdivision: 9 sub-triangles.  Standard Lagrange node
            # ordering in VTK_LAGRANGE_TRIANGLE: 0..2 corners; 3..5
            # edge (0,1); 6..8 edge (1,2); 9..11 edge (2,0); 12 interior
            # -- but that's 13, not 10.  Actually P3 has (3+1)(3+2)/2
            # = 10 nodes; VTK orders corners, then 2 per edge, then 1
            # interior: 0,1,2, [edge01]: 3,4, [edge12]: 5,6, [edge20]:
            # 7,8, [interior]: 9.  We subdivide into 9 sub-triangles.
            cells = cblock.data
            sub = np.empty((cells.shape[0] * 9, 3), dtype=cells.dtype)
            # Outer ring of 6 sub-triangles around each corner + 3
            # interior sub-triangles; layout shown as a lookup table.
            pattern = np.array([
                [0, 3, 8], [3, 9, 8], [8, 9, 7], [3, 4, 9],
                [4, 1, 5], [4, 5, 9], [9, 5, 6], [9, 6, 7],
                [6, 2, 7],
            ], dtype=cells.dtype)
            for k in range(9):
                sub[k::9] = cells[:, pattern[k]]
            return Triangulation(pts_xy[:, 0], pts_xy[:, 1], sub)
    raise SystemExit(
        "unsupported cell type; got "
        + ", ".join(c.type for c in mesh.cells)
    )


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "pvd_or_dir",
        type=Path,
        help="a .pvd collection file, or a directory containing .vtu files",
    )
    parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help="output animation path (.mp4 or .gif).  Defaults to "
             "<pvd_stem>.mp4 next to the input.",
    )
    parser.add_argument(
        "-f", "--field", default=None,
        help="point-data field name(s); comma-separated for multi-panel "
             "(e.g. `-f rho,p,|v|`).  Default: first available.",
    )
    parser.add_argument(
        "--list-fields", action="store_true",
        help="print the available field names from frame 0 and exit",
    )
    parser.add_argument(
        "--fps", type=int, default=10,
        help="frames per second (default 10)",
    )
    parser.add_argument(
        "--dpi", type=int, default=120,
        help="figure DPI (default 120)",
    )
    parser.add_argument(
        "--cmap", default="viridis",
        help="matplotlib colormap (default viridis)",
    )
    args = parser.parse_args(argv)

    if args.pvd_or_dir.is_file():
        frames = parse_pvd(args.pvd_or_dir)
        default_out = args.pvd_or_dir.with_suffix(".mp4")
    else:
        frames = infer_from_dir(args.pvd_or_dir)
        default_out = args.pvd_or_dir / "animation.mp4"

    if not frames:
        raise SystemExit("no frames found")

    # Peek at frame 0 to pick the field and build the triangulation.
    mesh0 = meshio.read(frames[0][0])
    available = list(mesh0.point_data.keys())
    if args.list_fields:
        print(f"{frames[0][0]}: {len(available)} field(s)")
        for name in available:
            shape = mesh0.point_data[name].shape
            print(f"  - {name} {shape}")
        return
    triangulation = build_triangulation(mesh0)
    if args.field is None:
        if not available:
            raise SystemExit("no point data in the VTU")
        field_list = [available[0]]
    else:
        field_list = [name.strip() for name in args.field.split(",")]
        for name in field_list:
            if name not in mesh0.point_data:
                raise SystemExit(
                    f"field '{name}' not found in {frames[0][0]}; "
                    f"available: {', '.join(available) or '(none)'}"
                )

    # Colour range per field: sweep the whole sequence once so each
    # panel's colourbar is steady across the animation.
    print(f"scanning {len(frames)} frames for colour ranges...", flush=True)
    vmin = {name: float("inf") for name in field_list}
    vmax = {name: -float("inf") for name in field_list}
    for p, _ in frames:
        pd = meshio.read(p).point_data
        for name in field_list:
            v = pd[name]
            vmin[name] = min(vmin[name], float(v.min()))
            vmax[name] = max(vmax[name], float(v.max()))
    for name in field_list:
        print(f"  field '{name}' range: [{vmin[name]:.4g}, {vmax[name]:.4g}]")

    # One panel per field, laid out 1 x N.  Per-panel colourbar uses
    # the panel's own range so disparate-magnitude fields (e.g. rho
    # near 1, |v| near 0.1, p near 1) all show useful detail.
    n_fields = len(field_list)
    fig, axes = plt.subplots(
        1, n_fields, figsize=(7 * n_fields, 7), squeeze=False,
    )
    axes = axes[0]  # 1 x N -> N
    tpcs = []
    for ax, name in zip(axes, field_list):
        ax.set_aspect("equal")
        ax.set_xlim(triangulation.x.min(), triangulation.x.max())
        ax.set_ylim(triangulation.y.min(), triangulation.y.max())
        v0 = mesh0.point_data[name]
        tpc = ax.tripcolor(
            triangulation, v0, shading="gouraud",
            vmin=vmin[name], vmax=vmax[name], cmap=args.cmap,
        )
        fig.colorbar(tpc, ax=ax, label=name, fraction=0.046, pad=0.04)
        tpcs.append(tpc)
    title = fig.suptitle("")

    out = args.output or default_out
    print(f"writing {out} ({args.fps} fps, {len(frames)} frames, "
          f"{n_fields} panel(s))", flush=True)
    writer: FFMpegWriter | PillowWriter
    if out.suffix.lower() == ".gif":
        writer = PillowWriter(fps=args.fps)
    else:
        writer = FFMpegWriter(fps=args.fps)
    with writer.saving(fig, str(out), dpi=args.dpi):
        for i, (p, t) in enumerate(frames):
            pd = meshio.read(p).point_data
            for tpc, name in zip(tpcs, field_list):
                tpc.set_array(pd[name])
            title.set_text(f"t = {t:g}   ({i+1}/{len(frames)})")
            writer.grab_frame()
    print("done", flush=True)


if __name__ == "__main__":
    matplotlib.use("Agg")
    main()
