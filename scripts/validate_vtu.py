#!/usr/bin/env python3
"""validate_vtu.py -- meshio-roundtrip sanity check for VTU output.

Reads each VTU file passed on the command line, asserts that meshio
can parse it cleanly, and verifies the cell type + connectivity
structure matches what a P-th-order Lagrange tetrahedron mesh
should produce:

    P=2  -> VTK_QUADRATIC_TETRA          (meshio name: tetra10), NP=10
    P>=3 -> VTK_LAGRANGE_TETRAHEDRON,
            NP = (P+1)(P+2)(P+3)/6  (20 / 35 / 56 / ...)

Usage:
    scripts/validate_vtu.py file1.vtu [file2.vtu ...]

Exits 0 if every file passes, 1 on the first failure (with a
description on stderr).

Catches:
  * binary VTU format regressions (mojo writer producing
    bytes meshio can't parse -- the in-Mojo XML-string check
    in vtu_3d_multi_test wouldn't see this).
  * cell-type-71 dispatch regressions at P >= 3.
  * point/cell count off-by-ones.

Doesn't validate actual node-position ordering against the VTK
spec -- that's a deeper TBD per project_p_propagation_scope.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import meshio


# (P+1)(P+2)(P+3)/6 -- the number of nodes per Lagrange tet at order P.
def num_tet_nodes(p: int) -> int:
    return (p + 1) * (p + 2) * (p + 3) // 6


# Path stems like "vtu_3d_multi_test_p3.vtu" or "frame_00000.vtu"
# -- we infer P from the "_pN" suffix when present.  When missing we
# infer P from the connectivity-array width found by meshio.
P_RE = re.compile(r"_p(\d+)\.vtu$|_p(\d+)$")


def parse_p_from_path(path: Path) -> int | None:
    m = P_RE.search(path.name)
    if not m:
        return None
    return int(m.group(1) or m.group(2))


def validate_one(path: Path) -> list[str]:
    failures: list[str] = []
    try:
        m = meshio.read(path)
    except Exception as e:
        return [f"meshio.read({path}) failed: {e}"]

    if not m.cells:
        return [f"{path}: no cells in mesh"]
    if len(m.cells) != 1:
        failures.append(
            f"{path}: expected 1 cell block, got {len(m.cells)}"
        )

    cells = m.cells[0]
    nodes_per = cells.data.shape[1]
    p_from_path = parse_p_from_path(path)

    # Infer P from the connectivity-array width if path doesn't tell.
    p_from_data: int | None = None
    for p in range(1, 8):
        if num_tet_nodes(p) == nodes_per:
            p_from_data = p
            break
    if p_from_data is None:
        failures.append(
            f"{path}: nodes_per_cell={nodes_per} doesn't match num_tet_nodes(P) "
            f"for any P in 1..7"
        )

    if p_from_path is not None and p_from_data is not None:
        if p_from_path != p_from_data:
            failures.append(
                f"{path}: P={p_from_path} from filename but data has "
                f"NP={nodes_per} which corresponds to P={p_from_data}"
            )

    # Cell type expectations.
    p_eff = p_from_data or p_from_path
    if p_eff == 2:
        expected_type = "tetra10"
    elif p_eff is not None and p_eff >= 3:
        expected_type = "VTK_LAGRANGE_TETRAHEDRON"
    else:
        expected_type = None  # can't infer

    if expected_type is not None and cells.type != expected_type:
        failures.append(
            f"{path}: cell type {cells.type!r}, expected {expected_type!r} "
            f"(P={p_eff}, NP={nodes_per})"
        )

    # Sanity: every connectivity index references a valid point.
    n_points = m.points.shape[0]
    if cells.data.min() < 0 or cells.data.max() >= n_points:
        failures.append(
            f"{path}: connectivity index out of [0, {n_points}) range -- "
            f"min={cells.data.min()}, max={cells.data.max()}"
        )

    # Sanity: each point has 3 coordinates (XYZ).
    if m.points.shape[1] != 3:
        failures.append(
            f"{path}: point coords have {m.points.shape[1]} components, expected 3"
        )

    # Per-cell invariants: scan up to 32 cells (covers small-mesh test
    # fixtures fully; samples driver output cheaply).
    import numpy as np

    n_cells_check = min(32, cells.data.shape[0])
    for ci in range(n_cells_check):
        row = cells.data[ci]
        # Connectivity uniqueness within a cell -- a duplicated index
        # would cause rendering glitches and indicates a writer bug.
        if len(set(int(x) for x in row)) != len(row):
            failures.append(
                f"{path}: cell {ci} has duplicate connectivity indices "
                f"(only {len(set(int(x) for x in row))} unique of {len(row)})"
            )
            break  # stop after first to keep noise down
        # First 4 nodes must be the 4 vertices -- distinct physical
        # positions.  VTK_LAGRANGE_TETRAHEDRON puts the corners first;
        # if they coincide the writer is mis-ordering nodes.
        corners = m.points[row[:4]]
        for a in range(4):
            for b in range(a + 1, 4):
                if np.allclose(corners[a], corners[b], atol=1e-9):
                    failures.append(
                        f"{path}: cell {ci} corner nodes {a} and {b} "
                        f"coincide at {tuple(corners[a])}"
                    )
                    break

    return failures


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("files", nargs="+", help="VTU files to validate")
    args = ap.parse_args()

    n_ok = 0
    for f in args.files:
        path = Path(f)
        if not path.exists():
            print(f"validate_vtu: {path} does not exist", file=sys.stderr)
            return 1
        failures = validate_one(path)
        if failures:
            print(f"validate_vtu FAILED: {path}", file=sys.stderr)
            for msg in failures:
                print(f"  * {msg}", file=sys.stderr)
            return 1
        n_ok += 1
    print(f"validate_vtu: {n_ok} file(s) OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
