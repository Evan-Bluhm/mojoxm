#!/usr/bin/env python3
"""
Compare two mpi_advection_test dump sets from disk.

Usage:
    diff_mpi_dumps.py --a <dir_a> --b <dir_b> [--tol-abs 1e-5] [--tol-rel 1e-4]

Each <dir> must contain one or more `final_q_rank_<N>.bin` files,
produced by examples/mpi_advection_test.mojo.  The script unions the
per-rank element coverage in each directory, verifies the two
directories describe the *same* global-element set, and reports the
max |a - b| and max |a - b| / |a| over all matched elements.

Returns exit code 0 on pass, 1 on fail.
"""

import argparse
import glob
import os
import struct
import sys

import numpy as np


MAGIC = 0x514D584D  # "MXMQ"
VERSION = 1
HEADER_FIELDS = 5


def load_directory(d: str) -> dict[int, np.ndarray]:
    """Union of (global_elem_id -> q[N_P]) across every rank dump in d."""
    paths = sorted(glob.glob(os.path.join(d, "final_q_rank_*.bin")))
    if not paths:
        raise SystemExit(f"no final_q_rank_*.bin under {d!r}")
    result: dict[int, np.ndarray] = {}
    for p in paths:
        with open(p, "rb") as f:
            raw = f.read()
        magic, version, n_owned, nc, np_nodes = struct.unpack(
            "<IIIII", raw[: HEADER_FIELDS * 4]
        )
        if magic != MAGIC or version != VERSION:
            raise SystemExit(f"{p}: bad header (magic={magic:#x}, v={version})")
        off = HEADER_FIELDS * 4
        ids = np.frombuffer(
            raw[off : off + n_owned * 4], dtype=np.uint32
        )
        off += n_owned * 4
        q = np.frombuffer(
            raw[off : off + n_owned * np_nodes * nc * 4],
            dtype=np.float32,
        ).reshape(n_owned, np_nodes * nc)
        for i, gid in enumerate(ids):
            gid = int(gid)
            if gid in result:
                # Duplicate ownership means a bug: two ranks claim the
                # same owned element.
                raise SystemExit(
                    f"duplicate owner for global_elem_id={gid} in {d}"
                )
            result[gid] = q[i]
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="first dump directory")
    ap.add_argument("--b", required=True, help="second dump directory")
    ap.add_argument("--tol-abs", type=float, default=1e-5)
    ap.add_argument("--tol-rel", type=float, default=1e-4)
    args = ap.parse_args()

    a = load_directory(args.a)
    b = load_directory(args.b)

    if set(a.keys()) != set(b.keys()):
        only_a = sorted(set(a.keys()) - set(b.keys()))[:8]
        only_b = sorted(set(b.keys()) - set(a.keys()))[:8]
        print(
            f"FAIL: element sets differ -- {len(a)} vs {len(b)}; "
            f"only_a[:8]={only_a}, only_b[:8]={only_b}"
        )
        return 1

    max_abs = 0.0
    max_rel = 0.0
    worst_gid = -1
    for gid, qa in a.items():
        qb = b[gid]
        diff = np.abs(qa - qb)
        da = float(diff.max())
        denom = np.maximum(np.abs(qa), 1.0)  # relative vs at most O(1)
        dr = float((diff / denom).max())
        if da > max_abs:
            max_abs = da
            worst_gid = gid
        if dr > max_rel:
            max_rel = dr

    print(f"compared {len(a):,} elements ({a[next(iter(a))].size} DOFs each)")
    print(f"  max |a - b|         = {max_abs:.3e}   (at global_elem={worst_gid})")
    print(f"  max |a - b| / max(|a|, 1) = {max_rel:.3e}")
    print(f"  tolerances: abs={args.tol_abs:.1e}, rel={args.tol_rel:.1e}")

    if max_abs <= args.tol_abs and max_rel <= args.tol_rel:
        print("PASS")
        return 0
    print("FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())
