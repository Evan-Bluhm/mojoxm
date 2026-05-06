#!/usr/bin/env python3
"""profile_summary.py -- rank benches by rk_stage_kernel cost.

Reads benchmarks/profile_reports/*.kern.txt (the cuda_gpu_kern_sum
stats already saved by `make profile-bench-<name>`) and prints a
sorted view of the dominant kernel's average launch time per bench.

Use cases:
    * Pick optimisation targets: heaviest kernel = best per-step
      ROI for a kernel-level speedup (NP=56 P=5 benches typically
      sit at the top of the list).
    * Regression baselining: re-run `make profile-bench-all` and
      diff against a previous summary to spot a launch-time
      regression in any single bench at a glance.

Usage
-----
    scripts/profile_summary.py              # top 10 heaviest benches
    scripts/profile_summary.py --top 25     # top 25
    scripts/profile_summary.py --all        # every bench, sorted
    scripts/profile_summary.py --filter mhd # only matching benches
    scripts/profile_summary.py --kernel rk_stage  # only rk_stage_kernel rows
                                            # (3D benches only; 2D benches
                                            # use per-physics kernels)
    scripts/profile_summary.py --sort total # sort by total kernel ms
                                            # (where the profile-bench-all
                                            # budget actually sits)

Doesn't depend on Mojo / pixi -- pure stdlib Python so it runs
anywhere the .kern.txt files do.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PROFILE_DIR = REPO / "benchmarks" / "profile_reports"

# Each data row in cuda_gpu_kern_sum has the layout:
#   Time(%)  TotalTime  Instances  Avg  Med  Min  Max  StdDev  Name
# all whitespace-separated.  Name is the trailing token (Mojo's
# mangled symbol form, e.g. src_solver_rk_stage_kernel_I...).
ROW_RE = re.compile(
    r"^\s*"
    r"(?P<time_pct>[\d.]+)\s+"
    r"(?P<total_ns>\d+)\s+"
    r"(?P<instances>\d+)\s+"
    r"(?P<avg_ns>[\d.]+)\s+"
    r"(?P<med_ns>[\d.]+)\s+"
    r"(?P<min_ns>\d+)\s+"
    r"(?P<max_ns>\d+)\s+"
    r"(?P<stddev_ns>[\d.]+)\s+"
    r"(?P<name>\S+)\s*$"
)


@dataclass(frozen=True)
class KernelRow:
    bench: str
    time_pct: float
    total_ns: int
    instances: int
    avg_ns: float
    name: str

    @property
    def avg_us(self) -> float:
        return self.avg_ns / 1000.0

    @property
    def total_ms(self) -> float:
        return self.total_ns / 1_000_000.0


def parse_file(path: Path) -> list[KernelRow]:
    bench = path.stem.removesuffix(".kern")
    rows: list[KernelRow] = []
    for line in path.read_text().splitlines():
        m = ROW_RE.match(line)
        if not m:
            continue
        rows.append(
            KernelRow(
                bench=bench,
                time_pct=float(m["time_pct"]),
                total_ns=int(m["total_ns"]),
                instances=int(m["instances"]),
                avg_ns=float(m["avg_ns"]),
                name=m["name"],
            )
        )
    return rows


def kernel_kind(name: str) -> str:
    # The mangler appends a content hash; strip it for a readable label.
    base = re.sub(r"_[0-9a-f]{16}$", "", name)
    if "rk_stage_kernel" in base:
        return "rk_stage"
    if "build_elements" in base:
        return "build_elem"
    if "build_faces" in base:
        return "build_face"
    if "bj_limiter_compute_theta" in base:
        return "bj_theta"
    if "bj_limiter_apply" in base:
        return "bj_apply"
    if "compute_cell_averages" in base:
        return "cell_avg"
    # 2D per-physics path mangling: src_local_mesh_2d_gpu_<phys>_<hash>.
    # Truncated by the mangler at ~30 chars so we just call them "2d_<phys>".
    if "local_mesh_2d_gpu_advectio" in base:
        return "2d_advect"
    if "local_mesh_2d_gpu_euler" in base:
        return "2d_euler"
    if "local_mesh_2d_gpu_sw" in base:
        return "2d_sw"
    if "local_mesh_2d_gpu_mhd_glm" in base:
        return "2d_mhd_glm"
    if "local_mesh_2d_gpu_mhd" in base:
        return "2d_mhd"
    if "local_mesh_2d_gpu_maxwell" in base:
        return "2d_maxwell"
    if "local_mesh_2d_gpu_limiter" in base:
        return "2d_limiter"
    return base.rsplit("_", 1)[-1] if "_" in base else base


def collect(filter_substr: str | None, kernel_substr: str | None) -> list[KernelRow]:
    out: list[KernelRow] = []
    for kf in sorted(PROFILE_DIR.glob("*.kern.txt")):
        if filter_substr and filter_substr not in kf.stem:
            continue
        for row in parse_file(kf):
            if kernel_substr and kernel_substr not in row.name:
                continue
            out.append(row)
    return out


def dominant_per_bench(rows: list[KernelRow]) -> list[KernelRow]:
    """Pick the highest-Time(%) row per bench."""
    by_bench: dict[str, KernelRow] = {}
    for row in rows:
        cur = by_bench.get(row.bench)
        if cur is None or row.time_pct > cur.time_pct:
            by_bench[row.bench] = row
    return list(by_bench.values())


def print_table(
    rows: list[KernelRow],
    header: str,
    total_ms_all_benches: float | None = None,
) -> None:
    if not rows:
        print("(no rows)")
        return
    print(f"{header}")
    print(
        f"  {'bench':<48} {'kernel':<14} {'avg us':>9} {'inst':>7} "
        f"{'total ms':>10}"
    )
    print(f"  {'-' * 48} {'-' * 14} {'-' * 9} {'-' * 7} {'-' * 10}")
    shown_ms_sum = 0.0
    for r in rows:
        shown_ms_sum += r.total_ms
        print(
            f"  {r.bench:<48} {kernel_kind(r.name):<14} "
            f"{r.avg_us:>9.1f} {r.instances:>7d} {r.total_ms:>10.1f}"
        )
    # Footer: total kernel time across the displayed benches, plus the
    # full-suite total when only a slice is shown.  Useful for sizing
    # how long `make profile-bench-all` will take to re-baseline.
    if total_ms_all_benches is not None and total_ms_all_benches > shown_ms_sum + 0.5:
        print(
            f"  -> shown {len(rows)} benches: {shown_ms_sum / 1000:.2f} s "
            f"of dominant-kernel time;"
        )
        print(
            f"     full suite ({total_ms_all_benches / 1000:.2f} s "
            f"across all benches dominant-kernel-only)"
        )
    else:
        print(
            f"  -> {len(rows)} benches, "
            f"{shown_ms_sum / 1000:.2f} s of dominant-kernel time"
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--top",
        type=int,
        default=10,
        help="Show the top-N heaviest benches (default 10; ignored with --all).",
    )
    ap.add_argument(
        "--all",
        action="store_true",
        help="Show every bench, sorted by avg us (overrides --top).",
    )
    ap.add_argument(
        "--filter",
        default=None,
        help="Only include bench files whose stem contains this substring.",
    )
    ap.add_argument(
        "--kernel",
        default="",
        help=(
            "Only consider rows whose mangled name contains this substring "
            "(default: '', i.e. all rows; the per-bench dominant row is "
            "picked automatically).  Use --kernel rk_stage to scope to "
            "the 3D Solver's main kernel."
        ),
    )
    ap.add_argument(
        "--sort",
        default="avg",
        choices=["avg", "total", "inst"],
        help=(
            "Sort key (default: avg us/launch).  'total' = total kernel ms "
            "across all instances (where the profile-bench-all budget sits); "
            "'inst' = number of kernel launches (a proxy for step count)."
        ),
    )
    ap.add_argument(
        "--csv",
        action="store_true",
        help=(
            "Emit machine-readable CSV instead of the human-readable table."
        ),
    )
    args = ap.parse_args()

    if not PROFILE_DIR.is_dir():
        print(f"profile_summary.py: {PROFILE_DIR} not found", file=sys.stderr)
        print(
            "  Run `make profile-bench-<name>` (or `make profile-bench-all`) first.",
            file=sys.stderr,
        )
        return 1

    rows = collect(args.filter, args.kernel or None)
    if not rows:
        print(
            "profile_summary.py: no matching kernel rows found in "
            f"{PROFILE_DIR}",
            file=sys.stderr,
        )
        return 1

    dom = dominant_per_bench(rows)
    sort_label_map = {
        "avg": ("avg us/launch", lambda r: -r.avg_ns),
        "total": ("total kernel ms", lambda r: -r.total_ns),
        "inst": ("instance count", lambda r: -r.instances),
    }
    sort_desc, sort_key = sort_label_map[args.sort]
    dom.sort(key=sort_key)

    label = (
        f"all {len(dom)} benches"
        if args.all
        else f"top {min(args.top, len(dom))} benches"
    )
    if args.kernel:
        label += f" (kernel ~ '{args.kernel}')"
    if args.filter:
        label += f" (bench ~ '{args.filter}')"
    label += f", sorted by {sort_desc}:"

    show = dom if args.all else dom[: args.top]
    if args.csv:
        # Stable column order; downstream tooling can pivot/aggregate.
        print("bench,kernel,avg_us,instances,total_ms")
        for r in show:
            print(
                f"{r.bench},{kernel_kind(r.name)},{r.avg_us:.3f},"
                f"{r.instances},{r.total_ms:.3f}"
            )
        return 0
    total_ms_all = sum(r.total_ms for r in dom)
    print_table(show, label, total_ms_all_benches=total_ms_all)
    return 0


if __name__ == "__main__":
    sys.exit(main())
