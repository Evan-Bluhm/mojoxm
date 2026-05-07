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
    stddev_ns: float
    name: str

    @property
    def avg_us(self) -> float:
        return self.avg_ns / 1000.0

    @property
    def total_ms(self) -> float:
        return self.total_ns / 1_000_000.0

    @property
    def cv_pct(self) -> float:
        """Coefficient of variation -- stddev / avg, percent.  >50%
        typically means the bench runs a refinement sweep (multiple
        mesh sizes back-to-back) so the per-launch distribution is
        bimodal, not a real timing anomaly."""
        if self.avg_ns <= 0:
            return 0.0
        return 100.0 * self.stddev_ns / self.avg_ns


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
                stddev_ns=float(m["stddev_ns"]),
                name=m["name"],
            )
        )
    return rows


def bench_physics(bench: str) -> str:
    """Map a bench name to its physics module (one of: advection / euler /
    mhd / shallow_water / maxwell / two_fluid).  Used by --by-physics
    aggregation."""
    # mhd_glm_psi_* + mhd_brio_wu_* + mhd_alfven_glm_* all roll up to "mhd".
    # shallow_water comes before "water" so we just check the prefix.
    if bench.startswith("bench_advection_"):
        return "advection"
    if bench.startswith("bench_euler_"):
        return "euler"
    if bench.startswith("bench_mhd_"):
        return "mhd"
    if bench.startswith("bench_shallow_water_"):
        return "shallow_water"
    if bench.startswith("bench_maxwell_"):
        return "maxwell"
    if bench.startswith("bench_two_fluid_"):
        return "two_fluid"
    return "other"


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
    show_cv: bool = False,
) -> None:
    if not rows:
        print("(no rows)")
        return
    print(f"{header}")
    if show_cv:
        print(
            f"  {'bench':<48} {'kernel':<14} {'avg us':>9} {'inst':>7} "
            f"{'total ms':>10} {'CV %':>6}"
        )
        print(
            f"  {'-' * 48} {'-' * 14} {'-' * 9} {'-' * 7} "
            f"{'-' * 10} {'-' * 6}"
        )
    else:
        print(
            f"  {'bench':<48} {'kernel':<14} {'avg us':>9} {'inst':>7} "
            f"{'total ms':>10}"
        )
        print(f"  {'-' * 48} {'-' * 14} {'-' * 9} {'-' * 7} {'-' * 10}")
    shown_ms_sum = 0.0
    for r in rows:
        shown_ms_sum += r.total_ms
        if show_cv:
            print(
                f"  {r.bench:<48} {kernel_kind(r.name):<14} "
                f"{r.avg_us:>9.1f} {r.instances:>7d} {r.total_ms:>10.1f} "
                f"{r.cv_pct:>6.1f}"
            )
        else:
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


SELF_TEST_FIXTURE = """\
[6/8] Executing 'cuda_gpu_kern_sum' stats report

 Time (%)  Total Time (ns)  Instances  Avg (ns)  Med (ns)  Min (ns)  Max (ns)  StdDev (ns)                             Name
 --------  ---------------  ---------  --------  --------  --------  --------  -----------  -----------------------------------------------------------
     99.8         87479447       4680   18692.2   12896.0      6528     30753       9595.0  src_solver_rk_stage_kernel_I6A6A6A6AcB6A6A_3c8d0c8e9d13eed8
      0.1           100128          3   33376.0   21728.0     21312     57088      20536.2  src_local_mesh_build_elements_6A6A6A6A_b4fb09db0125e176
      0.0            31745          3   10581.7    8768.0      5856     17121       5847.4  src_local_mesh_build_faces_ker6A6A6A6A_3680e9f99a1681d1

[7/8] Executing 'cuda_gpu_mem_time_sum' stats report
"""


def run_self_test() -> int:
    """Validates the parser + classifiers against a hardcoded fixture
    snapshot of nsys cuda_gpu_kern_sum output.  Returns 0 on pass,
    1 on any assertion failure (with a description on stderr)."""
    failures: list[str] = []

    def expect(cond: bool, msg: str) -> None:
        if not cond:
            failures.append(msg)

    # Parse the fixture by writing it to a temp file (parse_file works on
    # Path inputs; that's the codepath we want to exercise).
    import tempfile
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".kern.txt", prefix="bench_self_test_", delete=False
    ) as f:
        f.write(SELF_TEST_FIXTURE)
        tmp_path = Path(f.name)
    try:
        rows = parse_file(tmp_path)
    finally:
        tmp_path.unlink()

    expect(len(rows) == 3, f"expected 3 rows, got {len(rows)}")
    if rows:
        expect(rows[0].time_pct == 99.8, f"row[0].time_pct = {rows[0].time_pct}")
        expect(rows[0].total_ns == 87479447, f"row[0].total_ns = {rows[0].total_ns}")
        expect(rows[0].instances == 4680, f"row[0].instances = {rows[0].instances}")
        expect(
            abs(rows[0].avg_ns - 18692.2) < 1e-6,
            f"row[0].avg_ns = {rows[0].avg_ns}",
        )
        expect(
            abs(rows[0].stddev_ns - 9595.0) < 1e-6,
            f"row[0].stddev_ns = {rows[0].stddev_ns}",
        )
        expect(
            "rk_stage_kernel" in rows[0].name,
            f"row[0].name = {rows[0].name}",
        )
        expect(
            abs(rows[0].avg_us - 18.6922) < 1e-3,
            f"row[0].avg_us = {rows[0].avg_us}",
        )
        # cv_pct = 100 * 9595 / 18692.2 = 51.331
        expect(
            abs(rows[0].cv_pct - 51.33) < 0.01,
            f"row[0].cv_pct = {rows[0].cv_pct}",
        )

    # dominant_per_bench: row[0] has 99.8% time_pct, should win.
    if len(rows) == 3:
        dom = dominant_per_bench(rows)
        expect(len(dom) == 1, f"expected 1 dominant row, got {len(dom)}")
        if dom:
            expect(
                "rk_stage_kernel" in dom[0].name,
                f"dominant kernel = {dom[0].name}",
            )

    # kernel_kind classifier sanity:
    test_cases_kk = [
        ("src_solver_rk_stage_kernel_I6_3c8d0c8e9d13eed8", "rk_stage"),
        ("src_local_mesh_build_elements_6A_b4fb09db0125e176", "build_elem"),
        ("src_local_mesh_build_faces_ker6A_3680e9f99a1681d1", "build_face"),
        ("src_local_mesh_2d_gpu_advectio6A_aaaaaaaaaaaaaaaa", "2d_advect"),
        ("src_local_mesh_2d_gpu_mhd_glm6A_bbbbbbbbbbbbbbbb", "2d_mhd_glm"),
        ("src_local_mesh_2d_gpu_mhd6A_cccccccccccccccc", "2d_mhd"),
    ]
    for name, expected_kind in test_cases_kk:
        got = kernel_kind(name)
        expect(
            got == expected_kind,
            f"kernel_kind({name!r}) = {got!r}, expected {expected_kind!r}",
        )

    # bench_physics classifier sanity:
    test_cases_phys = [
        ("bench_advection_translation_2d", "advection"),
        ("bench_euler_smooth_wave_3d_p5", "euler"),
        ("bench_mhd_alfven_3d_p5", "mhd"),
        ("bench_mhd_glm_psi_damp_3d", "mhd"),
        ("bench_mhd_brio_wu_3d_p3", "mhd"),
        ("bench_shallow_water_dam_break_2d", "shallow_water"),
        ("bench_maxwell_plane_wave_3d_p5", "maxwell"),
        ("bench_two_fluid_walls_3d_p5", "two_fluid"),
        ("bench_unknown_physics", "other"),
    ]
    for bench, expected_phys in test_cases_phys:
        got = bench_physics(bench)
        expect(
            got == expected_phys,
            f"bench_physics({bench!r}) = {got!r}, expected {expected_phys!r}",
        )

    # dominant_per_bench: synthesize rows from multiple distinct benches
    # and verify the highest-time_pct row wins for each.
    synth = [
        KernelRow("bench_a", 50.0, 1000, 10, 100.0, 5.0, "kernel_x"),
        KernelRow("bench_a", 25.0, 500, 10, 50.0, 2.5, "kernel_y"),
        KernelRow("bench_a", 25.0, 500, 10, 50.0, 2.5, "kernel_z"),
        KernelRow("bench_b", 60.0, 6000, 100, 60.0, 30.0, "kernel_x"),
        KernelRow("bench_b", 40.0, 4000, 100, 40.0, 1.0, "kernel_w"),
    ]
    dom_synth = dominant_per_bench(synth)
    expect(
        len(dom_synth) == 2,
        f"dominant_per_bench: expected 2 distinct benches, got {len(dom_synth)}",
    )
    by_bench = {r.bench: r for r in dom_synth}
    expect(
        by_bench["bench_a"].name == "kernel_x",
        f"bench_a dominant = {by_bench['bench_a'].name!r}, expected 'kernel_x'",
    )
    expect(
        by_bench["bench_b"].name == "kernel_x",
        f"bench_b dominant = {by_bench['bench_b'].name!r}, expected 'kernel_x'",
    )

    # cv_pct: stddev_ns=30, avg_ns=60 -> 50%
    cv_row = KernelRow("synth", 100.0, 6000, 100, 60.0, 30.0, "k")
    expect(
        abs(cv_row.cv_pct - 50.0) < 1e-9,
        f"cv_pct = {cv_row.cv_pct}, expected 50.0",
    )

    # --by-physics + --show-cv: max-CV reduction per group should pick
    # the highest per-bench CV.  Two synthetic euler benches with CVs
    # 50% and 10%; the group max should be 50%.
    cv_synth = [
        KernelRow("bench_euler_a", 100.0, 1000, 10, 100.0, 50.0, "kernel_x"),  # CV=50%
        KernelRow("bench_euler_b", 100.0, 2000, 20, 100.0, 10.0, "kernel_x"),  # CV=10%
        KernelRow("bench_advection_a", 100.0, 500, 5, 100.0, 80.0, "kernel_y"),  # CV=80%
    ]
    cv_by_phys: dict[str, list[KernelRow]] = {}
    for r in cv_synth:
        cv_by_phys.setdefault(bench_physics(r.bench), []).append(r)
    euler_max_cv = max(r.cv_pct for r in cv_by_phys["euler"])
    advection_max_cv = max(r.cv_pct for r in cv_by_phys["advection"])
    expect(
        abs(euler_max_cv - 50.0) < 1e-6,
        f"euler group max CV = {euler_max_cv}, expected 50.0",
    )
    expect(
        abs(advection_max_cv - 80.0) < 1e-6,
        f"advection group max CV = {advection_max_cv}, expected 80.0",
    )

    if failures:
        print("profile_summary.py self-test FAILED:", file=sys.stderr)
        for msg in failures:
            print(f"  * {msg}", file=sys.stderr)
        return 1
    n_dom_synth = 3  # 2 length checks + 2 winner checks ~= 3 named asserts
    n_cv_group = 2  # max-CV reduction per group
    print(
        f"profile_summary.py self-test PASSED ("
        f"{3 + len(test_cases_kk) + len(test_cases_phys) + n_dom_synth + 1 + n_cv_group} "
        f"assertions)"
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
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
    fmt_group = ap.add_mutually_exclusive_group()
    fmt_group.add_argument(
        "--csv",
        action="store_true",
        help=(
            "Emit machine-readable CSV instead of the human-readable table.  "
            "Mutually exclusive with --markdown."
        ),
    )
    fmt_group.add_argument(
        "--markdown",
        action="store_true",
        help=(
            "Emit a markdown table instead of the human-readable text "
            "table -- useful for pasting into PRs / issues / docs.  "
            "Mutually exclusive with --csv."
        ),
    )
    ap.add_argument(
        "--show-cv",
        action="store_true",
        help=(
            "Add a 'CV %%' column showing coefficient of variation "
            "(stddev / avg).  >50%% typically signals a refinement-sweep "
            "bench (multi-resolution back-to-back) rather than a real "
            "timing anomaly.  Combined with --by-physics, surfaces the "
            "max per-bench CV in each group (catches refinement-sweep "
            "modules at a glance)."
        ),
    )
    ap.add_argument(
        "--by-physics",
        action="store_true",
        help=(
            "Aggregate dominant-kernel time by physics module (advection / "
            "euler / mhd / shallow_water / maxwell / two_fluid).  Useful "
            "for sizing where the suite-wide compute budget actually sits."
        ),
    )
    ap.add_argument(
        "--self-test",
        action="store_true",
        help=(
            "Run a small in-script self-test of the parser + classifiers "
            "against hardcoded fixture rows.  Exits 0 if all assertions "
            "pass.  Use as a CI gate after editing this script (or the "
            "regex), or to detect a future nsys-output-format change."
        ),
    )
    args = ap.parse_args()

    if args.self_test:
        return run_self_test()

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
    if args.by_physics:
        # Group by physics module; sum total_ms and instance count.
        # Sort groups descending by total_ms.  Always operates on the
        # full dom set (not the --top slice), since aggregating just
        # the top-N benches isn't a meaningful suite roll-up.
        by_phys: dict[str, list[KernelRow]] = {}
        for r in dom:
            by_phys.setdefault(bench_physics(r.bench), []).append(r)
        # Optional max-CV summary per group when --show-cv is set --
        # surfaces refinement-sweep benches (multi-resolution back-to-
        # back) clustered into one physics module.  Group rows are
        # tuples; the trailing max_cv field is None when --show-cv is off.
        groups = [
            (
                phys,
                len(rows),
                sum(r.instances for r in rows),
                sum(r.total_ms for r in rows),
                max((r.cv_pct for r in rows), default=0.0)
                if args.show_cv
                else None,
            )
            for phys, rows in by_phys.items()
        ]
        groups.sort(key=lambda g: -g[3])
        suite_total_ms = sum(g[3] for g in groups)
        if args.csv:
            header = "physics,benches,instances,total_ms,share_pct"
            if args.show_cv:
                header += ",max_cv_pct"
            print(header)
            for phys, n_benches, n_inst, t_ms, max_cv in groups:
                share = 100.0 * t_ms / suite_total_ms if suite_total_ms > 0 else 0.0
                row = f"{phys},{n_benches},{n_inst},{t_ms:.3f},{share:.2f}"
                if args.show_cv:
                    row += f",{max_cv:.2f}"
                print(row)
            return 0
        if args.markdown:
            if args.show_cv:
                print("| physics | benches | launches | total ms | share | max CV % |")
                print("|---|---:|---:|---:|---:|---:|")
            else:
                print("| physics | benches | launches | total ms | share |")
                print("|---|---:|---:|---:|---:|")
            for phys, n_benches, n_inst, t_ms, max_cv in groups:
                share = 100.0 * t_ms / suite_total_ms if suite_total_ms > 0 else 0.0
                row = (
                    f"| {phys} | {n_benches} | {n_inst} | "
                    f"{t_ms:.1f} | {share:.1f}% |"
                )
                if args.show_cv:
                    row += f" {max_cv:.1f} |"
                print(row)
            return 0
        # Different label semantics in --by-physics: --top is irrelevant
        # since we're aggregating, and the rows are sorted by group total
        # not the per-bench --sort key.
        phys_label = f"all {len(dom)} benches grouped by physics module"
        if args.kernel:
            phys_label += f" (kernel ~ '{args.kernel}')"
        if args.filter:
            phys_label += f" (bench ~ '{args.filter}')"
        phys_label += ", sorted by group total ms:"
        print(phys_label)
        if args.show_cv:
            print(
                f"  {'physics':<14} {'benches':>8} {'launches':>10} "
                f"{'total ms':>11} {'share':>7} {'max CV %':>9}"
            )
            print(
                f"  {'-' * 14} {'-' * 8} {'-' * 10} {'-' * 11} "
                f"{'-' * 7} {'-' * 9}"
            )
            for phys, n_benches, n_inst, t_ms, max_cv in groups:
                share = 100.0 * t_ms / suite_total_ms if suite_total_ms > 0 else 0.0
                print(
                    f"  {phys:<14} {n_benches:>8d} {n_inst:>10d} "
                    f"{t_ms:>11.1f} {share:>6.1f}% {max_cv:>8.1f}%"
                )
        else:
            print(
                f"  {'physics':<14} {'benches':>8} {'launches':>10} "
                f"{'total ms':>11} {'share':>7}"
            )
            print(f"  {'-' * 14} {'-' * 8} {'-' * 10} {'-' * 11} {'-' * 7}")
            for phys, n_benches, n_inst, t_ms, _ in groups:
                share = 100.0 * t_ms / suite_total_ms if suite_total_ms > 0 else 0.0
                print(
                    f"  {phys:<14} {n_benches:>8d} {n_inst:>10d} "
                    f"{t_ms:>11.1f} {share:>6.1f}%"
                )
        print(
            f"  -> {len(dom)} benches across {len(groups)} physics modules, "
            f"{suite_total_ms / 1000:.2f} s total"
        )
        return 0
    if args.csv:
        # Stable column order; downstream tooling can pivot/aggregate.
        print("bench,kernel,avg_us,instances,total_ms,cv_pct")
        for r in show:
            print(
                f"{r.bench},{kernel_kind(r.name)},{r.avg_us:.3f},"
                f"{r.instances},{r.total_ms:.3f},{r.cv_pct:.3f}"
            )
        return 0
    if args.markdown:
        cv_col = " CV % |" if args.show_cv else ""
        cv_sep = " ---: |" if args.show_cv else ""
        print(
            f"| bench | kernel | avg us | inst | total ms |{cv_col}"
        )
        print(
            f"|---|---|---:|---:|---:|{cv_sep}"
        )
        for r in show:
            cv_cell = f" {r.cv_pct:.1f} |" if args.show_cv else ""
            print(
                f"| `{r.bench}` | {kernel_kind(r.name)} | "
                f"{r.avg_us:.1f} | {r.instances} | "
                f"{r.total_ms:.1f} |{cv_cell}"
            )
        return 0
    total_ms_all = sum(r.total_ms for r in dom)
    print_table(
        show,
        label,
        total_ms_all_benches=total_ms_all,
        show_cv=args.show_cv,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
