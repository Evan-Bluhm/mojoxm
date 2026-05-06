# ======================================================================
# memory_report_test -- smoke test for memory + throughput reports
# ======================================================================
#
# Builds a tiny Mesh + Solver[Advection], runs a few SSPRK3 steps,
# asks for both reports, and checks that:
#   (1) Every memory category has a positive byte count except halo
#       (zero at np=1 since ring_count[d] == 0 for every dir).
#   (2) total_device_bytes() equals the per-category sum.
#   (3) `solver.dof_count()` matches the expected formula.
#   (4) ThroughputReport derives a positive DOF/s after a real step
#       loop.
#   (5) Both `.print()` methods run without raising.
#
# Parameterised over P in {2, 3, 4, 5} so a P-specific bug in
# `Solver.memory_report()` -- e.g. a per-category byte calculation
# that uses the wrong NP and silently underreports at NP=20/35/56 --
# would be caught at test-utils latency rather than only by reading
# a real driver's startup banner.  All five invariants above hold at
# every P; only the absolute byte counts differ.
# ======================================================================

from src import mpi
from std.sys import has_accelerator
from std.gpu.host import DeviceContext

from src.partition import build_partition
from src.reference import ReferenceElement, to_float32, num_tet_nodes
from src.mesh import Mesh
from src.boundary import BoundaryConditions
from src.halo_exchange import HaloExchange
from src.solver import Solver
from src.advection import Advection
from src.nvtx import NvtxContext


def check[P: Int](mut nvtx: NvtxContext, rank: Int, size: Int) raises:
    print("  P=", P)
    var ctx = DeviceContext()

    var re = ReferenceElement[P]()
    var D_ref = to_float32(re.D_ref)
    var Lift_ref = to_float32(re.Lift_ref)
    var node_weights = to_float32(re.node_weights)

    var mesh = Mesh[P](
        ctx=ctx,
        part=build_partition(rank=rank, nprocs=size, nx=4, ny=4, nz=4),
        Lx=1.0,
        Ly=1.0,
        Lz=1.0,
        bcs=BoundaryConditions.periodic(),
    )
    var halo = HaloExchange(
        ctx=ctx,
        part=mesh.part,
        nc=Advection.NUM_COMPONENTS,
        d_perm=mesh.d_perm.unsafe_ptr(),
    )
    var physics = Advection(
        vx=Float32(1.0),
        vy=Float32(0.0),
        vz=Float32(0.0),
    )
    var solver = Solver[Advection, P](
        ctx=ctx^,
        mesh=mesh^,
        halo=halo^,
        physics=physics^,
        D_ref=D_ref^,
        Lift_ref=Lift_ref^,
        node_weights=node_weights^,
    )

    var rep = solver.memory_report()

    print("    rk_stage_bytes        =", rep.rk_stage_bytes)
    print("    dg_operators_bytes    =", rep.dg_operators_bytes)
    print("    limiter_bytes         =", rep.limiter_bytes)
    print("    mesh_connectivity_bytes =", rep.mesh_connectivity_bytes)
    print("    halo_device_bytes     =", rep.halo_device_bytes)
    print("    halo_pinned_bytes     =", rep.halo_pinned_bytes)

    if rep.rk_stage_bytes <= 0:
        raise Error(
            "memory_report_test P="
            + String(P)
            + ": rk_stage_bytes non-positive"
        )
    if rep.dg_operators_bytes <= 0:
        raise Error(
            "memory_report_test P="
            + String(P)
            + ": dg_operators_bytes non-positive"
        )
    if rep.limiter_bytes <= 0:
        raise Error(
            "memory_report_test P=" + String(P) + ": limiter_bytes non-positive"
        )
    if rep.mesh_connectivity_bytes <= 0:
        raise Error(
            "memory_report_test P="
            + String(P)
            + ": mesh_connectivity_bytes non-positive"
        )
    # At np=1 every ring has zero entries.
    if rep.halo_device_bytes != 0:
        raise Error(
            "memory_report_test P="
            + String(P)
            + ": halo_device_bytes nonzero at np=1"
        )
    if rep.halo_pinned_bytes != 0:
        raise Error(
            "memory_report_test P="
            + String(P)
            + ": halo_pinned_bytes nonzero at np=1"
        )

    var sum_check = (
        rep.rk_stage_bytes
        + rep.dg_operators_bytes
        + rep.limiter_bytes
        + rep.mesh_connectivity_bytes
        + rep.halo_device_bytes
    )
    if sum_check != rep.total_device_bytes():
        raise Error(
            "memory_report_test P="
            + String(P)
            + ": total_device_bytes ("
            + String(rep.total_device_bytes())
            + ") != per-category sum ("
            + String(sum_check)
            + ")"
        )

    # Monotone byte counts: rk_stage_bytes scales as 3 * num_owned * NP * NC
    # so it grows monotonically with NP at fixed mesh + NC.
    # (Sanity: NP at P=2..5 is 10/20/35/56, so rk_stage_bytes at P=5 should
    # be 5.6x its P=2 value at this mesh.)
    var nominal_rk = 3 * solver.num_owned_elements * num_tet_nodes(P) * 4
    if rep.rk_stage_bytes != nominal_rk:
        raise Error(
            "memory_report_test P="
            + String(P)
            + ": rk_stage_bytes "
            + String(rep.rk_stage_bytes)
            + " != nominal 3*num_owned*NP*sizeof(F32) = "
            + String(nominal_rk)
        )

    rep.print()

    # --- Throughput report after a short step loop ----------------------
    var expected_dof = solver.num_owned_elements * num_tet_nodes(P) * 1
    if solver.dof_count() != expected_dof:
        raise Error(
            "memory_report_test P="
            + String(P)
            + ": dof_count "
            + String(solver.dof_count())
            + " != expected "
            + String(expected_dof)
        )

    var dt = Float32(1.0e-3)
    var tput = solver.bench_step_loop(
        dt,
        nvtx,
        warmup_steps=2,
        measure_steps=10,
    )
    if tput.dof_per_second() <= 0.0:
        raise Error(
            "memory_report_test P="
            + String(P)
            + ": dof_per_second non-positive"
        )
    if tput.per_step_seconds() <= 0.0:
        raise Error(
            "memory_report_test P="
            + String(P)
            + ": per_step_seconds non-positive"
        )
    if tput.state_bandwidth_bytes_per_second() <= 0.0:
        raise Error(
            "memory_report_test P="
            + String(P)
            + ": state_bandwidth non-positive"
        )
    # Internal consistency: dof_per_second == dof_count / per_step_seconds.
    # Catches a regression where one of the two methods forgets to scale
    # by num_steps (e.g. dof_per_second using wall_seconds in place of
    # per_step_seconds, or vice versa).
    var derived_rate = Float64(tput.dof_count) / tput.per_step_seconds()
    var rate_err = derived_rate - tput.dof_per_second()
    if rate_err < 0.0:
        rate_err = -rate_err
    var rate_rel = rate_err / tput.dof_per_second()
    if rate_rel > 1.0e-9:
        raise Error(
            "memory_report_test P="
            + String(P)
            + ": throughput consistency violated -- dof_per_second="
            + String(tput.dof_per_second())
            + " vs dof_count/per_step_seconds="
            + String(derived_rate)
            + " (rel err "
            + String(rate_rel)
            + ")"
        )
    # Upper-bound sanity: physically plausible range.  This catches the
    # async-enqueue-only timing bug that originally produced ~9 TB/s
    # numbers (which is impossible on any real GPU).  Modern HBM peaks
    # near 3 TB/s; any STATE-bandwidth lower bound exceeding 1 TB/s
    # means we likely measured kernel enqueue, not actual GPU work.
    var bw = tput.state_bandwidth_bytes_per_second()
    if bw > 1.0e12:
        raise Error(
            "memory_report_test P="
            + String(P)
            + ": state_bandwidth "
            + String(bw)
            + " B/s exceeds 1 TB/s sanity ceiling -- timing likely "
            "missing a ctx.synchronize()"
        )
    # Sanity: 8 * total_q_len * 4 bytes.
    var expected_state_bytes = 8 * solver.total_q_len * 4
    if solver.state_bytes_per_step() != expected_state_bytes:
        raise Error(
            "memory_report_test P="
            + String(P)
            + ": state_bytes_per_step "
            + String(solver.state_bytes_per_step())
            + " != expected "
            + String(expected_state_bytes)
        )

    tput.print()


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    mpi.init()
    var size = mpi.world_size()
    if size > 1:
        mpi.finalize()
        print("memory_report_test: runs at np=1 only")
        return

    print("memory_report_test (Solver.memory_report() smoke, P=2..5)")
    var rank = mpi.world_rank()
    var nvtx = NvtxContext()
    check[2](nvtx, rank, size)
    check[3](nvtx, rank, size)
    check[4](nvtx, rank, size)
    check[5](nvtx, rank, size)
    print("=== memory_report_test PASSED ===")
    mpi.finalize()
