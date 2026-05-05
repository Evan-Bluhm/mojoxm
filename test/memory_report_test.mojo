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
# ======================================================================

from src import mpi
from std.math import ceildiv
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


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    mpi.init()
    var size = mpi.world_size()
    if size > 1:
        mpi.finalize()
        print("memory_report_test: runs at np=1 only")
        return

    print("memory_report_test (Solver.memory_report() smoke)")

    comptime P = 2
    var rank = mpi.world_rank()
    var nvtx = NvtxContext()
    var ctx = DeviceContext()

    var re = ReferenceElement[P]()
    var D_ref = to_float32(re.D_ref)
    var Lift_ref = to_float32(re.Lift_ref)
    var node_weights = to_float32(re.node_weights)

    var mesh = Mesh[P](
        ctx=ctx,
        part=build_partition(rank=rank, nprocs=size, nx=4, ny=4, nz=4),
        Lx=1.0, Ly=1.0, Lz=1.0,
        bcs=BoundaryConditions.periodic(),
    )
    var halo = HaloExchange(
        ctx=ctx,
        part=mesh.part,
        nc=Advection.NUM_COMPONENTS,
        d_perm=mesh.d_perm.unsafe_ptr(),
    )
    var physics = Advection(
        vx=Float32(1.0), vy=Float32(0.0), vz=Float32(0.0),
    )
    var solver = Solver[Advection, P](
        ctx=ctx^, mesh=mesh^, halo=halo^, physics=physics^,
        D_ref=D_ref^, Lift_ref=Lift_ref^, node_weights=node_weights^,
    )

    var rep = solver.memory_report()

    print("  rk_stage_bytes        =", rep.rk_stage_bytes)
    print("  dg_operators_bytes    =", rep.dg_operators_bytes)
    print("  limiter_bytes         =", rep.limiter_bytes)
    print("  mesh_connectivity_bytes =", rep.mesh_connectivity_bytes)
    print("  halo_device_bytes     =", rep.halo_device_bytes)
    print("  halo_pinned_bytes     =", rep.halo_pinned_bytes)

    if rep.rk_stage_bytes <= 0:
        raise Error("memory_report_test: rk_stage_bytes non-positive")
    if rep.dg_operators_bytes <= 0:
        raise Error("memory_report_test: dg_operators_bytes non-positive")
    if rep.limiter_bytes <= 0:
        raise Error("memory_report_test: limiter_bytes non-positive")
    if rep.mesh_connectivity_bytes <= 0:
        raise Error("memory_report_test: mesh_connectivity_bytes non-positive")
    # At np=1 every ring has zero entries.
    if rep.halo_device_bytes != 0:
        raise Error("memory_report_test: halo_device_bytes nonzero at np=1")
    if rep.halo_pinned_bytes != 0:
        raise Error("memory_report_test: halo_pinned_bytes nonzero at np=1")

    var sum_check = (
        rep.rk_stage_bytes
        + rep.dg_operators_bytes
        + rep.limiter_bytes
        + rep.mesh_connectivity_bytes
        + rep.halo_device_bytes
    )
    if sum_check != rep.total_device_bytes():
        raise Error(
            "memory_report_test: total_device_bytes ("
            + String(rep.total_device_bytes())
            + ") != per-category sum ("
            + String(sum_check) + ")"
        )

    print()
    rep.print()
    print()

    # --- Throughput report after a short step loop ----------------------
    var expected_dof = solver.num_owned_elements * num_tet_nodes(P) * 1
    if solver.dof_count() != expected_dof:
        raise Error(
            "memory_report_test: dof_count "
            + String(solver.dof_count())
            + " != expected " + String(expected_dof)
        )

    var dt = Float32(1.0e-3)
    var tput = solver.bench_step_loop(
        dt, nvtx, warmup_steps=2, measure_steps=10,
    )
    if tput.dof_per_second() <= 0.0:
        raise Error("memory_report_test: dof_per_second non-positive")
    if tput.per_step_seconds() <= 0.0:
        raise Error("memory_report_test: per_step_seconds non-positive")
    if tput.state_bandwidth_bytes_per_second() <= 0.0:
        raise Error("memory_report_test: state_bandwidth non-positive")
    # Upper-bound sanity: physically plausible range.  This catches the
    # async-enqueue-only timing bug that originally produced ~9 TB/s
    # numbers (which is impossible on any real GPU).  Modern HBM peaks
    # near 3 TB/s; any STATE-bandwidth lower bound exceeding 1 TB/s
    # means we likely measured kernel enqueue, not actual GPU work.
    var bw = tput.state_bandwidth_bytes_per_second()
    if bw > 1.0e12:
        raise Error(
            "memory_report_test: state_bandwidth "
            + String(bw)
            + " B/s exceeds 1 TB/s sanity ceiling -- timing likely "
            "missing a ctx.synchronize()"
        )
    # Sanity: 8 * total_q_len * 4 bytes.
    var expected_state_bytes = 8 * solver.total_q_len * 4
    if solver.state_bytes_per_step() != expected_state_bytes:
        raise Error(
            "memory_report_test: state_bytes_per_step "
            + String(solver.state_bytes_per_step())
            + " != expected " + String(expected_state_bytes)
        )

    print()
    tput.print()
    print()

    print("=== memory_report_test PASSED ===")
    mpi.finalize()
