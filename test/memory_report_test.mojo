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
from std.time import perf_counter_ns

from src.partition import build_partition
from src.reference import ReferenceElement, to_float32, num_tet_nodes
from src.mesh import Mesh
from src.memory_report import ThroughputReport
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
        ctx, build_partition(rank, size, 4, 4, 4), 1.0, 1.0, 1.0,
        BoundaryConditions.periodic(),
    )
    var halo = HaloExchange(
        ctx, mesh.part, Advection.NUM_COMPONENTS,
        mesh.d_perm.unsafe_ptr(),
    )
    var physics = Advection(Float32(1.0), Float32(0.0), Float32(0.0))
    var solver = Solver[Advection, P](
        ctx^, mesh^, halo^, physics^,
        D_ref^, Lift_ref^, node_weights^,
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

    var num_steps = 20
    var dt = Float32(1.0e-3)
    var t_start = perf_counter_ns()
    for _ in range(num_steps):
        solver.step_ssprk3(dt, nvtx)
    solver.ctx.synchronize()
    var t_end = perf_counter_ns()
    var wall_seconds = Float64(t_end - t_start) * 1.0e-9

    var tput = ThroughputReport(num_steps, wall_seconds, solver.dof_count())
    if tput.dof_per_second() <= 0.0:
        raise Error("memory_report_test: dof_per_second non-positive")
    if tput.per_step_seconds() <= 0.0:
        raise Error("memory_report_test: per_step_seconds non-positive")

    print()
    tput.print()
    print()

    print("=== memory_report_test PASSED ===")
    mpi.finalize()
