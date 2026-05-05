# ======================================================================
# memory_report_test -- smoke test for Solver.memory_report()
# ======================================================================
#
# Builds a tiny Mesh + Solver[Advection], asks for the memory report,
# checks that:
#   (1) Every category has a positive byte count except halo (which
#       is zero at np=1 since ring_count[d] == 0 for every dir).
#   (2) total_device_bytes() equals the per-category sum.
#   (3) The print() method runs without raising.
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

    print("=== memory_report_test PASSED ===")
    mpi.finalize()
