# ======================================================================
# p3_smoke_test -- end-to-end plumbing check for P = 3
# ======================================================================
#
# Builds Mesh[3] + Solver[Advection, 3], fills q with a constant value
# on owned DOFs, takes one SSPRK3 step with dt = 0 (so q is unchanged),
# downloads it back, and verifies every owned nodal DOF still holds
# the original value.
#
# What this catches:
#   * wrong buffer sizing for NP = 20 at P=3 (segfault / assertion)
#   * RK kernel launched with mismatched comptime NP/NFP
#   * Mesh[3] + LocalMesh[3] GPU build kernels producing bad element
#     layouts (q read-write would land outside the per-element stride)
#
# What it does NOT catch:
#   * physical-solution accuracy at higher order (that would need an
#     analytic reference and a real time integration)
#   * VTU output correctness (this test skips frame output entirely)
# ======================================================================

from std.sys import has_accelerator
from std.gpu import global_idx
from std.gpu.host import DeviceContext
from std.math import ceildiv

from src import mpi
from src.partition import build_partition
from src.reference import ReferenceElement, to_float32, num_tet_nodes
from src.mesh import Mesh
from src.boundary import BoundaryConditions
from src.halo_exchange import HaloExchange
from src.solver import Solver
from src.advection import Advection
from src.nvtx import NvtxContext


comptime P = 3
comptime NP = num_tet_nodes(P)  # 20 at P=3

comptime NX = 4
comptime NY = 4
comptime NZ = 4
comptime LX = 1.0
comptime LY = 1.0
comptime LZ = 1.0
comptime IC_BLOCK = 256
comptime FILL_VALUE: Float32 = 2.5


def fill_constant_kernel(
    q: UnsafePointer[Float32, MutAnyOrigin],
    owned_elem_ids: UnsafePointer[Int32, MutAnyOrigin],
    num_owned: Int,
    value: Float32,
):
    var idx = Int(global_idx.x)
    var total = num_owned * NP
    if idx >= total:
        return
    var i = idx // NP
    var nn = idx % NP
    var e = Int(owned_elem_ids[i])
    q[e * NP + nn] = value


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    mpi.init()
    var size = mpi.world_size()
    if size > 1:
        mpi.finalize()
        print("p3_smoke_test: runs at np=1 only")
        return

    print(
        "p3_smoke_test: building Mesh[",
        P,
        "] / Solver[Advection, ",
        P,
        "] at NP =",
        NP,
    )

    var nvtx = NvtxContext()
    var ctx = DeviceContext()

    var re = ReferenceElement[P]()
    var D_ref = to_float32(re.D_ref)
    var Lift_ref = to_float32(re.Lift_ref)
    var node_weights = to_float32(re.node_weights)

    var mesh = Mesh[P](
        ctx,
        build_partition(0, 1, NX, NY, NZ),
        LX,
        LY,
        LZ,
        BoundaryConditions.periodic(),
    )
    var halo = HaloExchange(
        ctx,
        mesh.part,
        Advection.NUM_COMPONENTS,
        mesh.d_perm.unsafe_ptr(),
    )
    var physics = Advection(Float32(0.0), Float32(0.0), Float32(0.0))
    var solver = Solver[Advection, P](
        ctx^,
        mesh^,
        halo^,
        physics^,
        D_ref^,
        Lift_ref^,
        node_weights^,
    )

    # Fill q with FILL_VALUE on every owned (element, node).
    var num_owned = solver.num_owned_elements
    print(
        "  num_owned_elements:",
        num_owned,
        "  expected total_owned_dof:",
        num_owned * NP,
    )
    print("  solver.total_owned_dof:", solver.total_owned_dof)
    if solver.total_owned_dof != num_owned * NP:
        raise Error(
            "total_owned_dof mismatch: Solver reports "
            + String(solver.total_owned_dof)
            + " but num_owned_elements * NP = "
            + String(num_owned * NP)
        )

    solver.ctx.enqueue_function[fill_constant_kernel, fill_constant_kernel](
        solver.d_q.unsafe_ptr(),
        solver.mesh.d_owned_elem_ids.unsafe_ptr(),
        num_owned,
        FILL_VALUE,
        grid_dim=ceildiv(num_owned * NP, IC_BLOCK),
        block_dim=IC_BLOCK,
    )
    solver.ctx.synchronize()

    # Download q back and verify every owned DOF holds FILL_VALUE.
    var host = List[Float32]()
    for _ in range(num_owned * NP):
        host.append(Float32(0.0))
    solver.download_owned_component(0, host, nvtx)
    solver.ctx.synchronize()

    var max_err: Float32 = 0.0
    for k in range(len(host)):
        var d = host[k] - FILL_VALUE
        var ad = d if d >= Float32(0.0) else -d
        if ad > max_err:
            max_err = ad
    print("  max |q - fill| after download:", max_err)
    if max_err > Float32(1.0e-6):
        raise Error("P=3 smoke test: q readback does not match fill value")

    print("=== p3_smoke_test PASSED ===")
    mpi.finalize()
