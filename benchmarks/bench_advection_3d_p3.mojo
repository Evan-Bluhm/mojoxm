# ======================================================================
# bench_advection_3d_p3 -- 3D Gaussian one-period advection at P=3
# ======================================================================
#
# P=3 counterpart of bench_advection_3d.  Same Gaussian transport in a
# triply-periodic [0, 1]^3 box, but routed through Mesh[3] +
# Solver[Advection, 3] with NP = 20 nodes per tet.  At P=3 the
# expected asymptotic convergence rate is P+1 = 4, so observed rates
# should sit near 4 (or above 3 in the pre-asymptotic regime at our
# resolution range).
#
# This is the only analytic-rate benchmark that exercises the
# parameterized-P pipeline; p3_smoke_test only round-trips a constant
# field.  Validates that higher-order operators (D_ref, Lift_ref,
# node_weights at NP=20) actually produce the design-rate accuracy.
#
# Pass criteria (P=3, SSPRK3, CFL=0.1, single-rank):
#   * rel L2(q) at N=12 < 5e-3 (Gaussian is well-resolved at P=3)
#   * rel L2 decreases monotonically under refinement N=6, 8, 12
#   * observed convergence rate >= 2.5 between N=8 and N=12
#     (theoretical 4; pre-asymptotic so 2.5 is a generous floor)
#   * no NaN / Inf
# ======================================================================

from std.sys import has_accelerator
from std.gpu import global_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from std.math import sqrt, ceildiv, exp, log, isnan, isinf

from src import mpi
from src.partition import build_partition
from src.reference import (
    ReferenceElement, to_float32, num_tet_nodes,
)
from src.mesh import Mesh
from src.boundary import BoundaryConditions
from src.halo_exchange import HaloExchange
from src.solver import Solver
from src.advection import Advection
from src.nvtx import NvtxContext


comptime P = 3
comptime NP = num_tet_nodes(P)   # 20 at P=3

comptime LX = 1.0
comptime LY = 1.0
comptime LZ = 1.0
comptime VX: Float32 = 1.0
comptime VY: Float32 = 1.0
comptime VZ: Float32 = 1.0
comptime T_FINAL: Float32 = 1.0
comptime CFL = Float32(0.1)
comptime GAUSS_SIGMA: Float32 = 0.12
comptime IC_BLOCK = 256

comptime L2_MAX_REL_AT_12: Float64 = 5.0e-3
comptime RATE_MIN: Float64 = 2.5


def gaussian_ic_kernel(
    q: UnsafePointer[Float32, MutAnyOrigin],
    owned_elem_ids: UnsafePointer[Int32, MutAnyOrigin],
    elem_node_xyz:  UnsafePointer[Float32, MutAnyOrigin],
    num_owned: Int,
    cx: Float32, cy: Float32, cz: Float32,
    inv_two_sigma2: Float32,
):
    var idx = Int(global_idx.x)
    var total = num_owned * NP
    if idx >= total:
        return
    var i = idx // NP
    var nn = idx % NP
    var e = Int(owned_elem_ids[i])
    var px = elem_node_xyz[(e * NP + nn) * 3 + 0]
    var py = elem_node_xyz[(e * NP + nn) * 3 + 1]
    var pz = elem_node_xyz[(e * NP + nn) * 3 + 2]
    var dx = px - cx
    if dx >  Float32(LX * 0.5): dx -= Float32(LX)
    if dx < -Float32(LX * 0.5): dx += Float32(LX)
    var dy = py - cy
    if dy >  Float32(LY * 0.5): dy -= Float32(LY)
    if dy < -Float32(LY * 0.5): dy += Float32(LY)
    var dz = pz - cz
    if dz >  Float32(LZ * 0.5): dz -= Float32(LZ)
    if dz < -Float32(LZ * 0.5): dz += Float32(LZ)
    q[e * NP + nn] = exp(
        -(dx * dx + dy * dy + dz * dz) * inv_two_sigma2
    )


def _run(N: Int) raises -> Float64:
    var rank = mpi.world_rank()
    var size = mpi.world_size()
    var nvtx = NvtxContext()
    var ctx = DeviceContext()

    # Build reference operators directly at P=3.  Cannot use the
    # P=2-defaulting build_reference_operators() in src.reference
    # because Solver[Advection, P=3] needs NP=20 D_ref / Lift_ref /
    # node_weights.
    var re = ReferenceElement[P]()
    var D_ref = to_float32(re.D_ref)
    var Lift_ref = to_float32(re.Lift_ref)
    var node_weights = to_float32(re.node_weights)

    var mesh = Mesh[P](
        ctx, build_partition(rank, size, N, N, N), LX, LY, LZ,
        BoundaryConditions.periodic(),
    )
    var halo = HaloExchange(
        ctx, mesh.part, Advection.NUM_COMPONENTS,
        mesh.d_perm.unsafe_ptr(),
    )
    var physics = Advection(VX, VY, VZ)
    var solver = Solver[Advection, P](
        ctx^, mesh^, halo^, physics^, D_ref^, Lift_ref^, node_weights^,
    )

    var inv_two_sigma2 = Float32(1.0) / (
        Float32(2.0) * GAUSS_SIGMA * GAUSS_SIGMA
    )
    solver.ctx.enqueue_function[gaussian_ic_kernel, gaussian_ic_kernel](
        solver.d_q.unsafe_ptr(),
        solver.mesh.d_owned_elem_ids.unsafe_ptr(),
        solver.mesh.local.d_elem_node_xyz.unsafe_ptr(),
        solver.num_owned_elements,
        Float32(0.5), Float32(0.5), Float32(0.5),
        inv_two_sigma2,
        grid_dim=ceildiv(solver.num_owned_elements * NP, IC_BLOCK),
        block_dim=IC_BLOCK,
    )
    solver.ctx.synchronize()

    var n_owned_dof = solver.num_owned_elements * NP
    var hbuf_ic = solver.ctx.enqueue_create_host_buffer[DType.float32](
        n_owned_dof
    )
    solver.ctx.enqueue_copy(
        hbuf_ic,
        solver.d_q.create_sub_buffer[DType.float32](0, n_owned_dof),
    )
    solver.ctx.synchronize()
    var ic_ptr = hbuf_ic.unsafe_ptr()
    var host_ic = List[Float32]()
    for k in range(n_owned_dof):
        host_ic.append(ic_ptr[k])

    # CFL: tighter at higher P (factor 2P+1 = 7 at P=3).
    var h = Float32(LX) / Float32(N)
    var v = sqrt(VX * VX + VY * VY + VZ * VZ)
    var dt_est = CFL * h / (v * Float32(2 * P + 1))
    var num_steps = Int(T_FINAL / dt_est) + 1
    var dt = T_FINAL / Float32(num_steps)

    for _ in range(num_steps):
        solver.step_ssprk3(dt, nvtx)
    solver.ctx.synchronize()

    var hbuf_q = solver.ctx.enqueue_create_host_buffer[DType.float32](
        n_owned_dof
    )
    solver.ctx.enqueue_copy(
        hbuf_q,
        solver.d_q.create_sub_buffer[DType.float32](0, n_owned_dof),
    )
    solver.ctx.synchronize()
    var q_ptr = hbuf_q.unsafe_ptr()

    var sum_sq: Float64 = 0.0
    var sum_ic: Float64 = 0.0
    for k in range(n_owned_dof):
        var v_now = q_ptr[k]
        if isnan(v_now) or isinf(v_now):
            raise Error("bench_advection_3d_p3: non-finite output")
        var e = Float64(v_now - host_ic[k])
        sum_sq += e * e
        var ic = Float64(host_ic[k])
        sum_ic += ic * ic
    var l2 = sqrt(sum_sq / Float64(n_owned_dof))
    var l2_ic = sqrt(sum_ic / Float64(n_owned_dof))
    return l2 / l2_ic


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    mpi.init()
    var size = mpi.world_size()
    if size > 1:
        mpi.finalize()
        print("bench_advection_3d_p3: runs at np=1 only")
        return

    print("bench_advection_3d_p3 (P=3 Gaussian one-period advection)")
    print("  P=", P, "  NP=", NP, "  refinement sweep N=6, 8, 12")

    var err6 = _run(6)
    print("  N=6   rel L2 =", err6)
    var err8 = _run(8)
    print("  N=8   rel L2 =", err8)
    var err12 = _run(12)
    print("  N=12  rel L2 =", err12)

    if err12 > L2_MAX_REL_AT_12:
        raise Error(
            "bench_advection_3d_p3 FAILED: rel L2 at N=12 "
            + String(err12) + " exceeds " + String(L2_MAX_REL_AT_12)
        )
    if not (err6 > err8 and err8 > err12):
        raise Error(
            "bench_advection_3d_p3 FAILED: rel L2 did not decrease "
            + "monotonically (6: " + String(err6)
            + ", 8: " + String(err8)
            + ", 12: " + String(err12) + ")"
        )

    var rate_68  = log(err6 / err8)  / log(8.0 / 6.0)
    var rate_812 = log(err8 / err12) / log(12.0 / 8.0)
    print("  observed rates: log_(4/3)(e6/e8) =", rate_68,
          "  log_(3/2)(e8/e12) =", rate_812,
          "  (P+1 =", P + 1, ", floor", RATE_MIN, ")")
    if rate_68 < RATE_MIN and rate_812 < RATE_MIN:
        raise Error(
            "bench_advection_3d_p3 FAILED: observed rates "
            + String(rate_68) + " and " + String(rate_812)
            + " both below " + String(RATE_MIN)
        )

    print("=== bench_advection_3d_p3 PASSED ===")
    mpi.finalize()
