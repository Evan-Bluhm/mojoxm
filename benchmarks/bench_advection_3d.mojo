# ======================================================================
# bench_advection_3d -- 3D Gaussian one-period advection convergence
# ======================================================================
#
# Scalar advection in 3D on a triply-periodic [0, 1]^3 Kuhn-tet mesh.
# A Gaussian bump is advected at v = (1, 1, 1) for exactly one period
# (T = 1), so the exact solution equals the IC; residual L2 is pure
# scheme dissipation.  Counterpart to bench_advection_translation_2d
# but through the full 3D Mesh / Solver / HaloExchange stack.
#
# Pass criteria (P=2, SSPRK3, CFL=0.2, single-rank):
#   * rel L2(q) at N=16 < 5%%
#   * rel L2 decreases monotonically under refinement N=8, 12, 16
#   * observed convergence rate log2(e_N / e_{3N/2}) >= 1.5 between
#     at least one pair (pre-asymptotic Rusanov on upwind is
#     typically ~2.0-2.5 at these resolutions; 1.5 is a generous
#     regression floor)
#   * no NaN / Inf
# ======================================================================

from std.sys import has_accelerator
from std.gpu import global_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from std.math import sqrt, ceildiv, exp, log, isnan, isinf

from src import mpi
from src.partition import build_partition
from src.reference import N_P, build_reference_operators
from src.mesh import Mesh
from src.boundary import BoundaryConditions
from src.halo_exchange import HaloExchange
from src.solver import Solver
from src.advection import Advection
from src.nvtx import NvtxContext


comptime P = 2
comptime LX = 1.0
comptime LY = 1.0
comptime LZ = 1.0
comptime VX: Float32 = 1.0
comptime VY: Float32 = 1.0
comptime VZ: Float32 = 1.0
comptime T_FINAL: Float32 = 1.0
comptime CFL = Float32(0.2)
comptime GAUSS_SIGMA: Float32 = 0.12
comptime IC_BLOCK = 256

comptime L2_MAX_REL_AT_16: Float64 = 0.05
comptime RATE_MIN:         Float64 = 1.5


def gaussian_ic_kernel(
    q: UnsafePointer[Float32, MutAnyOrigin],
    owned_elem_ids: UnsafePointer[Int32, MutAnyOrigin],
    elem_node_xyz:  UnsafePointer[Float32, MutAnyOrigin],
    num_owned: Int,
    cx: Float32, cy: Float32, cz: Float32,
    inv_two_sigma2: Float32,
):
    var idx = Int(global_idx.x)
    var total = num_owned * N_P
    if idx >= total:
        return
    var i = idx // N_P
    var nn = idx % N_P
    var e = Int(owned_elem_ids[i])
    var px = elem_node_xyz[(e * N_P + nn) * 3 + 0]
    var py = elem_node_xyz[(e * N_P + nn) * 3 + 1]
    var pz = elem_node_xyz[(e * N_P + nn) * 3 + 2]
    var dx = px - cx
    if dx >  Float32(LX * 0.5): dx -= Float32(LX)
    if dx < -Float32(LX * 0.5): dx += Float32(LX)
    var dy = py - cy
    if dy >  Float32(LY * 0.5): dy -= Float32(LY)
    if dy < -Float32(LY * 0.5): dy += Float32(LY)
    var dz = pz - cz
    if dz >  Float32(LZ * 0.5): dz -= Float32(LZ)
    if dz < -Float32(LZ * 0.5): dz += Float32(LZ)
    q[e * N_P + nn] = exp(
        -(dx * dx + dy * dy + dz * dz) * inv_two_sigma2
    )


def _run(N: Int) raises -> Float64:
    """Run one period at mesh NxNxN and return rel L2(q - q_ic)."""
    var rank = mpi.world_rank()
    var size = mpi.world_size()
    var nvtx = NvtxContext()
    var refs = build_reference_operators(nvtx)
    var ctx = DeviceContext()

    var mesh = Mesh(
        ctx, build_partition(rank, size, N, N, N), LX, LY, LZ,
        BoundaryConditions.periodic(),
    )
    var halo = HaloExchange(
        ctx, mesh.part, Advection.NUM_COMPONENTS,
        mesh.d_perm.unsafe_ptr(),
    )
    var physics = Advection(VX, VY, VZ)
    var solver = Solver[Advection](
        ctx^, mesh^, halo^, physics^, refs.D_ref^, refs.Lift_ref^, refs.node_weights^,
    )

    # Initial condition: Gaussian centered at (0.5, 0.5, 0.5).
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
        grid_dim=ceildiv(solver.num_owned_elements * N_P, IC_BLOCK),
        block_dim=IC_BLOCK,
    )
    solver.ctx.synchronize()

    # Snapshot the IC on host for later L2 comparison.
    var n_owned_dof = solver.num_owned_elements * N_P
    var hbuf_ic = solver.ctx.enqueue_create_host_buffer[DType.float32](n_owned_dof)
    solver.ctx.enqueue_copy(hbuf_ic, solver.d_q.create_sub_buffer[DType.float32](0, n_owned_dof))
    solver.ctx.synchronize()
    var ic_ptr = hbuf_ic.unsafe_ptr()
    var host_ic = List[Float32]()
    for k in range(n_owned_dof):
        host_ic.append(ic_ptr[k])

    # Choose dt and step count.  CFL for P=2 tet: ~ 1/(2p+1) = 1/5.
    var h = Float32(LX) / Float32(N)
    var v = sqrt(VX * VX + VY * VY + VZ * VZ)
    var dt_est = CFL * h / (v * Float32(2 * P + 1))
    var num_steps = Int(T_FINAL / dt_est) + 1
    var dt = T_FINAL / Float32(num_steps)

    for _ in range(num_steps):
        solver.step_ssprk3(dt, nvtx)
    solver.ctx.synchronize()

    # Download final state + compute L2.
    var hbuf_q = solver.ctx.enqueue_create_host_buffer[DType.float32](n_owned_dof)
    solver.ctx.enqueue_copy(hbuf_q, solver.d_q.create_sub_buffer[DType.float32](0, n_owned_dof))
    solver.ctx.synchronize()
    var q_ptr = hbuf_q.unsafe_ptr()

    var sum_sq: Float64 = 0.0
    var sum_ic: Float64 = 0.0
    for k in range(n_owned_dof):
        var v_now = q_ptr[k]
        if isnan(v_now) or isinf(v_now):
            raise Error("bench_advection_3d: non-finite output")
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
        print("bench_advection_3d: runs at np=1 only")
        return

    print("bench_advection_3d (3D Gaussian one-period advection)")
    print("  P=", P, "  refinement sweep N=8, 12, 16")

    var err8  = _run(8)
    print("  N=8   rel L2 =", err8)
    var err12 = _run(12)
    print("  N=12  rel L2 =", err12)
    var err16 = _run(16)
    print("  N=16  rel L2 =", err16)

    if err16 > L2_MAX_REL_AT_16:
        raise Error(
            "bench_advection_3d FAILED: rel L2 at N=16 "
            + String(err16) + " exceeds " + String(L2_MAX_REL_AT_16)
        )
    if not (err8 > err12 and err12 > err16):
        raise Error(
            "bench_advection_3d FAILED: rel L2 did not decrease "
            + "monotonically (8: " + String(err8)
            + ", 12: " + String(err12)
            + ", 16: " + String(err16) + ")"
        )

    # Observed rate: log(e8/e12)/log(1.5), log(e12/e16)/log(4/3).
    var rate_812  = log(err8 / err12)  / log(1.5)
    var rate_1216 = log(err12 / err16) / log(4.0 / 3.0)
    print("  observed rates: log_1.5(e8/e12) =", rate_812,
          "  log_4/3(e12/e16) =", rate_1216,
          "  (P+1 =", P + 1, ", floor", RATE_MIN, ")")
    if rate_812 < RATE_MIN and rate_1216 < RATE_MIN:
        raise Error(
            "bench_advection_3d FAILED: observed rates "
            + String(rate_812) + " and " + String(rate_1216)
            + " both below " + String(RATE_MIN)
        )

    print("=== bench_advection_3d PASSED ===")
    mpi.finalize()
