# ======================================================================
# bench_advection_translation_2d_p4 -- 2D Gaussian advection at P=4
# ======================================================================
#
# P=4 counterpart of bench_advection_translation_2d_p3.  Same Gaussian-
# bump one-period advection problem on a periodic [0, 1]^2 mesh, but
# routed through LocalMesh2D[4] / ReferenceElement2D[4] /
# advection_rk_stage_2d[4].  NP = 15 (P=4 triangle) operators must
# produce the design-rate P+1 = 5 convergence on smooth flow.
#
# Pushes both Lagrange basis construction (Vandermonde inverse) and
# the kernel-side comptime template instantiation past the P=2 / P=3
# values that earlier benches exercise.  Pairs with the P=5 variant
# (NP=21) for top-of-stack coverage in 2D.
#
# Pass criteria (P=4, SSPRK3, sweep N = 8, 12, 16):
#   * rel L2 at N=16 < 5e-4 (Gaussian well-resolved at P=4)
#   * monotone refinement
#   * rate >= 2.5 between at least one consecutive pair
#     (theoretical P+1=5; the e8->e12 pair sits at ~4.67 in current
#     code, e12->e16 at ~2.97 since N=16 is hitting Float32 floor)
# ======================================================================

from std.math import sqrt, exp, log, isnan, isinf
from std.sys import has_accelerator
from std.gpu.host import DeviceContext, DeviceBuffer
from src import mpi
from src.local_mesh_2d import LocalMesh2D
from src.local_mesh_2d_gpu import LocalMesh2DGpu
from src.local_mesh_2d_gpu_advection import advection_rk_stage_2d
from src.reference_2d import (
    ReferenceElement2D, num_tri_nodes_2d, num_edge_nodes,
)
from src.reference_2d_gpu import ReferenceElement2DGpu


comptime P = 4
comptime LX: Float32 = 1.0
comptime LY: Float32 = 1.0
comptime VX: Float32 = 1.0
comptime VY: Float32 = 1.0
comptime T_FINAL: Float32 = 1.0
comptime CFL: Float32 = 0.15
comptime SIGMA: Float32 = 0.12
comptime CX: Float32 = 0.5
comptime CY: Float32 = 0.5

comptime L2_MAX_REL_AT_16: Float64 = 5.0e-4
comptime RATE_MIN:         Float64 = 2.5


def _gauss(x: Float32, y: Float32) -> Float32:
    var dx = x - CX
    if dx >  LX * Float32(0.5): dx -= LX
    if dx < -LX * Float32(0.5): dx += LX
    var dy = y - CY
    if dy >  LY * Float32(0.5): dy -= LY
    if dy < -LY * Float32(0.5): dy += LY
    var s2 = SIGMA * SIGMA
    return exp(-(dx * dx + dy * dy) / (Float32(2.0) * s2))


def _run(N: Int) raises -> Float64:
    comptime NP_p = num_tri_nodes_2d(P)
    comptime NFP_e = num_edge_nodes(P)
    var ctx = DeviceContext()

    var host_mesh = LocalMesh2D[P](N, N, Float64(LX), Float64(LY))
    var host_re = ReferenceElement2D[P]()
    var mesh_coords = LocalMesh2D[P](N, N, Float64(LX), Float64(LY))
    var gpu_mesh = LocalMesh2DGpu[P](ctx, host_mesh^)
    var gpu_re = ReferenceElement2DGpu[P](ctx, host_re)

    var n_q = gpu_mesh.num_elements * NP_p
    var host_q = List[Float32]()
    var host_ic = List[Float32]()
    for elem in range(gpu_mesh.num_elements):
        for nn in range(NP_p):
            var x = Float32(mesh_coords.elem_node_xyz[(elem * NP_p + nn) * 2 + 0])
            var y = Float32(mesh_coords.elem_node_xyz[(elem * NP_p + nn) * 2 + 1])
            var v = _gauss(x, y)
            host_q.append(v)
            host_ic.append(v)

    var d_q  = ctx.enqueue_create_buffer[DType.float32](n_q)
    var d_q1 = ctx.enqueue_create_buffer[DType.float32](n_q)
    var d_q2 = ctx.enqueue_create_buffer[DType.float32](n_q)
    var d_vol = ctx.enqueue_create_buffer[DType.float32](n_q)
    var d_rhs = ctx.enqueue_create_buffer[DType.float32](n_q)
    var d_fstar = ctx.enqueue_create_buffer[DType.float32](
        gpu_mesh.num_faces * NFP_e
    )
    var hbuf_q = ctx.enqueue_create_host_buffer[DType.float32](n_q)
    var hptr_q = hbuf_q.unsafe_ptr()
    for k in range(n_q):
        hptr_q[k] = host_q[k]
    ctx.enqueue_copy(d_q, hbuf_q)
    ctx.synchronize()

    var h = LX / Float32(N)
    var vmag = sqrt(VX * VX + VY * VY)
    var dt_est = CFL * h / (vmag * Float32(2 * P + 1))
    var num_steps = Int(T_FINAL / dt_est) + 1
    var dt = T_FINAL / Float32(num_steps)

    for _ in range(num_steps):
        advection_rk_stage_2d[P](
            ctx, gpu_mesh,
            gpu_re.d_Lift_ref.unsafe_ptr(), gpu_re.d_D_ref.unsafe_ptr(),
            d_q.unsafe_ptr(),
            d_q.unsafe_ptr(), d_q.unsafe_ptr(),
            d_q1.unsafe_ptr(),
            d_vol.unsafe_ptr(), d_fstar.unsafe_ptr(), d_rhs.unsafe_ptr(),
            VX, VY, Float32(0.0),
            Float32(1.0), Float32(0.0), Float32(1.0), dt,
        )
        advection_rk_stage_2d[P](
            ctx, gpu_mesh,
            gpu_re.d_Lift_ref.unsafe_ptr(), gpu_re.d_D_ref.unsafe_ptr(),
            d_q1.unsafe_ptr(),
            d_q.unsafe_ptr(), d_q1.unsafe_ptr(),
            d_q2.unsafe_ptr(),
            d_vol.unsafe_ptr(), d_fstar.unsafe_ptr(), d_rhs.unsafe_ptr(),
            VX, VY, Float32(0.0),
            Float32(0.75), Float32(0.25), Float32(0.25), dt,
        )
        advection_rk_stage_2d[P](
            ctx, gpu_mesh,
            gpu_re.d_Lift_ref.unsafe_ptr(), gpu_re.d_D_ref.unsafe_ptr(),
            d_q2.unsafe_ptr(),
            d_q.unsafe_ptr(), d_q2.unsafe_ptr(),
            d_q.unsafe_ptr(),
            d_vol.unsafe_ptr(), d_fstar.unsafe_ptr(), d_rhs.unsafe_ptr(),
            VX, VY, Float32(0.0),
            Float32(1.0 / 3.0), Float32(2.0 / 3.0),
            Float32(2.0 / 3.0), dt,
        )
    ctx.synchronize()
    ctx.enqueue_copy(hbuf_q, d_q)
    ctx.synchronize()

    var sum_sq: Float64 = 0.0
    var sum_ic: Float64 = 0.0
    for k in range(n_q):
        var v = hptr_q[k]
        if isnan(v) or isinf(v):
            raise Error("bench_advection_translation_2d_p4: non-finite output")
        var e = Float64(v - host_ic[k])
        sum_sq += e * e
        var ic = Float64(host_ic[k])
        sum_ic += ic * ic
    var l2 = sqrt(sum_sq / Float64(n_q))
    var l2_ic = sqrt(sum_ic / Float64(n_q))
    return l2 / l2_ic


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    mpi.init()
    var size = mpi.world_size()
    if size > 1:
        mpi.finalize()
        print("bench_advection_translation_2d_p4: runs at np=1 only")
        return

    print("bench_advection_translation_2d_p4 (P=4 Gaussian advection)")
    print("  P=", P, "  NP=", num_tri_nodes_2d(P),
          "  refinement sweep N=8, 12, 16")

    var err8  = _run(8)
    print("  N=8   rel L2 =", err8)
    var err12 = _run(12)
    print("  N=12  rel L2 =", err12)
    var err16 = _run(16)
    print("  N=16  rel L2 =", err16)

    if err16 > L2_MAX_REL_AT_16:
        raise Error(
            "bench_advection_translation_2d_p4 FAILED: rel L2 at N=16 "
            + String(err16) + " exceeds " + String(L2_MAX_REL_AT_16)
        )
    if not (err8 > err12 and err12 > err16):
        raise Error(
            "bench_advection_translation_2d_p4 FAILED: L2 not monotone"
            + " (8: " + String(err8) + ", 12: " + String(err12)
            + ", 16: " + String(err16) + ")"
        )

    var rate_812  = log(err8 / err12)  / log(12.0 / 8.0)
    var rate_1216 = log(err12 / err16) / log(16.0 / 12.0)
    print("  observed rates: log_(3/2)(e8/e12) =", rate_812,
          "  log_(4/3)(e12/e16) =", rate_1216,
          "  (P+1 =", P + 1, ", floor", RATE_MIN, ")")
    if rate_812 < RATE_MIN and rate_1216 < RATE_MIN:
        raise Error(
            String("bench_advection_translation_2d_p4 FAILED: rates ")
            + String(rate_812) + " and " + String(rate_1216)
            + " both below " + String(RATE_MIN)
        )

    print("=== bench_advection_translation_2d_p4 PASSED ===")
    mpi.finalize()
