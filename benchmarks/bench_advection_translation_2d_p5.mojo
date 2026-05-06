# ======================================================================
# bench_advection_translation_2d_p5 -- 2D Gaussian advection at P=5
# ======================================================================
#
# P=5 counterpart of bench_advection_translation_2d_p4.  Same Gaussian
# transport, but routed through LocalMesh2D[5] / ReferenceElement2D[5]
# / advection_rk_stage_2d[5].  NP = 21 (P=5 triangle) operators must
# produce the design-rate P+1 = 6 convergence on smooth flow.
#
# This is the highest-order analytic gate in the suite.  Float32
# precision becomes the binding constraint: even at N=12 the error
# sits at ~3e-5, so monotonicity and rate checks both fail at
# higher N.  We use a sweep of N=4, 6, 8 instead, where the
# truncation error is large enough to dominate roundoff.
#
# Pass criteria (P=5, SSPRK3, sweep N=4, 6, 8):
#   * rel L2 at N=8 < 1e-3
#   * rate >= 2.5 between at least one pair
# ======================================================================

from std.math import sqrt, exp, log, isnan, isinf
from std.sys import has_accelerator
from std.gpu.host import DeviceContext, DeviceBuffer
from src import mpi
from src.local_mesh_2d import LocalMesh2D
from src.local_mesh_2d_gpu import LocalMesh2DGpu
from src.local_mesh_2d_gpu_advection import advection_rk_stage_2d
from src.reference_2d import (
    ReferenceElement2D,
    num_tri_nodes_2d,
    num_edge_nodes,
)
from src.reference_2d_gpu import ReferenceElement2DGpu
from src.ssprk3 import ssprk3_stage_plans


comptime P = 5
comptime LX: Float32 = 1.0
comptime LY: Float32 = 1.0
comptime VX: Float32 = 1.0
comptime VY: Float32 = 1.0
comptime T_FINAL: Float32 = 1.0
comptime CFL: Float32 = 0.15
comptime SIGMA: Float32 = 0.12
comptime CX: Float32 = 0.5
comptime CY: Float32 = 0.5

comptime L2_MAX_REL_AT_8: Float64 = 1.0e-3
comptime RATE_MIN: Float64 = 2.5


def _gauss(x: Float32, y: Float32) -> Float32:
    var dx = x - CX
    if dx > LX * Float32(0.5):
        dx -= LX
    if dx < -LX * Float32(0.5):
        dx += LX
    var dy = y - CY
    if dy > LY * Float32(0.5):
        dy -= LY
    if dy < -LY * Float32(0.5):
        dy += LY
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
            var x = Float32(
                mesh_coords.elem_node_xyz[(elem * NP_p + nn) * 2 + 0]
            )
            var y = Float32(
                mesh_coords.elem_node_xyz[(elem * NP_p + nn) * 2 + 1]
            )
            var v = _gauss(x, y)
            host_q.append(v)
            host_ic.append(v)

    var d_q = ctx.enqueue_create_buffer[DType.float32](n_q)
    var d_q1 = ctx.enqueue_create_buffer[DType.float32](n_q)
    var d_q2 = ctx.enqueue_create_buffer[DType.float32](n_q)
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

    var stage_plans = ssprk3_stage_plans(
        d_q.unsafe_ptr(),
        d_q1.unsafe_ptr(),
        d_q2.unsafe_ptr(),
    )
    for _ in range(num_steps):
        for stage in stage_plans:
            advection_rk_stage_2d[P](
                ctx,
                gpu_mesh,
                gpu_re.d_Lift_ref.unsafe_ptr(),
                gpu_re.d_D_ref.unsafe_ptr(),
                stage.q_in,
                stage.q_a,
                stage.q_b,
                stage.q_out,
                d_fstar.unsafe_ptr(),
                VX,
                VY,
                stage.a,
                stage.b,
                stage.c,
                dt,
            )
    ctx.synchronize()
    ctx.enqueue_copy(hbuf_q, d_q)
    ctx.synchronize()

    var sum_sq: Float64 = 0.0
    var sum_ic: Float64 = 0.0
    for k in range(n_q):
        var v = hptr_q[k]
        if isnan(v) or isinf(v):
            raise Error("bench_advection_translation_2d_p5: non-finite output")
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
        print("bench_advection_translation_2d_p5: runs at np=1 only")
        return

    print("bench_advection_translation_2d_p5 (P=5 Gaussian advection)")
    print(
        "  P=", P, "  NP=", num_tri_nodes_2d(P), "  refinement sweep N=4, 6, 8"
    )

    var err4 = _run(4)
    print("  N=4   rel L2 =", err4)
    var err6 = _run(6)
    print("  N=6   rel L2 =", err6)
    var err8 = _run(8)
    print("  N=8   rel L2 =", err8)

    if err8 > L2_MAX_REL_AT_8:
        raise Error(
            "bench_advection_translation_2d_p5 FAILED: rel L2 at N=8 "
            + String(err8)
            + " exceeds "
            + String(L2_MAX_REL_AT_8)
        )

    var rate_46 = log(err4 / err6) / log(6.0 / 4.0)
    var rate_68 = log(err6 / err8) / log(8.0 / 6.0)
    print(
        "  observed rates: log_(3/2)(e4/e6) =",
        rate_46,
        "  log_(4/3)(e6/e8) =",
        rate_68,
        "  (P+1 =",
        P + 1,
        ", floor",
        RATE_MIN,
        ")",
    )
    if rate_46 < RATE_MIN and rate_68 < RATE_MIN:
        raise Error(
            String("bench_advection_translation_2d_p5 FAILED: rates ")
            + String(rate_46)
            + " and "
            + String(rate_68)
            + " both below "
            + String(RATE_MIN)
        )

    print("=== bench_advection_translation_2d_p5 PASSED ===")
    mpi.finalize()
