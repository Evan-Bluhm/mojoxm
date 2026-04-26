# ======================================================================
# bench_advection_translation_2d -- Gaussian one-period advection
# ======================================================================
#
# Analytic problem: a Gaussian bump in the 2D scalar advection equation
# under uniform velocity v = (1, 1) on a periodic [0, 1]^2 domain.
# After T = LX / vx = 1 the bump has translated by exactly one period
# so the exact solution equals the IC -- any L2 difference is pure
# scheme dissipation.
#
# Pass criteria (P=2, SSPRK3, CFL=0.3):
#   * Rel L2 at N=32 < 1.0%
#   * Rel L2 strictly decreases when refining 16 -> 32 -> 64, and the
#     observed convergence rate log2(e_N / e_{2N}) >= 2.0 between at
#     least one pair of consecutive resolutions (P+1 = 3 asymptotically,
#     but loose floor at 2.0 to allow pre-asymptotic / Rusanov-upwind
#     dissipation wiggle).
#   * No NaN / Inf.
#
# Convergence at the expected rate is the real physical gate: a bug
# that still produces a finite L2 but breaks the rate signals a scheme
# problem that single-resolution L2 alone would miss.
# ======================================================================

from std.math import sqrt, exp, log, isnan, isinf
from std.sys import has_accelerator
from std.gpu.host import DeviceContext, DeviceBuffer
from std.time import perf_counter_ns
from src import mpi
from src.local_mesh_2d import LocalMesh2D
from src.local_mesh_2d_gpu import LocalMesh2DGpu
from src.local_mesh_2d_gpu_advection import advection_rk_stage_2d
from src.reference_2d import (
    ReferenceElement2D, num_tri_nodes_2d, num_edge_nodes,
)
from src.reference_2d_gpu import ReferenceElement2DGpu


comptime P = 2
comptime LX: Float32 = 1.0
comptime LY: Float32 = 1.0
comptime VX: Float32 = 1.0
comptime VY: Float32 = 1.0
comptime T_FINAL: Float32 = 1.0
comptime CFL: Float32 = 0.3
comptime SIGMA: Float32 = 0.12
comptime CX: Float32 = 0.5
comptime CY: Float32 = 0.5

# Measured ~7.2e-4 at N=32 on current code; 1.5e-3 is ~2x margin.
# (Old gate at 1e-2 was 14x looser -- only caught catastrophic regressions.)
comptime L2_MAX_REL_AT_32: Float64 = 1.5e-3
comptime RATE_MIN:         Float64 = 2.0


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
    """Run one period at resolution NxN and return the relative L2 error."""
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
            d_fstar.unsafe_ptr(),
            VX, VY, Float32(0.0),
            Float32(1.0), Float32(0.0), Float32(1.0), dt,
        )
        advection_rk_stage_2d[P](
            ctx, gpu_mesh,
            gpu_re.d_Lift_ref.unsafe_ptr(), gpu_re.d_D_ref.unsafe_ptr(),
            d_q1.unsafe_ptr(),
            d_q.unsafe_ptr(), d_q1.unsafe_ptr(),
            d_q2.unsafe_ptr(),
            d_fstar.unsafe_ptr(),
            VX, VY, Float32(0.0),
            Float32(0.75), Float32(0.25), Float32(0.25), dt,
        )
        advection_rk_stage_2d[P](
            ctx, gpu_mesh,
            gpu_re.d_Lift_ref.unsafe_ptr(), gpu_re.d_D_ref.unsafe_ptr(),
            d_q2.unsafe_ptr(),
            d_q.unsafe_ptr(), d_q2.unsafe_ptr(),
            d_q.unsafe_ptr(),
            d_fstar.unsafe_ptr(),
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
            raise Error("bench_advection_translation_2d: non-finite output")
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
        print("bench_advection_translation_2d: runs at np=1 only")
        return

    print("bench_advection_translation_2d (Gaussian one-period advection)")
    print("  P=", P, "  refinement sweep N=16, 32, 64")

    var err16 = _run(16)
    print("  N=16  rel L2 =", err16)
    var err32 = _run(32)
    print("  N=32  rel L2 =", err32)
    var err64 = _run(64)
    print("  N=64  rel L2 =", err64)

    # Single-resolution L2 at the target N=32.
    if err32 > L2_MAX_REL_AT_32:
        raise Error(
            "bench_advection_translation_2d FAILED: rel L2 at N=32 "
            + String(err32) + " exceeds " + String(L2_MAX_REL_AT_32)
        )

    # Strict monotone decrease under refinement.
    if not (err16 > err32 and err32 > err64):
        raise Error(
            "bench_advection_translation_2d FAILED: L2 did not decrease "
            + "monotonically under refinement (16: " + String(err16)
            + ", 32: " + String(err32) + ", 64: " + String(err64) + ")"
        )

    # Observed rate 16->32 and 32->64.  Assert at least one pair
    # makes RATE_MIN -- that's enough to catch a bug that destroys the
    # convergence order (e.g. an incorrect Jacobian or a flux
    # dissipation that dominates the truncation error).
    var rate_1632 = log(err16 / err32) / log(2.0)
    var rate_3264 = log(err32 / err64) / log(2.0)
    print("  observed rates: log2(e16/e32) =", rate_1632,
          "  log2(e32/e64) =", rate_3264,
          "  (P+1 =", P + 1, ", floor", RATE_MIN, ")")
    if rate_1632 < RATE_MIN and rate_3264 < RATE_MIN:
        raise Error(
            "bench_advection_translation_2d FAILED: observed rates "
            + String(rate_1632) + " and " + String(rate_3264)
            + " both below " + String(RATE_MIN)
        )

    print("=== bench_advection_translation_2d PASSED ===")
    mpi.finalize()
