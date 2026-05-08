# ======================================================================
# bench_euler_smooth_wave_2d_p4 -- 2D entropy-wave Euler at P=4
# ======================================================================
#
# P=4 counterpart of bench_euler_smooth_wave_2d_p3.  Same entropy-wave
# IC (rho = rho0 + A sin(2 pi x) sin(2 pi y), uniform p, uniform u, v)
# on a periodic [0, 1]^2 mesh, but routed through LocalMesh2D[4] /
# euler_rk_stage_hllc_2d[4] with NP = 15 nodes per triangle.
#
# Mirrors the P=4 advection coverage (`bench_advection_translation_2d_p4`,
# NP=15, single-component) on the multi-component HLLC Euler path.
# Closes a P-parity gap: the Euler 2D kernels are comptime-templated
# on P, so the same vol+lift+rk and HLLC face-flux kernels that pass
# at P=2 / P=3 should also produce correct output at P=4 -- but
# Vandermonde inversion at P=4 hasn't been exercised on the multi-
# component path before this gate.
#
# Pass criteria (P=4, HLLC, periodic, sweep N=4, 6, 8):
#   * rel L2 at every N < 1.5e-4  (Float32-floor regime)
#   * no NaN / Inf
#
# At P=4 with NP=15 the entropy wave sits at the Float32 round-off
# floor across the entire sweep, so monotone refinement and rate
# checks don't apply -- this gate is an absolute-L2 regression
# sentinel for the higher-P Euler kernels.
# ======================================================================

from std.math import sqrt, sin, pi, isnan, isinf
from std.sys import has_accelerator
from std.gpu.host import DeviceContext, DeviceBuffer
from src import mpi
from src.local_mesh_2d import LocalMesh2D
from src.local_mesh_2d_gpu import LocalMesh2DGpu
from src.local_mesh_2d_gpu_euler import euler_rk_stage_hllc_2d
from src.ssprk3 import ssprk3_stage_plans
from src.reference_2d import (
    ReferenceElement2D,
    num_tri_nodes_2d,
    num_edge_nodes,
)
from src.reference_2d_gpu import ReferenceElement2DGpu


comptime P = 4
comptime LX = 1.0
comptime LY = 1.0
comptime GAMMA = 1.4
comptime RHO0 = 1.0
comptime U0 = 1.0
comptime V0 = 1.0
comptime P0 = 1.0
comptime AMPLITUDE = 0.1
comptime T_FINAL = 1.0
comptime CFL = 0.08

# Measured ~5e-5 across N = 4, 6, 8 (Float32 floor; same plateau as
# at P=3 since the scheme is more accurate than Float32 representation
# at this resolution).  1.5e-4 is ~3x the empirical floor, catches any
# HLLC P=4 regression that meaningfully degrades smooth-flow accuracy.
comptime L2_MAX_REL: Float64 = 1.5e-4


def _run(N: Int) raises -> Float64:
    comptime NP_p = num_tri_nodes_2d(P)
    comptime NFP_e = num_edge_nodes(P)
    comptime NC = 4
    var ctx = DeviceContext()

    var host_mesh = LocalMesh2D[P](N, N, LX, LY)
    var host_re = ReferenceElement2D[P]()
    var mesh_coords = LocalMesh2D[P](N, N, LX, LY)
    var gpu_mesh = LocalMesh2DGpu[P](ctx, host_mesh^)
    var gpu_re = ReferenceElement2DGpu[P](ctx, host_re)

    var two_pi = 2.0 * pi
    var n_q = gpu_mesh.num_elements * NP_p * NC
    var host_q = List[Float32]()
    var host_ic = List[Float32]()
    for elem in range(gpu_mesh.num_elements):
        for nn in range(NP_p):
            var x = mesh_coords.elem_node_xyz[(elem * NP_p + nn) * 2 + 0]
            var y = mesh_coords.elem_node_xyz[(elem * NP_p + nn) * 2 + 1]
            var rho = RHO0 + AMPLITUDE * sin(two_pi * x) * sin(two_pi * y)
            var E = P0 / (GAMMA - 1.0) + 0.5 * rho * (U0 * U0 + V0 * V0)
            host_q.append(Float32(rho))
            host_ic.append(Float32(rho))
            host_q.append(Float32(rho * U0))
            host_ic.append(Float32(rho * U0))
            host_q.append(Float32(rho * V0))
            host_ic.append(Float32(rho * V0))
            host_q.append(Float32(E))
            host_ic.append(Float32(E))

    var d_q = ctx.enqueue_create_buffer[DType.float32](n_q)
    var d_q1 = ctx.enqueue_create_buffer[DType.float32](n_q)
    var d_q2 = ctx.enqueue_create_buffer[DType.float32](n_q)
    var d_fstar = ctx.enqueue_create_buffer[DType.float32](gpu_mesh.num_faces * NFP_e * NC)
    var hbuf_q = ctx.enqueue_create_host_buffer[DType.float32](n_q)
    var hptr_q = hbuf_q.unsafe_ptr()
    for k in range(n_q):
        hptr_q[k] = host_q[k]
    ctx.enqueue_copy(d_q, hbuf_q)
    ctx.synchronize()

    var h = LX / Float64(N)
    var c0 = sqrt(GAMMA * P0 / RHO0)
    var wave_max = sqrt(U0 * U0 + V0 * V0) + c0
    var dt_est = CFL * h / (wave_max * Float64(2 * P + 1))
    var num_steps = Int(T_FINAL / dt_est) + 1
    var dt = Float32(T_FINAL / Float64(num_steps))

    var gamma = Float32(GAMMA)
    var min_rho = Float32(1.0e-6)
    var min_p = Float32(1.0e-6)

    var stage_plans = ssprk3_stage_plans(
        d_q.unsafe_ptr(),
        d_q1.unsafe_ptr(),
        d_q2.unsafe_ptr(),
    )
    for _ in range(num_steps):
        for stage in stage_plans:
            euler_rk_stage_hllc_2d[P](
                ctx,
                gpu_mesh,
                gpu_re.d_Lift_ref.unsafe_ptr(),
                gpu_re.d_D_ref.unsafe_ptr(),
                stage.q_in,
                stage.q_a,
                stage.q_b,
                stage.q_out,
                d_fstar.unsafe_ptr(),
                gamma,
                min_rho,
                min_p,
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
            raise Error("bench_euler_smooth_wave_2d_p4: non-finite output")
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
        print("bench_euler_smooth_wave_2d_p4: runs at np=1 only")
        return

    print("bench_euler_smooth_wave_2d_p4 (P=4 entropy wave, HLLC)")
    print("  P=", P, "  NP=", num_tri_nodes_2d(P), "  refinement sweep N=4, 6, 8")

    var err4 = _run(4)
    print("  N=4   rel L2 =", err4)
    var err6 = _run(6)
    print("  N=6   rel L2 =", err6)
    var err8 = _run(8)
    print("  N=8   rel L2 =", err8)

    if err4 > L2_MAX_REL:
        raise Error(
            "bench_euler_smooth_wave_2d_p4 FAILED: rel L2 at N=4 " + String(err4) + " exceeds " + String(L2_MAX_REL)
        )
    if err6 > L2_MAX_REL:
        raise Error(
            "bench_euler_smooth_wave_2d_p4 FAILED: rel L2 at N=6 " + String(err6) + " exceeds " + String(L2_MAX_REL)
        )
    if err8 > L2_MAX_REL:
        raise Error(
            "bench_euler_smooth_wave_2d_p4 FAILED: rel L2 at N=8 " + String(err8) + " exceeds " + String(L2_MAX_REL)
        )

    print("=== bench_euler_smooth_wave_2d_p4 PASSED ===")
    mpi.finalize()
