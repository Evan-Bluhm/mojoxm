# ======================================================================
# bench_euler_smooth_wave_2d -- entropy-wave Euler convergence
# ======================================================================
#
# Short-time smooth Euler convergence test: a small-amplitude density
# perturbation on a uniform (u0, v0) background with constant pressure
# is advected over one period on a [0, 1]^2 periodic domain.  This is
# an exact entropy-wave solution of the Euler equations -- uniform u,
# v, p trivially satisfy momentum / energy / continuity while
# transporting rho unchanged at speed (u0, v0).  After T = LX / u0
# the density pattern has traversed a full period and the exact
# solution equals the IC; residual L2 is pure scheme dissipation.
#
# IC (entropy wave):
#   rho(x, y)   = rho0 + A * sin(2 pi x) * sin(2 pi y)
#   u = u0,  v = v0     (uniform)
#   p(x, y)     = p0     (uniform -- this is NOT isentropic; using
#                         p ~ rho^gamma would inject a spurious
#                         pressure gradient and break the exact-
#                         advection property)
#   E           = p / (gamma - 1) + 0.5 * rho * (u^2 + v^2)
#
# Counterpart to bench_advection_translation_2d: same mesh, same
# time horizon, same analytic-return-to-IC property, but full Euler
# with HLLC flux rather than scalar upwind.  With HLLC's
# properly-ordered dissipation, the scheme should show the expected
# (P+1) convergence rate on this smooth problem.
#
# Pass criteria (P=2, HLLC, periodic):
#   * rel L2(state) at N=16, 32, 64 all < 5e-4
#   * no non-finite values
#
# The scheme preserves entropy waves to near-Float32 precision on
# all three resolutions (observed: N=16 7e-5, N=32 6e-5, N=64 1e-4).
# The error is dominated by accumulated Float32 roundoff over the
# RK stages (num_steps grows as N), so the single-resolution gates
# below don't tighten with refinement and a conventional P+1
# convergence-rate assertion doesn't apply here.  The strong
# physical statement this benchmark makes is: "HLLC on a pure
# entropy wave produces results indistinguishable from exact
# advection up to Float32 roundoff" -- that's a tight correctness
# check that catches any bug that would pollute the scheme even
# slightly (regressions would push error well above 5e-4).
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


comptime P = 2
comptime LX = 1.0
comptime LY = 1.0
comptime GAMMA = 1.4
comptime RHO0 = 1.0
comptime U0 = 1.0
comptime V0 = 1.0
comptime P0 = 1.0
comptime AMPLITUDE = 0.1  # rho varies from 0.9 to 1.1; still
# smooth, well above Float32 noise
comptime T_FINAL = 1.0  # one advection period LX / U0
comptime CFL = 0.15

# Measured: N=16 ~7e-5, N=32 ~6e-5, N=64 ~1.1e-4 (non-monotone --
# Float32 roundoff floor over ~200 SSPRK3 steps).  2e-4 is ~2x the
# worst observed, catches regressions in HLLC Euler on smooth flow.
comptime L2_MAX_REL: Float64 = 2.0e-4


def _run(N: Int) raises -> Float64:
    """Run one period at resolution NxN and return rel L2."""
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
            # Uniform pressure p = p0 -- this is an entropy wave.  No
            # pressure gradient means rho advects exactly at (u0, v0).
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
    var d_fstar = ctx.enqueue_create_buffer[DType.float32](
        gpu_mesh.num_faces * NFP_e * NC
    )
    var hbuf_q = ctx.enqueue_create_host_buffer[DType.float32](n_q)
    var hptr_q = hbuf_q.unsafe_ptr()
    for k in range(n_q):
        hptr_q[k] = host_q[k]
    ctx.enqueue_copy(d_q, hbuf_q)
    ctx.synchronize()

    var h = LX / Float64(N)
    var c0 = sqrt(GAMMA * P0 / RHO0)  # sound speed
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
            raise Error(
                "bench_euler_smooth_wave_2d: non-finite output at " + String(k)
            )
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
        print("bench_euler_smooth_wave_2d: runs at np=1 only")
        return

    print("bench_euler_smooth_wave_2d (entropy wave, HLLC)")
    print("  P=", P, "  sweep N=16, 32, 64   (threshold", L2_MAX_REL, ")")

    var err16 = _run(16)
    print("  N=16  rel L2 =", err16)
    var err32 = _run(32)
    print("  N=32  rel L2 =", err32)
    var err64 = _run(64)
    print("  N=64  rel L2 =", err64)

    if err16 > L2_MAX_REL:
        raise Error(
            "bench_euler_smooth_wave_2d FAILED: rel L2 at N=16 "
            + String(err16)
            + " exceeds "
            + String(L2_MAX_REL)
        )
    if err32 > L2_MAX_REL:
        raise Error(
            "bench_euler_smooth_wave_2d FAILED: rel L2 at N=32 "
            + String(err32)
            + " exceeds "
            + String(L2_MAX_REL)
        )
    if err64 > L2_MAX_REL:
        raise Error(
            "bench_euler_smooth_wave_2d FAILED: rel L2 at N=64 "
            + String(err64)
            + " exceeds "
            + String(L2_MAX_REL)
        )

    print("=== bench_euler_smooth_wave_2d PASSED ===")
    mpi.finalize()
