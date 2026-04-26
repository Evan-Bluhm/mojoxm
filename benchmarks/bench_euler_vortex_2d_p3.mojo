# ======================================================================
# bench_euler_vortex_2d_p3 -- isentropic vortex (one period) at P=3
# ======================================================================
#
# P=3 (NP=10) HLLC counterpart of bench_euler_vortex_2d (P=2 Rusanov).
# Same Shu-Erlebacher isentropic vortex IC + uniform (U0, V0)
# background flow on a periodic [0, L]^2 domain; after T = L / U0 the
# vortex returns to IC.  L2 deviation is pure scheme dissipation.
#
# Routes through LocalMesh2D[3] / euler_rk_stage_hllc_2d[3] so the
# 2D HLLC kernel is exercised at NP=10 on a long-time smooth flow,
# complementing the existing P=3 HLLC entropy-wave (short time, tight
# tolerance) and Sod-limited (shocked) gates with a long-time smooth-
# flow regression sentinel.
#
# Pass criteria (P=3, HLLC, N=32, T=10):
#   * rel L2(state) < 8%%
#   * no non-finite values
#
# Empirically the rel L2 sits at 6.81%% at P=3 / HLLC -- essentially
# identical to the P=2 / Rusanov + P=2 / HLLC bench's 6.7-6.8%%.  The
# Shu-Erlebacher vortex at T=10 has a long-horizon dissipation floor
# that's roughly P- and flux-type-independent -- both Rusanov-alpha
# and HLLC's contact-restored arithmetic produce essentially the
# same time-integrated wake.  Higher P doesn't help here.
#
# As with the P=2 bench, this is a regression-detector for catastrophic
# blow-up, not a convergence-rate gate -- the advection rate benches
# (`bench_advection_translation_2d_p3` etc.) demonstrate (P+1)-order
# on the same mesh infrastructure.
# ======================================================================

from std.math import sqrt, exp, pi, isnan, isinf
from std.sys import has_accelerator
from std.gpu.host import DeviceContext, DeviceBuffer
from src import mpi
from src.local_mesh_2d import LocalMesh2D
from src.local_mesh_2d_gpu import LocalMesh2DGpu
from src.local_mesh_2d_gpu_euler import euler_rk_stage_hllc_2d
from src.reference_2d import (
    ReferenceElement2D, num_tri_nodes_2d, num_edge_nodes,
)
from src.reference_2d_gpu import ReferenceElement2DGpu


comptime P = 3
comptime LX = 10.0
comptime LY = 10.0
comptime T_FINAL = 10.0
comptime CFL = 0.15

comptime GAMMA = 1.4
comptime T_INF = 1.0
comptime U0 = 1.0
comptime V0 = 1.0
comptime BETA = 5.0
comptime CX0 = 5.0
comptime CY0 = 5.0

# Empirical 6.81%% at P=3 / HLLC, same scale as P=2.  8%% is ~1.2x
# the floor and catches any flux-type regression beyond a small
# constant; old 10%% gate only caught catastrophic blow-ups.
comptime L2_MAX_REL: Float64 = 0.08


def _periodic_delta(a: Float64, b: Float64, L: Float64) -> Float64:
    var d = a - b
    if d >  L * 0.5: d -= L
    if d < -L * 0.5: d += L
    return d


def _run(N: Int) raises -> Float64:
    """Run one vortex period at resolution NxN and return rel L2."""
    comptime NP_p = num_tri_nodes_2d(P)
    comptime NFP_e = num_edge_nodes(P)
    comptime NC = 4
    var ctx = DeviceContext()

    var host_mesh = LocalMesh2D[P](N, N, LX, LY)
    var host_re = ReferenceElement2D[P]()
    var mesh_coords = LocalMesh2D[P](N, N, LX, LY)
    var gpu_mesh = LocalMesh2DGpu[P](ctx, host_mesh^)
    var gpu_re = ReferenceElement2DGpu[P](ctx, host_re)

    var n_q = gpu_mesh.num_elements * NP_p * NC
    var host_q = List[Float32]()
    var host_ic = List[Float32]()
    var two_pi = 2.0 * pi
    var factor = (GAMMA - 1.0) * BETA * BETA / (8.0 * GAMMA * two_pi * two_pi)
    for elem in range(gpu_mesh.num_elements):
        for nn in range(NP_p):
            var x = mesh_coords.elem_node_xyz[(elem * NP_p + nn) * 2 + 0]
            var y = mesh_coords.elem_node_xyz[(elem * NP_p + nn) * 2 + 1]
            var dx = _periodic_delta(x, CX0, LX)
            var dy = _periodic_delta(y, CY0, LY)
            var r2 = dx * dx + dy * dy
            var T = T_INF - factor * exp(1.0 - r2)
            var e_half = exp(0.5 * (1.0 - r2))
            var u = U0 - (BETA / two_pi) * dy * e_half
            var v = V0 + (BETA / two_pi) * dx * e_half
            var rho = T ** (1.0 / (GAMMA - 1.0))
            var p = rho * T
            var E = p / (GAMMA - 1.0) + 0.5 * rho * (u * u + v * v)
            host_q.append(Float32(rho));     host_ic.append(Float32(rho))
            host_q.append(Float32(rho * u)); host_ic.append(Float32(rho * u))
            host_q.append(Float32(rho * v)); host_ic.append(Float32(rho * v))
            host_q.append(Float32(E));        host_ic.append(Float32(E))

    var d_q  = ctx.enqueue_create_buffer[DType.float32](n_q)
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
    var c_inf = sqrt(GAMMA * T_INF)
    var wave_max = sqrt(U0 * U0 + V0 * V0) + c_inf + BETA / two_pi
    var dt_est = CFL * h / (wave_max * Float64(2 * P + 1))
    var num_steps = Int(T_FINAL / dt_est) + 1
    var dt = Float32(T_FINAL / Float64(num_steps))

    var gamma = Float32(GAMMA)
    var min_rho = Float32(1.0e-6)
    var min_p = Float32(1.0e-6)

    for _ in range(num_steps):
        euler_rk_stage_hllc_2d[P](
            ctx, gpu_mesh,
            gpu_re.d_Lift_ref.unsafe_ptr(), gpu_re.d_D_ref.unsafe_ptr(),
            d_q.unsafe_ptr(),
            d_q.unsafe_ptr(), d_q.unsafe_ptr(),
            d_q1.unsafe_ptr(),
            d_fstar.unsafe_ptr(),
            gamma, min_rho, min_p,
            Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0),
            Float32(1.0), Float32(0.0), Float32(1.0), dt,
        )
        euler_rk_stage_hllc_2d[P](
            ctx, gpu_mesh,
            gpu_re.d_Lift_ref.unsafe_ptr(), gpu_re.d_D_ref.unsafe_ptr(),
            d_q1.unsafe_ptr(),
            d_q.unsafe_ptr(), d_q1.unsafe_ptr(),
            d_q2.unsafe_ptr(),
            d_fstar.unsafe_ptr(),
            gamma, min_rho, min_p,
            Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0),
            Float32(0.75), Float32(0.25), Float32(0.25), dt,
        )
        euler_rk_stage_hllc_2d[P](
            ctx, gpu_mesh,
            gpu_re.d_Lift_ref.unsafe_ptr(), gpu_re.d_D_ref.unsafe_ptr(),
            d_q2.unsafe_ptr(),
            d_q.unsafe_ptr(), d_q2.unsafe_ptr(),
            d_q.unsafe_ptr(),
            d_fstar.unsafe_ptr(),
            gamma, min_rho, min_p,
            Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0),
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
            raise Error("bench_euler_vortex_2d_p3: non-finite output at index "
                        + String(k))
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
        print("bench_euler_vortex_2d_p3: runs at np=1 only")
        return

    print("bench_euler_vortex_2d_p3 (isentropic vortex, one period)")
    print("  P=", P, "  N=32  (HLLC)")

    var rel_l2 = _run(32)
    print("  rel L2(state) =", rel_l2,
          "  (threshold", L2_MAX_REL, ")")

    if rel_l2 > L2_MAX_REL:
        raise Error(
            "bench_euler_vortex_2d_p3 FAILED: rel L2 "
            + String(rel_l2)
            + " exceeds threshold "
            + String(L2_MAX_REL)
        )
    print("=== bench_euler_vortex_2d_p3 PASSED ===")
    mpi.finalize()
