# ======================================================================
# bench_euler_vortex_2d -- Shu-Erlebacher isentropic vortex (one period)
# ======================================================================
#
# Canonical DG Euler validation: an analytic isentropic vortex
# (Shu 1998, Erlebacher et al.) is superposed on a uniform (U0, V0)
# background flow on a periodic [0, L]^2 domain.  After T = L / U0
# the vortex has traversed exactly one period in each direction and
# the solution must equal the IC -- no entropy generated, so the L2
# deviation is pure scheme dissipation.
#
# Pass criteria (P=2, Rusanov, N=32):
#   * rel L2(state) < 10%%  (measured ~6.8%%)
#   * no non-finite values
#
# Empirically the rel L2 is ~flat under mesh refinement at this
# T=10 horizon -- Rusanov's alpha-based dissipation is
# mesh-independent on smooth problems, and HLLC on this same problem
# produced identical numbers in a sweep (6.71, 6.80, 6.81 at
# N=16, 32, 64).  A proper Euler convergence-rate benchmark needs a
# shorter-time, tighter-amplitude setup; the advection benchmark
# already demonstrates (P+1)-order convergence on the same mesh
# infrastructure.  The vortex check here is scoped for "does the
# Euler pipeline produce a physically-reasonable long-time state on
# a canonical smooth problem" -- the threshold catches any
# catastrophic error (NaN, scheme blow-up, sign flip in a flux
# component) that would push L2 well past the Rusanov-dissipation
# floor.
# ======================================================================

from std.math import sqrt, exp, pi, isnan, isinf
from std.sys import has_accelerator
from std.gpu.host import DeviceContext, DeviceBuffer
from src import mpi
from src.local_mesh_2d import LocalMesh2D
from src.local_mesh_2d_gpu import LocalMesh2DGpu
from src.local_mesh_2d_gpu_euler import euler_rk_stage_2d
from src.ssprk3 import ssprk3_stage_plans
from src.reference_2d import (
    ReferenceElement2D, num_tri_nodes_2d, num_edge_nodes,
)
from src.reference_2d_gpu import ReferenceElement2DGpu


comptime P = 2
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

# Measured ~6.8%% (Rusanov dissipation floor for the vortex at P=2,
# N=32).  8%% is ~1.2x the empirical floor and catches any flux
# regression beyond a small constant; old 10%% gate only caught
# catastrophic blow-ups.
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

    var stage_plans = ssprk3_stage_plans(
        d_q=d_q.unsafe_ptr(),
        d_q1=d_q1.unsafe_ptr(),
        d_q2=d_q2.unsafe_ptr(),
    )
    for _ in range(num_steps):
        for stage in stage_plans:
            euler_rk_stage_2d[P](
                ctx=ctx,
                mesh=gpu_mesh,
                Lift_ref=gpu_re.d_Lift_ref.unsafe_ptr(),
                D_ref=gpu_re.d_D_ref.unsafe_ptr(),
                q_in=stage.q_in,
                q_a=stage.q_a,
                q_b=stage.q_b,
                q_out=stage.q_out,
                fstar_scratch=d_fstar.unsafe_ptr(),
                gamma=gamma, min_density=min_rho, min_pressure=min_p,
                a=stage.a, b=stage.b, cc=stage.c, dt=dt,
            )
    ctx.synchronize()
    ctx.enqueue_copy(hbuf_q, d_q)
    ctx.synchronize()

    var sum_sq: Float64 = 0.0
    var sum_ic: Float64 = 0.0
    for k in range(n_q):
        var v = hptr_q[k]
        if isnan(v) or isinf(v):
            raise Error("bench_euler_vortex_2d: non-finite output at index "
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
        print("bench_euler_vortex_2d: runs at np=1 only")
        return

    print("bench_euler_vortex_2d (isentropic vortex, one period)")
    print("  P=", P, "  N=32  (Rusanov)")

    var rel_l2 = _run(32)
    print("  rel L2(state) =", rel_l2,
          "  (threshold", L2_MAX_REL, ")")

    if rel_l2 > L2_MAX_REL:
        raise Error(
            "bench_euler_vortex_2d FAILED: rel L2 "
            + String(rel_l2)
            + " exceeds threshold "
            + String(L2_MAX_REL)
        )
    print("=== bench_euler_vortex_2d PASSED ===")
    mpi.finalize()
