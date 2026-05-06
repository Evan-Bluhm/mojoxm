# ======================================================================
# bench_maxwell_plane_wave_2d -- 2D TM plane wave on a periodic box
# ======================================================================
#
# Companion to bench_maxwell_cavity_2d (PEC cavity standing wave).
# Periodic [0, 1]^2 box, TM-mode plane wave traveling in +x at speed c:
#
#   Ez(x, y, t) = cos(2 pi (x - c t))
#   By(x, y, t) = -(1 / c) cos(2 pi (x - c t))
#   Bx = Ex = Ey = Bz = 0  identically
#
# This satisfies vacuum Maxwell's equations with k = (2 pi, 0) and
# omega = c |k| = 2 pi c.  Period T = 1/c.  After T = 1/c (with c=1)
# the wave has translated by exactly one period and the exact
# solution returns to the IC -- residual L2 is pure scheme
# dissipation.
#
# Why this and not just the cavity:
#   * Cavity test uses BC_WALL (PEC) and a *standing* wave.  All modes
#     are stationary; the gate exercises wall reflection and time
#     stepping but not actual wave propagation across the mesh.
#   * Plane wave uses BC_INTERIOR (periodic) and an actual *traveling*
#     wave.  Bugs that get the propagation speed wrong, or that break
#     the upwind symmetry of the Rusanov flux at non-PEC boundaries,
#     show up here but not in the cavity test.
#
# Pass criteria (P=2, NX=NY=16, single-rank):
#   * rel L2(state, all 6 components) < 1e-3
#   * leakage into the analytically-zero components (Ex, Ey, Bx, Bz)
#     stays bounded -- the diagonal Kuhn-style triangulation breaks
#     y-symmetry under Rusanov, so we allow ~5e-4 here (empirical
#     ~2.7e-4 on current code) but trip a hard regression beyond
#     that.
#   * no NaN / Inf
# ======================================================================

from std.math import sqrt, sin, cos, pi, isnan, isinf
from std.sys import has_accelerator
from std.gpu.host import DeviceContext, DeviceBuffer
from src import mpi
from src.local_mesh_2d import LocalMesh2D
from src.local_mesh_2d_gpu import LocalMesh2DGpu
from src.local_mesh_2d_gpu_maxwell import maxwell_rk_stage_2d
from src.ssprk3 import ssprk3_stage_plans
from src.reference_2d import (
    ReferenceElement2D,
    num_tri_nodes_2d,
    num_edge_nodes,
)
from src.reference_2d_gpu import ReferenceElement2DGpu
from src.boundary import BoundaryConditions2D


comptime P = 2
comptime NX = 16
comptime NY = 16
comptime LX = 1.0
comptime LY = 1.0
comptime C_LIGHT: Float32 = 1.0
comptime CFL = 0.2
# omega = c * 2 pi at k = (2 pi, 0); period T = 2 pi / omega = 1 / c.
comptime T_FINAL: Float64 = 1.0  # exactly one period at c = 1

# Empirical: ~5.8e-4 at P=2, NX=NY=16; 1e-3 leaves ~2x headroom and
# catches any regression in the propagation speed or upwind symmetry
# beyond a small constant.
comptime L2_MAX_REL: Float64 = 1.0e-3
# Empirical zero-component leakage ~2.7e-4 (triangulation asymmetry
# under Rusanov).  5e-4 is ~2x the empirical floor; a hard regression
# would push this 10x or more.
comptime ZERO_COMPONENT_MAX: Float64 = 5.0e-4


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    mpi.init()
    var size = mpi.world_size()
    if size > 1:
        mpi.finalize()
        print("bench_maxwell_plane_wave_2d: runs at np=1 only")
        return

    print("bench_maxwell_plane_wave_2d (2D TM plane wave, periodic box)")
    print("  P=", P, "  mesh=", NX, "x", NY, "  T=", T_FINAL)

    comptime NP_p = num_tri_nodes_2d(P)
    comptime NFP_e = num_edge_nodes(P)
    comptime NC = 6
    var ctx = DeviceContext()

    var bcs = BoundaryConditions2D.periodic()
    var host_mesh = LocalMesh2D[P](NX, NY, LX, LY, bcs)
    var mesh_coords = LocalMesh2D[P](NX, NY, LX, LY, bcs)
    var host_re = ReferenceElement2D[P]()
    var gpu_mesh = LocalMesh2DGpu[P](ctx, host_mesh^)
    var gpu_re = ReferenceElement2DGpu[P](ctx, host_re)

    # IC at t=0: Ez = cos(2 pi x), By = -(1/c) cos(2 pi x), rest 0.
    var two_pi = 2.0 * pi
    var inv_c = 1.0 / Float64(C_LIGHT)
    var n_q = gpu_mesh.num_elements * NP_p * NC
    var host_q = List[Float32]()
    var host_ic = List[Float32]()
    for elem in range(gpu_mesh.num_elements):
        for nn in range(NP_p):
            var x = mesh_coords.elem_node_xyz[(elem * NP_p + nn) * 2 + 0]
            var Ez = cos(two_pi * x)
            var By = -inv_c * cos(two_pi * x)
            host_q.append(Float32(0.0))
            host_ic.append(Float32(0.0))  # Ex
            host_q.append(Float32(0.0))
            host_ic.append(Float32(0.0))  # Ey
            host_q.append(Float32(Ez))
            host_ic.append(Float32(Ez))  # Ez
            host_q.append(Float32(0.0))
            host_ic.append(Float32(0.0))  # Bx
            host_q.append(Float32(By))
            host_ic.append(Float32(By))  # By
            host_q.append(Float32(0.0))
            host_ic.append(Float32(0.0))  # Bz

    var d_q = ctx.enqueue_create_buffer[DType.float32](n_q)
    var d_q1 = ctx.enqueue_create_buffer[DType.float32](n_q)
    var d_q2 = ctx.enqueue_create_buffer[DType.float32](n_q)
    var d_fstar = ctx.enqueue_create_buffer[DType.float32](
        gpu_mesh.num_faces * NFP_e * NC
    )
    var hbuf_q = ctx.enqueue_create_host_buffer[DType.float32](n_q)
    var hptr_q = hbuf_q.unsafe_ptr()
    for i in range(n_q):
        hptr_q[i] = host_q[i]
    ctx.enqueue_copy(d_q, hbuf_q)
    ctx.synchronize()

    var h_cell = LX / Float64(NX)
    var dt_est = CFL * h_cell / (Float64(C_LIGHT) * Float64(2 * P + 1))
    var num_steps = Int(T_FINAL / dt_est) + 1
    var dt = Float32(T_FINAL / Float64(num_steps))
    print("  dt=", dt, "  steps=", num_steps)

    var stage_plans = ssprk3_stage_plans(
        d_q.unsafe_ptr(),
        d_q1.unsafe_ptr(),
        d_q2.unsafe_ptr(),
    )
    for _ in range(num_steps):
        for stage in stage_plans:
            maxwell_rk_stage_2d[P](
                ctx,
                gpu_mesh,
                gpu_re.d_Lift_ref.unsafe_ptr(),
                gpu_re.d_D_ref.unsafe_ptr(),
                stage.q_in,
                stage.q_a,
                stage.q_b,
                stage.q_out,
                d_fstar.unsafe_ptr(),
                C_LIGHT,
                stage.a,
                stage.b,
                stage.c,
                dt,
            )
    ctx.synchronize()
    ctx.enqueue_copy(hbuf_q, d_q)
    ctx.synchronize()

    # Per-component sums for the zero-component leakage check, plus
    # the global state-vector L2 against the IC.
    var sum_sq: Float64 = 0.0
    var sum_ic: Float64 = 0.0
    var max_zero_leak: Float64 = 0.0
    for elem in range(gpu_mesh.num_elements):
        for nn in range(NP_p):
            for c in range(NC):
                var k_idx = (elem * NP_p + nn) * NC + c
                var v = hptr_q[k_idx]
                if isnan(v) or isinf(v):
                    raise Error(
                        "bench_maxwell_plane_wave_2d: non-finite output"
                    )
                var err = Float64(v - host_ic[k_idx])
                sum_sq += err * err
                var ic = Float64(host_ic[k_idx])
                sum_ic += ic * ic
                # Components Ex (0), Ey (1), Bx (3), Bz (5) are
                # identically zero in the analytic solution.
                if c == 0 or c == 1 or c == 3 or c == 5:
                    var av = Float64(v)
                    if av < 0.0:
                        av = -av
                    if av > max_zero_leak:
                        max_zero_leak = av

    var l2 = sqrt(sum_sq / Float64(n_q))
    var l2_ic = sqrt(sum_ic / Float64(n_q))
    var rel = l2 / l2_ic
    print("  rel L2(state) =", rel, "  (threshold", L2_MAX_REL, ")")
    print(
        "  max |Ex|/|Ey|/|Bx|/|Bz| =",
        max_zero_leak,
        "  (threshold",
        ZERO_COMPONENT_MAX,
        ")",
    )

    if rel > L2_MAX_REL:
        raise Error(
            String("bench_maxwell_plane_wave_2d FAILED: rel L2 ")
            + String(rel)
            + " > "
            + String(L2_MAX_REL)
        )
    if max_zero_leak > ZERO_COMPONENT_MAX:
        raise Error(
            String("bench_maxwell_plane_wave_2d FAILED: zero-component ")
            + "leakage "
            + String(max_zero_leak)
            + " > "
            + String(ZERO_COMPONENT_MAX)
        )

    print("=== bench_maxwell_plane_wave_2d PASSED ===")
    mpi.finalize()
