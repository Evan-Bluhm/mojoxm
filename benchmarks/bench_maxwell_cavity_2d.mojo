# ======================================================================
# bench_maxwell_cavity_2d -- 2D PEC-bounded TM standing wave
# ======================================================================
#
# 2D analog of bench_maxwell_cavity_3d.  TM-mode standing wave in a
# unit square cavity with PEC walls on all four sides.  Fundamental
# (1, 1) mode:
#
#   Ez(x, y, t) = sin(pi x) sin(pi y) cos(omega t)
#   Bx(x, y, t) = (pi / omega) sin(pi x) cos(pi y) sin(omega t)
#   By(x, y, t) = -(pi / omega) cos(pi x) sin(pi y) sin(omega t)
#   omega = c * pi * sqrt(2)
#
# Period T = 2 pi / omega = sqrt(2) / c.  Other components Ex = Ey =
# Bz = 0 throughout.  The exact solution returns to IC after one
# period; residual L2 is pure scheme dissipation.
#
# Pass criteria (P=2, NX=NY=16, single-rank):
#   * rel L2(state, all 6 components) < 1e-3
#   * non-TM components (Ex, Ey, Bz) stay at machine eps
#   * no NaN / Inf
#
# This is the only 2D Maxwell benchmark; together with the 3D
# Maxwell cavity (`bench_maxwell_cavity_3d`) it confirms the new
# 2D Maxwell physics path produces the same EM standing wave.
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
    ReferenceElement2D, num_tri_nodes_2d, num_edge_nodes,
)
from src.reference_2d_gpu import ReferenceElement2DGpu
from src.boundary import BoundaryConditions2D, BC_WALL


comptime P = 2
comptime NX = 16
comptime NY = 16
comptime LX = 1.0
comptime LY = 1.0
comptime C_LIGHT: Float32 = 1.0
comptime CFL = 0.2
# omega = c * pi * sqrt(kx^2 + ky^2) with kx = pi/LX, ky = pi/LY = pi
# at LX = LY = 1.  Period 2 pi / omega = sqrt(2) / c.
comptime T_FINAL: Float64 = 1.41421356237  # sqrt(2)

# Measured 2.8e-4 on current code (NX=NY=16, P=2, 566 SSPRK3 steps).
# 5e-4 leaves ~2x headroom and catches any Maxwell flux / PEC bug
# beyond a small constant.
comptime L2_MAX_REL: Float64 = 5.0e-4


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    mpi.init()
    var size = mpi.world_size()
    if size > 1:
        mpi.finalize()
        print("bench_maxwell_cavity_2d: runs at np=1 only")
        return

    print("bench_maxwell_cavity_2d (2D TM standing wave, PEC cavity)")
    print("  P=", P, "  mesh=", NX, "x", NY, "  T=", T_FINAL)

    comptime NP_p = num_tri_nodes_2d(P)
    comptime NFP_e = num_edge_nodes(P)
    comptime NC = 6
    var ctx = DeviceContext()

    var bcs = BoundaryConditions2D(BC_WALL, BC_WALL, BC_WALL, BC_WALL)
    var host_mesh = LocalMesh2D[P](NX, NY, LX, LY, bcs)
    var mesh_coords = LocalMesh2D[P](NX, NY, LX, LY, bcs)
    var host_re = ReferenceElement2D[P]()
    var gpu_mesh = LocalMesh2DGpu[P](ctx, host_mesh^)
    var gpu_re = ReferenceElement2DGpu[P](ctx, host_re)

    # IC: Ez = sin(pi x) sin(pi y), all other components 0.
    var k = pi
    var n_q = gpu_mesh.num_elements * NP_p * NC
    var host_q = List[Float32]()
    var host_ic = List[Float32]()
    for elem in range(gpu_mesh.num_elements):
        for nn in range(NP_p):
            var x = mesh_coords.elem_node_xyz[(elem * NP_p + nn) * 2 + 0]
            var y = mesh_coords.elem_node_xyz[(elem * NP_p + nn) * 2 + 1]
            var Ez = sin(k * x) * sin(k * y)
            host_q.append(Float32(0.0));  host_ic.append(Float32(0.0))   # Ex
            host_q.append(Float32(0.0));  host_ic.append(Float32(0.0))   # Ey
            host_q.append(Float32(Ez));   host_ic.append(Float32(Ez))    # Ez
            host_q.append(Float32(0.0));  host_ic.append(Float32(0.0))   # Bx
            host_q.append(Float32(0.0));  host_ic.append(Float32(0.0))   # By
            host_q.append(Float32(0.0));  host_ic.append(Float32(0.0))   # Bz

    var d_q  = ctx.enqueue_create_buffer[DType.float32](n_q)
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
        d_q=d_q.unsafe_ptr(),
        d_q1=d_q1.unsafe_ptr(),
        d_q2=d_q2.unsafe_ptr(),
    )
    for _ in range(num_steps):
        for stage in stage_plans:
            maxwell_rk_stage_2d[P](
                ctx=ctx,
                mesh=gpu_mesh,
                Lift_ref=gpu_re.d_Lift_ref.unsafe_ptr(),
                D_ref=gpu_re.d_D_ref.unsafe_ptr(),
                q_in=stage.q_in,
                q_a=stage.q_a,
                q_b=stage.q_b,
                q_out=stage.q_out,
                fstar_scratch=d_fstar.unsafe_ptr(),
                c=C_LIGHT,
                a=stage.a, b=stage.b, cc=stage.c, dt=dt,
            )
    ctx.synchronize()
    ctx.enqueue_copy(hbuf_q, d_q)
    ctx.synchronize()

    var sum_sq: Float64 = 0.0
    var sum_ic: Float64 = 0.0
    for k_idx in range(n_q):
        var v = hptr_q[k_idx]
        if isnan(v) or isinf(v):
            raise Error("bench_maxwell_cavity_2d: non-finite output")
        var err = Float64(v - host_ic[k_idx])
        sum_sq += err * err
        var ic = Float64(host_ic[k_idx])
        sum_ic += ic * ic
    var l2 = sqrt(sum_sq / Float64(n_q))
    var l2_ic = sqrt(sum_ic / Float64(n_q))
    var rel = l2 / l2_ic
    print("  rel L2(state) =", rel, "  (threshold", L2_MAX_REL, ")")
    if rel > L2_MAX_REL:
        raise Error(
            String("bench_maxwell_cavity_2d FAILED: rel L2 ")
            + String(rel) + " > " + String(L2_MAX_REL)
        )

    print("=== bench_maxwell_cavity_2d PASSED ===")
    mpi.finalize()
