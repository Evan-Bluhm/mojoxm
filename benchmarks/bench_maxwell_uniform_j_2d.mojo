# ======================================================================
# bench_maxwell_uniform_j_2d -- 2D Maxwell J source-term gate
# ======================================================================
#
# 2D analog of bench_maxwell_uniform_j_3d.  Closes a 2D feature-parity
# gap: the 3D Maxwell physics struct accepts a uniform current J and
# magnetization M and applies them via Maxwell.source_term, but the
# 2D Maxwell GPU kernel previously had no J / M plumbing at all
# (the volume kernel computed RHS = -inv_2A * face only).  This
# commit adds J / M parameters to maxwell_vol_lift_combine_rk_kernel_2d
# and threads them through maxwell_rk_stage_2d as default-zero args.
#
# Cleanest analytic test: uniform J on a periodic box with q = 0
# at t = 0.  Since q is uniform, all flux divergences vanish and
# only the source contributes:
#
#   dEx/dt = -c^2 * Jx  ->  Ex(T) = -c^2 * Jx * T   (exact)
#   B(T) = 0  identically (M = 0)
#
# Pass criteria (P=2, periodic 8x8, T=0.5):
#   * |Ex - Ex_exact| / |Ex_exact| < 1e-5 (Float32 roundoff floor)
#   * max |Ey, Ez, Bx, By, Bz| < 1e-5
#   * no NaN / Inf
# ======================================================================

from std.math import sqrt, isnan, isinf
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


comptime P = 2
comptime NX = 8
comptime NY = 8
comptime LX = 1.0
comptime LY = 1.0
comptime C_LIGHT: Float32 = 1.0
comptime JX: Float32 = 1.0
comptime CFL = 0.2
comptime T_FINAL: Float64 = 0.5

# Float32 roundoff floor over ~100 SSPRK3 steps.  Same threshold as
# bench_maxwell_uniform_j_3d.
comptime EX_REL_TOL: Float64 = 1.0e-5
comptime ZERO_COMPONENT_TOL: Float64 = 1.0e-5


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    mpi.init()
    var size = mpi.world_size()
    if size > 1:
        mpi.finalize()
        print("bench_maxwell_uniform_j_2d: runs at np=1 only")
        return

    print("bench_maxwell_uniform_j_2d (uniform J source-term gate)")
    print("  P=", P, "  mesh=", NX, "x", NY, "  Jx=", JX, "  T=", T_FINAL)

    comptime NP_p = num_tri_nodes_2d(P)
    comptime NFP_e = num_edge_nodes(P)
    comptime NC = 6
    var ctx = DeviceContext()

    var host_mesh = LocalMesh2D[P](NX, NY, LX, LY)
    var host_re = ReferenceElement2D[P]()
    var gpu_mesh = LocalMesh2DGpu[P](ctx, host_mesh^)
    var gpu_re = ReferenceElement2DGpu[P](ctx, host_re)

    # Zero IC -- q = 0 everywhere.
    var n_q = gpu_mesh.num_elements * NP_p * NC
    var d_q = ctx.enqueue_create_buffer[DType.float32](n_q)
    var d_q1 = ctx.enqueue_create_buffer[DType.float32](n_q)
    var d_q2 = ctx.enqueue_create_buffer[DType.float32](n_q)
    var d_fstar = ctx.enqueue_create_buffer[DType.float32](gpu_mesh.num_faces * NFP_e * NC)
    var hbuf_q = ctx.enqueue_create_host_buffer[DType.float32](n_q)
    var hptr_q = hbuf_q.unsafe_ptr()
    for k in range(n_q):
        hptr_q[k] = Float32(0.0)
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
                a=stage.a,
                b=stage.b,
                cc=stage.c,
                dt=dt,
                Jx=JX,
            )
    ctx.synchronize()
    ctx.enqueue_copy(hbuf_q, d_q)
    ctx.synchronize()

    var Ex_expect = Float64(-C_LIGHT * C_LIGHT * JX * Float32(T_FINAL))
    var max_ex_rel: Float64 = 0.0
    var max_zero: Float64 = 0.0
    var n_nodes = gpu_mesh.num_elements * NP_p
    for i in range(n_nodes):
        var ex = hptr_q[i * 6 + 0]
        var ey = hptr_q[i * 6 + 1]
        var ez = hptr_q[i * 6 + 2]
        var bx = hptr_q[i * 6 + 3]
        var by = hptr_q[i * 6 + 4]
        var bz = hptr_q[i * 6 + 5]
        if (
            isnan(ex)
            or isinf(ex)
            or isnan(ey)
            or isinf(ey)
            or isnan(ez)
            or isinf(ez)
            or isnan(bx)
            or isinf(bx)
            or isnan(by)
            or isinf(by)
            or isnan(bz)
            or isinf(bz)
        ):
            raise Error("bench_maxwell_uniform_j_2d: non-finite output")
        var d_ex = Float64(ex) - Ex_expect
        if d_ex < 0.0:
            d_ex = -d_ex
        var rel = d_ex / abs(Ex_expect)
        if rel > max_ex_rel:
            max_ex_rel = rel
        var av_ey = Float64(ey if ey >= Float32(0.0) else -ey)
        var av_ez = Float64(ez if ez >= Float32(0.0) else -ez)
        var av_bx = Float64(bx if bx >= Float32(0.0) else -bx)
        var av_by = Float64(by if by >= Float32(0.0) else -by)
        var av_bz = Float64(bz if bz >= Float32(0.0) else -bz)
        if av_ey > max_zero:
            max_zero = av_ey
        if av_ez > max_zero:
            max_zero = av_ez
        if av_bx > max_zero:
            max_zero = av_bx
        if av_by > max_zero:
            max_zero = av_by
        if av_bz > max_zero:
            max_zero = av_bz

    print("  Ex(T) expected =", Ex_expect)
    print(
        "  max |Ex - exact| / |Ex_exact| =",
        max_ex_rel,
        "  (threshold",
        EX_REL_TOL,
        ")",
    )
    print(
        "  max |Ey,Ez,Bx,By,Bz| =",
        max_zero,
        "  (threshold",
        ZERO_COMPONENT_TOL,
        ")",
    )

    if max_ex_rel > EX_REL_TOL:
        raise Error(
            String("bench_maxwell_uniform_j_2d FAILED: Ex rel err ") + String(max_ex_rel) + " > " + String(EX_REL_TOL)
        )
    if max_zero > ZERO_COMPONENT_TOL:
        raise Error(
            String("bench_maxwell_uniform_j_2d FAILED: zero-component drift ")
            + String(max_zero)
            + " > "
            + String(ZERO_COMPONENT_TOL)
        )

    print("=== bench_maxwell_uniform_j_2d PASSED ===")
    mpi.finalize()


def abs(x: Float64) -> Float64:
    return x if x >= 0.0 else -x
