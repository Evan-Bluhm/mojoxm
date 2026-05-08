# ======================================================================
# bench_maxwell_inflow_2d -- 2D Maxwell BC_INFLOW preservation gate
# ======================================================================
#
# Companion to bench_maxwell_outflow_2d.  Exercises the BC_INFLOW
# dispatch arm of `maxwell_face_flux_kernel_2d`, which writes the
# user-set 6-component ghost state (Ex, Ey, Ez, Bx, By, Bz) before
# the Rusanov arbiter decides which side's information propagates
# into the domain.
#
# Cleanest test: uniform constant state matched to the inflow ghost.
# Since the IC matches the BC_INFLOW ghost exactly, the analytic
# solution is the IC unchanged for all time -- by the same
# divergence-theorem argument as the BC_OUTFLOW analog.
#
# Pass criteria (P=2, NX=NY=8, T=0.5):
#   * max |q - q_IC| < 1e-3 (Float32 epsilon * step accumulation;
#     same threshold as bench_maxwell_outflow_2d)
#   * no NaN / Inf
# ======================================================================

from std.math import isnan, isinf
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
from src.boundary import BoundaryConditions2D, BC_INFLOW


comptime P = 2
comptime NX = 8
comptime NY = 8
comptime LX = 1.0
comptime LY = 1.0
comptime C_LIGHT: Float32 = 1.0
comptime CFL = 0.2
comptime T_FINAL: Float64 = 0.5

# Uniform IC components -- non-zero in every component to exercise
# every flux term.  Must match the BC_INFLOW ghost exactly.
comptime EX0: Float32 = 0.3
comptime EY0: Float32 = -0.2
comptime EZ0: Float32 = 0.5
comptime BX0: Float32 = 0.4
comptime BY0: Float32 = 0.1
comptime BZ0: Float32 = -0.6

comptime DRIFT_TOL: Float64 = 1.0e-3


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    mpi.init()
    var size = mpi.world_size()
    if size > 1:
        mpi.finalize()
        print("bench_maxwell_inflow_2d: runs at np=1 only")
        return

    print("bench_maxwell_inflow_2d (BC_INFLOW preservation gate)")
    print("  P=", P, "  mesh=", NX, "x", NY, "  T=", T_FINAL)

    comptime NP_p = num_tri_nodes_2d(P)
    comptime NFP_e = num_edge_nodes(P)
    comptime NC = 6
    var ctx = DeviceContext()

    # BC_INFLOW on all four faces -- the path that is otherwise
    # untested in 2D Maxwell.
    var bcs = BoundaryConditions2D(BC_INFLOW, BC_INFLOW, BC_INFLOW, BC_INFLOW)
    var host_mesh = LocalMesh2D[P](NX, NY, LX, LY, bcs)
    var host_re = ReferenceElement2D[P]()
    var gpu_mesh = LocalMesh2DGpu[P](ctx, host_mesh^)
    var gpu_re = ReferenceElement2DGpu[P](ctx, host_re)

    # Uniform IC = inflow ghost.
    var n_q = gpu_mesh.num_elements * NP_p * NC
    var host_q = List[Float32]()
    var host_ic = List[Float32]()
    for _ in range(gpu_mesh.num_elements * NP_p):
        host_q.append(EX0)
        host_ic.append(EX0)
        host_q.append(EY0)
        host_ic.append(EY0)
        host_q.append(EZ0)
        host_ic.append(EZ0)
        host_q.append(BX0)
        host_ic.append(BX0)
        host_q.append(BY0)
        host_ic.append(BY0)
        host_q.append(BZ0)
        host_ic.append(BZ0)

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
                inflow_Ex=EX0,
                inflow_Ey=EY0,
                inflow_Ez=EZ0,
                inflow_Bx=BX0,
                inflow_By=BY0,
                inflow_Bz=BZ0,
            )
    ctx.synchronize()
    ctx.enqueue_copy(hbuf_q, d_q)
    ctx.synchronize()

    var max_drift: Float64 = 0.0
    for k in range(n_q):
        var v = hptr_q[k]
        if isnan(v) or isinf(v):
            raise Error("bench_maxwell_inflow_2d: non-finite output")
        var d = Float64(v - host_ic[k])
        if d < 0.0:
            d = -d
        if d > max_drift:
            max_drift = d

    print("  max |q - q_IC| =", max_drift, "  (threshold", DRIFT_TOL, ")")
    if max_drift > DRIFT_TOL:
        raise Error("bench_maxwell_inflow_2d FAILED: drift " + String(max_drift) + " > " + String(DRIFT_TOL))

    print("=== bench_maxwell_inflow_2d PASSED ===")
    mpi.finalize()
