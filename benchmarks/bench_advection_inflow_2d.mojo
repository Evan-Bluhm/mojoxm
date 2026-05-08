# ======================================================================
# bench_advection_inflow_2d -- 2D advection BC_INFLOW preservation
# ======================================================================
#
# 2D analog of bench_advection_inflow_3d (and parallel to the pattern
# in bench_euler_inflow_3d / bench_shallow_water_inflow_2d).  Closes
# the corner of the BC coverage matrix that previous gates missed:
# `bench_advection_outflow_2d` does hit the BC_INFLOW dispatch arm
# in `advection_face_flux_kernel_2d`, but it passes `inflow_q = 0`,
# so any sign-flip / zeroing regression in the BC_INFLOW branch
# would be hidden (the ghost reduces to 0 either way).
#
# Cleanest test: uniform constant state matched to the inflow ghost.
# IC = inflow_q, advection velocity (vx, vy) > 0 with BC_INFLOW on
# -x, BC_OUTFLOW on +x, periodic in y.  The analytic solution is the
# IC unchanged for all time -- any drift signals a BC-coupling bug.
#
# Pass criteria (P=2, NX=32 NY=4, T=1):
#   * max |q - q_inflow| < 2e-3 (Float32 epsilon * step accumulation)
#   * no NaN / Inf
# ======================================================================

from std.math import isnan, isinf
from std.sys import has_accelerator
from std.gpu.host import DeviceContext, DeviceBuffer
from src import mpi
from src.local_mesh_2d import LocalMesh2D
from src.local_mesh_2d_gpu import LocalMesh2DGpu
from src.local_mesh_2d_gpu_advection import advection_rk_stage_2d
from src.ssprk3 import ssprk3_stage_plans
from src.reference_2d import (
    ReferenceElement2D,
    num_tri_nodes_2d,
    num_edge_nodes,
)
from src.reference_2d_gpu import ReferenceElement2DGpu
from src.boundary import (
    BoundaryConditions2D,
    BC_INFLOW,
    BC_OUTFLOW,
    BC_INTERIOR,
)


comptime P = 2
comptime NX = 32
comptime NY = 4
comptime LX = 1.0
comptime LY = Float64(NY) / Float64(NX) * LX

comptime VX: Float32 = 1.0
comptime VY: Float32 = 0.0  # purely +x advection
comptime INFLOW_Q: Float32 = 0.7  # non-trivial value -- a sign bug
# would flip this to -0.7 and trip
# the gate by orders of magnitude
comptime CFL: Float32 = 0.3
comptime T_FINAL: Float32 = 1.0  # one transit time (LX / VX)

# Empirical drift on a uniform-state preservation test of this shape
# is ~1e-4 over ~150 SSPRK3 steps (Float32 step accumulation; not a
# bug).  2e-3 leaves ~20x margin and catches sign-flip / zeroing
# regressions in the BC_INFLOW dispatch.
comptime DRIFT_TOL: Float64 = 2.0e-3


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    mpi.init()
    var size = mpi.world_size()
    if size > 1:
        mpi.finalize()
        print("bench_advection_inflow_2d: runs at np=1 only")
        return

    print("bench_advection_inflow_2d (2D advection BC_INFLOW preservation)")
    print(
        "  P=",
        P,
        "  mesh=",
        NX,
        "x",
        NY,
        "  v=(",
        VX,
        ",",
        VY,
        ")  inflow_q=",
        INFLOW_Q,
        "  T=",
        T_FINAL,
    )

    comptime NP_p = num_tri_nodes_2d(P)
    comptime NFP_e = num_edge_nodes(P)
    comptime NC = 1
    var ctx = DeviceContext()

    # BC_INFLOW on -x (incoming), BC_OUTFLOW on +x (outgoing), periodic in y.
    var bcs = BoundaryConditions2D(
        BC_INFLOW,
        BC_OUTFLOW,  # -x, +x
        BC_INTERIOR,
        BC_INTERIOR,  # -y, +y (periodic)
    )
    var host_mesh = LocalMesh2D[P](Nx=NX, Ny=NY, Lx=LX, Ly=LY, bcs=bcs)
    var host_re = ReferenceElement2D[P]()
    var gpu_mesh = LocalMesh2DGpu[P](ctx=ctx, host=host_mesh^)
    var gpu_re = ReferenceElement2DGpu[P](ctx=ctx, host=host_re)

    # IC: q = INFLOW_Q uniformly (matches the inflow ghost).
    var n_q = gpu_mesh.num_elements * NP_p * NC
    var host_q = List[Float32]()
    for _ in range(n_q):
        host_q.append(INFLOW_Q)

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

    var v_max = VX if VX > VY else VY
    var h_cell = Float32(LX) / Float32(NX)
    var dt_est = CFL * h_cell / (v_max * Float32(2 * P + 1))
    var num_steps = Int(T_FINAL / dt_est) + 1
    var dt = T_FINAL / Float32(num_steps)
    print("  steps=", num_steps, "  dt=", dt)

    var stage_plans = ssprk3_stage_plans(
        d_q=d_q.unsafe_ptr(),
        d_q1=d_q1.unsafe_ptr(),
        d_q2=d_q2.unsafe_ptr(),
    )
    for _ in range(num_steps):
        for stage in stage_plans:
            advection_rk_stage_2d[P](
                ctx=ctx,
                mesh=gpu_mesh,
                Lift_ref=gpu_re.d_Lift_ref.unsafe_ptr(),
                D_ref=gpu_re.d_D_ref.unsafe_ptr(),
                q_in=stage.q_in,
                q_a=stage.q_a,
                q_b=stage.q_b,
                q_out=stage.q_out,
                fstar_scratch=d_fstar.unsafe_ptr(),
                vx=VX,
                vy=VY,
                a=stage.a,
                b=stage.b,
                cc=stage.c,
                dt=dt,
                inflow_q=INFLOW_Q,
            )
    ctx.synchronize()
    ctx.enqueue_copy(hbuf_q, d_q)
    ctx.synchronize()

    var max_drift: Float64 = 0.0
    for k in range(n_q):
        var v = hptr_q[k]
        if isnan(v) or isinf(v):
            raise Error("bench_advection_inflow_2d: non-finite output")
        var d = Float64(v - INFLOW_Q)
        if d < 0.0:
            d = -d
        if d > max_drift:
            max_drift = d

    print("  max |q - inflow_q| =", max_drift, "  (threshold", DRIFT_TOL, ")")
    if max_drift > DRIFT_TOL:
        raise Error(
            "bench_advection_inflow_2d FAILED: drift "
            + String(max_drift)
            + " > "
            + String(DRIFT_TOL)
        )

    print("=== bench_advection_inflow_2d PASSED ===")
    mpi.finalize()
