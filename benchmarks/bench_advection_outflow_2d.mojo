# ======================================================================
# bench_advection_outflow_2d -- BC_OUTFLOW drainage gate
# ======================================================================
#
# Companion benchmark to advection_outflow_2d_gpu (the example).
# Gaussian bump at (0.3, 0.3) with velocity v = (1, 1); BC_OUTFLOW on
# all four sides.  After T = 1.2 the bump has translated to (1.5, 1.5)
# which is well outside the domain [0, 1]^2, so the analytic mass on
# the domain is zero (the entire bump has drained out).
#
# Pass criteria (P=2, NX=NY=32, T=1.2):
#   * mass-matrix-weighted integral of q at t = T < 1e-3 of the IC
#     mass (measured ~5e-7 in current code -- the bump is gone modulo
#     scheme dissipation tail)
#   * no NaN / Inf
#
# This is the only 2D analytic-solution benchmark that exercises
# BC_OUTFLOW on all four boundaries (bench_euler_sod_2d uses outflow
# only on +/- x; bench_euler_channel_steady_2d uses inflow + outflow).
# ======================================================================

from std.math import sqrt, exp, isnan, isinf
from std.sys import has_accelerator
from std.gpu.host import DeviceContext, DeviceBuffer
from src import mpi
from src.local_mesh_2d import LocalMesh2D
from src.local_mesh_2d_gpu import LocalMesh2DGpu
from src.local_mesh_2d_gpu_advection import advection_rk_stage_2d
from src.ssprk3 import ssprk3_stage_plans
from src.reference_2d import (
    ReferenceElement2D, num_tri_nodes_2d, num_edge_nodes,
)
from src.reference_2d_gpu import ReferenceElement2DGpu
from src.boundary import BoundaryConditions2D, BC_OUTFLOW


comptime P = 2
comptime NX = 32
comptime NY = 32
comptime LX = 1.0
comptime LY = 1.0
comptime VX: Float32 = 1.0
comptime VY: Float32 = 1.0
comptime T_FINAL: Float32 = 1.2
comptime CFL: Float32 = 0.3

comptime CX: Float32 = 0.3
comptime CY: Float32 = 0.3
comptime SIGMA: Float32 = 0.1

# Measured residual mass ~1e-12 of IC (essentially noise) on
# current code.  1e-9 leaves 3 orders of headroom and would catch
# any regression that leaves more than O(1e-9) mass behind.
comptime DRAIN_TOL_REL: Float64 = 1.0e-9


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    mpi.init()
    var size = mpi.world_size()
    if size > 1:
        mpi.finalize()
        print("bench_advection_outflow_2d: runs at np=1 only")
        return

    print("bench_advection_outflow_2d (BC_OUTFLOW x 4, drainage)")
    print("  P=", P, "  NX=", NX, "  T=", T_FINAL)

    comptime NP_p = num_tri_nodes_2d(P)
    comptime NFP_e = num_edge_nodes(P)
    var ctx = DeviceContext()

    var bcs = BoundaryConditions2D(
        BC_OUTFLOW, BC_OUTFLOW, BC_OUTFLOW, BC_OUTFLOW,
    )
    var host_mesh = LocalMesh2D[P](Nx=NX, Ny=NY, Lx=LX, Ly=LY, bcs=bcs)
    var host_re = ReferenceElement2D[P]()
    var mesh_coords = LocalMesh2D[P](
        Nx=NX, Ny=NY, Lx=LX, Ly=LY, bcs=bcs,
    )
    var gpu_mesh = LocalMesh2DGpu[P](ctx=ctx, host=host_mesh^)
    var gpu_re = ReferenceElement2DGpu[P](ctx=ctx, host=host_re)

    var n_q = gpu_mesh.num_elements * NP_p
    var host_q = List[Float32]()
    var inv_two_sigma2 = Float32(1.0) / (Float32(2.0) * SIGMA * SIGMA)
    for elem in range(gpu_mesh.num_elements):
        for nn in range(NP_p):
            var x = mesh_coords.elem_node_xyz[(elem * NP_p + nn) * 2 + 0]
            var y = mesh_coords.elem_node_xyz[(elem * NP_p + nn) * 2 + 1]
            var dx = Float32(x) - CX
            var dy = Float32(y) - CY
            var v = exp(-(dx * dx + dy * dy) * inv_two_sigma2)
            host_q.append(v)

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

    # IC mass (mass-matrix-weighted nodal quadrature).
    var node_w = host_re.node_weights.copy()
    var mass_ic: Float64 = 0.0
    for elem in range(gpu_mesh.num_elements):
        for nn in range(NP_p):
            mass_ic += Float64(host_q[elem * NP_p + nn]) * node_w[nn]

    var h = LX / Float32(NX)
    var v_mag = sqrt(VX * VX + VY * VY)
    var dt_est = CFL * h / (v_mag * Float32(2 * P + 1))
    var num_steps = Int(T_FINAL / dt_est) + 1
    var dt = T_FINAL / Float32(num_steps)

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
                vx=VX, vy=VY,
                a=stage.a, b=stage.b, cc=stage.c, dt=dt,
            )
    ctx.synchronize()

    ctx.enqueue_copy(hbuf_q, d_q)
    ctx.synchronize()

    var mass_fin: Float64 = 0.0
    var max_abs: Float32 = 0.0
    for elem in range(gpu_mesh.num_elements):
        for nn in range(NP_p):
            var v = hptr_q[elem * NP_p + nn]
            if isnan(v) or isinf(v):
                raise Error("bench_advection_outflow_2d: non-finite output")
            mass_fin += Float64(v) * node_w[nn]
            var av = v if v >= Float32(0.0) else -v
            if av > max_abs: max_abs = av

    var rel = mass_fin / mass_ic
    if rel < 0.0: rel = -rel
    print("  mass(IC)=", mass_ic,
          "  mass(t=T)=", mass_fin,
          "  rel=", rel,
          "  max |q|=", max_abs)

    if rel > DRAIN_TOL_REL:
        raise Error(
            String("bench_advection_outflow_2d FAILED: residual mass ")
            + String(rel) + " > " + String(DRAIN_TOL_REL)
        )

    print("=== bench_advection_outflow_2d PASSED ===")
    mpi.finalize()
