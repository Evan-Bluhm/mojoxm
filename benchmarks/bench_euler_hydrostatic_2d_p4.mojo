# ======================================================================
# bench_euler_hydrostatic_2d_p4 -- 2D Euler gravity at P=4 (NP=15)
# ======================================================================
#
# P=4 (NP=15) counterpart of bench_euler_hydrostatic_2d_p3.  Same
# stratified rest IC, same gravity gy<0, but routed through
# LocalMesh2D[4] / euler_rk_stage_hllc_2d[4] so the 2D Euler
# gravity source path is exercised at NP=15.  Mesh size unchanged
# (4x12) since the per-element volume work scales with NP^2 and dt
# tightens to 1/(2P+1)=1/9; the suite still runs in well under a
# second.
#
# Pass criteria (P=4, NX=4, NY=12, T=1):
#   * max |v_mag| < 1e-3 (state remains at rest)
#   * max |rho - rho0| / rho0 < 1e-4 (density unchanged)
#   * max |p_measured - p_exact| / p0 < 5e-4 (pressure profile)
#   * no NaN / Inf
# ======================================================================

from std.math import sqrt, isnan, isinf
from std.sys import has_accelerator
from std.gpu.host import DeviceContext, DeviceBuffer
from src import mpi
from src.local_mesh_2d import LocalMesh2D
from src.local_mesh_2d_gpu import LocalMesh2DGpu
from src.local_mesh_2d_gpu_euler import euler_rk_stage_hllc_2d
from src.ssprk3 import ssprk3_stage_plans
from src.reference_2d import (
    ReferenceElement2D, num_tri_nodes_2d, num_edge_nodes,
)
from src.reference_2d_gpu import ReferenceElement2DGpu
from src.boundary import BoundaryConditions2D, BC_WALL, BC_INTERIOR


comptime P = 4
comptime NX = 4
comptime NY = 12
comptime LX = 1.0
comptime LY = 2.0

comptime GAMMA = 1.4
comptime RHO0  = 1.0
comptime P0    = 1.0
comptime GY_NEG: Float64 = -0.1
comptime T_FINAL: Float64 = 1.0
comptime CFL = 0.2

comptime VMAX_TOL: Float64 = 1.0e-3
comptime RHO_REL_TOL: Float64 = 1.0e-4
comptime P_REL_TOL: Float64 = 5.0e-4


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    mpi.init()
    var size = mpi.world_size()
    if size > 1:
        mpi.finalize()
        print("bench_euler_hydrostatic_2d_p4: runs at np=1 only")
        return

    print("bench_euler_hydrostatic_2d_p4 (2D hydrostatic balance, gravity gate)")
    print("  P=", P, "  mesh=", NX, "x", NY, "  gy=", GY_NEG, "  T=", T_FINAL)

    comptime NP_p = num_tri_nodes_2d(P)
    comptime NFP_e = num_edge_nodes(P)
    comptime NC = 4
    var ctx = DeviceContext()

    # x-periodic, y-slip-wall (the stratification breaks y-periodicity).
    var bcs = BoundaryConditions2D(BC_INTERIOR, BC_INTERIOR, BC_WALL, BC_WALL)

    var host_mesh = LocalMesh2D[P](Nx=NX, Ny=NY, Lx=LX, Ly=LY, bcs=bcs)
    var host_re = ReferenceElement2D[P]()
    var mesh_coords = LocalMesh2D[P](Nx=NX, Ny=NY, Lx=LX, Ly=LY, bcs=bcs)
    var gpu_mesh = LocalMesh2DGpu[P](ctx=ctx, host=host_mesh^)
    var gpu_re = ReferenceElement2DGpu[P](ctx=ctx, host=host_re)

    # Hydrostatic IC: p(y) = P0 + RHO0 * GY_NEG * y, u = v = 0.
    var n_q = gpu_mesh.num_elements * NP_p * NC
    var host_q = List[Float32]()
    var host_p = List[Float64]()
    var host_y = List[Float64]()
    for elem in range(gpu_mesh.num_elements):
        for nn in range(NP_p):
            var y = Float64(mesh_coords.elem_node_xyz[(elem * NP_p + nn) * 2 + 1])
            var p_y = P0 + RHO0 * GY_NEG * y
            var rho = RHO0
            var E = p_y / (GAMMA - 1.0)   # u = v = 0 so KE = 0
            host_q.append(Float32(rho))
            host_q.append(Float32(0.0))
            host_q.append(Float32(0.0))
            host_q.append(Float32(E))
            host_p.append(p_y)
            host_y.append(y)

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

    # CFL on sound speed (no advection).
    var cs = sqrt(GAMMA * P0 / RHO0)
    var h_cell = LX / Float64(NX)
    var dt_est = CFL * h_cell / (cs * Float64(2 * P + 1))
    var num_steps = Int(T_FINAL / dt_est) + 1
    var dt = Float32(T_FINAL / Float64(num_steps))
    print("  cs=", cs, "  steps=", num_steps, "  dt=", dt)

    var gamma_f = Float32(GAMMA)
    var min_rho = Float32(1.0e-6)
    var min_p   = Float32(1.0e-6)
    var gx_f = Float32(0.0)
    var gy_f = Float32(GY_NEG)

    var stage_plans = ssprk3_stage_plans(
        d_q=d_q.unsafe_ptr(),
        d_q1=d_q1.unsafe_ptr(),
        d_q2=d_q2.unsafe_ptr(),
    )
    for _ in range(num_steps):
        for stage in stage_plans:
            euler_rk_stage_hllc_2d[P](
                ctx=ctx,
                mesh=gpu_mesh,
                Lift_ref=gpu_re.d_Lift_ref.unsafe_ptr(),
                D_ref=gpu_re.d_D_ref.unsafe_ptr(),
                q_in=stage.q_in,
                q_a=stage.q_a,
                q_b=stage.q_b,
                q_out=stage.q_out,
                fstar_scratch=d_fstar.unsafe_ptr(),
                gamma=gamma_f,
                min_density=min_rho,
                min_pressure=min_p,
                a=stage.a, b=stage.b, cc=stage.c, dt=dt,
                gx=gx_f, gy=gy_f,
            )
    ctx.synchronize()
    ctx.enqueue_copy(hbuf_q, d_q)
    ctx.synchronize()

    var max_v: Float64 = 0.0
    var max_rho_dev: Float64 = 0.0
    var max_p_dev: Float64 = 0.0
    var n_nodes = gpu_mesh.num_elements * NP_p
    for i in range(n_nodes):
        var rho_now = hptr_q[i * NC + 0]
        var rhou    = hptr_q[i * NC + 1]
        var rhov    = hptr_q[i * NC + 2]
        var E_now   = hptr_q[i * NC + 3]
        if isnan(rho_now) or isinf(rho_now):
            raise Error("bench_euler_hydrostatic_2d_p4: non-finite output")
        var u = rhou / rho_now
        var v = rhov / rho_now
        var v_mag = sqrt(Float64(u) * Float64(u) + Float64(v) * Float64(v))
        if v_mag > max_v: max_v = v_mag
        var rho_dev = Float64(rho_now) - RHO0
        if rho_dev < 0.0: rho_dev = -rho_dev
        var rho_dev_rel = rho_dev / RHO0
        if rho_dev_rel > max_rho_dev: max_rho_dev = rho_dev_rel
        var ke = Float32(0.5) * rho_now * (u * u + v * v)
        var p_now = (GAMMA - 1.0) * Float64(E_now - ke)
        var p_dev = p_now - host_p[i]
        if p_dev < 0.0: p_dev = -p_dev
        var p_dev_rel = p_dev / P0
        if p_dev_rel > max_p_dev: max_p_dev = p_dev_rel

    print("  max |v_mag|       =", max_v,
          "  (threshold", VMAX_TOL, ")")
    print("  max |rho-rho0|/rho0 =", max_rho_dev,
          "  (threshold", RHO_REL_TOL, ")")
    print("  max |p-p_exact|/p0  =", max_p_dev,
          "  (threshold", P_REL_TOL, ")")

    if max_v > VMAX_TOL:
        raise Error(
            String("bench_euler_hydrostatic_2d_p4 FAILED: |v_mag| ")
            + String(max_v) + " > " + String(VMAX_TOL)
        )
    if max_rho_dev > RHO_REL_TOL:
        raise Error(
            String("bench_euler_hydrostatic_2d_p4 FAILED: density drift ")
            + String(max_rho_dev) + " > " + String(RHO_REL_TOL)
        )
    if max_p_dev > P_REL_TOL:
        raise Error(
            String("bench_euler_hydrostatic_2d_p4 FAILED: pressure drift ")
            + String(max_p_dev) + " > " + String(P_REL_TOL)
        )

    print("=== bench_euler_hydrostatic_2d_p4 PASSED ===")
    mpi.finalize()
