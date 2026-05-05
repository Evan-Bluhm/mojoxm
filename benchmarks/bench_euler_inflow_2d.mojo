# ======================================================================
# bench_euler_inflow_2d -- 2D Euler BC_INFLOW preservation gate
# ======================================================================
#
# Closes a real coverage gap.  The 2D Euler face-flux kernels
# (`launch_euler_face_flux_2d` and `launch_euler_face_flux_hllc_2d`
# in src/local_mesh_2d_gpu_euler.mojo) accept BC_INFLOW with a
# prescribed (inflow_rho, inflow_rhou, inflow_rhov, inflow_E) ghost
# state, but every existing 2D Euler bench/test uses periodic or
# wall BCs and never exercises the BC_INFLOW branch.  A regression
# in the 2D Euler BC_INFLOW path -- in either the Rusanov or HLLC
# face-flux kernel -- would not have tripped any gate.
#
# Mirrors `bench_shallow_water_inflow_2d`: matched-state inflow on
# the -x face, BC_OUTFLOW on the +x face, periodic in y.  The IC
# matches the inflow ghost exactly so the analytic solution is the
# IC, unchanged for all time.  Any drift signals a BC bug.
#
# Setup: rho0 = 1.0, u0 = 0.5, p0 = 1.0  (Mach number 0.42 < 1, so
# subsonic -- BC_OUTFLOW is well-posed).  HLLC flux (the more-used
# real-world choice; the Rusanov path shares the BC ghost helper
# so its inflow handling is structurally identical).
#
# Pass criteria (P=2, NX=32 NY=4, T=1):
#   * max |rho - rho0| / rho0 < 2e-3
#   * max |rhou - rho0*u0| / |rho0*u0| < 2e-3
#   * max |rhov| < 1e-3
#   * max |E - E0| / E0 < 2e-3
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
from src.boundary import (
    BoundaryConditions2D, BC_INFLOW, BC_OUTFLOW, BC_INTERIOR,
)


comptime P = 2
comptime NX = 32
comptime NY = 4
comptime LX = 1.0
comptime LY = Float64(NY) / Float64(NX) * LX

comptime GAMMA:   Float32 = 1.4
comptime RHO0:    Float32 = 1.0
comptime U0:      Float32 = 0.5
comptime P0:      Float32 = 1.0
comptime MIN_RHO: Float32 = 1.0e-6
comptime MIN_P:   Float32 = 1.0e-6
comptime CFL: Float64 = 0.15
comptime T_FINAL: Float64 = 1.0

# Empirical: state drift sits at the Float32 floor on this uniform
# IC (~few * 1e-5).  2e-3 leaves ample margin for any meaningful
# BC-coupling regression while staying well above the noise floor.
comptime RHO_REL_TOL:  Float64 = 2.0e-3
comptime RHOU_REL_TOL: Float64 = 2.0e-3
comptime RHOV_TOL:     Float64 = 1.0e-3
comptime E_REL_TOL:    Float64 = 2.0e-3


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    mpi.init()
    var size = mpi.world_size()
    if size > 1:
        mpi.finalize()
        print("bench_euler_inflow_2d: runs at np=1 only")
        return

    print("bench_euler_inflow_2d (BC_INFLOW preservation gate, HLLC)")
    print("  P=", P, "  mesh=", NX, "x", NY,
          "   M=", U0 / sqrt(GAMMA * P0 / RHO0),
          "   T=", T_FINAL)

    comptime NP_p = num_tri_nodes_2d(P)
    comptime NFP_e = num_edge_nodes(P)
    comptime NC = 4
    var ctx = DeviceContext()

    # Periodic in y (BC_INTERIOR), inflow on -x, outflow on +x.
    var bcs = BoundaryConditions2D(
        BC_INFLOW, BC_OUTFLOW,         # -x, +x
        BC_INTERIOR, BC_INTERIOR,      # -y, +y (periodic)
    )
    var host_mesh = LocalMesh2D[P](Nx=NX, Ny=NY, Lx=LX, Ly=LY, bcs=bcs)
    var host_re = ReferenceElement2D[P]()
    var gpu_mesh = LocalMesh2DGpu[P](ctx=ctx, host=host_mesh^)
    var gpu_re = ReferenceElement2DGpu[P](ctx=ctx, host=host_re)

    # IC: uniform rho = RHO0, rhou = RHO0*U0, rhov = 0, E = E0.
    # Matches the inflow ghost exactly so the analytic solution is
    # the IC for all time.
    var E0 = P0 / (GAMMA - Float32(1.0)) + Float32(0.5) * RHO0 * U0 * U0
    var n_q = gpu_mesh.num_elements * NP_p * NC
    var host_q = List[Float32]()
    for _ in range(gpu_mesh.num_elements * NP_p):
        host_q.append(RHO0)
        host_q.append(RHO0 * U0)
        host_q.append(Float32(0.0))
        host_q.append(E0)

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

    # CFL on c + |u|.
    var c = sqrt(GAMMA * P0 / RHO0)
    var wave = c + U0
    var h_cell = LX / Float64(NX)
    var dt_est = CFL * Float64(h_cell) / (Float64(wave) * Float64(2 * P + 1))
    var num_steps = Int(T_FINAL / dt_est) + 1
    var dt = Float32(T_FINAL / Float64(num_steps))
    print("  steps=", num_steps, "  dt=", dt)

    # Inflow ghost state passed to every RK stage.
    var inflow_rho  = RHO0
    var inflow_rhou = RHO0 * U0
    var inflow_rhov = Float32(0.0)
    var inflow_E    = E0

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
                gamma=GAMMA, min_density=MIN_RHO, min_pressure=MIN_P,
                a=stage.a, b=stage.b, cc=stage.c, dt=dt,
                inflow_rho=inflow_rho, inflow_rhou=inflow_rhou,
                inflow_rhov=inflow_rhov, inflow_E=inflow_E,
            )
    ctx.synchronize()
    ctx.enqueue_copy(hbuf_q, d_q)
    ctx.synchronize()

    var max_rho_dev: Float64 = 0.0
    var max_rhou_dev: Float64 = 0.0
    var max_rhov: Float64 = 0.0
    var max_E_dev: Float64 = 0.0
    var n_owned_nodes = gpu_mesh.num_elements * NP_p
    for i in range(n_owned_nodes):
        var rho  = hptr_q[i * 4 + 0]
        var rhou = hptr_q[i * 4 + 1]
        var rhov = hptr_q[i * 4 + 2]
        var E    = hptr_q[i * 4 + 3]
        if (isnan(rho) or isinf(rho) or isnan(rhou) or isinf(rhou)
            or isnan(rhov) or isinf(rhov) or isnan(E) or isinf(E)):
            raise Error("bench_euler_inflow_2d: non-finite output")
        var drho = Float64(rho - RHO0)
        if drho < 0.0: drho = -drho
        if drho > max_rho_dev: max_rho_dev = drho
        var drhou = Float64(rhou - RHO0 * U0)
        if drhou < 0.0: drhou = -drhou
        if drhou > max_rhou_dev: max_rhou_dev = drhou
        var arhov = Float64(rhov)
        if arhov < 0.0: arhov = -arhov
        if arhov > max_rhov: max_rhov = arhov
        var dE = Float64(E - E0)
        if dE < 0.0: dE = -dE
        if dE > max_E_dev: max_E_dev = dE

    var rho_rel  = max_rho_dev  / Float64(RHO0)
    var rhou_rel = max_rhou_dev / Float64(RHO0 * U0)
    var E_rel    = max_E_dev    / Float64(E0)
    print("  max |rho - rho0| / rho0       =", rho_rel,  "  (threshold", RHO_REL_TOL,  ")")
    print("  max |rhou - rho0*u0|/rho0*u0  =", rhou_rel, "  (threshold", RHOU_REL_TOL, ")")
    print("  max |rhov|                    =", max_rhov, "  (threshold", RHOV_TOL,     ")")
    print("  max |E - E0| / E0             =", E_rel,    "  (threshold", E_REL_TOL,    ")")

    if rho_rel > RHO_REL_TOL:
        raise Error(
            "bench_euler_inflow_2d FAILED: rho rel err "
            + String(rho_rel) + " > " + String(RHO_REL_TOL)
        )
    if rhou_rel > RHOU_REL_TOL:
        raise Error(
            "bench_euler_inflow_2d FAILED: rhou rel err "
            + String(rhou_rel) + " > " + String(RHOU_REL_TOL)
        )
    if max_rhov > RHOV_TOL:
        raise Error(
            "bench_euler_inflow_2d FAILED: max |rhov| "
            + String(max_rhov) + " > " + String(RHOV_TOL)
        )
    if E_rel > E_REL_TOL:
        raise Error(
            "bench_euler_inflow_2d FAILED: E rel err "
            + String(E_rel) + " > " + String(E_REL_TOL)
        )

    print("=== bench_euler_inflow_2d PASSED ===")
    mpi.finalize()
