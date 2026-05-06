# ======================================================================
# bench_mhd_inflow_2d_glm -- first BC gate for 2D GLM-MHD
# ======================================================================
#
# Closes a real coverage + feature gap.  Until this iteration, the
# 2D GLM-MHD face-flux kernel had no BC_INFLOW path -- only
# BC_INTERIOR + BC_WALL with the else branch acting as zero-gradient
# outflow.  The README's "full BC menu (periodic / wall / outflow /
# inflow)" claim was inaccurate for 2D MHD.  This iteration adds the
# BC_INFLOW path to mhd_glm_face_flux_kernel_2d (NC=7 ghost) and
# closes the loop with this bench.
#
# Setup mirrors bench_mhd_inflow_3d: matched-state subsonic inflow
# on -x with BC_OUTFLOW on +x and periodic in y.  IC = inflow ghost
# so the analytic solution is the IC unchanged.  GLM is disabled
# (c_h=0) per project memory `project_glm_bc_coupling.md` -- the
# Dedner GLM source path subtly couples to BC_OUTFLOW at sub-Alfvenic
# flow leaving ~1% drift on E even from psi=0.  Disabling GLM
# isolates the BC path.
#
# Pass criteria (P=2, NX=32 NY=4, T=1):
#   * max relative drift in any of the 7 components < 5e-3
#   * no NaN / Inf
# ======================================================================

from std.math import sqrt, isnan, isinf
from std.sys import has_accelerator
from std.gpu.host import DeviceContext, DeviceBuffer
from src import mpi
from src.local_mesh_2d import LocalMesh2D
from src.local_mesh_2d_gpu import LocalMesh2DGpu
from src.local_mesh_2d_gpu_mhd_glm import mhd_glm_rk_stage_2d
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

comptime GAMMA: Float32 = Float32(5.0 / 3.0)
comptime RHO0: Float32 = 1.0
comptime U0: Float32 = 0.5
comptime B0: Float32 = 1.0
comptime P0: Float32 = 0.1
comptime C_H: Float32 = 0.0  # GLM disabled; psi stays 0 trivially
comptime MIN_RHO: Float32 = 1.0e-6
comptime MIN_P: Float32 = 1.0e-6
comptime CFL: Float64 = 0.15
comptime T_FINAL: Float64 = 1.0

comptime REL_TOL: Float64 = 5.0e-3


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    mpi.init()
    var size = mpi.world_size()
    if size > 1:
        mpi.finalize()
        print("bench_mhd_inflow_2d_glm: runs at np=1 only")
        return

    print("bench_mhd_inflow_2d_glm (BC_INFLOW preservation, GLM disabled)")
    print(
        "  P=",
        P,
        "  mesh=",
        NX,
        "x",
        NY,
        "   U0=",
        U0,
        "   B0=",
        B0,
        "   T=",
        T_FINAL,
    )

    comptime NP_p = num_tri_nodes_2d(P)
    comptime NFP_e = num_edge_nodes(P)
    comptime NC = 7
    var ctx = DeviceContext()

    # Inflow on -x, outflow on +x, periodic in y.
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

    # IC: uniform (rho, rhou, rhov, Bx, By, E, psi) matched to the
    # inflow ghost.  E = p/(g-1) + 0.5*rho*|u|^2 + 0.5*|B|^2.
    var ke = Float32(0.5) * RHO0 * (U0 * U0)
    var pe = Float32(0.5) * (B0 * B0)
    var E0 = P0 / (GAMMA - Float32(1.0)) + ke + pe

    var n_q = gpu_mesh.num_elements * NP_p * NC
    var host_q = List[Float32]()
    for _ in range(gpu_mesh.num_elements * NP_p):
        host_q.append(RHO0)
        host_q.append(RHO0 * U0)
        host_q.append(Float32(0.0))
        host_q.append(B0)
        host_q.append(Float32(0.0))
        host_q.append(E0)
        host_q.append(Float32(0.0))

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

    # CFL on fast-magnetosonic + |u| with B aligned to flow.
    var cf = sqrt(GAMMA * P0 / RHO0 + B0 * B0 / RHO0)
    var wave = cf + U0
    var h_cell = LX / Float64(NX)
    var dt_est = CFL * Float64(h_cell) / (Float64(wave) * Float64(2 * P + 1))
    var num_steps = Int(T_FINAL / dt_est) + 1
    var dt = Float32(T_FINAL / Float64(num_steps))
    print("  c_f=", cf, "  steps=", num_steps, "  dt=", dt)

    var stage_plans = ssprk3_stage_plans(
        d_q=d_q.unsafe_ptr(),
        d_q1=d_q1.unsafe_ptr(),
        d_q2=d_q2.unsafe_ptr(),
    )
    for _ in range(num_steps):
        for stage in stage_plans:
            mhd_glm_rk_stage_2d[P](
                ctx=ctx,
                mesh=gpu_mesh,
                Lift_ref=gpu_re.d_Lift_ref.unsafe_ptr(),
                D_ref=gpu_re.d_D_ref.unsafe_ptr(),
                q_in=stage.q_in,
                q_a=stage.q_a,
                q_b=stage.q_b,
                q_out=stage.q_out,
                fstar_scratch=d_fstar.unsafe_ptr(),
                gamma=GAMMA,
                min_density=MIN_RHO,
                min_pressure=MIN_P,
                c_h=C_H,
                a=stage.a,
                b=stage.b,
                cc=stage.c,
                dt=dt,
                inflow_rho=RHO0,
                inflow_rhou=RHO0 * U0,
                inflow_rhov=Float32(0.0),
                inflow_Bx=B0,
                inflow_By=Float32(0.0),
                inflow_E=E0,
                inflow_psi=Float32(0.0),
            )
    ctx.synchronize()
    ctx.enqueue_copy(hbuf_q, d_q)
    ctx.synchronize()

    var ref_scale = Float64(E0)
    var max_drift: Float64 = 0.0
    var n_owned_nodes = gpu_mesh.num_elements * NP_p
    for i in range(n_owned_nodes):
        for c in range(NC):
            var qv = hptr_q[i * NC + c]
            if isnan(qv) or isinf(qv):
                raise Error("bench_mhd_inflow_2d_glm: non-finite output")
            var qref = host_q[i * NC + c]
            var d = Float64(qv) - Float64(qref)
            if d < 0.0:
                d = -d
            var rel = d / ref_scale
            if rel > max_drift:
                max_drift = rel

    print("  max relative drift   =", max_drift, "  (threshold", REL_TOL, ")")

    if max_drift > REL_TOL:
        raise Error(
            "bench_mhd_inflow_2d_glm FAILED: max relative drift "
            + String(max_drift)
            + " > "
            + String(REL_TOL)
        )

    print("=== bench_mhd_inflow_2d_glm PASSED ===")
    mpi.finalize()
