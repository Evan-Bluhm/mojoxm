# ======================================================================
# bench_euler_channel_steady_2d -- non-periodic BC steady-state gate
# ======================================================================
#
# Mach-2 supersonic channel in a [0, 1] x [0, 0.25] box with three
# kinds of non-periodic boundaries:
#   -x: BC_INFLOW  (Dirichlet on the supersonic inflow state)
#   +x: BC_OUTFLOW (zero-gradient ghost; supersonic outflow)
#   y:  BC_WALL    (slip-wall reflection of normal momentum)
#
# IC = the inflow state filled across the whole domain.  This is an
# exact steady solution of Euler: zero gradients, the inflow state
# is constant, and supersonic flow guarantees no information
# propagates upstream from +x.  Any drift over T = 0.20 is pure
# numerical error in the BC kernels or HLLC.
#
# Pass criteria:
#   * max |rho - rho_inflow| < 1.0e-5 anywhere in the domain
#   * max |mx - rhou_inflow| < 1.0e-5
#   * no NaN / Inf
#
# This is the tightest gate in the harness -- the analytic solution
# is exact (constant), so the only error sources are the GPU's
# Float32 round-off in the volume / face-flux / lift / rk-update
# kernels and any subtle BC plumbing bug that breaks symmetry.  The
# existing channel driver reports rho_max_drift = 1.2e-7 (Float32
# epsilon); a 1e-5 threshold is two orders looser to absorb harmless
# Float32 jitter while still catching any real BC regression.
# ======================================================================

from std.math import sqrt, isnan, isinf
from std.sys import has_accelerator
from std.gpu.host import DeviceContext, DeviceBuffer
from src import mpi
from src.local_mesh_2d import LocalMesh2D
from src.local_mesh_2d_gpu import LocalMesh2DGpu
from src.local_mesh_2d_gpu_euler import euler_rk_stage_2d
from src.reference_2d import (
    ReferenceElement2D, num_tri_nodes_2d, num_edge_nodes,
)
from src.reference_2d_gpu import ReferenceElement2DGpu
from src.boundary import (
    BoundaryConditions2D, BC_WALL, BC_OUTFLOW, BC_INFLOW,
)


comptime P = 2
comptime NX = 64
comptime NY = 16
comptime LX = 1.0
comptime LY = 0.25
comptime GAMMA     = 1.4
comptime RHO_0     = 1.0
comptime P_0       = 1.0
comptime MACH      = 2.0
comptime T_FINAL   = 0.20
comptime CFL       = 0.15

comptime DRIFT_TOL: Float64 = 1.0e-5


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    mpi.init()
    var size = mpi.world_size()
    if size > 1:
        mpi.finalize()
        print("bench_euler_channel_steady_2d: runs at np=1 only")
        return

    print("bench_euler_channel_steady_2d (Mach-2 channel steady state)")
    print("  P=", P, "  mesh=", NX, "x", NY,
          "  BC: -x INFLOW, +x OUTFLOW, y WALL")

    comptime NP_p = num_tri_nodes_2d(P)
    comptime NFP_e = num_edge_nodes(P)
    comptime NC = 4
    var ctx = DeviceContext()

    var c_inf = sqrt(GAMMA * P_0 / RHO_0)
    var u_inf = MACH * c_inf
    var rhou_inf = RHO_0 * u_inf
    var E_inf = P_0 / (GAMMA - 1.0) + 0.5 * RHO_0 * u_inf * u_inf

    var bcs = BoundaryConditions2D(
        BC_INFLOW, BC_OUTFLOW, BC_WALL, BC_WALL,
    )
    var host_mesh = LocalMesh2D[P](NX, NY, LX, LY, bcs)
    var host_re = ReferenceElement2D[P]()
    var gpu_mesh = LocalMesh2DGpu[P](ctx, host_mesh^)
    var gpu_re = ReferenceElement2DGpu[P](ctx, host_re)

    var n_q = gpu_mesh.num_elements * NP_p * NC
    var host_q = List[Float32]()
    for _ in range(gpu_mesh.num_elements * NP_p):
        host_q.append(Float32(RHO_0))
        host_q.append(Float32(rhou_inf))
        host_q.append(Float32(0.0))
        host_q.append(Float32(E_inf))

    var d_q  = ctx.enqueue_create_buffer[DType.float32](n_q)
    var d_q1 = ctx.enqueue_create_buffer[DType.float32](n_q)
    var d_q2 = ctx.enqueue_create_buffer[DType.float32](n_q)
    var d_vol = ctx.enqueue_create_buffer[DType.float32](n_q)
    var d_rhs = ctx.enqueue_create_buffer[DType.float32](n_q)
    var d_fstar = ctx.enqueue_create_buffer[DType.float32](
        gpu_mesh.num_faces * NFP_e * NC
    )
    var hbuf_q = ctx.enqueue_create_host_buffer[DType.float32](n_q)
    var hptr_q = hbuf_q.unsafe_ptr()
    for k in range(n_q):
        hptr_q[k] = host_q[k]
    ctx.enqueue_copy(d_q, hbuf_q)
    ctx.synchronize()

    var h_cell = LX / Float64(NX)
    var wave_max = u_inf + c_inf
    var dt_est = CFL * h_cell / (wave_max * Float64(2 * P + 1))
    var num_steps = Int(T_FINAL / dt_est) + 1
    var dt = Float32(T_FINAL / Float64(num_steps))

    var gamma = Float32(GAMMA)
    var min_rho = Float32(1.0e-6)
    var min_p   = Float32(1.0e-6)
    var inflow_rho  = Float32(RHO_0)
    var inflow_rhou = Float32(rhou_inf)
    var inflow_rhov = Float32(0.0)
    var inflow_E    = Float32(E_inf)

    for _ in range(num_steps):
        euler_rk_stage_2d[P](
            ctx, gpu_mesh,
            gpu_re.d_Lift_ref.unsafe_ptr(), gpu_re.d_D_ref.unsafe_ptr(),
            d_q.unsafe_ptr(),
            d_q.unsafe_ptr(), d_q.unsafe_ptr(),
            d_q1.unsafe_ptr(),
            d_vol.unsafe_ptr(), d_fstar.unsafe_ptr(), d_rhs.unsafe_ptr(),
            gamma, min_rho, min_p,
            inflow_rho, inflow_rhou, inflow_rhov, inflow_E,
            Float32(1.0), Float32(0.0), Float32(1.0), dt,
        )
        euler_rk_stage_2d[P](
            ctx, gpu_mesh,
            gpu_re.d_Lift_ref.unsafe_ptr(), gpu_re.d_D_ref.unsafe_ptr(),
            d_q1.unsafe_ptr(),
            d_q.unsafe_ptr(), d_q1.unsafe_ptr(),
            d_q2.unsafe_ptr(),
            d_vol.unsafe_ptr(), d_fstar.unsafe_ptr(), d_rhs.unsafe_ptr(),
            gamma, min_rho, min_p,
            inflow_rho, inflow_rhou, inflow_rhov, inflow_E,
            Float32(0.75), Float32(0.25), Float32(0.25), dt,
        )
        euler_rk_stage_2d[P](
            ctx, gpu_mesh,
            gpu_re.d_Lift_ref.unsafe_ptr(), gpu_re.d_D_ref.unsafe_ptr(),
            d_q2.unsafe_ptr(),
            d_q.unsafe_ptr(), d_q2.unsafe_ptr(),
            d_q.unsafe_ptr(),
            d_vol.unsafe_ptr(), d_fstar.unsafe_ptr(), d_rhs.unsafe_ptr(),
            gamma, min_rho, min_p,
            inflow_rho, inflow_rhou, inflow_rhov, inflow_E,
            Float32(1.0 / 3.0), Float32(2.0 / 3.0),
            Float32(2.0 / 3.0), dt,
        )
    ctx.synchronize()

    ctx.enqueue_copy(hbuf_q, d_q)
    ctx.synchronize()

    var rho_drift_max: Float64 = 0.0
    var rhou_drift_max: Float64 = 0.0
    for elem in range(gpu_mesh.num_elements):
        for nn in range(NP_p):
            var base = (elem * NP_p + nn) * NC
            var rho = hptr_q[base + 0]
            var mx  = hptr_q[base + 1]
            if isnan(rho) or isinf(rho) or isnan(mx) or isinf(mx):
                raise Error("bench_euler_channel_steady_2d: non-finite output")
            var d_rho = Float64(rho) - RHO_0
            if d_rho < 0.0: d_rho = -d_rho
            if d_rho > rho_drift_max: rho_drift_max = d_rho
            var d_mx = Float64(mx) - rhou_inf
            if d_mx < 0.0: d_mx = -d_mx
            if d_mx > rhou_drift_max: rhou_drift_max = d_mx

    print("  steps=", num_steps, "  dt=", dt)
    print("  max |rho - rho_inflow|  =", rho_drift_max,
          " (tol", DRIFT_TOL, ")")
    print("  max |mx - rhou_inflow|  =", rhou_drift_max,
          " (tol", DRIFT_TOL, ")")

    if rho_drift_max > DRIFT_TOL:
        raise Error(
            String("bench_euler_channel_steady_2d FAILED: rho_drift ")
            + String(rho_drift_max)
            + " exceeds " + String(DRIFT_TOL)
        )
    if rhou_drift_max > DRIFT_TOL:
        raise Error(
            String("bench_euler_channel_steady_2d FAILED: rhou_drift ")
            + String(rhou_drift_max)
            + " exceeds " + String(DRIFT_TOL)
        )

    print("=== bench_euler_channel_steady_2d PASSED ===")
    mpi.finalize()
