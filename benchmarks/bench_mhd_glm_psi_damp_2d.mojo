# ======================================================================
# bench_mhd_glm_psi_damp_2d -- pure GLM psi damping (no transport)
# ======================================================================
#
# Validates the GLM operator-splitting psi damping in the 2D MHD path.
# Background: rho=1, p=1, u=v=0, B=0; psi = A0 uniformly.  With
# c_h = 0 (no GLM transport) the only psi dynamics is the splitting
# decay
#     dpsi/dt = -alpha_d * psi
# applied once per timestep via launch_mhd_glm_psi_damp_2d.  The
# analytic solution is
#     psi(T) = A0 * exp(-alpha_d * T)
# (uniform in space because the IC was uniform and there is no flux).
#
# Pass criteria (P=2, sweep N = 8, 12, T = 1, alpha_d = 1.0,
# c_h = 0, A0 = 0.1):
#   * |psi(T) - A0/e| / A0 < 1e-3 at every owned node
#   * non-psi state unchanged from IC (rho, momenta, B, E)
#   * no NaN / Inf
#
# This benchmark also serves as a regression gate for the 2026-04-25
# damping-over-application fix.  Before that commit, mhd_glm_rk_stage_2d
# ran the damp kernel inside every SSPRK3 stage with full dt, so per-
# step decay was exp(-3*alpha_d*dt) instead of exp(-alpha_d*dt) -- the
# observed psi here would have been A0 * exp(-3) ~= 0.05 * A0 instead
# of A0/e ~= 0.37 * A0, far outside the tolerance.
# ======================================================================

from std.math import sqrt, exp, isnan, isinf
from std.sys import has_accelerator
from std.gpu.host import DeviceContext, DeviceBuffer
from src import mpi
from src.local_mesh_2d import LocalMesh2D
from src.local_mesh_2d_gpu import LocalMesh2DGpu
from src.local_mesh_2d_gpu_mhd_glm import (
    mhd_glm_rk_stage_2d, launch_mhd_glm_psi_damp_2d,
)
from src.reference_2d import (
    ReferenceElement2D, num_tri_nodes_2d, num_edge_nodes,
)
from src.reference_2d_gpu import ReferenceElement2DGpu


comptime P = 2
comptime LX = 1.0
comptime LY = 1.0

comptime GAMMA: Float64 = 5.0 / 3.0
comptime RHO0:  Float64 = 1.0
comptime P0:    Float64 = 1.0
comptime A0:    Float64 = 0.1            # psi amplitude
comptime C_H:   Float64 = 0.0            # no transport
comptime ALPHA_D: Float64 = 1.0          # damping rate
comptime T_FINAL: Float64 = 1.0          # so psi(T) = A0/e

comptime CFL: Float64 = 0.15

# Measured rel err ~4e-6 on current code; 5e-5 leaves ~10x margin.
comptime PSI_TOL_REL: Float64 = 5.0e-5
# State drift is at ~1e-5 (Float32 floor), keep at threshold = floor.
comptime STATE_TOL: Float32 = Float32(2.0e-5)


def _run(NX: Int) raises -> Bool:
    comptime NP_p = num_tri_nodes_2d(P)
    comptime NFP_e = num_edge_nodes(P)
    comptime NC = 7
    var ctx = DeviceContext()
    var NY = 4
    var LY_run = Float64(NY) / Float64(NX) * LX

    var host_mesh = LocalMesh2D[P](NX, NY, LX, LY_run)
    var host_re = ReferenceElement2D[P]()
    var gpu_mesh = LocalMesh2DGpu[P](ctx, host_mesh^)
    var gpu_re = ReferenceElement2DGpu[P](ctx, host_re)

    var n_q = gpu_mesh.num_elements * NP_p * NC
    var host_q = List[Float32]()
    # IC: rest state, uniform psi = A0.
    var rho0_f = Float32(RHO0)
    var E0 = Float32(P0 / (GAMMA - 1.0))   # u=v=0, B=0 -> no kinetic / mag
    var A0_f = Float32(A0)
    for _ in range(gpu_mesh.num_elements * NP_p):
        host_q.append(rho0_f)        # rho
        host_q.append(Float32(0.0))  # rho*u
        host_q.append(Float32(0.0))  # rho*v
        host_q.append(Float32(0.0))  # Bx
        host_q.append(Float32(0.0))  # By
        host_q.append(E0)            # E
        host_q.append(A0_f)          # psi

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

    # Time stepping: SSPRK3 + psi damp once per step.  With c_h=0 the
    # flux for psi is identically zero, so q_out for psi after each
    # stage equals the SSPRK3 combination of psi values from previous
    # stages.  After the third stage psi is unchanged from IC (no
    # flux source); then the damping kernel multiplies by exp(-alpha_d*dt).
    var c = 1.0   # advection speed irrelevant; psi has no flux
    var h_cell = LX / Float64(NX)
    var dt_est = CFL * h_cell / (c * Float64(2 * P + 1))
    var num_steps = Int(T_FINAL / dt_est) + 1
    var dt = Float32(T_FINAL / Float64(num_steps))

    var gamma_f = Float32(GAMMA)
    var min_rho = Float32(1.0e-6)
    var min_p   = Float32(1.0e-6)
    var c_h_f   = Float32(C_H)
    var alpha_d_f = Float32(ALPHA_D)

    for _ in range(num_steps):
        # SSPRK3 stages.  alpha_d on rk_stage is now ignored (post-fix);
        # damping is applied once below.
        mhd_glm_rk_stage_2d[P](
            ctx, gpu_mesh,
            gpu_re.d_Lift_ref.unsafe_ptr(), gpu_re.d_D_ref.unsafe_ptr(),
            d_q.unsafe_ptr(),
            d_q.unsafe_ptr(), d_q.unsafe_ptr(),
            d_q1.unsafe_ptr(),
            d_fstar.unsafe_ptr(),
            gamma_f, min_rho, min_p, c_h_f, alpha_d_f,
            Float32(1.0), Float32(0.0), Float32(1.0), dt,
        )
        mhd_glm_rk_stage_2d[P](
            ctx, gpu_mesh,
            gpu_re.d_Lift_ref.unsafe_ptr(), gpu_re.d_D_ref.unsafe_ptr(),
            d_q1.unsafe_ptr(),
            d_q.unsafe_ptr(), d_q1.unsafe_ptr(),
            d_q2.unsafe_ptr(),
            d_fstar.unsafe_ptr(),
            gamma_f, min_rho, min_p, c_h_f, alpha_d_f,
            Float32(0.75), Float32(0.25), Float32(0.25), dt,
        )
        mhd_glm_rk_stage_2d[P](
            ctx, gpu_mesh,
            gpu_re.d_Lift_ref.unsafe_ptr(), gpu_re.d_D_ref.unsafe_ptr(),
            d_q2.unsafe_ptr(),
            d_q.unsafe_ptr(), d_q2.unsafe_ptr(),
            d_q.unsafe_ptr(),
            d_fstar.unsafe_ptr(),
            gamma_f, min_rho, min_p, c_h_f, alpha_d_f,
            Float32(1.0 / 3.0), Float32(2.0 / 3.0),
            Float32(2.0 / 3.0), dt,
        )
        # Operator-splitting psi damping: ONCE per timestep.
        comptime NP_t = num_tri_nodes_2d(P)
        launch_mhd_glm_psi_damp_2d[NP_t](
            ctx, d_q.unsafe_ptr(), gpu_mesh.num_elements,
            alpha_d_f, dt,
        )
    ctx.synchronize()

    ctx.enqueue_copy(hbuf_q, d_q)
    ctx.synchronize()

    var psi_expect = Float64(A0) * exp(-Float64(ALPHA_D) * Float64(T_FINAL))
    var max_psi_err: Float64 = 0.0
    var max_state_err: Float32 = 0.0
    var n_nodes = gpu_mesh.num_elements * NP_p
    for i in range(n_nodes):
        var base = i * NC
        var psi = hptr_q[base + 6]
        if isnan(psi) or isinf(psi):
            raise Error("bench_mhd_glm_psi_damp_2d: non-finite psi")
        var d = Float64(psi) - psi_expect
        if d < 0.0: d = -d
        if d > max_psi_err: max_psi_err = d
        # Check that non-psi state unchanged.
        var d_rho = hptr_q[base + 0] - rho0_f
        if d_rho < 0.0: d_rho = -d_rho
        if d_rho > max_state_err: max_state_err = d_rho
        # Momenta and B should still be 0.
        for c_idx in range(1, 5):
            var v = hptr_q[base + c_idx]
            var av = v if v >= Float32(0.0) else -v
            if av > max_state_err: max_state_err = av
        var d_E = hptr_q[base + 5] - E0
        if d_E < 0.0: d_E = -d_E
        if d_E > max_state_err: max_state_err = d_E

    var rel_err = max_psi_err / Float64(A0)
    print("  N=", NX, "  steps=", num_steps,
          "  psi_expect=", psi_expect,
          "  max |psi-expect|=", max_psi_err,
          "  rel=", rel_err,
          "  max state drift=", max_state_err)
    if rel_err > PSI_TOL_REL:
        raise Error(
            String("bench_mhd_glm_psi_damp_2d FAILED at NX=") + String(NX)
            + ": psi rel err " + String(rel_err)
            + " > tol " + String(PSI_TOL_REL)
        )
    if max_state_err > STATE_TOL:
        raise Error(
            String("bench_mhd_glm_psi_damp_2d FAILED at NX=") + String(NX)
            + ": non-psi state drifted by " + String(max_state_err)
        )
    return True


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    mpi.init()
    print("bench_mhd_glm_psi_damp_2d (GLM operator-splitting decay)")
    print("  P=", P, "  alpha_d=", ALPHA_D, "  T=", T_FINAL,
          "  expected psi/A0 = exp(-alpha_d*T) = ",
          exp(-Float64(ALPHA_D) * Float64(T_FINAL)))

    _ = _run(8)
    _ = _run(12)

    print("=== bench_mhd_glm_psi_damp_2d PASSED ===")
    mpi.finalize()
