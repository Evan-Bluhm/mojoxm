# ======================================================================
# bench_mhd_glm_psi_damp_2d_p3 -- 2D GLM operator-splitting decay at P=3
# ======================================================================
#
# P=3 (NP=10) counterpart of bench_mhd_glm_psi_damp_2d.  Same uniform
# rest IC + uniform psi=A0, same alpha_d>0 / c_h=0 setup, same
# operator-splitting damp kernel applied once per timestep, but
# routed through LocalMesh2D[3] / mhd_glm_rk_stage_2d[3] /
# launch_mhd_glm_psi_damp_2d[NP=10] so the NC=7 GLM stack is
# exercised at P=3.
#
# Closes the last 2D GLM P=3 gap: the existing 2D GLM coverage at
# P=3 (alfven_glm_2d_p3, glm_psi_transport_2d_p3) tests the flux-
# coupled paths but not the operator-splitting damp kernel itself.
# This bench gates `launch_mhd_glm_psi_damp_2d[NP=10]` directly --
# a different kernel from the 3D source-term path (already covered
# by bench_mhd_glm_psi_damp_3d_p3 in the previous iteration).
#
# Pass criteria (P=3, sweep N=8, 12, T=1, alpha_d=1, A0=0.1):
#   * |psi(T) - A0/e| / A0 < 5e-5 at every owned node (~Float32 floor;
#     same threshold as P=2 because the splitting decay is exact each
#     step regardless of P -- the only P-dependent error is the
#     accumulated Float32 round-off from the SSPRK3 stages doing
#     nothing on the constant state)
#   * non-psi state unchanged from IC to ~Float32 epsilon
#   * no NaN / Inf
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


comptime P = 3
comptime LX = 1.0
comptime LY = 1.0

comptime GAMMA: Float64 = 5.0 / 3.0
comptime RHO0:  Float64 = 1.0
comptime P0:    Float64 = 1.0
comptime A0:    Float64 = 0.1
comptime C_H:   Float64 = 0.0
comptime ALPHA_D: Float64 = 1.0
comptime T_FINAL: Float64 = 1.0

comptime CFL: Float64 = 0.15

comptime PSI_TOL_REL: Float64 = 5.0e-5
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
    var rho0_f = Float32(RHO0)
    var E0 = Float32(P0 / (GAMMA - 1.0))
    var A0_f = Float32(A0)
    for _ in range(gpu_mesh.num_elements * NP_p):
        host_q.append(rho0_f)
        host_q.append(Float32(0.0))
        host_q.append(Float32(0.0))
        host_q.append(Float32(0.0))
        host_q.append(Float32(0.0))
        host_q.append(E0)
        host_q.append(A0_f)

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

    var c = 1.0
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
        mhd_glm_rk_stage_2d[P](
            ctx, gpu_mesh,
            gpu_re.d_Lift_ref.unsafe_ptr(), gpu_re.d_D_ref.unsafe_ptr(),
            d_q.unsafe_ptr(),
            d_q.unsafe_ptr(), d_q.unsafe_ptr(),
            d_q1.unsafe_ptr(),
            d_fstar.unsafe_ptr(),
            gamma_f, min_rho, min_p, c_h_f,
            Float32(1.0), Float32(0.0), Float32(1.0), dt,
        )
        mhd_glm_rk_stage_2d[P](
            ctx, gpu_mesh,
            gpu_re.d_Lift_ref.unsafe_ptr(), gpu_re.d_D_ref.unsafe_ptr(),
            d_q1.unsafe_ptr(),
            d_q.unsafe_ptr(), d_q1.unsafe_ptr(),
            d_q2.unsafe_ptr(),
            d_fstar.unsafe_ptr(),
            gamma_f, min_rho, min_p, c_h_f,
            Float32(0.75), Float32(0.25), Float32(0.25), dt,
        )
        mhd_glm_rk_stage_2d[P](
            ctx, gpu_mesh,
            gpu_re.d_Lift_ref.unsafe_ptr(), gpu_re.d_D_ref.unsafe_ptr(),
            d_q2.unsafe_ptr(),
            d_q.unsafe_ptr(), d_q2.unsafe_ptr(),
            d_q.unsafe_ptr(),
            d_fstar.unsafe_ptr(),
            gamma_f, min_rho, min_p, c_h_f,
            Float32(1.0 / 3.0), Float32(2.0 / 3.0),
            Float32(2.0 / 3.0), dt,
        )
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
            raise Error("bench_mhd_glm_psi_damp_2d_p3: non-finite psi")
        var d = Float64(psi) - psi_expect
        if d < 0.0: d = -d
        if d > max_psi_err: max_psi_err = d
        var d_rho = hptr_q[base + 0] - rho0_f
        if d_rho < 0.0: d_rho = -d_rho
        if d_rho > max_state_err: max_state_err = d_rho
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
            String("bench_mhd_glm_psi_damp_2d_p3 FAILED at NX=") + String(NX)
            + ": psi rel err " + String(rel_err)
            + " > tol " + String(PSI_TOL_REL)
        )
    if max_state_err > STATE_TOL:
        raise Error(
            String("bench_mhd_glm_psi_damp_2d_p3 FAILED at NX=") + String(NX)
            + ": non-psi state drifted by " + String(max_state_err)
        )
    return True


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    mpi.init()
    print("bench_mhd_glm_psi_damp_2d_p3 (GLM operator-splitting decay at P=3)")
    print("  P=", P, "  alpha_d=", ALPHA_D, "  T=", T_FINAL,
          "  expected psi/A0 = exp(-alpha_d*T) = ",
          exp(-Float64(ALPHA_D) * Float64(T_FINAL)))

    _ = _run(8)
    _ = _run(12)

    print("=== bench_mhd_glm_psi_damp_2d_p3 PASSED ===")
    mpi.finalize()
