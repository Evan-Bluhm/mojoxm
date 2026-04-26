# ======================================================================
# bench_mhd_alfven_glm_2d_p3 -- Alfven via GLM 2D MHD at P=3 (NP=10)
# ======================================================================
#
# P=3 counterpart of bench_mhd_alfven_glm_2d.  Same Alfven-wave IC and
# the same GLM-with-c_h=0 setup (so the GLM path should reduce to
# plain ideal MHD), but routed through LocalMesh2D[3] /
# mhd_glm_rk_stage_2d[3] with NP=10 nodes per triangle.
#
# Pairs with bench_advection_translation_2d_p3, bench_euler_smooth_
# wave_2d_p3, bench_shallow_water_wave_2d_p3, and bench_maxwell_plane_
# wave_2d to bring 2D MHD GLM (NC=7) up to the same higher-order
# coverage parity for smooth physics.
#
# Pass criteria (P=3, NX=32, thin y-strip):
#   * rel L2(state, including psi) < 5e-3 (the IC amplitude A=0.1
#     gives a nonlinear A^2~0.01 correction over one period that sets
#     the error floor at ~3.6e-3; same floor as the P=2 gate because
#     it's set by the IC, not the scheme.  Threshold catches
#     catastrophic GLM-operator regressions; a tighter accuracy gate
#     would need a smaller-amplitude IC.)
#   * |psi|_max < 1e-5 (psi must remain ~0 since the IC has no
#     div(B) error and c_h = alpha_d = 0; this catches GLM-coupling
#     bugs that wrongly excite psi).
#   * no NaN / Inf
#
# Primary value of this bench: gates that the GLM 2D MHD kernels
# (mhd_glm_*) work correctly at NP=10, parallel to the same coverage
# we have for advection / Euler / SW / Maxwell at P=3.
# ======================================================================

from std.math import sqrt, sin, pi, isnan, isinf
from std.sys import has_accelerator
from std.gpu.host import DeviceContext, DeviceBuffer
from src import mpi
from src.local_mesh_2d import LocalMesh2D
from src.local_mesh_2d_gpu import LocalMesh2DGpu
from src.local_mesh_2d_gpu_mhd_glm import mhd_glm_rk_stage_2d
from src.reference_2d import (
    ReferenceElement2D, num_tri_nodes_2d, num_edge_nodes,
)
from src.reference_2d_gpu import ReferenceElement2DGpu


comptime P = 3
comptime NX = 32
comptime NY = 4
comptime LX = 1.0

comptime GAMMA     = 5.0 / 3.0
comptime RHO0      = 1.0
comptime B0        = 1.0
comptime P0        = 0.1
comptime AMPLITUDE = 0.1

comptime T_FINAL   = 1.0
comptime CFL       = 0.10

comptime L2_MAX_REL: Float64 = 5.0e-3
comptime PSI_TOL: Float32 = Float32(1.0e-5)


def _run() raises -> Float64:
    comptime NP_p = num_tri_nodes_2d(P)
    comptime NFP_e = num_edge_nodes(P)
    comptime NC = 7
    var ctx = DeviceContext()
    var LY = Float64(NY) / Float64(NX) * LX

    var host_mesh = LocalMesh2D[P](NX, NY, LX, LY)
    var host_re = ReferenceElement2D[P]()
    var mesh_coords = LocalMesh2D[P](NX, NY, LX, LY)
    var gpu_mesh = LocalMesh2DGpu[P](ctx, host_mesh^)
    var gpu_re = ReferenceElement2DGpu[P](ctx, host_re)

    var k_wave = 2.0 * pi / LX
    var n_q = gpu_mesh.num_elements * NP_p * NC
    var host_q = List[Float32]()
    var host_ic = List[Float32]()
    for elem in range(gpu_mesh.num_elements):
        for nn in range(NP_p):
            var x = mesh_coords.elem_node_xyz[(elem * NP_p + nn) * 2 + 0]
            var sv = sin(k_wave * x)
            var uy = AMPLITUDE * sv
            var By = -AMPLITUDE * sv
            var Bx = B0
            var mx = 0.0
            var my = RHO0 * uy
            var ke = 0.5 * RHO0 * (uy * uy)
            var mp = 0.5 * (Bx * Bx + By * By)
            var E  = P0 / (GAMMA - 1.0) + ke + mp
            host_q.append(Float32(RHO0));   host_ic.append(Float32(RHO0))
            host_q.append(Float32(mx));     host_ic.append(Float32(mx))
            host_q.append(Float32(my));     host_ic.append(Float32(my))
            host_q.append(Float32(Bx));     host_ic.append(Float32(Bx))
            host_q.append(Float32(By));     host_ic.append(Float32(By))
            host_q.append(Float32(E));      host_ic.append(Float32(E))
            host_q.append(Float32(0.0));    host_ic.append(Float32(0.0))   # psi

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

    var cs = sqrt(GAMMA * P0 / RHO0)
    var cA = B0 / sqrt(RHO0)
    var cf = sqrt(cs * cs + cA * cA)
    var h_cell = LX / Float64(NX)
    var dt_est = CFL * h_cell / (cf * Float64(2 * P + 1))
    var num_steps = Int(T_FINAL / dt_est) + 1
    var dt = Float32(T_FINAL / Float64(num_steps))

    var gamma_f = Float32(GAMMA)
    var min_rho = Float32(1.0e-6)
    var min_p   = Float32(1.0e-6)
    # GLM disabled (c_h=0, alpha_d=0): the GLM kernels should reduce
    # to plain ideal MHD on this smooth IC.
    var c_h_f = Float32(0.0)

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
    ctx.synchronize()

    ctx.enqueue_copy(hbuf_q, d_q)
    ctx.synchronize()

    var sum_sq: Float64 = 0.0
    var sum_ic: Float64 = 0.0
    var psi_max: Float32 = 0.0
    for k in range(n_q):
        var v = hptr_q[k]
        if isnan(v) or isinf(v):
            raise Error("bench_mhd_alfven_glm_2d_p3: non-finite at index "
                        + String(k))
        var err = Float64(v - host_ic[k])
        sum_sq += err * err
        var ic = Float64(host_ic[k])
        sum_ic += ic * ic
    var n_elem_nodes = gpu_mesh.num_elements * NP_p
    for i in range(n_elem_nodes):
        var p = hptr_q[i * NC + 6]
        var a = p if p >= Float32(0.0) else -p
        if a > psi_max: psi_max = a

    var l2 = sqrt(sum_sq / Float64(n_q))
    var l2_ic = sqrt(sum_ic / Float64(n_q))
    var rel_l2 = l2 / l2_ic
    if psi_max > PSI_TOL:
        raise Error(
            String("bench_mhd_alfven_glm_2d_p3 FAILED: psi_max ")
            + String(psi_max) + " > tol " + String(PSI_TOL)
        )
    return rel_l2


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    mpi.init()
    print("bench_mhd_alfven_glm_2d_p3 (Alfven via GLM 2D MHD, P=3, c_h=0)")
    print("  P=", P, "  NP=", num_tri_nodes_2d(P), "  NX=", NX)

    var rel_l2 = _run()
    print("  rel L2(state) =", rel_l2,
          "  (threshold", L2_MAX_REL, ")")
    if rel_l2 > L2_MAX_REL:
        raise Error(
            "bench_mhd_alfven_glm_2d_p3 FAILED: rel L2 "
            + String(rel_l2)
            + " exceeds threshold "
            + String(L2_MAX_REL)
        )
    print("=== bench_mhd_alfven_glm_2d_p3 PASSED ===")
    mpi.finalize()
