# ======================================================================
# mhd_glm_2d_gpu_test -- GLM-enabled IdealMHD 2D constant-state
# ======================================================================
#
# Periodic 2D mesh, uniform 7-component GLM-MHD state
# (rho, rho*u, rho*v, Bx, By, E, psi).  With c_h > 0 the GLM transport
# couples to the divergence of B; on a uniform B field that divergence
# is identically zero, so psi stays at its IC value.  alpha_d damping
# is applied as a separate operator-split call -- not exercised here
# (alpha_d = 0 means damp is a no-op).
#
# Tests the new mhd_glm_face_flux_kernel_2d and
# mhd_glm_vol_lift_combine_rk_kernel_2d for a uniform-state regression.
# Counterpart to mhd_2d_gpu_test (NC=6 non-GLM path).
#
# Parameterised over P in {2, 3, 4, 5} to catch P-specific kernel
# regressions at NP=10 / NP=15 / NP=21 (GLM is the largest 2D NC at 7).
# ======================================================================

from std.sys import has_accelerator
from std.gpu.host import DeviceContext, DeviceBuffer
from std.math import isnan, isinf
from src import mpi
from src.local_mesh_2d import LocalMesh2D
from src.local_mesh_2d_gpu import LocalMesh2DGpu
from src.local_mesh_2d_gpu_mhd_glm import mhd_glm_rk_stage_2d
from src.reference_2d import ReferenceElement2D, num_tri_nodes_2d, num_edge_nodes
from src.reference_2d_gpu import ReferenceElement2DGpu


def _abs32(x: Float32) -> Float32:
    return x if x >= Float32(0.0) else -x


def check[P: Int]() raises:
    print("  P=", P)
    comptime Nx = 5
    comptime Ny = 4
    comptime NP_p = num_tri_nodes_2d(P)
    comptime NFP_e = num_edge_nodes(P)
    comptime NC = 7

    var host = LocalMesh2D[P](Nx, Ny, 1.0, 1.0)
    var ctx = DeviceContext()
    var gpu = LocalMesh2DGpu[P](ctx, host^)
    var re_host = ReferenceElement2D[P]()
    var re_gpu = ReferenceElement2DGpu[P](ctx, re_host)

    var gamma = Float32(5.0 / 3.0)
    var min_rho = Float32(1.0e-8)
    var min_p = Float32(1.0e-8)
    var c_h = Float32(1.5)
    var rho0 = Float32(1.0)
    var u0 = Float32(0.2)
    var v0 = Float32(0.1)
    var Bx0 = Float32(0.3)
    var By0 = Float32(0.15)
    var p0 = Float32(0.5)
    var psi0 = Float32(0.05)
    var mx0 = rho0 * u0
    var my0 = rho0 * v0
    var ke = Float32(0.5) * rho0 * (u0 * u0 + v0 * v0)
    var mp = Float32(0.5) * (Bx0 * Bx0 + By0 * By0)
    var E0 = p0 / (gamma - Float32(1.0)) + ke + mp

    var n_q = gpu.num_elements * NP_p * NC
    var d_q = ctx.enqueue_create_buffer[DType.float32](n_q)
    var d_q1 = ctx.enqueue_create_buffer[DType.float32](n_q)
    var d_q2 = ctx.enqueue_create_buffer[DType.float32](n_q)
    var d_fstar = ctx.enqueue_create_buffer[DType.float32](gpu.num_faces * NFP_e * NC)
    var hbuf = ctx.enqueue_create_host_buffer[DType.float32](n_q)
    var hptr = hbuf.unsafe_ptr()

    for k in range(gpu.num_elements * NP_p):
        hptr[k * NC + 0] = rho0
        hptr[k * NC + 1] = mx0
        hptr[k * NC + 2] = my0
        hptr[k * NC + 3] = Bx0
        hptr[k * NC + 4] = By0
        hptr[k * NC + 5] = E0
        hptr[k * NC + 6] = psi0
    ctx.enqueue_copy(d_q, hbuf)
    ctx.synchronize()

    var dt = Float32(2.0e-4)
    mhd_glm_rk_stage_2d[P](
        ctx,
        gpu,
        re_gpu.d_Lift_ref.unsafe_ptr(),
        re_gpu.d_D_ref.unsafe_ptr(),
        d_q.unsafe_ptr(),
        d_q.unsafe_ptr(),
        d_q.unsafe_ptr(),
        d_q1.unsafe_ptr(),
        d_fstar.unsafe_ptr(),
        gamma,
        min_rho,
        min_p,
        c_h,
        Float32(1.0),
        Float32(0.0),
        Float32(1.0),
        dt,
    )
    mhd_glm_rk_stage_2d[P](
        ctx,
        gpu,
        re_gpu.d_Lift_ref.unsafe_ptr(),
        re_gpu.d_D_ref.unsafe_ptr(),
        d_q1.unsafe_ptr(),
        d_q.unsafe_ptr(),
        d_q1.unsafe_ptr(),
        d_q2.unsafe_ptr(),
        d_fstar.unsafe_ptr(),
        gamma,
        min_rho,
        min_p,
        c_h,
        Float32(0.75),
        Float32(0.25),
        Float32(0.25),
        dt,
    )
    mhd_glm_rk_stage_2d[P](
        ctx,
        gpu,
        re_gpu.d_Lift_ref.unsafe_ptr(),
        re_gpu.d_D_ref.unsafe_ptr(),
        d_q2.unsafe_ptr(),
        d_q.unsafe_ptr(),
        d_q2.unsafe_ptr(),
        d_q.unsafe_ptr(),
        d_fstar.unsafe_ptr(),
        gamma,
        min_rho,
        min_p,
        c_h,
        Float32(1.0 / 3.0),
        Float32(2.0 / 3.0),
        Float32(2.0 / 3.0),
        dt,
    )
    ctx.enqueue_copy(hbuf, d_q)
    ctx.synchronize()

    var max_err: Float32 = 0.0
    var ic = List[Float32]()
    ic.append(rho0)
    ic.append(mx0)
    ic.append(my0)
    ic.append(Bx0)
    ic.append(By0)
    ic.append(E0)
    ic.append(psi0)
    for k in range(gpu.num_elements * NP_p):
        var base = k * NC
        for c_idx in range(NC):
            var v = hptr[base + c_idx]
            if isnan(v) or isinf(v):
                raise Error("GLM-MHD: non-finite at component " + String(c_idx))
            var err = _abs32(v - ic[c_idx])
            if err > max_err:
                max_err = err
    print("    GLM-MHD max |q - q_IC| =", max_err)
    if max_err > Float32(1.0e-4):
        raise Error("GLM-MHD P=" + String(P) + ": constant state not preserved (max err " + String(max_err) + ")")


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    mpi.init()
    var size = mpi.world_size()
    if size > 1:
        mpi.finalize()
        print("mhd_glm_2d_gpu_test: runs at np=1 only")
        return
    print("mhd_glm_2d_gpu_test (constant-state preservation, NC=7, P=2..5)")
    check[2]()
    check[3]()
    check[4]()
    check[5]()
    print("=== mhd_glm_2d_gpu_test PASSED ===")
    mpi.finalize()
