# ======================================================================
# mhd_2d_gpu_test -- IdealMHD2D GPU smoke test
# ======================================================================
#
# Constant-state preservation on a periodic mesh.  With a uniform
# (rho, u, v, Bx, By, E) state, the MHD volume flux divergence cancels
# exactly and face fluxes cancel pairwise -- one SSPRK3 step must
# leave the state unchanged to Float32 roundoff.
# ======================================================================

from std.sys import has_accelerator
from std.gpu.host import DeviceContext, DeviceBuffer
from std.math import isnan, isinf
from src import mpi
from src.local_mesh_2d import LocalMesh2D
from src.local_mesh_2d_gpu import LocalMesh2DGpu
from src.local_mesh_2d_gpu_mhd import mhd_rk_stage_2d
from src.reference_2d import ReferenceElement2D, num_tri_nodes_2d, num_edge_nodes
from src.reference_2d_gpu import ReferenceElement2DGpu


def _abs32(x: Float32) -> Float32:
    return x if x >= Float32(0.0) else -x


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    mpi.init()
    var size = mpi.world_size()
    if size > 1:
        mpi.finalize()
        print("mhd_2d_gpu_test: runs at np=1 only")
        return
    print("mhd_2d_gpu_test (constant-state preservation)")

    comptime P = 2
    comptime Nx = 5
    comptime Ny = 4
    comptime NP_p = num_tri_nodes_2d(P)
    comptime NFP_e = num_edge_nodes(P)
    comptime NC = 6

    var host = LocalMesh2D[P](Nx, Ny, 1.0, 1.0)
    var ctx = DeviceContext()
    var gpu = LocalMesh2DGpu[P](ctx, host^)
    var re_host = ReferenceElement2D[P]()
    var re_gpu = ReferenceElement2DGpu[P](ctx, re_host)

    var gamma = Float32(5.0 / 3.0)
    var min_rho = Float32(1.0e-8)
    var min_p   = Float32(1.0e-8)
    var rho0 = Float32(1.0)
    var u0   = Float32(0.2)
    var v0   = Float32(0.1)
    var Bx0  = Float32(0.3)
    var By0  = Float32(0.15)
    var p0   = Float32(0.5)
    var mx0 = rho0 * u0
    var my0 = rho0 * v0
    var ke  = Float32(0.5) * rho0 * (u0 * u0 + v0 * v0)
    var mp  = Float32(0.5) * (Bx0 * Bx0 + By0 * By0)
    var E0  = p0 / (gamma - Float32(1.0)) + ke + mp

    var n_q = gpu.num_elements * NP_p * NC
    var d_q  = ctx.enqueue_create_buffer[DType.float32](n_q)
    var d_q1 = ctx.enqueue_create_buffer[DType.float32](n_q)
    var d_q2 = ctx.enqueue_create_buffer[DType.float32](n_q)
    var d_vol = ctx.enqueue_create_buffer[DType.float32](n_q)
    var d_rhs = ctx.enqueue_create_buffer[DType.float32](n_q)
    var d_fstar = ctx.enqueue_create_buffer[DType.float32](
        gpu.num_faces * NFP_e * NC
    )
    var hbuf = ctx.enqueue_create_host_buffer[DType.float32](n_q)
    var hptr = hbuf.unsafe_ptr()

    for k in range(gpu.num_elements * NP_p):
        hptr[k * NC + 0] = rho0
        hptr[k * NC + 1] = mx0
        hptr[k * NC + 2] = my0
        hptr[k * NC + 3] = Bx0
        hptr[k * NC + 4] = By0
        hptr[k * NC + 5] = E0
    ctx.enqueue_copy(d_q, hbuf)
    ctx.synchronize()

    var dt = Float32(2.0e-4)
    mhd_rk_stage_2d[P](
        ctx, gpu,
        re_gpu.d_Lift_ref.unsafe_ptr(), re_gpu.d_D_ref.unsafe_ptr(),
        d_q.unsafe_ptr(),
        d_q.unsafe_ptr(), d_q.unsafe_ptr(),
        d_q1.unsafe_ptr(),
        d_vol.unsafe_ptr(), d_fstar.unsafe_ptr(), d_rhs.unsafe_ptr(),
        gamma, min_rho, min_p,
        Float32(1.0), Float32(0.0), Float32(1.0), dt,
    )
    mhd_rk_stage_2d[P](
        ctx, gpu,
        re_gpu.d_Lift_ref.unsafe_ptr(), re_gpu.d_D_ref.unsafe_ptr(),
        d_q1.unsafe_ptr(),
        d_q.unsafe_ptr(), d_q1.unsafe_ptr(),
        d_q2.unsafe_ptr(),
        d_vol.unsafe_ptr(), d_fstar.unsafe_ptr(), d_rhs.unsafe_ptr(),
        gamma, min_rho, min_p,
        Float32(0.75), Float32(0.25), Float32(0.25), dt,
    )
    mhd_rk_stage_2d[P](
        ctx, gpu,
        re_gpu.d_Lift_ref.unsafe_ptr(), re_gpu.d_D_ref.unsafe_ptr(),
        d_q2.unsafe_ptr(),
        d_q.unsafe_ptr(), d_q2.unsafe_ptr(),
        d_q.unsafe_ptr(),
        d_vol.unsafe_ptr(), d_fstar.unsafe_ptr(), d_rhs.unsafe_ptr(),
        gamma, min_rho, min_p,
        Float32(1.0 / 3.0), Float32(2.0 / 3.0),
        Float32(2.0 / 3.0), dt,
    )
    ctx.enqueue_copy(hbuf, d_q)
    ctx.synchronize()

    var max_err: Float32 = 0.0
    for k in range(gpu.num_elements * NP_p):
        var base = k * NC
        if isnan(hptr[base + 0]) or isinf(hptr[base + 0]):
            raise Error("MHD: non-finite rho at node " + String(k))
        var err0 = _abs32(hptr[base + 0] - rho0)
        var err1 = _abs32(hptr[base + 1] - mx0)
        var err2 = _abs32(hptr[base + 2] - my0)
        var err3 = _abs32(hptr[base + 3] - Bx0)
        var err4 = _abs32(hptr[base + 4] - By0)
        var err5 = _abs32(hptr[base + 5] - E0)
        var local_max = err0
        if err1 > local_max: local_max = err1
        if err2 > local_max: local_max = err2
        if err3 > local_max: local_max = err3
        if err4 > local_max: local_max = err4
        if err5 > local_max: local_max = err5
        if local_max > max_err:
            max_err = local_max
    print("  MHD Rusanov max |q - q_IC| =", max_err)
    if max_err > Float32(1.0e-4):
        raise Error(
            "MHD: constant state not preserved (max err "
            + String(max_err) + ")"
        )

    print("=== mhd_2d_gpu_test PASSED ===")
    mpi.finalize()
