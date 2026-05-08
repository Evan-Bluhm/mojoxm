# ======================================================================
# maxwell_2d_gpu_test -- Maxwell 2D constant-state preservation
# ======================================================================
#
# Periodic 2D mesh, uniform (Ex, Ey, Ez, Bx, By, Bz) state, one SSPRK3
# step.  Maxwell flux on a uniform field is identically zero (every
# spatial derivative vanishes), so q must equal q_IC to Float32
# roundoff after one step.
#
# Catches sign / direction bugs in maxwell_face_flux_kernel_2d and
# maxwell_vol_lift_combine_rk_kernel_2d that would inject spurious
# wave activity from a uniform field (e.g. a misrouted curl term or a
# normal-direction sign flip on the Faraday equation).
#
# Parameterised over P in {2, 3, 4, 5} to catch a P-specific
# regression in the comptime-templated kernels (e.g. a shared-mem
# index that broke at NP=15 / NP=21 but happened to work at NP=6).
# ======================================================================

from std.sys import has_accelerator
from std.gpu.host import DeviceContext, DeviceBuffer
from std.math import isnan, isinf
from src import mpi
from src.local_mesh_2d import LocalMesh2D
from src.local_mesh_2d_gpu import LocalMesh2DGpu
from src.local_mesh_2d_gpu_maxwell import maxwell_rk_stage_2d
from src.reference_2d import (
    ReferenceElement2D,
    num_tri_nodes_2d,
    num_edge_nodes,
)
from src.reference_2d_gpu import ReferenceElement2DGpu


def _abs32(x: Float32) -> Float32:
    return x if x >= Float32(0.0) else -x


def check[P: Int]() raises:
    print("  P=", P)
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

    var c = Float32(1.0)
    var Ex0 = Float32(0.3)
    var Ey0 = Float32(-0.2)
    var Ez0 = Float32(0.5)
    var Bx0 = Float32(0.4)
    var By0 = Float32(0.1)
    var Bz0 = Float32(-0.6)

    var n_q = gpu.num_elements * NP_p * NC
    var d_q = ctx.enqueue_create_buffer[DType.float32](n_q)
    var d_q1 = ctx.enqueue_create_buffer[DType.float32](n_q)
    var d_q2 = ctx.enqueue_create_buffer[DType.float32](n_q)
    var d_fstar = ctx.enqueue_create_buffer[DType.float32](gpu.num_faces * NFP_e * NC)
    var hbuf = ctx.enqueue_create_host_buffer[DType.float32](n_q)
    var hptr = hbuf.unsafe_ptr()
    for k in range(gpu.num_elements * NP_p):
        hptr[k * NC + 0] = Ex0
        hptr[k * NC + 1] = Ey0
        hptr[k * NC + 2] = Ez0
        hptr[k * NC + 3] = Bx0
        hptr[k * NC + 4] = By0
        hptr[k * NC + 5] = Bz0
    ctx.enqueue_copy(d_q, hbuf)
    ctx.synchronize()

    var dt = Float32(2.0e-4)
    maxwell_rk_stage_2d[P](
        ctx,
        gpu,
        re_gpu.d_Lift_ref.unsafe_ptr(),
        re_gpu.d_D_ref.unsafe_ptr(),
        d_q.unsafe_ptr(),
        d_q.unsafe_ptr(),
        d_q.unsafe_ptr(),
        d_q1.unsafe_ptr(),
        d_fstar.unsafe_ptr(),
        c,
        Float32(1.0),
        Float32(0.0),
        Float32(1.0),
        dt,
    )
    maxwell_rk_stage_2d[P](
        ctx,
        gpu,
        re_gpu.d_Lift_ref.unsafe_ptr(),
        re_gpu.d_D_ref.unsafe_ptr(),
        d_q1.unsafe_ptr(),
        d_q.unsafe_ptr(),
        d_q1.unsafe_ptr(),
        d_q2.unsafe_ptr(),
        d_fstar.unsafe_ptr(),
        c,
        Float32(0.75),
        Float32(0.25),
        Float32(0.25),
        dt,
    )
    maxwell_rk_stage_2d[P](
        ctx,
        gpu,
        re_gpu.d_Lift_ref.unsafe_ptr(),
        re_gpu.d_D_ref.unsafe_ptr(),
        d_q2.unsafe_ptr(),
        d_q.unsafe_ptr(),
        d_q2.unsafe_ptr(),
        d_q.unsafe_ptr(),
        d_fstar.unsafe_ptr(),
        c,
        Float32(1.0 / 3.0),
        Float32(2.0 / 3.0),
        Float32(2.0 / 3.0),
        dt,
    )
    ctx.enqueue_copy(hbuf, d_q)
    ctx.synchronize()

    var max_err: Float32 = 0.0
    var ic = List[Float32]()
    ic.append(Ex0)
    ic.append(Ey0)
    ic.append(Ez0)
    ic.append(Bx0)
    ic.append(By0)
    ic.append(Bz0)
    for k in range(gpu.num_elements * NP_p):
        var base = k * NC
        for c_idx in range(NC):
            var v = hptr[base + c_idx]
            if isnan(v) or isinf(v):
                raise Error("Maxwell: non-finite at node " + String(k) + " comp " + String(c_idx))
            var err = _abs32(v - ic[c_idx])
            if err > max_err:
                max_err = err
    print("    Maxwell max |q - q_IC| =", max_err)
    if max_err > Float32(1.0e-4):
        raise Error("Maxwell P=" + String(P) + ": constant state not preserved (max err " + String(max_err) + ")")


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    mpi.init()
    var size = mpi.world_size()
    if size > 1:
        mpi.finalize()
        print("maxwell_2d_gpu_test: runs at np=1 only")
        return
    print("maxwell_2d_gpu_test (constant-state preservation, P=2..5)")
    check[2]()
    check[3]()
    check[4]()
    check[5]()
    print("=== maxwell_2d_gpu_test PASSED ===")
    mpi.finalize()
