# ======================================================================
# sw_2d_gpu_test -- ShallowWater2D GPU smoke test (Rusanov + HLL)
# ======================================================================
#
# Constant-state preservation on a periodic mesh.  With a uniform
# state (h=2, u=0.4, v=0.2), the volume divergence cancels exactly
# and face fluxes cancel pairwise -- one SSPRK3 step must leave the
# state unchanged to Float32 roundoff.  Covers both Rusanov and HLL.
#
# Parameterised over P in {2, 3, 4, 5} so a P-specific regression in
# the comptime-templated kernels (e.g. a shared-mem index that broke
# at NP=15 / NP=21 but happened to work at NP=6) gets caught at
# test-quick latency rather than only by bench-p5.
# ======================================================================

from std.sys import has_accelerator
from std.gpu.host import DeviceContext, DeviceBuffer
from std.math import isnan, isinf
from src import mpi
from src.local_mesh_2d import LocalMesh2D
from src.local_mesh_2d_gpu import LocalMesh2DGpu
from src.local_mesh_2d_gpu_sw import sw_rk_stage_2d, sw_rk_stage_hll_2d
from src.reference_2d import (
    ReferenceElement2D,
    num_tri_nodes_2d,
    num_edge_nodes,
)
from src.reference_2d_gpu import ReferenceElement2DGpu


def _abs32(x: Float32) -> Float32:
    return x if x >= Float32(0.0) else -x


def _check_constant(
    label: String,
    hptr: UnsafePointer[Float32, MutAnyOrigin],
    n: Int,
    expect_h: Float32,
    expect_hu: Float32,
    expect_hv: Float32,
) raises:
    var max_err: Float32 = 0.0
    for k in range(n):
        var v = hptr[k]
        if isnan(v) or isinf(v):
            raise Error(label + ": non-finite value at index " + String(k))
        var expect: Float32 = expect_h
        var c = k % 3
        if c == 1:
            expect = expect_hu
        elif c == 2:
            expect = expect_hv
        var err = _abs32(v - expect)
        if err > max_err:
            max_err = err
    print("  ", label, " max |q - q_IC| =", max_err)
    if max_err > Float32(1.0e-4):
        raise Error(label + ": constant state not preserved (max err " + String(max_err) + ")")


def check[P: Int]() raises:
    print("  P=", P)
    comptime Nx = 5
    comptime Ny = 4
    comptime NP_p = num_tri_nodes_2d(P)
    comptime NFP_e = num_edge_nodes(P)
    comptime NC = 3

    var host = LocalMesh2D[P](Nx, Ny, 1.0, 1.0)
    var ctx = DeviceContext()
    var gpu = LocalMesh2DGpu[P](ctx, host^)
    var re_host = ReferenceElement2D[P]()
    var re_gpu = ReferenceElement2DGpu[P](ctx, re_host)

    var h0 = Float32(2.0)
    var u0 = Float32(0.4)
    var v0 = Float32(0.2)
    var hu0 = h0 * u0
    var hv0 = h0 * v0
    var gsw = Float32(9.81)
    var min_h = Float32(1.0e-6)

    var n_q = gpu.num_elements * NP_p * NC
    var d_q = ctx.enqueue_create_buffer[DType.float32](n_q)
    var d_q1 = ctx.enqueue_create_buffer[DType.float32](n_q)
    var d_q2 = ctx.enqueue_create_buffer[DType.float32](n_q)
    var d_fstar = ctx.enqueue_create_buffer[DType.float32](gpu.num_faces * NFP_e * NC)
    var hbuf = ctx.enqueue_create_host_buffer[DType.float32](n_q)
    var hptr = hbuf.unsafe_ptr()

    var dt = Float32(2.0e-4)

    # ---- Rusanov ---------------------------------------------------
    for k in range(gpu.num_elements * NP_p):
        hptr[k * NC + 0] = h0
        hptr[k * NC + 1] = hu0
        hptr[k * NC + 2] = hv0
    ctx.enqueue_copy(d_q, hbuf)
    ctx.synchronize()
    sw_rk_stage_2d[P](
        ctx,
        gpu,
        re_gpu.d_Lift_ref.unsafe_ptr(),
        re_gpu.d_D_ref.unsafe_ptr(),
        d_q.unsafe_ptr(),
        d_q.unsafe_ptr(),
        d_q.unsafe_ptr(),
        d_q1.unsafe_ptr(),
        d_fstar.unsafe_ptr(),
        gsw,
        min_h,
        Float32(1.0),
        Float32(0.0),
        Float32(1.0),
        dt,
    )
    sw_rk_stage_2d[P](
        ctx,
        gpu,
        re_gpu.d_Lift_ref.unsafe_ptr(),
        re_gpu.d_D_ref.unsafe_ptr(),
        d_q1.unsafe_ptr(),
        d_q.unsafe_ptr(),
        d_q1.unsafe_ptr(),
        d_q2.unsafe_ptr(),
        d_fstar.unsafe_ptr(),
        gsw,
        min_h,
        Float32(0.75),
        Float32(0.25),
        Float32(0.25),
        dt,
    )
    sw_rk_stage_2d[P](
        ctx,
        gpu,
        re_gpu.d_Lift_ref.unsafe_ptr(),
        re_gpu.d_D_ref.unsafe_ptr(),
        d_q2.unsafe_ptr(),
        d_q.unsafe_ptr(),
        d_q2.unsafe_ptr(),
        d_q.unsafe_ptr(),
        d_fstar.unsafe_ptr(),
        gsw,
        min_h,
        Float32(1.0 / 3.0),
        Float32(2.0 / 3.0),
        Float32(2.0 / 3.0),
        dt,
    )
    ctx.enqueue_copy(hbuf, d_q)
    ctx.synchronize()
    _check_constant(String("SW Rusanov"), hptr, n_q, h0, hu0, hv0)

    # ---- HLL -------------------------------------------------------
    for k in range(gpu.num_elements * NP_p):
        hptr[k * NC + 0] = h0
        hptr[k * NC + 1] = hu0
        hptr[k * NC + 2] = hv0
    ctx.enqueue_copy(d_q, hbuf)
    ctx.synchronize()
    sw_rk_stage_hll_2d[P](
        ctx,
        gpu,
        re_gpu.d_Lift_ref.unsafe_ptr(),
        re_gpu.d_D_ref.unsafe_ptr(),
        d_q.unsafe_ptr(),
        d_q.unsafe_ptr(),
        d_q.unsafe_ptr(),
        d_q1.unsafe_ptr(),
        d_fstar.unsafe_ptr(),
        gsw,
        min_h,
        Float32(1.0),
        Float32(0.0),
        Float32(1.0),
        dt,
    )
    sw_rk_stage_hll_2d[P](
        ctx,
        gpu,
        re_gpu.d_Lift_ref.unsafe_ptr(),
        re_gpu.d_D_ref.unsafe_ptr(),
        d_q1.unsafe_ptr(),
        d_q.unsafe_ptr(),
        d_q1.unsafe_ptr(),
        d_q2.unsafe_ptr(),
        d_fstar.unsafe_ptr(),
        gsw,
        min_h,
        Float32(0.75),
        Float32(0.25),
        Float32(0.25),
        dt,
    )
    sw_rk_stage_hll_2d[P](
        ctx,
        gpu,
        re_gpu.d_Lift_ref.unsafe_ptr(),
        re_gpu.d_D_ref.unsafe_ptr(),
        d_q2.unsafe_ptr(),
        d_q.unsafe_ptr(),
        d_q2.unsafe_ptr(),
        d_q.unsafe_ptr(),
        d_fstar.unsafe_ptr(),
        gsw,
        min_h,
        Float32(1.0 / 3.0),
        Float32(2.0 / 3.0),
        Float32(2.0 / 3.0),
        dt,
    )
    ctx.enqueue_copy(hbuf, d_q)
    ctx.synchronize()
    _check_constant(String("SW HLL"), hptr, n_q, h0, hu0, hv0)


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    mpi.init()
    var size = mpi.world_size()
    if size > 1:
        mpi.finalize()
        print("sw_2d_gpu_test: runs at np=1 only")
        return
    print("sw_2d_gpu_test (constant-state preservation, Rusanov + HLL, P=2..5)")
    check[2]()
    check[3]()
    check[4]()
    check[5]()
    print("=== sw_2d_gpu_test PASSED ===")
    mpi.finalize()
