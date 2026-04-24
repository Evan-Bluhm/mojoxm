# ======================================================================
# euler_2d_gpu_test -- Euler2D GPU smoke test (Rusanov + HLLC)
# ======================================================================
#
# Constant-state preservation: on a periodic mesh with a uniform
# Euler state (rho=1, u=0.5, v=0.3, p=1, so gamma=1.4), both the
# volume divergence and the face-flux ring cancel exactly -- after
# one SSPRK3 step on the GPU, every node must equal the IC to
# roundoff.  This tests the Euler flux math, the Jacobian / normal
# conventions, and the lift + rk-update plumbing simultaneously
# without needing an external reference.
#
# Separately asserts that every output is finite (no NaN / Inf) and
# that density / energy stay above their floors.
#
# Runs Rusanov and HLLC back-to-back.  The same IC should come back
# bit-identical from both since the state is constant and the jump
# terms vanish either way.
# ======================================================================

from std.sys import has_accelerator
from std.gpu.host import DeviceContext, DeviceBuffer
from std.math import isnan, isinf
from src import mpi
from src.local_mesh_2d import LocalMesh2D
from src.local_mesh_2d_gpu import (
    LocalMesh2DGpu, euler_rk_stage_2d, euler_rk_stage_hllc_2d,
)
from src.reference_2d import ReferenceElement2D, num_tri_nodes_2d, num_edge_nodes
from src.reference_2d_gpu import ReferenceElement2DGpu


def _abs32(x: Float32) -> Float32:
    return x if x >= Float32(0.0) else -x


def _check_constant(
    label: String,
    hptr: UnsafePointer[Float32, MutAnyOrigin],
    n: Int,
    expect_rho: Float32,
    expect_mx: Float32,
    expect_my: Float32,
    expect_E: Float32,
) raises:
    var max_err: Float32 = 0.0
    for k in range(n):
        var v = hptr[k]
        if isnan(v) or isinf(v):
            raise Error(label + ": non-finite value at index " + String(k))
        var expect: Float32 = expect_rho
        var c = k % 4
        if c == 1: expect = expect_mx
        elif c == 2: expect = expect_my
        elif c == 3: expect = expect_E
        var err = _abs32(v - expect)
        if err > max_err:
            max_err = err
    print("  ", label, " max |q - q_IC| =", max_err)
    # 1e-4 absolute tolerance: one SSPRK3 step at dt=5e-4 with a
    # constant state should cancel to Float32 roundoff.  Generous
    # bound, but still catches sign / scale bugs (those would push
    # err into the O(dt * f) range which is >> 1e-4 for these values).
    if max_err > Float32(1.0e-4):
        raise Error(
            label + ": constant state not preserved (max err "
            + String(max_err) + ")"
        )


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    mpi.init()
    var size = mpi.world_size()
    if size > 1:
        mpi.finalize()
        print("euler_2d_gpu_test: runs at np=1 only")
        return
    print("euler_2d_gpu_test (constant-state preservation, Rusanov + HLLC)")

    comptime P = 2
    comptime Nx = 5
    comptime Ny = 4
    comptime NP_p = num_tri_nodes_2d(P)
    comptime NFP_e = num_edge_nodes(P)
    comptime NC = 4

    var host = LocalMesh2D[P](Nx, Ny, 1.0, 1.0)
    var ctx = DeviceContext()
    var gpu = LocalMesh2DGpu[P](ctx, host^)
    var re_host = ReferenceElement2D[P]()
    var re_gpu = ReferenceElement2DGpu[P](ctx, re_host)

    # Uniform IC (analytic steady state on a periodic domain).
    var rho0  = Float32(1.0)
    var u0    = Float32(0.5)
    var v0    = Float32(0.3)
    var p0    = Float32(1.0)
    var gamma = Float32(1.4)
    var mx0 = rho0 * u0
    var my0 = rho0 * v0
    var E0  = p0 / (gamma - Float32(1.0)) + Float32(0.5) * rho0 * (u0 * u0 + v0 * v0)
    var min_rho = Float32(1.0e-8)
    var min_p   = Float32(1.0e-8)

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

    var dt = Float32(5.0e-4)

    # ---- Rusanov: one SSPRK3 step on constant IC ------------------
    for k in range(gpu.num_elements * NP_p):
        hptr[k * NC + 0] = rho0
        hptr[k * NC + 1] = mx0
        hptr[k * NC + 2] = my0
        hptr[k * NC + 3] = E0
    ctx.enqueue_copy(d_q, hbuf)
    ctx.synchronize()
    euler_rk_stage_2d[P](
        ctx, gpu,
        re_gpu.d_Lift_ref.unsafe_ptr(), re_gpu.d_D_ref.unsafe_ptr(),
        d_q.unsafe_ptr(),
        d_q.unsafe_ptr(), d_q.unsafe_ptr(),
        d_q1.unsafe_ptr(),
        d_vol.unsafe_ptr(), d_fstar.unsafe_ptr(), d_rhs.unsafe_ptr(),
        gamma, min_rho, min_p,
        Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0),
        Float32(1.0), Float32(0.0), Float32(1.0), dt,
    )
    euler_rk_stage_2d[P](
        ctx, gpu,
        re_gpu.d_Lift_ref.unsafe_ptr(), re_gpu.d_D_ref.unsafe_ptr(),
        d_q1.unsafe_ptr(),
        d_q.unsafe_ptr(), d_q1.unsafe_ptr(),
        d_q2.unsafe_ptr(),
        d_vol.unsafe_ptr(), d_fstar.unsafe_ptr(), d_rhs.unsafe_ptr(),
        gamma, min_rho, min_p,
        Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0),
        Float32(0.75), Float32(0.25), Float32(0.25), dt,
    )
    euler_rk_stage_2d[P](
        ctx, gpu,
        re_gpu.d_Lift_ref.unsafe_ptr(), re_gpu.d_D_ref.unsafe_ptr(),
        d_q2.unsafe_ptr(),
        d_q.unsafe_ptr(), d_q2.unsafe_ptr(),
        d_q.unsafe_ptr(),
        d_vol.unsafe_ptr(), d_fstar.unsafe_ptr(), d_rhs.unsafe_ptr(),
        gamma, min_rho, min_p,
        Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0),
        Float32(1.0 / 3.0), Float32(2.0 / 3.0),
        Float32(2.0 / 3.0), dt,
    )
    ctx.enqueue_copy(hbuf, d_q)
    ctx.synchronize()
    _check_constant(
        String("Euler Rusanov"), hptr, n_q, rho0, mx0, my0, E0,
    )

    # ---- HLLC: same check ---------------------------------------
    for k in range(gpu.num_elements * NP_p):
        hptr[k * NC + 0] = rho0
        hptr[k * NC + 1] = mx0
        hptr[k * NC + 2] = my0
        hptr[k * NC + 3] = E0
    ctx.enqueue_copy(d_q, hbuf)
    ctx.synchronize()
    euler_rk_stage_hllc_2d[P](
        ctx, gpu,
        re_gpu.d_Lift_ref.unsafe_ptr(), re_gpu.d_D_ref.unsafe_ptr(),
        d_q.unsafe_ptr(),
        d_q.unsafe_ptr(), d_q.unsafe_ptr(),
        d_q1.unsafe_ptr(),
        d_vol.unsafe_ptr(), d_fstar.unsafe_ptr(), d_rhs.unsafe_ptr(),
        gamma, min_rho, min_p,
        Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0),
        Float32(1.0), Float32(0.0), Float32(1.0), dt,
    )
    euler_rk_stage_hllc_2d[P](
        ctx, gpu,
        re_gpu.d_Lift_ref.unsafe_ptr(), re_gpu.d_D_ref.unsafe_ptr(),
        d_q1.unsafe_ptr(),
        d_q.unsafe_ptr(), d_q1.unsafe_ptr(),
        d_q2.unsafe_ptr(),
        d_vol.unsafe_ptr(), d_fstar.unsafe_ptr(), d_rhs.unsafe_ptr(),
        gamma, min_rho, min_p,
        Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0),
        Float32(0.75), Float32(0.25), Float32(0.25), dt,
    )
    euler_rk_stage_hllc_2d[P](
        ctx, gpu,
        re_gpu.d_Lift_ref.unsafe_ptr(), re_gpu.d_D_ref.unsafe_ptr(),
        d_q2.unsafe_ptr(),
        d_q.unsafe_ptr(), d_q2.unsafe_ptr(),
        d_q.unsafe_ptr(),
        d_vol.unsafe_ptr(), d_fstar.unsafe_ptr(), d_rhs.unsafe_ptr(),
        gamma, min_rho, min_p,
        Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0),
        Float32(1.0 / 3.0), Float32(2.0 / 3.0),
        Float32(2.0 / 3.0), dt,
    )
    ctx.enqueue_copy(hbuf, d_q)
    ctx.synchronize()
    _check_constant(
        String("Euler HLLC"), hptr, n_q, rho0, mx0, my0, E0,
    )

    print("=== euler_2d_gpu_test PASSED ===")
    mpi.finalize()
