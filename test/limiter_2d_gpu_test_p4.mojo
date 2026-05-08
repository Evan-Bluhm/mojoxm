# ======================================================================
# limiter_2d_gpu_test_p4 -- P=4 BJ limiter smoke test (NP=15)
# ======================================================================
#
# P=4 counterpart of limiter_2d_gpu_test_p3.  The mass-matrix-weighted
# cell mean differs at every P (different node-weight distributions
# per `ReferenceElement2D[P].node_weights`), so each P needs a
# direct test to catch a regression in either the kernel weights or
# the BJ limiter's downstream use of them.
#
# At P=4 the per-element node count NP=15 is the largest 2D NP that
# the bench harness exercises in shocked-flow gates
# (`bench_euler_sod_limited_2d_p4`).  This test keeps the same two
# checks as the P=2/P=3 tests in a self-contained smoke form:
#   (1) Smooth passthrough -- a constant state must come out
#       bit-unchanged.
#   (2) Within-cell spike monotonicity -- a node-spike rho=5 in an
#       otherwise-rho=1 sea gets clamped well below 5 by BJ; the
#       far-from-spike half of the mesh stays untouched.
#
# Runs at P=4, 4 components.
# ======================================================================

from std.sys import has_accelerator
from std.gpu.host import DeviceContext, DeviceBuffer
from std.math import isnan, isinf
from src import mpi
from src.local_mesh_2d import LocalMesh2D
from src.local_mesh_2d_gpu import LocalMesh2DGpu
from src.local_mesh_2d_gpu_limiter import bj_limit_full_2d
from src.reference_2d import ReferenceElement2D, num_tri_nodes_2d
from src.reference_2d_gpu import ReferenceElement2DGpu


def _abs32(x: Float32) -> Float32:
    return x if x >= Float32(0.0) else -x


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    mpi.init()
    var size = mpi.world_size()
    if size > 1:
        mpi.finalize()
        print("limiter_2d_gpu_test_p4: runs at np=1 only")
        return
    print("limiter_2d_gpu_test_p4 (P=4 / NP=15 BJ limiter)")

    comptime P = 4
    comptime Nx = 6
    comptime Ny = 4
    comptime NP_p = num_tri_nodes_2d(P)
    comptime NC = 4

    var host = LocalMesh2D[P](Nx=Nx, Ny=Ny, Lx=1.0, Ly=1.0)
    var ctx = DeviceContext()
    var gpu = LocalMesh2DGpu[P](ctx=ctx, host=host^)
    var re_host = ReferenceElement2D[P]()
    var re_gpu = ReferenceElement2DGpu[P](ctx=ctx, host=re_host)

    var n_q = gpu.num_elements * NP_p * NC
    var d_q = ctx.enqueue_create_buffer[DType.float32](n_q)
    var d_ca = ctx.enqueue_create_buffer[DType.float32](gpu.num_elements * NC)
    var hbuf = ctx.enqueue_create_host_buffer[DType.float32](n_q)
    var hptr = hbuf.unsafe_ptr()

    # ---- (1) smooth passthrough on a uniform state ----------------
    var rho0 = Float32(1.0)
    var mx0 = Float32(0.2)
    var my0 = Float32(0.1)
    var E0 = Float32(2.52)
    for k in range(gpu.num_elements * NP_p):
        hptr[k * NC + 0] = rho0
        hptr[k * NC + 1] = mx0
        hptr[k * NC + 2] = my0
        hptr[k * NC + 3] = E0
    ctx.enqueue_copy(d_q, hbuf)
    ctx.synchronize()
    bj_limit_full_2d[P, NC](
        ctx,
        gpu,
        d_q.unsafe_ptr(),
        re_gpu.d_node_weights.unsafe_ptr(),
        d_ca.unsafe_ptr(),
        Float32(0.1),
    )
    ctx.enqueue_copy(hbuf, d_q)
    ctx.synchronize()
    var max_uniform_err: Float32 = 0.0
    for k in range(gpu.num_elements * NP_p):
        var e0 = _abs32(hptr[k * NC + 0] - rho0)
        var e1 = _abs32(hptr[k * NC + 1] - mx0)
        var e2 = _abs32(hptr[k * NC + 2] - my0)
        var e3 = _abs32(hptr[k * NC + 3] - E0)
        var m = e0
        if e1 > m:
            m = e1
        if e2 > m:
            m = e2
        if e3 > m:
            m = e3
        if m > max_uniform_err:
            max_uniform_err = m
    print("  (1) smooth passthrough max |q - q_IC| =", max_uniform_err)
    if max_uniform_err > Float32(1.0e-6):
        raise Error("limiter perturbed a uniform state (max err " + String(max_uniform_err) + ")")

    # ---- (2) within-cell spike monotonicity -----------------------
    # Same setup as the P=2/P=3 tests: rho=1 everywhere, then perturb
    # node 0 of element 0 to rho=5.  BJ should clamp the spike well
    # below 5 (the cell mean stays near 1 because at NP=15 there are
    # 14 other nodes weighing in via the mass-matrix-weighted mean).
    # The threshold (max rho < 2.5) is the same as the P=2/P=3 tests
    # since BJ pulls toward the cell mean either way -- if anything
    # it should clamp harder at higher NP since the mean is closer
    # to the unperturbed value.
    for k in range(gpu.num_elements * NP_p):
        hptr[k * NC + 0] = rho0
        hptr[k * NC + 1] = mx0
        hptr[k * NC + 2] = my0
        hptr[k * NC + 3] = E0
    hptr[0 * NC + 0] = Float32(5.0)
    ctx.enqueue_copy(d_q, hbuf)
    ctx.synchronize()
    bj_limit_full_2d[P, NC](
        ctx,
        gpu,
        d_q.unsafe_ptr(),
        re_gpu.d_node_weights.unsafe_ptr(),
        d_ca.unsafe_ptr(),
        Float32(0.1),
    )
    ctx.enqueue_copy(hbuf, d_q)
    ctx.synchronize()

    var max_rho: Float32 = 0.0
    for k in range(gpu.num_elements * NP_p):
        var rho = hptr[k * NC + 0]
        if isnan(rho) or isinf(rho):
            raise Error("limiter: non-finite density at index " + String(k))
        if rho > max_rho:
            max_rho = rho
    print("  (2) spike: max rho after limiting (pre-limit = 5.0) =", max_rho)
    if max_rho >= Float32(5.0):
        raise Error("limiter did nothing -- max rho stayed at " + String(max_rho))
    if max_rho > Float32(2.5):
        raise Error("limiter under-scaled the spike (max rho " + String(max_rho) + ", expected < 2.5)")

    # Check (2b): far-from-spike elements should be unperturbed.
    var max_far_err: Float32 = 0.0
    var start_elem = gpu.num_elements // 2
    for elem in range(start_elem, gpu.num_elements):
        for nn in range(NP_p):
            var base = (elem * NP_p + nn) * NC
            var e0 = _abs32(hptr[base + 0] - rho0)
            var e1 = _abs32(hptr[base + 1] - mx0)
            var e2 = _abs32(hptr[base + 2] - my0)
            var e3 = _abs32(hptr[base + 3] - E0)
            var m = e0
            if e1 > m:
                m = e1
            if e2 > m:
                m = e2
            if e3 > m:
                m = e3
            if m > max_far_err:
                max_far_err = m
    print("  (2) far-from-spike max |q - q_IC| =", max_far_err)
    if max_far_err > Float32(1.0e-6):
        raise Error("limiter perturbed far-from-spike elements (max err " + String(max_far_err) + ")")

    print("=== limiter_2d_gpu_test_p4 PASSED ===")
    mpi.finalize()
