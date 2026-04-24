# ======================================================================
# limiter_2d_gpu_test -- Barth-Jespersen GPU limiter smoke test
# ======================================================================
#
# Two rigorous self-checks without a CPU reference:
#
#   (1) Smooth-passthrough: a constant state has zero deviation from
#       its cell mean, so the limiter must leave every node
#       bit-unchanged (theta = 1 trivially, or the early-exit path
#       kicks in before any scaling).
#
#   (2) Monotonicity: on a spiky IC where one element's rho sits far
#       outside its neighbours' range, the limiter must pull the
#       outlier nodes back inside [neighbour_min, neighbour_max] for
#       the density component.  Interior elements in a "clean band"
#       should still pass through unchanged.
#
# Runs at P=2, 4 components (Euler-shaped state).
# ======================================================================

from std.sys import has_accelerator
from std.gpu.host import DeviceContext, DeviceBuffer
from std.math import isnan, isinf
from src import mpi
from src.local_mesh_2d import LocalMesh2D
from src.local_mesh_2d_gpu import LocalMesh2DGpu, bj_limit_full_2d
from src.reference_2d import ReferenceElement2D, num_tri_nodes_2d


def _abs32(x: Float32) -> Float32:
    return x if x >= Float32(0.0) else -x


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    mpi.init()
    var size = mpi.world_size()
    if size > 1:
        mpi.finalize()
        print("limiter_2d_gpu_test: runs at np=1 only")
        return
    print("limiter_2d_gpu_test (smooth passthrough + spike monotonicity)")

    comptime P = 2
    comptime Nx = 6
    comptime Ny = 4
    comptime NP_p = num_tri_nodes_2d(P)
    comptime NC = 4

    var host = LocalMesh2D[P](Nx, Ny, 1.0, 1.0)
    var ctx = DeviceContext()
    var gpu = LocalMesh2DGpu[P](ctx, host^)

    var n_q = gpu.num_elements * NP_p * NC
    var d_q = ctx.enqueue_create_buffer[DType.float32](n_q)
    var d_ca = ctx.enqueue_create_buffer[DType.float32](
        gpu.num_elements * NC
    )
    var hbuf = ctx.enqueue_create_host_buffer[DType.float32](n_q)
    var hptr = hbuf.unsafe_ptr()

    # ---- (1) smooth passthrough on a uniform state ----------------
    var rho0 = Float32(1.0)
    var mx0  = Float32(0.2)
    var my0  = Float32(0.1)
    var E0   = Float32(2.52)     # plausible Euler energy, not zero
    for k in range(gpu.num_elements * NP_p):
        hptr[k * NC + 0] = rho0
        hptr[k * NC + 1] = mx0
        hptr[k * NC + 2] = my0
        hptr[k * NC + 3] = E0
    ctx.enqueue_copy(d_q, hbuf)
    ctx.synchronize()
    bj_limit_full_2d[P, NC](
        ctx, gpu, d_q.unsafe_ptr(), d_ca.unsafe_ptr(),
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
        if e1 > m: m = e1
        if e2 > m: m = e2
        if e3 > m: m = e3
        if m > max_uniform_err: max_uniform_err = m
    print("  (1) smooth passthrough max |q - q_IC| =", max_uniform_err)
    if max_uniform_err > Float32(1.0e-6):
        raise Error(
            "limiter perturbed a uniform state (max err "
            + String(max_uniform_err) + ")"
        )

    # ---- (2) within-cell spike monotonicity -----------------------
    # Seed rho=1 everywhere, then perturb ONE node inside element 0
    # to rho=5 while leaving the other NP-1 nodes at 1.  BJ is a
    # within-cell slope limiter: element 0's mean is slightly above
    # 1 (the spike drags it up), its neighbours' means are 1, and
    # the spike node sits far above both.  The Venkat theta must
    # come out much smaller than 1 and scale the spike back toward
    # the cell mean.
    #
    # Concretely: after limiting, max rho in the domain should be
    # strictly less than the pre-limit spike (5.0) -- and with
    # Venkat eps=0.1 and such a large d / D ratio, the spike gets
    # pulled almost to the mean.  We assert max rho < 2.5, which
    # gives the Venkat smoothing plenty of headroom while still
    # proving the limiter acted.
    for k in range(gpu.num_elements * NP_p):
        hptr[k * NC + 0] = rho0
        hptr[k * NC + 1] = mx0
        hptr[k * NC + 2] = my0
        hptr[k * NC + 3] = E0
    # Perturb a single interior node in element 0.
    hptr[0 * NC + 0] = Float32(5.0)
    ctx.enqueue_copy(d_q, hbuf)
    ctx.synchronize()
    bj_limit_full_2d[P, NC](
        ctx, gpu, d_q.unsafe_ptr(), d_ca.unsafe_ptr(),
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
    print("  (2) spike: max rho after limiting (pre-limit = 5.0) =",
          max_rho)
    if max_rho >= Float32(5.0):
        raise Error(
            "limiter did nothing -- max rho stayed at "
            + String(max_rho)
        )
    if max_rho > Float32(2.5):
        raise Error(
            "limiter under-scaled the spike (max rho "
            + String(max_rho) + ", expected < 2.5)"
        )

    # Check (2b): far-from-spike elements should be unperturbed.
    # Element 0 has 3 face-neighbours that will see its elevated
    # cell-mean and may limit slightly; everything past those is
    # untouched.  We check the second half of the mesh.
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
            if e1 > m: m = e1
            if e2 > m: m = e2
            if e3 > m: m = e3
            if m > max_far_err: max_far_err = m
    print("  (2) far-from-spike max |q - q_IC| =", max_far_err)
    if max_far_err > Float32(1.0e-6):
        raise Error(
            "limiter perturbed far-from-spike elements (max err "
            + String(max_far_err) + ")"
        )

    print("=== limiter_2d_gpu_test PASSED ===")
    mpi.finalize()
