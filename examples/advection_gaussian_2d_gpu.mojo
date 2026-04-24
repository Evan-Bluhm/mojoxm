# ======================================================================
# advection_gaussian_2d_gpu -- GPU 2D DG scalar advection
# ======================================================================
#
# GPU analog of `advection_gaussian_2d_cpu.mojo`.  Integrates a
# periodic Gaussian bump on a [0, 1]^2 triangulated mesh under
# v = (1, 1) for one full period, using the Float32 GPU kernels in
# `src/local_mesh_2d_gpu.mojo`.  Reports the final L2 error vs the
# analytic (translated-back-to-IC) reference and the wall-clock time
# for the step loop.
#
# This driver is the first end-to-end GPU simulation on the 2D stack;
# a ParaView frame-writer can follow.
# ======================================================================

from std.math import sqrt, exp
from std.sys import has_accelerator
from std.gpu.host import DeviceContext, DeviceBuffer
from std.time import perf_counter_ns
from src import mpi
from src.local_mesh_2d import LocalMesh2D
from src.local_mesh_2d_gpu import LocalMesh2DGpu, advection_rk_stage_2d
from src.reference_2d import (
    ReferenceElement2D, num_tri_nodes_2d, num_edge_nodes,
)
from src.reference_2d_gpu import ReferenceElement2DGpu


comptime P = 2
comptime NX = 32
comptime NY = 32
comptime LX = 1.0
comptime LY = 1.0
comptime VX: Float32 = 1.0
comptime VY: Float32 = 1.0
comptime T_FINAL: Float32 = 1.0
comptime CFL: Float32 = 0.3
comptime SIGMA: Float32 = 0.12
comptime CX: Float32 = 0.5
comptime CY: Float32 = 0.5


def _gauss(x: Float32, y: Float32) -> Float32:
    var dx = x - CX
    if dx >  LX * Float32(0.5): dx -= Float32(LX)
    if dx < -LX * Float32(0.5): dx += Float32(LX)
    var dy = y - CY
    if dy >  LY * Float32(0.5): dy -= Float32(LY)
    if dy < -LY * Float32(0.5): dy += Float32(LY)
    var s2 = SIGMA * SIGMA
    return exp(-(dx * dx + dy * dy) / (Float32(2.0) * s2))


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    mpi.init()
    var size = mpi.world_size()
    if size > 1:
        mpi.finalize()
        print("advection_gaussian_2d_gpu: runs at np=1 only")
        return

    print("advection_gaussian_2d_gpu (GPU, P=", P, ",", NX, "x", NY, ")")

    comptime NP_p = num_tri_nodes_2d(P)
    comptime NFP_e = num_edge_nodes(P)
    var ctx = DeviceContext()

    var host_mesh = LocalMesh2D[P](NX, NY, LX, LY)
    var host_re = ReferenceElement2D[P]()
    var mesh_host_copy = LocalMesh2D[P](NX, NY, LX, LY)  # keep for IC coords
    var gpu_mesh = LocalMesh2DGpu[P](ctx, host_mesh^)
    var gpu_re = ReferenceElement2DGpu[P](ctx, host_re)
    print("  elements:", gpu_mesh.num_elements, " faces:", gpu_mesh.num_faces,
          "  nodes/elem:", NP_p, "  total DOF:", gpu_mesh.num_elements * NP_p)

    # IC: Gaussian on host, uploaded to d_q.
    var n_q = gpu_mesh.num_elements * NP_p
    var host_q = List[Float32]()
    var host_ic = List[Float32]()
    for elem in range(gpu_mesh.num_elements):
        for nn in range(NP_p):
            var x = Float32(mesh_host_copy.elem_node_xyz[(elem * NP_p + nn) * 2 + 0])
            var y = Float32(mesh_host_copy.elem_node_xyz[(elem * NP_p + nn) * 2 + 1])
            var v = _gauss(x, y)
            host_q.append(v)
            host_ic.append(v)

    var d_q  = ctx.enqueue_create_buffer[DType.float32](n_q)
    var d_q1 = ctx.enqueue_create_buffer[DType.float32](n_q)
    var d_q2 = ctx.enqueue_create_buffer[DType.float32](n_q)
    var d_vol = ctx.enqueue_create_buffer[DType.float32](n_q)
    var d_rhs = ctx.enqueue_create_buffer[DType.float32](n_q)
    var d_fstar = ctx.enqueue_create_buffer[DType.float32](
        gpu_mesh.num_faces * NFP_e
    )

    var hbuf_q = ctx.enqueue_create_host_buffer[DType.float32](n_q)
    var hptr_q = hbuf_q.unsafe_ptr()
    for k in range(n_q):
        hptr_q[k] = host_q[k]
    ctx.enqueue_copy(d_q, hbuf_q)
    ctx.synchronize()

    # dt from CFL / DG scaling (same factor (2P+1) the CPU driver uses).
    var h = Float32(LX) / Float32(NX)
    var vmag = sqrt(VX * VX + VY * VY)
    var dt_est = CFL * h / (vmag * Float32(2 * P + 1))
    var num_steps = Int(T_FINAL / dt_est) + 1
    var dt = T_FINAL / Float32(num_steps)
    print("  dt=", dt, "  num_steps=", num_steps)

    var step_start = perf_counter_ns()
    for _ in range(num_steps):
        # Stage 1: q1 = q + dt L(q)
        advection_rk_stage_2d[P](
            ctx, gpu_mesh,
            gpu_re.d_Lift_ref.unsafe_ptr(), gpu_re.d_D_ref.unsafe_ptr(),
            d_q.unsafe_ptr(),
            d_q.unsafe_ptr(), d_q.unsafe_ptr(),
            d_q1.unsafe_ptr(),
            d_vol.unsafe_ptr(), d_fstar.unsafe_ptr(), d_rhs.unsafe_ptr(),
            VX, VY, Float32(0.0),
            Float32(1.0), Float32(0.0), Float32(1.0), dt,
        )
        # Stage 2: q2 = 3/4 q + 1/4 (q1 + dt L(q1))
        advection_rk_stage_2d[P](
            ctx, gpu_mesh,
            gpu_re.d_Lift_ref.unsafe_ptr(), gpu_re.d_D_ref.unsafe_ptr(),
            d_q1.unsafe_ptr(),
            d_q.unsafe_ptr(), d_q1.unsafe_ptr(),
            d_q2.unsafe_ptr(),
            d_vol.unsafe_ptr(), d_fstar.unsafe_ptr(), d_rhs.unsafe_ptr(),
            VX, VY, Float32(0.0),
            Float32(0.75), Float32(0.25), Float32(0.25), dt,
        )
        # Stage 3: q <- 1/3 q + 2/3 (q2 + dt L(q2))
        advection_rk_stage_2d[P](
            ctx, gpu_mesh,
            gpu_re.d_Lift_ref.unsafe_ptr(), gpu_re.d_D_ref.unsafe_ptr(),
            d_q2.unsafe_ptr(),
            d_q.unsafe_ptr(), d_q2.unsafe_ptr(),
            d_q.unsafe_ptr(),
            d_vol.unsafe_ptr(), d_fstar.unsafe_ptr(), d_rhs.unsafe_ptr(),
            VX, VY, Float32(0.0),
            Float32(1.0 / 3.0), Float32(2.0 / 3.0),
            Float32(2.0 / 3.0), dt,
        )
    ctx.synchronize()
    var step_end = perf_counter_ns()
    var wall_sec = Float64(step_end - step_start) * 1.0e-9
    print("  wall time:", wall_sec, "s")
    print("  throughput:", Float64(num_steps) / wall_sec, "steps/s")

    # Download final q; compare to IC (exact: translate by (vx T, vy T)
    # = (1, 1), which on a periodic [0, 1]^2 wraps back to the IC).
    ctx.enqueue_copy(hbuf_q, d_q)
    ctx.synchronize()
    var sum_sq: Float64 = 0.0
    var sum_ic: Float64 = 0.0
    for k in range(n_q):
        var e = Float64(hptr_q[k] - host_ic[k])
        sum_sq += e * e
        var ic = Float64(host_ic[k])
        sum_ic += ic * ic
    var l2 = sqrt(sum_sq / Float64(n_q))
    var l2_ic = sqrt(sum_ic / Float64(n_q))
    print("  L2 err =", l2, "  rel err =", l2 / l2_ic,
          "  (IC L2 =", l2_ic, ")")

    mpi.finalize()
