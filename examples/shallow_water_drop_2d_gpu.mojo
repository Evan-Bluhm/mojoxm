# ======================================================================
# shallow_water_drop_2d_gpu -- GPU 2D DG shallow-water drop
# ======================================================================
#
# GPU analog of `shallow_water_drop_2d_cpu.mojo`.  A circular elevated
# water column collapses under gravity on a [0, 1]^2 periodic basin.
# Waves propagate outward, wrap around, and interfere -- a non-trivial
# multi-component run with a discontinuous-like IC (smooth Gaussian
# bump) to exercise the SW GPU kernels.
#
# No VTU output here; reports wall time, throughput, and mass / mean-h
# conservation (should be bit-stable on periodic meshes).
# ======================================================================

from std.math import sqrt, exp, pi
from std.sys import has_accelerator
from std.gpu.host import DeviceContext, DeviceBuffer
from std.time import perf_counter_ns
from src import mpi
from src.local_mesh_2d import LocalMesh2D
from src.local_mesh_2d_gpu import LocalMesh2DGpu, sw_rk_stage_2d
from src.reference_2d import (
    ReferenceElement2D, num_tri_nodes_2d, num_edge_nodes,
)
from src.reference_2d_gpu import ReferenceElement2DGpu


comptime P = 2
comptime NX = 48
comptime NY = 48
comptime LX = 1.0
comptime LY = 1.0
comptime T_FINAL = 0.4
comptime CFL = 0.2

comptime G = 9.81
comptime H_BG = 1.0
comptime H_AMP = 0.4
comptime SIGMA = 0.08
comptime DROP_CX = 0.5
comptime DROP_CY = 0.5


def _periodic_delta(a: Float64, b: Float64, L: Float64) -> Float64:
    var d = a - b
    if d >  L * 0.5: d -= L
    if d < -L * 0.5: d += L
    return d


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    mpi.init()
    var size = mpi.world_size()
    if size > 1:
        mpi.finalize()
        print("shallow_water_drop_2d_gpu: runs at np=1 only")
        return

    print("shallow_water_drop_2d_gpu (GPU, P=", P, ",", NX, "x", NY, ")")

    comptime NP_p = num_tri_nodes_2d(P)
    comptime NFP_e = num_edge_nodes(P)
    comptime NC = 3
    var ctx = DeviceContext()

    var host_mesh = LocalMesh2D[P](NX, NY, LX, LY)
    var host_re = ReferenceElement2D[P]()
    var mesh_coords = LocalMesh2D[P](NX, NY, LX, LY)
    var gpu_mesh = LocalMesh2DGpu[P](ctx, host_mesh^)
    var gpu_re = ReferenceElement2DGpu[P](ctx, host_re)
    print("  elements:", gpu_mesh.num_elements,
          " faces:", gpu_mesh.num_faces,
          "  nodes/elem:", NP_p,
          "  total DOF:", gpu_mesh.num_elements * NP_p * NC)

    var n_q = gpu_mesh.num_elements * NP_p * NC
    var host_q = List[Float32]()
    var total_h0: Float64 = 0.0
    for elem in range(gpu_mesh.num_elements):
        for nn in range(NP_p):
            var x = mesh_coords.elem_node_xyz[(elem * NP_p + nn) * 2 + 0]
            var y = mesh_coords.elem_node_xyz[(elem * NP_p + nn) * 2 + 1]
            var dx = _periodic_delta(x, DROP_CX, LX)
            var dy = _periodic_delta(y, DROP_CY, LY)
            var r2 = dx * dx + dy * dy
            var h = H_BG + H_AMP * exp(-r2 / (2.0 * SIGMA * SIGMA))
            host_q.append(Float32(h))
            host_q.append(Float32(0.0))
            host_q.append(Float32(0.0))
            total_h0 += h

    var d_q  = ctx.enqueue_create_buffer[DType.float32](n_q)
    var d_q1 = ctx.enqueue_create_buffer[DType.float32](n_q)
    var d_q2 = ctx.enqueue_create_buffer[DType.float32](n_q)
    var d_vol = ctx.enqueue_create_buffer[DType.float32](n_q)
    var d_rhs = ctx.enqueue_create_buffer[DType.float32](n_q)
    var d_fstar = ctx.enqueue_create_buffer[DType.float32](
        gpu_mesh.num_faces * NFP_e * NC
    )

    var hbuf_q = ctx.enqueue_create_host_buffer[DType.float32](n_q)
    var hptr_q = hbuf_q.unsafe_ptr()
    for k in range(n_q):
        hptr_q[k] = host_q[k]
    ctx.enqueue_copy(d_q, hbuf_q)
    ctx.synchronize()

    var h_cell = LX / Float64(NX)
    var c_peak = sqrt(G * (H_BG + H_AMP))
    var dt_est = CFL * h_cell / (c_peak * Float64(2 * P + 1))
    var num_steps = Int(T_FINAL / dt_est) + 1
    var dt = Float32(T_FINAL / Float64(num_steps))
    print("  dt=", dt, "  num_steps=", num_steps,
          "  c_peak=", c_peak)

    var g = Float32(G)
    var min_h = Float32(1.0e-6)

    var step_start = perf_counter_ns()
    for _ in range(num_steps):
        sw_rk_stage_2d[P](
            ctx, gpu_mesh,
            gpu_re.d_Lift_ref.unsafe_ptr(), gpu_re.d_D_ref.unsafe_ptr(),
            d_q.unsafe_ptr(),
            d_q.unsafe_ptr(), d_q.unsafe_ptr(),
            d_q1.unsafe_ptr(),
            d_vol.unsafe_ptr(), d_fstar.unsafe_ptr(), d_rhs.unsafe_ptr(),
            g, min_h,
            Float32(0.0), Float32(0.0), Float32(0.0),
            Float32(1.0), Float32(0.0), Float32(1.0), dt,
        )
        sw_rk_stage_2d[P](
            ctx, gpu_mesh,
            gpu_re.d_Lift_ref.unsafe_ptr(), gpu_re.d_D_ref.unsafe_ptr(),
            d_q1.unsafe_ptr(),
            d_q.unsafe_ptr(), d_q1.unsafe_ptr(),
            d_q2.unsafe_ptr(),
            d_vol.unsafe_ptr(), d_fstar.unsafe_ptr(), d_rhs.unsafe_ptr(),
            g, min_h,
            Float32(0.0), Float32(0.0), Float32(0.0),
            Float32(0.75), Float32(0.25), Float32(0.25), dt,
        )
        sw_rk_stage_2d[P](
            ctx, gpu_mesh,
            gpu_re.d_Lift_ref.unsafe_ptr(), gpu_re.d_D_ref.unsafe_ptr(),
            d_q2.unsafe_ptr(),
            d_q.unsafe_ptr(), d_q2.unsafe_ptr(),
            d_q.unsafe_ptr(),
            d_vol.unsafe_ptr(), d_fstar.unsafe_ptr(), d_rhs.unsafe_ptr(),
            g, min_h,
            Float32(0.0), Float32(0.0), Float32(0.0),
            Float32(1.0 / 3.0), Float32(2.0 / 3.0),
            Float32(2.0 / 3.0), dt,
        )
    ctx.synchronize()
    var step_end = perf_counter_ns()
    var wall_sec = Float64(step_end - step_start) * 1.0e-9
    print("  wall time:", wall_sec, "s")
    print("  throughput:", Float64(num_steps) / wall_sec, "steps/s")

    ctx.enqueue_copy(hbuf_q, d_q)
    ctx.synchronize()
    var total_h: Float64 = 0.0
    var h_min = Float64(hptr_q[0])
    var h_max = Float64(hptr_q[0])
    for elem in range(gpu_mesh.num_elements):
        for nn in range(NP_p):
            var h = Float64(hptr_q[(elem * NP_p + nn) * 3 + 0])
            total_h += h
            if h < h_min: h_min = h
            if h > h_max: h_max = h
    var mean_h = total_h / Float64(gpu_mesh.num_elements * NP_p)
    var mean_h0 = total_h0 / Float64(gpu_mesh.num_elements * NP_p)
    print("  mean h (t=0):", mean_h0,
          "  mean h (final):", mean_h)
    print("  h range (final): [", h_min, ",", h_max, "]")

    mpi.finalize()
