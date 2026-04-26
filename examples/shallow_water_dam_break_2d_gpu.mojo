# ======================================================================
# shallow_water_dam_break_2d_gpu -- GPU 2D SW dam break in a closed box
# ======================================================================
#
# 2D Riemann problem
#   h(x, y, 0) = h_L for x < Lx/2,  h_R otherwise;  u = v = 0
# with WALL boundaries on all four sides.  The dam collapses into a
# rightward bore + leftward rarefaction that reflect off the walls
# -- a non-trivial non-periodic SW run that exercises the BC_WALL
# path in sw_face_flux_kernel_2d end-to-end.
#
# Writes 21 depth VTU frames + `output/solution_dam_gpu.pvd`.
# ======================================================================

from std.math import sqrt
from std.pathlib import Path
from std.sys import has_accelerator
from std.gpu.host import DeviceContext, DeviceBuffer
from std.time import perf_counter_ns
from src import mpi
from src.local_mesh_2d import LocalMesh2D
from src.local_mesh_2d_gpu import LocalMesh2DGpu
from src.local_mesh_2d_gpu_sw import sw_rk_stage_2d
from src.reference_2d import (
    ReferenceElement2D, num_tri_nodes_2d, num_edge_nodes,
)
from src.reference_2d_gpu import ReferenceElement2DGpu
from src.boundary import BoundaryConditions2D, BC_WALL
from src.vtu_2d import dump_vtu_2d_frame


comptime P = 2
comptime NX = 64
comptime NY = 16
comptime LX = 2.0
comptime LY = 0.5
comptime G   = 9.81
comptime H_L = 2.0
comptime H_R = 1.0
comptime T_FINAL = 0.5
comptime NUM_FRAMES = 20
comptime CFL = 0.2


def _frame_name(i: Int) raises -> String:
    var s = String("frame_dam_gpu_")
    var idx = String(i)
    for _ in range(5 - idx.byte_length()):
        s += "0"
    s += idx
    s += ".vtu"
    return s^


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    mpi.init()
    var size = mpi.world_size()
    if size > 1:
        mpi.finalize()
        print("shallow_water_dam_break_2d_gpu: runs at np=1 only")
        return

    print("shallow_water_dam_break_2d_gpu (walls on all 4 sides)")
    print("  mesh:", NX, "x", NY, " domain:", LX, "x", LY,
          " (h_L=", H_L, ", h_R=", H_R, ")")

    comptime NP_p = num_tri_nodes_2d(P)
    comptime NFP_e = num_edge_nodes(P)
    comptime NC = 3
    var ctx = DeviceContext()

    var bcs = BoundaryConditions2D(BC_WALL, BC_WALL, BC_WALL, BC_WALL)
    var host_mesh = LocalMesh2D[P](NX, NY, LX, LY, bcs)
    var host_re = ReferenceElement2D[P]()
    var mesh_coords = LocalMesh2D[P](NX, NY, LX, LY, bcs)
    var gpu_mesh = LocalMesh2DGpu[P](ctx, host_mesh^)
    var gpu_re = ReferenceElement2DGpu[P](ctx, host_re)
    print("  elements:", gpu_mesh.num_elements,
          "  faces:", gpu_mesh.num_faces)

    var n_q = gpu_mesh.num_elements * NP_p * NC
    var host_q = List[Float32]()
    var total_h0: Float64 = 0.0
    for elem in range(gpu_mesh.num_elements):
        for nn in range(NP_p):
            var x = mesh_coords.elem_node_xyz[(elem * NP_p + nn) * 2 + 0]
            var h: Float64 = H_L if x < LX * 0.5 else H_R
            host_q.append(Float32(h))
            host_q.append(Float32(0.0))
            host_q.append(Float32(0.0))
            total_h0 += h
    var mean_h0 = total_h0 / Float64(gpu_mesh.num_elements * NP_p)
    print("  initial mean h =", mean_h0)

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

    var c_peak = sqrt(G * H_L)
    var h_cell = LX / Float64(NX)
    var dt_est = CFL * h_cell / (c_peak * Float64(2 * P + 1))
    var steps_per_frame = Int(T_FINAL / (Float64(NUM_FRAMES) * dt_est)) + 1
    var total_steps = NUM_FRAMES * steps_per_frame
    var dt = Float32(T_FINAL / Float64(total_steps))
    print("  dt=", dt, "  steps/frame=", steps_per_frame,
          "  total steps=", total_steps)

    var g = Float32(G)
    var min_h = Float32(1.0e-6)

    var depth = List[Float64]()
    for _ in range(gpu_mesh.num_elements * NP_p):
        depth.append(0.0)
    var paths = List[String]()
    var times = List[Float64]()

    for elem in range(gpu_mesh.num_elements):
        for nn in range(NP_p):
            depth[elem * NP_p + nn] = Float64(host_q[(elem * NP_p + nn) * NC + 0])
    var f0_path = String("output/") + _frame_name(0)
    dump_vtu_2d_frame[P](mesh_coords, depth, f0_path, String("h"))
    paths.append(_frame_name(0))
    times.append(0.0)

    var run_start = perf_counter_ns()
    var compute_ns: UInt = 0
    for fi in range(1, NUM_FRAMES + 1):
        var c_start = perf_counter_ns()
        for _ in range(steps_per_frame):
            sw_rk_stage_2d[P](
                ctx, gpu_mesh,
                gpu_re.d_Lift_ref.unsafe_ptr(), gpu_re.d_D_ref.unsafe_ptr(),
                d_q.unsafe_ptr(),
                d_q.unsafe_ptr(), d_q.unsafe_ptr(),
                d_q1.unsafe_ptr(),
                d_fstar.unsafe_ptr(),
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
                d_fstar.unsafe_ptr(),
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
                d_fstar.unsafe_ptr(),
                g, min_h,
                Float32(0.0), Float32(0.0), Float32(0.0),
                Float32(1.0 / 3.0), Float32(2.0 / 3.0),
                Float32(2.0 / 3.0), dt,
            )
        ctx.synchronize()
        var c_end = perf_counter_ns()
        compute_ns += c_end - c_start

        ctx.enqueue_copy(hbuf_q, d_q)
        ctx.synchronize()
        for elem in range(gpu_mesh.num_elements):
            for nn in range(NP_p):
                depth[elem * NP_p + nn] = Float64(
                    hptr_q[(elem * NP_p + nn) * NC + 0]
                )
        var t = Float64(fi) * Float64(steps_per_frame) * Float64(dt)
        var path_i = String("output/") + _frame_name(fi)
        dump_vtu_2d_frame[P](mesh_coords, depth, path_i, String("h"))
        paths.append(_frame_name(fi))
        times.append(t)
        print("    frame", fi, "/", NUM_FRAMES, " t=", t)
    var run_end = perf_counter_ns()

    var total_sec = Float64(run_end - run_start) * 1.0e-9
    var compute_sec = Float64(compute_ns) * 1.0e-9
    print("  compute time:", compute_sec, "s")
    print("  total time  :", total_sec, "s (incl. frame I/O)")
    print("  throughput  :", Float64(total_steps) / compute_sec,
          "steps/s (compute only)")

    var total_h: Float64 = 0.0
    var max_h: Float64 = 0.0
    var min_h_f: Float64 = 1.0e30
    for elem in range(gpu_mesh.num_elements):
        for nn in range(NP_p):
            var h = Float64(hptr_q[(elem * NP_p + nn) * 3 + 0])
            total_h += h
            if h > max_h: max_h = h
            if h < min_h_f: min_h_f = h
    var mean_hf = total_h / Float64(gpu_mesh.num_elements * NP_p)
    var mass_err = (mean_hf - mean_h0) / mean_h0
    print("  final mean h =", mean_hf,
          "  relative drift =", mass_err,
          "  (h range [", min_h_f, ",", max_h, "])")

    var pvd = String()
    pvd += '<?xml version="1.0"?>\n'
    pvd += ('<VTKFile type="Collection" version="0.1"'
            ' byte_order="LittleEndian">\n')
    pvd += '<Collection>\n'
    for i in range(len(paths)):
        pvd += '<DataSet timestep="'
        pvd += String(times[i])
        pvd += '" group="" part="0" file="'
        pvd += paths[i]
        pvd += '"/>\n'
    pvd += '</Collection>\n'
    pvd += '</VTKFile>\n'
    Path("output/solution_dam_gpu.pvd").write_text(pvd)
    print("  wrote output/solution_dam_gpu.pvd +",
          NUM_FRAMES + 1, "VTU frames")

    mpi.finalize()
