# ======================================================================
# advection_outflow_2d_gpu -- GPU advection with BC_OUTFLOW x 4
# ======================================================================
#
# Gaussian bump at (0.3, 0.3) with v = (1, 1); BC_OUTFLOW on all four
# sides.  The bump
# exits through the +x / +y boundaries by t ~ 1 -- mass drops to
# ~zero.  Exercises the BC_OUTFLOW path in
# `advection_face_flux_kernel_2d` end-to-end (the periodic / channel
# drivers don't touch this path).
#
# Writes 21 VTU frames + `output/solution_advout_gpu.pvd`.
# ======================================================================

from std.math import sqrt, exp
from std.pathlib import Path
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
from src.boundary import BoundaryConditions2D, BC_OUTFLOW
from src.vtu_2d import dump_vtu_2d_frame


comptime P = 2
comptime NX = 32
comptime NY = 32
comptime LX = 1.0
comptime LY = 1.0
comptime VX: Float32 = 1.0
comptime VY: Float32 = 1.0
comptime T_FINAL: Float32 = 1.0
comptime NUM_FRAMES = 20
comptime CFL: Float32 = 0.3

comptime CX: Float32 = 0.3
comptime CY: Float32 = 0.3
comptime SIGMA: Float32 = 0.1


def _frame_name(i: Int) raises -> String:
    var s = String("frame_advout_gpu_")
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
        print("advection_outflow_2d_gpu: runs at np=1 only")
        return

    print("advection_outflow_2d_gpu (BC_OUTFLOW x 4, P=", P, ")")

    comptime NP_p = num_tri_nodes_2d(P)
    comptime NFP_e = num_edge_nodes(P)
    var ctx = DeviceContext()

    var bcs = BoundaryConditions2D(
        BC_OUTFLOW, BC_OUTFLOW, BC_OUTFLOW, BC_OUTFLOW,
    )
    var host_mesh = LocalMesh2D[P](NX, NY, LX, LY, bcs)
    var host_re = ReferenceElement2D[P]()
    var mesh_coords = LocalMesh2D[P](NX, NY, LX, LY, bcs)
    var gpu_mesh = LocalMesh2DGpu[P](ctx, host_mesh^)
    var gpu_re = ReferenceElement2DGpu[P](ctx, host_re)
    print("  elements:", gpu_mesh.num_elements,
          " faces:", gpu_mesh.num_faces,
          "  (", gpu_mesh.num_faces - 3 * NX * NY, "boundary faces)")

    var n_q = gpu_mesh.num_elements * NP_p
    var host_q = List[Float32]()
    var mass_ic: Float64 = 0.0
    for elem in range(gpu_mesh.num_elements):
        for nn in range(NP_p):
            var x = mesh_coords.elem_node_xyz[(elem * NP_p + nn) * 2 + 0]
            var y = mesh_coords.elem_node_xyz[(elem * NP_p + nn) * 2 + 1]
            var dx = Float32(x) - CX
            var dy = Float32(y) - CY
            var r2 = dx * dx + dy * dy
            var v = exp(-r2 / (Float32(2.0) * SIGMA * SIGMA))
            host_q.append(v)
            mass_ic += Float64(v)

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

    var h = Float32(LX) / Float32(NX)
    var vmag = sqrt(VX * VX + VY * VY)
    var dt_est = CFL * h / (vmag * Float32(2 * P + 1))
    var steps_per_frame = Int(T_FINAL / (Float32(NUM_FRAMES) * dt_est)) + 1
    var total_steps = NUM_FRAMES * steps_per_frame
    var dt = T_FINAL / Float32(total_steps)
    print("  dt=", dt, "  steps/frame=", steps_per_frame,
          "  total steps=", total_steps)

    var q_scalar = List[Float64]()
    for _ in range(n_q):
        q_scalar.append(0.0)
    var paths = List[String]()
    var times = List[Float64]()

    for k in range(n_q):
        q_scalar[k] = Float64(host_q[k])
    var f0_path = String("output/") + _frame_name(0)
    dump_vtu_2d_frame[P](mesh_coords, q_scalar, f0_path, String("q"))
    paths.append(_frame_name(0))
    times.append(0.0)
    print("    t=0.0       mass =", mass_ic)

    var run_start = perf_counter_ns()
    var compute_ns: UInt = 0
    for fi in range(1, NUM_FRAMES + 1):
        var c_start = perf_counter_ns()
        for _ in range(steps_per_frame):
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
        var c_end = perf_counter_ns()
        compute_ns += c_end - c_start

        ctx.enqueue_copy(hbuf_q, d_q)
        ctx.synchronize()
        var mass_now: Float64 = 0.0
        for k in range(n_q):
            q_scalar[k] = Float64(hptr_q[k])
            mass_now += q_scalar[k]
        var t = Float64(fi) * Float64(steps_per_frame) * Float64(dt)
        var path_i = String("output/") + _frame_name(fi)
        dump_vtu_2d_frame[P](mesh_coords, q_scalar, path_i, String("q"))
        paths.append(_frame_name(fi))
        times.append(t)
        if fi % 4 == 0 or fi == NUM_FRAMES:
            print("    t=", t, " mass =", mass_now,
                  " (frac of IC =", mass_now / mass_ic, ")")
    var run_end = perf_counter_ns()

    var total_sec = Float64(run_end - run_start) * 1.0e-9
    var compute_sec = Float64(compute_ns) * 1.0e-9
    print("  compute time:", compute_sec, "s")
    print("  total time  :", total_sec, "s (incl. frame I/O)")
    print("  throughput  :", Float64(total_steps) / compute_sec,
          "steps/s (compute only)")

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
    Path("output/solution_advout_gpu.pvd").write_text(pvd)
    print("  wrote output/solution_advout_gpu.pvd +",
          NUM_FRAMES + 1, "VTU frames")

    mpi.finalize()
