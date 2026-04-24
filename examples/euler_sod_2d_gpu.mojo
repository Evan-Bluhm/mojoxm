# ======================================================================
# euler_sod_2d_gpu -- GPU 2D Sod shock tube (HLLC + BJ limiter)
# ======================================================================
#
# 1D Sod shock tube (rho_L=1, p_L=1, rho_R=0.125, p_R=0.1, u=v=0)
# lifted to a 2D strip
# on [0, 1] x [0, 0.125], with BC_OUTFLOW on the x ends and BC_WALL on
# the y ends.  Runs to T=0.20, the classical Sod horizon.
#
# Uses the HLLC Riemann solver (less dissipative than Rusanov on
# contact waves) + the Barth-Jespersen / Venkat-smoothed limiter
# between every SSPRK3 stage.  Together they keep Gibbs overshoots
# bounded even with a discontinuous initial jump; the CPU driver has
# to tanh-smooth the IC to stay stable because the CPU limiter wasn't
# enabled by default.
#
# Writes 21 density VTU frames + `output/solution_sod2d_gpu.pvd`.
# ======================================================================

from std.math import sqrt, tanh
from std.pathlib import Path
from std.sys import has_accelerator
from std.gpu.host import DeviceContext, DeviceBuffer
from std.time import perf_counter_ns
from src import mpi
from src.local_mesh_2d import LocalMesh2D
from src.local_mesh_2d_gpu import (
    LocalMesh2DGpu, euler_rk_stage_hllc_2d, bj_limit_full_2d,
)
from src.reference_2d import (
    ReferenceElement2D, num_tri_nodes_2d, num_edge_nodes,
)
from src.reference_2d_gpu import ReferenceElement2DGpu
from src.boundary import BoundaryConditions2D, BC_WALL, BC_OUTFLOW
from src.vtu_2d import dump_vtu_2d_frame


comptime P = 2
comptime NX = 128
comptime NY = 16
comptime LX = 1.0
comptime LY = 0.125
comptime GAMMA = 1.4
comptime RHO_L = 1.0
comptime P_L   = 1.0
comptime RHO_R = 0.125
comptime P_R   = 0.1
comptime T_FINAL = 0.20
comptime NUM_FRAMES = 20
comptime CFL = 0.15
comptime VENKAT_EPS = 0.1


def _frame_name(i: Int) raises -> String:
    var s = String("frame_sod2d_gpu_")
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
        print("euler_sod_2d_gpu: runs at np=1 only")
        return

    print("euler_sod_2d_gpu (Sod shock tube, HLLC+BJ, P=", P, ")")
    print("  mesh:", NX, "x", NY, " domain:", LX, "x", LY)
    print("  BCs: x-ends OUTFLOW, y-ends WALL")

    comptime NP_p = num_tri_nodes_2d(P)
    comptime NFP_e = num_edge_nodes(P)
    comptime NC = 4
    var ctx = DeviceContext()

    var bcs = BoundaryConditions2D(
        BC_OUTFLOW, BC_OUTFLOW,
        BC_WALL,    BC_WALL,
    )
    var host_mesh = LocalMesh2D[P](NX, NY, LX, LY, bcs)
    var host_re = ReferenceElement2D[P]()
    var mesh_coords = LocalMesh2D[P](NX, NY, LX, LY, bcs)
    var gpu_mesh = LocalMesh2DGpu[P](ctx, host_mesh^)
    var gpu_re = ReferenceElement2DGpu[P](ctx, host_re)
    print("  elements:", gpu_mesh.num_elements,
          " faces:", gpu_mesh.num_faces)

    # IC: tanh-smoothed over ~8 cells so P>=2 can resolve the jump.
    # The BJ limiter + HLLC can handle a fully discontinuous IC too,
    # but the smoothed version keeps rho / p clamped away from the
    # floors on the first few steps and gives a cleaner comparison
    # with the CPU driver.
    var smooth_width = 8.0 * (LX / Float64(NX))
    var n_q = gpu_mesh.num_elements * NP_p * NC
    var host_q = List[Float32]()
    for elem in range(gpu_mesh.num_elements):
        for nn in range(NP_p):
            var x = mesh_coords.elem_node_xyz[(elem * NP_p + nn) * 2 + 0]
            var s = 0.5 * (tanh((x - 0.5 * LX) / smooth_width) + 1.0)
            var rho = RHO_L + s * (RHO_R - RHO_L)
            var p   = P_L   + s * (P_R   - P_L)
            var E = p / (GAMMA - 1.0)
            host_q.append(Float32(rho))
            host_q.append(Float32(0.0))
            host_q.append(Float32(0.0))
            host_q.append(Float32(E))

    var d_q  = ctx.enqueue_create_buffer[DType.float32](n_q)
    var d_q1 = ctx.enqueue_create_buffer[DType.float32](n_q)
    var d_q2 = ctx.enqueue_create_buffer[DType.float32](n_q)
    var d_vol = ctx.enqueue_create_buffer[DType.float32](n_q)
    var d_rhs = ctx.enqueue_create_buffer[DType.float32](n_q)
    var d_fstar = ctx.enqueue_create_buffer[DType.float32](
        gpu_mesh.num_faces * NFP_e * NC
    )
    var d_cell_avg = ctx.enqueue_create_buffer[DType.float32](
        gpu_mesh.num_elements * NC
    )

    var hbuf_q = ctx.enqueue_create_host_buffer[DType.float32](n_q)
    var hptr_q = hbuf_q.unsafe_ptr()
    for k in range(n_q):
        hptr_q[k] = host_q[k]
    ctx.enqueue_copy(d_q, hbuf_q)
    ctx.synchronize()

    var c_peak = sqrt(GAMMA * P_L / RHO_L)
    var h_cell = LX / Float64(NX)
    var dt_est = CFL * h_cell / (2.0 * c_peak * Float64(2 * P + 1))
    var steps_per_frame = Int(T_FINAL / (Float64(NUM_FRAMES) * dt_est)) + 1
    var total_steps = NUM_FRAMES * steps_per_frame
    var dt = Float32(T_FINAL / Float64(total_steps))
    print("  dt=", dt, "  steps/frame=", steps_per_frame,
          "  total steps=", total_steps)

    var gamma = Float32(GAMMA)
    var min_rho = Float32(1.0e-6)
    var min_p   = Float32(1.0e-6)
    var venkat_eps = Float32(VENKAT_EPS)

    var density = List[Float64]()
    for _ in range(gpu_mesh.num_elements * NP_p):
        density.append(0.0)
    var paths = List[String]()
    var times = List[Float64]()

    for elem in range(gpu_mesh.num_elements):
        for nn in range(NP_p):
            density[elem * NP_p + nn] = Float64(host_q[(elem * NP_p + nn) * NC + 0])
    var f0_path = String("output/") + _frame_name(0)
    dump_vtu_2d_frame[P](mesh_coords, density, f0_path, String("rho"))
    paths.append(_frame_name(0))
    times.append(0.0)

    var run_start = perf_counter_ns()
    var compute_ns: UInt = 0
    for fi in range(1, NUM_FRAMES + 1):
        var c_start = perf_counter_ns()
        for _ in range(steps_per_frame):
            # Stage 1 + limit
            euler_rk_stage_hllc_2d[P](
                ctx, gpu_mesh,
                gpu_re.d_Lift_ref.unsafe_ptr(), gpu_re.d_D_ref.unsafe_ptr(),
                d_q.unsafe_ptr(),
                d_q.unsafe_ptr(), d_q.unsafe_ptr(),
                d_q1.unsafe_ptr(),
                d_vol.unsafe_ptr(), d_fstar.unsafe_ptr(), d_rhs.unsafe_ptr(),
                gamma, min_rho, min_p,
                Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0),
                Float32(1.0), Float32(0.0), Float32(1.0), dt,
            )
            bj_limit_full_2d[P, NC](
                ctx, gpu_mesh, d_q1.unsafe_ptr(), d_cell_avg.unsafe_ptr(),
                venkat_eps,
            )
            # Stage 2 + limit
            euler_rk_stage_hllc_2d[P](
                ctx, gpu_mesh,
                gpu_re.d_Lift_ref.unsafe_ptr(), gpu_re.d_D_ref.unsafe_ptr(),
                d_q1.unsafe_ptr(),
                d_q.unsafe_ptr(), d_q1.unsafe_ptr(),
                d_q2.unsafe_ptr(),
                d_vol.unsafe_ptr(), d_fstar.unsafe_ptr(), d_rhs.unsafe_ptr(),
                gamma, min_rho, min_p,
                Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0),
                Float32(0.75), Float32(0.25), Float32(0.25), dt,
            )
            bj_limit_full_2d[P, NC](
                ctx, gpu_mesh, d_q2.unsafe_ptr(), d_cell_avg.unsafe_ptr(),
                venkat_eps,
            )
            # Stage 3 + limit
            euler_rk_stage_hllc_2d[P](
                ctx, gpu_mesh,
                gpu_re.d_Lift_ref.unsafe_ptr(), gpu_re.d_D_ref.unsafe_ptr(),
                d_q2.unsafe_ptr(),
                d_q.unsafe_ptr(), d_q2.unsafe_ptr(),
                d_q.unsafe_ptr(),
                d_vol.unsafe_ptr(), d_fstar.unsafe_ptr(), d_rhs.unsafe_ptr(),
                gamma, min_rho, min_p,
                Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0),
                Float32(1.0 / 3.0), Float32(2.0 / 3.0),
                Float32(2.0 / 3.0), dt,
            )
            bj_limit_full_2d[P, NC](
                ctx, gpu_mesh, d_q.unsafe_ptr(), d_cell_avg.unsafe_ptr(),
                venkat_eps,
            )
        ctx.synchronize()
        var c_end = perf_counter_ns()
        compute_ns += c_end - c_start

        ctx.enqueue_copy(hbuf_q, d_q)
        ctx.synchronize()
        for elem in range(gpu_mesh.num_elements):
            for nn in range(NP_p):
                density[elem * NP_p + nn] = Float64(
                    hptr_q[(elem * NP_p + nn) * NC + 0]
                )
        var t = Float64(fi) * Float64(steps_per_frame) * Float64(dt)
        var path_i = String("output/") + _frame_name(fi)
        dump_vtu_2d_frame[P](mesh_coords, density, path_i, String("rho"))
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

    # End-state density check: left side should still be ~rho_L
    # (rarefaction head hasn't reached the wall at T=0.2), right side
    # still ~rho_R (shock hasn't reached the wall either).
    var rho_left: Float64 = 0.0
    var rho_right: Float64 = 0.0
    var n_left = 0
    var n_right = 0
    var rho_min = Float64(hptr_q[0])
    var rho_max = Float64(hptr_q[0])
    for elem in range(gpu_mesh.num_elements):
        for nn in range(NP_p):
            var x = mesh_coords.elem_node_xyz[(elem * NP_p + nn) * 2 + 0]
            var rho = Float64(hptr_q[(elem * NP_p + nn) * NC + 0])
            if x < 0.02:
                rho_left += rho
                n_left += 1
            if x > LX - 0.02:
                rho_right += rho
                n_right += 1
            if rho < rho_min: rho_min = rho
            if rho > rho_max: rho_max = rho
    if n_left > 0: rho_left /= Float64(n_left)
    if n_right > 0: rho_right /= Float64(n_right)
    print("  density @ -x boundary =", rho_left, " (expected ~", RHO_L, ")")
    print("  density @ +x boundary =", rho_right, " (expected ~", RHO_R, ")")
    print("  rho range over domain = [", rho_min, ",", rho_max, "]")
    # With the limiter, rho should stay within a few percent of
    # [rho_R, rho_L] = [0.125, 1.0].  Without it the Gibbs overshoot
    # would push it above 1.0 and potentially below rho_R.
    print("  rho overshoot ratio = (rho_max - rho_L) / rho_L =",
          (rho_max - RHO_L) / RHO_L)

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
    Path("output/solution_sod2d_gpu.pvd").write_text(pvd)
    print("  wrote output/solution_sod2d_gpu.pvd +",
          NUM_FRAMES + 1, "VTU frames")

    mpi.finalize()
