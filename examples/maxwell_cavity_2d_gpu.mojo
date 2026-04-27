# ======================================================================
# maxwell_cavity_2d_gpu -- GPU 2D Maxwell PEC cavity standing wave
# ======================================================================
#
# 2D analog of the 3D `maxwell_cavity` example.  TM-mode (1, 1) standing
# wave in a unit-square PEC cavity:
#
#   Ez(x, y, t) = sin(pi x) sin(pi y) cos(omega t)
#   Bx(x, y, t) =  (pi / omega) sin(pi x) cos(pi y) sin(omega t)
#   By(x, y, t) = -(pi / omega) cos(pi x) sin(pi y) sin(omega t)
#   omega = c * pi * sqrt(2)
#
# Other components Ex = Ey = Bz = 0 throughout.  Period
#   T = 2 pi / omega = sqrt(2) / c
# After one period the exact solution returns to the IC; the residual
# rel-L2 vs IC is pure scheme dissipation.  The same setup is gated
# numerically by `bench_maxwell_cavity_2d` (rel L2 < 5e-4 at NX=NY=16).
#
# Output: NUM_FRAMES VTU snapshots of Ez +
# `output/solution_maxwell_cav2d_gpu.pvd` (drop into ParaView, or feed
# the PVD to `scripts/animate_2d.py` for an MP4).  Reports rel L2 vs
# IC at one full period plus wall time split between compute and I/O.
# ======================================================================

from std.math import sqrt, sin, pi
from std.sys import has_accelerator
from std.gpu.host import DeviceContext, DeviceBuffer
from std.time import perf_counter_ns
from src import mpi
from src.local_mesh_2d import LocalMesh2D
from src.local_mesh_2d_gpu import LocalMesh2DGpu
from src.local_mesh_2d_gpu_maxwell import maxwell_rk_stage_2d
from src.reference_2d import (
    ReferenceElement2D, num_tri_nodes_2d, num_edge_nodes,
)
from src.reference_2d_gpu import ReferenceElement2DGpu
from src.boundary import BoundaryConditions2D, BC_WALL
from src.vtu_2d import dump_vtu_2d_frame, dump_pvd_collection, vtu_frame_name


comptime P = 2
comptime NX = 32
comptime NY = 32
comptime LX = 1.0
comptime LY = 1.0
comptime C_LIGHT: Float32 = 1.0
comptime CFL = 0.2
comptime T_FINAL: Float64 = 1.41421356237   # one period = sqrt(2) / c
comptime NUM_FRAMES = 20


comptime FRAME_PREFIX = "frame_maxwell_cav2d_gpu_"


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    mpi.init()
    var size = mpi.world_size()
    if size > 1:
        mpi.finalize()
        print("maxwell_cavity_2d_gpu: runs at np=1 only")
        return

    print("maxwell_cavity_2d_gpu (GPU 2D Maxwell PEC cavity, P=", P,
          ",", NX, "x", NY, ")")

    comptime NP_p = num_tri_nodes_2d(P)
    comptime NFP_e = num_edge_nodes(P)
    comptime NC = 6
    var ctx = DeviceContext()

    var bcs = BoundaryConditions2D(BC_WALL, BC_WALL, BC_WALL, BC_WALL)
    var host_mesh = LocalMesh2D[P](NX, NY, LX, LY, bcs)
    var mesh_coords = LocalMesh2D[P](NX, NY, LX, LY, bcs)
    var host_re = ReferenceElement2D[P]()
    var gpu_mesh = LocalMesh2DGpu[P](ctx, host_mesh^)
    var gpu_re = ReferenceElement2DGpu[P](ctx, host_re)
    print("  elements:", gpu_mesh.num_elements,
          " faces:", gpu_mesh.num_faces,
          "  nodes/elem:", NP_p,
          "  total DOF:", gpu_mesh.num_elements * NP_p * NC)

    # IC: Ez = sin(pi x) sin(pi y), all other components 0.
    var k = pi
    var n_q = gpu_mesh.num_elements * NP_p * NC
    var host_q = List[Float32]()
    var host_ic = List[Float32]()
    for elem in range(gpu_mesh.num_elements):
        for nn in range(NP_p):
            var x = mesh_coords.elem_node_xyz[(elem * NP_p + nn) * 2 + 0]
            var y = mesh_coords.elem_node_xyz[(elem * NP_p + nn) * 2 + 1]
            var Ez = sin(k * x) * sin(k * y)
            host_q.append(Float32(0.0));  host_ic.append(Float32(0.0))   # Ex
            host_q.append(Float32(0.0));  host_ic.append(Float32(0.0))   # Ey
            host_q.append(Float32(Ez));   host_ic.append(Float32(Ez))    # Ez
            host_q.append(Float32(0.0));  host_ic.append(Float32(0.0))   # Bx
            host_q.append(Float32(0.0));  host_ic.append(Float32(0.0))   # By
            host_q.append(Float32(0.0));  host_ic.append(Float32(0.0))   # Bz

    var d_q  = ctx.enqueue_create_buffer[DType.float32](n_q)
    var d_q1 = ctx.enqueue_create_buffer[DType.float32](n_q)
    var d_q2 = ctx.enqueue_create_buffer[DType.float32](n_q)
    var d_fstar = ctx.enqueue_create_buffer[DType.float32](
        gpu_mesh.num_faces * NFP_e * NC
    )

    var hbuf_q = ctx.enqueue_create_host_buffer[DType.float32](n_q)
    var hptr_q = hbuf_q.unsafe_ptr()
    for i in range(n_q):
        hptr_q[i] = host_q[i]
    ctx.enqueue_copy(d_q, hbuf_q)
    ctx.synchronize()

    # dt: pick num_steps as a multiple of NUM_FRAMES so each frame
    # closes exactly on a stage-3 boundary.
    var h_cell = LX / Float64(NX)
    var dt_est = CFL * h_cell / (Float64(C_LIGHT) * Float64(2 * P + 1))
    var steps_per_frame = Int(T_FINAL / (Float64(NUM_FRAMES) * dt_est)) + 1
    var total_steps = NUM_FRAMES * steps_per_frame
    var dt = Float32(T_FINAL / Float64(total_steps))
    print("  dt=", dt, "  steps/frame=", steps_per_frame,
          "  total steps=", total_steps)

    var Ez_field = List[Float64]()
    for _ in range(gpu_mesh.num_elements * NP_p):
        Ez_field.append(0.0)
    var paths = List[String]()
    var times = List[Float64]()

    # Frame 0: dump IC before stepping.
    for elem in range(gpu_mesh.num_elements):
        for nn in range(NP_p):
            Ez_field[elem * NP_p + nn] = Float64(
                host_q[(elem * NP_p + nn) * NC + 2]   # component 2 = Ez
            )
    var f0_name = vtu_frame_name(FRAME_PREFIX, 0)
    dump_vtu_2d_frame[P](
        mesh_coords, Ez_field,
        String("output/") + f0_name, String("Ez"),
    )
    paths.append(f0_name)
    times.append(0.0)

    var run_start = perf_counter_ns()
    var compute_ns: UInt = 0
    for fi in range(1, NUM_FRAMES + 1):
        var c_start = perf_counter_ns()
        for _ in range(steps_per_frame):
            # Stage 1
            maxwell_rk_stage_2d[P](
                ctx, gpu_mesh,
                gpu_re.d_Lift_ref.unsafe_ptr(), gpu_re.d_D_ref.unsafe_ptr(),
                d_q.unsafe_ptr(),
                d_q.unsafe_ptr(), d_q.unsafe_ptr(),
                d_q1.unsafe_ptr(),
                d_fstar.unsafe_ptr(),
                C_LIGHT,
                Float32(1.0), Float32(0.0), Float32(1.0), dt,
            )
            # Stage 2
            maxwell_rk_stage_2d[P](
                ctx, gpu_mesh,
                gpu_re.d_Lift_ref.unsafe_ptr(), gpu_re.d_D_ref.unsafe_ptr(),
                d_q1.unsafe_ptr(),
                d_q.unsafe_ptr(), d_q1.unsafe_ptr(),
                d_q2.unsafe_ptr(),
                d_fstar.unsafe_ptr(),
                C_LIGHT,
                Float32(0.75), Float32(0.25), Float32(0.25), dt,
            )
            # Stage 3
            maxwell_rk_stage_2d[P](
                ctx, gpu_mesh,
                gpu_re.d_Lift_ref.unsafe_ptr(), gpu_re.d_D_ref.unsafe_ptr(),
                d_q2.unsafe_ptr(),
                d_q.unsafe_ptr(), d_q2.unsafe_ptr(),
                d_q.unsafe_ptr(),
                d_fstar.unsafe_ptr(),
                C_LIGHT,
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
                Ez_field[elem * NP_p + nn] = Float64(
                    hptr_q[(elem * NP_p + nn) * NC + 2]
                )
        var t = Float64(fi) * Float64(steps_per_frame) * Float64(dt)
        var fname = vtu_frame_name(FRAME_PREFIX, fi)
        dump_vtu_2d_frame[P](
            mesh_coords, Ez_field,
            String("output/") + fname, String("Ez"),
        )
        paths.append(fname)
        times.append(t)
        print("    frame", fi, "/", NUM_FRAMES, " t=", t)
    var run_end = perf_counter_ns()

    var total_sec = Float64(run_end - run_start) * 1.0e-9
    var compute_sec = Float64(compute_ns) * 1.0e-9
    print("  compute time:", compute_sec, "s")
    print("  total time  :", total_sec, "s (incl. frame I/O)")
    print("  throughput  :", Float64(total_steps) / compute_sec,
          "steps/s (compute only)")

    # Final relative L2 vs IC (one period of TM(1,1) returns to IC).
    var sum_sq: Float64 = 0.0
    var sum_ic: Float64 = 0.0
    for k_idx in range(n_q):
        var e = Float64(hptr_q[k_idx] - host_ic[k_idx])
        sum_sq += e * e
        var ic = Float64(host_ic[k_idx])
        sum_ic += ic * ic
    var l2 = sqrt(sum_sq / Float64(n_q))
    var l2_ic = sqrt(sum_ic / Float64(n_q))
    print("  L2 err =", l2, "  rel err =", l2 / l2_ic,
          "  (IC L2 =", l2_ic, ")")

    # PVD collection -- `scripts/animate_2d.py` walks this directly.
    dump_pvd_collection(
        String("output/solution_maxwell_cav2d_gpu.pvd"), paths, times,
    )
    print("  wrote output/solution_maxwell_cav2d_gpu.pvd +",
          NUM_FRAMES + 1, "VTU frames")

    mpi.finalize()
