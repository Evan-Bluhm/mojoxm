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
from src.reference_2d import ReferenceElement2D, num_tri_nodes_2d, num_edge_nodes
from src.reference_2d_gpu import ReferenceElement2DGpu
from src.boundary import BoundaryConditions2D, BC_WALL
from src.ssprk3 import ssprk3_stage_plans
from src.memory_report import MemoryReport, ThroughputReport, format_seconds
from src.vtu_2d import dump_vtu_2d_frame_multi, dump_pvd_collection, vtu_frame_name


comptime P = 2
comptime NX = 32
comptime NY = 32
comptime LX = 1.0
comptime LY = 1.0
comptime C_LIGHT: Float32 = 1.0
comptime CFL = 0.2
comptime T_FINAL: Float64 = 1.41421356237  # one period = sqrt(2) / c
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

    print("maxwell_cavity_2d_gpu (GPU 2D Maxwell PEC cavity, P=", P, ",", NX, "x", NY, ")")

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
    print("  elements:", gpu_mesh.num_elements, " faces:", gpu_mesh.num_faces, "  nodes/elem:", NP_p, "  total DOF:", gpu_mesh.num_elements * NP_p * NC)

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
            host_q.append(Float32(0.0))
            host_ic.append(Float32(0.0))  # Ex
            host_q.append(Float32(0.0))
            host_ic.append(Float32(0.0))  # Ey
            host_q.append(Float32(Ez))
            host_ic.append(Float32(Ez))  # Ez
            host_q.append(Float32(0.0))
            host_ic.append(Float32(0.0))  # Bx
            host_q.append(Float32(0.0))
            host_ic.append(Float32(0.0))  # By
            host_q.append(Float32(0.0))
            host_ic.append(Float32(0.0))  # Bz

    var d_q = ctx.enqueue_create_buffer[DType.float32](n_q)
    var d_q1 = ctx.enqueue_create_buffer[DType.float32](n_q)
    var d_q2 = ctx.enqueue_create_buffer[DType.float32](n_q)
    var d_fstar_count = gpu_mesh.num_faces * NFP_e * NC
    var d_fstar = ctx.enqueue_create_buffer[DType.float32](d_fstar_count)

    # WARPXM-style device-memory accounting (Maxwell: linear vacuum
    # flux, no BJ limiter; 2D np=1 only).
    MemoryReport(
        rk_stage_bytes=3 * n_q * 4,
        dg_operators_bytes=gpu_re.device_bytes(),
        limiter_bytes=0,
        mesh_connectivity_bytes=(gpu_mesh.device_bytes() + d_fstar_count * 4),
        halo_device_bytes=0,
        halo_pinned_bytes=0,
    ).print()

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
    print("  dt=", dt, "  steps/frame=", steps_per_frame, "  total steps=", total_steps)

    var Ez_field = List[Float64]()
    var Emag = List[Float64]()
    var Bmag = List[Float64]()
    for _ in range(gpu_mesh.num_elements * NP_p):
        Ez_field.append(0.0)
        Emag.append(0.0)
        Bmag.append(0.0)
    var paths = List[String]()
    var times = List[Float64]()

    var field_names = List[String]()
    field_names.append(String("Ez"))  # signed -- shows TM(1,1) phase
    field_names.append(String("|E|"))
    field_names.append(String("|B|"))

    # Vacuum Maxwell (NC=6) state layout: Ex, Ey, Ez, Bx, By, Bz
    @parameter
    def fill_derived(src: UnsafePointer[Float32, MutAnyOrigin]) raises:
        var n_nodes = gpu_mesh.num_elements * NP_p
        for i in range(n_nodes):
            var Ex = Float64(src[i * NC + 0])
            var Ey = Float64(src[i * NC + 1])
            var Ez = Float64(src[i * NC + 2])
            var Bx = Float64(src[i * NC + 3])
            var By = Float64(src[i * NC + 4])
            var Bz = Float64(src[i * NC + 5])
            Ez_field[i] = Ez
            Emag[i] = sqrt(Ex * Ex + Ey * Ey + Ez * Ez)
            Bmag[i] = sqrt(Bx * Bx + By * By + Bz * Bz)

    # Frame 0: dump IC before stepping.
    fill_derived(hptr_q)
    var fields = List[List[Float64]]()
    fields.append(Ez_field.copy())
    fields.append(Emag.copy())
    fields.append(Bmag.copy())
    var f0_name = vtu_frame_name(FRAME_PREFIX, 0)
    dump_vtu_2d_frame_multi[P](mesh_coords, field_names, fields, String("output/") + f0_name)
    paths.append(f0_name)
    times.append(0.0)

    var run_start = perf_counter_ns()
    var compute_ns: UInt = 0
    var stage_plans = ssprk3_stage_plans(d_q.unsafe_ptr(), d_q1.unsafe_ptr(), d_q2.unsafe_ptr())
    for fi in range(1, NUM_FRAMES + 1):
        var c_start = perf_counter_ns()
        for _ in range(steps_per_frame):
            for stage in stage_plans:
                maxwell_rk_stage_2d[P](
                    ctx,
                    gpu_mesh,
                    gpu_re.d_Lift_ref.unsafe_ptr(),
                    gpu_re.d_D_ref.unsafe_ptr(),
                    stage.q_in,
                    stage.q_a,
                    stage.q_b,
                    stage.q_out,
                    d_fstar.unsafe_ptr(),
                    C_LIGHT,
                    stage.a,
                    stage.b,
                    stage.c,
                    dt,
                )
        ctx.synchronize()
        var c_end = perf_counter_ns()
        compute_ns += c_end - c_start

        ctx.enqueue_copy(hbuf_q, d_q)
        ctx.synchronize()
        fill_derived(hptr_q)
        var fi_fields = List[List[Float64]]()
        fi_fields.append(Ez_field.copy())
        fi_fields.append(Emag.copy())
        fi_fields.append(Bmag.copy())
        var t = Float64(fi) * Float64(steps_per_frame) * Float64(dt)
        var fname = vtu_frame_name(FRAME_PREFIX, fi)
        dump_vtu_2d_frame_multi[P](mesh_coords, field_names, fi_fields, String("output/") + fname)
        paths.append(fname)
        times.append(t)
        print("    frame", fi, "/", NUM_FRAMES, " t=", t)
    var run_end = perf_counter_ns()

    var total_sec = Float64(run_end - run_start) * 1.0e-9
    var compute_sec = Float64(compute_ns) * 1.0e-9
    print("  total time  :", format_seconds(total_sec), "(incl. frame I/O)")
    ThroughputReport(num_steps=total_steps, wall_seconds=compute_sec, dof_count=gpu_mesh.num_elements * NP_p * NC, state_bytes_per_step=8 * n_q * 4).print()

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
    print("  L2 err =", l2, "  rel err =", l2 / l2_ic, "  (IC L2 =", l2_ic, ")")

    # PVD collection -- `scripts/animate_2d.py` walks this directly.
    dump_pvd_collection(String("output/solution_maxwell_cav2d_gpu.pvd"), paths, times)
    print("  wrote output/solution_maxwell_cav2d_gpu.pvd +", NUM_FRAMES + 1, "VTU frames")

    mpi.finalize()
