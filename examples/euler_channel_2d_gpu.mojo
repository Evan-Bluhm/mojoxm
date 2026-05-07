# ======================================================================
# euler_channel_2d_gpu -- GPU Mach-2 supersonic channel
# ======================================================================
#
# Exercises the non-periodic BC paths in `euler_face_flux_kernel_2d`
# (BC_INFLOW on -x, BC_OUTFLOW on +x, BC_WALL on +/- y) end-to-end.
# The IC is the inflow state everywhere, so the analytic solution is
# steady -- `rho_max_drift` reports how much the GPU scheme departs
# from the exact constant.
#
# Writes NUM_FRAMES density snapshots + `solution_channel_gpu.pvd`
# into `output/`.
# ======================================================================

from std.math import sqrt
from std.sys import has_accelerator
from std.gpu.host import DeviceContext, DeviceBuffer
from std.time import perf_counter_ns
from src import mpi
from src.local_mesh_2d import LocalMesh2D
from src.local_mesh_2d_gpu import LocalMesh2DGpu
from src.local_mesh_2d_gpu_euler import euler_rk_stage_2d
from src.reference_2d import (
    ReferenceElement2D,
    num_tri_nodes_2d,
    num_edge_nodes,
)
from src.reference_2d_gpu import ReferenceElement2DGpu
from src.boundary import (
    BoundaryConditions2D,
    BC_WALL,
    BC_OUTFLOW,
    BC_INFLOW,
)
from src.ssprk3 import ssprk3_stage_plans
from src.memory_report import MemoryReport, ThroughputReport
from src.vtu_2d import (
    dump_vtu_2d_frame_multi,
    dump_pvd_collection,
    vtu_frame_name,
)


comptime P = 2
comptime NX = 64
comptime NY = 16
comptime LX = 1.0
comptime LY = 0.25
comptime GAMMA = 1.4
comptime RHO_0 = 1.0
comptime P_0 = 1.0
comptime MACH = 2.0
comptime T_FINAL = 0.2
comptime NUM_FRAMES = 20
comptime CFL = 0.15


comptime FRAME_PREFIX = "frame_channel_gpu_"


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    mpi.init()
    var size = mpi.world_size()
    if size > 1:
        mpi.finalize()
        print("euler_channel_2d_gpu: runs at np=1 only")
        return

    print("euler_channel_2d_gpu (Mach", MACH, ", P=", P, ",", NX, "x", NY, ")")
    print("  BCs: -x INFLOW, +x OUTFLOW, y WALL")

    comptime NP_p = num_tri_nodes_2d(P)
    comptime NFP_e = num_edge_nodes(P)
    comptime NC = 4
    var ctx = DeviceContext()

    var c_inf = sqrt(GAMMA * P_0 / RHO_0)
    var u_inf = MACH * c_inf
    var rhou_inf = RHO_0 * u_inf
    var E_inf = P_0 / (GAMMA - 1.0) + 0.5 * RHO_0 * u_inf * u_inf
    print("  c_inf =", c_inf, " u_inf =", u_inf, " E_inf =", E_inf)

    var bcs = BoundaryConditions2D(
        BC_INFLOW,
        BC_OUTFLOW,
        BC_WALL,
        BC_WALL,
    )
    var host_mesh = LocalMesh2D[P](NX, NY, LX, LY, bcs)
    var host_re = ReferenceElement2D[P]()
    var mesh_coords = LocalMesh2D[P](NX, NY, LX, LY, bcs)
    var gpu_mesh = LocalMesh2DGpu[P](ctx, host_mesh^)
    var gpu_re = ReferenceElement2DGpu[P](ctx, host_re)
    print(
        "  elements:",
        gpu_mesh.num_elements,
        " faces:",
        gpu_mesh.num_faces,
        "  total DOF:",
        gpu_mesh.num_elements * NP_p * NC,
    )

    # IC = inflow state everywhere (analytically steady).
    var n_q = gpu_mesh.num_elements * NP_p * NC
    var host_q = List[Float32]()
    for _ in range(gpu_mesh.num_elements * NP_p):
        host_q.append(Float32(RHO_0))
        host_q.append(Float32(rhou_inf))
        host_q.append(Float32(0.0))
        host_q.append(Float32(E_inf))

    var d_q = ctx.enqueue_create_buffer[DType.float32](n_q)
    var d_q1 = ctx.enqueue_create_buffer[DType.float32](n_q)
    var d_q2 = ctx.enqueue_create_buffer[DType.float32](n_q)
    var d_fstar_count = gpu_mesh.num_faces * NFP_e * NC
    var d_fstar = ctx.enqueue_create_buffer[DType.float32](d_fstar_count)

    # WARPXM-style device-memory accounting (Euler channel: smooth flow,
    # no BJ limiter; 2D np=1 only).
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
    for k in range(n_q):
        hptr_q[k] = host_q[k]
    ctx.enqueue_copy(d_q, hbuf_q)
    ctx.synchronize()

    var h_cell = LX / Float64(NX)
    var wave_max = u_inf + c_inf
    var dt_est = CFL * h_cell / (wave_max * Float64(2 * P + 1))
    var steps_per_frame = Int(T_FINAL / (Float64(NUM_FRAMES) * dt_est)) + 1
    var total_steps = NUM_FRAMES * steps_per_frame
    var dt = Float32(T_FINAL / Float64(total_steps))
    print(
        "  dt=",
        dt,
        "  steps/frame=",
        steps_per_frame,
        "  total steps=",
        total_steps,
    )

    var gamma = Float32(GAMMA)
    var min_rho = Float32(1.0e-6)
    var min_p = Float32(1.0e-6)
    # Kernel-level inflow state passed into the BC_INFLOW arm of
    # `euler_face_flux_kernel_2d` as the matched ghost.
    var inflow_rho = Float32(RHO_0)
    var inflow_rhou = Float32(rhou_inf)
    var inflow_rhov = Float32(0.0)
    var inflow_E = Float32(E_inf)

    var density = List[Float64]()
    var pressure = List[Float64]()
    var vmag = List[Float64]()
    for _ in range(gpu_mesh.num_elements * NP_p):
        density.append(0.0)
        pressure.append(0.0)
        vmag.append(0.0)
    var paths = List[String]()
    var times = List[Float64]()

    var field_names = List[String]()
    field_names.append(String("rho"))
    field_names.append(String("p"))
    field_names.append(String("|v|"))

    @parameter
    def fill_derived(src: UnsafePointer[Float32, MutAnyOrigin]) raises:
        var n_nodes = gpu_mesh.num_elements * NP_p
        for i in range(n_nodes):
            var rho = Float64(src[i * NC + 0])
            var mx = Float64(src[i * NC + 1])
            var my = Float64(src[i * NC + 2])
            var E = Float64(src[i * NC + 3])
            var rho_safe = rho if rho > 1.0e-12 else 1.0e-12
            var u = mx / rho_safe
            var v = my / rho_safe
            var ke = 0.5 * rho_safe * (u * u + v * v)
            density[i] = rho
            pressure[i] = (GAMMA - 1.0) * (E - ke)
            vmag[i] = sqrt(u * u + v * v)

    # Frame 0: IC.
    fill_derived(hptr_q)
    var fields = List[List[Float64]]()
    fields.append(density.copy())
    fields.append(pressure.copy())
    fields.append(vmag.copy())
    var f0_name = vtu_frame_name(FRAME_PREFIX, 0)
    dump_vtu_2d_frame_multi[P](
        mesh_coords,
        field_names,
        fields,
        String("output/") + f0_name,
    )
    paths.append(f0_name)
    times.append(0.0)

    var rho_max_drift: Float64 = 0.0
    var run_start = perf_counter_ns()
    var compute_ns: UInt = 0
    var stage_plans = ssprk3_stage_plans(
        d_q.unsafe_ptr(),
        d_q1.unsafe_ptr(),
        d_q2.unsafe_ptr(),
    )
    for fi in range(1, NUM_FRAMES + 1):
        var c_start = perf_counter_ns()
        for _ in range(steps_per_frame):
            for stage in stage_plans:
                euler_rk_stage_2d[P](
                    ctx,
                    gpu_mesh,
                    gpu_re.d_Lift_ref.unsafe_ptr(),
                    gpu_re.d_D_ref.unsafe_ptr(),
                    stage.q_in,
                    stage.q_a,
                    stage.q_b,
                    stage.q_out,
                    d_fstar.unsafe_ptr(),
                    gamma,
                    min_rho,
                    min_p,
                    stage.a,
                    stage.b,
                    stage.c,
                    dt,
                    inflow_rho=inflow_rho,
                    inflow_rhou=inflow_rhou,
                    inflow_rhov=inflow_rhov,
                    inflow_E=inflow_E,
                )
        ctx.synchronize()
        var c_end = perf_counter_ns()
        compute_ns += c_end - c_start

        ctx.enqueue_copy(hbuf_q, d_q)
        ctx.synchronize()
        fill_derived(hptr_q)
        for i in range(gpu_mesh.num_elements * NP_p):
            var drift = density[i] - RHO_0
            var adr = drift if drift >= 0.0 else -drift
            if adr > rho_max_drift:
                rho_max_drift = adr
        var fi_fields = List[List[Float64]]()
        fi_fields.append(density.copy())
        fi_fields.append(pressure.copy())
        fi_fields.append(vmag.copy())
        var t = Float64(fi) * Float64(steps_per_frame) * Float64(dt)
        var fname = vtu_frame_name(FRAME_PREFIX, fi)
        dump_vtu_2d_frame_multi[P](
            mesh_coords,
            field_names,
            fi_fields,
            String("output/") + fname,
        )
        paths.append(fname)
        times.append(t)
        print(
            "    frame",
            fi,
            "/",
            NUM_FRAMES,
            " t=",
            t,
            "  rho max |drift|:",
            rho_max_drift,
        )
    var run_end = perf_counter_ns()

    var total_sec = Float64(run_end - run_start) * 1.0e-9
    var compute_sec = Float64(compute_ns) * 1.0e-9
    print("  total time  :", total_sec, "s (incl. frame I/O)")
    ThroughputReport(
        num_steps=total_steps,
        wall_seconds=compute_sec,
        dof_count=gpu_mesh.num_elements * NP_p * NC,
        state_bytes_per_step=8 * n_q * 4,
    ).print()
    print(
        "  rho max |drift| vs IC:",
        rho_max_drift,
        " (expected ~ 0 for exact steady solution)",
    )

    dump_pvd_collection(
        String("output/solution_channel_gpu.pvd"),
        paths,
        times,
    )
    print(
        "  wrote output/solution_channel_gpu.pvd +",
        NUM_FRAMES + 1,
        "VTU frames",
    )

    mpi.finalize()
