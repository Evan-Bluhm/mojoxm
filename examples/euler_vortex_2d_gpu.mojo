# ======================================================================
# euler_vortex_2d_gpu -- GPU 2D DG isentropic Euler vortex
# ======================================================================
#
# Isentropic vortex -- the classical DG Euler validation: after one
# full period on a
# periodic domain it must equal the initial condition, so the final L2
# discrepancy is pure scheme dissipation.
#
# Uses the Float32 Euler kernels from `src/local_mesh_2d_gpu_euler.mojo`:
#   euler_face_flux_kernel_2d + euler_vol_lift_combine_rk_kernel_2d
# (2 launches per SSPRK3 stage), orchestrated by `euler_rk_stage_2d`.
# Every NUM_FRAMES-th step
# downloads the state, extracts density, and writes a VTU frame to
# `output/frame_euler2d_gpu_NNNNN.vtu` + a `.pvd` collection -- same
# layout as the CPU driver, so `scripts/animate_2d.py` works unchanged.
# Prints wall time (compute only vs total), throughput, final rel L2.
# ======================================================================

from std.math import sqrt, exp, pi
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
from src.vtu_2d import (
    dump_vtu_2d_frame_multi,
    dump_pvd_collection,
    vtu_frame_name,
)
from src.memory_report import MemoryReport, ThroughputReport
from src.ssprk3 import ssprk3_stage_plans


comptime P = 2
comptime NX = 32
comptime NY = 32
comptime LX = 10.0
comptime LY = 10.0
comptime T_FINAL = 10.0
comptime NUM_FRAMES = 20
comptime CFL = 0.15

comptime GAMMA = 1.4
comptime T_INF = 1.0
comptime U0 = 1.0
comptime V0 = 1.0
comptime BETA = 5.0
comptime CX0 = 5.0
comptime CY0 = 5.0


def _periodic_delta(a: Float64, b: Float64, L: Float64) -> Float64:
    var d = a - b
    if d > L * 0.5:
        d -= L
    if d < -L * 0.5:
        d += L
    return d


comptime FRAME_PREFIX = "frame_euler2d_gpu_"


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    mpi.init()
    var size = mpi.world_size()
    if size > 1:
        mpi.finalize()
        print("euler_vortex_2d_gpu: runs at np=1 only")
        return

    print("euler_vortex_2d_gpu (GPU, P=", P, ",", NX, "x", NY, ")")

    comptime NP_p = num_tri_nodes_2d(P)
    comptime NFP_e = num_edge_nodes(P)
    comptime NC = 4
    var ctx = DeviceContext()

    var host_mesh = LocalMesh2D[P](Nx=NX, Ny=NY, Lx=LX, Ly=LY)
    var host_re = ReferenceElement2D[P]()
    # kept for IC + VTU
    var mesh_coords = LocalMesh2D[P](Nx=NX, Ny=NY, Lx=LX, Ly=LY)
    var gpu_mesh = LocalMesh2DGpu[P](ctx=ctx, host=host_mesh^)
    var gpu_re = ReferenceElement2DGpu[P](ctx=ctx, host=host_re)
    print(
        "  elements:",
        gpu_mesh.num_elements,
        " faces:",
        gpu_mesh.num_faces,
        "  nodes/elem:",
        NP_p,
        "  total DOF:",
        gpu_mesh.num_elements * NP_p * NC,
    )

    # IC: isentropic vortex (Shu 1998).
    var n_q = gpu_mesh.num_elements * NP_p * NC
    var host_q = List[Float32]()
    var host_ic = List[Float32]()
    var two_pi = 2.0 * pi
    var factor = (GAMMA - 1.0) * BETA * BETA / (8.0 * GAMMA * two_pi * two_pi)

    for elem in range(gpu_mesh.num_elements):
        for nn in range(NP_p):
            var x = mesh_coords.elem_node_xyz[(elem * NP_p + nn) * 2 + 0]
            var y = mesh_coords.elem_node_xyz[(elem * NP_p + nn) * 2 + 1]
            var dx = _periodic_delta(x, CX0, LX)
            var dy = _periodic_delta(y, CY0, LY)
            var r2 = dx * dx + dy * dy
            var T = T_INF - factor * exp(1.0 - r2)
            var e_half = exp(0.5 * (1.0 - r2))
            var u = U0 - (BETA / two_pi) * dy * e_half
            var v = V0 + (BETA / two_pi) * dx * e_half
            var rho = T ** (1.0 / (GAMMA - 1.0))
            var p = rho * T
            var E = p / (GAMMA - 1.0) + 0.5 * rho * (u * u + v * v)
            host_q.append(Float32(rho))
            host_q.append(Float32(rho * u))
            host_q.append(Float32(rho * v))
            host_q.append(Float32(E))
            host_ic.append(Float32(rho))
            host_ic.append(Float32(rho * u))
            host_ic.append(Float32(rho * v))
            host_ic.append(Float32(E))

    var d_q = ctx.enqueue_create_buffer[DType.float32](n_q)
    var d_q1 = ctx.enqueue_create_buffer[DType.float32](n_q)
    var d_q2 = ctx.enqueue_create_buffer[DType.float32](n_q)
    var d_fstar_count = gpu_mesh.num_faces * NFP_e * NC
    var d_fstar = ctx.enqueue_create_buffer[DType.float32](d_fstar_count)

    # Pre-step perf snapshot: device-memory accounting.  2D drivers
    # don't have a Solver wrapper so we assemble the report from each
    # owner's device_bytes() helper plus the locally-allocated state /
    # face-flux scratch.
    MemoryReport(
        rk_stage_bytes=3 * n_q * 4,  # d_q + d_q1 + d_q2
        dg_operators_bytes=gpu_re.device_bytes(),
        limiter_bytes=0,  # (vortex doesn't use BJ)
        mesh_connectivity_bytes=(gpu_mesh.device_bytes() + d_fstar_count * 4),
        halo_device_bytes=0,  # 2D is np=1 only
        halo_pinned_bytes=0,
    ).print()

    var hbuf_q = ctx.enqueue_create_host_buffer[DType.float32](n_q)
    var hptr_q = hbuf_q.unsafe_ptr()
    for k in range(n_q):
        hptr_q[k] = host_q[k]
    ctx.enqueue_copy(d_q, hbuf_q)
    ctx.synchronize()

    # dt: choose num_steps as a multiple of NUM_FRAMES so each frame
    # closes exactly on a stage-3 boundary.
    var h = LX / Float64(NX)
    var c_inf = sqrt(GAMMA * T_INF)
    var wave_max = sqrt(U0 * U0 + V0 * V0) + c_inf + BETA / two_pi
    var dt_est = CFL * h / (wave_max * Float64(2 * P + 1))
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

    var density = List[Float64]()
    var pressure = List[Float64]()
    var vmag = List[Float64]()
    for _ in range(gpu_mesh.num_elements * NP_p):
        density.append(0.0)
        pressure.append(0.0)
        vmag.append(0.0)
    var paths = List[String]()
    var times = List[Float64]()

    # Build the (rho, p, |v|) field-name list once -- same for every frame.
    var field_names = List[String]()
    field_names.append(String("rho"))
    field_names.append(String("p"))
    field_names.append(String("|v|"))

    # Helper: fill (rho, p, |v|) from a flat NC-component buffer.
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
            var p = (GAMMA - 1.0) * (E - ke)
            density[i] = rho
            pressure[i] = p
            vmag[i] = sqrt(u * u + v * v)

    # Frame 0: dump IC before stepping.
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

    var run_start = perf_counter_ns()
    var compute_ns: UInt = 0
    var stage_plans = ssprk3_stage_plans(
        d_q=d_q.unsafe_ptr(),
        d_q1=d_q1.unsafe_ptr(),
        d_q2=d_q2.unsafe_ptr(),
    )
    for fi in range(1, NUM_FRAMES + 1):
        var c_start = perf_counter_ns()
        for _ in range(steps_per_frame):
            for stage in stage_plans:
                euler_rk_stage_2d[P](
                    ctx=ctx,
                    mesh=gpu_mesh,
                    Lift_ref=gpu_re.d_Lift_ref.unsafe_ptr(),
                    D_ref=gpu_re.d_D_ref.unsafe_ptr(),
                    q_in=stage.q_in,
                    q_a=stage.q_a,
                    q_b=stage.q_b,
                    q_out=stage.q_out,
                    fstar_scratch=d_fstar.unsafe_ptr(),
                    gamma=gamma,
                    min_density=min_rho,
                    min_pressure=min_p,
                    a=stage.a,
                    b=stage.b,
                    cc=stage.c,
                    dt=dt,
                )
        ctx.synchronize()
        var c_end = perf_counter_ns()
        compute_ns += c_end - c_start

        ctx.enqueue_copy(hbuf_q, d_q)
        ctx.synchronize()
        fill_derived(hptr_q)
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
        print("    frame", fi, "/", NUM_FRAMES, " t=", t)
    var run_end = perf_counter_ns()

    var total_sec = Float64(run_end - run_start) * 1.0e-9
    var compute_sec = Float64(compute_ns) * 1.0e-9
    print("  total time  :", total_sec, "s (incl. frame I/O)")
    # WARPXM-style throughput summary.  compute_sec is already a
    # sync'd measurement -- the per-frame timing block above ends in
    # ctx.synchronize() before stopping the counter.  state_bytes_per_
    # step = 8 * n_q * 4 mirrors the 3D `Solver.state_bytes_per_step`
    # formula (8 q-buffer touches per SSPRK3 step from the 2-3-3
    # read/write pattern of the three Shu-Osher stages).
    ThroughputReport(
        num_steps=total_steps,
        wall_seconds=compute_sec,
        dof_count=gpu_mesh.num_elements * NP_p * NC,
        state_bytes_per_step=8 * n_q * 4,
    ).print()

    # Final relative L2 vs IC (one period == exact IC on periodic).
    var sum_sq: Float64 = 0.0
    var sum_ic: Float64 = 0.0
    for k in range(n_q):
        var e = Float64(hptr_q[k] - host_ic[k])
        sum_sq += e * e
        var ic = Float64(host_ic[k])
        sum_ic += ic * ic
    var l2 = sqrt(sum_sq / Float64(n_q))
    var l2_ic = sqrt(sum_ic / Float64(n_q))
    print("  L2 err =", l2, "  rel err =", l2 / l2_ic, "  (IC L2 =", l2_ic, ")")

    # PVD collection -- `scripts/animate_2d.py` walks this directly.
    dump_pvd_collection(
        String("output/solution_euler2d_gpu.pvd"),
        paths,
        times,
    )
    print(
        "  wrote output/solution_euler2d_gpu.pvd +",
        NUM_FRAMES + 1,
        "VTU frames",
    )

    mpi.finalize()
