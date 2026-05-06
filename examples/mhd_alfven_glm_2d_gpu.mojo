# ======================================================================
# mhd_alfven_glm_2d_gpu -- 2D MHD with Dedner GLM divB cleaning
# ======================================================================
#
# Companion to `mhd_alfven_2d_gpu` (NC=6 plain ideal MHD) that
# demonstrates the NC=7 GLM (Generalised Lagrange Multiplier) path.
# Same Alfven-wave background, plus an additional psi pseudo-field
# that GLM transports + damps.  The IC seeds a small psi
# perturbation
#
#   psi(x, 0) = A_psi sin(2 k x)
#
# along with the canonical right-going Alfven wave (uy, By non-trivial
# and rho, p, Bx uniform).  At t > 0 the GLM equations
#
#   dB/dt + ... + grad psi   = 0
#   dpsi/dt + c_h^2 div B    = -alpha_d psi
#
# advect psi at speeds +-c_h while damping its amplitude at rate
# alpha_d.  c_h = 1.5 (faster than the fast-magnetosonic speed cf =
# sqrt(cs^2 + cA^2) ~ 1.08, so any numerical divB error gets swept
# out of the box) and alpha_d = 0.5 (half-life ~1.4 / alpha_d) make
# the cleaning visible in animation.
#
# Visualisation: each VTU frame ships rho + |v| + |B| + psi.
# `psi` is the diagnostic field -- it should ride one or two waves
# left/right at speed +-c_h while smoothly decaying.
#
#   ./mhd_alfven_glm_2d_gpu                       # produces the .pvd
#   scripts/animate_2d.py output/solution_mhd_alfven_glm_2d_gpu.pvd \
#       -f 'rho,|B|,psi' -o glm.mp4               # 3-panel animation
#
# Numerically validated by `bench_mhd_alfven_glm_2d` (c_h=0 path)
# and `bench_mhd_glm_psi_transport_2d` (psi/Bx coupling); this
# example exercises both with non-trivial physical parameters.
# ======================================================================

from std.math import sqrt, sin, pi
from std.sys import has_accelerator
from std.gpu.host import DeviceContext, DeviceBuffer
from std.time import perf_counter_ns
from src import mpi
from src.local_mesh_2d import LocalMesh2D
from src.local_mesh_2d_gpu import LocalMesh2DGpu
from src.local_mesh_2d_gpu_mhd_glm import (
    mhd_glm_rk_stage_2d,
    launch_mhd_glm_psi_damp_2d,
)
from src.reference_2d import (
    ReferenceElement2D,
    num_tri_nodes_2d,
    num_edge_nodes,
)
from src.reference_2d_gpu import ReferenceElement2DGpu
from src.ssprk3 import ssprk3_stage_plans
from src.memory_report import MemoryReport, ThroughputReport
from src.vtu_2d import (
    dump_vtu_2d_frame_multi,
    dump_pvd_collection,
    vtu_frame_name,
)


comptime P = 2
comptime NX = 64
comptime NY = 4
comptime LX = 1.0
comptime LY = Float64(4.0 / 64.0)

comptime GAMMA = 5.0 / 3.0
comptime RHO0 = 1.0
comptime B0 = 1.0
comptime P0 = 0.1
comptime AMPLITUDE = 0.1  # Alfven u_y / B_y amplitude
comptime PSI_AMP = 0.05  # initial psi perturbation
comptime C_H = 1.5  # GLM transport speed (> cf)
comptime ALPHA_D = 0.5  # GLM damping rate

comptime T_FINAL = 1.0
comptime NUM_FRAMES = 20
comptime CFL = 0.15

comptime FRAME_PREFIX = "frame_mhd_alfven_glm_2d_gpu_"


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    mpi.init()
    var size = mpi.world_size()
    if size > 1:
        mpi.finalize()
        print("mhd_alfven_glm_2d_gpu: runs at np=1 only")
        return

    print(
        "mhd_alfven_glm_2d_gpu (GPU 2D MHD + Dedner GLM, P=",
        P,
        ",",
        NX,
        "x",
        NY,
        ")",
    )

    comptime NP_p = num_tri_nodes_2d(P)
    comptime NFP_e = num_edge_nodes(P)
    comptime NC = 7  # GLM adds psi to the 6-component plain MHD state
    var ctx = DeviceContext()

    var host_mesh = LocalMesh2D[P](NX, NY, LX, LY)
    var host_re = ReferenceElement2D[P]()
    var mesh_coords = LocalMesh2D[P](NX, NY, LX, LY)
    var gpu_mesh = LocalMesh2DGpu[P](ctx, host_mesh^)
    var gpu_re = ReferenceElement2DGpu[P](ctx, host_re)
    print(
        "  elements:",
        gpu_mesh.num_elements,
        "  nodes/elem:",
        NP_p,
        "  total DOF:",
        gpu_mesh.num_elements * NP_p * NC,
        "  (NC=",
        NC,
        " incl. psi)",
    )

    # IC: Alfven wave background + small psi perturbation.
    # NC=7 layout: rho, mom_x, mom_y, Bx, By, E, psi.
    var k_wave = 2.0 * pi / LX
    var n_q = gpu_mesh.num_elements * NP_p * NC
    var host_q = List[Float32]()
    for elem in range(gpu_mesh.num_elements):
        for nn in range(NP_p):
            var x = mesh_coords.elem_node_xyz[(elem * NP_p + nn) * 2 + 0]
            var sv = sin(k_wave * x)
            var rho = RHO0
            var uy = AMPLITUDE * sv
            var By = -AMPLITUDE * sv
            var Bx = B0
            var ke = 0.5 * rho * (uy * uy)
            var mp = 0.5 * (Bx * Bx + By * By)
            var E = P0 / (GAMMA - 1.0) + ke + mp
            var psi = PSI_AMP * sin(2.0 * k_wave * x)
            host_q.append(Float32(rho))
            host_q.append(Float32(0.0))
            host_q.append(Float32(rho * uy))
            host_q.append(Float32(Bx))
            host_q.append(Float32(By))
            host_q.append(Float32(E))
            host_q.append(Float32(psi))

    var d_q = ctx.enqueue_create_buffer[DType.float32](n_q)
    var d_q1 = ctx.enqueue_create_buffer[DType.float32](n_q)
    var d_q2 = ctx.enqueue_create_buffer[DType.float32](n_q)
    var d_fstar_count = gpu_mesh.num_faces * NFP_e * NC
    var d_fstar = ctx.enqueue_create_buffer[DType.float32](d_fstar_count)

    # WARPXM-style device-memory accounting (NC=7 GLM-MHD smooth Alfven,
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

    # CFL: bound by max(c_h, cf) since GLM transport adds c_h waves.
    var cs = sqrt(GAMMA * P0 / RHO0)
    var cA = B0 / sqrt(RHO0)
    var cf = sqrt(cs * cs + cA * cA)
    var wave_max = cf if cf > C_H else C_H
    var h_cell = LX / Float64(NX)
    var dt_est = CFL * h_cell / (wave_max * Float64(2 * P + 1))
    var steps_per_frame = Int(T_FINAL / (Float64(NUM_FRAMES) * dt_est)) + 1
    var total_steps = NUM_FRAMES * steps_per_frame
    var dt = Float32(T_FINAL / Float64(total_steps))
    print(
        "  cf=",
        cf,
        "  c_h=",
        C_H,
        "  alpha_d=",
        ALPHA_D,
        "  dt=",
        dt,
        "  steps/frame=",
        steps_per_frame,
        "  total steps=",
        total_steps,
    )

    var gamma_f = Float32(GAMMA)
    var min_rho = Float32(1.0e-6)
    var min_p = Float32(1.0e-6)
    var c_h_f = Float32(C_H)
    var alpha_d_f = Float32(ALPHA_D)

    # Multi-field VTU: rho + |v| + |B| + psi.
    var density = List[Float64]()
    var vmag = List[Float64]()
    var bmag = List[Float64]()
    var psi_field = List[Float64]()
    for _ in range(gpu_mesh.num_elements * NP_p):
        density.append(0.0)
        vmag.append(0.0)
        bmag.append(0.0)
        psi_field.append(0.0)
    var paths = List[String]()
    var times = List[Float64]()

    var field_names = List[String]()
    field_names.append(String("rho"))
    field_names.append(String("|v|"))
    field_names.append(String("|B|"))
    field_names.append(String("psi"))

    @parameter
    def fill_derived(src: UnsafePointer[Float32, MutAnyOrigin]) raises:
        var n_nodes = gpu_mesh.num_elements * NP_p
        for i in range(n_nodes):
            var rho = Float64(src[i * NC + 0])
            var mx = Float64(src[i * NC + 1])
            var my = Float64(src[i * NC + 2])
            var Bx = Float64(src[i * NC + 3])
            var By = Float64(src[i * NC + 4])
            var psi = Float64(src[i * NC + 6])
            var rho_safe = rho if rho > 1.0e-12 else 1.0e-12
            var u = mx / rho_safe
            var v = my / rho_safe
            density[i] = rho
            vmag[i] = sqrt(u * u + v * v)
            bmag[i] = sqrt(Bx * Bx + By * By)
            psi_field[i] = psi

    # Frame 0: IC.
    fill_derived(hptr_q)
    var fields = List[List[Float64]]()
    fields.append(density.copy())
    fields.append(vmag.copy())
    fields.append(bmag.copy())
    fields.append(psi_field.copy())
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
        d_q.unsafe_ptr(),
        d_q1.unsafe_ptr(),
        d_q2.unsafe_ptr(),
    )
    for fi in range(1, NUM_FRAMES + 1):
        var c_start = perf_counter_ns()
        for _ in range(steps_per_frame):
            for stage in stage_plans:
                mhd_glm_rk_stage_2d[P](
                    ctx,
                    gpu_mesh,
                    gpu_re.d_Lift_ref.unsafe_ptr(),
                    gpu_re.d_D_ref.unsafe_ptr(),
                    stage.q_in,
                    stage.q_a,
                    stage.q_b,
                    stage.q_out,
                    d_fstar.unsafe_ptr(),
                    gamma_f,
                    min_rho,
                    min_p,
                    c_h_f,
                    stage.a,
                    stage.b,
                    stage.c,
                    dt,
                )
            # Operator-split psi damping after each full SSPRK3 step.
            launch_mhd_glm_psi_damp_2d[NP_p](
                ctx,
                d_q.unsafe_ptr(),
                gpu_mesh.num_elements,
                alpha_d_f,
                dt,
            )
        ctx.synchronize()
        var c_end = perf_counter_ns()
        compute_ns += c_end - c_start

        ctx.enqueue_copy(hbuf_q, d_q)
        ctx.synchronize()
        fill_derived(hptr_q)
        var fi_fields = List[List[Float64]]()
        fi_fields.append(density.copy())
        fi_fields.append(vmag.copy())
        fi_fields.append(bmag.copy())
        fi_fields.append(psi_field.copy())
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
        # Track psi peak amplitude per frame for textual diagnostic.
        var psi_peak: Float64 = 0.0
        for i in range(gpu_mesh.num_elements * NP_p):
            var ap = psi_field[i] if psi_field[i] >= 0.0 else -psi_field[i]
            if ap > psi_peak:
                psi_peak = ap
        print(
            "    frame", fi, "/", NUM_FRAMES, " t=", t, "  |psi|_max=", psi_peak
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

    dump_pvd_collection(
        String("output/solution_mhd_alfven_glm_2d_gpu.pvd"),
        paths,
        times,
    )
    print(
        "  wrote output/solution_mhd_alfven_glm_2d_gpu.pvd +",
        NUM_FRAMES + 1,
        "VTU frames",
    )

    mpi.finalize()
