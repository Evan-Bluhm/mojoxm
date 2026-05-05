# ======================================================================
# mhd_alfven_2d_gpu -- GPU 2D DG ideal-MHD linear Alfven wave
# ======================================================================
#
# Canonical ideal-MHD validation: a transverse Alfven wave travels
# along +x with background B = (B0, 0), background rho = rho0, and
# initial perturbation
#
#   u_y(x, 0) =  A sin(k x)
#   B_y(x, 0) = -A sin(k x)       (right-going, c_A = B0 / sqrt(rho0))
#
# After one period T = L / c_A the state returns exactly to the IC.
# Any residual L2 error in the 6-component state is pure scheme
# dissipation + Float32 accumulation.  A 2D-only analog of the 3D
# `mhd_alfven` driver, using the Float32 kernels in
# `src/local_mesh_2d_gpu.mojo` (mhd_rk_stage_2d).
#
# Writes NUM_FRAMES VTU snapshots of the By field +
# `output/solution_mhd_alfven_2d_gpu.pvd`.  Drop the .pvd into
# ParaView to watch the wave; the perturbation should translate +x
# without distortion.
# ======================================================================

from std.math import sqrt, sin, pi
from std.sys import has_accelerator
from std.gpu.host import DeviceContext, DeviceBuffer
from std.time import perf_counter_ns
from src import mpi
from src.local_mesh_2d import LocalMesh2D
from src.local_mesh_2d_gpu import LocalMesh2DGpu
from src.local_mesh_2d_gpu_mhd import mhd_rk_stage_2d
from src.reference_2d import (
    ReferenceElement2D, num_tri_nodes_2d, num_edge_nodes,
)
from src.reference_2d_gpu import ReferenceElement2DGpu
from src.vtu_2d import (
    dump_vtu_2d_frame_multi, dump_pvd_collection, vtu_frame_name,
)


comptime P = 2
comptime NX = 64
comptime NY = 4
comptime LX = 1.0
comptime LY = Float64(4.0 / 64.0)   # dx = dy so triangles are isotropic

comptime GAMMA     = 5.0 / 3.0
comptime RHO0      = 1.0
comptime B0        = 1.0
comptime P0        = 0.1
comptime AMPLITUDE = 0.1

# c_A = B0 / sqrt(rho0) = 1 here; wave period = LX / c_A = LX.
comptime T_FINAL   = 1.0
comptime NUM_FRAMES = 20
comptime CFL       = 0.15


comptime FRAME_PREFIX = "frame_mhd_alfven_2d_gpu_"


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    mpi.init()
    var size = mpi.world_size()
    if size > 1:
        mpi.finalize()
        print("mhd_alfven_2d_gpu: runs at np=1 only")
        return

    print("mhd_alfven_2d_gpu (GPU, P=", P, ",", NX, "x", NY, ")")

    comptime NP_p = num_tri_nodes_2d(P)
    comptime NFP_e = num_edge_nodes(P)
    comptime NC = 6
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

    # IC: rho0, (mx, my) = (0, rho0 A sin(k x)),
    #     (Bx, By) = (B0, -A sin(k x)), E from gas + mag pressure.
    var k_wave = 2.0 * pi / LX
    var n_q = gpu_mesh.num_elements * NP_p * NC
    var host_q = List[Float32]()
    var host_ic = List[Float32]()
    for elem in range(gpu_mesh.num_elements):
        for nn in range(NP_p):
            var x = mesh_coords.elem_node_xyz[(elem * NP_p + nn) * 2 + 0]
            var sv = sin(k_wave * x)
            var rho = RHO0
            var uy  = AMPLITUDE * sv
            var By  = -AMPLITUDE * sv
            var Bx  = B0
            var mx  = 0.0
            var my  = rho * uy
            var ke  = 0.5 * rho * (uy * uy)
            var mp  = 0.5 * (Bx * Bx + By * By)
            var E   = P0 / (GAMMA - 1.0) + ke + mp
            host_q.append(Float32(rho))
            host_q.append(Float32(mx))
            host_q.append(Float32(my))
            host_q.append(Float32(Bx))
            host_q.append(Float32(By))
            host_q.append(Float32(E))
            host_ic.append(Float32(rho))
            host_ic.append(Float32(mx))
            host_ic.append(Float32(my))
            host_ic.append(Float32(Bx))
            host_ic.append(Float32(By))
            host_ic.append(Float32(E))

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

    # dt: fast-magnetosonic bound.  For Bx=B0 propagation and no
    # perpendicular B, cf = sqrt(cs^2 + ca^2) with cs^2 = gamma p / rho,
    # ca^2 = B^2 / rho.  Choose dt to divide evenly into NUM_FRAMES.
    var cs = sqrt(GAMMA * P0 / RHO0)
    var cA = B0 / sqrt(RHO0)
    var cf = sqrt(cs * cs + cA * cA)
    var h_cell = LX / Float64(NX)
    var dt_est = CFL * h_cell / (cf * Float64(2 * P + 1))
    var steps_per_frame = Int(T_FINAL / (Float64(NUM_FRAMES) * dt_est)) + 1
    var total_steps = NUM_FRAMES * steps_per_frame
    var dt = Float32(T_FINAL / Float64(total_steps))
    print("  cf=", cf, "  dt=", dt, "  steps/frame=", steps_per_frame,
          "  total steps=", total_steps)

    var gamma_f = Float32(GAMMA)
    var min_rho = Float32(1.0e-6)
    var min_p   = Float32(1.0e-6)

    var density = List[Float64]()
    var vmag = List[Float64]()
    var bmag = List[Float64]()
    var By_scalar = List[Float64]()
    for _ in range(gpu_mesh.num_elements * NP_p):
        density.append(0.0)
        vmag.append(0.0)
        bmag.append(0.0)
        By_scalar.append(0.0)
    var paths = List[String]()
    var times = List[Float64]()

    var field_names = List[String]()
    field_names.append(String("rho"))
    field_names.append(String("|v|"))
    field_names.append(String("|B|"))
    field_names.append(String("By"))   # signed component, useful for wave phase

    # 2D plain MHD (NC=6) state layout: rho, mom_x, mom_y, Bx, By, E
    @parameter
    def fill_derived(src: UnsafePointer[Float32, MutAnyOrigin]) raises:
        var n_nodes = gpu_mesh.num_elements * NP_p
        for i in range(n_nodes):
            var rho = Float64(src[i * NC + 0])
            var mx  = Float64(src[i * NC + 1])
            var my  = Float64(src[i * NC + 2])
            var Bx  = Float64(src[i * NC + 3])
            var By  = Float64(src[i * NC + 4])
            var rho_safe = rho if rho > 1.0e-12 else 1.0e-12
            var u = mx / rho_safe
            var v = my / rho_safe
            density[i] = rho
            vmag[i] = sqrt(u * u + v * v)
            bmag[i] = sqrt(Bx * Bx + By * By)
            By_scalar[i] = By

    # Frame 0: IC.
    fill_derived(hptr_q)
    var fields = List[List[Float64]]()
    fields.append(density.copy())
    fields.append(vmag.copy())
    fields.append(bmag.copy())
    fields.append(By_scalar.copy())
    var f0_name = vtu_frame_name(FRAME_PREFIX, 0)
    dump_vtu_2d_frame_multi[P](
        mesh_coords, field_names, fields,
        String("output/") + f0_name,
    )
    paths.append(f0_name)
    times.append(0.0)

    var run_start = perf_counter_ns()
    var compute_ns: UInt = 0
    for fi in range(1, NUM_FRAMES + 1):
        var c_start = perf_counter_ns()
        for _ in range(steps_per_frame):
            mhd_rk_stage_2d[P](
                ctx, gpu_mesh,
                gpu_re.d_Lift_ref.unsafe_ptr(), gpu_re.d_D_ref.unsafe_ptr(),
                d_q.unsafe_ptr(),
                d_q.unsafe_ptr(), d_q.unsafe_ptr(),
                d_q1.unsafe_ptr(),
                d_fstar.unsafe_ptr(),
                gamma_f, min_rho, min_p,
                Float32(1.0), Float32(0.0), Float32(1.0), dt,
            )
            mhd_rk_stage_2d[P](
                ctx, gpu_mesh,
                gpu_re.d_Lift_ref.unsafe_ptr(), gpu_re.d_D_ref.unsafe_ptr(),
                d_q1.unsafe_ptr(),
                d_q.unsafe_ptr(), d_q1.unsafe_ptr(),
                d_q2.unsafe_ptr(),
                d_fstar.unsafe_ptr(),
                gamma_f, min_rho, min_p,
                Float32(0.75), Float32(0.25), Float32(0.25), dt,
            )
            mhd_rk_stage_2d[P](
                ctx, gpu_mesh,
                gpu_re.d_Lift_ref.unsafe_ptr(), gpu_re.d_D_ref.unsafe_ptr(),
                d_q2.unsafe_ptr(),
                d_q.unsafe_ptr(), d_q2.unsafe_ptr(),
                d_q.unsafe_ptr(),
                d_fstar.unsafe_ptr(),
                gamma_f, min_rho, min_p,
                Float32(1.0 / 3.0), Float32(2.0 / 3.0),
                Float32(2.0 / 3.0), dt,
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
        fi_fields.append(By_scalar.copy())
        var t = Float64(fi) * Float64(steps_per_frame) * Float64(dt)
        var fname = vtu_frame_name(FRAME_PREFIX, fi)
        dump_vtu_2d_frame_multi[P](
            mesh_coords, field_names, fi_fields,
            String("output/") + fname,
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

    # Final L2 vs IC (exact: one period on periodic domain -> IC).
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

    dump_pvd_collection(
        String("output/solution_mhd_alfven_2d_gpu.pvd"), paths, times,
    )
    print("  wrote output/solution_mhd_alfven_2d_gpu.pvd +",
          NUM_FRAMES + 1, "VTU frames")

    mpi.finalize()
