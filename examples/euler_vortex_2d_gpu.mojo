# ======================================================================
# euler_vortex_2d_gpu -- GPU 2D DG isentropic Euler vortex
# ======================================================================
#
# GPU analog of `euler_vortex_2d_cpu.mojo`.  The isentropic vortex is
# the classical DG Euler validation: after one full period on a
# periodic domain it must equal the initial condition, so the final L2
# discrepancy is pure scheme dissipation.
#
# Uses the Float32 Euler kernels from `src/local_mesh_2d_gpu.mojo`:
#   euler_volume_rhs_kernel_2d + euler_face_flux_kernel_2d +
#   lift_combine_kernel_2d[NC=4] + rk_update_kernel_2d[NC=4],
# orchestrated by `euler_rk_stage_2d`.  Prints wall-clock time and
# final relative L2 error on state (rho, mx, my, E).
# ======================================================================

from std.math import sqrt, exp, pi
from std.sys import has_accelerator
from std.gpu.host import DeviceContext, DeviceBuffer
from std.time import perf_counter_ns
from src import mpi
from src.local_mesh_2d import LocalMesh2D
from src.local_mesh_2d_gpu import LocalMesh2DGpu, euler_rk_stage_2d
from src.reference_2d import (
    ReferenceElement2D, num_tri_nodes_2d, num_edge_nodes,
)
from src.reference_2d_gpu import ReferenceElement2DGpu


comptime P = 2
comptime NX = 32
comptime NY = 32
comptime LX = 10.0
comptime LY = 10.0
comptime T_FINAL = 10.0
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
    if d >  L * 0.5: d -= L
    if d < -L * 0.5: d += L
    return d


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

    var host_mesh = LocalMesh2D[P](NX, NY, LX, LY)
    var host_re = ReferenceElement2D[P]()
    var mesh_coords = LocalMesh2D[P](NX, NY, LX, LY)  # keep coords for IC
    var gpu_mesh = LocalMesh2DGpu[P](ctx, host_mesh^)
    var gpu_re = ReferenceElement2DGpu[P](ctx, host_re)
    print("  elements:", gpu_mesh.num_elements,
          " faces:", gpu_mesh.num_faces,
          "  nodes/elem:", NP_p,
          "  total DOF:", gpu_mesh.num_elements * NP_p * NC)

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

    # dt from CFL / DG scaling (same (2P+1) factor the CPU driver uses).
    var h = LX / Float64(NX)
    var c_inf = sqrt(GAMMA * T_INF)
    var wave_max = sqrt(U0 * U0 + V0 * V0) + c_inf + BETA / two_pi
    var dt_est = CFL * h / (wave_max * Float64(2 * P + 1))
    var num_steps = Int(T_FINAL / dt_est) + 1
    var dt = Float32(T_FINAL / Float64(num_steps))
    print("  dt=", dt, "  num_steps=", num_steps,
          "  wave_max=", wave_max)

    var gamma = Float32(GAMMA)
    var min_rho = Float32(1.0e-6)
    var min_p = Float32(1.0e-6)

    var step_start = perf_counter_ns()
    for _ in range(num_steps):
        # Stage 1: q1 = q + dt L(q)
        euler_rk_stage_2d[P](
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
        # Stage 2: q2 = 3/4 q + 1/4 (q1 + dt L(q1))
        euler_rk_stage_2d[P](
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
        # Stage 3: q <- 1/3 q + 2/3 (q2 + dt L(q2))
        euler_rk_stage_2d[P](
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
    ctx.synchronize()
    var step_end = perf_counter_ns()
    var wall_sec = Float64(step_end - step_start) * 1.0e-9
    print("  wall time:", wall_sec, "s")
    print("  throughput:", Float64(num_steps) / wall_sec, "steps/s")

    # Download final q and compare to IC (one period == exact IC).
    ctx.enqueue_copy(hbuf_q, d_q)
    ctx.synchronize()
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

    mpi.finalize()
