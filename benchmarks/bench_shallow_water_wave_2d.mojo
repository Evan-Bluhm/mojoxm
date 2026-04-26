# ======================================================================
# bench_shallow_water_wave_2d -- linear SW wave periodic-return gate
# ======================================================================
#
# 2D counterpart of bench_shallow_water_wave_3d.  Small-amplitude
# linear shallow-water wave on a periodic-x domain, constant in y:
#
#   h = H + A * sin(2 pi x / Lx)
#   u = 0,  v = 0
#
# Linear SW theory: IC splits into two equal amplitude ±c waves with
# c = sqrt(g H), so eta(x,t) = A sin(kx) cos(kct).  After T = L/c the
# solution returns exactly to the IC modulo nonlinear O((A/H)^2)
# corrections per period (here ~1e-4 at A/H=0.01).
#
# Pass criteria (P=2, sweep N = 32, 48, 64 with square cells, HLL flux):
#   * rel L2(state) < 1e-3 at every N
#   * mass conservation: total h sum drift < 1e-4 (float32 roundoff)
#   * no NaN / Inf
#
# Covers the 2D SW pipeline (sw_rk_stage_hll_2d / sw_volume_rhs /
# sw_face_flux / lift_combine_rk) with an exact one-line analytic
# reference, parallel to bench_euler_smooth_wave_2d for Euler.  The
# 2D SW stack previously had no analytic benchmark; shallow_water_drop
# examples are demonstrations, not regression gates.
# ======================================================================

from std.math import sqrt, pi, sin, isnan, isinf
from std.sys import has_accelerator
from std.gpu.host import DeviceContext, DeviceBuffer
from src import mpi
from src.local_mesh_2d import LocalMesh2D
from src.local_mesh_2d_gpu import LocalMesh2DGpu
from src.local_mesh_2d_gpu_sw import sw_rk_stage_hll_2d
from src.reference_2d import (
    ReferenceElement2D, num_tri_nodes_2d, num_edge_nodes,
)
from src.reference_2d_gpu import ReferenceElement2DGpu


comptime P = 2
comptime NY = 4
comptime LX = 1.0
# LY sized per-run so cells stay square (LY = NY/NX * LX).

comptime GRAVITY: Float64 = 1.0
comptime H_REST:  Float64 = 1.0
comptime AMPLITUDE: Float64 = 0.01
comptime H_MIN: Float64 = 1.0e-6
comptime CFL: Float64 = 0.15

# T = L / c with c = sqrt(gH) = 1 for our values -> one wave period.
comptime T_FINAL: Float64 = LX / 1.0

# Measured ~1.7e-4 across N = 32, 48, 64 (nonlinear O((A/H)^2)
# correction at A=0.01 is the error floor, not scheme dissipation).
# 3e-4 is ~1.7x the actual error, catches any SW flux regression.
comptime L2_MAX_REL: Float64 = 3.0e-4
# 1e-4 is realistic for float32 mass summation over N*N*2 cells and
# ~200 RK steps; conservation bugs would change this by orders of
# magnitude.
comptime MASS_TOL_REL: Float64 = 1.0e-4


def _run(NX: Int) raises -> Float64:
    """Run one period at NX x NY with cells kept square (LY = NY/NX * LX).
    Returns rel L2(q - q_IC)."""
    comptime NP_p = num_tri_nodes_2d(P)
    comptime NFP_e = num_edge_nodes(P)
    comptime NC = 3
    var ctx = DeviceContext()
    var LY = Float64(NY) / Float64(NX) * LX

    var host_mesh = LocalMesh2D[P](NX, NY, LX, LY)
    var host_re = ReferenceElement2D[P]()
    var mesh_coords = LocalMesh2D[P](NX, NY, LX, LY)
    var gpu_mesh = LocalMesh2DGpu[P](ctx, host_mesh^)
    var gpu_re = ReferenceElement2DGpu[P](ctx, host_re)

    var k_wave = 2.0 * pi / LX
    var n_q = gpu_mesh.num_elements * NP_p * NC
    var host_q = List[Float32]()
    var host_ic = List[Float32]()
    for elem in range(gpu_mesh.num_elements):
        for nn in range(NP_p):
            var x = mesh_coords.elem_node_xyz[(elem * NP_p + nn) * 2 + 0]
            var h = H_REST + AMPLITUDE * sin(k_wave * x)
            host_q.append(Float32(h));    host_ic.append(Float32(h))
            host_q.append(Float32(0.0));  host_ic.append(Float32(0.0))
            host_q.append(Float32(0.0));  host_ic.append(Float32(0.0))

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

    var c = sqrt(GRAVITY * H_REST)
    var h_cell = LX / Float64(NX)
    var dt_est = CFL * h_cell / (c * Float64(2 * P + 1))
    var num_steps = Int(T_FINAL / dt_est) + 1
    var dt = Float32(T_FINAL / Float64(num_steps))

    var g_f = Float32(GRAVITY)
    var h_min_f = Float32(H_MIN)
    # BC_INFLOW ghosts (unused under periodic BCs but required by signature).
    var inflow_h = Float32(0.0)
    var inflow_hu = Float32(0.0)
    var inflow_hv = Float32(0.0)

    for _ in range(num_steps):
        sw_rk_stage_hll_2d[P](
            ctx, gpu_mesh,
            gpu_re.d_Lift_ref.unsafe_ptr(), gpu_re.d_D_ref.unsafe_ptr(),
            d_q.unsafe_ptr(),
            d_q.unsafe_ptr(), d_q.unsafe_ptr(),
            d_q1.unsafe_ptr(),
            d_vol.unsafe_ptr(), d_fstar.unsafe_ptr(), d_rhs.unsafe_ptr(),
            g_f, h_min_f, inflow_h, inflow_hu, inflow_hv,
            Float32(1.0), Float32(0.0), Float32(1.0), dt,
        )
        sw_rk_stage_hll_2d[P](
            ctx, gpu_mesh,
            gpu_re.d_Lift_ref.unsafe_ptr(), gpu_re.d_D_ref.unsafe_ptr(),
            d_q1.unsafe_ptr(),
            d_q.unsafe_ptr(), d_q1.unsafe_ptr(),
            d_q2.unsafe_ptr(),
            d_vol.unsafe_ptr(), d_fstar.unsafe_ptr(), d_rhs.unsafe_ptr(),
            g_f, h_min_f, inflow_h, inflow_hu, inflow_hv,
            Float32(0.75), Float32(0.25), Float32(0.25), dt,
        )
        sw_rk_stage_hll_2d[P](
            ctx, gpu_mesh,
            gpu_re.d_Lift_ref.unsafe_ptr(), gpu_re.d_D_ref.unsafe_ptr(),
            d_q2.unsafe_ptr(),
            d_q.unsafe_ptr(), d_q2.unsafe_ptr(),
            d_q.unsafe_ptr(),
            d_vol.unsafe_ptr(), d_fstar.unsafe_ptr(), d_rhs.unsafe_ptr(),
            g_f, h_min_f, inflow_h, inflow_hu, inflow_hv,
            Float32(1.0 / 3.0), Float32(2.0 / 3.0),
            Float32(2.0 / 3.0), dt,
        )
    ctx.synchronize()

    ctx.enqueue_copy(hbuf_q, d_q)
    ctx.synchronize()

    var sum_sq: Float64 = 0.0
    var sum_ic: Float64 = 0.0
    var mass_ic_local: Float64 = 0.0
    var mass_fin: Float64 = 0.0
    for k in range(n_q):
        var v = hptr_q[k]
        if isnan(v) or isinf(v):
            raise Error("bench_shallow_water_wave_2d: non-finite")
        var err = Float64(v - host_ic[k])
        sum_sq += err * err
        var ic = Float64(host_ic[k])
        sum_ic += ic * ic
    # Mass: component-0 (h) summed over all nodes.
    var n_elem_nodes = gpu_mesh.num_elements * NP_p
    for i in range(n_elem_nodes):
        mass_ic_local += Float64(host_ic[i * NC + 0])
        mass_fin += Float64(hptr_q[i * NC + 0])
    var l2 = sqrt(sum_sq / Float64(n_q))
    var l2_ic = sqrt(sum_ic / Float64(n_q))
    var rel_l2 = l2 / l2_ic

    var dmass = mass_fin - mass_ic_local
    if dmass < 0.0: dmass = -dmass
    var mass_rel = dmass / mass_ic_local
    if mass_rel > MASS_TOL_REL:
        raise Error(
            String("bench_shallow_water_wave_2d FAILED: mass drift ")
            + String(mass_rel) + " > tol " + String(MASS_TOL_REL)
            + " (at NX=" + String(NX) + ")"
        )

    return rel_l2


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    mpi.init()
    var size = mpi.world_size()
    if size > 1:
        mpi.finalize()
        print("bench_shallow_water_wave_2d: runs at np=1 only")
        return

    print("bench_shallow_water_wave_2d (linear SW wave, 2D HLL)")
    print("  P=", P, "  sweep NX=32, 48, 64   (threshold", L2_MAX_REL, ")")

    var err32 = _run(32)
    print("  NX=32  rel L2 =", err32)
    var err48 = _run(48)
    print("  NX=48  rel L2 =", err48)
    var err64 = _run(64)
    print("  NX=64  rel L2 =", err64)

    if err32 > L2_MAX_REL:
        raise Error(
            "bench_shallow_water_wave_2d FAILED: NX=32 rel L2 "
            + String(err32) + " exceeds " + String(L2_MAX_REL)
        )
    if err48 > L2_MAX_REL:
        raise Error(
            "bench_shallow_water_wave_2d FAILED: NX=48 rel L2 "
            + String(err48) + " exceeds " + String(L2_MAX_REL)
        )
    if err64 > L2_MAX_REL:
        raise Error(
            "bench_shallow_water_wave_2d FAILED: NX=64 rel L2 "
            + String(err64) + " exceeds " + String(L2_MAX_REL)
        )

    print("=== bench_shallow_water_wave_2d PASSED ===")
    mpi.finalize()
