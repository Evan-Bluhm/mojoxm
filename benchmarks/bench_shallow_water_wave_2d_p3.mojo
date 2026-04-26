# ======================================================================
# bench_shallow_water_wave_2d_p3 -- linear SW wave at P=3 (NP=10)
# ======================================================================
#
# P=3 counterpart of bench_shallow_water_wave_2d.  Same small-amplitude
# linear shallow-water wave on a periodic-x domain, constant in y:
#
#   h(x, 0) = H + A * sin(2 pi x / Lx)
#   u = v = 0
#
# After T = Lx / sqrt(g H) (one wave period) the solution returns to
# the IC modulo nonlinear O((A/H)^2) corrections.  Routed through
# LocalMesh2D[3] / sw_rk_stage_hll_2d[3] with NP=10 nodes per
# triangle.
#
# Pairs with bench_euler_smooth_wave_2d_p3 / bench_advection_
# translation_2d_p3 / bench_mhd_alfven_2d (smooth) /
# bench_maxwell_plane_wave_2d to give every 2D smooth physics a
# higher-order analytic gate at NP=10.
#
# Pass criteria (P=3, sweep N = 16, 24, 32, square cells, HLL flux):
#   * rel L2 at every N < 5e-4
#   * mass conservation drift < 1e-4 (Float32 roundoff)
#   * no NaN / Inf
#
# At P=3 the dispersion error for one wave period drops below the
# A/H=0.01 nonlinear correction floor (~1e-4) at the coarsest tested
# resolution, so this is an absolute-L2 sentinel rather than a
# convergence-rate gate.  Catches any SW HLL flux / volume-RHS
# regression that pushes L2 well above the nonlinear floor.
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


comptime P = 3
comptime NY = 4
comptime LX = 1.0
# LY sized per-run so cells stay square (LY = NY/NX * LX).

comptime GRAVITY: Float64 = 1.0
comptime H_REST:  Float64 = 1.0
comptime AMPLITUDE: Float64 = 0.01
comptime H_MIN: Float64 = 1.0e-6
comptime CFL: Float64 = 0.10

# T = L / c with c = sqrt(gH) = 1 -> one wave period.
comptime T_FINAL: Float64 = LX / 1.0

# At P=3 the rel L2 sits at the A/H=0.01 nonlinear correction floor
# across the entire sweep (~1.7e-4 measured).  5e-4 is ~3x the
# empirical floor; tighter would risk false positives on Float32
# roundoff variation across hardware.
comptime L2_MAX_REL: Float64 = 5.0e-4
# Same realistic Float32 mass-summation tolerance as the P=2 bench.
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
            d_fstar.unsafe_ptr(),
            g_f, h_min_f, inflow_h, inflow_hu, inflow_hv,
            Float32(1.0), Float32(0.0), Float32(1.0), dt,
        )
        sw_rk_stage_hll_2d[P](
            ctx, gpu_mesh,
            gpu_re.d_Lift_ref.unsafe_ptr(), gpu_re.d_D_ref.unsafe_ptr(),
            d_q1.unsafe_ptr(),
            d_q.unsafe_ptr(), d_q1.unsafe_ptr(),
            d_q2.unsafe_ptr(),
            d_fstar.unsafe_ptr(),
            g_f, h_min_f, inflow_h, inflow_hu, inflow_hv,
            Float32(0.75), Float32(0.25), Float32(0.25), dt,
        )
        sw_rk_stage_hll_2d[P](
            ctx, gpu_mesh,
            gpu_re.d_Lift_ref.unsafe_ptr(), gpu_re.d_D_ref.unsafe_ptr(),
            d_q2.unsafe_ptr(),
            d_q.unsafe_ptr(), d_q2.unsafe_ptr(),
            d_q.unsafe_ptr(),
            d_fstar.unsafe_ptr(),
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
            raise Error("bench_shallow_water_wave_2d_p3: non-finite")
        var err = Float64(v - host_ic[k])
        sum_sq += err * err
        var ic = Float64(host_ic[k])
        sum_ic += ic * ic
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
            String("bench_shallow_water_wave_2d_p3 FAILED: mass drift ")
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
        print("bench_shallow_water_wave_2d_p3: runs at np=1 only")
        return

    print("bench_shallow_water_wave_2d_p3 (linear SW wave, 2D HLL, P=3)")
    print("  P=", P, "  NP=", num_tri_nodes_2d(P),
          "  sweep NX=16, 24, 32   (threshold", L2_MAX_REL, ")")

    var err16 = _run(16)
    print("  NX=16  rel L2 =", err16)
    var err24 = _run(24)
    print("  NX=24  rel L2 =", err24)
    var err32 = _run(32)
    print("  NX=32  rel L2 =", err32)

    if err16 > L2_MAX_REL:
        raise Error(
            "bench_shallow_water_wave_2d_p3 FAILED: NX=16 rel L2 "
            + String(err16) + " exceeds " + String(L2_MAX_REL)
        )
    if err24 > L2_MAX_REL:
        raise Error(
            "bench_shallow_water_wave_2d_p3 FAILED: NX=24 rel L2 "
            + String(err24) + " exceeds " + String(L2_MAX_REL)
        )
    if err32 > L2_MAX_REL:
        raise Error(
            "bench_shallow_water_wave_2d_p3 FAILED: NX=32 rel L2 "
            + String(err32) + " exceeds " + String(L2_MAX_REL)
        )

    print("=== bench_shallow_water_wave_2d_p3 PASSED ===")
    mpi.finalize()
