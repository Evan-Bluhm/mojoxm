# ======================================================================
# bench_shallow_water_wave_2d_p5 -- linear SW wave at P=5 (NP=21)
# ======================================================================
#
# P=5 counterpart of bench_shallow_water_wave_2d_p4.  Same small-
# amplitude linear shallow-water wave on a periodic-x domain,
# constant in y, but routed through LocalMesh2D[5] /
# sw_rk_stage_hll_2d[5] with NP=21 nodes per triangle.
#
# Closes the 2D SW P-parity sweep at P=5, mirroring the 2D Euler
# (`bench_euler_smooth_wave_2d_p5`) and 2D Maxwell
# (`bench_maxwell_plane_wave_2d_p5`) gates at the same NP=21 -- and
# the just-added 3D SW P=5 (NP=56) gate.
#
# Pass criteria (P=5, sweep NX = 6, 8, 12, square cells, HLL flux):
#   * rel L2 at every NX < 5e-4
#   * mass conservation drift < 1e-4 (Float32 roundoff floor)
#   * no NaN / Inf
#
# At P=5 the dispersion error is well below the A/H=0.01 nonlinear
# correction floor (~1.7e-4 same as P=3/P=4), so this is an
# absolute-L2 regression sentinel rather than a convergence-rate gate.
# ======================================================================

from std.math import sqrt, pi, sin, isnan, isinf
from std.sys import has_accelerator
from std.gpu.host import DeviceContext, DeviceBuffer
from src import mpi
from src.local_mesh_2d import LocalMesh2D
from src.local_mesh_2d_gpu import LocalMesh2DGpu
from src.local_mesh_2d_gpu_sw import sw_rk_stage_hll_2d
from src.ssprk3 import ssprk3_stage_plans
from src.reference_2d import ReferenceElement2D, num_tri_nodes_2d, num_edge_nodes
from src.reference_2d_gpu import ReferenceElement2DGpu


comptime P = 5
comptime NY = 4
comptime LX = 1.0

comptime GRAVITY: Float64 = 1.0
comptime H_REST: Float64 = 1.0
comptime AMPLITUDE: Float64 = 0.01
comptime H_MIN: Float64 = 1.0e-6
comptime CFL: Float64 = 0.08

comptime T_FINAL: Float64 = LX / 1.0  # one period at c = sqrt(gH) = 1

# At P=5 the rel L2 sits at the A/H=0.01 nonlinear correction floor
# across the entire sweep (~1.7e-4 expected, same as P=3/P=4).
# 5e-4 keeps the threshold convention used at P=3/P=4.
comptime L2_MAX_REL: Float64 = 5.0e-4
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
            host_q.append(Float32(h))
            host_ic.append(Float32(h))
            host_q.append(Float32(0.0))
            host_ic.append(Float32(0.0))
            host_q.append(Float32(0.0))
            host_ic.append(Float32(0.0))

    var d_q = ctx.enqueue_create_buffer[DType.float32](n_q)
    var d_q1 = ctx.enqueue_create_buffer[DType.float32](n_q)
    var d_q2 = ctx.enqueue_create_buffer[DType.float32](n_q)
    var d_fstar = ctx.enqueue_create_buffer[DType.float32](gpu_mesh.num_faces * NFP_e * NC)
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

    var stage_plans = ssprk3_stage_plans(d_q.unsafe_ptr(), d_q1.unsafe_ptr(), d_q2.unsafe_ptr())
    for _ in range(num_steps):
        for stage in stage_plans:
            sw_rk_stage_hll_2d[P](
                ctx,
                gpu_mesh,
                gpu_re.d_Lift_ref.unsafe_ptr(),
                gpu_re.d_D_ref.unsafe_ptr(),
                stage.q_in,
                stage.q_a,
                stage.q_b,
                stage.q_out,
                d_fstar.unsafe_ptr(),
                g_f,
                h_min_f,
                stage.a,
                stage.b,
                stage.c,
                dt,
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
            raise Error("bench_shallow_water_wave_2d_p5: non-finite")
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
    if dmass < 0.0:
        dmass = -dmass
    var mass_rel = dmass / mass_ic_local
    if mass_rel > MASS_TOL_REL:
        raise Error(String("bench_shallow_water_wave_2d_p5 FAILED: mass drift ") + String(mass_rel) + " > tol " + String(MASS_TOL_REL) + " (at NX=" + String(NX) + ")")

    return rel_l2


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    mpi.init()
    var size = mpi.world_size()
    if size > 1:
        mpi.finalize()
        print("bench_shallow_water_wave_2d_p5: runs at np=1 only")
        return

    print("bench_shallow_water_wave_2d_p5 (linear SW wave, 2D HLL, P=5)")
    print("  P=", P, "  NP=", num_tri_nodes_2d(P), "  sweep NX=6, 8, 12   (threshold", L2_MAX_REL, ")")

    var err6 = _run(6)
    print("  NX=6   rel L2 =", err6)
    var err8 = _run(8)
    print("  NX=8   rel L2 =", err8)
    var err12 = _run(12)
    print("  NX=12  rel L2 =", err12)

    if err6 > L2_MAX_REL:
        raise Error("bench_shallow_water_wave_2d_p5 FAILED: NX=6 rel L2 " + String(err6) + " exceeds " + String(L2_MAX_REL))
    if err8 > L2_MAX_REL:
        raise Error("bench_shallow_water_wave_2d_p5 FAILED: NX=8 rel L2 " + String(err8) + " exceeds " + String(L2_MAX_REL))
    if err12 > L2_MAX_REL:
        raise Error("bench_shallow_water_wave_2d_p5 FAILED: NX=12 rel L2 " + String(err12) + " exceeds " + String(L2_MAX_REL))

    print("=== bench_shallow_water_wave_2d_p5 PASSED ===")
    mpi.finalize()
