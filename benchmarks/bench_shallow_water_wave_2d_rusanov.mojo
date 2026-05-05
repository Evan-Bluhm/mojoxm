# ======================================================================
# bench_shallow_water_wave_2d_rusanov -- 2D SW Rusanov flux gate
# ======================================================================
#
# Closes a coverage gap parallel to bench_euler_flux_coverage_3d:
# every 2D SW benchmark and test (bench_shallow_water_wave_2d /_p3,
# bench_shallow_water_dam_break_2d) and the sw_2d_gpu_test go through
# `sw_rk_stage_hll_2d` (HLL flux).  The Rusanov variant
# `sw_rk_stage_2d` lives only in
# examples/shallow_water_dam_break_2d_gpu.mojo and is otherwise
# untested -- a regression in `sw_face_flux_kernel_2d` (Rusanov) or
# in `sw_rk_stage_2d` itself would not trip any gate.
#
# Same linear shallow-water periodic-return IC as the HLL gate
# (h = H + A sin(2 pi x / Lx); IC splits into +-c waves; one period
# returns to IC modulo nonlinear O((A/H)^2) corrections) but routed
# through `sw_rk_stage_2d` (Rusanov flux) at P=2.
#
# Pass criteria (P=2, sweep N = 32, 48, 64, square cells):
#   * rel L2(state) < 5e-4 at every N (Rusanov is more dissipative
#     than HLL but at A/H=0.01 both sit at the nonlinear floor)
#   * mass conservation drift < 1e-4 (Float32 roundoff)
#   * no NaN / Inf
# ======================================================================

from std.math import sqrt, pi, sin, isnan, isinf
from std.sys import has_accelerator
from std.gpu.host import DeviceContext, DeviceBuffer
from src import mpi
from src.local_mesh_2d import LocalMesh2D
from src.local_mesh_2d_gpu import LocalMesh2DGpu
from src.local_mesh_2d_gpu_sw import sw_rk_stage_2d
from src.ssprk3 import ssprk3_stage_plans
from src.reference_2d import (
    ReferenceElement2D, num_tri_nodes_2d, num_edge_nodes,
)
from src.reference_2d_gpu import ReferenceElement2DGpu


comptime P = 2
comptime NY = 4
comptime LX = 1.0

comptime GRAVITY: Float64 = 1.0
comptime H_REST:  Float64 = 1.0
comptime AMPLITUDE: Float64 = 0.01
comptime H_MIN: Float64 = 1.0e-6
comptime CFL: Float64 = 0.15

comptime T_FINAL: Float64 = LX / 1.0   # one wave period

# Rusanov is more dissipative than HLL but at A/H=0.01 both are
# dominated by the nonlinear O((A/H)^2) floor (~1.7e-4 measured
# under HLL; Rusanov here measures 1.83e-4).  5e-4 is ~2.7x the
# Rusanov floor and catches Rusanov-flux operator regressions.
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

    var stage_plans = ssprk3_stage_plans(
        d_q=d_q.unsafe_ptr(),
        d_q1=d_q1.unsafe_ptr(),
        d_q2=d_q2.unsafe_ptr(),
    )
    for _ in range(num_steps):
        for stage in stage_plans:
            sw_rk_stage_2d[P](
                ctx=ctx,
                mesh=gpu_mesh,
                Lift_ref=gpu_re.d_Lift_ref.unsafe_ptr(),
                D_ref=gpu_re.d_D_ref.unsafe_ptr(),
                q_in=stage.q_in,
                q_a=stage.q_a,
                q_b=stage.q_b,
                q_out=stage.q_out,
                fstar_scratch=d_fstar.unsafe_ptr(),
                g=g_f, min_h=h_min_f,
                a=stage.a, b=stage.b, cc=stage.c, dt=dt,
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
            raise Error("bench_shallow_water_wave_2d_rusanov: non-finite")
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
            String("bench_shallow_water_wave_2d_rusanov FAILED: ")
            + "mass drift " + String(mass_rel)
            + " > tol " + String(MASS_TOL_REL)
            + " (at NX=" + String(NX) + ")"
        )

    return rel_l2


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    mpi.init()
    var size = mpi.world_size()
    if size > 1:
        mpi.finalize()
        print("bench_shallow_water_wave_2d_rusanov: runs at np=1 only")
        return

    print("bench_shallow_water_wave_2d_rusanov (linear SW wave, 2D Rusanov)")
    print("  P=", P, "  sweep NX=32, 48, 64   (threshold", L2_MAX_REL, ")")

    var err32 = _run(32)
    print("  NX=32  rel L2 =", err32)
    var err48 = _run(48)
    print("  NX=48  rel L2 =", err48)
    var err64 = _run(64)
    print("  NX=64  rel L2 =", err64)

    if err32 > L2_MAX_REL:
        raise Error(
            "bench_shallow_water_wave_2d_rusanov FAILED: NX=32 rel L2 "
            + String(err32) + " exceeds " + String(L2_MAX_REL)
        )
    if err48 > L2_MAX_REL:
        raise Error(
            "bench_shallow_water_wave_2d_rusanov FAILED: NX=48 rel L2 "
            + String(err48) + " exceeds " + String(L2_MAX_REL)
        )
    if err64 > L2_MAX_REL:
        raise Error(
            "bench_shallow_water_wave_2d_rusanov FAILED: NX=64 rel L2 "
            + String(err64) + " exceeds " + String(L2_MAX_REL)
        )

    print("=== bench_shallow_water_wave_2d_rusanov PASSED ===")
    mpi.finalize()
