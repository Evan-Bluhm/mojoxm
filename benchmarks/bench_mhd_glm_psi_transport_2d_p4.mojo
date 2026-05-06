# ======================================================================
# bench_mhd_glm_psi_transport_2d_p4 -- GLM psi/Bx wave at P=4 (NP=15)
# ======================================================================
#
# P=4 (NP=15) intermediate between bench_mhd_glm_psi_transport_2d_p3
# (NP=10) and bench_mhd_glm_psi_transport_2d_p5 (NP=21).  Same linear-
# wave (psi, Bx) coupling test routed through LocalMesh2D[4] /
# mhd_glm_rk_stage_2d[4] with NP=15 nodes per triangle.  Mesh sweep
# 10/14/20 sits between P=3's 12/16/24 and P=5's 8/12/16, balancing
# per-element NP^2 work against dt's 1/(2P+1)=1/9 tightening.
#
# Closes the last 2D GLM transport P-parity hole.  Sweep is now
# P=2/P=3/P=4/P=5 across both 2D and 3D for the GLM transport
# rate gate.
#
# Pass criteria (P=4, sweep N = 10, 14, 20, c_h=1):
#   * rel L2(state) < 5e-4 at every N (Float32 floor).
#   * |psi| <= 1.1 * A throughout -- no spurious amplification.
#   * no NaN / Inf
# ======================================================================

from std.math import sqrt, pi, sin, isnan, isinf
from std.sys import has_accelerator
from std.gpu.host import DeviceContext, DeviceBuffer
from src import mpi
from src.local_mesh_2d import LocalMesh2D
from src.local_mesh_2d_gpu import LocalMesh2DGpu
from src.local_mesh_2d_gpu_mhd_glm import mhd_glm_rk_stage_2d
from src.ssprk3 import ssprk3_stage_plans
from src.reference_2d import (
    ReferenceElement2D, num_tri_nodes_2d, num_edge_nodes,
)
from src.reference_2d_gpu import ReferenceElement2DGpu


comptime P = 4
comptime NY = 4
comptime LX = 1.0

comptime GAMMA: Float64    = 5.0 / 3.0
comptime RHO0:  Float64    = 1.0
comptime P0:    Float64    = 1.0
comptime AMPLITUDE: Float64 = 0.01
comptime C_H:   Float64    = 1.0
comptime ALPHA_D: Float64  = 0.0   # transport-only; no damping

comptime CFL: Float64 = 0.10
comptime T_FINAL: Float64 = LX / C_H   # one wave period

comptime L2_MAX_REL: Float64 = 5.0e-4


def _run(NX: Int) raises -> Float64:
    comptime NP_p = num_tri_nodes_2d(P)
    comptime NFP_e = num_edge_nodes(P)
    comptime NC = 7
    var ctx = DeviceContext()
    var LY = Float64(NY) / Float64(NX) * LX

    var host_mesh = LocalMesh2D[P](Nx=NX, Ny=NY, Lx=LX, Ly=LY)
    var host_re = ReferenceElement2D[P]()
    var mesh_coords = LocalMesh2D[P](Nx=NX, Ny=NY, Lx=LX, Ly=LY)
    var gpu_mesh = LocalMesh2DGpu[P](ctx=ctx, host=host_mesh^)
    var gpu_re = ReferenceElement2DGpu[P](ctx=ctx, host=host_re)

    var k_wave = 2.0 * pi / LX
    var n_q = gpu_mesh.num_elements * NP_p * NC
    var host_q = List[Float32]()
    var host_ic = List[Float32]()

    var E0 = P0 / (GAMMA - 1.0)
    for elem in range(gpu_mesh.num_elements):
        for nn in range(NP_p):
            var x = mesh_coords.elem_node_xyz[(elem * NP_p + nn) * 2 + 0]
            var psi = AMPLITUDE * sin(k_wave * x)
            host_q.append(Float32(RHO0));   host_ic.append(Float32(RHO0))
            host_q.append(Float32(0.0));    host_ic.append(Float32(0.0))
            host_q.append(Float32(0.0));    host_ic.append(Float32(0.0))
            host_q.append(Float32(0.0));    host_ic.append(Float32(0.0))
            host_q.append(Float32(0.0));    host_ic.append(Float32(0.0))
            host_q.append(Float32(E0));     host_ic.append(Float32(E0))
            host_q.append(Float32(psi));    host_ic.append(Float32(psi))

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

    # CFL: factor 2P+1 = 9 at P=4.
    var h_cell = LX / Float64(NX)
    var dt_est = CFL * h_cell / (C_H * Float64(2 * P + 1))
    var num_steps = Int(T_FINAL / dt_est) + 1
    var dt = Float32(T_FINAL / Float64(num_steps))

    var gamma_f = Float32(GAMMA)
    var min_rho = Float32(1.0e-6)
    var min_p   = Float32(1.0e-6)
    var c_h_f   = Float32(C_H)

    var stage_plans = ssprk3_stage_plans(
        d_q=d_q.unsafe_ptr(),
        d_q1=d_q1.unsafe_ptr(),
        d_q2=d_q2.unsafe_ptr(),
    )
    for _ in range(num_steps):
        for stage in stage_plans:
            mhd_glm_rk_stage_2d[P](
                ctx=ctx,
                mesh=gpu_mesh,
                Lift_ref=gpu_re.d_Lift_ref.unsafe_ptr(),
                D_ref=gpu_re.d_D_ref.unsafe_ptr(),
                q_in=stage.q_in,
                q_a=stage.q_a,
                q_b=stage.q_b,
                q_out=stage.q_out,
                fstar_scratch=d_fstar.unsafe_ptr(),
                gamma=gamma_f,
                min_density=min_rho,
                min_pressure=min_p,
                c_h=c_h_f,
                a=stage.a, b=stage.b, cc=stage.c, dt=dt,
            )
    ctx.synchronize()

    ctx.enqueue_copy(hbuf_q, d_q)
    ctx.synchronize()

    var sum_sq: Float64 = 0.0
    var sum_ic: Float64 = 0.0
    var psi_max: Float32 = 0.0
    for k in range(n_q):
        var v = hptr_q[k]
        if isnan(v) or isinf(v):
            raise Error("bench_mhd_glm_psi_transport_2d_p4: non-finite")
        var err = Float64(v - host_ic[k])
        sum_sq += err * err
        var ic = Float64(host_ic[k])
        sum_ic += ic * ic
    var n_elem_nodes = gpu_mesh.num_elements * NP_p
    for i in range(n_elem_nodes):
        var p = hptr_q[i * NC + 6]
        var a = p if p >= Float32(0.0) else -p
        if a > psi_max: psi_max = a

    var amp_bound = Float32(AMPLITUDE) * Float32(1.1)
    if psi_max > amp_bound:
        raise Error(
            String("bench_mhd_glm_psi_transport_2d_p4 FAILED: psi_max ")
            + String(psi_max) + " exceeds 1.1 * AMPLITUDE "
            + String(amp_bound)
        )

    var l2 = sqrt(sum_sq / Float64(n_q))
    var l2_ic = sqrt(sum_ic / Float64(n_q))
    return l2 / l2_ic


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    mpi.init()
    print("bench_mhd_glm_psi_transport_2d_p4 (GLM linear psi/Bx wave, P=4)")
    print("  P=", P, "  NP=", num_tri_nodes_2d(P),
          "  c_h=", C_H, "  T=", T_FINAL, "  amplitude=", AMPLITUDE)

    var err10 = _run(10)
    print("  N=10  rel L2 =", err10)
    var err14 = _run(14)
    print("  N=14  rel L2 =", err14)
    var err20 = _run(20)
    print("  N=20  rel L2 =", err20)

    if err10 > L2_MAX_REL:
        raise Error(
            "bench_mhd_glm_psi_transport_2d_p4 FAILED: NX=10 rel L2 "
            + String(err10) + " exceeds " + String(L2_MAX_REL)
        )
    if err14 > L2_MAX_REL:
        raise Error(
            "bench_mhd_glm_psi_transport_2d_p4 FAILED: NX=14 rel L2 "
            + String(err14) + " exceeds " + String(L2_MAX_REL)
        )
    if err20 > L2_MAX_REL:
        raise Error(
            "bench_mhd_glm_psi_transport_2d_p4 FAILED: NX=20 rel L2 "
            + String(err20) + " exceeds " + String(L2_MAX_REL)
        )

    print("=== bench_mhd_glm_psi_transport_2d_p4 PASSED ===")
    mpi.finalize()
