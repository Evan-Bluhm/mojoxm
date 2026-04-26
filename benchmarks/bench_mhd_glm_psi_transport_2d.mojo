# ======================================================================
# bench_mhd_glm_psi_transport_2d -- GLM linear wave (psi/Bx coupling)
# ======================================================================
#
# Direct test of the c_h transport in the new mhd_glm_* 2D kernels.
# Background: rho=1, p=1, u=v=0, By=Bz=0; only Bx and psi are non-
# trivial.  IC perturbation: psi = A sin(2 pi x / Lx), Bx = 0, all
# other components at rest values.
#
# Linearised GLM on a rest state (no Lorentz feedback because u=0,
# B0=0):
#   d Bx/dt + d psi/dx = 0
#   d psi/dt + c_h^2 d Bx/dx = 0
# This is the wave equation in (Bx, psi) with characteristic speeds
# +-c_h.  d'Alembert: a stationary sin(kx) psi IC splits into two
# equal-amplitude waves moving at +-c_h.  After T = Lx / c_h each
# wave has translated one full wavelength and the superposition
# returns to the IC.  In closed form:
#   psi(x, t) = A sin(kx) cos(k c_h t)
#   Bx(x, t) = (A / c_h) cos(kx) sin(k c_h t) - shifted to match IC
#
# At T = Lx / c_h: cos(k * c_h * T) = cos(2 pi) = 1, sin = 0, so
# psi returns exactly to IC and Bx returns to 0.
#
# Pass criteria (P=2, sweep N = 16, 24, 32 with square cells, c_h=1):
#   * rel L2(state) < 5e-4 at every N (small amplitude A=0.01 keeps
#     linear theory exact to O(A^2); roundoff is the floor).
#   * psi_max sane: |psi| <= A (1 + epsilon) throughout the run --
#     proves no spurious amplification.  Sampled only at t=T because
#     intermediate snapshots aren't stored.
#   * no NaN / Inf.
# ======================================================================

from std.math import sqrt, pi, sin, isnan, isinf
from std.sys import has_accelerator
from std.gpu.host import DeviceContext, DeviceBuffer
from src import mpi
from src.local_mesh_2d import LocalMesh2D
from src.local_mesh_2d_gpu import LocalMesh2DGpu
from src.local_mesh_2d_gpu_mhd_glm import mhd_glm_rk_stage_2d
from src.reference_2d import (
    ReferenceElement2D, num_tri_nodes_2d, num_edge_nodes,
)
from src.reference_2d_gpu import ReferenceElement2DGpu


comptime P = 2
comptime NY = 4
comptime LX = 1.0

comptime GAMMA: Float64    = 5.0 / 3.0
comptime RHO0:  Float64    = 1.0
comptime P0:    Float64    = 1.0
comptime AMPLITUDE: Float64 = 0.01
comptime C_H:   Float64    = 1.0
comptime ALPHA_D: Float64  = 0.0   # transport-only; no damping

comptime CFL: Float64 = 0.15
comptime T_FINAL: Float64 = LX / C_H   # one wave period

comptime L2_MAX_REL: Float64 = 5.0e-4


def _run(NX: Int) raises -> Float64:
    comptime NP_p = num_tri_nodes_2d(P)
    comptime NFP_e = num_edge_nodes(P)
    comptime NC = 7
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

    # IC: rest state + sinusoidal psi.  Energy = p / (gamma - 1) since
    # u = 0 and |B| = 0 in the IC; B will pick up tiny wave amplitudes
    # but those reset to 0 at t = T under linear theory.
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

    var h_cell = LX / Float64(NX)
    var dt_est = CFL * h_cell / (C_H * Float64(2 * P + 1))
    var num_steps = Int(T_FINAL / dt_est) + 1
    var dt = Float32(T_FINAL / Float64(num_steps))

    var gamma_f = Float32(GAMMA)
    var min_rho = Float32(1.0e-6)
    var min_p   = Float32(1.0e-6)
    var c_h_f   = Float32(C_H)

    for _ in range(num_steps):
        mhd_glm_rk_stage_2d[P](
            ctx, gpu_mesh,
            gpu_re.d_Lift_ref.unsafe_ptr(), gpu_re.d_D_ref.unsafe_ptr(),
            d_q.unsafe_ptr(),
            d_q.unsafe_ptr(), d_q.unsafe_ptr(),
            d_q1.unsafe_ptr(),
            d_fstar.unsafe_ptr(),
            gamma_f, min_rho, min_p, c_h_f,
            Float32(1.0), Float32(0.0), Float32(1.0), dt,
        )
        mhd_glm_rk_stage_2d[P](
            ctx, gpu_mesh,
            gpu_re.d_Lift_ref.unsafe_ptr(), gpu_re.d_D_ref.unsafe_ptr(),
            d_q1.unsafe_ptr(),
            d_q.unsafe_ptr(), d_q1.unsafe_ptr(),
            d_q2.unsafe_ptr(),
            d_fstar.unsafe_ptr(),
            gamma_f, min_rho, min_p, c_h_f,
            Float32(0.75), Float32(0.25), Float32(0.25), dt,
        )
        mhd_glm_rk_stage_2d[P](
            ctx, gpu_mesh,
            gpu_re.d_Lift_ref.unsafe_ptr(), gpu_re.d_D_ref.unsafe_ptr(),
            d_q2.unsafe_ptr(),
            d_q.unsafe_ptr(), d_q2.unsafe_ptr(),
            d_q.unsafe_ptr(),
            d_fstar.unsafe_ptr(),
            gamma_f, min_rho, min_p, c_h_f,
            Float32(1.0 / 3.0), Float32(2.0 / 3.0),
            Float32(2.0 / 3.0), dt,
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
            raise Error("bench_mhd_glm_psi_transport_2d: non-finite")
        var err = Float64(v - host_ic[k])
        sum_sq += err * err
        var ic = Float64(host_ic[k])
        sum_ic += ic * ic
    var n_elem_nodes = gpu_mesh.num_elements * NP_p
    for i in range(n_elem_nodes):
        var p = hptr_q[i * NC + 6]
        var a = p if p >= Float32(0.0) else -p
        if a > psi_max: psi_max = a

    # Bounded amplitude check: psi should not exceed A by more than
    # some margin (linear waves can shift phase but conserve amplitude).
    var amp_bound = Float32(AMPLITUDE) * Float32(1.1)
    if psi_max > amp_bound:
        raise Error(
            String("bench_mhd_glm_psi_transport_2d FAILED: psi_max ")
            + String(psi_max) + " exceeds 1.1 * AMPLITUDE "
            + String(amp_bound)
        )

    var l2 = sqrt(sum_sq / Float64(n_q))
    var l2_ic = sqrt(sum_ic / Float64(n_q))
    return l2 / l2_ic


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    mpi.init()
    print("bench_mhd_glm_psi_transport_2d (GLM linear psi/Bx wave)")
    print("  P=", P, "  c_h=", C_H, "  T=", T_FINAL,
          "  amplitude=", AMPLITUDE)

    var err16 = _run(16)
    print("  N=16  rel L2 =", err16)
    var err24 = _run(24)
    print("  N=24  rel L2 =", err24)
    var err32 = _run(32)
    print("  N=32  rel L2 =", err32)

    if err16 > L2_MAX_REL:
        raise Error(
            "bench_mhd_glm_psi_transport_2d FAILED: N=16 rel L2 "
            + String(err16) + " exceeds " + String(L2_MAX_REL)
        )
    if err24 > L2_MAX_REL:
        raise Error(
            "bench_mhd_glm_psi_transport_2d FAILED: N=24 rel L2 "
            + String(err24) + " exceeds " + String(L2_MAX_REL)
        )
    if err32 > L2_MAX_REL:
        raise Error(
            "bench_mhd_glm_psi_transport_2d FAILED: N=32 rel L2 "
            + String(err32) + " exceeds " + String(L2_MAX_REL)
        )

    print("=== bench_mhd_glm_psi_transport_2d PASSED ===")
    mpi.finalize()
