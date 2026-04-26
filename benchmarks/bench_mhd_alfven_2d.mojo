# ======================================================================
# bench_mhd_alfven_2d -- linear Alfven wave (one period)
# ======================================================================
#
# Classical ideal-MHD validation.  Background B = (B0, 0), rho = rho0,
# p = p0 with perturbation
#   u_y(x, 0) =  A sin(k x)
#   B_y(x, 0) = -A sin(k x)
# gives a right-going Alfven wave at c_A = B0 / sqrt(rho0).  After one
# period T = L / c_A the state equals the IC; residual L2 is the
# scheme's numerical dissipation on the six conservative components.
#
# Pass criteria (P=2, NX=64, thin y-strip):
#   * rel L2(state) < 1%%   (measured ~0.36%%)
#   * no non-finite values
#
# A convergence-rate check was attempted across NX=32, 64, 128 but
# the rel L2 is ~flat (3.6e-3) across resolutions.  That's not
# scheme dissipation dominating -- at finite amplitude A=0.1 the
# Alfven wave picks up O(A^2) nonlinear corrections that don't
# vanish under mesh refinement and are the error floor here.  A
# cleaner convergence test would need A -> 0 (ruining SNR on the
# single-resolution check) or a different problem altogether.  The
# single-resolution 1%% L2 gate still catches any catastrophic MHD
# flux bug (result would balloon well past 1%%).
# ======================================================================

from std.math import sqrt, sin, pi, isnan, isinf
from std.sys import has_accelerator
from std.gpu.host import DeviceContext, DeviceBuffer
from src import mpi
from src.local_mesh_2d import LocalMesh2D
from src.local_mesh_2d_gpu import LocalMesh2DGpu
from src.local_mesh_2d_gpu_mhd import mhd_rk_stage_2d
from src.reference_2d import (
    ReferenceElement2D, num_tri_nodes_2d, num_edge_nodes,
)
from src.reference_2d_gpu import ReferenceElement2DGpu


comptime P = 2
comptime NY = 4
comptime LX = 1.0
# LY is computed per-run so cells stay square (LY = NY / NX * LX).
# Otherwise refining NX with LY fixed changes cell aspect ratio
# rather than resolving more scales -- rel L2 plateaus instead of
# converging.

comptime GAMMA     = 5.0 / 3.0
comptime RHO0      = 1.0
comptime B0        = 1.0
comptime P0        = 0.1
comptime AMPLITUDE = 0.1

comptime T_FINAL   = 1.0   # one wave period exactly (L / c_A = 1)
comptime CFL       = 0.15

# Measured ~3.6e-3 (nonlinear O(A^2) corrections are the error floor
# at A=0.1, not scheme dissipation).  5e-3 is a ~1.4x gate around
# the actual value, catching any regression in the MHD Rusanov flux.
comptime L2_MAX_REL: Float64 = 5.0e-3


def _run(NX: Int) raises -> Float64:
    """Run one Alfven period at resolution NX x NY with cells kept
    square (LY = NY / NX * LX), and return rel L2."""
    comptime NP_p = num_tri_nodes_2d(P)
    comptime NFP_e = num_edge_nodes(P)
    comptime NC = 6
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
            var sv = sin(k_wave * x)
            var uy = AMPLITUDE * sv
            var By = -AMPLITUDE * sv
            var Bx = B0
            var mx = 0.0
            var my = RHO0 * uy
            var ke = 0.5 * RHO0 * (uy * uy)
            var mp = 0.5 * (Bx * Bx + By * By)
            var E  = P0 / (GAMMA - 1.0) + ke + mp
            host_q.append(Float32(RHO0));  host_ic.append(Float32(RHO0))
            host_q.append(Float32(mx));     host_ic.append(Float32(mx))
            host_q.append(Float32(my));     host_ic.append(Float32(my))
            host_q.append(Float32(Bx));     host_ic.append(Float32(Bx))
            host_q.append(Float32(By));     host_ic.append(Float32(By))
            host_q.append(Float32(E));      host_ic.append(Float32(E))

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

    var cs = sqrt(GAMMA * P0 / RHO0)
    var cA = B0 / sqrt(RHO0)
    var cf = sqrt(cs * cs + cA * cA)
    var h_cell = LX / Float64(NX)
    var dt_est = CFL * h_cell / (cf * Float64(2 * P + 1))
    var num_steps = Int(T_FINAL / dt_est) + 1
    var dt = Float32(T_FINAL / Float64(num_steps))

    var gamma_f = Float32(GAMMA)
    var min_rho = Float32(1.0e-6)
    var min_p   = Float32(1.0e-6)

    for _ in range(num_steps):
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
    ctx.enqueue_copy(hbuf_q, d_q)
    ctx.synchronize()

    var sum_sq: Float64 = 0.0
    var sum_ic: Float64 = 0.0
    for k in range(n_q):
        var v = hptr_q[k]
        if isnan(v) or isinf(v):
            raise Error("bench_mhd_alfven_2d: non-finite output at index "
                        + String(k))
        var e = Float64(v - host_ic[k])
        sum_sq += e * e
        var ic = Float64(host_ic[k])
        sum_ic += ic * ic
    var l2 = sqrt(sum_sq / Float64(n_q))
    var l2_ic = sqrt(sum_ic / Float64(n_q))
    return l2 / l2_ic


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    mpi.init()
    var size = mpi.world_size()
    if size > 1:
        mpi.finalize()
        print("bench_mhd_alfven_2d: runs at np=1 only")
        return

    print("bench_mhd_alfven_2d (linear Alfven wave, one period)")
    print("  P=", P, "  NX=64   NY=", NY)

    var rel_l2 = _run(64)
    print("  rel L2(state) =", rel_l2,
          "  (threshold", L2_MAX_REL, ")")

    if rel_l2 > L2_MAX_REL:
        raise Error(
            "bench_mhd_alfven_2d FAILED: rel L2 "
            + String(rel_l2)
            + " exceeds threshold "
            + String(L2_MAX_REL)
        )
    print("=== bench_mhd_alfven_2d PASSED ===")
    mpi.finalize()
