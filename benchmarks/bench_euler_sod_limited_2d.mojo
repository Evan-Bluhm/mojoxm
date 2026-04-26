# ======================================================================
# bench_euler_sod_limited_2d -- Sod vs exact Riemann with BJ limiter
# ======================================================================
#
# Runs the classical Sod shock tube with HLLC + BJ cell-level limiter
# (applied after every RK stage) and asserts the result matches the
# analytic Riemann solution to the same tolerances as the unlimited
# bench_euler_sod_2d.  This is the combined-correctness gate for the
# limited path: any regression in the limiter or its mean-computation
# would show up as shock displacement here.
#
# Before the mass-matrix-weighted cell mean landed (commit pre-#34
# fix), this benchmark would have failed with a ~10-cell shock
# displacement -- the old `cell_avg_kernel_2d` computed an unweighted
# nodal average, which at P=2 is not the true cell mean (3 vertex
# weights are 0, 3 midpoint weights are 1/3).  Post-fix, the limiter
# is strictly cell-mean-preserving and the shock lands within 2 cells
# of analytic.
#
# Pass criteria (P=2, NX=256, NY=16, HLLC + BJ limiter, T=0.20):
#   * Left/right plateaus within 2%%
#   * Star-left plateau within 5%%
#   * Shock position within 2 cells of Rankine-Hugoniot
#   * No NaN / Inf
# ======================================================================

from std.math import sqrt, pow, tanh, isnan, isinf
from std.sys import has_accelerator
from std.gpu.host import DeviceContext, DeviceBuffer
from src import mpi
from src.local_mesh_2d import LocalMesh2D
from src.local_mesh_2d_gpu import LocalMesh2DGpu
from src.local_mesh_2d_gpu_limiter import bj_limit_full_2d
from src.local_mesh_2d_gpu_euler import euler_rk_stage_hllc_2d
from src.reference_2d import (
    ReferenceElement2D, num_tri_nodes_2d, num_edge_nodes,
)
from src.reference_2d_gpu import ReferenceElement2DGpu
from src.boundary import BoundaryConditions2D, BC_WALL, BC_OUTFLOW


comptime P = 2
comptime NX = 256
comptime NY = 16
comptime LX = 1.0
comptime LY = 0.0625
comptime GAMMA = 1.4
comptime RHO_L = 1.0
comptime P_L   = 1.0
comptime RHO_R = 0.125
comptime P_R   = 0.1
comptime T_FINAL = 0.20
comptime CFL = 0.15
comptime VENKAT_EPS = 0.1

comptime PLATEAU_TOL_REL: Float64 = 0.02
comptime STAR_TOL_REL:    Float64 = 0.05
comptime SHOCK_TOL_CELLS: Float64 = 2.0


def _f_K(p: Float64, rho_K: Float64, p_K: Float64, a_K: Float64) -> Float64:
    if p > p_K:
        var A = 2.0 / ((GAMMA + 1.0) * rho_K)
        var B = (GAMMA - 1.0) / (GAMMA + 1.0) * p_K
        return (p - p_K) * sqrt(A / (p + B))
    else:
        var e = (GAMMA - 1.0) / (2.0 * GAMMA)
        return (2.0 * a_K / (GAMMA - 1.0)) * (pow(p / p_K, e) - 1.0)


def _solve_star(
    rho_L: Float64, u_L: Float64, p_L: Float64, a_L: Float64,
    rho_R: Float64, u_R: Float64, p_R: Float64, a_R: Float64,
) raises -> Float64:
    var e = (GAMMA - 1.0) / (2.0 * GAMMA)
    var p_tr = pow(
        (a_L + a_R - 0.5 * (GAMMA - 1.0) * (u_R - u_L))
        / (a_L / pow(p_L, e) + a_R / pow(p_R, e)),
        1.0 / e,
    )
    var p = p_tr if p_tr > 0.0 else 0.5 * (p_L + p_R)
    var tol: Float64 = 1.0e-10
    for _ in range(50):
        var fL = _f_K(p, rho_L, p_L, a_L)
        var fR = _f_K(p, rho_R, p_R, a_R)
        var dfL: Float64
        var dfR: Float64
        if p > p_L:
            var A_L = 2.0 / ((GAMMA + 1.0) * rho_L)
            var B_L = (GAMMA - 1.0) / (GAMMA + 1.0) * p_L
            var sL = sqrt(A_L / (p + B_L))
            dfL = sL * (1.0 - 0.5 * (p - p_L) / (p + B_L))
        else:
            dfL = (1.0 / (rho_L * a_L)) * pow(p / p_L, -(GAMMA + 1.0) / (2.0 * GAMMA))
        if p > p_R:
            var A_R = 2.0 / ((GAMMA + 1.0) * rho_R)
            var B_R = (GAMMA - 1.0) / (GAMMA + 1.0) * p_R
            var sR = sqrt(A_R / (p + B_R))
            dfR = sR * (1.0 - 0.5 * (p - p_R) / (p + B_R))
        else:
            dfR = (1.0 / (rho_R * a_R)) * pow(p / p_R, -(GAMMA + 1.0) / (2.0 * GAMMA))
        var resid = fL + fR + (u_R - u_L)
        var dp = -resid / (dfL + dfR)
        var p_new = p + dp
        if p_new <= 0.0:
            p_new = 0.5 * p
        var rel = 2.0 * (p_new - p) / (p_new + p)
        var arel = rel if rel >= 0.0 else -rel
        p = p_new
        if arel < tol:
            return p
    raise Error("Sod Newton did not converge")


def _sod_exact_rho(x: Float64, t: Float64) raises -> Float64:
    var xi = (x - 0.5) / t
    var a_L = sqrt(GAMMA * P_L / RHO_L)
    var a_R = sqrt(GAMMA * P_R / RHO_R)
    var p_star = _solve_star(RHO_L, 0.0, P_L, a_L, RHO_R, 0.0, P_R, a_R)
    var u_star = 0.5 * (
        _f_K(p_star, RHO_R, P_R, a_R) - _f_K(p_star, RHO_L, P_L, a_L)
    )
    var rho_star_L = RHO_L * pow(p_star / P_L, 1.0 / GAMMA)
    var a_star_L = a_L * pow(p_star / P_L, (GAMMA - 1.0) / (2.0 * GAMMA))
    var xi_head_L = -a_L
    var xi_tail_L = u_star - a_star_L
    var rho_star_R = RHO_R * (
        (p_star / P_R + (GAMMA - 1.0) / (GAMMA + 1.0))
        / ((GAMMA - 1.0) / (GAMMA + 1.0) * p_star / P_R + 1.0)
    )
    var S_R = a_R * sqrt(
        (GAMMA + 1.0) / (2.0 * GAMMA) * p_star / P_R
        + (GAMMA - 1.0) / (2.0 * GAMMA)
    )
    if xi < xi_head_L:
        return RHO_L
    elif xi < xi_tail_L:
        var v = 2.0 / (GAMMA + 1.0) * (a_L + xi)
        var a = a_L - 0.5 * (GAMMA - 1.0) * v
        return RHO_L * pow(a / a_L, 2.0 / (GAMMA - 1.0))
    elif xi < u_star:
        return rho_star_L
    elif xi < S_R:
        return rho_star_R
    return RHO_R


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    mpi.init()
    var size = mpi.world_size()
    if size > 1:
        mpi.finalize()
        print("bench_euler_sod_limited_2d: runs at np=1 only")
        return

    print("bench_euler_sod_limited_2d (Sod HLLC + BJ limiter vs exact)")
    print("  P=", P, "  mesh=", NX, "x", NY, "  T=", T_FINAL)

    comptime NP_p = num_tri_nodes_2d(P)
    comptime NFP_e = num_edge_nodes(P)
    comptime NC = 4
    var ctx = DeviceContext()

    var bcs = BoundaryConditions2D(
        BC_OUTFLOW, BC_OUTFLOW,
        BC_WALL,    BC_WALL,
    )
    var host_mesh = LocalMesh2D[P](NX, NY, LX, LY, bcs)
    var host_re = ReferenceElement2D[P]()
    var mesh_coords = LocalMesh2D[P](NX, NY, LX, LY, bcs)
    var gpu_mesh = LocalMesh2DGpu[P](ctx, host_mesh^)
    var gpu_re = ReferenceElement2DGpu[P](ctx, host_re)

    # Smoothed IC over 4 cells -- same as the unlimited Sod benchmark.
    var smooth_width = 4.0 * (LX / Float64(NX))
    var n_q = gpu_mesh.num_elements * NP_p * NC
    var host_q = List[Float32]()
    for elem in range(gpu_mesh.num_elements):
        for nn in range(NP_p):
            var x = mesh_coords.elem_node_xyz[(elem * NP_p + nn) * 2 + 0]
            var s = 0.5 * (tanh((x - 0.5 * LX) / smooth_width) + 1.0)
            var rho = RHO_L + s * (RHO_R - RHO_L)
            var p   = P_L   + s * (P_R   - P_L)
            var E = p / (GAMMA - 1.0)
            host_q.append(Float32(rho))
            host_q.append(Float32(0.0))
            host_q.append(Float32(0.0))
            host_q.append(Float32(E))

    var d_q  = ctx.enqueue_create_buffer[DType.float32](n_q)
    var d_q1 = ctx.enqueue_create_buffer[DType.float32](n_q)
    var d_q2 = ctx.enqueue_create_buffer[DType.float32](n_q)
    var d_fstar = ctx.enqueue_create_buffer[DType.float32](
        gpu_mesh.num_faces * NFP_e * NC
    )
    var d_cell_mean = ctx.enqueue_create_buffer[DType.float32](
        gpu_mesh.num_elements * NC
    )
    var hbuf_q = ctx.enqueue_create_host_buffer[DType.float32](n_q)
    var hptr_q = hbuf_q.unsafe_ptr()
    for k in range(n_q):
        hptr_q[k] = host_q[k]
    ctx.enqueue_copy(d_q, hbuf_q)
    ctx.synchronize()

    var c_peak = sqrt(GAMMA * P_L / RHO_L)
    var h_cell = LX / Float64(NX)
    var dt_est = CFL * h_cell / (2.0 * c_peak * Float64(2 * P + 1))
    var num_steps = Int(T_FINAL / dt_est) + 1
    var dt = Float32(T_FINAL / Float64(num_steps))

    var gamma = Float32(GAMMA)
    var min_rho = Float32(1.0e-6)
    var min_p   = Float32(1.0e-6)
    var venkat_eps = Float32(VENKAT_EPS)

    for _ in range(num_steps):
        euler_rk_stage_hllc_2d[P](
            ctx, gpu_mesh,
            gpu_re.d_Lift_ref.unsafe_ptr(), gpu_re.d_D_ref.unsafe_ptr(),
            d_q.unsafe_ptr(),
            d_q.unsafe_ptr(), d_q.unsafe_ptr(),
            d_q1.unsafe_ptr(),
            d_fstar.unsafe_ptr(),
            gamma, min_rho, min_p,
            Float32(1.0), Float32(0.0), Float32(1.0), dt,
        )
        bj_limit_full_2d[P, NC](
            ctx, gpu_mesh, d_q1.unsafe_ptr(),
            gpu_re.d_node_weights.unsafe_ptr(),
            d_cell_mean.unsafe_ptr(), venkat_eps,
        )
        euler_rk_stage_hllc_2d[P](
            ctx, gpu_mesh,
            gpu_re.d_Lift_ref.unsafe_ptr(), gpu_re.d_D_ref.unsafe_ptr(),
            d_q1.unsafe_ptr(),
            d_q.unsafe_ptr(), d_q1.unsafe_ptr(),
            d_q2.unsafe_ptr(),
            d_fstar.unsafe_ptr(),
            gamma, min_rho, min_p,
            Float32(0.75), Float32(0.25), Float32(0.25), dt,
        )
        bj_limit_full_2d[P, NC](
            ctx, gpu_mesh, d_q2.unsafe_ptr(),
            gpu_re.d_node_weights.unsafe_ptr(),
            d_cell_mean.unsafe_ptr(), venkat_eps,
        )
        euler_rk_stage_hllc_2d[P](
            ctx, gpu_mesh,
            gpu_re.d_Lift_ref.unsafe_ptr(), gpu_re.d_D_ref.unsafe_ptr(),
            d_q2.unsafe_ptr(),
            d_q.unsafe_ptr(), d_q2.unsafe_ptr(),
            d_q.unsafe_ptr(),
            d_fstar.unsafe_ptr(),
            gamma, min_rho, min_p,
            Float32(1.0 / 3.0), Float32(2.0 / 3.0),
            Float32(2.0 / 3.0), dt,
        )
        bj_limit_full_2d[P, NC](
            ctx, gpu_mesh, d_q.unsafe_ptr(),
            gpu_re.d_node_weights.unsafe_ptr(),
            d_cell_mean.unsafe_ptr(), venkat_eps,
        )
    ctx.synchronize()

    ctx.enqueue_copy(hbuf_q, d_q)
    ctx.synchronize()

    # Bin centerline profile.
    var bins = List[Float64]()
    var counts = List[Int]()
    for _ in range(NX):
        bins.append(0.0)
        counts.append(0)
    var dx_cell = LX / Float64(NX)
    var y_lo = 0.5 * LY - dx_cell
    var y_hi = 0.5 * LY + dx_cell
    for elem in range(gpu_mesh.num_elements):
        for nn in range(NP_p):
            var v = hptr_q[(elem * NP_p + nn) * NC + 0]
            if isnan(v) or isinf(v):
                raise Error("bench_euler_sod_limited_2d: non-finite density")
            var x = mesh_coords.elem_node_xyz[(elem * NP_p + nn) * 2 + 0]
            var y = mesh_coords.elem_node_xyz[(elem * NP_p + nn) * 2 + 1]
            if y < y_lo or y > y_hi: continue
            var ix = Int((x / LX) * Float64(NX))
            if ix < 0: ix = 0
            if ix >= NX: ix = NX - 1
            bins[ix] += Float64(v)
            counts[ix] += 1
    var profile = List[Float64]()
    for ix in range(NX):
        if counts[ix] > 0:
            profile.append(bins[ix] / Float64(counts[ix]))
        else:
            profile.append(0.0)

    var rho_ref_L    = _sod_exact_rho(0.10, T_FINAL)
    var rho_ref_R    = _sod_exact_rho(0.95, T_FINAL)
    var rho_ref_star = _sod_exact_rho(0.60, T_FINAL)
    var ix_L    = Int((0.10 / LX) * Float64(NX))
    var ix_R    = Int((0.95 / LX) * Float64(NX))
    var ix_star = Int((0.60 / LX) * Float64(NX))
    var meas_L    = profile[ix_L]
    var meas_R    = profile[ix_R]
    var meas_star = profile[ix_star]
    print("  rho_L plateau:   ", meas_L, " vs", rho_ref_L)
    print("  rho_R plateau:   ", meas_R, " vs", rho_ref_R)
    print("  rho_star plateau:", meas_star, " vs", rho_ref_star)

    var err_L = (meas_L - rho_ref_L) / rho_ref_L
    if err_L < 0.0: err_L = -err_L
    if err_L > PLATEAU_TOL_REL:
        raise Error(
            String("bench_euler_sod_limited_2d FAILED: left plateau err ")
            + String(err_L)
        )
    var err_R = (meas_R - rho_ref_R) / rho_ref_R
    if err_R < 0.0: err_R = -err_R
    if err_R > PLATEAU_TOL_REL:
        raise Error(
            String("bench_euler_sod_limited_2d FAILED: right plateau err ")
            + String(err_R)
        )
    var err_star = (meas_star - rho_ref_star) / rho_ref_star
    if err_star < 0.0: err_star = -err_star
    if err_star > STAR_TOL_REL:
        raise Error(
            String("bench_euler_sod_limited_2d FAILED: star plateau err ")
            + String(err_star)
        )

    # Shock position: gradient-based detector on x > 0.7.
    var ix_start = Int((0.70 / LX) * Float64(NX))
    var max_grad: Float64 = 0.0
    var ix_shock: Int = -1
    for ix in range(ix_start, NX):
        var grad = profile[ix - 1] - profile[ix]
        var a = grad if grad >= 0.0 else -grad
        if a > max_grad:
            max_grad = a
            ix_shock = ix
    if ix_shock < 0:
        raise Error("bench_euler_sod_limited_2d FAILED: no shock front found")
    var x_shock_meas = (Float64(ix_shock) - 0.5) * dx_cell

    var a_R0 = sqrt(GAMMA * P_R / RHO_R)
    var a_L0 = sqrt(GAMMA * P_L / RHO_L)
    var p_star = _solve_star(RHO_L, 0.0, P_L, a_L0, RHO_R, 0.0, P_R, a_R0)
    var S_R = a_R0 * sqrt(
        (GAMMA + 1.0) / (2.0 * GAMMA) * p_star / P_R
        + (GAMMA - 1.0) / (2.0 * GAMMA)
    )
    var x_shock_exact = 0.5 + T_FINAL * S_R
    var shock_err_cells = (x_shock_meas - x_shock_exact) / dx_cell
    var a_shock_err_cells = shock_err_cells if shock_err_cells >= 0.0 else -shock_err_cells
    print("  shock x:          ", x_shock_meas, " vs", x_shock_exact,
          "  err:", a_shock_err_cells, "cells (tol", SHOCK_TOL_CELLS, ")")

    if a_shock_err_cells > SHOCK_TOL_CELLS:
        raise Error(
            String("bench_euler_sod_limited_2d FAILED: shock position off by ")
            + String(a_shock_err_cells)
            + " cells (tol " + String(SHOCK_TOL_CELLS) + ")"
        )

    print("=== bench_euler_sod_limited_2d PASSED ===")
    mpi.finalize()
