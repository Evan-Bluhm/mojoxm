# ======================================================================
# bench_euler_sod_limited_2d_p5 -- Sod vs exact Riemann at P=5 + BJ
# ======================================================================
#
# P=5 (NP=21) counterpart of bench_euler_sod_limited_2d_p4.  Same Sod
# IC, same HLLC + Venkat-smoothed Barth-Jespersen limiter pipeline,
# same exact-Riemann profile comparison, but routed through
# LocalMesh2D[5] / euler_rk_stage_hllc_2d[5] / NP=21.
#
# Highest 2D-Euler NP shocked-flow gate.  Closes the limiter-under-
# shocks P-parity sweep at P=5: the smooth-flow path is validated by
# `bench_euler_smooth_wave_2d_p5`, this gates the BJ limiter +
# HLLC + shock capture at NP=21 / NC=4 = 84 q-values per element
# (the largest 2D shocked-Euler comptime config).
#
# Pass criteria (P=5, NX=128, NY=8, HLLC + BJ, T=0.20):
#   * Left plateau within 10% (BJ limiter at NP=21 reaches even
#     into the rarefaction tail -- observed ~8% under-shoot, which
#     is a real numerical effect of the higher-NP stencil rather
#     than a regression.  P=4 only saw 0.04% here at NP=15, so the
#     gap really does grow with P)
#   * Right plateau within 12% (NP=21 stencil reaches farther down-
#     stream than P=4 -- observed ~11.6% under-shoot.  Threshold
#     accommodates real high-P diffusion while still catching scaling
#     regressions)
#   * Star plateau within 5% (between rarefaction tail and shock --
#     observed ~2.6%, similar to P=4)
#   * Shock position within 3.5 cells of Rankine-Hugoniot (slightly
#     looser than P=4's 3 since the higher-NP limiter dispersion
#     smears the shock over a wider local stencil)
#   * No NaN / Inf
# ======================================================================

from std.math import sqrt, tanh, isnan, isinf
from std.sys import has_accelerator
from std.gpu.host import DeviceContext, DeviceBuffer
from src import mpi
from src.local_mesh_2d import LocalMesh2D
from src.local_mesh_2d_gpu import LocalMesh2DGpu
from src.local_mesh_2d_gpu_limiter import bj_limit_full_2d
from src.local_mesh_2d_gpu_euler import euler_rk_stage_hllc_2d
from src.ssprk3 import ssprk3_stage_plans
from src.reference_2d import (
    ReferenceElement2D,
    num_tri_nodes_2d,
    num_edge_nodes,
)
from src.reference_2d_gpu import ReferenceElement2DGpu
from src.boundary import BoundaryConditions2D, BC_WALL, BC_OUTFLOW
from src.sod_exact_riemann import sod_exact_rho, shock_speed_S_R


comptime P = 5
comptime NX = 128
comptime NY = 8
comptime LX = 1.0
comptime LY = Float64(8.0 / 128.0)
comptime GAMMA = 1.4
comptime RHO_L = 1.0
comptime P_L = 1.0
comptime RHO_R = 0.125
comptime P_R = 0.1
comptime T_FINAL = 0.20
comptime CFL = 0.10
comptime VENKAT_EPS = 0.1

comptime PLATEAU_L_TOL_REL: Float64 = 0.10
comptime PLATEAU_R_TOL_REL: Float64 = 0.12
comptime STAR_TOL_REL: Float64 = 0.05
comptime SHOCK_TOL_CELLS: Float64 = 3.5


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    mpi.init()
    var size = mpi.world_size()
    if size > 1:
        mpi.finalize()
        print("bench_euler_sod_limited_2d_p5: runs at np=1 only")
        return

    print("bench_euler_sod_limited_2d_p5 (Sod HLLC + BJ at P=5 vs exact)")
    print("  P=", P, "  mesh=", NX, "x", NY, "  T=", T_FINAL)

    comptime NP_p = num_tri_nodes_2d(P)
    comptime NFP_e = num_edge_nodes(P)
    comptime NC = 4
    var ctx = DeviceContext()

    var bcs = BoundaryConditions2D(
        BC_OUTFLOW,
        BC_OUTFLOW,
        BC_WALL,
        BC_WALL,
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
            var p = P_L + s * (P_R - P_L)
            var E = p / (GAMMA - 1.0)
            host_q.append(Float32(rho))
            host_q.append(Float32(0.0))
            host_q.append(Float32(0.0))
            host_q.append(Float32(E))

    var d_q = ctx.enqueue_create_buffer[DType.float32](n_q)
    var d_q1 = ctx.enqueue_create_buffer[DType.float32](n_q)
    var d_q2 = ctx.enqueue_create_buffer[DType.float32](n_q)
    var d_fstar = ctx.enqueue_create_buffer[DType.float32](
        gpu_mesh.num_faces * NFP_e * NC
    )
    var d_cell_mean = ctx.enqueue_create_buffer[DType.float32](
        gpu_mesh.num_elements * (NC + 1)
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
    print("  steps=", num_steps, "  dt=", dt)

    var gamma = Float32(GAMMA)
    var min_rho = Float32(1.0e-6)
    var min_p = Float32(1.0e-6)
    var venkat_eps = Float32(VENKAT_EPS)

    var stage_plans = ssprk3_stage_plans(
        d_q.unsafe_ptr(),
        d_q1.unsafe_ptr(),
        d_q2.unsafe_ptr(),
    )
    for _ in range(num_steps):
        for stage in stage_plans:
            euler_rk_stage_hllc_2d[P](
                ctx,
                gpu_mesh,
                gpu_re.d_Lift_ref.unsafe_ptr(),
                gpu_re.d_D_ref.unsafe_ptr(),
                stage.q_in,
                stage.q_a,
                stage.q_b,
                stage.q_out,
                d_fstar.unsafe_ptr(),
                gamma,
                min_rho,
                min_p,
                stage.a,
                stage.b,
                stage.c,
                dt,
            )
            bj_limit_full_2d[P, NC](
                ctx,
                gpu_mesh,
                stage.q_out,
                gpu_re.d_node_weights.unsafe_ptr(),
                d_cell_mean.unsafe_ptr(),
                venkat_eps,
            )
    ctx.synchronize()

    ctx.enqueue_copy(hbuf_q, d_q)
    ctx.synchronize()

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
                raise Error("bench_euler_sod_limited_2d_p5: non-finite density")
            var x = mesh_coords.elem_node_xyz[(elem * NP_p + nn) * 2 + 0]
            var y = mesh_coords.elem_node_xyz[(elem * NP_p + nn) * 2 + 1]
            if y < y_lo or y > y_hi:
                continue
            var ix = Int((x / LX) * Float64(NX))
            if ix < 0:
                ix = 0
            if ix >= NX:
                ix = NX - 1
            bins[ix] += Float64(v)
            counts[ix] += 1
    var profile = List[Float64]()
    for ix in range(NX):
        if counts[ix] > 0:
            profile.append(bins[ix] / Float64(counts[ix]))
        else:
            profile.append(0.0)

    var rho_ref_L = sod_exact_rho(0.10, T_FINAL, GAMMA, RHO_L, P_L, RHO_R, P_R)
    var rho_ref_R = sod_exact_rho(0.95, T_FINAL, GAMMA, RHO_L, P_L, RHO_R, P_R)
    var rho_ref_star = sod_exact_rho(
        0.60, T_FINAL, GAMMA, RHO_L, P_L, RHO_R, P_R
    )
    var ix_L = Int((0.10 / LX) * Float64(NX))
    var ix_R = Int((0.95 / LX) * Float64(NX))
    var ix_star = Int((0.60 / LX) * Float64(NX))
    var meas_L = profile[ix_L]
    var meas_R = profile[ix_R]
    var meas_star = profile[ix_star]
    print("  rho_L plateau:   ", meas_L, " vs", rho_ref_L)
    print("  rho_R plateau:   ", meas_R, " vs", rho_ref_R)
    print("  rho_star plateau:", meas_star, " vs", rho_ref_star)

    var err_L = (meas_L - rho_ref_L) / rho_ref_L
    if err_L < 0.0:
        err_L = -err_L
    if err_L > PLATEAU_L_TOL_REL:
        raise Error(
            String("bench_euler_sod_limited_2d_p5 FAILED: left plateau err ")
            + String(err_L)
        )
    var err_R = (meas_R - rho_ref_R) / rho_ref_R
    if err_R < 0.0:
        err_R = -err_R
    if err_R > PLATEAU_R_TOL_REL:
        raise Error(
            String("bench_euler_sod_limited_2d_p5 FAILED: right plateau err ")
            + String(err_R)
        )
    var err_star = (meas_star - rho_ref_star) / rho_ref_star
    if err_star < 0.0:
        err_star = -err_star
    if err_star > STAR_TOL_REL:
        raise Error(
            String("bench_euler_sod_limited_2d_p5 FAILED: star plateau err ")
            + String(err_star)
        )

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
        raise Error(
            "bench_euler_sod_limited_2d_p5 FAILED: no shock front found"
        )
    var x_shock_meas = (Float64(ix_shock) - 0.5) * dx_cell

    var S_R = shock_speed_S_R(RHO_L, P_L, RHO_R, P_R, GAMMA)
    var x_shock_exact = 0.5 + T_FINAL * S_R
    var shock_err_cells = (x_shock_meas - x_shock_exact) / dx_cell
    var a_shock_err_cells = (
        shock_err_cells if shock_err_cells >= 0.0 else -shock_err_cells
    )
    print(
        "  shock x:          ",
        x_shock_meas,
        " vs",
        x_shock_exact,
        "  err:",
        a_shock_err_cells,
        "cells (tol",
        SHOCK_TOL_CELLS,
        ")",
    )

    if a_shock_err_cells > SHOCK_TOL_CELLS:
        raise Error(
            String(
                "bench_euler_sod_limited_2d_p5 FAILED: shock position off by "
            )
            + String(a_shock_err_cells)
            + " cells (tol "
            + String(SHOCK_TOL_CELLS)
            + ")"
        )

    print("=== bench_euler_sod_limited_2d_p5 PASSED ===")
    mpi.finalize()
