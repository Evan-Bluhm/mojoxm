# ======================================================================
# bench_euler_sod_2d -- classical Sod shock tube vs exact solution
# ======================================================================
#
# Runs Sod (rho_L=1, p_L=1, rho_R=0.125, p_R=0.1, u_L=u_R=0, gamma=1.4)
# on a 2D strip with HLLC and a mildly-smoothed IC, then compares the
# 1D profile along y = LY/2 to the exact self-similar Riemann solution
# at t = 0.20.
#
# Exact solution: find p_star via Newton iteration on Toro's
# pressure-balance equation (Toro 2009 ch. 4), then the five wave
# regions (left plateau, rarefaction fan, star-left, star-right,
# right plateau) each have analytic (rho, u, p).
#
# The IC is tanh-smoothed over 4 cells (~1.5%% of the domain) so P=2
# nodes don't straddle a discontinuity at t=0; HLLC is well-behaved
# on that profile without a slope limiter, and the physics matches
# the Riemann solution within one cell.  The BJ-limited variant
# `bench_euler_sod_limited_2d` exercises the same setup with the
# Venkat-smoothed Barth-Jespersen limiter on top (used to compare
# the limiter against the unlimited reference).
#
# Pass criteria (P=2, NX=256, NY=16, HLLC, T=0.20):
#   * Left plateau at x=0.10 matches rho_L = 1.0 to within 2%%
#   * Right plateau at x=0.95 matches rho_R = 0.125 to within 2%%
#   * Star-left plateau at x=0.60 matches rho_star_L to within 5%%
#   * Shock position (max |drho/dx| in x > 0.7) within 2 cells of
#     the analytic Rankine-Hugoniot shock position -- a direct test
#     that HLLC conserves momentum + energy across the shock and
#     the Davis wave speeds are right.
#   * No NaN / Inf anywhere.
# ======================================================================

from std.math import sqrt, tanh, isnan, isinf
from std.sys import has_accelerator
from std.gpu.host import DeviceContext, DeviceBuffer
from std.time import perf_counter_ns
from src import mpi
from src.local_mesh_2d import LocalMesh2D
from src.local_mesh_2d_gpu import LocalMesh2DGpu
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
from src.memory_report import format_seconds


comptime P = 2
comptime NX = 256
comptime NY = 16
comptime LX = 1.0
comptime LY = 0.0625
comptime GAMMA = 1.4
comptime RHO_L = 1.0
comptime P_L = 1.0
comptime RHO_R = 0.125
comptime P_R = 0.1
comptime T_FINAL = 0.20
comptime CFL = 0.15

# Tolerances on the analytic comparisons.
comptime PLATEAU_TOL_REL: Float64 = 0.02
comptime STAR_TOL_REL: Float64 = 0.05
comptime SHOCK_TOL_CELLS: Float64 = 2.0


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    mpi.init()
    var size = mpi.world_size()
    if size > 1:
        mpi.finalize()
        print("bench_euler_sod_2d: runs at np=1 only")
        return

    print("bench_euler_sod_2d (Sod shock tube vs exact Riemann)")
    print("  P=", P, "  mesh=", NX, "x", NY, "  HLLC (no limiter), T=", T_FINAL)

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

    # IC: tanh-smoothed jump over 4 cells -- wide enough that P=2
    # nodes don't straddle a discontinuity at t=0, narrow enough that
    # the first wave structure emerges very close to the analytic
    # Riemann solution.  Debug sweep (smooth_cells in {2, 4, ..., 16})
    # showed shock position within 1 cell of analytic across that
    # entire range with HLLC alone.
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
    var min_p = Float32(1.0e-6)

    var stage_plans = ssprk3_stage_plans(
        d_q=d_q.unsafe_ptr(),
        d_q1=d_q1.unsafe_ptr(),
        d_q2=d_q2.unsafe_ptr(),
    )
    var t0 = perf_counter_ns()
    for _ in range(num_steps):
        for stage in stage_plans:
            euler_rk_stage_hllc_2d[P](
                ctx=ctx,
                mesh=gpu_mesh,
                Lift_ref=gpu_re.d_Lift_ref.unsafe_ptr(),
                D_ref=gpu_re.d_D_ref.unsafe_ptr(),
                q_in=stage.q_in,
                q_a=stage.q_a,
                q_b=stage.q_b,
                q_out=stage.q_out,
                fstar_scratch=d_fstar.unsafe_ptr(),
                gamma=gamma,
                min_density=min_rho,
                min_pressure=min_p,
                a=stage.a,
                b=stage.b,
                cc=stage.c,
                dt=dt,
            )
    ctx.synchronize()
    var t1 = perf_counter_ns()
    var wall_sec = Float64(t1 - t0) * 1.0e-9
    print(
        "  steps=",
        num_steps,
        "  dt=",
        dt,
        "  wall=",
        format_seconds(wall_sec),
        "  throughput=",
        Float64(num_steps) / wall_sec,
        "steps/s",
    )

    ctx.enqueue_copy(hbuf_q, d_q)
    ctx.synchronize()

    # Build a 1D profile rho(x) along the centerline y ≈ LY/2 by
    # averaging each x-bin (one column of triangulated cells) over
    # its nodes.  Simple and deterministic: the mesh's elem_node_xyz
    # gives us nodal positions; we bin by floor((x - 0) / dx_cell).
    var bins = List[Float64]()
    var counts = List[Int]()
    for _ in range(NX):
        bins.append(0.0)
        counts.append(0)
    var dx_cell = LX / Float64(NX)
    var y_lo = 0.5 * LY - dx_cell
    var y_hi = 0.5 * LY + dx_cell
    var finite = True
    for elem in range(gpu_mesh.num_elements):
        for nn in range(NP_p):
            var v = hptr_q[(elem * NP_p + nn) * NC + 0]
            if isnan(v) or isinf(v):
                finite = False
                break
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
        if not finite:
            break
    if not finite:
        raise Error("bench_euler_sod_2d: non-finite output")
    var profile = List[Float64]()
    for ix in range(NX):
        if counts[ix] > 0:
            profile.append(bins[ix] / Float64(counts[ix]))
        else:
            profile.append(0.0)

    # Analytic reference at T_FINAL.
    var rho_ref_L = sod_exact_rho(0.10, T_FINAL, GAMMA, RHO_L, P_L, RHO_R, P_R)
    var rho_ref_R = sod_exact_rho(0.95, T_FINAL, GAMMA, RHO_L, P_L, RHO_R, P_R)
    var rho_ref_star = sod_exact_rho(
        0.60, T_FINAL, GAMMA, RHO_L, P_L, RHO_R, P_R
    )

    # Bin helper inlined (no nested def -- Mojo nested defs can't
    # capture the NX/LX comptime constants).
    var ix_L = Int((0.10 / LX) * Float64(NX))
    var ix_R = Int((0.95 / LX) * Float64(NX))
    var ix_star = Int((0.60 / LX) * Float64(NX))

    var meas_L = profile[ix_L]
    var meas_R = profile[ix_R]
    var meas_star = profile[ix_star]

    print(
        "  exact rho_L plateau   =", rho_ref_L, "  measured (x=0.10) =", meas_L
    )
    print(
        "  exact rho_R plateau   =", rho_ref_R, "  measured (x=0.95) =", meas_R
    )
    print(
        "  exact rho_star (left) =",
        rho_ref_star,
        "  measured (x=0.60) =",
        meas_star,
    )

    var err_L = (meas_L - rho_ref_L) / rho_ref_L
    if err_L < 0.0:
        err_L = -err_L
    if err_L > PLATEAU_TOL_REL:
        raise Error(
            String("bench_euler_sod_2d FAILED: left plateau rel err ")
            + String(err_L)
            + " exceeds "
            + String(PLATEAU_TOL_REL)
        )
    var err_R = (meas_R - rho_ref_R) / rho_ref_R
    if err_R < 0.0:
        err_R = -err_R
    if err_R > PLATEAU_TOL_REL:
        raise Error(
            String("bench_euler_sod_2d FAILED: right plateau rel err ")
            + String(err_R)
            + " exceeds "
            + String(PLATEAU_TOL_REL)
        )
    var err_star = (meas_star - rho_ref_star) / rho_ref_star
    if err_star < 0.0:
        err_star = -err_star
    if err_star > STAR_TOL_REL:
        raise Error(
            String("bench_euler_sod_2d FAILED: star-left plateau rel err ")
            + String(err_star)
            + " exceeds "
            + String(STAR_TOL_REL)
        )

    # Shock position: locate the largest |drho/dx| in the right half
    # of the domain.  HLLC alone should have the shock at the exact
    # Rankine-Hugoniot position within a cell or two -- this is the
    # tightest gate in the benchmark because the shock speed comes
    # directly from the flux's conservation properties.
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
        raise Error("bench_euler_sod_2d FAILED: could not locate shock front")
    var x_shock_meas = (Float64(ix_shock) - 0.5) * dx_cell

    # Analytic shock x: 0.5 + t * S_R  via Rankine-Hugoniot.
    var S_R = shock_speed_S_R(RHO_L, P_L, RHO_R, P_R, GAMMA)
    var x_shock_exact = 0.5 + T_FINAL * S_R
    var shock_err_cells = (x_shock_meas - x_shock_exact) / dx_cell
    var a_shock_err_cells = (
        shock_err_cells if shock_err_cells >= 0.0 else -shock_err_cells
    )
    print(
        "  exact shock x =",
        x_shock_exact,
        "  measured =",
        x_shock_meas,
        "  err =",
        a_shock_err_cells,
        "cells",
        "  (threshold",
        SHOCK_TOL_CELLS,
        ")",
    )

    if a_shock_err_cells > SHOCK_TOL_CELLS:
        raise Error(
            String("bench_euler_sod_2d FAILED: shock position off by ")
            + String(a_shock_err_cells)
            + " cells (tol "
            + String(SHOCK_TOL_CELLS)
            + ")"
        )

    print("=== bench_euler_sod_2d PASSED ===")
    mpi.finalize()
