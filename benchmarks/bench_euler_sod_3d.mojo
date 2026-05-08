# ======================================================================
# bench_euler_sod_3d -- 3D Sod shock tube (bounds + conservation gate)
# ======================================================================
#
# 3D analog of bench_euler_sod_limited_2d.  Sod (rho_L=1, p_L=1,
# rho_R=0.125, p_R=0.1, u_L=u_R=0, gamma=1.4) on a long-x rectangular
# domain (1 x 0.04 x 0.04) with transmissive outflow on x and slip
# walls on y/z.  The IC is uniform in y and z so the 1D Riemann
# solution is the exact 3D solution (u_y=u_z=0 keeps slip walls a
# no-op), which means the analytic plateau bounds [RHO_R, RHO_L]
# must hold everywhere in rho at every time.  Running with the BJ
# cell limiter enabled (without which P=2 DG on an unsmoothed Sod
# blows up past t~0.15).
#
# IC smoothing: 8-cell tanh.  Without it the P=2 Lagrange nodes
# straddle the discontinuity and the IC already violates the bounds.
#
# Pass criteria (P=2, NX=100, NY=NZ=4, HLLEC, 8-cell IC smoothing,
# BJ + Venkat(eps=0.1), T=0.20):
#   * rho_max in [RHO_R, RHO_L] to 1e-3 (no over/under-shoot from
#     the limiter -- this is the conservation / bounds guarantee the
#     mass-matrix-weighted BJ limiter is supposed to give).
#   * rho_min > 0 (positivity).
#   * total mass change relative to IC < 0.5% over the run (the BJ
#     limiter is exactly conservative in the mean; outflow is the
#     only mass sink, and at t=0.20 the shock sits at x~0.85, so
#     very little mass has left -- large drift here signals a
#     non-conservative limiter).
#   * no NaN / Inf
#
# This is a weaker gate than the 2D Sod driver (which compares the
# full centerline rho profile to the exact Riemann solution), but
# it exercises the full 3D Sod-with-limiter pipeline and catches
# any regression that violates TVD or mass conservation.  A finer
# Toro-Riemann comparison is on the table once a per-cell global-id
# decoder with x-bucketing is plumbed through to this driver.
# ======================================================================

from std.sys import has_accelerator
from std.gpu import global_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from std.math import sqrt, ceildiv, tanh, isnan, isinf

from src import mpi
from src.partition import build_partition
from src.reference import N_P, build_reference_operators
from src.mesh import Mesh
from src.boundary import (
    BoundaryConditions,
    BC_INTERIOR,
    BC_WALL,
    BC_OUTFLOW,
)
from src.halo_exchange import HaloExchange
from src.solver import Solver
from src.euler import Euler, FLUX_HLLEC
from src.nvtx import NvtxContext


comptime NX = 100
comptime NY = 4
comptime NZ = 4
comptime LX = 1.0
comptime LY = Float64(4.0 / 100.0)
comptime LZ = Float64(4.0 / 100.0)

comptime GAMMA: Float32 = 1.4
comptime RHO_L: Float32 = 1.0
comptime P_L: Float32 = 1.0
comptime RHO_R: Float32 = 0.125
comptime P_R: Float32 = 0.1
comptime T_FINAL: Float32 = 0.20
comptime CFL = Float32(0.2)
comptime IC_BLOCK = 256
comptime SMOOTH_WIDTH: Float32 = Float32(8.0 * (LX / NX))

comptime BOUNDS_SLACK: Float32 = Float32(1.0e-3)
comptime MASS_TOL_REL: Float64 = 5.0e-3


def sod_ic_kernel(
    q: UnsafePointer[Float32, MutAnyOrigin],
    owned_elem_ids: UnsafePointer[Int32, MutAnyOrigin],
    elem_node_xyz: UnsafePointer[Float32, MutAnyOrigin],
    num_owned: Int,
):
    var idx = Int(global_idx.x)
    var total = num_owned * N_P
    if idx >= total:
        return
    var i = idx // N_P
    var nn = idx % N_P
    var e = Int(owned_elem_ids[i])
    var px = elem_node_xyz[(e * N_P + nn) * 3 + 0]
    var s = (tanh((px - Float32(0.5)) / SMOOTH_WIDTH) + Float32(1.0)) * Float32(
        0.5
    )
    var rho = RHO_L + s * (RHO_R - RHO_L)
    var p = P_L + s * (P_R - P_L)
    var E = p / (GAMMA - Float32(1.0))
    var base = (e * N_P + nn) * 5
    q[base + 0] = rho
    q[base + 1] = Float32(0.0)
    q[base + 2] = Float32(0.0)
    q[base + 3] = Float32(0.0)
    q[base + 4] = E


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    mpi.init()
    var size = mpi.world_size()
    if size > 1:
        mpi.finalize()
        print("bench_euler_sod_3d: runs at np=1 only")
        return

    print("bench_euler_sod_3d (3D Sod shock tube, BJ-limited bounds gate)")
    print("  P=2  mesh=", NX, "x", NY, "x", NZ, "  T=", T_FINAL)

    var rank = mpi.world_rank()
    var nvtx = NvtxContext()
    var refs = build_reference_operators(nvtx)
    var ctx = DeviceContext()

    var bcs = BoundaryConditions(
        BC_OUTFLOW,
        BC_OUTFLOW,
        BC_WALL,
        BC_WALL,
        BC_WALL,
        BC_WALL,
    )
    var mesh = Mesh(
        ctx,
        build_partition(rank, size, NX, NY, NZ),
        LX,
        LY,
        LZ,
        bcs,
    )
    var halo = HaloExchange(
        ctx,
        mesh.part,
        Euler.NUM_COMPONENTS,
        mesh.d_perm.unsafe_ptr(),
        bcs,
    )
    var physics = Euler(
        GAMMA,
        Float32(1.0e-6),
        Float32(1.0e-6),
        FLUX_HLLEC,
        False,
    )
    var solver = Solver[Euler](
        ctx^,
        mesh^,
        halo^,
        physics^,
        refs.D_ref^,
        refs.Lift_ref^,
        refs.node_weights^,
    )
    solver.enable_cell_limiter(True, Float32(0.1))

    solver.ctx.enqueue_function[sod_ic_kernel](
        solver.d_q.unsafe_ptr(),
        solver.mesh.d_owned_elem_ids.unsafe_ptr(),
        solver.mesh.local.d_elem_node_xyz.unsafe_ptr(),
        solver.num_owned_elements,
        grid_dim=ceildiv(solver.num_owned_elements * N_P, IC_BLOCK),
        block_dim=IC_BLOCK,
    )
    solver.ctx.synchronize()

    # Snapshot initial mean density for the mass-conservation check.
    var n_owned = solver.num_owned_elements
    var n_dof = n_owned * N_P
    var rho_buf = List[Float32]()
    for _ in range(n_dof):
        rho_buf.append(Float32(0.0))
    solver.download_owned_component(0, rho_buf, nvtx)
    var mass_ic: Float64 = 0.0
    for i in range(n_dof):
        mass_ic += Float64(rho_buf[i])
    mass_ic /= Float64(n_dof)

    var h = Float32(LX) / Float32(NX)
    var c_L = sqrt(GAMMA * P_L / RHO_L)
    var wave = Float32(2.0) * c_L
    var dt_est = CFL * h / (wave * Float32(5.0))
    var num_steps = Int(T_FINAL / dt_est) + 1
    var dt = T_FINAL / Float32(num_steps)
    print("  steps=", num_steps, "  dt=", dt)

    for _ in range(num_steps):
        solver.step_ssprk3(dt, nvtx)
    solver.ctx.synchronize()

    # Pull final rho and compute summary stats.
    solver.download_owned_component(0, rho_buf, nvtx)
    var rho_max: Float32 = Float32(-1.0e30)
    var rho_min: Float32 = Float32(1.0e30)
    var mass_fin: Float64 = 0.0
    for i in range(n_dof):
        var v = rho_buf[i]
        if isnan(v) or isinf(v):
            raise Error("bench_euler_sod_3d: non-finite density")
        if v > rho_max:
            rho_max = v
        if v < rho_min:
            rho_min = v
        mass_fin += Float64(v)
    mass_fin /= Float64(n_dof)

    print("  rho_max =", rho_max, "  (bound =", RHO_L, ")")
    print("  rho_min =", rho_min, "  (bound =", RHO_R, ", positive)")
    print("  mass(IC) =", mass_ic, "  mass(t=T) =", mass_fin)

    # Bounds: no over/undershoot past the analytic plateau.  A strict
    # bound on the over-shoot side; rho_min can legitimately be below
    # RHO_R only in machine-epsilon territory after the shock passes,
    # but for a well-limited P=2 scheme it should stay above RHO_R -
    # slack.
    if rho_max > RHO_L + BOUNDS_SLACK:
        raise Error(
            String("bench_euler_sod_3d FAILED: rho_max ")
            + String(rho_max)
            + " overshot RHO_L="
            + String(RHO_L)
        )
    if rho_min < Float32(0.0):
        raise Error(
            String("bench_euler_sod_3d FAILED: rho_min ")
            + String(rho_min)
            + " negative (positivity lost)"
        )
    if rho_min < RHO_R - BOUNDS_SLACK:
        raise Error(
            String("bench_euler_sod_3d FAILED: rho_min ")
            + String(rho_min)
            + " undershot RHO_R="
            + String(RHO_R)
        )

    # Mass conservation: outflow at t=0.20 has barely started (shock
    # at x~0.85), so mass drift should be << 1%.
    var dmass = mass_fin - mass_ic
    if dmass < 0.0:
        dmass = -dmass
    var rel = dmass / mass_ic
    if rel > MASS_TOL_REL:
        raise Error(
            String("bench_euler_sod_3d FAILED: mass drift ")
            + String(rel * 100.0)
            + "%% > tol "
            + String(MASS_TOL_REL * 100.0)
            + "%%"
        )

    print("=== bench_euler_sod_3d PASSED ===")
    mpi.finalize()
