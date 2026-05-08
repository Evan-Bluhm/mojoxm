# ======================================================================
# euler_rising_bubble -- buoyant thermal bubble in a stratified box
# ======================================================================
#
# Demo of the Euler module's gravity vector (gx, gy, gz) source term
# in a non-trivial buoyancy setup (vs the static hydrostatic balance
# gated by `bench_euler_hydrostatic_3d` + `_p3`).  Setup:
#
#   * Closed box with reflecting walls on all 6 sides.
#   * Gravity g = (0, -1, 0) (dimensionless units).
#   * Background: isothermal hydrostatic atmosphere,
#       rho_bg(y) = exp(-y / H),  p_bg(y) = exp(-y / H),  H = 1,
#     satisfying dp/dy = -rho * g identically, so the hydrostatic
#     state is a stationary solution of Euler + gravity source.
#   * Bubble perturbation: a Gaussian drop in density (keeping pressure
#     the same) centred at (0.5, 0.2, Lz/2) with radius ~0.1.  The
#     bubble is therefore warmer (higher T = p / rho) than the ambient
#     stratification at the same altitude, so buoyancy drives it up.
#
# The "did it rise?" check: mass-weighted y of the low-density region
# at t = T_FINAL should be noticeably larger than at t = 0.  That's
# only possible if the gravity source term is being applied; without
# it the initial hydrostatic profile + bubble perturbation just
# oscillates around the starting state.
# ======================================================================

from std.sys import has_accelerator
from std.gpu import global_idx
from std.math import sqrt, ceildiv, exp

from src import mpi
from src.reference import N_P
from src.boundary import BoundaryConditions, BC_WALL
from src.solver import Solver
from src.euler import Euler, FLUX_HLLEC
from src.driver3d import Driver3D
from src.nvtx import NvtxContext
from src.frame_writer import FrameWriter, DownloadedSnapshot
from src.time_integrator import run_ssprk3_loop_with_diagnostics
from src.diagnostics import DiagnosticsWriter, DiagComponents


# Thin 3D box; the bubble physics is essentially 2D in the (x, y)
# plane and z is kept minimal just to keep the mesh 3D.
comptime NX = 32
comptime NY = 32
comptime NZ = 2
comptime LX = 1.0
comptime LY = 1.0
comptime LZ = Float64(2.0 / 32.0)  # dz matches dx = dy

comptime GAMMA: Float32 = 1.4
comptime MIN_DENSITY: Float32 = 1.0e-6
comptime MIN_PRESSURE: Float32 = 1.0e-6

# Gravity: -y direction, unit magnitude.  Scale height H = p0 / (rho0 * g)
# = 1 with rho0 = p0 = g = 1.
comptime GX: Float32 = 0.0
comptime GY: Float32 = -1.0
comptime GZ: Float32 = 0.0
comptime SCALE_H: Float32 = 1.0

# Bubble: Gaussian density deficit, keeping pressure matched to the
# ambient hydrostatic profile so the bubble is warmer and rises.
comptime BUBBLE_X0: Float32 = 0.5
comptime BUBBLE_Y0: Float32 = 0.2
comptime BUBBLE_RADIUS: Float32 = 0.1
comptime BUBBLE_RHO_FACTOR: Float32 = 0.8  # 20% density deficit at core

comptime T_FINAL: Float32 = 2.5
comptime NUM_FRAMES = 25

comptime CFL = Float32(0.2)
comptime IC_BLOCK = 256


def bubble_ic_kernel(
    q: UnsafePointer[Float32, MutAnyOrigin],
    owned_elem_ids: UnsafePointer[Int32, MutAnyOrigin],
    elem_node_xyz: UnsafePointer[Float32, MutAnyOrigin],
    num_owned: Int,
    gamma: Float32,
    H: Float32,
    x0: Float32,
    y0: Float32,
    z_mid: Float32,
    radius: Float32,
    rho_factor: Float32,
):
    var idx = Int(global_idx.x)
    var total = num_owned * N_P
    if idx >= total:
        return
    var i = idx // N_P
    var nn = idx % N_P
    var e = Int(owned_elem_ids[i])
    var px = elem_node_xyz[(e * N_P + nn) * 3 + 0]
    var py = elem_node_xyz[(e * N_P + nn) * 3 + 1]
    var pz = elem_node_xyz[(e * N_P + nn) * 3 + 2]

    # Hydrostatic background.
    var rho_bg = exp(-py / H)
    var p_bg = exp(-py / H)

    # Smooth Gaussian bubble mask in (x, y) with a mild z-dependence.
    var dx = px - x0
    var dy = py - y0
    var dz = pz - z_mid
    var r2 = dx * dx + dy * dy + dz * dz
    var bubble_mask = exp(-r2 / (radius * radius))

    # Keep pressure = hydrostatic; deplete density inside the bubble.
    var rho = rho_bg * (Float32(1.0) - (Float32(1.0) - rho_factor) * bubble_mask)
    var p = p_bg
    var E = p / (gamma - Float32(1.0))  # velocities are zero
    var base = (e * N_P + nn) * 5
    q[base + 0] = rho
    q[base + 1] = Float32(0.0)
    q[base + 2] = Float32(0.0)
    q[base + 3] = Float32(0.0)
    q[base + 4] = E


def choose_dt() raises -> Float32:
    # Max sound speed c_bg = sqrt(gamma * p / rho) = sqrt(gamma) at any
    # altitude in the isothermal atmosphere (p / rho = const).  Bubble
    # dynamics are much slower than acoustic speed, so c bounds wave.
    var h = Float32(LX) / Float32(NX)
    var c = sqrt(GAMMA)
    return CFL * h / (c * Float32(5.0))


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"

    # All walls.
    var bcs = BoundaryConditions(BC_WALL, BC_WALL, BC_WALL, BC_WALL, BC_WALL, BC_WALL)
    var physics = Euler(GAMMA, MIN_DENSITY, MIN_PRESSURE, FLUX_HLLEC, True, GX, GY, GZ)
    var d = Driver3D[Euler](
        problem_name="euler_rising_bubble: GPU DG Euler + gravity",
        nx=NX,
        ny=NY,
        nz=NZ,
        lx=LX,
        ly=LY,
        lz=LZ,
        bcs=bcs,
        physics=physics^,
    )

    d.solver.ctx.enqueue_function[bubble_ic_kernel](
        d.solver.d_q.unsafe_ptr(),
        d.solver.mesh.d_owned_elem_ids.unsafe_ptr(),
        d.solver.mesh.local.d_elem_node_xyz.unsafe_ptr(),
        d.solver.num_owned_elements,
        GAMMA,
        SCALE_H,
        BUBBLE_X0,
        BUBBLE_Y0,
        Float32(LZ) * Float32(0.5),
        BUBBLE_RADIUS,
        BUBBLE_RHO_FACTOR,
        grid_dim=ceildiv(d.solver.num_owned_elements * N_P, IC_BLOCK),
        block_dim=IC_BLOCK,
    )
    d.solver.ctx.synchronize()

    # IC-time centroid of the density deficit (where the bubble is).
    # At np>1 each rank computes the centroid of ITS owned subset,
    # which is not globally meaningful without an allreduce -- so the
    # diagnostic prints are suppressed outside np=1.
    if d.size == 1:
        var y_bubble_ic = _bubble_centroid_y(d.solver, d.nvtx)
        print("  bubble centroid y at t=0 :", y_bubble_ic, " (expected ~", BUBBLE_Y0, ")")

    # Pre-step perf snapshot: device memory accounting (rank 0 only).
    if d.rank == 0:
        d.solver.memory_report().print()

    var writer = FrameWriter[Euler](d.solver, d.nvtx, component=0)

    # Conserved-quantity time series for the animated dashboard.
    # Euler's state layout: q[0]=rho, q[1..3]=rho u_{x,y,z}, q[4]=E.
    # Integrating each component over the domain gives the domain-
    # integrated mass, 3 momentum components, and total energy -- all
    # conserved to the flux scheme's precision (walls on all sides
    # carry no net momentum or energy transport, gravity source shifts
    # y-momentum and energy as expected).
    var components = DiagComponents().linear("mass", 0).linear("momentum_x", 1).linear("momentum_y", 2).linear("momentum_z", 3).linear("total_energy", 4)
    var diag = DiagnosticsWriter[Euler](d.solver, "output/diagnostics.csv", components.linear_list, components.squared_list, components.maxabs_list, LX, LY, LZ)

    var dt = choose_dt()
    if d.rank == 0:
        print("  dt =", dt, " (", Int(T_FINAL / dt), " steps estimated)")

    var result = run_ssprk3_loop_with_diagnostics[Euler](d.solver, writer, diag, dt, T_FINAL, NUM_FRAMES, d.nvtx)

    writer.finalize("output/solution.pvd", d.nvtx)

    # Final-state multi-field snapshot (rho + p + |v|) for richer
    # ParaView inspection.  Independent of the per-frame async pipeline.
    if d.is_single_rank():
        var snap = DownloadedSnapshot[Euler](d.solver, d.nvtx, components=[0, 1, 2, 3, 4])
        var f_rho = snap.alloc_field()
        var f_p = snap.alloc_field()
        var f_vmag = snap.alloc_field()
        for k in range(snap.n_owned_dof):
            var rho = snap.snaps[0][k]
            var u = snap.snaps[1][k] / rho
            var v = snap.snaps[2][k] / rho
            var w = snap.snaps[3][k] / rho
            var ke = Float32(0.5) * rho * (u * u + v * v + w * w)
            var p = (GAMMA - Float32(1.0)) * (snap.snaps[4][k] - ke)
            f_rho[k] = Float64(rho)
            f_p[k] = Float64(p)
            f_vmag[k] = Float64(sqrt(u * u + v * v + w * w))
        snap.add_field("rho", f_rho^)
        snap.add_field("p", f_p^)
        snap.add_field("|v|", f_vmag^)
        snap.write(d.solver, d.nvtx, "output/snapshot_t_final.vtu")
        if d.rank == 0:
            print("  wrote output/snapshot_t_final.vtu (rho + p + |v|, t=", T_FINAL, ")")

    if d.size == 1:
        var y_bubble_final = _bubble_centroid_y(d.solver, d.nvtx)
        print("  bubble centroid y at t=", T_FINAL, ":", y_bubble_final)
    if d.rank == 0:
        result.print_summary()
        print("  wrote output/solution.pvd")
    # Post-run sync'd throughput measurement.
    var tput = d.solver.bench_step_loop(dt, d.nvtx)
    if d.rank == 0:
        tput.print()

    mpi.finalize()


# ----------------------------------------------------------------------
# Bubble-centroid diagnostic
# ----------------------------------------------------------------------
# Locates the weighted centroid of the density deficit relative to the
# ambient hydrostatic profile.  A rising bubble means this centroid
# moves in the +y direction over time -- the minimum "the gravity
# source is doing something" check for this driver.
# ----------------------------------------------------------------------


def _bubble_centroid_y(mut solver: Solver[Euler], mut nvtx: NvtxContext) raises -> Float32:
    var num_owned = solver.num_owned_elements
    var h_q = List[Float32]()
    for _ in range(num_owned * N_P):
        h_q.append(Float32(0.0))
    solver.download_owned_component(0, h_q, nvtx)
    var xyz_ptr = solver.mesh.owned_node_xyz_f32_ptr

    var weighted_y = Float32(0.0)
    var total_w = Float32(0.0)
    for i in range(num_owned):
        for nn in range(N_P):
            var py = xyz_ptr[(i * N_P + nn) * 3 + 1]
            var rho = h_q[i * N_P + nn]
            var rho_bg = exp(-py / SCALE_H)
            var deficit = rho_bg - rho
            # Only count positive deficits; noise below this floor
            # is interior oscillation, not the bubble itself.
            if deficit > Float32(1.0e-3):
                weighted_y += py * deficit
                total_w += deficit
    if total_w <= Float32(0.0):
        return Float32(-1.0)
    return weighted_y / total_w
