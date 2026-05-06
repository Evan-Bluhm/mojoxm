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
from std.gpu.host import DeviceContext
from std.math import sqrt, ceildiv, exp

from src import mpi
from src.partition import build_partition
from src.reference import N_P, build_reference_operators
from src.mesh import Mesh
from src.boundary import BoundaryConditions, BC_WALL
from src.halo_exchange import HaloExchange
from src.solver import Solver
from src.euler import Euler, FLUX_HLLEC
from src.nvtx import NvtxContext
from src.frame_writer import FrameWriter
from src.time_integrator import run_ssprk3_loop_with_diagnostics
from src.diagnostics import DiagnosticsWriter, NamedComponent
from src.vtu import dump_vtu_3d_frame_multi


# Thin 3D box; the bubble physics is essentially 2D in the (x, y)
# plane and z is kept minimal just to keep the mesh 3D.
comptime NX = 32
comptime NY = 32
comptime NZ = 2
comptime LX = 1.0
comptime LY = 1.0
comptime LZ = Float64(2.0 / 32.0)   # dz matches dx = dy

comptime GAMMA: Float32 = 1.4
comptime MIN_DENSITY:  Float32 = 1.0e-6
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
comptime BUBBLE_RHO_FACTOR: Float32 = 0.8   # 20% density deficit at core

comptime T_FINAL: Float32 = 2.5
comptime NUM_FRAMES = 25

comptime CFL = Float32(0.2)
comptime IC_BLOCK = 256


def bubble_ic_kernel(
    q: UnsafePointer[Float32, MutAnyOrigin],
    owned_elem_ids: UnsafePointer[Int32, MutAnyOrigin],
    elem_node_xyz:  UnsafePointer[Float32, MutAnyOrigin],
    num_owned: Int,
    gamma: Float32,
    H: Float32,
    x0: Float32, y0: Float32, z_mid: Float32,
    radius: Float32, rho_factor: Float32,
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
    var p_bg   = exp(-py / H)

    # Smooth Gaussian bubble mask in (x, y) with a mild z-dependence.
    var dx = px - x0
    var dy = py - y0
    var dz = pz - z_mid
    var r2 = dx * dx + dy * dy + dz * dz
    var bubble_mask = exp(-r2 / (radius * radius))

    # Keep pressure = hydrostatic; deplete density inside the bubble.
    var rho = rho_bg * (Float32(1.0) - (Float32(1.0) - rho_factor) * bubble_mask)
    var p   = p_bg
    var E   = p / (gamma - Float32(1.0))   # velocities are zero
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
    mpi.init()
    var rank = mpi.world_rank()
    var size = mpi.world_size()

    if rank == 0:
        print(
            "euler_rising_bubble: GPU DG Euler + gravity,",
            size, "rank(s)",
        )
        print("  global mesh: ", NX, "x", NY, "x", NZ,
              " cells -> ", NX * NY * NZ * 6, "tets")

    var nvtx = NvtxContext()

    var refs = build_reference_operators(nvtx)

    var ctx = DeviceContext()

    # All walls.
    var bcs = BoundaryConditions(
        BC_WALL, BC_WALL,   # -x, +x
        BC_WALL, BC_WALL,   # -y, +y
        BC_WALL, BC_WALL,   # -z, +z
    )

    var mesh = Mesh(
        ctx, build_partition(rank, size, NX, NY, NZ), LX, LY, LZ, bcs,
    )

    var halo = HaloExchange(
        ctx, mesh.part, Euler.NUM_COMPONENTS,
        mesh.d_perm.unsafe_ptr(), bcs,
    )

    var physics = Euler(
        GAMMA, MIN_DENSITY, MIN_PRESSURE, FLUX_HLLEC, True,
        GX, GY, GZ,
    )

    var solver = Solver[Euler](
        ctx^, mesh^, halo^, physics^, refs.D_ref^, refs.Lift_ref^, refs.node_weights^,
    )

    solver.ctx.enqueue_function[bubble_ic_kernel, bubble_ic_kernel](
        solver.d_q.unsafe_ptr(),
        solver.mesh.d_owned_elem_ids.unsafe_ptr(),
        solver.mesh.local.d_elem_node_xyz.unsafe_ptr(),
        solver.num_owned_elements,
        GAMMA, SCALE_H,
        BUBBLE_X0, BUBBLE_Y0, Float32(LZ) * Float32(0.5),
        BUBBLE_RADIUS, BUBBLE_RHO_FACTOR,
        grid_dim=ceildiv(
            solver.num_owned_elements * N_P, IC_BLOCK
        ),
        block_dim=IC_BLOCK,
    )
    solver.ctx.synchronize()

    # IC-time centroid of the density deficit (where the bubble is).
    # At np>1 each rank computes the centroid of ITS owned subset,
    # which is not globally meaningful without an allreduce -- so the
    # diagnostic prints are suppressed outside np=1.
    if size == 1:
        var y_bubble_ic = _bubble_centroid_y(solver, nvtx)
        print("  bubble centroid y at t=0 :", y_bubble_ic,
              " (expected ~", BUBBLE_Y0, ")")

    # Pre-step perf snapshot: device memory accounting (rank 0 only).
    if rank == 0:
        solver.memory_report().print()

    var writer = FrameWriter[Euler](solver, nvtx, component=0)

    # Conserved-quantity time series for the animated dashboard.
    # Euler's state layout: q[0]=rho, q[1..3]=rho u_{x,y,z}, q[4]=E.
    # Integrating each component over the domain gives the domain-
    # integrated mass, 3 momentum components, and total energy -- all
    # conserved to the flux scheme's precision (walls on all sides
    # carry no net momentum or energy transport, gravity source shifts
    # y-momentum and energy as expected).
    var diag_linear = List[NamedComponent]()
    diag_linear.append(NamedComponent("mass",         0))
    diag_linear.append(NamedComponent("momentum_x",   1))
    diag_linear.append(NamedComponent("momentum_y",   2))
    diag_linear.append(NamedComponent("momentum_z",   3))
    diag_linear.append(NamedComponent("total_energy", 4))
    var diag = DiagnosticsWriter[Euler](
        solver, "output/diagnostics.csv",
        diag_linear, List[NamedComponent](), List[NamedComponent](),
        LX, LY, LZ,
    )

    var dt = choose_dt()
    if rank == 0:
        print("  dt =", dt, " (", Int(T_FINAL / dt), " steps estimated)")

    var result = run_ssprk3_loop_with_diagnostics[Euler](
        solver, writer, diag, dt, T_FINAL, NUM_FRAMES, nvtx,
    )

    writer.finalize("output/solution.pvd", nvtx)

    # Final-state multi-field snapshot (rho + p + |v|) for richer
    # ParaView inspection.  Independent of the per-frame async pipeline.
    var nprocs = solver.mesh.part.px * solver.mesh.part.py * solver.mesh.part.pz
    if nprocs == 1:
        nvtx.push_range("snapshot_t_final")
        var n_owned_dof = solver.num_owned_elements * N_P
        var snap_rho = List[Float32]()
        var snap_rhou = List[Float32]()
        var snap_rhov = List[Float32]()
        var snap_rhow = List[Float32]()
        var snap_E = List[Float32]()
        for _ in range(n_owned_dof):
            snap_rho.append(Float32(0.0))
            snap_rhou.append(Float32(0.0))
            snap_rhov.append(Float32(0.0))
            snap_rhow.append(Float32(0.0))
            snap_E.append(Float32(0.0))
        solver.download_owned_component(0, snap_rho,  nvtx)
        solver.download_owned_component(1, snap_rhou, nvtx)
        solver.download_owned_component(2, snap_rhov, nvtx)
        solver.download_owned_component(3, snap_rhow, nvtx)
        solver.download_owned_component(4, snap_E,    nvtx)
        var f_rho  = List[Float64]()
        var f_p    = List[Float64]()
        var f_vmag = List[Float64]()
        for k in range(n_owned_dof):
            var rho = snap_rho[k]
            var u = snap_rhou[k] / rho
            var v = snap_rhov[k] / rho
            var w = snap_rhow[k] / rho
            var ke = Float32(0.5) * rho * (u*u + v*v + w*w)
            var p = (GAMMA - Float32(1.0)) * (snap_E[k] - ke)
            f_rho.append(Float64(rho))
            f_p.append(Float64(p))
            f_vmag.append(Float64(sqrt(u*u + v*v + w*w)))
        var fields = List[List[Float64]]()
        fields.append(f_rho^)
        fields.append(f_p^)
        fields.append(f_vmag^)
        var names = List[String]()
        names.append(String("rho"))
        names.append(String("p"))
        names.append(String("|v|"))
        dump_vtu_3d_frame_multi(
            num_elements=solver.num_owned_elements,
            nodes_per_elem=N_P,
            elem_node_xyz=rebind[UnsafePointer[Float32, MutAnyOrigin]](
                solver.mesh.owned_node_xyz_f32_ptr
            ),
            field_names=names,
            field_data=fields,
            path=String("output/snapshot_t_final.vtu"),
        )
        nvtx.pop_range()
        if rank == 0:
            print("  wrote output/snapshot_t_final.vtu (rho + p + |v|, t=", T_FINAL, ")")

    if size == 1:
        var y_bubble_final = _bubble_centroid_y(solver, nvtx)
        print("  bubble centroid y at t=", T_FINAL, ":", y_bubble_final)
    if rank == 0:
        print("  total steps:", result.total_steps,
              " wall time:", result.wall_sec, "s")
        print("  wrote output/solution.pvd")
    # Post-run sync'd throughput measurement.
    var tput = solver.bench_step_loop(dt, nvtx)
    if rank == 0:
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

def _bubble_centroid_y(
    mut solver: Solver[Euler], mut nvtx: NvtxContext,
) raises -> Float32:
    var num_owned = solver.num_owned_elements
    var h_q = List[Float32]()
    for _ in range(num_owned * N_P):
        h_q.append(Float32(0.0))
    solver.download_owned_component(0, h_q, nvtx)
    var xyz_ptr = solver.mesh.owned_node_xyz_f32_ptr

    var weighted_y = Float32(0.0)
    var total_w    = Float32(0.0)
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
                total_w    += deficit
    if total_w <= Float32(0.0):
        return Float32(-1.0)
    return weighted_y / total_w
