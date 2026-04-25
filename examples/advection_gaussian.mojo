# ======================================================================
# advection_gaussian -- 3D periodic advection of a Gaussian pulse
# ======================================================================
#
# Single-compilation-unit DG driver using:
#   * `Advection` physics (scalar, constant velocity)
#   * Kuhn-tet Cartesian mesh, triply periodic
#   * Gaussian initial condition, integrated over one period so the
#     exact solution returns to the IC.
#
# Hard-coded setup:
#   domain     : [0, 1]^3
#   mesh       : NX x NY x NZ Cartesian cells, each split into 6 tets
#   velocity   : v = (1, 1, 1)
#   time       : integrate 0 -> 1
#   frames     : NUM_FRAMES evenly-spaced checkpoint VTU files
#
# Output:
#   output/frame_00000.vtu ... output/frame_NNNNN.vtu
#   output/solution.pvd   (ParaView collection file)
#
# MPI: runs at any rank count.  At np=1 the Mesh has no ghost ring and
# no halo exchange runs; at np>1 each rank owns a sub-box plus a 1-cube
# ghost ring, and the halo is refreshed once per RK stage.  The driver
# is identical in both cases -- just `mpirun -np N ./advection_gaussian`.
# ======================================================================

from std.sys import has_accelerator
from std.gpu import global_idx
from std.gpu.host import DeviceContext
from std.math import sqrt, ceildiv, exp

from src import mpi
from src.partition import build_partition
from src.reference import N_P, build_reference_operators
from src.mesh import Mesh
from src.boundary import BoundaryConditions
from src.halo_exchange import HaloExchange
from src.solver import Solver
from src.advection import Advection
from src.nvtx import NvtxContext
from src.frame_writer import FrameWriter
from src.time_integrator import run_ssprk3_loop_with_diagnostics
from src.diagnostics import DiagnosticsWriter, NamedComponent

comptime NX = 48
comptime NY = 48
comptime NZ = 48
comptime LX = 1.0
comptime LY = 1.0
comptime LZ = 1.0

comptime VX: Float32 = 1.0
comptime VY: Float32 = 1.0
comptime VZ: Float32 = 1.0

comptime T_FINAL: Float32 = 1.0
comptime NUM_FRAMES = 20

# CFL safety factor (explicit P2 DG on tet typically requires CFL ~ 0.1
# times the element-scale / wave-speed).
comptime CFL = Float32(0.2)

# Gaussian pulse parameters -- evaluated by `gaussian_ic_kernel` directly
# on the device at every DG node position.
comptime GAUSS_CX: Float32 = 0.5
comptime GAUSS_CY: Float32 = 0.5
comptime GAUSS_CZ: Float32 = 0.5
comptime GAUSS_SIGMA: Float32 = 0.12

comptime IC_BLOCK = 256


# ----------------------------------------------------------------------
# Initial-condition kernel (periodic Gaussian, single-component)
# ----------------------------------------------------------------------
# Evaluates
#   q(x, y, z) = exp( - (dx^2 + dy^2 + dz^2) / (2 sigma^2) )
# where (dx, dy, dz) is the nearest-image displacement from the pulse
# center (cx, cy, cz) under periodic identification with box size
# (Lx, Ly, Lz).  Threads cover owned DOFs only -- ghost slots are
# populated via halo exchange at the top of each RK stage.
# ----------------------------------------------------------------------

def gaussian_ic_kernel(
    q: UnsafePointer[Float32, MutAnyOrigin],
    owned_elem_ids: UnsafePointer[Int32, MutAnyOrigin],
    elem_node_xyz:  UnsafePointer[Float32, MutAnyOrigin],
    num_owned: Int,
    cx: Float32, cy: Float32, cz: Float32,
    Lx: Float32, Ly: Float32, Lz: Float32,
    inv_two_sigma2: Float32,
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
    var dx = px - cx
    if dx >  Lx * Float32(0.5): dx -= Lx
    if dx < -Lx * Float32(0.5): dx += Lx
    var dy = py - cy
    if dy >  Ly * Float32(0.5): dy -= Ly
    if dy < -Ly * Float32(0.5): dy += Ly
    var dz = pz - cz
    if dz >  Lz * Float32(0.5): dz -= Lz
    if dz < -Lz * Float32(0.5): dz += Lz
    q[e * N_P + nn] = exp(
        -(dx * dx + dy * dy + dz * dz) * inv_two_sigma2
    )


# Characteristic element length: ~ cube_side / NX = 1/NX.  Further
# shrink by 1/p^2 for P2 DG stability (explicit RK).  SSPRK3 cfl_max
# ~ 1 on a "standard" DG basis; we include CFL factor for safety.
def choose_dt() raises -> Float32:
    var h = Float32(LX) / Float32(NX)
    var v = sqrt(VX * VX + VY * VY + VZ * VZ)
    # P2 tet empirical CFL factor ~ 1 / (2 p + 1)
    var dt_est = CFL * h / (v * Float32(2 * 2 + 1))
    return dt_est

def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    mpi.init()
    var rank = mpi.world_rank()
    var size = mpi.world_size()

    if rank == 0:
        print("advection_gaussian: GPU DG advection, P2 tet,",
              size, "rank(s)")
        print("  global mesh: ", NX, "x", NY, "x", NZ,
              " cells -> ", NX * NY * NZ * 6, "tets")
        print("  nodes per element:", N_P, " total DOF:",
              NX * NY * NZ * 6 * N_P)

    var nvtx = NvtxContext()
    if rank == 0:
        print("  NVTX:", "enabled" if nvtx.is_enabled() else "unavailable")

    var refs = build_reference_operators(nvtx)

    nvtx.push_range("device_context_create")
    var ctx = DeviceContext()
    nvtx.pop_range()

    nvtx.push_range("build_mesh")
    var mesh = Mesh(
        ctx, build_partition(rank, size, NX, NY, NZ), LX, LY, LZ,
        BoundaryConditions.periodic(),
    )
    nvtx.pop_range()

    nvtx.push_range("halo_setup")
    var halo = HaloExchange(
        ctx, mesh.part, Advection.NUM_COMPONENTS,
        mesh.d_perm.unsafe_ptr(),
    )
    nvtx.pop_range()

    if rank == 0:
        print("  proc-grid: ",
              mesh.part.px, "x", mesh.part.py, "x", mesh.part.pz,
              "  owned cubes per rank: ",
              mesh.part.nx, "x", mesh.part.ny, "x", mesh.part.nz)
        print("  per-rank: ", mesh.num_owned_elements,
              "owned elements (halo=", mesh.num_halo_elements,
              ", interior=", mesh.num_interior_elements, ")")

    var physics = Advection(VX, VY, VZ)

    nvtx.push_range("solver_setup")
    var solver = Solver[Advection](
        ctx^, mesh^, halo^, physics^, refs.D_ref^, refs.Lift_ref^, refs.node_weights^,
    )
    nvtx.pop_range()

    # Initial condition on owned elements.
    nvtx.push_range("initial_condition")
    var inv_two_sigma2 = Float32(1.0) / (
        Float32(2.0) * GAUSS_SIGMA * GAUSS_SIGMA
    )
    solver.ctx.enqueue_function[gaussian_ic_kernel, gaussian_ic_kernel](
        solver.d_q.unsafe_ptr(),
        solver.mesh.d_owned_elem_ids.unsafe_ptr(),
        solver.mesh.local.d_elem_node_xyz.unsafe_ptr(),
        solver.num_owned_elements,
        Float32(GAUSS_CX), Float32(GAUSS_CY), Float32(GAUSS_CZ),
        Float32(LX), Float32(LY), Float32(LZ),
        inv_two_sigma2,
        grid_dim=ceildiv(
            solver.num_owned_elements * N_P, IC_BLOCK
        ),
        block_dim=IC_BLOCK,
    )
    solver.ctx.synchronize()
    nvtx.pop_range()

    # Per-rank VTU output (density == full scalar solution for Advection).
    var writer = FrameWriter[Advection](solver, nvtx)

    # Diagnostics.  For a scalar conservation law the only conserved
    # integral is int(q) dV ("mass"); tracking int(q^2) dV (L2 norm
    # squared) gives a direct read-out of Rusanov dissipation, which
    # should decay slowly from the initial Gaussian's analytic value.
    var diag_comps = List[NamedComponent]()
    diag_comps.append(NamedComponent("mass", 0))
    var diag_squared = List[NamedComponent]()
    diag_squared.append(NamedComponent("l2_squared", 0))
    var diag = DiagnosticsWriter[Advection](
        solver, "output/diagnostics.csv",
        diag_comps, diag_squared, List[NamedComponent](),
        LX, LY, LZ,
    )

    var dt = choose_dt()
    if rank == 0:
        print("  dt =", dt, " (", Int(T_FINAL / dt), " steps estimated)")

    var result = run_ssprk3_loop_with_diagnostics[Advection](
        solver, writer, diag, dt, T_FINAL, NUM_FRAMES, nvtx,
    )

    writer.finalize("output/solution.pvd", nvtx)

    if rank == 0:
        print("  final sync:", result.final_sync_sec, "s")
        print("  total steps:", result.total_steps,
              " wall time:", result.wall_sec, "s")
        print("    step-loop time (enqueue only, no sync):",
              result.step_loop_sec, "s")
        print("    frame-write time (download + VTU):",
              result.frame_write_sec, "s")
        print("  wrote output/solution.pvd")

    mpi.finalize()
