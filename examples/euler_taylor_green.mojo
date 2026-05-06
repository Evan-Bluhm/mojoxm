# ======================================================================
# euler_taylor_green -- compressible Taylor-Green vortex
# ======================================================================
#
# The canonical smooth test for high-order compressible solvers.  Two
# counter-rotating vortex sheets on a 2 pi cube evolve under the Euler
# equations, stretch, tangle, and cascade energy into smaller scales.
# The flow is fully smooth and triply periodic, so it works cleanly
# inside our current framework (no limiter, no wall BCs).
#
# Initial condition (constant-density form):
#   u   =  U0 * sin(x) cos(y) cos(z)
#   v   = -U0 * cos(x) sin(y) cos(z)
#   w   =  0
#   rho =  rho0
#   p   =  p0 + (rho0 U0^2 / 16) * (cos(2x) + cos(2y)) * (cos(2z) + 2)
#   E   =  p/(gamma-1) + 0.5 rho (u^2 + v^2 + w^2)
#
# Mach number is set by the ratio U0 / sqrt(gamma p0 / rho0).  At
# Ma ~ 0.3 the flow develops visible density oscillations (~10% of rho0)
# and remains shocklet-free for the first few eddy turnover times.
#
# Reference: Taylor & Green, Proc. Roy. Soc. A 158 (1937); and the
# "Taylor-Green Vortex" as popularised by Brachet et al. (1983).  Used
# as the AIAA high-order CFD workshop benchmark.
#
# MPI: `mpirun -np N ./euler_taylor_green` partitions the domain across
# N ranks; np=1 runs single-rank with no halo exchange.
# ======================================================================

from std.sys import has_accelerator
from std.gpu import global_idx
from std.gpu.host import DeviceContext
from std.math import sqrt, ceildiv, sin, cos

from src import mpi
from src.partition import build_partition
from src.reference import N_P, build_reference_operators
from src.mesh import Mesh
from src.boundary import BoundaryConditions
from src.halo_exchange import HaloExchange
from src.solver import Solver
from src.euler import Euler, FLUX_HLLEC
from src.nvtx import NvtxContext
from src.frame_writer import FrameWriter
from src.time_integrator import run_ssprk3_loop_with_diagnostics
from src.diagnostics import DiagnosticsWriter, NamedComponent
from src.vtu import dump_vtu_3d_frame_multi

# Domain is the natural 2 pi cube so sin/cos of coordinates are periodic
# without any wrap-around algebra.
comptime TWO_PI: Float32 = 6.283185307179586
comptime LX = 6.283185307179586
comptime LY = 6.283185307179586
comptime LZ = 6.283185307179586

comptime NX = 32
comptime NY = 32
comptime NZ = 32

# Flow parameters.  Ma = U0 / sqrt(gamma p0 / rho0) ~ 0.3.
comptime GAMMA: Float32 = 1.4
comptime RHO0:  Float32 = 1.0
comptime U0:    Float32 = 1.0
comptime P0:    Float32 = 7.9365079    # chosen so c0 = sqrt(1.4 * p0) ~ 3.33 -> Ma = 0.3

# Numerical-flux floors.
comptime MIN_DENSITY:  Float32 = 1.0e-6
comptime MIN_PRESSURE: Float32 = 1.0e-6

# Integration window.  ~3 eddy turnover times is enough to see the
# large counter-rotating cells stretch, merge, and begin to cascade.
comptime T_FINAL: Float32 = 2.0
comptime NUM_FRAMES = 80

# SSPRK3 safety factor for P2 DG on tets.
comptime CFL = Float32(0.2)

comptime IC_BLOCK = 256


# ----------------------------------------------------------------------
# Initial-condition kernel (Taylor-Green vortex)
# ----------------------------------------------------------------------
# One thread per owned DOF.  Reads node coordinates for this rank's
# owned elements from the already-resident mesh buffer and writes the
# 5-component conserved state into q.  Layout: q[(e * N_P + nn) * 5 + c].
# ----------------------------------------------------------------------

def taylor_green_ic_kernel(
    q: UnsafePointer[Float32, MutAnyOrigin],
    owned_elem_ids: UnsafePointer[Int32, MutAnyOrigin],
    elem_node_xyz:  UnsafePointer[Float32, MutAnyOrigin],
    num_owned: Int,
    u0: Float32, rho0: Float32, p0: Float32, gamma: Float32,
):
    var idx = Int(global_idx.x)
    var total = num_owned * N_P
    if idx >= total:
        return
    var i = idx // N_P
    var nn = idx % N_P
    var e = Int(owned_elem_ids[i])
    var x = elem_node_xyz[(e * N_P + nn) * 3 + 0]
    var y = elem_node_xyz[(e * N_P + nn) * 3 + 1]
    var z = elem_node_xyz[(e * N_P + nn) * 3 + 2]

    var sx = sin(x); var cx = cos(x)
    var sy = sin(y); var cy = cos(y)
    var cz = cos(z)
    var c2x = cos(Float32(2.0) * x)
    var c2y = cos(Float32(2.0) * y)
    var c2z = cos(Float32(2.0) * z)

    var u =  u0 * sx * cy * cz
    var v = -u0 * cx * sy * cz
    var w =  Float32(0.0)

    var rho = rho0
    var p = p0 + (rho0 * u0 * u0 / Float32(16.0)) * (c2x + c2y) * (c2z + Float32(2.0))
    var E = p / (gamma - Float32(1.0)) + Float32(0.5) * rho * (u * u + v * v + w * w)

    var base = (e * N_P + nn) * 5
    q[base + 0] = rho
    q[base + 1] = rho * u
    q[base + 2] = rho * v
    q[base + 3] = rho * w
    q[base + 4] = E


# Estimated dt bound.  Max wave speed is |U| + c; pick |U| <= U0 and
# c ~ c0 since density/pressure don't move much at Ma ~ 0.3.
def choose_dt() raises -> Float32:
    var h = Float32(LX) / Float32(NX)
    var c0 = sqrt(GAMMA * P0 / RHO0)
    var wave = U0 + c0
    var dt_est = CFL * h / (wave * Float32(2 * 2 + 1))
    return dt_est

def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    mpi.init()
    var rank = mpi.world_rank()
    var size = mpi.world_size()

    if rank == 0:
        print("euler_taylor_green: GPU DG Euler, P2 tet, HLLEC flux,",
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
        ctx, mesh.part, Euler.NUM_COMPONENTS,
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

    var physics = Euler(
        GAMMA, MIN_DENSITY, MIN_PRESSURE, FLUX_HLLEC, True,
        Float32(0.0), Float32(0.0), Float32(0.0),
    )

    nvtx.push_range("solver_setup")
    var solver = Solver[Euler](
        ctx^, mesh^, halo^, physics^, refs.D_ref^, refs.Lift_ref^, refs.node_weights^,
    )
    nvtx.pop_range()

    nvtx.push_range("initial_condition")
    solver.ctx.enqueue_function[taylor_green_ic_kernel, taylor_green_ic_kernel](
        solver.d_q.unsafe_ptr(),
        solver.mesh.d_owned_elem_ids.unsafe_ptr(),
        solver.mesh.local.d_elem_node_xyz.unsafe_ptr(),
        solver.num_owned_elements,
        U0, RHO0, P0, GAMMA,
        grid_dim=ceildiv(
            solver.num_owned_elements * N_P, IC_BLOCK
        ),
        block_dim=IC_BLOCK,
    )
    solver.ctx.synchronize()
    nvtx.pop_range()

    # Pre-step perf snapshot: device memory accounting (rank 0 only).
    if rank == 0:
        solver.memory_report().print()

    # Frame output (density is component 0 of the 5-component Euler state).
    var writer = FrameWriter[Euler](solver, nvtx, component=0)

    # Diagnostics: Taylor-Green develops into turbulence, so we track
    # both the 5 linear conserved integrals AND the momentum L2^2
    # components -- the sum (0.5 / rho) * int|rho u|^2 approximates
    # kinetic energy (the enstrophy cascade's observable).
    var diag_linear = List[NamedComponent]()
    diag_linear.append(NamedComponent("mass",         0))
    diag_linear.append(NamedComponent("momentum_x",   1))
    diag_linear.append(NamedComponent("momentum_y",   2))
    diag_linear.append(NamedComponent("momentum_z",   3))
    diag_linear.append(NamedComponent("total_energy", 4))
    var diag_squared = List[NamedComponent]()
    diag_squared.append(NamedComponent("momentum_sq_x", 1))
    diag_squared.append(NamedComponent("momentum_sq_y", 2))
    diag_squared.append(NamedComponent("momentum_sq_z", 3))
    var diag = DiagnosticsWriter[Euler](
        solver, "output/diagnostics.csv",
        diag_linear, diag_squared, List[NamedComponent](),
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

    if rank == 0:
        print("  final sync:", result.final_sync_sec, "s")
        print("  total steps:", result.total_steps,
              " wall time:", result.wall_sec, "s")
        print("    step-loop time (enqueue only, no sync):",
              result.step_loop_sec, "s")
        print("    frame-write time (download + VTU):",
              result.frame_write_sec, "s")
        print("  wrote output/solution.pvd")
    # Post-run sync'd throughput measurement (5-step warmup + 50-step
    # measure).  Run AFTER finalize so we don't pollute the production
    # state mid-simulation.
    var tput = solver.bench_step_loop(dt, nvtx)
    if rank == 0:
        tput.print()

    mpi.finalize()
