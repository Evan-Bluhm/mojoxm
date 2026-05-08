# ======================================================================
# euler_vortex -- isentropic Euler vortex, triply periodic
# ======================================================================
#
# Classic smooth-periodic test problem for the compressible Euler
# equations.  A 2D vortex in the xy plane, constant along z, superposed
# on a uniform background flow v_bg = (u_bg, v_bg, w_bg).  The exact
# solution is the vortex translated by v_bg; with periodic BCs and
# T_final chosen so v_bg * T_final is a lattice vector, the solution
# returns to the initial condition.
#
# Perturbations (Shu 1998 form):
#   du = -eps * (y - y0) * exp((1 - r^2)/2) / (2 pi)
#   dv = +eps * (x - x0) * exp((1 - r^2)/2) / (2 pi)
#   dw = 0
#   T  = 1 - (gamma - 1) eps^2 / (8 gamma pi^2) * exp(1 - r^2)
#   rho = T^(1/(gamma - 1))
#   p   = rho^gamma
#
# where r^2 = (x-x0)^2 + (y-y0)^2.  The vortex is isentropic so
# p/rho^gamma = const = 1.
#
# Single-compilation-unit DG driver using:
#   * `Euler` physics (5 components) with HLLEC numerical flux
#   * Kuhn-tet Cartesian mesh, triply periodic
#   * SSPRK3 time integration
#   * VTU output of the density field
#
# MPI: `mpirun -np N ./euler_vortex` partitions the domain across N
# ranks; np=1 runs single-rank with no halo exchange.
# ======================================================================

from std.sys import has_accelerator
from std.gpu import global_idx
from std.gpu.host import DeviceContext
from std.math import sqrt, ceildiv, exp, pow

from src import mpi
from src.partition import build_partition
from src.reference import N_P, build_reference_operators
from src.mesh import Mesh
from src.boundary import BoundaryConditions
from src.halo_exchange import HaloExchange
from src.solver import Solver
from src.euler import Euler, FLUX_HLLEC
from src.nvtx import NvtxContext
from src.frame_writer import FrameWriter, write_snapshot_3d_multi
from src.time_integrator import run_ssprk3_loop_with_diagnostics
from src.diagnostics import DiagnosticsWriter, NamedComponent
from src.memory_report import ThroughputReport

comptime NX = 32
comptime NY = 32
comptime NZ = 32
comptime LX = 10.0
comptime LY = 10.0
comptime LZ = 10.0

# Background velocity (unit translation of the vortex per t).
comptime UBG: Float32 = 1.0
comptime VBG: Float32 = 1.0
comptime WBG: Float32 = 0.0

# Vortex parameters.
comptime GAMMA: Float32 = 1.4
comptime VORTEX_EPS: Float32 = 5.0  # peak tangential speed
comptime VORTEX_X0: Float32 = 5.0
comptime VORTEX_Y0: Float32 = 5.0

# Numerical-flux choices.
comptime MIN_DENSITY: Float32 = 1.0e-6
comptime MIN_PRESSURE: Float32 = 1.0e-6

comptime T_FINAL: Float32 = 1.0  # one period for demonstration
comptime NUM_FRAMES = 10

# SSPRK3 safety factor for P2 DG on tets.
comptime CFL = Float32(0.2)

comptime IC_BLOCK = 256


# ----------------------------------------------------------------------
# Initial-condition kernel (isentropic vortex)
# ----------------------------------------------------------------------
# One thread per owned DOF.  Reads node coordinates for this rank's
# owned elements from the already-resident mesh buffer and writes the
# 5-component conserved state into q.  Layout: q[(e * N_P + nn) * 5 + c].
# ----------------------------------------------------------------------


def vortex_ic_kernel(
    q: UnsafePointer[Float32, MutAnyOrigin],
    owned_elem_ids: UnsafePointer[Int32, MutAnyOrigin],
    elem_node_xyz: UnsafePointer[Float32, MutAnyOrigin],
    num_owned: Int,
    x0: Float32,
    y0: Float32,
    ubg: Float32,
    vbg: Float32,
    wbg: Float32,
    gamma: Float32,
    eps: Float32,
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
    var dx = px - x0
    var dy = py - y0
    var r2 = dx * dx + dy * dy
    var expf = exp((Float32(1.0) - r2) * Float32(0.5))
    var eps_over_2pi = eps / (Float32(2.0) * Float32(3.14159265358979323846))
    var du = -eps_over_2pi * dy * expf
    var dv = eps_over_2pi * dx * expf
    var u = ubg + du
    var v = vbg + dv
    var w = wbg
    var g1 = gamma - Float32(1.0)
    # T = 1 - (gamma-1) eps^2 / (8 gamma pi^2) * exp(1 - r^2).  Rewritten
    # via eps_over_2pi (= eps/(2 pi)) so eps^2/(8 gamma pi^2) becomes
    # eps_over_2pi^2 / (2 gamma) -- one expression, formatter-clean.
    var T = Float32(1.0) - g1 * eps_over_2pi * eps_over_2pi / (Float32(2.0) * gamma) * exp(Float32(1.0) - r2)
    var rho = pow(T, Float32(1.0) / g1)
    var p = pow(rho, gamma)
    var E = p / g1 + Float32(0.5) * rho * (u * u + v * v + w * w)
    var base = (e * N_P + nn) * 5
    q[base + 0] = rho
    q[base + 1] = rho * u
    q[base + 2] = rho * v
    q[base + 3] = rho * w
    q[base + 4] = E


# Estimated dt bound based on the background + peak tangential speed
# plus the sound speed at rest (c ~ sqrt(gamma)).
def choose_dt() raises -> Float32:
    var h = Float32(LX) / Float32(NX)
    var v_bg_mag = sqrt(UBG * UBG + VBG * VBG + WBG * WBG)
    var v_peak = VORTEX_EPS / Float32(2.0 * 3.14159265358979323846)  # rough
    var c_ref = sqrt(GAMMA)
    var wave = v_bg_mag + v_peak + c_ref
    var dt_est = CFL * h / (wave * Float32(2 * 2 + 1))
    return dt_est


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    mpi.init()
    var rank = mpi.world_rank()
    var size = mpi.world_size()

    if rank == 0:
        print("euler_vortex: GPU DG Euler, P2 tet, HLLEC flux,", size, "rank(s)")
        print("  global mesh: ", NX, "x", NY, "x", NZ, " cells -> ", NX * NY * NZ * 6, "tets")
        print("  nodes per element:", N_P, " total DOF:", NX * NY * NZ * 6 * N_P)

    var nvtx = NvtxContext()
    if rank == 0:
        print("  NVTX:", "enabled" if nvtx.is_enabled() else "unavailable")

    var refs = build_reference_operators(nvtx)

    nvtx.push_range("device_context_create")
    var ctx = DeviceContext()
    nvtx.pop_range()

    nvtx.push_range("build_mesh")
    var mesh = Mesh(ctx, build_partition(rank, size, NX, NY, NZ), LX, LY, LZ, BoundaryConditions.periodic())
    nvtx.pop_range()

    nvtx.push_range("halo_setup")
    var halo = HaloExchange(ctx, mesh.part, Euler.NUM_COMPONENTS, mesh.d_perm.unsafe_ptr())
    nvtx.pop_range()

    if rank == 0:
        print("  proc-grid: ", mesh.part.px, "x", mesh.part.py, "x", mesh.part.pz, "  owned cubes per rank: ", mesh.part.nx, "x", mesh.part.ny, "x", mesh.part.nz)
        print("  per-rank: ", mesh.num_owned_elements, "owned elements (halo=", mesh.num_halo_elements, ", interior=", mesh.num_interior_elements, ")")

    var physics = Euler(GAMMA, MIN_DENSITY, MIN_PRESSURE, FLUX_HLLEC, True, Float32(0.0), Float32(0.0), Float32(0.0))

    nvtx.push_range("solver_setup")
    var solver = Solver[Euler](ctx^, mesh^, halo^, physics^, refs.D_ref^, refs.Lift_ref^, refs.node_weights^)
    nvtx.pop_range()

    nvtx.push_range("initial_condition")
    solver.ctx.enqueue_function[vortex_ic_kernel](
        solver.d_q.unsafe_ptr(),
        solver.mesh.d_owned_elem_ids.unsafe_ptr(),
        solver.mesh.local.d_elem_node_xyz.unsafe_ptr(),
        solver.num_owned_elements,
        VORTEX_X0,
        VORTEX_Y0,
        UBG,
        VBG,
        WBG,
        GAMMA,
        VORTEX_EPS,
        grid_dim=ceildiv(solver.num_owned_elements * N_P, IC_BLOCK),
        block_dim=IC_BLOCK,
    )
    solver.ctx.synchronize()
    nvtx.pop_range()

    # Pre-step perf snapshot: device memory accounting (rank 0 only).
    if rank == 0:
        solver.memory_report().print()

    # Frame output (density is component 0 of the 5-component Euler state).
    var writer = FrameWriter[Euler](solver, nvtx, component=0)

    # Diagnostics: the classical vortex is a smooth periodic solution,
    # so all 5 conserved integrals should be EXACTLY conserved.  Any
    # drift visible in the CSV = scheme-level dissipation plus
    # nodal-quadrature aliasing.
    var diag_linear = List[NamedComponent]()
    diag_linear.append(NamedComponent("mass", 0))
    diag_linear.append(NamedComponent("momentum_x", 1))
    diag_linear.append(NamedComponent("momentum_y", 2))
    diag_linear.append(NamedComponent("momentum_z", 3))
    diag_linear.append(NamedComponent("total_energy", 4))
    var diag = DiagnosticsWriter[Euler](solver, "output/diagnostics.csv", diag_linear, List[NamedComponent](), List[NamedComponent](), LX, LY, LZ)

    var dt = choose_dt()
    if rank == 0:
        print("  dt =", dt, " (", Int(T_FINAL / dt), " steps estimated)")

    var result = run_ssprk3_loop_with_diagnostics[Euler](solver, writer, diag, dt, T_FINAL, NUM_FRAMES, nvtx)

    writer.finalize("output/solution.pvd", nvtx)

    # Final-state multi-field snapshot (rho + p + |v|) for richer
    # ParaView inspection.  Independent of the per-frame async pipeline.
    # Gated on np=1 since each rank would dump only its owned slab.
    var nprocs = solver.mesh.part.px * solver.mesh.part.py * solver.mesh.part.pz
    if nprocs == 1:
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
        solver.download_owned_component(0, snap_rho, nvtx)
        solver.download_owned_component(1, snap_rhou, nvtx)
        solver.download_owned_component(2, snap_rhov, nvtx)
        solver.download_owned_component(3, snap_rhow, nvtx)
        solver.download_owned_component(4, snap_E, nvtx)
        var f_rho = List[Float64]()
        var f_p = List[Float64]()
        var f_vmag = List[Float64]()
        for k in range(n_owned_dof):
            var rho = snap_rho[k]
            var u = snap_rhou[k] / rho
            var v = snap_rhov[k] / rho
            var w = snap_rhow[k] / rho
            var ke = Float32(0.5) * rho * (u * u + v * v + w * w)
            var p = (GAMMA - Float32(1.0)) * (snap_E[k] - ke)
            f_rho.append(Float64(rho))
            f_p.append(Float64(p))
            f_vmag.append(Float64(sqrt(u * u + v * v + w * w)))
        var fields = List[List[Float64]]()
        fields.append(f_rho^)
        fields.append(f_p^)
        fields.append(f_vmag^)
        var names = List[String]()
        names.append(String("rho"))
        names.append(String("p"))
        names.append(String("|v|"))
        write_snapshot_3d_multi(solver=solver, field_names=names, field_data=fields, path=String("output/snapshot_t_final.vtu"), nvtx=nvtx)
        if rank == 0:
            print("  wrote output/snapshot_t_final.vtu (rho + p + |v|, t=", T_FINAL, ")")

    if rank == 0:
        result.print_summary()
        print("  wrote output/solution.pvd")
    # Post-run sync'd throughput measurement (5-step warmup + 50-step
    # measure).  Run AFTER finalize so we don't pollute the production
    # state mid-simulation; the bench loop intentionally advances q
    # past T_FINAL so the printed numbers reflect steady-state per-
    # step compute cost.
    var tput = solver.bench_step_loop(dt, nvtx)
    if rank == 0:
        tput.print()

    mpi.finalize()
