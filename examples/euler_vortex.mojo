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
from std.math import sqrt, ceildiv, exp, pow

from src import mpi
from src.reference import N_P
from src.boundary import BoundaryConditions
from src.solver import Solver
from src.euler import Euler, FLUX_HLLEC
from src.driver3d import Driver3D
from src.frame_writer import FrameWriter, DownloadedSnapshot
from src.time_integrator import run_ssprk3_loop_with_diagnostics
from src.diagnostics import DiagnosticsWriter, DiagComponents

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

    var physics = Euler(GAMMA, MIN_DENSITY, MIN_PRESSURE, FLUX_HLLEC, True, Float32(0.0), Float32(0.0), Float32(0.0))
    var d = Driver3D[Euler](
        problem_name="euler_vortex: GPU DG Euler, P2 tet, HLLEC flux",
        nx=NX,
        ny=NY,
        nz=NZ,
        lx=LX,
        ly=LY,
        lz=LZ,
        bcs=BoundaryConditions.periodic(),
        physics=physics^,
    )

    d.nvtx.push_range("initial_condition")
    d.solver.ctx.enqueue_function[vortex_ic_kernel](
        d.solver.d_q.unsafe_ptr(),
        d.solver.mesh.d_owned_elem_ids.unsafe_ptr(),
        d.solver.mesh.local.d_elem_node_xyz.unsafe_ptr(),
        d.solver.num_owned_elements,
        VORTEX_X0,
        VORTEX_Y0,
        UBG,
        VBG,
        WBG,
        GAMMA,
        VORTEX_EPS,
        grid_dim=ceildiv(d.solver.num_owned_elements * N_P, IC_BLOCK),
        block_dim=IC_BLOCK,
    )
    d.solver.ctx.synchronize()
    d.nvtx.pop_range()

    # Pre-step perf snapshot: device memory accounting (rank 0 only).
    if d.rank == 0:
        d.solver.memory_report().print()

    # Frame output (density is component 0 of the 5-component Euler state).
    var writer = FrameWriter[Euler](d.solver, d.nvtx, component=0)

    # Diagnostics: the classical vortex is a smooth periodic solution,
    # so all 5 conserved integrals should be EXACTLY conserved.  Any
    # drift visible in the CSV = scheme-level dissipation plus
    # nodal-quadrature aliasing.
    var components = DiagComponents().linear("mass", 0).linear("momentum_x", 1).linear("momentum_y", 2).linear("momentum_z", 3).linear("total_energy", 4)
    var diag = DiagnosticsWriter[Euler](d.solver, "output/diagnostics.csv", components.linear_list, components.squared_list, components.maxabs_list, LX, LY, LZ)

    var dt = choose_dt()
    if d.rank == 0:
        print("  dt =", dt, " (", Int(T_FINAL / dt), " steps estimated)")

    var result = run_ssprk3_loop_with_diagnostics[Euler](d.solver, writer, diag, dt, T_FINAL, NUM_FRAMES, d.nvtx)

    writer.finalize("output/solution.pvd", d.nvtx)

    # Final-state multi-field snapshot (rho + p + |v|) for richer
    # ParaView inspection.  Independent of the per-frame async pipeline.
    # Gated on np=1 since each rank would dump only its owned slab.
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

    if d.rank == 0:
        result.print_summary()
        print("  wrote output/solution.pvd")
    # Post-run sync'd throughput measurement (5-step warmup + 50-step
    # measure).  Run AFTER finalize so we don't pollute the production
    # state mid-simulation; the bench loop intentionally advances q
    # past T_FINAL so the printed numbers reflect steady-state per-
    # step compute cost.
    var tput = d.solver.bench_step_loop(dt, d.nvtx)
    if d.rank == 0:
        tput.print()

    mpi.finalize()
