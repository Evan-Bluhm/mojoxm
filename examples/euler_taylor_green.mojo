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
from std.math import sqrt, ceildiv, sin, cos

from src import mpi
from src.reference import N_P
from src.boundary import BoundaryConditions
from src.solver import Solver
from src.euler import Euler, FLUX_HLLEC
from src.driver3d import Driver3D
from src.frame_writer import FrameWriter, DownloadedSnapshot
from src.time_integrator import run_ssprk3_loop_with_diagnostics
from src.diagnostics import DiagnosticsWriter, DiagComponents

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
comptime RHO0: Float32 = 1.0
comptime U0: Float32 = 1.0
comptime P0: Float32 = 7.9365079  # chosen so c0 = sqrt(1.4 * p0) ~ 3.33 -> Ma = 0.3

# Numerical-flux floors.
comptime MIN_DENSITY: Float32 = 1.0e-6
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
    elem_node_xyz: UnsafePointer[Float32, MutAnyOrigin],
    num_owned: Int,
    u0: Float32,
    rho0: Float32,
    p0: Float32,
    gamma: Float32,
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

    var sx = sin(x)
    var cx = cos(x)
    var sy = sin(y)
    var cy = cos(y)
    var cz = cos(z)
    var c2x = cos(Float32(2.0) * x)
    var c2y = cos(Float32(2.0) * y)
    var c2z = cos(Float32(2.0) * z)

    var u = u0 * sx * cy * cz
    var v = -u0 * cx * sy * cz
    var w = Float32(0.0)

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

    var physics = Euler(GAMMA, MIN_DENSITY, MIN_PRESSURE, FLUX_HLLEC, True, Float32(0.0), Float32(0.0), Float32(0.0))
    var d = Driver3D[Euler](
        problem_name="euler_taylor_green: GPU DG Euler, P2 tet, HLLEC flux",
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
    d.solver.ctx.enqueue_function[taylor_green_ic_kernel](
        d.solver.d_q.unsafe_ptr(),
        d.solver.mesh.d_owned_elem_ids.unsafe_ptr(),
        d.solver.mesh.local.d_elem_node_xyz.unsafe_ptr(),
        d.solver.num_owned_elements,
        U0,
        RHO0,
        P0,
        GAMMA,
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

    # Diagnostics: Taylor-Green develops into turbulence, so we track
    # both the 5 linear conserved integrals AND the momentum L2^2
    # components -- the sum (0.5 / rho) * int|rho u|^2 approximates
    # kinetic energy (the enstrophy cascade's observable).
    var components = (
        DiagComponents()
        .linear("mass", 0)
        .linear("momentum_x", 1)
        .linear("momentum_y", 2)
        .linear("momentum_z", 3)
        .linear("total_energy", 4)
        .squared("momentum_sq_x", 1)
        .squared("momentum_sq_y", 2)
        .squared("momentum_sq_z", 3)
    )
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

    if d.rank == 0:
        result.print_summary()
        print("  wrote output/solution.pvd")
    # Post-run sync'd throughput measurement (5-step warmup + 50-step
    # measure).  Run AFTER finalize so we don't pollute the production
    # state mid-simulation.
    var tput = d.solver.bench_step_loop(dt, d.nvtx)
    if d.rank == 0:
        tput.print()

    mpi.finalize()
