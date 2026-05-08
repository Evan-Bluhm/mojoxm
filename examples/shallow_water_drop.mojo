# ======================================================================
# shallow_water_drop -- Gaussian water drop spreading in a closed basin
# ======================================================================
#
# 2D shallow-water test:  at t = 0 the free surface has a Gaussian
# bump at the domain centre; the surface height flattens over time as
# gravity drives surface waves outward, reflecting off the four
# lateral slip walls.  Total mass (integrated h) is exactly conserved
# by the DG scheme under Rusanov + wall BCs, which this driver checks.
#
#   domain: [0, 1]^2 in xy; thin in z (z periodic so it's effectively 2D)
#   IC:     h(x,y,0) = h0 + a * exp(-r^2 / sigma^2)
#                      with r^2 = (x - 0.5)^2 + (y - 0.5)^2
#           hu = hv = 0
#   BC:     slip walls on +/- x and +/- y; periodic in z
#   g:      1
#
# Demonstrates the ShallowWater physics module + the non-periodic BC
# infrastructure operating on a 3-component conservation law.
# ======================================================================

from std.sys import has_accelerator
from std.gpu import global_idx
from std.math import sqrt, ceildiv, exp

from src import mpi
from src.reference import N_P
from src.boundary import BoundaryConditions, BC_INTERIOR, BC_WALL
from src.solver import Solver
from src.shallow_water import ShallowWater
from src.driver3d import Driver3D
from src.nvtx import NvtxContext
from src.frame_writer import FrameWriter, DownloadedSnapshot
from src.time_integrator import run_ssprk3_loop_with_diagnostics
from src.diagnostics import DiagnosticsWriter, DiagComponents


comptime NX = 32
comptime NY = 32
comptime NZ = 2
comptime LX = 1.0
comptime LY = 1.0
comptime LZ = Float64(2.0 / 32.0)  # dz matches dx = dy

comptime GRAVITY: Float32 = 1.0
comptime H_MIN: Float32 = 1.0e-4

comptime H_REST: Float32 = 1.0  # still-water depth
comptime DROP_AMPLITUDE: Float32 = 0.2
comptime DROP_SIGMA: Float32 = 0.07
comptime DROP_X0: Float32 = 0.5
comptime DROP_Y0: Float32 = 0.5

comptime T_FINAL: Float32 = 1.0
comptime NUM_FRAMES = 20

# SSPRK3 safety factor for P2 DG on tets (same as every other driver).
comptime CFL = Float32(0.2)
comptime IC_BLOCK = 256


def drop_ic_kernel(
    q: UnsafePointer[Float32, MutAnyOrigin],
    owned_elem_ids: UnsafePointer[Int32, MutAnyOrigin],
    elem_node_xyz: UnsafePointer[Float32, MutAnyOrigin],
    num_owned: Int,
    h0: Float32,
    amp: Float32,
    sigma: Float32,
    x0: Float32,
    y0: Float32,
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
    var bump = amp * exp(-r2 / (sigma * sigma))
    var base = (e * N_P + nn) * 3
    q[base + 0] = h0 + bump
    q[base + 1] = Float32(0.0)
    q[base + 2] = Float32(0.0)


def choose_dt() raises -> Float32:
    var h = Float32(LX) / Float32(NX)
    # Peak wave speed sqrt(g * (h0 + amp)) right at the drop centre.
    var c_peak = sqrt(GRAVITY * (H_REST + DROP_AMPLITUDE))
    return CFL * h / (c_peak * Float32(5.0))


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"

    # Slip walls on x and y; z is periodic (z-direction is trivial for
    # pure 2D shallow water).
    var bcs = BoundaryConditions(BC_WALL, BC_WALL, BC_WALL, BC_WALL, BC_INTERIOR, BC_INTERIOR)
    var physics = ShallowWater(GRAVITY, H_MIN)
    var d = Driver3D[ShallowWater](
        problem_name="shallow_water_drop: GPU DG shallow water, P2 tet, Rusanov",
        nx=NX,
        ny=NY,
        nz=NZ,
        lx=LX,
        ly=LY,
        lz=LZ,
        bcs=bcs,
        physics=physics^,
    )

    d.solver.ctx.enqueue_function[drop_ic_kernel](
        d.solver.d_q.unsafe_ptr(),
        d.solver.mesh.d_owned_elem_ids.unsafe_ptr(),
        d.solver.mesh.local.d_elem_node_xyz.unsafe_ptr(),
        d.solver.num_owned_elements,
        H_REST,
        DROP_AMPLITUDE,
        DROP_SIGMA,
        DROP_X0,
        DROP_Y0,
        grid_dim=ceildiv(d.solver.num_owned_elements * N_P, IC_BLOCK),
        block_dim=IC_BLOCK,
    )
    d.solver.ctx.synchronize()

    # Mass diagnostic sums over this rank's owned elements only; at
    # np>1 the correct global number would need an allreduce, which
    # isn't worth adding here -- suppress outside np=1 rather than
    # print misleading partial sums.
    var mass_ic = Float32(0.0)
    if d.size == 1:
        mass_ic = _total_mass(d.solver, d.nvtx)
        print("  integrated mass at t=0     :", mass_ic)

    # Pre-step perf snapshot: device memory accounting (rank 0 only).
    if d.rank == 0:
        d.solver.memory_report().print()

    var writer = FrameWriter[ShallowWater](d.solver, d.nvtx, component=0)

    # Diagnostics.  Slip walls conserve mass exactly; normal momentum
    # flips on impact so x-momentum and y-momentum oscillate around
    # zero as the radial wave hits and rebounds.  max|h| shows the
    # peak amplitude decay (Rusanov dissipation smooths the ring).
    var components = DiagComponents().linear("mass", 0).linear("momentum_x", 1).linear("momentum_y", 2).maxabs("max_h", 0)
    var diag = DiagnosticsWriter[ShallowWater](d.solver, "output/diagnostics.csv", components.linear_list, components.squared_list, components.maxabs_list, LX, LY, LZ)

    var dt = choose_dt()
    if d.rank == 0:
        print("  dt =", dt, " (", Int(T_FINAL / dt), " steps estimated)")

    var result = run_ssprk3_loop_with_diagnostics[ShallowWater](d.solver, writer, diag, dt, T_FINAL, NUM_FRAMES, d.nvtx)

    writer.finalize("output/solution.pvd", d.nvtx)

    # Final-state multi-field snapshot for richer ParaView inspection
    # (the per-frame async pipeline writes one h field per frame).
    # Emits h + |u| at t=T_FINAL: h is the surface elevation, |u| is
    # the depth-averaged velocity magnitude.  Gated on np=1 since
    # each rank dumps only its owned slab.
    if d.is_single_rank():
        var snap = DownloadedSnapshot[ShallowWater](d.solver, d.nvtx, components=[0, 1, 2])
        var f_h = snap.alloc_field()
        var f_umag = snap.alloc_field()
        for k in range(snap.n_owned_dof):
            var h = snap.snaps[0][k]
            var u = snap.snaps[1][k] / h
            var v = snap.snaps[2][k] / h
            f_h[k] = Float64(h)
            f_umag[k] = Float64(sqrt(u * u + v * v))
        snap.add_field("h", f_h^)
        snap.add_field("|u|", f_umag^)
        snap.write(d.solver, d.nvtx, "output/snapshot_t_final.vtu")
        if d.rank == 0:
            print("  wrote output/snapshot_t_final.vtu (h + |u|, t=", T_FINAL, ")")

    if d.size == 1:
        var mass_final = _total_mass(d.solver, d.nvtx)
        var mass_drift = mass_final - mass_ic
        print("  integrated mass at t=", T_FINAL, " :", mass_final)
        print("  mass drift                   :", mass_drift, "  (relative:", mass_drift / mass_ic, ")")
    if d.rank == 0:
        result.print_summary()
        print("  wrote output/solution.pvd")
    # Post-run sync'd throughput measurement.
    var tput = d.solver.bench_step_loop(dt, d.nvtx)
    if d.rank == 0:
        tput.print()

    mpi.finalize()


# Integral of h via nodal sampling over owned elements, normalised so
# the IC value reads as the domain volume * mean depth -- a zero-th
# order but monotone proxy for true mass.  Slip walls + divergence-form
# flux should conserve this across the run; drift is a numerical
# diagnostic, not a scheme invariant (nodal sum is not exact
# L^2(ref-tet) integration at P2).
def _total_mass(mut solver: Solver[ShallowWater], mut nvtx: NvtxContext) raises -> Float32:
    var num_owned = solver.num_owned_elements
    var total_dof = num_owned * N_P
    var h_buf = List[Float32]()
    for _ in range(total_dof):
        h_buf.append(Float32(0.0))
    solver.download_owned_component(0, h_buf, nvtx)
    var tot: Float64 = 0.0
    for i in range(total_dof):
        tot += Float64(h_buf[i])
    return Float32(tot / Float64(total_dof) * Float64(LX) * Float64(LY) * Float64(LZ))
