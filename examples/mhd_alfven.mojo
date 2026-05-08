# ======================================================================
# mhd_alfven -- linearly polarised Alfven wave, periodic box
# ======================================================================
#
# Canonical Ideal-MHD test: a transverse Alfven wave travelling along
# +x with background B = (B0, 0, 0), background density rho0, and
# initial perturbation
#
#   u_y(x, 0) =  A sin(k x)
#   B_y(x, 0) = -A sin(k x)        (right-going, Alfven speed c_A = B0 / sqrt(rho0))
#
# gives the exact solution
#
#   u_y(x, t) =  A sin(k x - omega t),   omega = c_A k
#   B_y(x, t) = -A sin(k x - omega t)
#
# with everything else constant in time and x.  After one period
# T = 2 pi / omega = L / c_A (with L = 2 pi / k the domain length) the
# state returns exactly to the IC; the round-trip L2 error in B_y vs
# the IC is a quantitative convergence check.
#
# div(B) is identically zero for this IC, so the GLM cleaner has
# nothing to damp -- psi stays at zero throughout.  A dirtier test
# (with sharp features or finite-precision div(B)) would see psi
# acquire nonzero values and get damped toward zero at rate alpha_d.
#
# Periodic BCs on all six sides.  Single rank only at present.
# ======================================================================

from std.sys import has_accelerator
from std.gpu import global_idx
from std.math import sqrt, ceildiv, sin, cos

from src import mpi
from src.reference import N_P
from src.boundary import BoundaryConditions
from src.solver import Solver
from src.mhd import IdealMHD
from src.driver3d import Driver3D
from src.frame_writer import FrameWriter, DownloadedSnapshot
from src.time_integrator import run_ssprk3_loop_with_diagnostics
from src.diagnostics import DiagnosticsWriter, DiagComponents


comptime NX = 32
comptime NY = 4
comptime NZ = 4
comptime LX = 1.0
comptime LY = Float64(4.0 / 32.0)  # dx = dy = dz = 1 / NX
comptime LZ = Float64(4.0 / 32.0)

# Alfven-wave parameters.  rho0 = B0 = 1 gives c_A = 1 and period = LX.
comptime GAMMA: Float32 = Float32(5.0 / 3.0)
comptime RHO0: Float32 = 1.0
comptime B0: Float32 = 1.0
comptime P0: Float32 = 0.1
comptime AMPLITUDE: Float32 = 0.1
comptime PI_F: Float32 = 3.14159265358979323846

# c_A = B0 / sqrt(rho0); wave period T = LX / c_A.
comptime T_FINAL: Float32 = 1.0  # exactly one period (LX / c_A)
comptime NUM_FRAMES = 20

# GLM cleaning: c_h set a bit above the expected max fast speed; a
# nonzero alpha_d damps any accumulated div(B) noise.  Both are
# unneeded for this analytic test (div(B) stays at zero to roundoff)
# but turning GLM on proves the full pipeline compiles and runs.
comptime C_H: Float32 = 1.5
comptime ALPHA_D: Float32 = 0.5

comptime MIN_DENSITY: Float32 = 1.0e-6
comptime MIN_PRESSURE: Float32 = 1.0e-6

comptime CFL = Float32(0.2)
comptime IC_BLOCK = 256


def alfven_ic_kernel(
    q: UnsafePointer[Float32, MutAnyOrigin],
    owned_elem_ids: UnsafePointer[Int32, MutAnyOrigin],
    elem_node_xyz: UnsafePointer[Float32, MutAnyOrigin],
    num_owned: Int,
    Lx: Float32,
    amp: Float32,
    B0_val: Float32,
    rho0_val: Float32,
    p0_val: Float32,
    gamma: Float32,
):
    var idx = Int(global_idx.x)
    var total = num_owned * N_P
    if idx >= total:
        return
    var i = idx // N_P
    var nn = idx % N_P
    var e = Int(owned_elem_ids[i])
    var px = elem_node_xyz[(e * N_P + nn) * 3 + 0]

    var k = Float32(2.0) * PI_F / Lx
    var s = sin(k * px)
    var uy = amp * s  # transverse velocity perturbation
    var By = -amp * s  # locked to u via Alfven relation

    var rho = rho0_val
    var u = Float32(0.0)
    var v = uy
    var w = Float32(0.0)
    var bx = B0_val
    var by = By
    var bz = Float32(0.0)
    var p_gas = p0_val
    var E = p_gas / (gamma - Float32(1.0)) + Float32(0.5) * rho * (u * u + v * v + w * w) + Float32(0.5) * (bx * bx + by * by + bz * bz)
    var base = (e * N_P + nn) * 9
    q[base + 0] = rho
    q[base + 1] = rho * u
    q[base + 2] = rho * v
    q[base + 3] = rho * w
    q[base + 4] = E
    q[base + 5] = bx
    q[base + 6] = by
    q[base + 7] = bz
    q[base + 8] = Float32(0.0)  # psi


def choose_dt() raises -> Float32:
    var h = Float32(LX) / Float32(NX)
    # Fast magnetosonic speed upper bound: sqrt(c_s^2 + c_a^2).
    var cs2 = GAMMA * P0 / RHO0
    var ca2 = (B0 * B0 + AMPLITUDE * AMPLITUDE) / RHO0  # worst-case |B|
    var cf = sqrt(cs2 + ca2)
    var wave = max(cf, C_H)
    return CFL * h / (wave * Float32(5.0))


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"

    var physics = IdealMHD(GAMMA, MIN_DENSITY, MIN_PRESSURE, C_H, ALPHA_D)
    var d = Driver3D[IdealMHD](
        problem_name="mhd_alfven: GPU DG ideal MHD + GLM, P2 tet, Rusanov",
        nx=NX,
        ny=NY,
        nz=NZ,
        lx=LX,
        ly=LY,
        lz=LZ,
        bcs=BoundaryConditions.periodic(),
        physics=physics^,
    )
    if d.rank == 0:
        print("  c_h =", C_H, "  alpha_d =", ALPHA_D)

    d.solver.ctx.enqueue_function[alfven_ic_kernel](
        d.solver.d_q.unsafe_ptr(),
        d.solver.mesh.d_owned_elem_ids.unsafe_ptr(),
        d.solver.mesh.local.d_elem_node_xyz.unsafe_ptr(),
        d.solver.num_owned_elements,
        Float32(LX),
        AMPLITUDE,
        B0,
        RHO0,
        P0,
        GAMMA,
        grid_dim=ceildiv(d.solver.num_owned_elements * N_P, IC_BLOCK),
        block_dim=IC_BLOCK,
    )
    d.solver.ctx.synchronize()

    # Save IC for By (component 6) for round-trip L2 comparison.
    var by_ic = List[Float32]()
    for _ in range(d.solver.num_owned_elements * N_P):
        by_ic.append(Float32(0.0))
    d.solver.download_owned_component(6, by_ic, d.nvtx)

    # density writer (component 0) -- watching the wave is more useful
    # via By, but the VTU writer only emits one scalar; density is the
    # conventional "density" slot.  Use ParaView's Calculator filter
    # to see By if curious.
    # Pre-step perf snapshot: device memory accounting (rank 0 only).
    if d.rank == 0:
        d.solver.memory_report().print()

    var writer = FrameWriter[IdealMHD](d.solver, d.nvtx, component=6)

    # Diagnostics.  Fluid invariants: mass, 3 momenta, total energy.
    # Magnetic energy int(|B|^2) dV is conserved too (via the squared
    # components).  `max_abs_psi` tracks the GLM cleaner's progress --
    # for a divergence-free IC this stays at roundoff the whole run.
    var components = (
        DiagComponents()
        .linear("mass", 0)
        .linear("momentum_x", 1)
        .linear("momentum_y", 2)
        .linear("momentum_z", 3)
        .linear("total_energy", 4)
        .squared("Bx_sq", 5)
        .squared("By_sq", 6)
        .squared("Bz_sq", 7)
        .maxabs("max_abs_psi", 8)
    )
    var diag = DiagnosticsWriter[IdealMHD](d.solver, "output/diagnostics.csv", components.linear_list, components.squared_list, components.maxabs_list, LX, LY, LZ)

    var dt = choose_dt()
    if d.rank == 0:
        print("  dt =", dt, " (", Int(T_FINAL / dt), " steps estimated)")

    var result = run_ssprk3_loop_with_diagnostics[IdealMHD](d.solver, writer, diag, dt, T_FINAL, NUM_FRAMES, d.nvtx)

    writer.finalize("output/solution.pvd", d.nvtx)

    # Final-state multi-field snapshot for richer ParaView inspection
    # (the per-frame async pipeline above writes one psi field per
    # frame for performance).  Emits By + |B| + psi at t=T_FINAL --
    # By is the dominant Alfven-wave perturbation, |B| shows the
    # magnetic-field magnitude, and psi exposes any GLM cleaning
    # residual.  Gated on np=1 since each rank dumps only its
    # owned slab.
    if d.is_single_rank():
        var snap = DownloadedSnapshot[IdealMHD](d.solver, d.nvtx, components=[5, 6, 7, 8])
        var f_by = snap.alloc_field()
        var f_bmag = snap.alloc_field()
        var f_psi = snap.alloc_field()
        for k in range(snap.n_owned_dof):
            var bx = snap.snaps[0][k]
            var by = snap.snaps[1][k]
            var bz = snap.snaps[2][k]
            f_by[k] = Float64(by)
            f_bmag[k] = Float64(sqrt(bx * bx + by * by + bz * bz))
            f_psi[k] = Float64(snap.snaps[3][k])
        snap.add_field("By", f_by^)
        snap.add_field("|B|", f_bmag^)
        snap.add_field("psi", f_psi^)
        snap.write(d.solver, d.nvtx, "output/snapshot_t_final.vtu")
        if d.rank == 0:
            print("  wrote output/snapshot_t_final.vtu (By + |B| + psi, t=", T_FINAL, ")")

    # The round-trip L2 and max-|psi| diagnostics below sum over this
    # rank's owned elements only; at np>1 the globally-correct numbers
    # would need an allreduce, which isn't worth adding for this demo
    # -- gate on np=1 so we don't print misleading partial sums.
    if d.size == 1:
        var by_fin = List[Float32]()
        for _ in range(d.solver.num_owned_elements * N_P):
            by_fin.append(Float32(0.0))
        d.solver.download_owned_component(6, by_fin, d.nvtx)
        var err2: Float64 = 0.0
        var ref2: Float64 = 0.0
        for i in range(len(by_ic)):
            var dd = Float64(by_fin[i] - by_ic[i])
            var r = Float64(by_ic[i])
            err2 += dd * dd
            ref2 += r * r
        var rel_l2 = sqrt(err2 / ref2) if ref2 > 0.0 else sqrt(err2)
        print("  relative L2(By) vs IC after one period:", Float32(rel_l2))

        var psi_fin = List[Float32]()
        for _ in range(d.solver.num_owned_elements * N_P):
            psi_fin.append(Float32(0.0))
        d.solver.download_owned_component(8, psi_fin, d.nvtx)
        var max_psi: Float32 = 0.0
        for i in range(len(psi_fin)):
            var p = psi_fin[i] if psi_fin[i] >= Float32(0.0) else -psi_fin[i]
            if p > max_psi:
                max_psi = p
        print("  max |psi| (GLM monopole tracer) :", max_psi)

    if d.rank == 0:
        result.print_summary()
        print("  wrote output/solution.pvd")
    # Post-run sync'd throughput measurement.
    var tput = d.solver.bench_step_loop(dt, d.nvtx)
    if d.rank == 0:
        tput.print()

    mpi.finalize()
