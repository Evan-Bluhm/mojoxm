# ======================================================================
# maxwell_cavity -- PEC-bounded standing wave in a rectangular cavity
# ======================================================================
#
# Canonical Maxwell test: a TE-like lowest-order mode in a 1 x 1 x 1
# cavity with perfect-electric-conductor walls on +/- y and periodic
# boundaries on x and z.  The solution is a pure standing wave,
#
#   E_x(y, t) = sin(pi y / L) cos(omega t)
#   B_z(y, t) = cos(pi y / L) sin(omega t)
#   omega = c pi / L
#
# satisfying the PEC conditions E_tangential = 0 at y = 0 and y = L
# (E_x vanishes there) and B_normal = 0 at both walls (B_y is
# identically zero throughout).  After one period T = 2 L / c the
# state returns bit-by-bit to the IC; the driver reports the L2 error
# in E_x vs the IC at T = 2 L / c, which should shrink with mesh
# refinement at the expected DG P2 rate.
#
# Mesh choice: periodic (-x, +x, -z, +z) and PEC-walled (-y, +y).
# Tests both the new BC infrastructure (PEC is a `BC_WALL`) and the
# Maxwell physics end to end.
# ======================================================================

from std.sys import has_accelerator
from std.gpu import global_idx
from std.math import sqrt, ceildiv, sin, cos

from src import mpi
from src.reference import N_P
from src.boundary import BoundaryConditions, BC_INTERIOR, BC_WALL
from src.solver import Solver
from src.maxwell import Maxwell
from src.driver3d import Driver3D
from src.nvtx import NvtxContext
from src.frame_writer import FrameWriter, DownloadedSnapshot
from src.time_integrator import run_ssprk3_loop_with_diagnostics
from src.diagnostics import DiagnosticsWriter, DiagComponents


comptime NX = 16
comptime NY = 32
comptime NZ = 2
comptime LX = 1.0
comptime LY = 1.0
comptime LZ = Float64(2.0 / 32.0)  # thin in z so dz = dy = dx

# Natural units: c = 1, so period T = 2 L / c = 2.
comptime C_LIGHT: Float32 = 1.0
comptime T_FINAL: Float32 = 2.0
comptime NUM_FRAMES = 20

comptime PI_F: Float32 = 3.14159265358979323846

comptime CFL = Float32(0.2)
comptime IC_BLOCK = 256


def cavity_ic_kernel(q: UnsafePointer[Float32, MutAnyOrigin], owned_elem_ids: UnsafePointer[Int32, MutAnyOrigin], elem_node_xyz: UnsafePointer[Float32, MutAnyOrigin], num_owned: Int, Ly: Float32):
    var idx = Int(global_idx.x)
    var total = num_owned * N_P
    if idx >= total:
        return
    var i = idx // N_P
    var nn = idx % N_P
    var e = Int(owned_elem_ids[i])
    var py = elem_node_xyz[(e * N_P + nn) * 3 + 1]
    var base = (e * N_P + nn) * 6
    q[base + 0] = sin(PI_F * py / Ly)  # Ex
    q[base + 1] = Float32(0.0)
    q[base + 2] = Float32(0.0)
    q[base + 3] = Float32(0.0)
    q[base + 4] = Float32(0.0)
    q[base + 5] = Float32(0.0)  # B_z = 0 at t = 0


def choose_dt() raises -> Float32:
    var h = Float32(LY) / Float32(NY)
    # Max wave speed = c for Maxwell.
    var wave = C_LIGHT
    return CFL * h / (wave * Float32(5.0))


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"

    # PEC walls on y, periodic on x and z.
    var bcs = BoundaryConditions(BC_INTERIOR, BC_INTERIOR, BC_WALL, BC_WALL, BC_INTERIOR, BC_INTERIOR)
    var physics = Maxwell(C_LIGHT, Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0))
    var d = Driver3D[Maxwell](
        problem_name="maxwell_cavity: GPU DG Maxwell, P2 tet, Rusanov",
        nx=NX,
        ny=NY,
        nz=NZ,
        lx=LX,
        ly=LY,
        lz=LZ,
        bcs=bcs,
        physics=physics^,
    )

    d.solver.ctx.enqueue_function[cavity_ic_kernel](
        d.solver.d_q.unsafe_ptr(),
        d.solver.mesh.d_owned_elem_ids.unsafe_ptr(),
        d.solver.mesh.local.d_elem_node_xyz.unsafe_ptr(),
        d.solver.num_owned_elements,
        Float32(LY),
        grid_dim=ceildiv(d.solver.num_owned_elements * N_P, IC_BLOCK),
        block_dim=IC_BLOCK,
    )
    d.solver.ctx.synchronize()

    # Save the IC for the round-trip L2 comparison at T = one period.
    # (Only needed at size==1 where we print the diagnostic; gate the
    # whole save so np>1 doesn't pay the download cost for nothing.)
    var h_ic = List[Float32]()
    if d.size == 1:
        for _ in range(d.solver.num_owned_elements * N_P):
            h_ic.append(Float32(0.0))
        d.solver.download_owned_component(0, h_ic, d.nvtx)
        var energy_ic = _em_energy(d.solver, d.nvtx)
        print("  EM energy at t=0    :", energy_ic)

    # Pre-step perf snapshot: device memory accounting (rank 0 only).
    if d.rank == 0:
        d.solver.memory_report().print()

    var writer = FrameWriter[Maxwell](d.solver, d.nvtx, component=0)

    # Diagnostics.  Maxwell has no linear conserved integrals (each E/B
    # component oscillates around zero), but the total EM energy
    #   U_EM = 0.5 * int(|E|^2 + c^2 |B|^2) dV
    # is conserved to scheme precision.  We record each quadratic
    # component separately; the postprocessor sums them.
    var components = DiagComponents().squared("Ex_sq", 0).squared("Ey_sq", 1).squared("Ez_sq", 2).squared("Bx_sq", 3).squared("By_sq", 4).squared("Bz_sq", 5)
    var diag = DiagnosticsWriter[Maxwell](d.solver, "output/diagnostics.csv", components.linear_list, components.squared_list, components.maxabs_list, LX, LY, LZ)

    var dt = choose_dt()
    if d.rank == 0:
        print("  dt =", dt, " (", Int(T_FINAL / dt), " steps estimated)")

    var result = run_ssprk3_loop_with_diagnostics[Maxwell](d.solver, writer, diag, dt, T_FINAL, NUM_FRAMES, d.nvtx)

    writer.finalize("output/solution.pvd", d.nvtx)

    # Final-state multi-field snapshot for richer ParaView inspection
    # (the per-frame async pipeline above writes one Ex field per
    # frame for performance).  Emits Ex + |E| + |B| at t=T_FINAL to
    # output/snapshot_t_final.vtu -- Ex shows the dominant cavity
    # standing-wave pattern, |E| and |B| show the energy localisation.
    # Gated on np=1 since each rank would dump only its owned slab.
    if d.is_single_rank():
        var snap = DownloadedSnapshot[Maxwell](d.solver, d.nvtx, components=[0, 1, 2, 3, 4, 5])
        var f_ex = snap.alloc_field()
        var f_emag = snap.alloc_field()
        var f_bmag = snap.alloc_field()
        for k in range(snap.n_owned_dof):
            var ex = snap.snaps[0][k]
            var ey = snap.snaps[1][k]
            var ez = snap.snaps[2][k]
            var bx = snap.snaps[3][k]
            var by = snap.snaps[4][k]
            var bz = snap.snaps[5][k]
            f_ex[k] = Float64(ex)
            f_emag[k] = Float64(sqrt(ex * ex + ey * ey + ez * ez))
            f_bmag[k] = Float64(sqrt(bx * bx + by * by + bz * bz))
        snap.add_field("Ex", f_ex^)
        snap.add_field("|E|", f_emag^)
        snap.add_field("|B|", f_bmag^)
        snap.write(d.solver, d.nvtx, "output/snapshot_t_final.vtu")
        if d.rank == 0:
            print("  wrote output/snapshot_t_final.vtu (Ex + |E| + |B|, t=", T_FINAL, ")")

    # Round-trip L2 + energy diagnostics are rank-local sums; at np>1
    # they'd need an allreduce to be meaningful, so gate on np=1.
    if d.size == 1:
        var energy_final = _em_energy(d.solver, d.nvtx)
        print("  EM energy at t=", T_FINAL, " :", energy_final)
        var h_fin = List[Float32]()
        for _ in range(d.solver.num_owned_elements * N_P):
            h_fin.append(Float32(0.0))
        d.solver.download_owned_component(0, h_fin, d.nvtx)
        var err2: Float64 = 0.0
        var ref2: Float64 = 0.0
        for i in range(len(h_ic)):
            var dd = Float64(h_fin[i] - h_ic[i])
            var r = Float64(h_ic[i])
            err2 += dd * dd
            ref2 += r * r
        var l2_rel = sqrt(err2 / ref2) if ref2 > 0.0 else sqrt(err2)
        print("  relative L2(Ex) vs IC after one period:", Float32(l2_rel))
    if d.rank == 0:
        result.print_summary()
        print("  wrote output/solution.pvd")
    # Post-run sync'd throughput measurement.
    var tput = d.solver.bench_step_loop(dt, d.nvtx)
    if d.rank == 0:
        tput.print()

    mpi.finalize()


# Total EM field energy: 0.5 * integral (|E|^2 + c^2 |B|^2) dV.
# In c = 1 units this reduces to 0.5 * (|E|^2 + |B|^2), summed at
# nodal quadrature.  Nodal collocation isn't a true L2 inner product on
# P2 tets, but drift in this sum is still a useful monotonicity check:
# for a closed PEC cavity with no sources, the semi-discrete Maxwell
# system conserves it to roundoff; Rusanov dissipation causes a slow,
# monotonic decay.
def _em_energy(mut solver: Solver[Maxwell], mut nvtx: NvtxContext) raises -> Float32:
    var num_owned = solver.num_owned_elements
    var total_dof = num_owned * N_P
    var tot: Float64 = 0.0
    var h_buf = List[Float32]()
    for _ in range(total_dof):
        h_buf.append(Float32(0.0))
    for c in range(6):
        solver.download_owned_component(c, h_buf, nvtx)
        var weight: Float64 = 1.0 if c < 3 else Float64(C_LIGHT * C_LIGHT)
        for i in range(total_dof):
            var v = Float64(h_buf[i])
            tot += weight * v * v
    return Float32(0.5 * tot / Float64(total_dof) * Float64(LX) * Float64(LY) * Float64(LZ))
