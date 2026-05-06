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
from std.gpu.host import DeviceContext
from std.math import sqrt, ceildiv, sin, cos

from src import mpi
from src.partition import build_partition
from src.reference import N_P, build_reference_operators
from src.mesh import Mesh
from src.boundary import BoundaryConditions, BC_INTERIOR, BC_WALL
from src.halo_exchange import HaloExchange
from src.solver import Solver
from src.maxwell import Maxwell
from src.nvtx import NvtxContext
from src.frame_writer import FrameWriter, write_snapshot_3d_multi
from src.time_integrator import run_ssprk3_loop_with_diagnostics
from src.diagnostics import DiagnosticsWriter, NamedComponent


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


def cavity_ic_kernel(
    q: UnsafePointer[Float32, MutAnyOrigin],
    owned_elem_ids: UnsafePointer[Int32, MutAnyOrigin],
    elem_node_xyz: UnsafePointer[Float32, MutAnyOrigin],
    num_owned: Int,
    Ly: Float32,
):
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
    mpi.init()
    var rank = mpi.world_rank()
    var size = mpi.world_size()

    if rank == 0:
        print(
            "maxwell_cavity: GPU DG Maxwell, P2 tet, Rusanov,",
            size,
            "rank(s)",
        )
        print(
            "  global mesh: ",
            NX,
            "x",
            NY,
            "x",
            NZ,
            " cells -> ",
            NX * NY * NZ * 6,
            "tets",
        )

    var nvtx = NvtxContext()

    var refs = build_reference_operators(nvtx)

    var ctx = DeviceContext()

    # PEC walls on y, periodic on x and z.
    var bcs = BoundaryConditions(
        BC_INTERIOR,
        BC_INTERIOR,  # -x, +x
        BC_WALL,
        BC_WALL,  # -y, +y
        BC_INTERIOR,
        BC_INTERIOR,  # -z, +z
    )

    var mesh = Mesh(
        ctx,
        build_partition(rank, size, NX, NY, NZ),
        LX,
        LY,
        LZ,
        bcs,
    )
    var halo = HaloExchange(
        ctx,
        mesh.part,
        Maxwell.NUM_COMPONENTS,
        mesh.d_perm.unsafe_ptr(),
        bcs,
    )
    var physics = Maxwell(
        C_LIGHT,
        Float32(0.0),
        Float32(0.0),
        Float32(0.0),  # J = 0
        Float32(0.0),
        Float32(0.0),
        Float32(0.0),  # M = 0
    )
    var solver = Solver[Maxwell](
        ctx^,
        mesh^,
        halo^,
        physics^,
        refs.D_ref^,
        refs.Lift_ref^,
        refs.node_weights^,
    )

    solver.ctx.enqueue_function[cavity_ic_kernel, cavity_ic_kernel](
        solver.d_q.unsafe_ptr(),
        solver.mesh.d_owned_elem_ids.unsafe_ptr(),
        solver.mesh.local.d_elem_node_xyz.unsafe_ptr(),
        solver.num_owned_elements,
        Float32(LY),
        grid_dim=ceildiv(solver.num_owned_elements * N_P, IC_BLOCK),
        block_dim=IC_BLOCK,
    )
    solver.ctx.synchronize()

    # Save the IC for the round-trip L2 comparison at T = one period.
    # (Only needed at size==1 where we print the diagnostic; gate the
    # whole save so np>1 doesn't pay the download cost for nothing.)
    var h_ic = List[Float32]()
    if size == 1:
        for _ in range(solver.num_owned_elements * N_P):
            h_ic.append(Float32(0.0))
        solver.download_owned_component(0, h_ic, nvtx)
        var energy_ic = _em_energy(solver, nvtx)
        print("  EM energy at t=0    :", energy_ic)

    # Pre-step perf snapshot: device memory accounting (rank 0 only).
    if rank == 0:
        solver.memory_report().print()

    var writer = FrameWriter[Maxwell](solver, nvtx, component=0)

    # Diagnostics.  Maxwell has no linear conserved integrals (each E/B
    # component oscillates around zero), but the total EM energy
    #   U_EM = 0.5 * int(|E|^2 + c^2 |B|^2) dV
    # is conserved to scheme precision.  We record each quadratic
    # component separately; the postprocessor sums them.
    var diag_squared = List[NamedComponent]()
    diag_squared.append(NamedComponent("Ex_sq", 0))
    diag_squared.append(NamedComponent("Ey_sq", 1))
    diag_squared.append(NamedComponent("Ez_sq", 2))
    diag_squared.append(NamedComponent("Bx_sq", 3))
    diag_squared.append(NamedComponent("By_sq", 4))
    diag_squared.append(NamedComponent("Bz_sq", 5))
    var diag = DiagnosticsWriter[Maxwell](
        solver,
        "output/diagnostics.csv",
        List[NamedComponent](),
        diag_squared,
        List[NamedComponent](),
        LX,
        LY,
        LZ,
    )

    var dt = choose_dt()
    if rank == 0:
        print("  dt =", dt, " (", Int(T_FINAL / dt), " steps estimated)")

    var result = run_ssprk3_loop_with_diagnostics[Maxwell](
        solver,
        writer,
        diag,
        dt,
        T_FINAL,
        NUM_FRAMES,
        nvtx,
    )

    writer.finalize("output/solution.pvd", nvtx)

    # Final-state multi-field snapshot for richer ParaView inspection
    # (the per-frame async pipeline above writes one Ex field per
    # frame for performance).  Emits Ex + |E| + |B| at t=T_FINAL to
    # output/snapshot_t_final.vtu -- Ex shows the dominant cavity
    # standing-wave pattern, |E| and |B| show the energy localisation.
    # Gated on np=1 since each rank would dump only its owned slab.
    var nprocs = solver.mesh.part.px * solver.mesh.part.py * solver.mesh.part.pz
    if nprocs == 1:
        var n_owned_dof = solver.num_owned_elements * N_P
        var snap_ex = List[Float32]()
        var snap_ey = List[Float32]()
        var snap_ez = List[Float32]()
        var snap_bx = List[Float32]()
        var snap_by = List[Float32]()
        var snap_bz = List[Float32]()
        for _ in range(n_owned_dof):
            snap_ex.append(Float32(0.0))
            snap_ey.append(Float32(0.0))
            snap_ez.append(Float32(0.0))
            snap_bx.append(Float32(0.0))
            snap_by.append(Float32(0.0))
            snap_bz.append(Float32(0.0))
        solver.download_owned_component(0, snap_ex, nvtx)
        solver.download_owned_component(1, snap_ey, nvtx)
        solver.download_owned_component(2, snap_ez, nvtx)
        solver.download_owned_component(3, snap_bx, nvtx)
        solver.download_owned_component(4, snap_by, nvtx)
        solver.download_owned_component(5, snap_bz, nvtx)
        var f_ex = List[Float64]()
        var f_emag = List[Float64]()
        var f_bmag = List[Float64]()
        for k in range(n_owned_dof):
            var ex = snap_ex[k]
            var ey = snap_ey[k]
            var ez = snap_ez[k]
            var bx = snap_bx[k]
            var by = snap_by[k]
            var bz = snap_bz[k]
            f_ex.append(Float64(ex))
            f_emag.append(Float64(sqrt(ex * ex + ey * ey + ez * ez)))
            f_bmag.append(Float64(sqrt(bx * bx + by * by + bz * bz)))
        var fields = List[List[Float64]]()
        fields.append(f_ex^)
        fields.append(f_emag^)
        fields.append(f_bmag^)
        var names = List[String]()
        names.append(String("Ex"))
        names.append(String("|E|"))
        names.append(String("|B|"))
        write_snapshot_3d_multi(
            solver=solver,
            field_names=names,
            field_data=fields,
            path=String("output/snapshot_t_final.vtu"),
            nvtx=nvtx,
        )
        if rank == 0:
            print(
                "  wrote output/snapshot_t_final.vtu (Ex + |E| + |B|, t=",
                T_FINAL,
                ")",
            )

    # Round-trip L2 + energy diagnostics are rank-local sums; at np>1
    # they'd need an allreduce to be meaningful, so gate on np=1.
    if size == 1:
        var energy_final = _em_energy(solver, nvtx)
        print(
            "  EM energy at t=",
            T_FINAL,
            " :",
            energy_final,
        )
        var h_fin = List[Float32]()
        for _ in range(solver.num_owned_elements * N_P):
            h_fin.append(Float32(0.0))
        solver.download_owned_component(0, h_fin, nvtx)
        var err2: Float64 = 0.0
        var ref2: Float64 = 0.0
        for i in range(len(h_ic)):
            var d = Float64(h_fin[i] - h_ic[i])
            var r = Float64(h_ic[i])
            err2 += d * d
            ref2 += r * r
        var l2_rel = sqrt(err2 / ref2) if ref2 > 0.0 else sqrt(err2)
        print("  relative L2(Ex) vs IC after one period:", Float32(l2_rel))
    if rank == 0:
        print(
            "  total steps:",
            result.total_steps,
            " wall time:",
            result.wall_sec,
            "s",
        )
        print("  wrote output/solution.pvd")
    # Post-run sync'd throughput measurement.
    var tput = solver.bench_step_loop(dt, nvtx)
    if rank == 0:
        tput.print()

    mpi.finalize()


# Total EM field energy: 0.5 * integral (|E|^2 + c^2 |B|^2) dV.
# In c = 1 units this reduces to 0.5 * (|E|^2 + |B|^2), summed at
# nodal quadrature.  Nodal collocation isn't a true L2 inner product on
# P2 tets, but drift in this sum is still a useful monotonicity check:
# for a closed PEC cavity with no sources, the semi-discrete Maxwell
# system conserves it to roundoff; Rusanov dissipation causes a slow,
# monotonic decay.
def _em_energy(
    mut solver: Solver[Maxwell],
    mut nvtx: NvtxContext,
) raises -> Float32:
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
    return Float32(
        0.5 * tot / Float64(total_dof) * Float64(LX) * Float64(LY) * Float64(LZ)
    )
