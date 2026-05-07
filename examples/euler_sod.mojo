# ======================================================================
# euler_sod -- smoothed Sod shock tube in a thin 3D box (BC demo)
# ======================================================================
#
# The Sod (1978) shock-tube Riemann problem with the jump at x = 0.5
# smoothed into a tanh transition of width ~8 dx, run through the
# Venkat-smoothed Barth-Jespersen slope limiter for shock stability
# (`solver.enable_cell_limiter(True)`).  Sod's discontinuous IC plus
# Gibbs ringing on the unlimited DG scheme used to drive density to
# ~1e18 before the per-node positivity floor could clamp it; the
# tanh-smoothed IC + BJ limiter combination now gives a clean
# shock / contact / rarefaction structure at this resolution.
#
#   initial state (after smoothing)
#     left  (x << 0.5): rho ~ 1.000, u = 0, p ~ 1.000
#     right (x >> 0.5): rho ~ 0.125, u = 0, p ~ 0.100
#   gamma   = 1.4
#   t_final = 0.10  (kept short for fast iteration; longer T runs
#                    fine with the limiter, e.g. T=0.20 matches
#                    Toro's Test 1 reference values per
#                    test/sod_exact_riemann_test.mojo)
#
# By t = 0.10 a left-moving rarefaction, a contact discontinuity, and
# a right-moving shock have formed.  The shock sits near x ~ 0.7; no
# wave has reached either x end, so the transmissive outflow BCs just
# hold the far-field states fixed.  y and z BCs are slip (reflecting)
# walls, which preserves the 1D structure exactly for an initial
# condition that is y/z-invariant and has no normal momentum at the
# walls.
#
# This is the first driver that exercises non-periodic boundaries end
# to end: the mesh builder allocates mirror -x/-y/-z faces, the RK
# kernel dispatches to `physics.boundary_flux` on face_bc_type != 0,
# and the Euler module reflects normal momentum (WALL) or extrapolates
# state (OUTFLOW) to synthesise the ghost state that the HLLEC solver
# then Riemann-solves against.
#
# Multi-rank: the underlying Mesh + HaloExchange + Solver pipeline
# supports np>=2 with non-periodic BCs (gated by `make test-bc`,
# np=1 vs np=4 bit-identical correctness on a periodic-vs-outflow
# mix).  This driver runs at np=1 by default but works under
# mpirun -np N with no code changes.
#
# Output: same VTU pipeline as euler_vortex; density is component 0.
#         Also dumps output/sod_density_line.txt at the end, a plain
#         x-vs-rho trace along y = LY/2, z = LZ/2 for a quick eyeball
#         comparison against the textbook Sod reference.
# ======================================================================

from std.sys import has_accelerator
from std.gpu import global_idx
from std.gpu.host import DeviceContext
from std.math import sqrt, ceildiv, tanh

from src import mpi
from src.partition import build_partition
from src.reference import N_P, build_reference_operators
from src.mesh import Mesh
from src.boundary import (
    BoundaryConditions,
    BC_INTERIOR,
    BC_WALL,
    BC_OUTFLOW,
)
from src.halo_exchange import HaloExchange
from src.solver import Solver
from src.euler import Euler, FLUX_HLLEC
from src.nvtx import NvtxContext
from src.frame_writer import FrameWriter, write_snapshot_3d_multi
from src.time_integrator import run_ssprk3_loop_with_diagnostics
from src.diagnostics import DiagnosticsWriter, NamedComponent

# Mesh: long in x, short in y/z so the cells stay roughly cubic.
# dx = LX/NX = 1/200 = 0.005; dy = dz = LY/NY = 0.04/8 = 0.005.
comptime NX = 200
comptime NY = 8
comptime NZ = 8
comptime LX = 1.0
comptime LY = 0.04
comptime LZ = 0.04

comptime GAMMA: Float32 = 1.4
comptime MIN_DENSITY: Float32 = 1.0e-6
comptime MIN_PRESSURE: Float32 = 1.0e-6

# Sod initial states.
comptime RHO_L: Float32 = 1.0
comptime P_L: Float32 = 1.0
comptime RHO_R: Float32 = 0.125
comptime P_R: Float32 = 0.1

comptime T_FINAL: Float32 = 0.20
comptime NUM_FRAMES = 10

# Smoothing width for the tanh IC: ~8 cell widths.  Small enough that
# the structure still looks Sod-like, wide enough to keep Gibbs
# oscillations from punching past the density / pressure floors.
comptime SMOOTH_WIDTH: Float32 = Float32(8.0 * (LX / NX))

# SSPRK3 safety factor for P2 DG on tets (same as euler_vortex).
comptime CFL = Float32(0.2)
comptime IC_BLOCK = 256


# ----------------------------------------------------------------------
# Initial condition: left/right constant states, split at x = 0.5.
# ----------------------------------------------------------------------


def sod_ic_kernel(
    q: UnsafePointer[Float32, MutAnyOrigin],
    owned_elem_ids: UnsafePointer[Int32, MutAnyOrigin],
    elem_node_xyz: UnsafePointer[Float32, MutAnyOrigin],
    num_owned: Int,
    gamma: Float32,
    rho_l: Float32,
    p_l: Float32,
    rho_r: Float32,
    p_r: Float32,
    smooth_width: Float32,
):
    var idx = Int(global_idx.x)
    var total = num_owned * N_P
    if idx >= total:
        return
    var i = idx // N_P
    var nn = idx % N_P
    var e = Int(owned_elem_ids[i])
    var px = elem_node_xyz[(e * N_P + nn) * 3 + 0]
    # Smooth right-state fraction s in [0, 1]; sharp-limit equivalent
    # when smooth_width -> 0.  The factor (tanh + 1) * 0.5 maps the
    # symmetric tanh to a 0->1 ramp centered at x = 0.5.
    var s = (tanh((px - Float32(0.5)) / smooth_width) + Float32(1.0)) * Float32(
        0.5
    )
    var rho = rho_l + s * (rho_r - rho_l)
    var p = p_l + s * (p_r - p_l)
    var E = p / (gamma - Float32(1.0))  # velocities are zero
    var base = (e * N_P + nn) * 5
    q[base + 0] = rho
    q[base + 1] = Float32(0.0)
    q[base + 2] = Float32(0.0)
    q[base + 3] = Float32(0.0)
    q[base + 4] = E


# Max initial wave speed: |u| + c.  On the high-pressure side,
# c_L = sqrt(gamma * p_L / rho_L) = sqrt(1.4) ~= 1.183.  Post-shock the
# peak wave speed climbs a bit; a factor of 2 on c_L covers it.
def choose_dt() raises -> Float32:
    var h = Float32(LX) / Float32(NX)
    var c_l = sqrt(GAMMA * P_L / RHO_L)
    var wave = Float32(2.0) * c_l
    # Denominator factor (2*P+1) = 5 matches euler_vortex for P=2.
    return CFL * h / (wave * Float32(5.0))


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    mpi.init()
    var rank = mpi.world_rank()
    var size = mpi.world_size()

    if rank == 0:
        print(
            "euler_sod: GPU DG Euler, P2 tet, HLLEC flux,",
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
        print("  BCs: x = transmissive outflow, y/z = slip walls")

    var nvtx = NvtxContext()

    var refs = build_reference_operators(nvtx)

    var ctx = DeviceContext()

    # Sod BCs: transmissive outflow on x, slip walls on y and z.
    var bcs = BoundaryConditions(
        BC_OUTFLOW,
        BC_OUTFLOW,  # -x, +x
        BC_WALL,
        BC_WALL,  # -y, +y
        BC_WALL,
        BC_WALL,  # -z, +z
    )

    nvtx.push_range("build_mesh")
    var mesh = Mesh(
        ctx,
        build_partition(rank, size, NX, NY, NZ),
        LX,
        LY,
        LZ,
        bcs,
    )
    nvtx.pop_range()

    var halo = HaloExchange(
        ctx,
        mesh.part,
        Euler.NUM_COMPONENTS,
        mesh.d_perm.unsafe_ptr(),
        bcs,
    )

    var physics = Euler(
        GAMMA,
        MIN_DENSITY,
        MIN_PRESSURE,
        FLUX_HLLEC,
        True,
        Float32(0.0),
        Float32(0.0),
        Float32(0.0),
    )

    var solver = Solver[Euler](
        ctx^,
        mesh^,
        halo^,
        physics^,
        refs.D_ref^,
        refs.Lift_ref^,
        refs.node_weights^,
    )
    # Shock stabilization: Barth-Jespersen slope limiter after every
    # RK stage.  Damps each element's nodal deviations by the tightest
    # theta that keeps every nodal value within the min/max cell
    # average of the element + its 4 face neighbours.  Conservation is
    # exact; smooth regions are untouched.  Without this, Gibbs
    # oscillations around the shock drive density to ~1e18 before the
    # per-node positivity floor can clamp it back.
    solver.enable_cell_limiter(True)

    nvtx.push_range("initial_condition")
    solver.ctx.enqueue_function[sod_ic_kernel, sod_ic_kernel](
        solver.d_q.unsafe_ptr(),
        solver.mesh.d_owned_elem_ids.unsafe_ptr(),
        solver.mesh.local.d_elem_node_xyz.unsafe_ptr(),
        solver.num_owned_elements,
        GAMMA,
        RHO_L,
        P_L,
        RHO_R,
        P_R,
        SMOOTH_WIDTH,
        grid_dim=ceildiv(solver.num_owned_elements * N_P, IC_BLOCK),
        block_dim=IC_BLOCK,
    )
    solver.ctx.synchronize()
    nvtx.pop_range()

    # Pre-step perf snapshot: device memory accounting (rank 0 only).
    if rank == 0:
        solver.memory_report().print()

    var writer = FrameWriter[Euler](solver, nvtx, component=0)

    # Diagnostics.  Mass is exactly conserved even with transmissive
    # outflow (no wave has reached either x-end by t=0.10).  Tracking
    # max|density| lets us see the right-plateau deplete from rho=1
    # only when the rarefaction head arrives; shocks later cause
    # density to peak above 1 transiently.
    var diag_linear = List[NamedComponent]()
    diag_linear.append(NamedComponent("mass", 0))
    diag_linear.append(NamedComponent("momentum_x", 1))
    diag_linear.append(NamedComponent("momentum_y", 2))
    diag_linear.append(NamedComponent("momentum_z", 3))
    diag_linear.append(NamedComponent("total_energy", 4))
    var diag_maxabs = List[NamedComponent]()
    diag_maxabs.append(NamedComponent("max_density", 0))
    var diag = DiagnosticsWriter[Euler](
        solver,
        "output/diagnostics.csv",
        diag_linear,
        List[NamedComponent](),
        diag_maxabs,
        LX,
        LY,
        LZ,
    )

    var dt = choose_dt()
    if rank == 0:
        print("  dt =", dt, " (", Int(T_FINAL / dt), " steps estimated)")

    var result = run_ssprk3_loop_with_diagnostics[Euler](
        solver,
        writer,
        diag,
        dt,
        T_FINAL,
        NUM_FRAMES,
        nvtx,
    )

    writer.finalize("output/solution.pvd", nvtx)

    # Final-state multi-field snapshot (rho + p + |v|) for richer
    # ParaView inspection.  Independent of the per-frame async pipeline.
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
        write_snapshot_3d_multi(
            solver=solver,
            field_names=names,
            field_data=fields,
            path=String("output/snapshot_t_final.vtu"),
            nvtx=nvtx,
        )
        if rank == 0:
            print(
                "  wrote output/snapshot_t_final.vtu (rho + p + |v|, t=",
                T_FINAL,
                ")",
            )

    if rank == 0:
        result.print_summary()
        print("  wrote output/solution.pvd")
    # Post-run sync'd throughput measurement.
    var tput = solver.bench_step_loop(dt, nvtx)
    if rank == 0:
        tput.print()

    # The density-line / boundary-state diagnostics below are single-rank
    # conveniences -- they'd need an all-gather (or a separate post-
    # process step) to be meaningful across rank counts, which isn't
    # worth the plumbing for this demo.
    if size == 1:
        _dump_density_line(solver, nvtx, "output/sod_density_line.txt")
        _validate_boundary_states(solver, nvtx)

    mpi.finalize()


def _validate_boundary_states(
    mut solver: Solver[Euler],
    mut nvtx: NvtxContext,
) raises:
    var num_owned = solver.num_owned_elements
    var h_q = List[Float32]()
    for _ in range(num_owned * N_P):
        h_q.append(Float32(0.0))
    solver.download_owned_component(0, h_q, nvtx)

    var xyz_ptr = solver.mesh.owned_node_xyz_f32_ptr
    # Sample the min-rho and max-rho across the leftmost and rightmost
    # cells, with element centers at roughly x in (0, dx) and (LX-dx, LX).
    var dx = Float32(LX) / Float32(NX)
    var rho_left_min = Float32(1.0e30)
    var rho_left_max = Float32(-1.0e30)
    var rho_right_min = Float32(1.0e30)
    var rho_right_max = Float32(-1.0e30)
    for i in range(num_owned):
        var elem_x = xyz_ptr[(i * N_P + 0) * 3 + 0]
        if elem_x < dx:
            for nn in range(N_P):
                var r = h_q[i * N_P + nn]
                if r < rho_left_min:
                    rho_left_min = r
                if r > rho_left_max:
                    rho_left_max = r
        elif elem_x > Float32(LX) - Float32(2.0) * dx:
            for nn in range(N_P):
                var r = h_q[i * N_P + nn]
                if r < rho_right_min:
                    rho_right_min = r
                if r > rho_right_max:
                    rho_right_max = r
    print(
        "  density at -x boundary cells: [",
        rho_left_min,
        ",",
        rho_left_max,
        "]  (expected ~",
        RHO_L,
        ")",
    )
    print(
        "  density at +x boundary cells: [",
        rho_right_min,
        ",",
        rho_right_max,
        "]  (expected ~",
        RHO_R,
        ")",
    )


# ----------------------------------------------------------------------
# Density-line dump along the tube centerline
# ----------------------------------------------------------------------
#
# Picks one P2 node per element whose (y, z) is closest to (LY/2, LZ/2),
# writes (x, rho) pairs sorted by x to `output/sod_density_line.txt`.
# Coarse -- good enough to sanity-check the shock / contact / rarefaction
# structure against the known Sod reference.
# ----------------------------------------------------------------------


def _dump_density_line(
    mut solver: Solver[Euler],
    mut nvtx: NvtxContext,
    out_path: String,
) raises:
    var num_owned = solver.num_owned_elements
    var total_dof = num_owned * N_P
    var h_q = List[Float32]()
    for _ in range(total_dof):
        h_q.append(Float32(0.0))
    solver.download_owned_component(0, h_q, nvtx)

    # Download owned-element node coordinates directly from the host
    # copy Mesh keeps around.  The host copy is laid out as
    # [owned_elem][node][xyz].
    var xyz_ptr = solver.mesh.owned_node_xyz_f32_ptr
    var y_mid = Float32(LY) * Float32(0.5)
    var z_mid = Float32(LZ) * Float32(0.5)

    # For each element, pick the P2 node whose (y, z) is closest to
    # the centerline, then emit (x, rho) at that node.
    var xs = List[Float32]()
    var rhos = List[Float32]()
    for i in range(num_owned):
        var best_nn = 0
        var best_d2 = Float32(1.0e30)
        for nn in range(N_P):
            var py = xyz_ptr[(i * N_P + nn) * 3 + 1]
            var pz = xyz_ptr[(i * N_P + nn) * 3 + 2]
            var dy = py - y_mid
            var dz = pz - z_mid
            var d2 = dy * dy + dz * dz
            if d2 < best_d2:
                best_d2 = d2
                best_nn = nn
        xs.append(xyz_ptr[(i * N_P + best_nn) * 3 + 0])
        rhos.append(h_q[i * N_P + best_nn])

    # Bucket-sort by x so the file is monotone for plotting.
    var order = List[Int]()
    for i in range(len(xs)):
        order.append(i)
    # Simple insertion sort; num_owned is small along the line sampling
    # so this is fine.
    for i in range(1, len(order)):
        var cur = order[i]
        var j = i - 1
        while j >= 0 and xs[order[j]] > xs[cur]:
            order[j + 1] = order[j]
            j -= 1
        order[j + 1] = cur

    with open(out_path, "w") as f:
        f.write("# x rho\n")
        for idx in range(len(order)):
            var k = order[idx]
            f.write(String(xs[k]))
            f.write(" ")
            f.write(String(rhos[k]))
            f.write("\n")
    print("  wrote", out_path)
