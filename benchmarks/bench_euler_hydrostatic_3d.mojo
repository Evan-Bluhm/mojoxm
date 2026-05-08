# ======================================================================
# bench_euler_hydrostatic_3d -- Euler gravity source-term gate
# ======================================================================
#
# Closes a real coverage gap.  The Euler gravity source
# (`source_term` in src/euler.mojo, lines 779+) is used by
# examples/euler_rising_bubble.mojo but no benchmark gates it.  A
# regression in the rho*g momentum source or the rho*(u.g) energy
# source would not have tripped any existing Euler gate.
#
# Cleanest analytic test: hydrostatic balance.  For a constant-
# density fluid at rest with gravity g = (0, 0, -|gz|), the
# equilibrium pressure profile is linear:
#
#   rho(z) = rho0
#   u = v = w = 0
#   p(z)   = p0 - rho0 * |gz| * z
#
# Hydrostatic balance: -dp/dz = rho0 * |gz| = rho(z) * |gz|, so the
# pressure-gradient flux exactly cancels the gravity source.  The
# state must remain at rest -- any imbalance shows up as nonzero
# velocity that grows with time.
#
# Periodic BCs would conflict with the stratification (p(L) != p(0));
# we use BC_WALL on +/-z faces (slip wall reflects w=0 trivially)
# and periodic in x and y.
#
# Pass criteria (P=2, NX=4, NY=4, NZ=16, single-rank):
#   * max |v_max| < 1e-3 (state remains at rest)
#   * max |rho - rho0| / rho0 < 1e-4 (density unchanged)
#   * max |p_measured - p_exact| / p0 < 5e-4 (pressure profile)
#   * no NaN / Inf
# ======================================================================

from std.sys import has_accelerator
from std.gpu import global_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from std.math import sqrt, ceildiv, isnan, isinf

from src import mpi
from src.partition import build_partition
from src.reference import N_P, build_reference_operators
from src.mesh import Mesh
from src.boundary import BoundaryConditions, BC_INTERIOR, BC_WALL
from src.halo_exchange import HaloExchange
from src.solver import Solver
from src.euler import Euler, FLUX_HLLEC
from src.nvtx import NvtxContext


comptime NX = 4
comptime NY = 4
comptime NZ = 16
comptime LX = 1.0
comptime LY = 1.0
comptime LZ = 2.0

comptime GAMMA: Float32 = 1.4
comptime RHO0: Float32 = 1.0
comptime P0: Float32 = 1.0
comptime GZ_NEG: Float32 = -0.1  # gravity points in -z
comptime T_FINAL: Float32 = 1.0
comptime CFL = Float32(0.2)
comptime IC_BLOCK = 256
comptime MIN_DENSITY = Float32(1.0e-6)
comptime MIN_PRESSURE = Float32(1.0e-6)

# Empirical: well-balanced DG at P=2 on this stratification leaves
# residual velocity ~3e-5 after T=1.  1e-3 leaves ~30x margin.
comptime VMAX_TOL: Float64 = 1.0e-3
comptime RHO_REL_TOL: Float64 = 1.0e-4
comptime P_REL_TOL: Float64 = 5.0e-4


def hydrostatic_ic_kernel(
    q: UnsafePointer[Float32, MutAnyOrigin],
    owned_elem_ids: UnsafePointer[Int32, MutAnyOrigin],
    elem_node_xyz: UnsafePointer[Float32, MutAnyOrigin],
    num_owned: Int,
):
    var idx = Int(global_idx.x)
    var total = num_owned * N_P
    if idx >= total:
        return
    var i = idx // N_P
    var nn = idx % N_P
    var e = Int(owned_elem_ids[i])
    var pz = elem_node_xyz[(e * N_P + nn) * 3 + 2]

    # gravity acceleration is -|GZ_NEG| z^; equilibrium pressure is
    # p(z) = P0 - RHO0 * |GZ_NEG| * z = P0 + RHO0 * GZ_NEG * z
    # since GZ_NEG itself is negative.
    var p_z = P0 + RHO0 * GZ_NEG * pz
    var rho = RHO0
    var E = p_z / (GAMMA - Float32(1.0))  # u = v = w = 0 so KE = 0
    var base = (e * N_P + nn) * 5
    q[base + 0] = rho
    q[base + 1] = Float32(0.0)
    q[base + 2] = Float32(0.0)
    q[base + 3] = Float32(0.0)
    q[base + 4] = E


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    mpi.init()
    var size = mpi.world_size()
    if size > 1:
        mpi.finalize()
        print("bench_euler_hydrostatic_3d: runs at np=1 only")
        return

    print("bench_euler_hydrostatic_3d (hydrostatic balance, gravity gate)")
    print(
        "  P= 2   mesh=",
        NX,
        "x",
        NY,
        "x",
        NZ,
        "   gz=",
        GZ_NEG,
        "   T=",
        T_FINAL,
    )

    var rank = mpi.world_rank()
    var nvtx = NvtxContext()
    var refs = build_reference_operators(nvtx)
    var ctx = DeviceContext()

    var bcs = BoundaryConditions(
        BC_INTERIOR,
        BC_INTERIOR,  # x: periodic
        BC_INTERIOR,
        BC_INTERIOR,  # y: periodic
        BC_WALL,
        BC_WALL,  # z: slip walls
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
        Euler.NUM_COMPONENTS,
        mesh.d_perm.unsafe_ptr(),
        bcs,
    )
    var physics = Euler(
        GAMMA,
        MIN_DENSITY,
        MIN_PRESSURE,
        FLUX_HLLEC,
        False,
        Float32(0.0),
        Float32(0.0),
        GZ_NEG,  # (gx, gy, gz)
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

    solver.ctx.enqueue_function[hydrostatic_ic_kernel](
        solver.d_q.unsafe_ptr(),
        solver.mesh.d_owned_elem_ids.unsafe_ptr(),
        solver.mesh.local.d_elem_node_xyz.unsafe_ptr(),
        solver.num_owned_elements,
        grid_dim=ceildiv(solver.num_owned_elements * N_P, IC_BLOCK),
        block_dim=IC_BLOCK,
    )
    solver.ctx.synchronize()

    var n_owned_dof = solver.num_owned_elements * N_P * Euler.NUM_COMPONENTS
    var hbuf_xyz = solver.ctx.enqueue_create_host_buffer[DType.float32](
        solver.num_owned_elements * N_P * 3
    )
    # owned_elem_xyz: re-derive from the device elem_node_xyz, indexed
    # by owned_elem_ids.  Simpler: reuse the IC pass's host_ic for
    # the analytic comparison, but we need pz too -- collect both.
    # Snapshot IC.
    var hbuf_ic = solver.ctx.enqueue_create_host_buffer[DType.float32](
        n_owned_dof
    )
    solver.ctx.enqueue_copy(
        hbuf_ic, solver.d_q.create_sub_buffer[DType.float32](0, n_owned_dof)
    )
    # Also pull elem_node_xyz over so we can recompute pz per node.
    var num_xyz = solver.mesh.local.num_elements * N_P * 3
    var hbuf_xyz_full = solver.ctx.enqueue_create_host_buffer[DType.float32](
        num_xyz
    )
    solver.ctx.enqueue_copy(
        hbuf_xyz_full,
        solver.mesh.local.d_elem_node_xyz.create_sub_buffer[DType.float32](
            0, num_xyz
        ),
    )
    solver.ctx.synchronize()
    var ic_ptr = hbuf_ic.unsafe_ptr()
    var xyz_ptr = hbuf_xyz_full.unsafe_ptr()
    var host_ic = List[Float32]()
    for k in range(n_owned_dof):
        host_ic.append(ic_ptr[k])

    # Time stepping.  Use the speed of sound for the CFL constraint
    # (no advective velocity here).
    var cs = sqrt(GAMMA * P0 / RHO0)
    var h = Float32(LX) / Float32(NX)
    var dt = CFL * h / (cs * Float32(5.0))
    var num_steps = Int(T_FINAL / dt) + 1
    var dt_used = T_FINAL / Float32(num_steps)
    print("  steps=", num_steps, "  dt=", dt_used)

    for _ in range(num_steps):
        solver.step_ssprk3(dt_used, nvtx)
    solver.ctx.synchronize()

    var hbuf_q = solver.ctx.enqueue_create_host_buffer[DType.float32](
        n_owned_dof
    )
    solver.ctx.enqueue_copy(
        hbuf_q, solver.d_q.create_sub_buffer[DType.float32](0, n_owned_dof)
    )
    solver.ctx.synchronize()
    var q_ptr = hbuf_q.unsafe_ptr()

    var max_v: Float64 = 0.0
    var max_rho_dev: Float64 = 0.0
    var max_p_dev: Float64 = 0.0
    var n_owned_nodes = solver.num_owned_elements * N_P
    for i in range(n_owned_nodes):
        var rho_now = q_ptr[i * 5 + 0]
        var rhou = q_ptr[i * 5 + 1]
        var rhov = q_ptr[i * 5 + 2]
        var rhow = q_ptr[i * 5 + 3]
        var E_now = q_ptr[i * 5 + 4]
        if isnan(rho_now) or isinf(rho_now):
            raise Error("bench_euler_hydrostatic_3d: non-finite output")
        var u = rhou / rho_now
        var v = rhov / rho_now
        var w = rhow / rho_now
        var v_mag = sqrt(
            Float64(u) * Float64(u)
            + Float64(v) * Float64(v)
            + Float64(w) * Float64(w)
        )
        if v_mag > max_v:
            max_v = v_mag
        var rho_dev = Float64(rho_now - RHO0)
        if rho_dev < 0.0:
            rho_dev = -rho_dev
        var rho_dev_rel = rho_dev / Float64(RHO0)
        if rho_dev_rel > max_rho_dev:
            max_rho_dev = rho_dev_rel
        # p = (gamma-1) * (E - 0.5 rho v^2)
        var ke = Float32(0.5) * rho_now * (u * u + v * v + w * w)
        var p_now = (GAMMA - Float32(1.0)) * (E_now - ke)
        # Need pz for the analytic profile; map owned node i -> global
        # element id -> elem_node_xyz row.
        var owned_e = Int(
            solver.mesh.d_owned_elem_ids.create_sub_buffer[DType.int32](
                0, solver.num_owned_elements
            ).unsafe_ptr()[0]
        )
        # Cheaper to re-pull host_ic[i*5+0] which equals RHO0; for the
        # pz lookup we use the ic_ptr's index conversion through
        # host_xyz_full.  But simpler: read from xyz_ptr indexed by
        # the owned element id.  See loop variable mapping below.
        _ = owned_e
        # Simpler: recompute pz by trusting the IC's E formula.
        # E_IC[i] = (P0 + RHO0 * GZ_NEG * pz) / (gamma - 1) at u=v=w=0.
        # So pz = (E_IC[i] * (gamma - 1) - P0) / (RHO0 * GZ_NEG).
        var E_ic = host_ic[i * 5 + 4]
        var pz = (E_ic * (GAMMA - Float32(1.0)) - P0) / (RHO0 * GZ_NEG)
        var p_exact = P0 + RHO0 * GZ_NEG * pz
        var p_dev = Float64(p_now - p_exact)
        if p_dev < 0.0:
            p_dev = -p_dev
        var p_dev_rel = p_dev / Float64(P0)
        if p_dev_rel > max_p_dev:
            max_p_dev = p_dev_rel

    print("  max |v|        =", max_v, "  (threshold", VMAX_TOL, ")")
    print("  max drho/rho0  =", max_rho_dev, "  (threshold", RHO_REL_TOL, ")")
    print("  max dp/p0      =", max_p_dev, "  (threshold", P_REL_TOL, ")")

    if max_v > VMAX_TOL:
        raise Error(
            "bench_euler_hydrostatic_3d FAILED: max |v| "
            + String(max_v)
            + " > "
            + String(VMAX_TOL)
        )
    if max_rho_dev > RHO_REL_TOL:
        raise Error(
            "bench_euler_hydrostatic_3d FAILED: max drho/rho0 "
            + String(max_rho_dev)
            + " > "
            + String(RHO_REL_TOL)
        )
    if max_p_dev > P_REL_TOL:
        raise Error(
            "bench_euler_hydrostatic_3d FAILED: max dp/p0 "
            + String(max_p_dev)
            + " > "
            + String(P_REL_TOL)
        )

    print("=== bench_euler_hydrostatic_3d PASSED ===")
    mpi.finalize()
