# ======================================================================
# bench_euler_hydrostatic_3d_p4 -- 3D Euler gravity at P=4 (NP=35)
# ======================================================================
#
# P=4 (NP=35 nodes per tet) counterpart of bench_euler_hydrostatic_3d_p3.
# Same constant-density hydrostatic-balance setup -- gravity g =
# (0, 0, -|gz|), p(z) = p0 - rho0*|gz|*z, slip walls in z, periodic
# in x/y -- but routed through Mesh[4] / Solver[Euler, 4] /
# rk_stage_kernel[4] so the 3D Euler gravity source-term path is
# exercised at NP=35.  Mesh shrunk to 2x2x4 since per-element volume
# work scales with NP^2 and dt tightens to 1/(2P+1)=1/9 at P=4 vs
# 1/7 at P=3.
#
# Pass criteria (P=4, NX=NY=2, NZ=4, single-rank, T=1):
#   * max |v_mag| < 1e-3 (state remains at rest)
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
from src.reference import ReferenceElement, to_float32, num_tet_nodes
from src.mesh import Mesh
from src.boundary import BoundaryConditions, BC_INTERIOR, BC_WALL
from src.halo_exchange import HaloExchange
from src.solver import Solver
from src.euler import Euler, FLUX_HLLEC
from src.nvtx import NvtxContext


comptime P = 4
comptime NP = num_tet_nodes(P)
comptime NX = 2
comptime NY = 2
comptime NZ = 4
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
    var total = num_owned * NP
    if idx >= total:
        return
    var i = idx // NP
    var nn = idx % NP
    var e = Int(owned_elem_ids[i])
    var pz = elem_node_xyz[(e * NP + nn) * 3 + 2]
    var p_z = P0 + RHO0 * GZ_NEG * pz
    var rho = RHO0
    var E = p_z / (GAMMA - Float32(1.0))  # u = v = w = 0 so KE = 0
    var base = (e * NP + nn) * 5
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
        print("bench_euler_hydrostatic_3d_p4: runs at np=1 only")
        return

    print("bench_euler_hydrostatic_3d_p4 (hydrostatic balance at P=4, NP=35)")
    print(
        "  P=",
        P,
        "  NP=",
        NP,
        "  mesh=",
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
    var ctx = DeviceContext()

    var re = ReferenceElement[P]()
    var D_ref = to_float32(re.D_ref)
    var Lift_ref = to_float32(re.Lift_ref)
    var node_weights = to_float32(re.node_weights)

    var bcs = BoundaryConditions(
        BC_INTERIOR,
        BC_INTERIOR,  # x: periodic
        BC_INTERIOR,
        BC_INTERIOR,  # y: periodic
        BC_WALL,
        BC_WALL,  # z: slip walls
    )
    var mesh = Mesh[P](
        ctx=ctx,
        part=build_partition(rank=rank, nprocs=size, nx=NX, ny=NY, nz=NZ),
        Lx=LX,
        Ly=LY,
        Lz=LZ,
        bcs=bcs,
    )
    var halo = HaloExchange(
        ctx=ctx,
        part=mesh.part,
        nc=Euler.NUM_COMPONENTS,
        d_perm=mesh.d_perm.unsafe_ptr(),
        bcs=bcs,
    )
    var physics = Euler(
        gamma=GAMMA,
        min_density=MIN_DENSITY,
        min_pressure=MIN_PRESSURE,
        flux_type=FLUX_HLLEC,
        entropy_fix=False,
        gx=Float32(0.0),
        gy=Float32(0.0),
        gz=GZ_NEG,
    )
    var solver = Solver[Euler, P](
        ctx=ctx^,
        mesh=mesh^,
        halo=halo^,
        physics=physics^,
        D_ref=D_ref^,
        Lift_ref=Lift_ref^,
        node_weights=node_weights^,
    )

    solver.ctx.enqueue_function[hydrostatic_ic_kernel](
        solver.d_q.unsafe_ptr(),
        solver.mesh.d_owned_elem_ids.unsafe_ptr(),
        solver.mesh.local.d_elem_node_xyz.unsafe_ptr(),
        solver.num_owned_elements,
        grid_dim=ceildiv(solver.num_owned_elements * NP, IC_BLOCK),
        block_dim=IC_BLOCK,
    )
    solver.ctx.synchronize()

    var n_owned_dof = solver.num_owned_elements * NP * Euler.NUM_COMPONENTS
    var hbuf_ic = solver.ctx.enqueue_create_host_buffer[DType.float32](
        n_owned_dof
    )
    solver.ctx.enqueue_copy(
        hbuf_ic, solver.d_q.create_sub_buffer[DType.float32](0, n_owned_dof)
    )
    solver.ctx.synchronize()
    var ic_ptr = hbuf_ic.unsafe_ptr()
    var host_ic = List[Float32]()
    for k in range(n_owned_dof):
        host_ic.append(ic_ptr[k])

    var cs = sqrt(GAMMA * P0 / RHO0)
    var h = Float32(LX) / Float32(NX)
    var dt = CFL * h / (cs * Float32(2 * P + 1))
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
    var n_owned_nodes = solver.num_owned_elements * NP
    for i in range(n_owned_nodes):
        var rho_now = q_ptr[i * 5 + 0]
        var rhou = q_ptr[i * 5 + 1]
        var rhov = q_ptr[i * 5 + 2]
        var rhow = q_ptr[i * 5 + 3]
        var E_now = q_ptr[i * 5 + 4]
        if isnan(rho_now) or isinf(rho_now):
            raise Error("bench_euler_hydrostatic_3d_p4: non-finite output")
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
        var ke = Float32(0.5) * rho_now * (u * u + v * v + w * w)
        var p_now = (GAMMA - Float32(1.0)) * (E_now - ke)
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
            "bench_euler_hydrostatic_3d_p4 FAILED: max |v| "
            + String(max_v)
            + " > "
            + String(VMAX_TOL)
        )
    if max_rho_dev > RHO_REL_TOL:
        raise Error(
            "bench_euler_hydrostatic_3d_p4 FAILED: max drho/rho0 "
            + String(max_rho_dev)
            + " > "
            + String(RHO_REL_TOL)
        )
    if max_p_dev > P_REL_TOL:
        raise Error(
            "bench_euler_hydrostatic_3d_p4 FAILED: max dp/p0 "
            + String(max_p_dev)
            + " > "
            + String(P_REL_TOL)
        )

    print("=== bench_euler_hydrostatic_3d_p4 PASSED ===")
    mpi.finalize()
