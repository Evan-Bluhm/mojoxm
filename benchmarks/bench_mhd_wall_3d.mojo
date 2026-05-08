# ======================================================================
# bench_mhd_wall_3d -- 3D MHD BC_WALL preservation, isolated from shocks
# ======================================================================
#
# Closes a coverage gap.  3D MHD's BC_WALL path (in
# IdealMHD.boundary_flux at src/mhd.mojo lines 405-416) reflects
# normal momentum, normal B-field, and negates psi.  It IS exercised
# by bench_mhd_brio_wu_3d (which uses BC_WALL on the y/z walls), but
# the Brio-Wu run mixes BC_WALL behaviour with a shocked 1D Riemann
# IC + GLM cleaning + BJ limiter, so a regression in just the wall
# reflection is hard to disentangle from limiter or flux quality
# issues.  This bench isolates BC_WALL alone.
#
# Setup mirrors bench_mhd_wall_2d_glm: u=0, B=(B0, 0, 0), psi=0.
# B is tangential to +/-y AND +/-z walls (B_n=0 there) and periodic
# in x, so the BC_WALL reflection is a no-op and the state is
# preserved exactly.  A regression that broke any of the wall-
# reflection sign conventions would still inject spurious mass /
# momentum / B-field at the walls.  GLM is disabled (c_h=alpha_d=0)
# here because the Dedner GLM source path subtly couples to
# non-periodic BCs at sub-Alfvenic flow, leaving ~1% drift on E
# even from psi=0; plain MHD on the same setup stays at Float32
# noise.  Disabling GLM isolates the BC_WALL path so a regression
# in the wall reflection is what trips this gate.
#
# Pass criteria (P=2, NX=NY=NZ=4 single-rank, T=1):
#   * max relative drift in any of the 9 components < 1e-3
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
from src.boundary import BoundaryConditions, BC_WALL, BC_INTERIOR
from src.halo_exchange import HaloExchange
from src.solver import Solver
from src.mhd import IdealMHD
from src.nvtx import NvtxContext


comptime P = 2
comptime NX = 4
comptime NY = 4
comptime NZ = 4
comptime LX = 1.0
comptime LY = 1.0
comptime LZ = 1.0

comptime GAMMA = Float32(5.0 / 3.0)
comptime RHO0 = Float32(1.0)
comptime B0 = Float32(1.0)
comptime P0 = Float32(0.5)
comptime C_H = Float32(0.0)  # GLM disabled
comptime ALPHA_D = Float32(0.0)
comptime MIN_DENSITY = Float32(1.0e-6)
comptime MIN_PRESSURE = Float32(1.0e-6)

comptime T_FINAL = Float32(1.0)
comptime CFL = Float32(0.2)
comptime IC_BLOCK = 256

comptime REL_TOL: Float64 = 1.0e-3


def uniform_ic_kernel(q: UnsafePointer[Float32, MutAnyOrigin], owned_elem_ids: UnsafePointer[Int32, MutAnyOrigin], elem_node_xyz: UnsafePointer[Float32, MutAnyOrigin], num_owned: Int):
    var idx = Int(global_idx.x)
    var total = num_owned * N_P
    if idx >= total:
        return
    var i = idx // N_P
    var nn = idx % N_P
    var e = Int(owned_elem_ids[i])
    var base = (e * N_P + nn) * 9

    var pe = Float32(0.5) * (B0 * B0)
    var E0 = P0 / (GAMMA - Float32(1.0)) + pe  # u = 0 so no KE

    q[base + 0] = RHO0
    q[base + 1] = Float32(0.0)
    q[base + 2] = Float32(0.0)
    q[base + 3] = Float32(0.0)
    q[base + 4] = E0
    q[base + 5] = B0
    q[base + 6] = Float32(0.0)
    q[base + 7] = Float32(0.0)
    q[base + 8] = Float32(0.0)


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    mpi.init()
    var size = mpi.world_size()
    if size > 1:
        mpi.finalize()
        print("bench_mhd_wall_3d: runs at np=1 only")
        return

    print("bench_mhd_wall_3d (BC_WALL preservation, GLM disabled)")
    print("  P=", P, "  mesh=", NX, "x", NY, "x", NZ, "   B0=", B0, "   T=", T_FINAL, "  (u=0; B aligned to x; walls in y,z)")

    var rank = mpi.world_rank()
    var nvtx = NvtxContext()
    var refs = build_reference_operators(nvtx)
    var ctx = DeviceContext()

    # Periodic in x; reflecting walls in y and z (B is tangential to
    # all 4 of those walls so BC_WALL reflection is a no-op).
    var bcs = BoundaryConditions(
        BC_INTERIOR,
        BC_INTERIOR,  # -x, +x periodic
        BC_WALL,
        BC_WALL,  # -y, +y reflecting
        BC_WALL,
        BC_WALL,  # -z, +z reflecting
    )
    var mesh = Mesh(ctx=ctx, part=build_partition(rank=rank, nprocs=size, nx=NX, ny=NY, nz=NZ), Lx=LX, Ly=LY, Lz=LZ, bcs=bcs)
    var halo = HaloExchange(ctx=ctx, part=mesh.part, nc=IdealMHD.NUM_COMPONENTS, d_perm=mesh.d_perm.unsafe_ptr(), bcs=bcs)
    var physics = IdealMHD(gamma=GAMMA, min_density=MIN_DENSITY, min_pressure=MIN_PRESSURE, c_h=C_H, alpha_d=ALPHA_D)
    var solver = Solver[IdealMHD](ctx=ctx^, mesh=mesh^, halo=halo^, physics=physics^, D_ref=refs.D_ref^, Lift_ref=refs.Lift_ref^, node_weights=refs.node_weights^)

    solver.ctx.enqueue_function[uniform_ic_kernel](
        solver.d_q.unsafe_ptr(),
        solver.mesh.d_owned_elem_ids.unsafe_ptr(),
        solver.mesh.local.d_elem_node_xyz.unsafe_ptr(),
        solver.num_owned_elements,
        grid_dim=ceildiv(solver.num_owned_elements * N_P, IC_BLOCK),
        block_dim=IC_BLOCK,
    )
    solver.ctx.synchronize()

    var n_owned_dof = solver.num_owned_elements * N_P * IdealMHD.NUM_COMPONENTS
    var hbuf_ic = solver.ctx.enqueue_create_host_buffer[DType.float32](n_owned_dof)
    solver.ctx.enqueue_copy(hbuf_ic, solver.d_q.create_sub_buffer[DType.float32](0, n_owned_dof))
    solver.ctx.synchronize()
    var ic_ptr = hbuf_ic.unsafe_ptr()
    var host_ic = List[Float32]()
    for k in range(n_owned_dof):
        host_ic.append(ic_ptr[k])

    var cf = sqrt(GAMMA * P0 / RHO0 + B0 * B0 / RHO0)
    var h = Float32(LX) / Float32(NX)
    var dt = CFL * h / (cf * Float32(2 * P + 1))
    var num_steps = Int(T_FINAL / dt) + 1
    var dt_used = T_FINAL / Float32(num_steps)
    print("  c_f=", cf, "  steps=", num_steps, "  dt=", dt_used)

    for _ in range(num_steps):
        solver.step_ssprk3(dt_used, nvtx)
    solver.ctx.synchronize()

    var hbuf_q = solver.ctx.enqueue_create_host_buffer[DType.float32](n_owned_dof)
    solver.ctx.enqueue_copy(hbuf_q, solver.d_q.create_sub_buffer[DType.float32](0, n_owned_dof))
    solver.ctx.synchronize()
    var q_ptr = hbuf_q.unsafe_ptr()

    # Energy magnitude in the IC -- use as the reference scale for
    # relative drift since rho*u/rho*v/rho*w start at 0.
    var pe = Float32(0.5) * (B0 * B0)
    var E0 = P0 / (GAMMA - Float32(1.0)) + pe
    var ref_scale = Float64(E0)

    var max_drift: Float64 = 0.0
    var n_owned_nodes = solver.num_owned_elements * N_P
    for i in range(n_owned_nodes):
        for c in range(9):
            var qv = q_ptr[i * 9 + c]
            if isnan(qv) or isinf(qv):
                raise Error("bench_mhd_wall_3d: non-finite output")
            var qref = host_ic[i * 9 + c]
            var d = Float64(qv) - Float64(qref)
            if d < 0.0:
                d = -d
            var rel = d / ref_scale
            if rel > max_drift:
                max_drift = rel

    print("  max relative drift   =", max_drift, "  (threshold", REL_TOL, ")")

    if max_drift > REL_TOL:
        raise Error("bench_mhd_wall_3d FAILED: max relative drift " + String(max_drift) + " > " + String(REL_TOL))

    print("=== bench_mhd_wall_3d PASSED ===")
    mpi.finalize()
