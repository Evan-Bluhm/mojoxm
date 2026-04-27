# ======================================================================
# bench_euler_sod_3d_p4 -- 3D Sod shock tube at P=4 (NP=35)
# ======================================================================
#
# P=4 (NP=35 nodes per tet) counterpart of bench_euler_sod_3d_p3.
# Same Sod IC, same HLLEC + BJ-limited pipeline, same long-x
# rectangular domain, but routed through Mesh[4] / Solver[Euler, 4]
# / rk_stage_kernel[4] so the higher-order limiter and HLLEC kernels
# are exercised on real shocked flow at NP=35.
#
# Pairs with bench_euler_sod_limited_2d_p4 (NP=15 in 2D) for cross-
# dimension P=4 limited-shocks parity.  Closes the 3D shocked-flow
# P-parity gap: smooth Euler at P=4 was validated by
# `bench_euler_smooth_wave_3d_p4`, but the 3D BJ limiter + HLLEC
# under hard shocks at NP=35 had not been gated.
#
# Pass criteria (P=4, NX=40, NY=NZ=4, HLLEC, 8-cell IC smoothing,
# BJ + Venkat(eps=0.1), T=0.20):
#   * rho_max <= RHO_L + 1e-2 (no over-shoot above the left plateau;
#     P=4 BJ is more diffusive than P=3 because at NP=35 the per-tet
#     deviation pool grows with NP -- expected plateau under-shoot
#     ~3-4%; 1e-2 = 8x the empirical floor)
#   * rho_min > 0 (positivity)
#   * rho_min >= RHO_R - 1e-2 (no under-shoot below the right plateau)
#   * total mass change relative to IC < 0.5% over the run
#   * no NaN / Inf
#
# NX=40 (vs P=3's NX=64): NP=35 in 3D means each tet has 1.75x more
# nodes than at NP=20, and dt is tighter by factor (2P+1)/(2P_p3+1)
# = 9/7 = 1.29x.  NX=40 keeps total work tractable while still
# covering the rarefaction region at NP=35 effective resolution.
# ======================================================================

from std.sys import has_accelerator
from std.gpu import global_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from std.math import sqrt, ceildiv, tanh, isnan, isinf

from src import mpi
from src.partition import build_partition
from src.reference import (
    ReferenceElement, to_float32, num_tet_nodes,
)
from src.mesh import Mesh
from src.boundary import (
    BoundaryConditions, BC_INTERIOR, BC_WALL, BC_OUTFLOW,
)
from src.halo_exchange import HaloExchange
from src.solver import Solver
from src.euler import Euler, FLUX_HLLEC
from src.nvtx import NvtxContext


comptime P = 4
comptime NP = num_tet_nodes(P)   # 35 at P=4
comptime NX = 40
comptime NY = 4
comptime NZ = 4
comptime LX = 1.0
comptime LY = Float64(4.0 / 40.0)
comptime LZ = Float64(4.0 / 40.0)

comptime GAMMA: Float32 = 1.4
comptime RHO_L: Float32 = 1.0
comptime P_L:   Float32 = 1.0
comptime RHO_R: Float32 = 0.125
comptime P_R:   Float32 = 0.1
comptime T_FINAL: Float32 = 0.20
comptime CFL = Float32(0.10)
comptime IC_BLOCK = 256
comptime SMOOTH_WIDTH: Float32 = Float32(8.0 * (LX / NX))

comptime BOUNDS_SLACK: Float32 = Float32(1.0e-2)
comptime MASS_TOL_REL: Float64 = 5.0e-3


def sod_ic_kernel_p4(
    q: UnsafePointer[Float32, MutAnyOrigin],
    owned_elem_ids: UnsafePointer[Int32, MutAnyOrigin],
    elem_node_xyz:  UnsafePointer[Float32, MutAnyOrigin],
    num_owned: Int,
):
    var idx = Int(global_idx.x)
    var total = num_owned * NP
    if idx >= total:
        return
    var i = idx // NP
    var nn = idx % NP
    var e = Int(owned_elem_ids[i])
    var px = elem_node_xyz[(e * NP + nn) * 3 + 0]
    var s = (tanh((px - Float32(0.5)) / SMOOTH_WIDTH) + Float32(1.0)) \
            * Float32(0.5)
    var rho = RHO_L + s * (RHO_R - RHO_L)
    var p   = P_L   + s * (P_R - P_L)
    var E   = p / (GAMMA - Float32(1.0))
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
        print("bench_euler_sod_3d_p4: runs at np=1 only")
        return

    print("bench_euler_sod_3d_p4 (3D Sod shock tube at P=4, BJ-limited)")
    print("  P=", P, "  NP=", NP, "  mesh=", NX, "x", NY, "x", NZ,
          "  T=", T_FINAL)

    var rank = mpi.world_rank()
    var nvtx = NvtxContext()
    var ctx = DeviceContext()

    var re = ReferenceElement[P]()
    var D_ref = to_float32(re.D_ref)
    var Lift_ref = to_float32(re.Lift_ref)
    var node_weights = to_float32(re.node_weights)

    var bcs = BoundaryConditions(
        BC_OUTFLOW, BC_OUTFLOW,
        BC_WALL,    BC_WALL,
        BC_WALL,    BC_WALL,
    )
    var mesh = Mesh[P](
        ctx, build_partition(rank, size, NX, NY, NZ), LX, LY, LZ, bcs,
    )
    var halo = HaloExchange(
        ctx, mesh.part, Euler.NUM_COMPONENTS,
        mesh.d_perm.unsafe_ptr(), bcs,
    )
    var physics = Euler(
        GAMMA, Float32(1.0e-6), Float32(1.0e-6),
        FLUX_HLLEC, False,
    )
    var solver = Solver[Euler, P](
        ctx^, mesh^, halo^, physics^, D_ref^, Lift_ref^, node_weights^,
    )
    solver.enable_cell_limiter(True, Float32(0.1))

    solver.ctx.enqueue_function[sod_ic_kernel_p4, sod_ic_kernel_p4](
        solver.d_q.unsafe_ptr(),
        solver.mesh.d_owned_elem_ids.unsafe_ptr(),
        solver.mesh.local.d_elem_node_xyz.unsafe_ptr(),
        solver.num_owned_elements,
        grid_dim=ceildiv(solver.num_owned_elements * NP, IC_BLOCK),
        block_dim=IC_BLOCK,
    )
    solver.ctx.synchronize()

    var n_owned = solver.num_owned_elements
    var n_dof = n_owned * NP
    var rho_buf = List[Float32]()
    for _ in range(n_dof):
        rho_buf.append(Float32(0.0))
    solver.download_owned_component(0, rho_buf, nvtx)
    var mass_ic: Float64 = 0.0
    for i in range(n_dof):
        mass_ic += Float64(rho_buf[i])
    mass_ic /= Float64(n_dof)

    # CFL: factor 2P+1 = 9 at P=4.
    var h = Float32(LX) / Float32(NX)
    var c_L = sqrt(GAMMA * P_L / RHO_L)
    var wave = Float32(2.0) * c_L
    var dt_est = CFL * h / (wave * Float32(2 * P + 1))
    var num_steps = Int(T_FINAL / dt_est) + 1
    var dt = T_FINAL / Float32(num_steps)
    print("  steps=", num_steps, "  dt=", dt)

    for _ in range(num_steps):
        solver.step_ssprk3(dt, nvtx)
    solver.ctx.synchronize()

    solver.download_owned_component(0, rho_buf, nvtx)
    var rho_max: Float32 = Float32(-1.0e30)
    var rho_min: Float32 = Float32( 1.0e30)
    var mass_fin: Float64 = 0.0
    for i in range(n_dof):
        var v = rho_buf[i]
        if isnan(v) or isinf(v):
            raise Error("bench_euler_sod_3d_p4: non-finite density")
        if v > rho_max: rho_max = v
        if v < rho_min: rho_min = v
        mass_fin += Float64(v)
    mass_fin /= Float64(n_dof)

    print("  rho_max =", rho_max, "  (bound =", RHO_L, ")")
    print("  rho_min =", rho_min, "  (bound =", RHO_R, ", positive)")
    print("  mass(IC) =", mass_ic, "  mass(t=T) =", mass_fin)

    if rho_max > RHO_L + BOUNDS_SLACK:
        raise Error(
            String("bench_euler_sod_3d_p4 FAILED: rho_max ")
            + String(rho_max) + " overshot RHO_L=" + String(RHO_L)
        )
    if rho_min < Float32(0.0):
        raise Error(
            String("bench_euler_sod_3d_p4 FAILED: rho_min ")
            + String(rho_min) + " negative (positivity lost)"
        )
    if rho_min < RHO_R - BOUNDS_SLACK:
        raise Error(
            String("bench_euler_sod_3d_p4 FAILED: rho_min ")
            + String(rho_min) + " undershot RHO_R=" + String(RHO_R)
        )

    var dmass = mass_fin - mass_ic
    if dmass < 0.0: dmass = -dmass
    var rel = dmass / mass_ic
    if rel > MASS_TOL_REL:
        raise Error(
            String("bench_euler_sod_3d_p4 FAILED: mass drift ")
            + String(rel * 100.0) + "%% > tol "
            + String(MASS_TOL_REL * 100.0) + "%%"
        )

    print("=== bench_euler_sod_3d_p4 PASSED ===")
    mpi.finalize()
