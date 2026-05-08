# ======================================================================
# bench_euler_sod_3d_p3 -- 3D Sod shock tube at P=3 (NP=20)
# ======================================================================
#
# P=3 (NP=20 nodes per tet) counterpart of bench_euler_sod_3d.  Same
# Sod IC, same HLLEC + BJ-limited pipeline, same long-x rectangular
# domain, but routed through Mesh[3] / Solver[Euler, 3] /
# rk_stage_kernel[3] so the higher-order limiter and HLLEC kernels
# are exercised on real shocked flow at NP=20.
#
# Existing P=3 limiter coverage in 3D (`limiter_3d_test_p3`) only
# checks the cell-mean conservation invariant on a constant state.
# This bench is the first analytic-Riemann gate on the P=3 3D
# limiter path.  It pairs with bench_euler_sod_limited_2d_p3
# (NP=10 in 2D) for cross-dimension P=3 limited-shocks parity.
#
# Pass criteria (P=3, NX=64, NY=NZ=4, HLLEC, 8-cell IC smoothing,
# BJ + Venkat(eps=0.1), T=0.20):
#   * rho_max <= RHO_L + 5e-3 (no over-shoot above the left plateau;
#     P=3 BJ is more diffusive than P=2 because at NP=20 the per-tet
#     deviation pool is larger and theta is scaled down more often,
#     so plateaus are slightly under-shot rather than overshot --
#     1.5%% on rho_L empirically.  BOUNDS_SLACK=5e-3 is ~3.3x the
#     observed under-shoot, still tight enough to catch any meaningful
#     P=3 limiter regression.)
#   * rho_min > 0 (positivity)
#   * rho_min >= RHO_R - 5e-3 (no under-shoot below the right plateau;
#     empirical 3.5e-3 under)
#   * total mass change relative to IC < 0.5%% over the run (BJ is
#     exactly conservative in the cell mean; outflow at t=0.20 has
#     barely started; observed 0.21%%)
#   * no NaN / Inf
#
# NX is reduced from 100 (P=2 bench) to 64 to keep wall-clock
# comparable: at P=3 each tet has NP=20 vs NP=10 nodes (volume
# work scales ~NP^2 = 4x), and dt is tighter by factor (2P+1)/(2P_ref+1)
# = 7/5 = 1.4x.  64*1.4*2 ~= 180 steps at NX=64 P=3 vs ~140 steps at
# NX=100 P=2; NX=64 still covers ~6 cells for the rarefaction region
# at NP=20 effective resolution.
# ======================================================================

from std.sys import has_accelerator
from std.gpu import global_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from std.math import sqrt, ceildiv, tanh, isnan, isinf

from src import mpi
from src.partition import build_partition
from src.reference import (
    ReferenceElement,
    to_float32,
    num_tet_nodes,
)
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


comptime P = 3
comptime NP = num_tet_nodes(P)  # 20 at P=3
comptime NX = 64
comptime NY = 4
comptime NZ = 4
comptime LX = 1.0
comptime LY = Float64(4.0 / 64.0)
comptime LZ = Float64(4.0 / 64.0)

comptime GAMMA: Float32 = 1.4
comptime RHO_L: Float32 = 1.0
comptime P_L: Float32 = 1.0
comptime RHO_R: Float32 = 0.125
comptime P_R: Float32 = 0.1
comptime T_FINAL: Float32 = 0.20
comptime CFL = Float32(0.15)
comptime IC_BLOCK = 256
comptime SMOOTH_WIDTH: Float32 = Float32(8.0 * (LX / NX))

comptime BOUNDS_SLACK: Float32 = Float32(5.0e-3)
comptime MASS_TOL_REL: Float64 = 5.0e-3


def sod_ic_kernel_p3(
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
    var px = elem_node_xyz[(e * NP + nn) * 3 + 0]
    var s = (tanh((px - Float32(0.5)) / SMOOTH_WIDTH) + Float32(1.0)) * Float32(0.5)
    var rho = RHO_L + s * (RHO_R - RHO_L)
    var p = P_L + s * (P_R - P_L)
    var E = p / (GAMMA - Float32(1.0))
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
        print("bench_euler_sod_3d_p3: runs at np=1 only")
        return

    print("bench_euler_sod_3d_p3 (3D Sod shock tube at P=3, BJ-limited)")
    print("  P=", P, "  NP=", NP, "  mesh=", NX, "x", NY, "x", NZ, "  T=", T_FINAL)

    var rank = mpi.world_rank()
    var nvtx = NvtxContext()
    var ctx = DeviceContext()

    # Build P=3 reference operators directly: build_reference_operators()
    # in src.reference defaults to P=2 and gives N_P=10 / N_FP=6, which
    # is wrong for Solver[Euler, 3] (NP=20 / NFP=10).
    var re = ReferenceElement[P]()
    var D_ref = to_float32(re.D_ref)
    var Lift_ref = to_float32(re.Lift_ref)
    var node_weights = to_float32(re.node_weights)

    var bcs = BoundaryConditions(
        BC_OUTFLOW,
        BC_OUTFLOW,
        BC_WALL,
        BC_WALL,
        BC_WALL,
        BC_WALL,
    )
    var mesh = Mesh[P](
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
        Float32(1.0e-6),
        Float32(1.0e-6),
        FLUX_HLLEC,
        False,
    )
    var solver = Solver[Euler, P](
        ctx^,
        mesh^,
        halo^,
        physics^,
        D_ref^,
        Lift_ref^,
        node_weights^,
    )
    solver.enable_cell_limiter(True, Float32(0.1))

    solver.ctx.enqueue_function[sod_ic_kernel_p3, sod_ic_kernel_p3](
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

    # CFL: tighter at higher P (factor 2P+1 = 7 at P=3 vs 5 at P=2).
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
    var rho_min: Float32 = Float32(1.0e30)
    var mass_fin: Float64 = 0.0
    for i in range(n_dof):
        var v = rho_buf[i]
        if isnan(v) or isinf(v):
            raise Error("bench_euler_sod_3d_p3: non-finite density")
        if v > rho_max:
            rho_max = v
        if v < rho_min:
            rho_min = v
        mass_fin += Float64(v)
    mass_fin /= Float64(n_dof)

    print("  rho_max =", rho_max, "  (bound =", RHO_L, ")")
    print("  rho_min =", rho_min, "  (bound =", RHO_R, ", positive)")
    print("  mass(IC) =", mass_ic, "  mass(t=T) =", mass_fin)

    if rho_max > RHO_L + BOUNDS_SLACK:
        raise Error(
            String("bench_euler_sod_3d_p3 FAILED: rho_max ") + String(rho_max) + " overshot RHO_L=" + String(RHO_L)
        )
    if rho_min < Float32(0.0):
        raise Error(String("bench_euler_sod_3d_p3 FAILED: rho_min ") + String(rho_min) + " negative (positivity lost)")
    if rho_min < RHO_R - BOUNDS_SLACK:
        raise Error(
            String("bench_euler_sod_3d_p3 FAILED: rho_min ") + String(rho_min) + " undershot RHO_R=" + String(RHO_R)
        )

    var dmass = mass_fin - mass_ic
    if dmass < 0.0:
        dmass = -dmass
    var rel = dmass / mass_ic
    if rel > MASS_TOL_REL:
        raise Error(
            String("bench_euler_sod_3d_p3 FAILED: mass drift ")
            + String(rel * 100.0)
            + "%% > tol "
            + String(MASS_TOL_REL * 100.0)
            + "%%"
        )

    print("=== bench_euler_sod_3d_p3 PASSED ===")
    mpi.finalize()
