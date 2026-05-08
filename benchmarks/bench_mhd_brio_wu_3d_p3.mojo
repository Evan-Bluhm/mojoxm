# ======================================================================
# bench_mhd_brio_wu_3d_p3 -- 3D Brio-Wu MHD shock tube at P=3
# ======================================================================
#
# P=3 (NP=20 nodes per tet) counterpart of bench_mhd_brio_wu_3d.  Same
# canonical Brio-Wu IC + boundary conditions, but routed through
# Mesh[3] / Solver[IdealMHD, 3] / NP=20 so the limiter, GLM kernels,
# and Rusanov flux are exercised at higher order on the seven-wave
# MHD Riemann fan.
#
# Pairs with bench_euler_sod_3d_p3 (NP=20 hydro shock-tube) and
# bench_mhd_alfven_3d_p3 (NP=20 smooth MHD wave) to bring 3D shocked
# MHD up to NP=20 coverage.  Closes a real gap: existing P=3 limiter
# coverage in 3D MHD only includes smooth-flow gates; this is the
# first analytic Riemann-style gate at P=3 on the ideal-MHD path.
#
# Tolerances loosen relative to the P=2 bench because the BJ limiter
# at NP=20 is more aggressive (per-tet deviation pool grows with NP)
# AND the P=3 dispersion error on the seven-wave fan is non-trivial
# at NX=120 (a lower mesh count is required to keep wall time
# reasonable; the work scales ~NP^2 = 4x and dt is tighter by 7/5 =
# 1.4x).
#
# Pass criteria (P=3, NX=120, NY=NZ=4, BJ + Venkat eps=0.1, T=0.08):
#   * rho in [0.03, 1.6] (slightly looser than the P=2 1.5 upper to
#     accommodate the higher-order overshoot at the fast-rarefaction
#     head)
#   * |Bx - 0.75| < 0.6 (looser than P=2's 0.5: at NP=20 the
#     transverse flux asymmetry from the tet split + GLM transport
#     produces somewhat larger |Bx| drift across the shock fronts)
#   * |psi| < 2.5 (P=3 limiter pushes back less aggressively against
#     the GLM-driven psi peaks)
#   * mass conservation drift < 2.5% over T=0.08
#   * no NaN / Inf
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
from src.mhd import IdealMHD
from src.nvtx import NvtxContext


comptime P = 3
comptime NP = num_tet_nodes(P)  # 20 at P=3
comptime NX = 120
comptime NY = 4
comptime NZ = 4
comptime LX = 1.0
comptime LY = Float64(4.0 / 120.0)
comptime LZ = Float64(4.0 / 120.0)

comptime GAMMA: Float32 = 2.0
comptime RHO_L: Float32 = 1.0
comptime P_L: Float32 = 1.0
comptime BX: Float32 = 0.75
comptime BY_L: Float32 = 1.0
comptime RHO_R: Float32 = 0.125
comptime P_R: Float32 = 0.1
comptime BY_R: Float32 = -1.0

comptime T_FINAL: Float32 = 0.08
comptime CFL = Float32(0.10)
comptime IC_BLOCK = 256
comptime SMOOTH_WIDTH: Float32 = Float32(8.0 * (LX / NX))
comptime MIN_DENSITY: Float32 = 1.0e-6
comptime MIN_PRESSURE: Float32 = 1.0e-6
comptime C_H: Float32 = 3.0
comptime ALPHA_D: Float32 = 0.1

comptime BX_TOL: Float32 = Float32(0.6)
comptime PSI_TOL: Float32 = Float32(2.5)
comptime RHO_MIN_OK: Float32 = Float32(0.03)
comptime RHO_MAX_OK: Float32 = Float32(1.6)
comptime MASS_TOL_REL: Float64 = 2.5e-2


def brio_wu_ic_kernel_p3(
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
    var by = BY_L + s * (BY_R - BY_L)
    var bx = BX
    var bz = Float32(0.0)
    var E = p / (GAMMA - Float32(1.0)) + Float32(0.5) * (bx * bx + by * by + bz * bz)
    var base = (e * NP + nn) * 9
    q[base + 0] = rho
    q[base + 1] = Float32(0.0)
    q[base + 2] = Float32(0.0)
    q[base + 3] = Float32(0.0)
    q[base + 4] = E
    q[base + 5] = bx
    q[base + 6] = by
    q[base + 7] = bz
    q[base + 8] = Float32(0.0)


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    mpi.init()
    var size = mpi.world_size()
    if size > 1:
        mpi.finalize()
        print("bench_mhd_brio_wu_3d_p3: runs at np=1 only")
        return

    print("bench_mhd_brio_wu_3d_p3 (3D Brio-Wu MHD at P=3)")
    print("  P=", P, "  NP=", NP, "  mesh=", NX, "x", NY, "x", NZ, "  T=", T_FINAL)

    var rank = mpi.world_rank()
    var nvtx = NvtxContext()
    var ctx = DeviceContext()

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
        IdealMHD.NUM_COMPONENTS,
        mesh.d_perm.unsafe_ptr(),
        bcs,
    )
    var physics = IdealMHD(
        GAMMA,
        MIN_DENSITY,
        MIN_PRESSURE,
        C_H,
        ALPHA_D,
    )
    var solver = Solver[IdealMHD, P](
        ctx^,
        mesh^,
        halo^,
        physics^,
        D_ref^,
        Lift_ref^,
        node_weights^,
    )
    solver.enable_cell_limiter(True, Float32(0.1))

    solver.ctx.enqueue_function[brio_wu_ic_kernel_p3, brio_wu_ic_kernel_p3](
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
    var scratch = List[Float32]()
    for _ in range(n_dof):
        scratch.append(Float32(0.0))
    solver.download_owned_component(0, scratch, nvtx)
    var mass_ic: Float64 = 0.0
    for i in range(n_dof):
        mass_ic += Float64(scratch[i])
    mass_ic /= Float64(n_dof)

    var cs2 = GAMMA * P_L / RHO_L
    var cA2 = (BX * BX + BY_L * BY_L) / RHO_L
    var c_fast = sqrt(cs2 + cA2)
    var h = Float32(LX) / Float32(NX)
    # CFL: factor 2P+1 = 7 at P=3.
    var dt_est = CFL * h / (c_fast * Float32(2 * P + 1))
    var num_steps = Int(T_FINAL / dt_est) + 1
    var dt = T_FINAL / Float32(num_steps)
    print("  steps=", num_steps, "  dt=", dt, "  c_fast=", c_fast)

    for _ in range(num_steps):
        solver.step_ssprk3(dt, nvtx)
    solver.ctx.synchronize()

    solver.download_owned_component(5, scratch, nvtx)
    var bx_max_err: Float32 = 0.0
    for i in range(n_dof):
        var v = scratch[i]
        if isnan(v) or isinf(v):
            raise Error("bench_mhd_brio_wu_3d_p3: Bx non-finite")
        var d = v - BX
        if d < 0.0:
            d = -d
        if d > bx_max_err:
            bx_max_err = d

    solver.download_owned_component(8, scratch, nvtx)
    var psi_max: Float32 = 0.0
    for i in range(n_dof):
        var v = scratch[i]
        if isnan(v) or isinf(v):
            raise Error("bench_mhd_brio_wu_3d_p3: psi non-finite")
        var a = v if v >= Float32(0.0) else -v
        if a > psi_max:
            psi_max = a

    solver.download_owned_component(0, scratch, nvtx)
    var rho_max: Float32 = Float32(-1.0e30)
    var rho_min: Float32 = Float32(1.0e30)
    var mass_fin: Float64 = 0.0
    for i in range(n_dof):
        var v = scratch[i]
        if isnan(v) or isinf(v):
            raise Error("bench_mhd_brio_wu_3d_p3: rho non-finite")
        if v > rho_max:
            rho_max = v
        if v < rho_min:
            rho_min = v
        mass_fin += Float64(v)
    mass_fin /= Float64(n_dof)

    print("  Bx max |dev|  =", bx_max_err, "  (tol=", BX_TOL, ")")
    print("  psi max |val| =", psi_max, "  (tol=", PSI_TOL, ")")
    print("  rho in [", rho_min, ",", rho_max, "]")
    print("  mass(IC)=", mass_ic, "  mass(t=T)=", mass_fin)

    if bx_max_err > BX_TOL:
        raise Error(
            String("bench_mhd_brio_wu_3d_p3 FAILED: Bx drifted ") + String(bx_max_err) + " > tol " + String(BX_TOL)
        )
    if psi_max > PSI_TOL:
        raise Error(String("bench_mhd_brio_wu_3d_p3 FAILED: psi ") + String(psi_max) + " > tol " + String(PSI_TOL))
    if rho_min < RHO_MIN_OK:
        raise Error(String("bench_mhd_brio_wu_3d_p3 FAILED: rho_min ") + String(rho_min) + " < " + String(RHO_MIN_OK))
    if rho_max > RHO_MAX_OK:
        raise Error(String("bench_mhd_brio_wu_3d_p3 FAILED: rho_max ") + String(rho_max) + " > " + String(RHO_MAX_OK))

    var dmass = mass_fin - mass_ic
    if dmass < 0.0:
        dmass = -dmass
    var rel = dmass / mass_ic
    if rel > MASS_TOL_REL:
        raise Error(
            String("bench_mhd_brio_wu_3d_p3 FAILED: mass drift ")
            + String(rel * 100.0)
            + "%% > tol "
            + String(MASS_TOL_REL * 100.0)
            + "%%"
        )

    print("=== bench_mhd_brio_wu_3d_p3 PASSED ===")
    mpi.finalize()
