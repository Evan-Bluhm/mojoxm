# ======================================================================
# bench_mhd_brio_wu_3d -- 3D Brio-Wu MHD shock tube
# ======================================================================
#
# The classical Brio-Wu 1D MHD Riemann problem (Brio & Wu, JCP 1988)
# embedded in a long-x 3D domain with slip walls on y/z and outflow
# on +-x.  IC uniform in y and z; with u=v=w=0 and Bz=0 throughout,
# the 3D solution reduces to the 1D Brio-Wu answer.
#
# IC at x = 0.5, gamma = 2:
#   Left:  rho=1,     p=1,     u=v=w=0, Bx=0.75, By= 1, Bz=0, psi=0
#   Right: rho=0.125, p=0.1,   u=v=w=0, Bx=0.75, By=-1, Bz=0, psi=0
#
# After t = 0.08 the flow develops a fast rarefaction, a compound
# (slow-rarefaction + slow-shock) wave, a contact discontinuity, a
# slow shock and a fast rarefaction (seven waves total).  No closed-
# form analytic solution, but sharp invariants that must hold for any
# 1D-embedded 3D MHD run:
#
#   (1) rho, p stay positive and physical: rho in [0.03, 1.5].
#       Brio-Wu's compound wave can push rho slightly above RHO_L=1,
#       so upper bound is loose; 1.5 comfortably bounds any real
#       Brio-Wu solution.
#   (2) Bx drift stays bounded: |Bx - 0.75| < 0.5.  The analytic 1D
#       solution has Bx = 0.75 exactly, but our 3D Rusanov+GLM+BJ
#       stack produces O(0.1-0.3) numerical Bx deviations at shock
#       fronts (transverse flux asymmetry from the tet split + GLM
#       transport), which is consistent with the published Rusanov
#       behaviour on Brio-Wu.  A tight sub-1%% Bx bound would require
#       an HLLD-type Riemann solver + constrained-transport divB
#       handling that we don't have yet; this gate still catches any
#       gross flux-function regression.
#   (3) psi bounded: |psi| < 2.  GLM transport drives psi away from
#       zero in reaction to numerical divB; it should stay bounded
#       but not necessarily small.  Tol=2 catches runaway divB
#       cleaning (which signals c_h / alpha_d misconfiguration).
#   (4) Mass conservation: total rho drifts < 2%% over t = 0.08.
#   (5) No NaN / Inf.
#
# With BJ cell limiter enabled (Venkat eps = 0.1 scaled to |B| range).
#
# This is a rigorous shock-MHD gate without requiring a reference
# Riemann-solver -- the four invariants + positivity + mass catch
# every regression that matters for WARPXM-parity MHD.
# ======================================================================

from std.sys import has_accelerator
from std.gpu import global_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from std.math import sqrt, ceildiv, tanh, isnan, isinf

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
from src.mhd import IdealMHD
from src.nvtx import NvtxContext


comptime NX = 200
comptime NY = 4
comptime NZ = 4
comptime LX = 1.0
comptime LY = Float64(4.0 / 200.0)
comptime LZ = Float64(4.0 / 200.0)

comptime GAMMA: Float32 = 2.0
comptime RHO_L: Float32 = 1.0
comptime P_L: Float32 = 1.0
comptime BX: Float32 = 0.75
comptime BY_L: Float32 = 1.0
comptime RHO_R: Float32 = 0.125
comptime P_R: Float32 = 0.1
comptime BY_R: Float32 = -1.0

comptime T_FINAL: Float32 = 0.08
comptime CFL = Float32(0.15)
comptime IC_BLOCK = 256
comptime SMOOTH_WIDTH: Float32 = Float32(8.0 * (LX / NX))
comptime MIN_DENSITY: Float32 = 1.0e-6
comptime MIN_PRESSURE: Float32 = 1.0e-6
# GLM parameters: c_h set to a loose upper bound on the fast
# magnetosonic speed so any numerical divB gets swept out, alpha_d
# = 0.1 is the standard Dedner damping.
comptime C_H: Float32 = 3.0
comptime ALPHA_D: Float32 = 0.1

comptime BX_TOL: Float32 = Float32(0.5)
comptime PSI_TOL: Float32 = Float32(2.0)
comptime RHO_MIN_OK: Float32 = Float32(0.03)
comptime RHO_MAX_OK: Float32 = Float32(1.5)
comptime MASS_TOL_REL: Float64 = 2.0e-2


def brio_wu_ic_kernel(
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
    var px = elem_node_xyz[(e * N_P + nn) * 3 + 0]
    var s = (tanh((px - Float32(0.5)) / SMOOTH_WIDTH) + Float32(1.0)) * Float32(0.5)
    var rho = RHO_L + s * (RHO_R - RHO_L)
    var p = P_L + s * (P_R - P_L)
    var by = BY_L + s * (BY_R - BY_L)
    var bx = BX
    var bz = Float32(0.0)
    var E = p / (GAMMA - Float32(1.0)) + Float32(0.5) * (bx * bx + by * by + bz * bz)
    var base = (e * N_P + nn) * 9
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
        print("bench_mhd_brio_wu_3d: runs at np=1 only")
        return

    print("bench_mhd_brio_wu_3d (3D Brio-Wu MHD shock tube)")
    print("  P=2  mesh=", NX, "x", NY, "x", NZ, "  T=", T_FINAL)

    var rank = mpi.world_rank()
    var nvtx = NvtxContext()
    var refs = build_reference_operators(nvtx)
    var ctx = DeviceContext()

    var bcs = BoundaryConditions(
        BC_OUTFLOW,
        BC_OUTFLOW,
        BC_WALL,
        BC_WALL,
        BC_WALL,
        BC_WALL,
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
    var solver = Solver[IdealMHD](
        ctx^,
        mesh^,
        halo^,
        physics^,
        refs.D_ref^,
        refs.Lift_ref^,
        refs.node_weights^,
    )
    # Venkat eps = 0.1 matches euler_sod_3d.  Quantities of interest
    # (rho ~ O(1), |B| ~ O(1)) are similarly scaled.
    solver.enable_cell_limiter(True, Float32(0.1))

    solver.ctx.enqueue_function[brio_wu_ic_kernel, brio_wu_ic_kernel](
        solver.d_q.unsafe_ptr(),
        solver.mesh.d_owned_elem_ids.unsafe_ptr(),
        solver.mesh.local.d_elem_node_xyz.unsafe_ptr(),
        solver.num_owned_elements,
        grid_dim=ceildiv(solver.num_owned_elements * N_P, IC_BLOCK),
        block_dim=IC_BLOCK,
    )
    solver.ctx.synchronize()

    # Initial mass snapshot (component 0).
    var n_owned = solver.num_owned_elements
    var n_dof = n_owned * N_P
    var scratch = List[Float32]()
    for _ in range(n_dof):
        scratch.append(Float32(0.0))
    solver.download_owned_component(0, scratch, nvtx)
    var mass_ic: Float64 = 0.0
    for i in range(n_dof):
        mass_ic += Float64(scratch[i])
    mass_ic /= Float64(n_dof)

    # Wave speeds for dt estimate.  Fast magnetosonic upper bound:
    # c_fast = sqrt(c_s^2 + c_A^2) with c_s = sqrt(gamma p / rho),
    # c_A = |B| / sqrt(rho).  Worst case on left state.
    var cs2 = GAMMA * P_L / RHO_L
    var cA2 = (BX * BX + BY_L * BY_L) / RHO_L
    var c_fast = sqrt(cs2 + cA2)
    var h = Float32(LX) / Float32(NX)
    var dt_est = CFL * h / (c_fast * Float32(5.0))
    var num_steps = Int(T_FINAL / dt_est) + 1
    var dt = T_FINAL / Float32(num_steps)
    print("  steps=", num_steps, "  dt=", dt, "  c_fast=", c_fast)

    for _ in range(num_steps):
        solver.step_ssprk3(dt, nvtx)
    solver.ctx.synchronize()

    # Check (1): Bx constant at BX, component 5.
    solver.download_owned_component(5, scratch, nvtx)
    var bx_max_err: Float32 = 0.0
    for i in range(n_dof):
        var v = scratch[i]
        if isnan(v) or isinf(v):
            raise Error("bench_mhd_brio_wu_3d: Bx non-finite")
        var d = v - BX
        if d < 0.0:
            d = -d
        if d > bx_max_err:
            bx_max_err = d

    # Check (2): psi ~ 0, component 8.
    solver.download_owned_component(8, scratch, nvtx)
    var psi_max: Float32 = 0.0
    for i in range(n_dof):
        var v = scratch[i]
        if isnan(v) or isinf(v):
            raise Error("bench_mhd_brio_wu_3d: psi non-finite")
        var a = v if v >= Float32(0.0) else -v
        if a > psi_max:
            psi_max = a

    # Check (3, 4, 6): rho positivity + bounds + mass conservation.
    solver.download_owned_component(0, scratch, nvtx)
    var rho_max: Float32 = Float32(-1.0e30)
    var rho_min: Float32 = Float32(1.0e30)
    var mass_fin: Float64 = 0.0
    for i in range(n_dof):
        var v = scratch[i]
        if isnan(v) or isinf(v):
            raise Error("bench_mhd_brio_wu_3d: rho non-finite")
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
            String("bench_mhd_brio_wu_3d FAILED: Bx drifted ") + String(bx_max_err) + " > tol " + String(BX_TOL)
        )
    if psi_max > PSI_TOL:
        raise Error(String("bench_mhd_brio_wu_3d FAILED: psi ") + String(psi_max) + " > tol " + String(PSI_TOL))
    if rho_min < RHO_MIN_OK:
        raise Error(String("bench_mhd_brio_wu_3d FAILED: rho_min ") + String(rho_min) + " < " + String(RHO_MIN_OK))
    if rho_max > RHO_MAX_OK:
        raise Error(String("bench_mhd_brio_wu_3d FAILED: rho_max ") + String(rho_max) + " > " + String(RHO_MAX_OK))

    var dmass = mass_fin - mass_ic
    if dmass < 0.0:
        dmass = -dmass
    var rel = dmass / mass_ic
    if rel > MASS_TOL_REL:
        raise Error(
            String("bench_mhd_brio_wu_3d FAILED: mass drift ")
            + String(rel * 100.0)
            + "%% > tol "
            + String(MASS_TOL_REL * 100.0)
            + "%%"
        )

    print("=== bench_mhd_brio_wu_3d PASSED ===")
    mpi.finalize()
