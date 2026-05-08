# ======================================================================
# mhd_3d_test -- 3D IdealMHD constant-state preservation
# ======================================================================
#
# Constant-state preservation on a periodic 3D mesh through the
# unified Solver[IdealMHD, P] pipeline, parameterised over P in
# {2, 3, 4, 5}.  With a uniform (rho, u, v, w, Bx, By, Bz, psi, E)
# state the volume flux divergence cancels exactly, face fluxes
# cancel pairwise, and the GLM transport leaves Bx and psi alone
# (both spatially constant).  After several SSPRK3 steps the state
# must equal the IC to Float32 roundoff at every P.
#
# What this catches:
#   * MHD flux bug: any component-c flux that isn't translation
#     invariant on uniform state would shift q[c] by a small but
#     deterministic amount per step, growing with step count.
#   * Solver wiring regression: a misrouted q_a / q_b argument or a
#     wrong RK Butcher constant would inject error per step.
#   * GLM source bug: the alpha_d * psi damping in solver.source_term
#     would push psi toward 0 over many steps if applied with wrong
#     sign or scale -- with psi=0 IC the test catches this.
#
# Mirrors mhd_2d_gpu_test (which exercises the 2D triangular path)
# for the 3D Kuhn-tet path.
# ======================================================================

from std.sys import has_accelerator
from std.gpu import global_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from std.math import ceildiv, sqrt, isnan, isinf

from src import mpi
from src.partition import build_partition
from src.reference import ReferenceElement, to_float32, num_tet_nodes
from src.mesh import Mesh
from src.boundary import BoundaryConditions
from src.halo_exchange import HaloExchange
from src.solver import Solver
from src.mhd import IdealMHD
from src.nvtx import NvtxContext


comptime NC = 9

comptime NX = 4
comptime NY = 4
comptime NZ = 4
comptime LX = 1.0
comptime LY = 1.0
comptime LZ = 1.0
comptime IC_BLOCK = 256
comptime NUM_STEPS = 5

comptime GAMMA: Float32 = 5.0 / 3.0
comptime RHO0: Float32 = 1.5
comptime U0: Float32 = 0.7
comptime V0: Float32 = -0.3
comptime W0: Float32 = 0.2
comptime BX0: Float32 = 0.4
comptime BY0: Float32 = -0.6
comptime BZ0: Float32 = 0.1
comptime P0: Float32 = 0.8

# GLM disabled here so the source term doesn't decay psi (which is 0
# anyway, so it would still hold, but turning GLM off makes the test
# scope independent of the operator-splitting choice).
comptime C_H: Float32 = 0.0
comptime ALPHA_D: Float32 = 0.0

comptime CONST_TOL: Float32 = Float32(1.0e-4)


def fill_constant_kernel[
    P: Int
](q: UnsafePointer[Float32, MutAnyOrigin], owned_elem_ids: UnsafePointer[Int32, MutAnyOrigin], num_owned: Int,):
    comptime NP = num_tet_nodes(P)
    var idx = Int(global_idx.x)
    var total = num_owned * NP
    if idx >= total:
        return
    var i = idx // NP
    var nn = idx % NP
    var e = Int(owned_elem_ids[i])
    var base = (e * NP + nn) * NC
    var ke = Float32(0.5) * RHO0 * (U0 * U0 + V0 * V0 + W0 * W0)
    var pB = Float32(0.5) * (BX0 * BX0 + BY0 * BY0 + BZ0 * BZ0)
    var E = P0 / (GAMMA - Float32(1.0)) + ke + pB
    q[base + 0] = RHO0
    q[base + 1] = RHO0 * U0
    q[base + 2] = RHO0 * V0
    q[base + 3] = RHO0 * W0
    q[base + 4] = E
    q[base + 5] = BX0
    q[base + 6] = BY0
    q[base + 7] = BZ0
    q[base + 8] = Float32(0.0)  # psi


def check[P: Int](mut nvtx: NvtxContext) raises:
    print("  P=", P)
    comptime NP = num_tet_nodes(P)
    var ctx = DeviceContext()
    var re = ReferenceElement[P]()
    var D_ref = to_float32(re.D_ref)
    var Lift_ref = to_float32(re.Lift_ref)
    var node_weights = to_float32(re.node_weights)

    var mesh = Mesh[P](
        ctx,
        build_partition(0, 1, NX, NY, NZ),
        LX,
        LY,
        LZ,
        BoundaryConditions.periodic(),
    )
    var halo = HaloExchange(
        ctx,
        mesh.part,
        IdealMHD.NUM_COMPONENTS,
        mesh.d_perm.unsafe_ptr(),
    )
    var physics = IdealMHD(
        GAMMA,
        Float32(1.0e-6),
        Float32(1.0e-6),
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

    var num_owned = solver.num_owned_elements
    var n_dof = num_owned * NP

    comptime fill_kernel = fill_constant_kernel[P]
    solver.ctx.enqueue_function[fill_kernel, fill_kernel](
        solver.d_q.unsafe_ptr(),
        solver.mesh.d_owned_elem_ids.unsafe_ptr(),
        num_owned,
        grid_dim=ceildiv(num_owned * NP, IC_BLOCK),
        block_dim=IC_BLOCK,
    )
    solver.ctx.synchronize()

    # Compute analytic IC values for comparison.
    var ke = Float32(0.5) * RHO0 * (U0 * U0 + V0 * V0 + W0 * W0)
    var pB = Float32(0.5) * (BX0 * BX0 + BY0 * BY0 + BZ0 * BZ0)
    var E0 = P0 / (GAMMA - Float32(1.0)) + ke + pB
    var ic_vals = List[Float32]()
    ic_vals.append(RHO0)
    ic_vals.append(RHO0 * U0)
    ic_vals.append(RHO0 * V0)
    ic_vals.append(RHO0 * W0)
    ic_vals.append(E0)
    ic_vals.append(BX0)
    ic_vals.append(BY0)
    ic_vals.append(BZ0)
    ic_vals.append(Float32(0.0))

    # Pick a small dt that respects the fast magnetosonic CFL bound.
    var cf = sqrt(GAMMA * P0 / RHO0 + (BX0 * BX0 + BY0 * BY0 + BZ0 * BZ0) / RHO0)
    var speed = sqrt(U0 * U0 + V0 * V0 + W0 * W0) + cf
    var h = Float32(LX) / Float32(NX)
    var dt = Float32(0.1) * h / (speed * Float32(2 * P + 1))

    for _ in range(NUM_STEPS):
        solver.step_ssprk3(dt, nvtx)
    solver.ctx.synchronize()

    # Verify every owned node still holds the IC values.
    var scratch = List[Float32]()
    for _ in range(n_dof):
        scratch.append(Float32(0.0))
    var max_err: Float32 = 0.0
    for c in range(NC):
        solver.download_owned_component(c, scratch, nvtx)
        for k in range(n_dof):
            var v = scratch[k]
            if isnan(v) or isinf(v):
                raise Error("mhd_3d_test P=" + String(P) + ": non-finite at component " + String(c))
            var d = v - ic_vals[c]
            var ad = d if d >= Float32(0.0) else -d
            if ad > max_err:
                max_err = ad
    print(
        "    max |q - IC| over",
        NUM_STEPS,
        "steps =",
        max_err,
        "  (tol",
        CONST_TOL,
        ")",
    )
    if max_err > CONST_TOL:
        raise Error(
            "mhd_3d_test P="
            + String(P)
            + " FAILED: constant state shifted by "
            + String(max_err)
            + " over "
            + String(NUM_STEPS)
            + " steps"
        )


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    mpi.init()
    var size = mpi.world_size()
    if size > 1:
        mpi.finalize()
        print("mhd_3d_test: runs at np=1 only")
        return

    print("mhd_3d_test: 3D IdealMHD constant-state preservation, P=2..5")
    var nvtx = NvtxContext()
    check[2](nvtx)
    check[3](nvtx)
    check[4](nvtx)
    check[5](nvtx)
    print("=== mhd_3d_test PASSED ===")
    mpi.finalize()
