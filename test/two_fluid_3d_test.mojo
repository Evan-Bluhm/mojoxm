# ======================================================================
# two_fluid_3d_test -- 3D FiveMomentTwoFluid constant-state preservation
# ======================================================================
#
# Constant-state preservation through Solver[FiveMomentTwoFluid, P]
# on a periodic 3D mesh, parameterised over P in {2, 3, 4, 5}.
# 17-component state (electrons, ions, Maxwell EM, GLM psi).  IC:
# charge-balanced uniform plasma (rho_e and rho_i set so net charge
# = 0), zero drift, E = B = 0, psi = 0.  In this rest state the
# source terms are all zero (no Lorentz force, no current, no
# charge density driving E) and uniform fluxes cancel, so SSPRK3
# must leave the state unchanged to Float32 roundoff.
#
# This is the most-components physics in the suite (NC=17) and
# exercises all the source-term arithmetic at every supported NP
# (10/20/35/56 = P=2/3/4/5).  A regression in the per-P comptime
# specialisation of the source-term kernel that broke at NC*NP=595
# (P=4) or NC*NP=952 (P=5) but worked at the smaller comptime
# configs would be caught here.
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
from src.two_fluid import FiveMomentTwoFluid
from src.nvtx import NvtxContext


comptime NC = 17

comptime NX = 4
comptime NY = 4
comptime NZ = 4
comptime LX = 1.0
comptime LY = 1.0
comptime LZ = 1.0
comptime IC_BLOCK = 256
comptime NUM_STEPS = 5

# Plasma parameters (matching bench_two_fluid_langmuir_3d).
comptime GAMMA_E: Float32 = Float32(5.0 / 3.0)
comptime GAMMA_I: Float32 = Float32(5.0 / 3.0)
comptime Q_E: Float32 = -1.0
comptime M_E: Float32 = 1.0
comptime Q_I: Float32 = 1.0
comptime M_I: Float32 = 25.0
comptime EPS0: Float32 = 1.0
comptime C_LIGHT: Float32 = 10.0
comptime C_H: Float32 = 0.0  # GLM off (psi=0 already in IC)
comptime ALPHA_D: Float32 = 0.0
comptime MIN_DENSITY: Float32 = 1.0e-6
comptime MIN_PRESSURE: Float32 = 1.0e-8

# Charge-balanced rest state: q_e * n_e + q_i * n_i = 0 ->
# n_e = n_i = N0 (since Q_E = -Q_I in our normalisation).
comptime N0: Float32 = 1.0
comptime P_E0: Float32 = 0.01
comptime P_I0: Float32 = 0.01
comptime CONST_TOL: Float32 = Float32(1.0e-4)


def fill_constant_kernel[P: Int](q: UnsafePointer[Float32, MutAnyOrigin], owned_elem_ids: UnsafePointer[Int32, MutAnyOrigin], num_owned: Int):
    comptime NP = num_tet_nodes(P)
    var idx = Int(global_idx.x)
    var total = num_owned * NP
    if idx >= total:
        return
    var i = idx // NP
    var nn = idx % NP
    var e = Int(owned_elem_ids[i])
    var base = (e * NP + nn) * NC
    var rho_e = M_E * N0
    var rho_i = M_I * N0
    var E_e = P_E0 / (GAMMA_E - Float32(1.0))  # u=0, no kinetic
    var E_i = P_I0 / (GAMMA_I - Float32(1.0))
    # Electrons (0..4)
    q[base + 0] = rho_e
    q[base + 1] = Float32(0.0)
    q[base + 2] = Float32(0.0)
    q[base + 3] = Float32(0.0)
    q[base + 4] = E_e
    # Ions (5..9)
    q[base + 5] = rho_i
    q[base + 6] = Float32(0.0)
    q[base + 7] = Float32(0.0)
    q[base + 8] = Float32(0.0)
    q[base + 9] = E_i
    # Maxwell E (10..12)
    q[base + 10] = Float32(0.0)
    q[base + 11] = Float32(0.0)
    q[base + 12] = Float32(0.0)
    # Maxwell B (13..15)
    q[base + 13] = Float32(0.0)
    q[base + 14] = Float32(0.0)
    q[base + 15] = Float32(0.0)
    # GLM psi (16)
    q[base + 16] = Float32(0.0)


def check[P: Int](mut nvtx: NvtxContext) raises:
    print("  P=", P)
    comptime NP = num_tet_nodes(P)
    var ctx = DeviceContext()
    var re = ReferenceElement[P]()
    var D_ref = to_float32(re.D_ref)
    var Lift_ref = to_float32(re.Lift_ref)
    var node_weights = to_float32(re.node_weights)

    var mesh = Mesh[P](ctx, build_partition(0, 1, NX, NY, NZ), LX, LY, LZ, BoundaryConditions.periodic())
    var halo = HaloExchange(ctx, mesh.part, FiveMomentTwoFluid.NUM_COMPONENTS, mesh.d_perm.unsafe_ptr())
    var physics = FiveMomentTwoFluid(GAMMA_E, GAMMA_I, Q_E, M_E, Q_I, M_I, EPS0, C_LIGHT, C_H, ALPHA_D, MIN_DENSITY, MIN_PRESSURE)
    var solver = Solver[FiveMomentTwoFluid, P](ctx^, mesh^, halo^, physics^, D_ref^, Lift_ref^, node_weights^)

    var num_owned = solver.num_owned_elements
    var n_dof = num_owned * NP

    comptime fill_kernel = fill_constant_kernel[P]
    solver.ctx.enqueue_function[fill_kernel, fill_kernel](solver.d_q.unsafe_ptr(), solver.mesh.d_owned_elem_ids.unsafe_ptr(), num_owned, grid_dim=ceildiv(num_owned * NP, IC_BLOCK), block_dim=IC_BLOCK)
    solver.ctx.synchronize()

    var rho_e = M_E * N0
    var rho_i = M_I * N0
    var E_e = P_E0 / (GAMMA_E - Float32(1.0))
    var E_i = P_I0 / (GAMMA_I - Float32(1.0))
    var ic_vals = List[Float32]()
    ic_vals.append(rho_e)
    ic_vals.append(Float32(0.0))
    ic_vals.append(Float32(0.0))
    ic_vals.append(Float32(0.0))
    ic_vals.append(E_e)
    ic_vals.append(rho_i)
    ic_vals.append(Float32(0.0))
    ic_vals.append(Float32(0.0))
    ic_vals.append(Float32(0.0))
    ic_vals.append(E_i)
    for _ in range(7):
        ic_vals.append(Float32(0.0))  # E, B, psi all 0

    # Wave speed: max of c_light (Maxwell) and sqrt(gamma*p/rho) (fluids).
    var cs_e = sqrt(GAMMA_E * P_E0 / rho_e)
    var cs_i = sqrt(GAMMA_I * P_I0 / rho_i)
    var max_c = C_LIGHT
    if cs_e > max_c:
        max_c = cs_e
    if cs_i > max_c:
        max_c = cs_i
    var h_cell = Float32(LX) / Float32(NX)
    var dt = Float32(0.05) * h_cell / (max_c * Float32(2 * P + 1))

    for _ in range(NUM_STEPS):
        solver.step_ssprk3(dt, nvtx)
    solver.ctx.synchronize()

    var scratch = List[Float32]()
    for _ in range(n_dof):
        scratch.append(Float32(0.0))
    var max_err: Float32 = 0.0
    for c_idx in range(NC):
        solver.download_owned_component(c_idx, scratch, nvtx)
        for k in range(n_dof):
            var v = scratch[k]
            if isnan(v) or isinf(v):
                raise Error("two_fluid_3d_test P=" + String(P) + ": non-finite at component " + String(c_idx))
            var d = v - ic_vals[c_idx]
            var ad = d if d >= Float32(0.0) else -d
            if ad > max_err:
                max_err = ad

    print("    max |q - IC| over", NUM_STEPS, "steps =", max_err, "  (tol", CONST_TOL, ")")
    if max_err > CONST_TOL:
        raise Error("two_fluid_3d_test P=" + String(P) + " FAILED: constant state shifted by " + String(max_err) + " over " + String(NUM_STEPS) + " steps")


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    mpi.init()
    var size = mpi.world_size()
    if size > 1:
        mpi.finalize()
        print("two_fluid_3d_test: runs at np=1 only")
        return
    print("two_fluid_3d_test: 3D FiveMomentTwoFluid constant-state, P=2..5")
    var nvtx = NvtxContext()
    check[2](nvtx)
    check[3](nvtx)
    check[4](nvtx)
    check[5](nvtx)
    print("=== two_fluid_3d_test PASSED ===")
    mpi.finalize()
