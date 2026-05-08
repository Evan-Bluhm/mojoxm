# ======================================================================
# euler_3d_test -- 3D Euler constant-state preservation
# ======================================================================
#
# Constant-state preservation through Solver[Euler, P] on a periodic
# 3D mesh.  Uniform (rho, u, v, w, E) state -> volume flux divergence
# cancels exactly, face fluxes cancel pairwise, several SSPRK3 steps
# leave the state unchanged to Float32 roundoff.
#
# Tests the 3D Kuhn-tet Euler kernel through HLLEC face flux (the
# most-commonly-used variant in benchmarks).
#
# Parameterised over P in {2, 3, 4, 5} so a P-specific bug in the
# comptime-templated rk_stage_kernel (e.g. shared-mem index that
# broke at NP=20 / 35 / 56 but happened to work at NP=10) gets
# caught at test-quick latency rather than only by bench-p5.
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
from src.euler import Euler, FLUX_HLLEC
from src.nvtx import NvtxContext


comptime NC = 5

comptime NX = 4
comptime NY = 4
comptime NZ = 4
comptime LX = 1.0
comptime LY = 1.0
comptime LZ = 1.0
comptime IC_BLOCK = 256
comptime NUM_STEPS = 5

comptime GAMMA: Float32 = 1.4
comptime RHO0: Float32 = 1.5
comptime U0: Float32 = 0.4
comptime V0: Float32 = -0.2
comptime W0: Float32 = 0.3
comptime P0: Float32 = 0.7
comptime CONST_TOL: Float32 = Float32(1.0e-4)


def fill_constant_kernel[
    P: Int
](
    q: UnsafePointer[Float32, MutAnyOrigin],
    owned_elem_ids: UnsafePointer[Int32, MutAnyOrigin],
    num_owned: Int,
):
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
    var E = P0 / (GAMMA - Float32(1.0)) + ke
    q[base + 0] = RHO0
    q[base + 1] = RHO0 * U0
    q[base + 2] = RHO0 * V0
    q[base + 3] = RHO0 * W0
    q[base + 4] = E


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
        Euler.NUM_COMPONENTS,
        mesh.d_perm.unsafe_ptr(),
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

    var ke = Float32(0.5) * RHO0 * (U0 * U0 + V0 * V0 + W0 * W0)
    var E0 = P0 / (GAMMA - Float32(1.0)) + ke
    var ic_vals = List[Float32]()
    ic_vals.append(RHO0)
    ic_vals.append(RHO0 * U0)
    ic_vals.append(RHO0 * V0)
    ic_vals.append(RHO0 * W0)
    ic_vals.append(E0)

    var c_s = sqrt(GAMMA * P0 / RHO0)
    var speed = sqrt(U0 * U0 + V0 * V0 + W0 * W0) + c_s
    var h = Float32(LX) / Float32(NX)
    var dt = Float32(0.1) * h / (speed * Float32(2 * P + 1))

    for _ in range(NUM_STEPS):
        solver.step_ssprk3(dt, nvtx)
    solver.ctx.synchronize()

    var scratch = List[Float32]()
    for _ in range(n_dof):
        scratch.append(Float32(0.0))
    var max_err: Float32 = 0.0
    for c in range(NC):
        solver.download_owned_component(c, scratch, nvtx)
        for k in range(n_dof):
            var v = scratch[k]
            if isnan(v) or isinf(v):
                raise Error(
                    "euler_3d_test P="
                    + String(P)
                    + ": non-finite at component "
                    + String(c)
                )
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
            "euler_3d_test P="
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
        print("euler_3d_test: runs at np=1 only")
        return
    print("euler_3d_test: 3D Euler constant-state preservation (HLLEC), P=2..5")
    var nvtx = NvtxContext()
    check[2](nvtx)
    check[3](nvtx)
    check[4](nvtx)
    check[5](nvtx)
    print("=== euler_3d_test PASSED ===")
    mpi.finalize()
