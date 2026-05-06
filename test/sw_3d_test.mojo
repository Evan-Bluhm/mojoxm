# ======================================================================
# sw_3d_test -- 3D ShallowWater constant-state preservation
# ======================================================================
#
# Constant-state preservation through Solver[ShallowWater, 2] on a
# periodic 3D mesh.  ShallowWater is 2D-embedded-in-3D (F^z = 0
# everywhere); a uniform (h, h*u, h*v) state -> volume flux divergence
# cancels exactly and face fluxes cancel pairwise, so SSPRK3 must
# leave the state unchanged to Float32 roundoff.
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
from src.shallow_water import ShallowWater
from src.nvtx import NvtxContext


comptime P = 2
comptime NP = num_tet_nodes(P)
comptime NC = 3

comptime NX = 4
comptime NY = 4
comptime NZ = 2
comptime LX = 1.0
comptime LY = 1.0
comptime LZ = 0.1
comptime IC_BLOCK = 256
comptime NUM_STEPS = 5

comptime GRAVITY: Float32 = 1.0
comptime H_MIN: Float32 = 1.0e-6
comptime H0: Float32 = 1.5
comptime U0: Float32 = 0.4
comptime V0: Float32 = -0.2
comptime CONST_TOL: Float32 = Float32(1.0e-5)


def fill_constant_kernel(
    q: UnsafePointer[Float32, MutAnyOrigin],
    owned_elem_ids: UnsafePointer[Int32, MutAnyOrigin],
    num_owned: Int,
):
    var idx = Int(global_idx.x)
    var total = num_owned * NP
    if idx >= total:
        return
    var i = idx // NP
    var nn = idx % NP
    var e = Int(owned_elem_ids[i])
    var base = (e * NP + nn) * NC
    q[base + 0] = H0
    q[base + 1] = H0 * U0
    q[base + 2] = H0 * V0


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    mpi.init()
    var size = mpi.world_size()
    if size > 1:
        mpi.finalize()
        print("sw_3d_test: runs at np=1 only")
        return
    print("sw_3d_test: 3D ShallowWater constant-state preservation")

    var nvtx = NvtxContext()
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
        ShallowWater.NUM_COMPONENTS,
        mesh.d_perm.unsafe_ptr(),
    )
    var physics = ShallowWater(GRAVITY, H_MIN)
    var solver = Solver[ShallowWater, P](
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

    solver.ctx.enqueue_function[fill_constant_kernel, fill_constant_kernel](
        solver.d_q.unsafe_ptr(),
        solver.mesh.d_owned_elem_ids.unsafe_ptr(),
        num_owned,
        grid_dim=ceildiv(num_owned * NP, IC_BLOCK),
        block_dim=IC_BLOCK,
    )
    solver.ctx.synchronize()

    var ic_vals = List[Float32]()
    ic_vals.append(H0)
    ic_vals.append(H0 * U0)
    ic_vals.append(H0 * V0)

    # Wave speed: |u| + sqrt(g*h)
    var c = sqrt(GRAVITY * H0)
    var speed = sqrt(U0 * U0 + V0 * V0) + c
    var h_cell = Float32(LX) / Float32(NX)
    var dt = Float32(0.1) * h_cell / (speed * Float32(2 * P + 1))

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
                raise Error(
                    "sw_3d_test: non-finite at component " + String(c_idx)
                )
            var d = v - ic_vals[c_idx]
            var ad = d if d >= Float32(0.0) else -d
            if ad > max_err:
                max_err = ad

    print(
        "  max |q - IC| over",
        NUM_STEPS,
        "steps =",
        max_err,
        "  (tol",
        CONST_TOL,
        ")",
    )
    if max_err > CONST_TOL:
        raise Error(
            "sw_3d_test FAILED: constant state shifted by "
            + String(max_err)
            + " over "
            + String(NUM_STEPS)
            + " steps"
        )

    print("=== sw_3d_test PASSED ===")
    mpi.finalize()
