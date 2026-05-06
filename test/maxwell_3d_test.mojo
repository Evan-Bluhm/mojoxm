# ======================================================================
# maxwell_3d_test -- 3D Maxwell constant-state preservation
# ======================================================================
#
# Constant-state preservation through Solver[Maxwell, P] on a periodic
# 3D mesh.  Uniform (Ex, Ey, Ez, Bx, By, Bz) with J = M = 0 -- the
# Maxwell flux on a uniform field is identically zero, so several
# SSPRK3 steps must leave the state unchanged to Float32 roundoff.
#
# Catches sign / direction bugs in the Maxwell flux kernel that would
# inject spurious wave activity from a uniform field (e.g. a misrouted
# curl term or a normal-direction sign flip on the Faraday equation).
#
# Parameterised over P in {2, 3, 4, 5} so a P-specific bug in the
# comptime-templated rk_stage_kernel (e.g. a shared-mem index that
# broke at NP=20 / 35 / 56 but happened to work at NP=10) gets
# caught at test-quick latency rather than only by bench-p5.
# ======================================================================

from std.sys import has_accelerator
from std.gpu import global_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from std.math import ceildiv, isnan, isinf

from src import mpi
from src.partition import build_partition
from src.reference import ReferenceElement, to_float32, num_tet_nodes
from src.mesh import Mesh
from src.boundary import BoundaryConditions
from src.halo_exchange import HaloExchange
from src.solver import Solver
from src.maxwell import Maxwell
from src.nvtx import NvtxContext


comptime NC = 6

comptime NX = 4
comptime NY = 4
comptime NZ = 4
comptime LX = 1.0
comptime LY = 1.0
comptime LZ = 1.0
comptime IC_BLOCK = 256
comptime NUM_STEPS = 5

comptime C_LIGHT: Float32 = 1.0
comptime EX0: Float32 = 0.3
comptime EY0: Float32 = -0.2
comptime EZ0: Float32 = 0.5
comptime BX0: Float32 = 0.4
comptime BY0: Float32 = 0.1
comptime BZ0: Float32 = -0.6
comptime CONST_TOL: Float32 = Float32(1.0e-5)


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
    q[base + 0] = EX0
    q[base + 1] = EY0
    q[base + 2] = EZ0
    q[base + 3] = BX0
    q[base + 4] = BY0
    q[base + 5] = BZ0


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
        Maxwell.NUM_COMPONENTS,
        mesh.d_perm.unsafe_ptr(),
    )
    var physics = Maxwell(
        C_LIGHT,
        Float32(0.0),
        Float32(0.0),
        Float32(0.0),  # J = 0
        Float32(0.0),
        Float32(0.0),
        Float32(0.0),  # M = 0
    )
    var solver = Solver[Maxwell, P](
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

    var ic_vals = List[Float32]()
    ic_vals.append(EX0)
    ic_vals.append(EY0)
    ic_vals.append(EZ0)
    ic_vals.append(BX0)
    ic_vals.append(BY0)
    ic_vals.append(BZ0)

    # Maxwell wave speed = c_light.
    var h = Float32(LX) / Float32(NX)
    var dt = Float32(0.1) * h / (C_LIGHT * Float32(2 * P + 1))

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
                    "maxwell_3d_test P="
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
            "maxwell_3d_test P="
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
        print("maxwell_3d_test: runs at np=1 only")
        return
    print("maxwell_3d_test: 3D Maxwell constant-state preservation, P=2..5")

    var nvtx = NvtxContext()
    check[2](nvtx)
    check[3](nvtx)
    check[4](nvtx)
    check[5](nvtx)
    print("=== maxwell_3d_test PASSED ===")
    mpi.finalize()
