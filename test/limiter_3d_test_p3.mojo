# ======================================================================
# limiter_3d_test_p3 -- P=3 (NP=20) 3D BJ slope limiter coverage
# ======================================================================
#
# P=3 counterpart of limiter_3d_test.  Same two checks
# (constant-state preservation + cell-mean conservation under
# perturbation) but at NP=20 nodes per tet.
#
# At P=2 the mass-matrix-weighted node weights are
# (-1/20, ..., 1/5, ...).  At P=3 they're a different distribution
# entirely, so a regression in either ReferenceElement[3].node_weights
# or the BJ limiter's downstream use of them at NP=20 wouldn't be
# caught by the P=2 test.  This test exercises that path.
# ======================================================================

from std.sys import has_accelerator
from std.gpu import global_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from std.math import ceildiv

from src import mpi
from src.partition import build_partition
from src.reference import ReferenceElement, to_float32, num_tet_nodes
from src.mesh import Mesh
from src.boundary import BoundaryConditions
from src.halo_exchange import HaloExchange
from src.solver import Solver
from src.euler import Euler, FLUX_HLLEC
from src.nvtx import NvtxContext


comptime P = 3
comptime NP = num_tet_nodes(P)         # 20 at P=3
comptime NC = 5

comptime NX = 4
comptime NY = 4
comptime NZ = 4
comptime LX = 1.0
comptime LY = 1.0
comptime LZ = 1.0
comptime IC_BLOCK = 256

comptime CONST_TOL: Float32 = Float32(1.0e-6)
comptime MEAN_TOL:  Float32 = Float32(1.0e-5)


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
    q[base + 0] = Float32(1.0)
    q[base + 1] = Float32(0.5)
    q[base + 2] = Float32(-0.25)
    q[base + 3] = Float32(0.1)
    q[base + 4] = Float32(2.5)


def fill_perturbed_kernel(
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
    var jitter = Float32(0.3) if (nn % 2) == 0 else Float32(-0.3)
    q[base + 0] = Float32(1.0) + jitter
    q[base + 1] = Float32(0.0)
    q[base + 2] = Float32(0.0)
    q[base + 3] = Float32(0.0)
    q[base + 4] = Float32(2.5)


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    mpi.init()
    var size = mpi.world_size()
    if size > 1:
        mpi.finalize()
        print("limiter_3d_test_p3: runs at np=1 only")
        return

    print("limiter_3d_test_p3: P=3 / NP=20 BJ limiter conservation")

    var nvtx = NvtxContext()
    var ctx = DeviceContext()
    var re = ReferenceElement[P]()
    var D_ref = to_float32(re.D_ref)
    var Lift_ref = to_float32(re.Lift_ref)
    var node_weights = to_float32(re.node_weights)
    var node_weights_host = List[Float32]()
    for k in range(NP):
        node_weights_host.append(node_weights[k])

    var mesh = Mesh[P](
        ctx, build_partition(0, 1, NX, NY, NZ), LX, LY, LZ,
        BoundaryConditions.periodic(),
    )
    var halo = HaloExchange(
        ctx, mesh.part, Euler.NUM_COMPONENTS,
        mesh.d_perm.unsafe_ptr(),
    )
    var physics = Euler(
        Float32(1.4), Float32(1.0e-6), Float32(1.0e-6),
        FLUX_HLLEC, False,
    )
    var solver = Solver[Euler, P](
        ctx^, mesh^, halo^, physics^, D_ref^, Lift_ref^, node_weights^,
    )
    solver.enable_cell_limiter(True, Float32(0.0))   # raw BJ, eps=0

    var num_owned = solver.num_owned_elements
    var n_dof = num_owned * NP

    # --- Test 1: constant state -> limiter is a no-op.
    solver.ctx.enqueue_function[fill_constant_kernel, fill_constant_kernel](
        solver.d_q.unsafe_ptr(),
        solver.mesh.d_owned_elem_ids.unsafe_ptr(),
        num_owned,
        grid_dim=ceildiv(num_owned * NP, IC_BLOCK),
        block_dim=IC_BLOCK,
    )
    solver.ctx.synchronize()

    var rho_buf = List[Float32]()
    for _ in range(n_dof):
        rho_buf.append(Float32(0.0))
    solver._launch_cell_limiter(solver.d_q.unsafe_ptr())
    solver.ctx.synchronize()
    solver.download_owned_component(0, rho_buf, nvtx)

    var max_err_const: Float32 = 0.0
    for k in range(n_dof):
        var d = rho_buf[k] - Float32(1.0)
        var ad = d if d >= Float32(0.0) else -d
        if ad > max_err_const: max_err_const = ad
    print("  constant-state max |rho - 1| =", max_err_const,
          "  (tol", CONST_TOL, ")")
    if max_err_const > CONST_TOL:
        raise Error(
            "limiter_3d_test_p3 FAILED: constant state altered by limiter "
            "by " + String(max_err_const)
        )

    # --- Test 2: perturbed state -> verify cell-mean preservation.
    solver.ctx.enqueue_function[fill_perturbed_kernel, fill_perturbed_kernel](
        solver.d_q.unsafe_ptr(),
        solver.mesh.d_owned_elem_ids.unsafe_ptr(),
        num_owned,
        grid_dim=ceildiv(num_owned * NP, IC_BLOCK),
        block_dim=IC_BLOCK,
    )
    solver.ctx.synchronize()

    solver.download_owned_component(0, rho_buf, nvtx)
    var means_before = List[Float32]()
    for e in range(num_owned):
        var s: Float32 = 0.0
        for nn in range(NP):
            s += node_weights_host[nn] * rho_buf[e * NP + nn]
        means_before.append(s)

    solver._launch_cell_limiter(solver.d_q.unsafe_ptr())
    solver.ctx.synchronize()
    solver.download_owned_component(0, rho_buf, nvtx)

    var max_mean_drift: Float32 = 0.0
    for e in range(num_owned):
        var s: Float32 = 0.0
        for nn in range(NP):
            s += node_weights_host[nn] * rho_buf[e * NP + nn]
        var d = s - means_before[e]
        var ad = d if d >= Float32(0.0) else -d
        if ad > max_mean_drift: max_mean_drift = ad
    print("  perturbed cell-mean max drift =", max_mean_drift,
          "  (tol", MEAN_TOL, ")")
    if max_mean_drift > MEAN_TOL:
        raise Error(
            "limiter_3d_test_p3 FAILED: cell mean drifted by "
            + String(max_mean_drift)
            + " under BJ limiter (conservation broken)"
        )

    var max_change: Float32 = 0.0
    for e in range(num_owned):
        for nn in range(NP):
            var jitter = Float32(0.3) if (nn % 2) == 0 else Float32(-0.3)
            var orig = Float32(1.0) + jitter
            var d = rho_buf[e * NP + nn] - orig
            var ad = d if d >= Float32(0.0) else -d
            if ad > max_change: max_change = ad
    print("  perturbed q max change vs IC =", max_change,
          "  (proves limiter fired)")
    if max_change < Float32(0.01):
        raise Error(
            "limiter_3d_test_p3 FAILED: limiter appears not to have "
            "modified q (max_change=" + String(max_change)
            + "); test setup may be invalid"
        )

    print("=== limiter_3d_test_p3 PASSED ===")
    mpi.finalize()
