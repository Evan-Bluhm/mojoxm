# ======================================================================
# bench_euler_smooth_wave_3d_p3 -- P=3 entropy-wave Euler convergence
# ======================================================================
#
# P=3 counterpart of bench_euler_smooth_wave_3d.  Same 3D entropy
# wave (uniform velocity + uniform pressure + sinusoidal density) on
# a triply-periodic [0, 1]^3 cube, but routed through Mesh[3] +
# Solver[Euler, 3] with NP = 20 nodes per tet.  Expected asymptotic
# rate is P+1 = 4 on smooth flow.
#
# Pairs with bench_advection_3d_p3 (single-component) to cover
# NP=20 operator correctness across single- and multi-component
# physics: a regression in mass-matrix inversion or quadrature
# weight computation would shift the convergence rate well below
# the asymptotic value here.
#
# Pass criteria (P=3, HLLEC, periodic, single-rank):
#   * rel L2(state) at N=12 < 5e-4 (Euler is well-behaved on
#     entropy waves; expected error ~1e-4 - 5e-4 at this resolution)
#   * monotone refinement N=6 -> 8 -> 12
#   * rate >= 2.5 between at least one consecutive pair
#     (theoretical 4; pre-asymptotic at coarse N)
#   * no NaN / Inf
# ======================================================================

from std.sys import has_accelerator
from std.gpu import global_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from std.math import sqrt, ceildiv, sin, log, isnan, isinf

from src import mpi
from src.partition import build_partition
from src.reference import (
    ReferenceElement, to_float32, num_tet_nodes,
)
from src.mesh import Mesh
from src.boundary import BoundaryConditions
from src.halo_exchange import HaloExchange
from src.solver import Solver
from src.euler import Euler, FLUX_HLLEC
from src.nvtx import NvtxContext


comptime P = 3
comptime NP = num_tet_nodes(P)
comptime NC = 5

comptime LX = 1.0
comptime LY = 1.0
comptime LZ = 1.0
comptime GAMMA: Float32 = 1.4
comptime RHO0:  Float32 = 1.0
comptime U0:    Float32 = 1.0
comptime V0:    Float32 = 1.0
comptime W0:    Float32 = 1.0
comptime P0:    Float32 = 1.0
comptime AMPLITUDE: Float32 = 0.1
comptime T_FINAL: Float32 = 1.0
comptime CFL = Float32(0.1)
comptime IC_BLOCK = 256
comptime PI_F: Float32 = 3.14159265358979323846

comptime L2_MAX_REL_AT_12: Float64 = 5.0e-4
comptime RATE_MIN: Float64 = 2.5


def entropy_wave_ic_kernel(
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
    var py = elem_node_xyz[(e * NP + nn) * 3 + 1]
    var pz = elem_node_xyz[(e * NP + nn) * 3 + 2]

    var k = Float32(2.0) * PI_F / Float32(LX)
    var rho = RHO0 + AMPLITUDE * sin(k * px) * sin(k * py) * sin(k * pz)
    var u = U0; var v = V0; var w = W0
    var p = P0
    var E = (
        p / (GAMMA - Float32(1.0))
        + Float32(0.5) * rho * (u * u + v * v + w * w)
    )

    var base = (e * NP + nn) * NC
    q[base + 0] = rho
    q[base + 1] = rho * u
    q[base + 2] = rho * v
    q[base + 3] = rho * w
    q[base + 4] = E


def _run(N: Int) raises -> Float64:
    var rank = mpi.world_rank()
    var size = mpi.world_size()
    var nvtx = NvtxContext()
    var ctx = DeviceContext()

    var re = ReferenceElement[P]()
    var D_ref = to_float32(re.D_ref)
    var Lift_ref = to_float32(re.Lift_ref)
    var node_weights = to_float32(re.node_weights)

    var mesh = Mesh[P](
        ctx, build_partition(rank, size, N, N, N), LX, LY, LZ,
        BoundaryConditions.periodic(),
    )
    var halo = HaloExchange(
        ctx, mesh.part, Euler.NUM_COMPONENTS,
        mesh.d_perm.unsafe_ptr(),
    )
    var physics = Euler(
        GAMMA, Float32(1.0e-6), Float32(1.0e-6), FLUX_HLLEC, False,
    )
    var solver = Solver[Euler, P](
        ctx^, mesh^, halo^, physics^, D_ref^, Lift_ref^, node_weights^,
    )

    solver.ctx.enqueue_function[
        entropy_wave_ic_kernel, entropy_wave_ic_kernel
    ](
        solver.d_q.unsafe_ptr(),
        solver.mesh.d_owned_elem_ids.unsafe_ptr(),
        solver.mesh.local.d_elem_node_xyz.unsafe_ptr(),
        solver.num_owned_elements,
        grid_dim=ceildiv(solver.num_owned_elements * NP, IC_BLOCK),
        block_dim=IC_BLOCK,
    )
    solver.ctx.synchronize()

    var n_owned_dof = solver.num_owned_elements * NP * NC
    var hbuf_ic = solver.ctx.enqueue_create_host_buffer[DType.float32](
        n_owned_dof
    )
    solver.ctx.enqueue_copy(
        hbuf_ic,
        solver.d_q.create_sub_buffer[DType.float32](0, n_owned_dof),
    )
    solver.ctx.synchronize()
    var ic_ptr = hbuf_ic.unsafe_ptr()
    var host_ic = List[Float32]()
    for k in range(n_owned_dof):
        host_ic.append(ic_ptr[k])

    var h = Float32(LX) / Float32(N)
    var c_inf = sqrt(GAMMA * P0 / RHO0)
    var wave_max = sqrt(U0 * U0 + V0 * V0 + W0 * W0) + c_inf
    var dt_est = CFL * h / (wave_max * Float32(2 * P + 1))
    var num_steps = Int(T_FINAL / dt_est) + 1
    var dt = T_FINAL / Float32(num_steps)

    for _ in range(num_steps):
        solver.step_ssprk3(dt, nvtx)
    solver.ctx.synchronize()

    var hbuf_q = solver.ctx.enqueue_create_host_buffer[DType.float32](
        n_owned_dof
    )
    solver.ctx.enqueue_copy(
        hbuf_q,
        solver.d_q.create_sub_buffer[DType.float32](0, n_owned_dof),
    )
    solver.ctx.synchronize()
    var q_ptr = hbuf_q.unsafe_ptr()

    var sum_sq: Float64 = 0.0
    var sum_ic: Float64 = 0.0
    for k in range(n_owned_dof):
        var v = q_ptr[k]
        if isnan(v) or isinf(v):
            raise Error("bench_euler_smooth_wave_3d_p3: non-finite output")
        var err = Float64(v - host_ic[k])
        sum_sq += err * err
        var ic = Float64(host_ic[k])
        sum_ic += ic * ic
    var l2 = sqrt(sum_sq / Float64(n_owned_dof))
    var l2_ic = sqrt(sum_ic / Float64(n_owned_dof))
    return l2 / l2_ic


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    mpi.init()
    var size = mpi.world_size()
    if size > 1:
        mpi.finalize()
        print("bench_euler_smooth_wave_3d_p3: runs at np=1 only")
        return

    print("bench_euler_smooth_wave_3d_p3 (P=3 entropy wave, HLLEC)")
    print("  P=", P, "  NP=", NP, "  sweep N=6, 8, 12   (threshold",
          L2_MAX_REL_AT_12, ")")

    var err6 = _run(6)
    print("  N=6   rel L2 =", err6)
    var err8 = _run(8)
    print("  N=8   rel L2 =", err8)
    var err12 = _run(12)
    print("  N=12  rel L2 =", err12)

    if err12 > L2_MAX_REL_AT_12:
        raise Error(
            "bench_euler_smooth_wave_3d_p3 FAILED: rel L2 at N=12 "
            + String(err12) + " exceeds " + String(L2_MAX_REL_AT_12)
        )
    if not (err6 > err8 and err8 > err12):
        raise Error(
            "bench_euler_smooth_wave_3d_p3 FAILED: rel L2 not monotone"
            + " (6: " + String(err6) + ", 8: " + String(err8)
            + ", 12: " + String(err12) + ")"
        )

    var rate_68  = log(err6 / err8)  / log(8.0 / 6.0)
    var rate_812 = log(err8 / err12) / log(12.0 / 8.0)
    print("  observed rates: log_(4/3)(e6/e8) =", rate_68,
          "  log_(3/2)(e8/e12) =", rate_812,
          "  (P+1 =", P + 1, ", floor", RATE_MIN, ")")
    if rate_68 < RATE_MIN and rate_812 < RATE_MIN:
        raise Error(
            String("bench_euler_smooth_wave_3d_p3 FAILED: rates ")
            + String(rate_68) + " and " + String(rate_812)
            + " both below " + String(RATE_MIN)
        )

    print("=== bench_euler_smooth_wave_3d_p3 PASSED ===")
    mpi.finalize()
