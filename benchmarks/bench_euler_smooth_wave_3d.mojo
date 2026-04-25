# ======================================================================
# bench_euler_smooth_wave_3d -- 3D entropy-wave preservation gate
# ======================================================================
#
# Short-description: 3D counterpart to bench_euler_smooth_wave_2d.
# Entropy wave on a periodic cube: constant (u0, v0, w0) and constant
# pressure, density carries a smooth 3D perturbation.  This is an
# exact entropy-wave solution of the compressible Euler equations --
# constant p kills the pressure-gradient term in momentum and the
# (E+p) * u term in energy both advects rho unchanged.
#
# IC:
#   rho = rho0 + A * sin(2 pi x / L) * sin(2 pi y / L) * sin(2 pi z / L)
#   u = u0,  v = v0,  w = w0     (uniform)
#   p = p0                        (uniform -- entropy wave, NOT isentropic)
#   E = p / (gamma - 1) + 0.5 * rho * (u^2 + v^2 + w^2)
#
# After T = L / u0 the density pattern has advected one full period
# in each direction and exactly returns to the IC.
#
# Pass criteria (P=2, HLLEC flux, periodic, single-rank):
#   * rel L2(state) at N=8, 12, 16 all < 1e-3
#   * no NaN / Inf
#
# Same framing as the 2D entropy-wave gate: on a smooth linearised
# problem HLLEC is so accurate that scheme error sits below Float32
# roundoff over any practical refinement sweep.  Convergence-rate
# doesn't apply; the tight absolute threshold catches any Euler bug
# that would pollute the scheme even slightly.
# ======================================================================

from std.sys import has_accelerator
from std.gpu import global_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from std.math import sqrt, ceildiv, sin, isnan, isinf

from src import mpi
from src.partition import build_partition
from src.reference import N_P, build_reference_operators
from src.mesh import Mesh
from src.boundary import BoundaryConditions
from src.halo_exchange import HaloExchange
from src.solver import Solver
from src.euler import Euler, FLUX_HLLEC
from src.nvtx import NvtxContext


comptime P = 2
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
comptime T_FINAL: Float32 = 1.0       # one advection period at u0 = 1
comptime CFL = Float32(0.2)
comptime IC_BLOCK = 256
comptime PI_F: Float32 = 3.14159265358979323846

comptime L2_MAX_REL: Float64 = 1.0e-3


def entropy_wave_ic_kernel(
    q: UnsafePointer[Float32, MutAnyOrigin],
    owned_elem_ids: UnsafePointer[Int32, MutAnyOrigin],
    elem_node_xyz:  UnsafePointer[Float32, MutAnyOrigin],
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
    var py = elem_node_xyz[(e * N_P + nn) * 3 + 1]
    var pz = elem_node_xyz[(e * N_P + nn) * 3 + 2]

    var k = Float32(2.0) * PI_F / Float32(LX)
    var rho = RHO0 + AMPLITUDE * sin(k * px) * sin(k * py) * sin(k * pz)
    var u = U0; var v = V0; var w = W0
    var p = P0
    var E = (
        p / (GAMMA - Float32(1.0))
        + Float32(0.5) * rho * (u * u + v * v + w * w)
    )

    var base = (e * N_P + nn) * 5
    q[base + 0] = rho
    q[base + 1] = rho * u
    q[base + 2] = rho * v
    q[base + 3] = rho * w
    q[base + 4] = E


def _run(N: Int) raises -> Float64:
    """Run one period at NxNxN resolution and return rel L2(q - q_ic)."""
    var rank = mpi.world_rank()
    var size = mpi.world_size()
    var nvtx = NvtxContext()
    var refs = build_reference_operators(nvtx)
    var ctx = DeviceContext()

    var mesh = Mesh(
        ctx, build_partition(rank, size, N, N, N), LX, LY, LZ,
        BoundaryConditions.periodic(),
    )
    var halo = HaloExchange(
        ctx, mesh.part, Euler.NUM_COMPONENTS,
        mesh.d_perm.unsafe_ptr(),
    )
    var physics = Euler(
        GAMMA, Float32(1.0e-6), Float32(1.0e-6),
        FLUX_HLLEC, False,
    )
    var solver = Solver[Euler](
        ctx^, mesh^, halo^, physics^, refs.D_ref^, refs.Lift_ref^, refs.node_weights^,
    )

    solver.ctx.enqueue_function[
        entropy_wave_ic_kernel, entropy_wave_ic_kernel
    ](
        solver.d_q.unsafe_ptr(),
        solver.mesh.d_owned_elem_ids.unsafe_ptr(),
        solver.mesh.local.d_elem_node_xyz.unsafe_ptr(),
        solver.num_owned_elements,
        grid_dim=ceildiv(solver.num_owned_elements * N_P, IC_BLOCK),
        block_dim=IC_BLOCK,
    )
    solver.ctx.synchronize()

    var n_owned_dof = solver.num_owned_elements * N_P * Euler.NUM_COMPONENTS
    var hbuf_ic = solver.ctx.enqueue_create_host_buffer[DType.float32](n_owned_dof)
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

    var hbuf_q = solver.ctx.enqueue_create_host_buffer[DType.float32](n_owned_dof)
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
            raise Error("bench_euler_smooth_wave_3d: non-finite output")
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
        print("bench_euler_smooth_wave_3d: runs at np=1 only")
        return

    print("bench_euler_smooth_wave_3d (3D entropy wave, HLLEC)")
    print("  P=", P, "  sweep N=8, 12, 16   (threshold", L2_MAX_REL, ")")

    var err8  = _run(8)
    print("  N=8   rel L2 =", err8)
    var err12 = _run(12)
    print("  N=12  rel L2 =", err12)
    var err16 = _run(16)
    print("  N=16  rel L2 =", err16)

    if err8 > L2_MAX_REL:
        raise Error(
            "bench_euler_smooth_wave_3d FAILED: N=8 rel L2 "
            + String(err8) + " exceeds " + String(L2_MAX_REL)
        )
    if err12 > L2_MAX_REL:
        raise Error(
            "bench_euler_smooth_wave_3d FAILED: N=12 rel L2 "
            + String(err12) + " exceeds " + String(L2_MAX_REL)
        )
    if err16 > L2_MAX_REL:
        raise Error(
            "bench_euler_smooth_wave_3d FAILED: N=16 rel L2 "
            + String(err16) + " exceeds " + String(L2_MAX_REL)
        )

    print("=== bench_euler_smooth_wave_3d PASSED ===")
    mpi.finalize()
