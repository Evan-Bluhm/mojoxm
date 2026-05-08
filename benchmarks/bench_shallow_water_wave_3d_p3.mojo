# ======================================================================
# bench_shallow_water_wave_3d_p3 -- 3D linear SW wave at P=3 (NP=20)
# ======================================================================
#
# P=3 counterpart of bench_shallow_water_wave_3d.  Same small-amplitude
# linear shallow-water wave embedded in the 3D DG pipeline (F^z = 0,
# NZ = 2 stacked sheet) but routed through Mesh[3] / Solver[
# ShallowWater, 3] with NP = 20 nodes per tet.
#
# Pairs with bench_advection_3d_p3 (single-component) and
# bench_euler_smooth_wave_3d_p3 (NC=5) to bring 3D SW (NC=3) up to
# the same NP=20 coverage parity for smooth physics.
#
# IC (rest state + stationary sinusoidal disturbance):
#   h = H + A * sin(2 pi x / Lx)
#   u = v = 0
# After T = Lx / sqrt(g H) the d'Alembert wave returns to IC modulo
# nonlinear O((A/H)^2) corrections.
#
# Pass criteria (P=3, periodic, single-rank, sweep N = 8, 12, 16):
#   * rel L2 < 5e-4 at every N
#   * mass conservation drift < 1e-4
#   * no NaN / Inf
#
# At P=3 the dispersion error sits at the A/H=0.01 nonlinear-
# correction floor (~1.7e-4) across the entire sweep, so this is an
# absolute-L2 sentinel rather than a rate gate -- catches any 3D SW
# Rusanov / volume-RHS regression that pushes L2 well above floor.
# ======================================================================

from std.sys import has_accelerator
from std.gpu import global_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from std.math import sqrt, ceildiv, sin, isnan, isinf

from src import mpi
from src.partition import build_partition
from src.reference import (
    ReferenceElement,
    to_float32,
    num_tet_nodes,
)
from src.mesh import Mesh
from src.boundary import BoundaryConditions
from src.halo_exchange import HaloExchange
from src.solver import Solver
from src.shallow_water import ShallowWater
from src.nvtx import NvtxContext


comptime P = 3
comptime NP = num_tet_nodes(P)  # 20 at P=3
comptime NC = 3  # ShallowWater.NUM_COMPONENTS

comptime LX = 1.0
comptime LY = 1.0
comptime LZ = 0.1
comptime GRAVITY: Float32 = 1.0
comptime H_REST: Float32 = 1.0
comptime AMPLITUDE: Float32 = 0.01
comptime H_MIN: Float32 = 1.0e-6
comptime CFL = Float32(0.10)
comptime IC_BLOCK = 256
comptime PI_F: Float32 = 3.14159265358979323846

# One wave period: T = Lx / c with c = sqrt(g H) = 1.
comptime T_FINAL: Float32 = Float32(LX / 1.0)

# Empirical: ~1.7e-4 across the sweep (A/H=0.01 nonlinear floor).
# 5e-4 is ~3x margin -- catches any SW flux/operator regression.
comptime L2_MAX_REL: Float64 = 5.0e-4
comptime MASS_TOL_REL: Float64 = 1.0e-4


def wave_ic_kernel(
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
    var k = Float32(2.0) * PI_F / Float32(LX)
    var h = H_REST + AMPLITUDE * sin(k * px)
    var base = (e * NP + nn) * NC
    q[base + 0] = h
    q[base + 1] = Float32(0.0)  # hu
    q[base + 2] = Float32(0.0)  # hv


@fieldwise_init
struct RunResult(Movable):
    var rel_l2: Float64
    var mass_rel: Float64


def _run(N: Int) raises -> RunResult:
    var rank = mpi.world_rank()
    var size = mpi.world_size()
    var nvtx = NvtxContext()
    var ctx = DeviceContext()

    # Build reference operators directly at P=3 (build_reference_operators
    # defaults to P=2).
    var re = ReferenceElement[P]()
    var D_ref = to_float32(re.D_ref)
    var Lift_ref = to_float32(re.Lift_ref)
    var node_weights = to_float32(re.node_weights)

    var NZ = 2
    var mesh = Mesh[P](
        ctx,
        build_partition(rank, size, N, N, NZ),
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

    solver.ctx.enqueue_function[wave_ic_kernel, wave_ic_kernel](
        solver.d_q.unsafe_ptr(),
        solver.mesh.d_owned_elem_ids.unsafe_ptr(),
        solver.mesh.local.d_elem_node_xyz.unsafe_ptr(),
        solver.num_owned_elements,
        grid_dim=ceildiv(solver.num_owned_elements * NP, IC_BLOCK),
        block_dim=IC_BLOCK,
    )
    solver.ctx.synchronize()

    var n_owned_dof = solver.num_owned_elements * NP * NC
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

    var h_cell = Float32(LX) / Float32(N)
    var c = sqrt(GRAVITY * H_REST)
    # CFL: factor 2P+1 = 7 at P=3.
    var dt_est = CFL * h_cell / (c * Float32(2 * P + 1))
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
            raise Error("bench_shallow_water_wave_3d_p3: non-finite output")
        var err = Float64(v - host_ic[k])
        sum_sq += err * err
        var ic = Float64(host_ic[k])
        sum_ic += ic * ic
    var l2 = sqrt(sum_sq / Float64(n_owned_dof))
    var l2_ic = sqrt(sum_ic / Float64(n_owned_dof))
    var rel_l2 = l2 / l2_ic

    var mass_ic: Float64 = 0.0
    var mass_fin: Float64 = 0.0
    var n_elem_nodes = solver.num_owned_elements * NP
    for i in range(n_elem_nodes):
        mass_ic += Float64(host_ic[i * NC + 0])
        mass_fin += Float64(q_ptr[i * NC + 0])
    var dmass = mass_fin - mass_ic
    if dmass < 0.0:
        dmass = -dmass
    var mass_rel = dmass / mass_ic

    return RunResult(rel_l2, mass_rel)


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    mpi.init()
    print("bench_shallow_water_wave_3d_p3 (linear SW wave, P=3)")
    print("  P=", P, "  NP=", NP, "  T=", T_FINAL, "  A/H=", AMPLITUDE / H_REST)

    var Ns = List[Int]()
    Ns.append(8)
    Ns.append(12)
    Ns.append(16)

    var all_ok = True
    for idx in range(len(Ns)):
        var N = Ns[idx]
        var result = _run(N)
        var l2 = result.rel_l2
        var mrel = result.mass_rel
        print("  N=", N, "  rel_L2=", l2, "  mass_drift=", mrel)
        if l2 > L2_MAX_REL:
            all_ok = False
            print("    -> L2 above threshold", L2_MAX_REL)
        if mrel > MASS_TOL_REL:
            all_ok = False
            print("    -> mass drift above threshold", MASS_TOL_REL)

    if not all_ok:
        raise Error("bench_shallow_water_wave_3d_p3 FAILED")
    print("=== bench_shallow_water_wave_3d_p3 PASSED ===")
    mpi.finalize()
