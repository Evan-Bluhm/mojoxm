# ======================================================================
# bench_mhd_alfven_3d_p4 -- 3D linear Alfven wave at P=4 (NP=35)
# ======================================================================
#
# P=4 counterpart of bench_mhd_alfven_3d_p3.  Same linearly-polarised
# right-going Alfven-wave IC in a thin 3D slab, but routed through
# Mesh[4] / Solver[IdealMHD, 4] / NP=35.  Highest-NC physics
# validated at NP=35 in 3D so far -- exercises the cooperative
# shared-memory rk_stage_kernel at NC=9 / NP=35 = 315 q values per
# element.
#
# Pairs with the just-added P=4 / NP=35 gates for advection, Euler,
# Maxwell, SW to bring 3D coverage parity for smooth physics up to
# P=4 across all 5 physics paths.
#
# Pass criteria (P=4, NX=8, NY=NZ=4, single-rank, periodic):
#   * rel L2(9-component state) < 5e-3 (same A^2~0.01 nonlinear
#     correction floor as P=3 -- IC sets the floor, not the scheme)
#   * no NaN / Inf
# ======================================================================

from std.sys import has_accelerator
from std.gpu import global_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from std.math import sqrt, ceildiv, sin, isnan, isinf

from src import mpi
from src.partition import build_partition
from src.reference import ReferenceElement, to_float32, num_tet_nodes
from src.mesh import Mesh
from src.boundary import BoundaryConditions
from src.halo_exchange import HaloExchange
from src.solver import Solver
from src.mhd import IdealMHD
from src.nvtx import NvtxContext


comptime P = 4
comptime NP = num_tet_nodes(P)  # 35 at P=4
comptime NC = 9  # IdealMHD.NUM_COMPONENTS

comptime NX = 8
comptime NY = 4
comptime NZ = 4
comptime LX = 1.0
comptime LY = Float64(4.0 / 8.0)
comptime LZ = Float64(4.0 / 8.0)

comptime GAMMA = Float32(5.0 / 3.0)
comptime RHO0 = Float32(1.0)
comptime B0 = Float32(1.0)
comptime P0 = Float32(0.1)
comptime AMPLITUDE = Float32(0.1)
comptime C_H = Float32(1.5)
comptime ALPHA_D = Float32(0.5)
comptime MIN_DENSITY = Float32(1.0e-6)
comptime MIN_PRESSURE = Float32(1.0e-6)

comptime T_FINAL = Float32(1.0)  # one period (LX / c_A = 1)
comptime CFL = Float32(0.08)
comptime IC_BLOCK = 256
comptime PI_F = Float32(3.14159265358979323846)

# Same nonlinear A^2~0.01 floor as P=2/P=3 (~3.6e-3).  5e-3 is the
# usual margin -- catches MHD operator regressions at NP=35 without
# false-firing on the IC-set floor.
comptime L2_MAX_REL: Float64 = 5.0e-3


def alfven_ic_kernel(q: UnsafePointer[Float32, MutAnyOrigin], owned_elem_ids: UnsafePointer[Int32, MutAnyOrigin], elem_node_xyz: UnsafePointer[Float32, MutAnyOrigin], num_owned: Int):
    var idx = Int(global_idx.x)
    var total = num_owned * NP
    if idx >= total:
        return
    var i = idx // NP
    var nn = idx % NP
    var e = Int(owned_elem_ids[i])
    var px = elem_node_xyz[(e * NP + nn) * 3 + 0]

    var k_wave = Float32(2.0) * PI_F / Float32(LX)
    var s = sin(k_wave * px)
    var uy = AMPLITUDE * s
    var By = -AMPLITUDE * s

    var rho = RHO0
    var u = Float32(0.0)
    var v = uy
    var w = Float32(0.0)
    var bx = B0
    var by = By
    var bz = Float32(0.0)
    var p_gas = P0
    var E = p_gas / (GAMMA - Float32(1.0)) + Float32(0.5) * rho * (u * u + v * v + w * w) + Float32(0.5) * (bx * bx + by * by + bz * bz)
    var base = (e * NP + nn) * NC
    q[base + 0] = rho
    q[base + 1] = rho * u
    q[base + 2] = rho * v
    q[base + 3] = rho * w
    q[base + 4] = E
    q[base + 5] = bx
    q[base + 6] = by
    q[base + 7] = bz
    q[base + 8] = Float32(0.0)  # psi


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    mpi.init()
    var size = mpi.world_size()
    if size > 1:
        mpi.finalize()
        print("bench_mhd_alfven_3d_p4: runs at np=1 only")
        return

    print("bench_mhd_alfven_3d_p4 (3D Alfven wave, P=4, one period)")
    print("  P=", P, "  NP=", NP, "  mesh=", NX, "x", NY, "x", NZ, "  (c_A = 1, T = 1)")

    var rank = mpi.world_rank()
    var nvtx = NvtxContext()
    var ctx = DeviceContext()

    var re = ReferenceElement[P]()
    var D_ref = to_float32(re.D_ref)
    var Lift_ref = to_float32(re.Lift_ref)
    var node_weights = to_float32(re.node_weights)

    var bcs = BoundaryConditions.periodic()
    var mesh = Mesh[P](ctx, build_partition(rank, size, NX, NY, NZ), LX, LY, LZ, bcs)
    var halo = HaloExchange(ctx, mesh.part, IdealMHD.NUM_COMPONENTS, mesh.d_perm.unsafe_ptr(), bcs)
    var physics = IdealMHD(GAMMA, MIN_DENSITY, MIN_PRESSURE, C_H, ALPHA_D)
    var solver = Solver[IdealMHD, P](ctx^, mesh^, halo^, physics^, D_ref^, Lift_ref^, node_weights^)

    solver.ctx.enqueue_function[alfven_ic_kernel](
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
    solver.ctx.enqueue_copy(hbuf_ic, solver.d_q.create_sub_buffer[DType.float32](0, n_owned_dof))
    solver.ctx.synchronize()
    var ic_ptr = hbuf_ic.unsafe_ptr()
    var host_ic = List[Float32]()
    for k in range(n_owned_dof):
        host_ic.append(ic_ptr[k])

    var cs2 = GAMMA * P0 / RHO0
    var ca2 = (B0 * B0 + AMPLITUDE * AMPLITUDE) / RHO0
    var cf = sqrt(cs2 + ca2)
    var wave = cf if cf > C_H else C_H
    var h = Float32(LX) / Float32(NX)
    # CFL: factor 2P+1 = 9 at P=4.
    var dt_est = CFL * h / (wave * Float32(2 * P + 1))
    var num_steps = Int(T_FINAL / dt_est) + 1
    var dt_used = T_FINAL / Float32(num_steps)
    print("  steps=", num_steps, "  dt=", dt_used)

    for _ in range(num_steps):
        solver.step_ssprk3(dt_used, nvtx)
    solver.ctx.synchronize()

    var hbuf_q = solver.ctx.enqueue_create_host_buffer[DType.float32](n_owned_dof)
    solver.ctx.enqueue_copy(hbuf_q, solver.d_q.create_sub_buffer[DType.float32](0, n_owned_dof))
    solver.ctx.synchronize()
    var q_ptr = hbuf_q.unsafe_ptr()

    var sum_sq: Float64 = 0.0
    var sum_ic: Float64 = 0.0
    for k in range(n_owned_dof):
        var v_now = q_ptr[k]
        if isnan(v_now) or isinf(v_now):
            raise Error("bench_mhd_alfven_3d_p4: non-finite output at " + String(k))
        var err = Float64(v_now - host_ic[k])
        sum_sq += err * err
        var ic = Float64(host_ic[k])
        sum_ic += ic * ic
    var l2 = sqrt(sum_sq / Float64(n_owned_dof))
    var l2_ic = sqrt(sum_ic / Float64(n_owned_dof))
    var rel_l2 = l2 / l2_ic
    print("  rel L2(state) =", rel_l2, "  (threshold", L2_MAX_REL, ")")

    if rel_l2 > L2_MAX_REL:
        raise Error("bench_mhd_alfven_3d_p4 FAILED: rel L2 " + String(rel_l2) + " exceeds threshold " + String(L2_MAX_REL))
    print("=== bench_mhd_alfven_3d_p4 PASSED ===")
    mpi.finalize()
