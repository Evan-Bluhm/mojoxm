# ======================================================================
# bench_maxwell_cavity_3d_p3 -- PEC cavity standing wave at P=3 (NP=20)
# ======================================================================
#
# P=3 counterpart of bench_maxwell_cavity_3d.  Same TM-like lowest-order
# mode in a 1 x 1 x thin-z rectangular cavity with perfect-electric-
# conductor walls on +/- y and periodic boundaries on x and z, but
# routed through Mesh[3] / Solver[Maxwell, 3] with NP = 20 nodes per
# tet.
#
# Closes the only coverage gap in the BC_WALL (PEC) dispatch arm of
# Maxwell at P>=3: every other higher-order Maxwell bench
# (plane_wave_3d_p3..p5, te_plane_wave_3d, uniform_j/m_3d_p3) uses
# either periodic BCs or BC_OUTFLOW.  A regression in the PEC
# reflection lift coefficient at P>=3 -- e.g. a stray sign on the
# odd-degree edge nodes -- would not trip any existing gate.
#
# Exact solution (same as P=2 cavity):
#
#   E_x(y, t) = sin(pi y / L) cos(omega t)
#   B_z(y, t) = -(1/c) cos(pi y / L) sin(omega t)
#   omega = c pi / L
#
# After one period T = 2 L / c = 2 (with c = 1, L = 1) the exact
# solution equals the IC.
#
# Pass criteria (P=3, NX=8, NY=16, NZ=2, single-rank):
#   * rel L2(6-component state) < 1e-4  (typical 1-3e-5 at this P)
#   * no NaN / Inf
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
from src.boundary import BoundaryConditions, BC_INTERIOR, BC_WALL
from src.halo_exchange import HaloExchange
from src.solver import Solver
from src.maxwell import Maxwell
from src.nvtx import NvtxContext


comptime P = 3
comptime NP = num_tet_nodes(P)  # 20 at P=3
comptime NC = 6  # Maxwell.NUM_COMPONENTS

comptime NX = 8
comptime NY = 16
comptime NZ = 2
comptime LX = 1.0
comptime LY = 1.0
comptime LZ = Float64(2.0 / 16.0)

comptime C_LIGHT: Float32 = 1.0
comptime T_FINAL: Float32 = 2.0  # one period: 2 L / c
comptime CFL = Float32(0.10)
comptime IC_BLOCK = 256
comptime PI_F: Float32 = 3.14159265358979323846

# At P=3 / NX=8 the cavity mode is well-resolved; rel L2 sits well
# below the P=2 floor (~3.5e-5 there).  1e-4 is a generous gate that
# catches PEC reflection / lift-operator bugs at P=3 without false-
# firing on Float32 wobble.
comptime L2_MAX_REL: Float64 = 1.0e-4


def cavity_ic_kernel(
    q: UnsafePointer[Float32, MutAnyOrigin],
    owned_elem_ids: UnsafePointer[Int32, MutAnyOrigin],
    elem_node_xyz: UnsafePointer[Float32, MutAnyOrigin],
    num_owned: Int,
    Ly: Float32,
):
    var idx = Int(global_idx.x)
    var total = num_owned * NP
    if idx >= total:
        return
    var i = idx // NP
    var nn = idx % NP
    var e = Int(owned_elem_ids[i])
    var py = elem_node_xyz[(e * NP + nn) * 3 + 1]
    var base = (e * NP + nn) * 6
    q[base + 0] = sin(PI_F * py / Ly)
    q[base + 1] = Float32(0.0)
    q[base + 2] = Float32(0.0)
    q[base + 3] = Float32(0.0)
    q[base + 4] = Float32(0.0)
    q[base + 5] = Float32(0.0)


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    mpi.init()
    var size = mpi.world_size()
    if size > 1:
        mpi.finalize()
        print("bench_maxwell_cavity_3d_p3: runs at np=1 only")
        return

    print(
        "bench_maxwell_cavity_3d_p3 (Maxwell PEC cavity standing wave, P=3, one"
        " period)"
    )
    print(
        "  P=", P, "  NP=", NP, "  mesh=", NX, "x", NY, "x", NZ, "  T=", T_FINAL
    )

    var rank = mpi.world_rank()
    var nvtx = NvtxContext()
    var ctx = DeviceContext()

    # Build reference operators directly at P=3.
    var re = ReferenceElement[P]()
    var D_ref = to_float32(re.D_ref)
    var Lift_ref = to_float32(re.Lift_ref)
    var node_weights = to_float32(re.node_weights)

    var bcs = BoundaryConditions(
        BC_INTERIOR,
        BC_INTERIOR,
        BC_WALL,
        BC_WALL,
        BC_INTERIOR,
        BC_INTERIOR,
    )
    var mesh = Mesh[P](
        ctx,
        build_partition(rank, size, NX, NY, NZ),
        LX,
        LY,
        LZ,
        bcs,
    )
    var halo = HaloExchange(
        ctx,
        mesh.part,
        Maxwell.NUM_COMPONENTS,
        mesh.d_perm.unsafe_ptr(),
        bcs,
    )
    var physics = Maxwell(
        C_LIGHT,
        Float32(0.0),
        Float32(0.0),
        Float32(0.0),
        Float32(0.0),
        Float32(0.0),
        Float32(0.0),
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

    solver.ctx.enqueue_function[cavity_ic_kernel](
        solver.d_q.unsafe_ptr(),
        solver.mesh.d_owned_elem_ids.unsafe_ptr(),
        solver.mesh.local.d_elem_node_xyz.unsafe_ptr(),
        solver.num_owned_elements,
        Float32(LY),
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

    var h = Float32(LY) / Float32(NY)
    # CFL: factor 2P+1 = 7 at P=3.
    var dt_est = CFL * h / (C_LIGHT * Float32(2 * P + 1))
    var num_steps = Int(T_FINAL / dt_est) + 1
    var dt = T_FINAL / Float32(num_steps)
    print("  steps=", num_steps, "  dt=", dt)

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
        var v_now = q_ptr[k]
        if isnan(v_now) or isinf(v_now):
            raise Error("bench_maxwell_cavity_3d_p3: non-finite output")
        var e = Float64(v_now - host_ic[k])
        sum_sq += e * e
        var ic = Float64(host_ic[k])
        sum_ic += ic * ic
    var l2 = sqrt(sum_sq / Float64(n_owned_dof))
    var l2_ic = sqrt(sum_ic / Float64(n_owned_dof))
    var rel_l2 = l2 / l2_ic
    print("  rel L2(state) =", rel_l2, "  (threshold", L2_MAX_REL, ")")

    if rel_l2 > L2_MAX_REL:
        raise Error(
            "bench_maxwell_cavity_3d_p3 FAILED: rel L2 "
            + String(rel_l2)
            + " exceeds threshold "
            + String(L2_MAX_REL)
        )
    print("=== bench_maxwell_cavity_3d_p3 PASSED ===")
    mpi.finalize()
