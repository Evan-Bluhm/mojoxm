# ======================================================================
# bench_maxwell_cavity_3d -- PEC-bounded standing wave, one period
# ======================================================================
#
# Canonical linear Maxwell test: TM-like lowest-order mode in a
# 1 x 1 x thin-z rectangular cavity with perfect-electric-conductor
# walls on +/- y and periodic boundaries on x and z.  Exact solution:
#
#   E_x(y, t) = sin(pi y / L) cos(omega t)
#   B_z(y, t) = cos(pi y / L) sin(omega t)
#   omega = c pi / L
#
# After one period T = 2 L / c = 2 (with c = 1, L = 1) the exact
# solution equals the IC.  Residual L2 is pure scheme dissipation
# and dispersion.
#
# Pass criteria (P=2, NX=16, NY=32, NZ=2, single-rank):
#   * rel L2(6-component state) < 1e-3
#   * no NaN / Inf
#
# Measured value is ~3.5e-5 (near Float32 roundoff over 1600 SSPRK3
# steps).  The 1e-3 gate is ~30x looser, still catches any bug that
# would push the Maxwell dispersion / PEC reflection off by more
# than a small fraction of the signal.
#
# Exercises the full 3D Maxwell path (6 conservative EM components,
# the BC_WALL path for PEC, Rusanov flux) through Mesh / HaloExchange
# / Solver.
# ======================================================================

from std.sys import has_accelerator
from std.gpu import global_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from std.math import sqrt, ceildiv, sin, isnan, isinf

from src import mpi
from src.partition import build_partition
from src.reference import N_P, build_reference_operators
from src.mesh import Mesh
from src.boundary import BoundaryConditions, BC_INTERIOR, BC_WALL
from src.halo_exchange import HaloExchange
from src.solver import Solver
from src.maxwell import Maxwell
from src.nvtx import NvtxContext


comptime NX = 16
comptime NY = 32
comptime NZ = 2
comptime LX = 1.0
comptime LY = 1.0
comptime LZ = Float64(2.0 / 32.0)

comptime C_LIGHT: Float32 = 1.0
comptime T_FINAL: Float32 = 2.0       # one period: 2 L / c
comptime CFL = Float32(0.2)
comptime IC_BLOCK = 256
comptime PI_F: Float32 = 3.14159265358979323846

comptime L2_MAX_REL: Float64 = 1.0e-3


def cavity_ic_kernel(
    q: UnsafePointer[Float32, MutAnyOrigin],
    owned_elem_ids: UnsafePointer[Int32, MutAnyOrigin],
    elem_node_xyz:  UnsafePointer[Float32, MutAnyOrigin],
    num_owned: Int,
    Ly: Float32,
):
    var idx = Int(global_idx.x)
    var total = num_owned * N_P
    if idx >= total:
        return
    var i = idx // N_P
    var nn = idx % N_P
    var e = Int(owned_elem_ids[i])
    var py = elem_node_xyz[(e * N_P + nn) * 3 + 1]
    var base = (e * N_P + nn) * 6
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
        print("bench_maxwell_cavity_3d: runs at np=1 only")
        return

    print("bench_maxwell_cavity_3d (Maxwell cavity standing wave, one period)")
    print("  P= 2   mesh=", NX, "x", NY, "x", NZ,
          "   T=", T_FINAL, " (one period)")

    var rank = mpi.world_rank()
    var nvtx = NvtxContext()
    var refs = build_reference_operators(nvtx)
    var ctx = DeviceContext()

    var bcs = BoundaryConditions(
        BC_INTERIOR, BC_INTERIOR,
        BC_WALL,     BC_WALL,
        BC_INTERIOR, BC_INTERIOR,
    )
    var mesh = Mesh(
        ctx, build_partition(rank, size, NX, NY, NZ), LX, LY, LZ, bcs,
    )
    var halo = HaloExchange(
        ctx, mesh.part, Maxwell.NUM_COMPONENTS,
        mesh.d_perm.unsafe_ptr(), bcs,
    )
    var physics = Maxwell(
        C_LIGHT,
        Float32(0.0), Float32(0.0), Float32(0.0),
        Float32(0.0), Float32(0.0), Float32(0.0),
    )
    var solver = Solver[Maxwell](
        ctx^, mesh^, halo^, physics^, refs.D_ref^, refs.Lift_ref^, refs.node_weights^,
    )

    solver.ctx.enqueue_function[cavity_ic_kernel, cavity_ic_kernel](
        solver.d_q.unsafe_ptr(),
        solver.mesh.d_owned_elem_ids.unsafe_ptr(),
        solver.mesh.local.d_elem_node_xyz.unsafe_ptr(),
        solver.num_owned_elements,
        Float32(LY),
        grid_dim=ceildiv(solver.num_owned_elements * N_P, IC_BLOCK),
        block_dim=IC_BLOCK,
    )
    solver.ctx.synchronize()

    var n_owned_dof = solver.num_owned_elements * N_P * Maxwell.NUM_COMPONENTS
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

    var h = Float32(LY) / Float32(NY)
    var dt_est = CFL * h / (C_LIGHT * Float32(5.0))
    var num_steps = Int(T_FINAL / dt_est) + 1
    var dt = T_FINAL / Float32(num_steps)
    print("  steps=", num_steps, "  dt=", dt)

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
        var v_now = q_ptr[k]
        if isnan(v_now) or isinf(v_now):
            raise Error("bench_maxwell_cavity_3d: non-finite output")
        var e = Float64(v_now - host_ic[k])
        sum_sq += e * e
        var ic = Float64(host_ic[k])
        sum_ic += ic * ic
    var l2 = sqrt(sum_sq / Float64(n_owned_dof))
    var l2_ic = sqrt(sum_ic / Float64(n_owned_dof))
    var rel_l2 = l2 / l2_ic
    print("  rel L2(state) =", rel_l2,
          "  (threshold", L2_MAX_REL, ")")

    if rel_l2 > L2_MAX_REL:
        raise Error(
            "bench_maxwell_cavity_3d FAILED: rel L2 "
            + String(rel_l2)
            + " exceeds threshold "
            + String(L2_MAX_REL)
        )
    print("=== bench_maxwell_cavity_3d PASSED ===")
    mpi.finalize()
