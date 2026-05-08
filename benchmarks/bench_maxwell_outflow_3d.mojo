# ======================================================================
# bench_maxwell_outflow_3d -- 3D Maxwell BC_OUTFLOW preservation gate
# ======================================================================
#
# 3D analog of bench_maxwell_outflow_2d.  Closes a 3D Maxwell BC
# coverage gap.  src/maxwell.mojo dispatch in `boundary_flux` has
# four arms (BC_INTERIOR, BC_WALL, BC_INFLOW, else=BC_OUTFLOW), but
# every existing 3D Maxwell bench / test exercises only periodic
# (`bench_maxwell_plane_wave_3d`, `bench_maxwell_uniform_j_3d`,
# `_m_3d`) or BC_WALL (`bench_maxwell_cavity_3d`).  The BC_OUTFLOW
# (zero-gradient) and BC_INFLOW arms in 3D were untested.
#
# Cleanest test: uniform-state preservation under BC_OUTFLOW on all
# six faces.  Same logic as the 2D version: any constant
# (Ex, Ey, Ez, Bx, By, Bz) satisfies vacuum Maxwell exactly; under
# BC_OUTFLOW the ghost equals the interior, the Rusanov dissipation
# vanishes, and the constant boundary flux integrates to zero around
# each tet by the discrete divergence theorem.  Net RHS = 0.
#
# Pass criteria (P=2, NX=NY=NZ=8, T=0.5):
#   * max |q - q_IC| < 1e-3 (Float32 epsilon * step accumulation)
#   * no NaN / Inf
# ======================================================================

from std.sys import has_accelerator
from std.gpu import global_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from std.math import ceildiv, isnan, isinf

from src import mpi
from src.partition import build_partition
from src.reference import N_P, build_reference_operators
from src.mesh import Mesh
from src.boundary import BoundaryConditions, BC_OUTFLOW
from src.halo_exchange import HaloExchange
from src.solver import Solver
from src.maxwell import Maxwell
from src.nvtx import NvtxContext


comptime NX = 8
comptime NY = 8
comptime NZ = 8
comptime LX = 1.0
comptime LY = 1.0
comptime LZ = 1.0

comptime C_LIGHT: Float32 = 1.0
comptime T_FINAL: Float32 = 0.5
comptime CFL = Float32(0.2)
comptime IC_BLOCK = 256

# Uniform IC components, all six non-zero so every flux term is
# exercised.
comptime EX0: Float32 = 0.3
comptime EY0: Float32 = -0.2
comptime EZ0: Float32 = 0.5
comptime BX0: Float32 = 0.4
comptime BY0: Float32 = 0.1
comptime BZ0: Float32 = -0.6

comptime DRIFT_TOL: Float64 = 1.0e-3


def uniform_ic_kernel(
    q: UnsafePointer[Float32, MutAnyOrigin],
    owned_elem_ids: UnsafePointer[Int32, MutAnyOrigin],
    elem_node_xyz: UnsafePointer[Float32, MutAnyOrigin],
    num_owned: Int,
):
    var idx = Int(global_idx.x)
    var total = num_owned * N_P
    if idx >= total:
        return
    var i = idx // N_P
    var nn = idx % N_P
    var e = Int(owned_elem_ids[i])
    var base = (e * N_P + nn) * 6
    q[base + 0] = EX0
    q[base + 1] = EY0
    q[base + 2] = EZ0
    q[base + 3] = BX0
    q[base + 4] = BY0
    q[base + 5] = BZ0


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    mpi.init()
    var size = mpi.world_size()
    if size > 1:
        mpi.finalize()
        print("bench_maxwell_outflow_3d: runs at np=1 only")
        return

    print("bench_maxwell_outflow_3d (3D Maxwell BC_OUTFLOW preservation)")
    print("  P= 2   mesh=", NX, "x", NY, "x", NZ, "   T=", T_FINAL)

    var rank = mpi.world_rank()
    var nvtx = NvtxContext()
    var refs = build_reference_operators(nvtx)
    var ctx = DeviceContext()

    var bcs = BoundaryConditions(
        BC_OUTFLOW,
        BC_OUTFLOW,
        BC_OUTFLOW,
        BC_OUTFLOW,
        BC_OUTFLOW,
        BC_OUTFLOW,
    )
    var mesh = Mesh(
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
        Float32(0.0),  # J = 0
        Float32(0.0),
        Float32(0.0),
        Float32(0.0),  # M = 0
    )
    var solver = Solver[Maxwell](
        ctx^,
        mesh^,
        halo^,
        physics^,
        refs.D_ref^,
        refs.Lift_ref^,
        refs.node_weights^,
    )

    solver.ctx.enqueue_function[uniform_ic_kernel](
        solver.d_q.unsafe_ptr(),
        solver.mesh.d_owned_elem_ids.unsafe_ptr(),
        solver.mesh.local.d_elem_node_xyz.unsafe_ptr(),
        solver.num_owned_elements,
        grid_dim=ceildiv(solver.num_owned_elements * N_P, IC_BLOCK),
        block_dim=IC_BLOCK,
    )
    solver.ctx.synchronize()

    var n_owned_dof = solver.num_owned_elements * N_P * Maxwell.NUM_COMPONENTS

    var h = Float32(LX) / Float32(NX)
    var dt_est = CFL * h / (C_LIGHT * Float32(5.0))
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

    var max_drift: Float64 = 0.0
    var n_owned_nodes = solver.num_owned_elements * N_P
    for i in range(n_owned_nodes):
        var qx = q_ptr[i * 6 + 0]
        var qy = q_ptr[i * 6 + 1]
        var qz = q_ptr[i * 6 + 2]
        var bx = q_ptr[i * 6 + 3]
        var by = q_ptr[i * 6 + 4]
        var bz = q_ptr[i * 6 + 5]
        if (
            isnan(qx)
            or isinf(qx)
            or isnan(qy)
            or isinf(qy)
            or isnan(qz)
            or isinf(qz)
            or isnan(bx)
            or isinf(bx)
            or isnan(by)
            or isinf(by)
            or isnan(bz)
            or isinf(bz)
        ):
            raise Error("bench_maxwell_outflow_3d: non-finite output")
        var d_ex = Float64(qx - EX0)
        if d_ex < 0.0:
            d_ex = -d_ex
        var d_ey = Float64(qy - EY0)
        if d_ey < 0.0:
            d_ey = -d_ey
        var d_ez = Float64(qz - EZ0)
        if d_ez < 0.0:
            d_ez = -d_ez
        var d_bx = Float64(bx - BX0)
        if d_bx < 0.0:
            d_bx = -d_bx
        var d_by = Float64(by - BY0)
        if d_by < 0.0:
            d_by = -d_by
        var d_bz = Float64(bz - BZ0)
        if d_bz < 0.0:
            d_bz = -d_bz
        var local_max = d_ex
        if d_ey > local_max:
            local_max = d_ey
        if d_ez > local_max:
            local_max = d_ez
        if d_bx > local_max:
            local_max = d_bx
        if d_by > local_max:
            local_max = d_by
        if d_bz > local_max:
            local_max = d_bz
        if local_max > max_drift:
            max_drift = local_max

    print("  max |q - q_IC| =", max_drift, "  (threshold", DRIFT_TOL, ")")
    if max_drift > DRIFT_TOL:
        raise Error(
            "bench_maxwell_outflow_3d FAILED: drift "
            + String(max_drift)
            + " > "
            + String(DRIFT_TOL)
        )

    print("=== bench_maxwell_outflow_3d PASSED ===")
    mpi.finalize()
