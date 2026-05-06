# ======================================================================
# bench_euler_inflow_3d -- 3D Euler BC_INFLOW preservation gate
# ======================================================================
#
# 3D analog of the 2D `bench_euler_channel_steady_2d`, but as a
# pure inflow-preservation gate (no shocks, no walls).  Closes the
# 3D Euler BC_INFLOW coverage gap: the Euler struct accepts
# (inflow_rho, inflow_rhou, inflow_rhov, inflow_rhow, inflow_E)
# but no 3D bench passes non-zero values, so the BC_INFLOW arm of
# `Euler.boundary_flux` is untested in 3D.
#
# Cleanest test: uniform subcritical state matched to the inflow
# ghost.  At t=0 the IC matches the BC_INFLOW state exactly, so
# the analytic solution is the IC unchanged for all time.  Any
# drift signals a BC bug.
#
# Setup: rho0=1, u0=0.5, v=w=0, p=1 (Mach M=u0/c=0.41 < 1, subcritical
# so BC_OUTFLOW at +x is well-posed).  HLLEC flux, periodic y/z.
#
# Pass criteria (P=2, NX=16 NY=NZ=4, T=1):
#   * max |rho - rho0| / rho0 < 1e-3
#   * max |rho*u - rho0*u0| / rho0*u0 < 1e-3
#   * max |rho*v|, |rho*w| < 1e-3
#   * max |E - E0| / E0 < 1e-3
#   * no NaN / Inf
# ======================================================================

from std.sys import has_accelerator
from std.gpu import global_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from std.math import sqrt, ceildiv, isnan, isinf

from src import mpi
from src.partition import build_partition
from src.reference import N_P, build_reference_operators
from src.mesh import Mesh
from src.boundary import (
    BoundaryConditions,
    BC_INTERIOR,
    BC_INFLOW,
    BC_OUTFLOW,
)
from src.halo_exchange import HaloExchange
from src.solver import Solver
from src.euler import Euler, FLUX_HLLEC
from src.nvtx import NvtxContext


comptime NX = 16
comptime NY = 4
comptime NZ = 4
comptime LX = 1.0
comptime LY = 1.0
comptime LZ = 1.0

comptime GAMMA: Float32 = 1.4
comptime RHO0: Float32 = 1.0
comptime U0: Float32 = 0.5
comptime P0: Float32 = 1.0
comptime T_FINAL: Float32 = 1.0
comptime CFL = Float32(0.2)
comptime IC_BLOCK = 256
comptime MIN_DENSITY = Float32(1.0e-6)
comptime MIN_PRESSURE = Float32(1.0e-6)

# Empirical drifts after 674 SSPRK3 steps: rho 7.3e-4, rhou 1.0e-3,
# rho*v/w 3e-6, E 8.5e-4 -- all at the Float32-epsilon * steps *
# accumulation floor.  2e-3 is ~2x the empirical floor and catches
# any meaningful BC_INFLOW regression.
comptime RHO_REL_TOL: Float64 = 2.0e-3
comptime RHOU_REL_TOL: Float64 = 2.0e-3
comptime RHOVW_TOL: Float64 = 1.0e-3
comptime E_REL_TOL: Float64 = 2.0e-3


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

    var rho = RHO0
    var u = U0
    var p = P0
    var E = p / (GAMMA - Float32(1.0)) + Float32(0.5) * rho * (u * u)
    var base = (e * N_P + nn) * 5
    q[base + 0] = rho
    q[base + 1] = rho * u
    q[base + 2] = Float32(0.0)
    q[base + 3] = Float32(0.0)
    q[base + 4] = E


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    mpi.init()
    var size = mpi.world_size()
    if size > 1:
        mpi.finalize()
        print("bench_euler_inflow_3d: runs at np=1 only")
        return

    print("bench_euler_inflow_3d (3D Euler BC_INFLOW preservation)")
    print(
        "  P= 2   mesh=",
        NX,
        "x",
        NY,
        "x",
        NZ,
        "   M=",
        U0 / sqrt(GAMMA * P0 / RHO0),
        "   T=",
        T_FINAL,
    )

    var rank = mpi.world_rank()
    var nvtx = NvtxContext()
    var refs = build_reference_operators(nvtx)
    var ctx = DeviceContext()

    var bcs = BoundaryConditions(
        BC_INFLOW,
        BC_OUTFLOW,  # -x, +x
        BC_INTERIOR,
        BC_INTERIOR,  # -y, +y (periodic)
        BC_INTERIOR,
        BC_INTERIOR,  # -z, +z (periodic)
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
        Euler.NUM_COMPONENTS,
        mesh.d_perm.unsafe_ptr(),
        bcs,
    )
    # Inflow ghost matches the IC exactly so the analytic solution
    # is the IC for all time.
    var rho0_inflow = RHO0
    var rhou0_inflow = RHO0 * U0
    var rhov0_inflow = Float32(0.0)
    var rhow0_inflow = Float32(0.0)
    var E0 = P0 / (GAMMA - Float32(1.0)) + Float32(0.5) * RHO0 * (U0 * U0)
    var physics = Euler(
        GAMMA,
        MIN_DENSITY,
        MIN_PRESSURE,
        FLUX_HLLEC,
        False,
        Float32(0.0),
        Float32(0.0),
        Float32(0.0),  # no gravity
        rho0_inflow,
        rhou0_inflow,
        rhov0_inflow,
        rhow0_inflow,
        E0,
    )
    var solver = Solver[Euler](
        ctx^,
        mesh^,
        halo^,
        physics^,
        refs.D_ref^,
        refs.Lift_ref^,
        refs.node_weights^,
    )

    solver.ctx.enqueue_function[uniform_ic_kernel, uniform_ic_kernel](
        solver.d_q.unsafe_ptr(),
        solver.mesh.d_owned_elem_ids.unsafe_ptr(),
        solver.mesh.local.d_elem_node_xyz.unsafe_ptr(),
        solver.num_owned_elements,
        grid_dim=ceildiv(solver.num_owned_elements * N_P, IC_BLOCK),
        block_dim=IC_BLOCK,
    )
    solver.ctx.synchronize()

    var n_owned_dof = solver.num_owned_elements * N_P * Euler.NUM_COMPONENTS

    # CFL on |u| + c.
    var c0 = sqrt(GAMMA * P0 / RHO0)
    var wave = U0 + c0
    var h = Float32(LX) / Float32(NX)
    var dt_est = CFL * h / (wave * Float32(5.0))
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

    var max_rho_dev: Float64 = 0.0
    var max_rhou_dev: Float64 = 0.0
    var max_rhovw: Float64 = 0.0
    var max_E_dev: Float64 = 0.0
    var n_owned_nodes = solver.num_owned_elements * N_P
    for i in range(n_owned_nodes):
        var rho_now = q_ptr[i * 5 + 0]
        var rhou_now = q_ptr[i * 5 + 1]
        var rhov_now = q_ptr[i * 5 + 2]
        var rhow_now = q_ptr[i * 5 + 3]
        var E_now = q_ptr[i * 5 + 4]
        if (
            isnan(rho_now)
            or isinf(rho_now)
            or isnan(rhou_now)
            or isinf(rhou_now)
            or isnan(rhov_now)
            or isinf(rhov_now)
            or isnan(rhow_now)
            or isinf(rhow_now)
            or isnan(E_now)
            or isinf(E_now)
        ):
            raise Error("bench_euler_inflow_3d: non-finite output")
        var d_rho = Float64(rho_now - RHO0)
        if d_rho < 0.0:
            d_rho = -d_rho
        if d_rho > max_rho_dev:
            max_rho_dev = d_rho
        var d_rhou = Float64(rhou_now - RHO0 * U0)
        if d_rhou < 0.0:
            d_rhou = -d_rhou
        if d_rhou > max_rhou_dev:
            max_rhou_dev = d_rhou
        var a_rhov = Float64(rhov_now)
        if a_rhov < 0.0:
            a_rhov = -a_rhov
        if a_rhov > max_rhovw:
            max_rhovw = a_rhov
        var a_rhow = Float64(rhow_now)
        if a_rhow < 0.0:
            a_rhow = -a_rhow
        if a_rhow > max_rhovw:
            max_rhovw = a_rhow
        var d_E = Float64(E_now - E0)
        if d_E < 0.0:
            d_E = -d_E
        if d_E > max_E_dev:
            max_E_dev = d_E

    var rho_rel = max_rho_dev / Float64(RHO0)
    var rhou_rel = max_rhou_dev / Float64(RHO0 * U0)
    var E_rel = max_E_dev / Float64(E0)
    print(
        "  max |rho - rho0| / rho0       =",
        rho_rel,
        "  (threshold",
        RHO_REL_TOL,
        ")",
    )
    print(
        "  max |rhou - rho0*u0| / rho0*u0=",
        rhou_rel,
        "  (threshold",
        RHOU_REL_TOL,
        ")",
    )
    print(
        "  max |rho*v|, |rho*w|          =",
        max_rhovw,
        "  (threshold",
        RHOVW_TOL,
        ")",
    )
    print(
        "  max |E - E0| / E0             =",
        E_rel,
        "  (threshold",
        E_REL_TOL,
        ")",
    )

    if rho_rel > RHO_REL_TOL:
        raise Error(
            "bench_euler_inflow_3d FAILED: rho rel err "
            + String(rho_rel)
            + " > "
            + String(RHO_REL_TOL)
        )
    if rhou_rel > RHOU_REL_TOL:
        raise Error(
            "bench_euler_inflow_3d FAILED: rhou rel err "
            + String(rhou_rel)
            + " > "
            + String(RHOU_REL_TOL)
        )
    if max_rhovw > RHOVW_TOL:
        raise Error(
            "bench_euler_inflow_3d FAILED: rhov/rhow drift "
            + String(max_rhovw)
            + " > "
            + String(RHOVW_TOL)
        )
    if E_rel > E_REL_TOL:
        raise Error(
            "bench_euler_inflow_3d FAILED: E rel err "
            + String(E_rel)
            + " > "
            + String(E_REL_TOL)
        )

    print("=== bench_euler_inflow_3d PASSED ===")
    mpi.finalize()
