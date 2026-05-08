# ======================================================================
# bench_two_fluid_inflow_3d -- 3D Two-Fluid BC_INFLOW + BC_OUTFLOW preservation
# ======================================================================
#
# Direct gate for the 17-component BC_INFLOW arm of
# FiveMomentTwoFluid.boundary_flux (src/two_fluid.mojo line 522),
# the highest-NC BC dispatch path in the suite.  Without this
# bench the BC_INFLOW arm would be reachable from no test or
# bench, so a regression in the inflow-ghost write (writing the
# wrong 17 components, or one of the 17 inflow_* fields not being
# threaded through to the kernel) would silently pass.
#
# Cleanest test: charge-balanced rest state under BC_INFLOW on -x
# and BC_OUTFLOW on +x (periodic in y/z).  The 17 `inflow_*` fields
# of the FiveMomentTwoFluid struct are matched to the IC: charge-
# balanced uniform plasma (q_e * n_e + q_i * n_i = 0, all velocities
# = 0, E = B = 0, psi = 0).  Inflow ghost == interior == IC, so all
# Rusanov dissipation vanishes at every face and the boundary flux
# divergence integrates to zero around each tet.  Net RHS = 0; state
# stays at rest indefinitely.
#
# Mirrors the pattern used by bench_mhd_inflow_3d (matched-state
# inflow + outflow, sub-Alfvenic / quiescent uniform IC) but at NC=17
# instead of NC=9.  Companion to bench_two_fluid_walls_3d (BC_WALL)
# and bench_two_fluid_outflow_3d (BC_OUTFLOW); together the three
# benches now exercise all four BC dispatch arms in
# FiveMomentTwoFluid.boundary_flux.
#
# Pass criteria (P=2, NX=NY=NZ=4, T=0.5):
#   * max |q - q_IC| < 5e-4 (Float32 epsilon * step accumulation;
#     same threshold as the BC_OUTFLOW / BC_WALL analogs since 17
#     components compound roundoff at the same rate)
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
from src.boundary import (
    BoundaryConditions,
    BC_INFLOW,
    BC_OUTFLOW,
    BC_INTERIOR,
)
from src.halo_exchange import HaloExchange
from src.solver import Solver
from src.two_fluid import FiveMomentTwoFluid
from src.nvtx import NvtxContext


comptime NX = 4
comptime NY = 4
comptime NZ = 4
comptime LX = 1.0
comptime LY = 1.0
comptime LZ = 1.0
comptime IC_BLOCK = 256

comptime GAMMA_E: Float32 = Float32(5.0 / 3.0)
comptime GAMMA_I: Float32 = Float32(5.0 / 3.0)
comptime Q_E: Float32 = -1.0
comptime M_E: Float32 = 1.0
comptime Q_I: Float32 = 1.0
comptime M_I: Float32 = 25.0
comptime EPS0: Float32 = 1.0
comptime C_LIGHT: Float32 = 10.0
comptime C_H: Float32 = Float32(0.0)
comptime ALPHA_D: Float32 = Float32(0.0)
comptime MIN_DENSITY: Float32 = Float32(1.0e-6)
comptime MIN_PRESSURE: Float32 = Float32(1.0e-6)

comptime N0: Float32 = 1.0
comptime P_E0: Float32 = 0.01
comptime P_I0: Float32 = 0.01

comptime CFL: Float32 = Float32(0.1)
comptime T_FINAL: Float32 = 0.5
comptime DRIFT_TOL: Float64 = 5.0e-4


def fill_constant_kernel(
    q: UnsafePointer[Float32, MutAnyOrigin],
    owned_elem_ids: UnsafePointer[Int32, MutAnyOrigin],
    num_owned: Int,
):
    var idx = Int(global_idx.x)
    var total = num_owned * N_P
    if idx >= total:
        return
    var i = idx // N_P
    var nn = idx % N_P
    var e = Int(owned_elem_ids[i])
    var base = (e * N_P + nn) * 17
    var rho_e = M_E * N0
    var rho_i = M_I * N0
    var E_e = P_E0 / (GAMMA_E - Float32(1.0))
    var E_i = P_I0 / (GAMMA_I - Float32(1.0))
    # Electrons (0..4)
    q[base + 0] = rho_e
    q[base + 1] = Float32(0.0)
    q[base + 2] = Float32(0.0)
    q[base + 3] = Float32(0.0)
    q[base + 4] = E_e
    # Ions (5..9)
    q[base + 5] = rho_i
    q[base + 6] = Float32(0.0)
    q[base + 7] = Float32(0.0)
    q[base + 8] = Float32(0.0)
    q[base + 9] = E_i
    # Maxwell E (10..12), B (13..15), psi (16) all zero
    q[base + 10] = Float32(0.0)
    q[base + 11] = Float32(0.0)
    q[base + 12] = Float32(0.0)
    q[base + 13] = Float32(0.0)
    q[base + 14] = Float32(0.0)
    q[base + 15] = Float32(0.0)
    q[base + 16] = Float32(0.0)


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    mpi.init()
    var size = mpi.world_size()
    if size > 1:
        mpi.finalize()
        print("bench_two_fluid_inflow_3d: runs at np=1 only")
        return

    print("bench_two_fluid_inflow_3d (3D Two-Fluid BC_INFLOW + BC_OUTFLOW)")
    print("  P= 2   mesh=", NX, "x", NY, "x", NZ, "   T=", T_FINAL)

    var rank = mpi.world_rank()
    var nvtx = NvtxContext()
    var refs = build_reference_operators(nvtx)
    var ctx = DeviceContext()

    # Inflow on -x, outflow on +x, periodic in y/z (matches
    # bench_mhd_inflow_3d's BC layout).
    var bcs = BoundaryConditions(
        BC_INFLOW,
        BC_OUTFLOW,  # -x, +x
        BC_INTERIOR,
        BC_INTERIOR,  # -y, +y
        BC_INTERIOR,
        BC_INTERIOR,  # -z, +z
    )
    var mesh = Mesh(
        ctx=ctx,
        part=build_partition(rank=rank, nprocs=size, nx=NX, ny=NY, nz=NZ),
        Lx=LX,
        Ly=LY,
        Lz=LZ,
        bcs=bcs,
    )
    var halo = HaloExchange(
        ctx=ctx,
        part=mesh.part,
        nc=FiveMomentTwoFluid.NUM_COMPONENTS,
        d_perm=mesh.d_perm.unsafe_ptr(),
        bcs=bcs,
    )

    # Inflow ghost = IC (charge-balanced rest state).
    var rho_e0 = M_E * N0
    var rho_i0 = M_I * N0
    var E_e0 = P_E0 / (GAMMA_E - Float32(1.0))
    var E_i0 = P_I0 / (GAMMA_I - Float32(1.0))

    var physics = FiveMomentTwoFluid(
        gamma_e=GAMMA_E,
        gamma_i=GAMMA_I,
        q_e=Q_E,
        m_e=M_E,
        q_i=Q_I,
        m_i=M_I,
        eps0=EPS0,
        c_light=C_LIGHT,
        c_h=C_H,
        alpha_d=ALPHA_D,
        min_density=MIN_DENSITY,
        min_pressure=MIN_PRESSURE,
        inflow_rho_e=rho_e0,
        inflow_E_e=E_e0,
        inflow_rho_i=rho_i0,
        inflow_E_i=E_i0,
        # All momenta, E-field, B-field, psi default to 0 -- matching
        # the IC's quiescent rest state.
    )
    var solver = Solver[FiveMomentTwoFluid](
        ctx^,
        mesh^,
        halo^,
        physics^,
        refs.D_ref^,
        refs.Lift_ref^,
        refs.node_weights^,
    )

    solver.ctx.enqueue_function[fill_constant_kernel, fill_constant_kernel](
        solver.d_q.unsafe_ptr(),
        solver.mesh.d_owned_elem_ids.unsafe_ptr(),
        solver.num_owned_elements,
        grid_dim=ceildiv(solver.num_owned_elements * N_P, IC_BLOCK),
        block_dim=IC_BLOCK,
    )
    solver.ctx.synchronize()

    var n_owned_dof = solver.num_owned_elements * N_P * 17
    var hbuf_ic = solver.ctx.enqueue_create_host_buffer[DType.float32](n_owned_dof)
    solver.ctx.enqueue_copy(hbuf_ic, solver.d_q.create_sub_buffer[DType.float32](0, n_owned_dof))
    solver.ctx.synchronize()
    var ic_ptr = hbuf_ic.unsafe_ptr()
    var host_ic = List[Float32]()
    for k in range(n_owned_dof):
        host_ic.append(ic_ptr[k])

    var h = Float32(LX) / Float32(NX)
    var dt_est = CFL * h / (C_LIGHT * Float32(5.0))
    var num_steps = Int(T_FINAL / dt_est) + 1
    var dt = T_FINAL / Float32(num_steps)
    print("  steps=", num_steps, "  dt=", dt)

    for _ in range(num_steps):
        solver.step_ssprk3(dt, nvtx)
    solver.ctx.synchronize()

    var hbuf_q = solver.ctx.enqueue_create_host_buffer[DType.float32](n_owned_dof)
    solver.ctx.enqueue_copy(hbuf_q, solver.d_q.create_sub_buffer[DType.float32](0, n_owned_dof))
    solver.ctx.synchronize()
    var q_ptr = hbuf_q.unsafe_ptr()

    var max_drift: Float64 = 0.0
    for k in range(n_owned_dof):
        var v = q_ptr[k]
        if isnan(v) or isinf(v):
            raise Error("bench_two_fluid_inflow_3d: non-finite output")
        var d = Float64(v - host_ic[k])
        if d < 0.0:
            d = -d
        if d > max_drift:
            max_drift = d

    print("  max |q - q_IC| =", max_drift, "  (threshold", DRIFT_TOL, ")")
    if max_drift > DRIFT_TOL:
        raise Error("bench_two_fluid_inflow_3d FAILED: drift " + String(max_drift) + " > " + String(DRIFT_TOL))

    print("=== bench_two_fluid_inflow_3d PASSED ===")
    mpi.finalize()
