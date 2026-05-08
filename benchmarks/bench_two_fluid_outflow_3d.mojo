# ======================================================================
# bench_two_fluid_outflow_3d -- 3D Two-Fluid BC_OUTFLOW preservation
# ======================================================================
#
# Direct gate for the BC_OUTFLOW dispatch arm of
# `FiveMomentTwoFluid.boundary_flux` (src/two_fluid.mojo line 487).
# Companion to bench_two_fluid_walls_3d (BC_WALL) and
# bench_two_fluid_inflow_3d (BC_INFLOW); together the three benches
# exercise all four BC dispatch arms in the highest-NC physics
# module (NC=17).
#
# Cleanest test: charge-balanced rest state under BC_OUTFLOW on all
# six faces.  q_e * n_e + q_i * n_i = 0, all velocities = 0, no E,
# no B, no psi -> no Lorentz force, no Ampere current source, no
# divB -> the state must remain at rest indefinitely.  Under
# BC_OUTFLOW the ghost equals the interior, the Rusanov dissipation
# vanishes at every face, and the constant boundary flux integrates
# to zero around each tet (discrete divergence theorem).  Net RHS = 0.
#
# Pass criteria (P=2, NX=NY=NZ=4, T=0.5):
#   * max |q - q_IC| < 5e-4 (Float32 epsilon * step accumulation;
#     17 components compound roundoff faster than the 6-component
#     Maxwell BC_OUTFLOW analog so threshold is 5x looser)
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
        print("bench_two_fluid_outflow_3d: runs at np=1 only")
        return

    print("bench_two_fluid_outflow_3d (3D Two-Fluid BC_OUTFLOW preservation)")
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
        FiveMomentTwoFluid.NUM_COMPONENTS,
        mesh.d_perm.unsafe_ptr(),
        bcs,
    )
    var physics = FiveMomentTwoFluid(
        GAMMA_E,
        GAMMA_I,
        Q_E,
        M_E,
        Q_I,
        M_I,
        EPS0,
        C_LIGHT,
        C_H,
        ALPHA_D,
        MIN_DENSITY,
        MIN_PRESSURE,
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
            raise Error("bench_two_fluid_outflow_3d: non-finite output")
        var d = Float64(v - host_ic[k])
        if d < 0.0:
            d = -d
        if d > max_drift:
            max_drift = d

    print("  max |q - q_IC| =", max_drift, "  (threshold", DRIFT_TOL, ")")
    if max_drift > DRIFT_TOL:
        raise Error("bench_two_fluid_outflow_3d FAILED: drift " + String(max_drift) + " > " + String(DRIFT_TOL))

    print("=== bench_two_fluid_outflow_3d PASSED ===")
    mpi.finalize()
