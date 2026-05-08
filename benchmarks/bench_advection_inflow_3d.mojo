# ======================================================================
# bench_advection_inflow_3d -- 3D BC_INFLOW + BC_OUTFLOW steady-state
# ======================================================================
#
# Wind-tunnel-style advection: q = 0 everywhere at t=0, velocity v =
# (1, 0, 0).  BC_INFLOW with inflow_q = 1 on -x; BC_OUTFLOW on +x;
# BC_WALL on +-y and +-z.  After T >= LX / vx the domain has been
# fully swept by the inflow and q should equal inflow_q at every
# owned node (uniform steady state).
#
# Pass criteria (P=2, NX=NY=NZ=12, T=1.5, single-rank):
#   * max |q - 1| < 5e-3 over all owned nodes (small float32 noise
#     plus DG transient near the inflow front)
#   * mean q within 1e-3 of 1.0 (global equilibrium)
#   * no NaN / Inf
#
# Closes the analytic-solution coverage gap on the 3D BC_INFLOW path.
# Combined with bench_advection_outflow_3d this exercises the three
# non-periodic 3D BCs (INFLOW, OUTFLOW, WALL) that previously had
# only the np-equivalence test (mpi_bc_test) for coverage.
# ======================================================================

from std.sys import has_accelerator
from std.gpu import global_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from std.math import sqrt, ceildiv, isnan, isinf

from src import mpi
from src.partition import build_partition
from src.reference import ReferenceElement, num_tet_nodes, build_reference_operators, N_P
from src.mesh import Mesh
from src.boundary import BoundaryConditions, BC_INFLOW, BC_OUTFLOW, BC_WALL
from src.halo_exchange import HaloExchange
from src.solver import Solver
from src.advection import Advection
from src.nvtx import NvtxContext


comptime LX = 1.0
comptime LY = 1.0
comptime LZ = 1.0
comptime VX: Float32 = 1.0
comptime VY: Float32 = 0.0
comptime VZ: Float32 = 0.0
comptime INFLOW_Q: Float32 = 1.0
comptime T_FINAL: Float32 = 1.5  # > LX / VX
comptime CFL = Float32(0.2)
comptime NX = 12
comptime NY = 12
comptime NZ = 12
comptime IC_BLOCK = 256

# Measured max |q-1| ~9.5e-6, mean error ~1e-6 (essentially Float32
# noise after a 1.5-period sweep).  1e-4 / 1e-5 leave ~10x headroom
# while catching any meaningful regression in the inflow / outflow
# coupling.
comptime Q_TOL: Float32 = Float32(1.0e-4)
comptime MEAN_TOL: Float64 = 1.0e-5


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    mpi.init()
    var size = mpi.world_size()
    if size > 1:
        mpi.finalize()
        print("bench_advection_inflow_3d: runs at np=1 only")
        return

    print("bench_advection_inflow_3d (BC_INFLOW + BC_OUTFLOW + BC_WALL)")
    print("  P=2  mesh=", NX, "x", NY, "x", NZ, "  T=", T_FINAL, "  inflow_q=", INFLOW_Q)

    var rank = mpi.world_rank()
    var nvtx = NvtxContext()
    var refs = build_reference_operators(nvtx)
    var ctx = DeviceContext()

    var bcs = BoundaryConditions(
        BC_INFLOW,
        BC_OUTFLOW,  # x
        BC_WALL,
        BC_WALL,  # y
        BC_WALL,
        BC_WALL,  # z
    )
    var mesh = Mesh(ctx, build_partition(rank, size, NX, NY, NZ), LX, LY, LZ, bcs)
    var halo = HaloExchange(ctx, mesh.part, Advection.NUM_COMPONENTS, mesh.d_perm.unsafe_ptr(), bcs)
    var physics = Advection(VX, VY, VZ, INFLOW_Q)
    var solver = Solver[Advection](ctx^, mesh^, halo^, physics^, refs.D_ref^, refs.Lift_ref^, refs.node_weights^)
    # IC: q = 0 everywhere (default solver state).
    solver.d_q.enqueue_fill(Float32(0.0))
    solver.ctx.synchronize()

    var v_mag = sqrt(VX * VX + VY * VY + VZ * VZ)
    var h = Float32(LX) / Float32(NX)
    var dt_est = CFL * h / (v_mag * Float32(2 * 2 + 1))
    var num_steps = Int(T_FINAL / dt_est) + 1
    var dt = T_FINAL / Float32(num_steps)

    for _ in range(num_steps):
        solver.step_ssprk3(dt, nvtx)
    solver.ctx.synchronize()

    var n_owned = solver.num_owned_elements
    var n_dof = n_owned * N_P
    var rho = List[Float32]()
    for _ in range(n_dof):
        rho.append(Float32(0.0))
    solver.download_owned_component(0, rho, nvtx)

    var max_err: Float32 = 0.0
    var mean_q: Float64 = 0.0
    for k in range(n_dof):
        var v = rho[k]
        if isnan(v) or isinf(v):
            raise Error("bench_advection_inflow_3d: non-finite output")
        var d = v - INFLOW_Q
        var ad = d if d >= Float32(0.0) else -d
        if ad > max_err:
            max_err = ad
        mean_q += Float64(v)
    mean_q /= Float64(n_dof)

    var mean_err = mean_q - Float64(INFLOW_Q)
    if mean_err < 0.0:
        mean_err = -mean_err
    print("  max |q - inflow_q| =", max_err, "  mean q =", mean_q, "  |mean - 1| =", mean_err)

    if max_err > Q_TOL:
        raise Error(String("bench_advection_inflow_3d FAILED: max |q-1| ") + String(max_err) + " > " + String(Q_TOL))
    if mean_err > MEAN_TOL:
        raise Error(String("bench_advection_inflow_3d FAILED: |<q>-1| ") + String(mean_err) + " > " + String(MEAN_TOL))

    print("=== bench_advection_inflow_3d PASSED ===")
    mpi.finalize()
