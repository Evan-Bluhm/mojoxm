# ======================================================================
# bench_two_fluid_walls_3d_p4 -- 3D Two-Fluid BC_WALL preservation at P=4
# ======================================================================
#
# P=4 (NP=35 nodes per tet) counterpart of bench_two_fluid_walls_3d_p3.
# Mesh shrunk to 2x2x2 since per-element volume work scales with NP^2
# and dt tightens to 1/(2P+1)=1/9 at P=4.  At NC=17 and NP=35 the
# per-element work is ~3.5x P=3's; the 1/8 mesh shrink keeps total
# wall-clock comparable.
#
# Pass criteria (P=4, NX=NY=NZ=2, T=0.5):
#   * max |q - q_IC| < 5e-4
#   * no NaN / Inf
# ======================================================================

from std.sys import has_accelerator
from std.gpu import global_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from std.math import ceildiv, isnan, isinf

from src import mpi
from src.partition import build_partition
from src.reference import ReferenceElement, to_float32, num_tet_nodes
from src.mesh import Mesh
from src.boundary import BoundaryConditions, BC_WALL
from src.halo_exchange import HaloExchange
from src.solver import Solver
from src.two_fluid import FiveMomentTwoFluid
from src.nvtx import NvtxContext


comptime P = 4
comptime NP = num_tet_nodes(P)        # 35 at P=4
comptime NC = 17

comptime NX = 2
comptime NY = 2
comptime NZ = 2
comptime LX = 1.0
comptime LY = 1.0
comptime LZ = 1.0
comptime IC_BLOCK = 256

comptime GAMMA_E: Float32 = Float32(5.0 / 3.0)
comptime GAMMA_I: Float32 = Float32(5.0 / 3.0)
comptime Q_E:     Float32 = -1.0
comptime M_E:     Float32 =  1.0
comptime Q_I:     Float32 =  1.0
comptime M_I:     Float32 = 25.0
comptime EPS0:    Float32 =  1.0
comptime C_LIGHT: Float32 = 10.0
comptime C_H:     Float32 = Float32(0.0)
comptime ALPHA_D: Float32 = Float32(0.0)
comptime MIN_DENSITY:  Float32 = Float32(1.0e-6)
comptime MIN_PRESSURE: Float32 = Float32(1.0e-6)

comptime N0:    Float32 = 1.0
comptime P_E0:  Float32 = 0.01
comptime P_I0:  Float32 = 0.01

comptime CFL: Float32 = Float32(0.1)
comptime T_FINAL: Float32 = 0.5
comptime DRIFT_TOL: Float64 = 5.0e-4


def fill_constant_kernel(
    q: UnsafePointer[Float32, MutAnyOrigin],
    owned_elem_ids: UnsafePointer[Int32, MutAnyOrigin],
    num_owned: Int,
):
    var idx = Int(global_idx.x)
    var total = num_owned * NP
    if idx >= total:
        return
    var i = idx // NP
    var nn = idx % NP
    var e = Int(owned_elem_ids[i])
    var base = (e * NP + nn) * NC
    var rho_e = M_E * N0
    var rho_i = M_I * N0
    var E_e = P_E0 / (GAMMA_E - Float32(1.0))
    var E_i = P_I0 / (GAMMA_I - Float32(1.0))
    q[base + 0] = rho_e
    q[base + 1] = Float32(0.0)
    q[base + 2] = Float32(0.0)
    q[base + 3] = Float32(0.0)
    q[base + 4] = E_e
    q[base + 5] = rho_i
    q[base + 6] = Float32(0.0)
    q[base + 7] = Float32(0.0)
    q[base + 8] = Float32(0.0)
    q[base + 9] = E_i
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
        print("bench_two_fluid_walls_3d_p4: runs at np=1 only")
        return

    print("bench_two_fluid_walls_3d_p4 (3D Two-Fluid BC_WALL preservation, P=4)")
    print("  P=", P, "  NP=", NP, "  mesh=", NX, "x", NY, "x", NZ,
          "   T=", T_FINAL)

    var rank = mpi.world_rank()
    var nvtx = NvtxContext()
    var ctx = DeviceContext()

    var re = ReferenceElement[P]()
    var D_ref = to_float32(re.D_ref)
    var Lift_ref = to_float32(re.Lift_ref)
    var node_weights = to_float32(re.node_weights)

    var bcs = BoundaryConditions(
        BC_WALL, BC_WALL,
        BC_WALL, BC_WALL,
        BC_WALL, BC_WALL,
    )
    var mesh = Mesh[P](
        ctx=ctx,
        part=build_partition(rank=rank, nprocs=size, nx=NX, ny=NY, nz=NZ),
        Lx=LX, Ly=LY, Lz=LZ, bcs=bcs,
    )
    var halo = HaloExchange(
        ctx=ctx, part=mesh.part, nc=FiveMomentTwoFluid.NUM_COMPONENTS,
        d_perm=mesh.d_perm.unsafe_ptr(), bcs=bcs,
    )
    var physics = FiveMomentTwoFluid(
        gamma_e=GAMMA_E, gamma_i=GAMMA_I,
        q_e=Q_E, m_e=M_E, q_i=Q_I, m_i=M_I,
        eps0=EPS0, c_light=C_LIGHT,
        c_h=C_H, alpha_d=ALPHA_D,
        min_density=MIN_DENSITY, min_pressure=MIN_PRESSURE,
    )
    var solver = Solver[FiveMomentTwoFluid, P](
        ctx=ctx^, mesh=mesh^, halo=halo^, physics=physics^,
        D_ref=D_ref^, Lift_ref=Lift_ref^, node_weights=node_weights^,
    )

    solver.ctx.enqueue_function[fill_constant_kernel, fill_constant_kernel](
        solver.d_q.unsafe_ptr(),
        solver.mesh.d_owned_elem_ids.unsafe_ptr(),
        solver.num_owned_elements,
        grid_dim=ceildiv(solver.num_owned_elements * NP, IC_BLOCK),
        block_dim=IC_BLOCK,
    )
    solver.ctx.synchronize()

    var n_owned_dof = solver.num_owned_elements * NP * NC
    var hbuf_ic = solver.ctx.enqueue_create_host_buffer[DType.float32](
        n_owned_dof
    )
    solver.ctx.enqueue_copy(
        hbuf_ic, solver.d_q.create_sub_buffer[DType.float32](0, n_owned_dof)
    )
    solver.ctx.synchronize()
    var ic_ptr = hbuf_ic.unsafe_ptr()
    var host_ic = List[Float32]()
    for k in range(n_owned_dof):
        host_ic.append(ic_ptr[k])

    var h = Float32(LX) / Float32(NX)
    var dt_est = CFL * h / (C_LIGHT * Float32(2 * P + 1))
    var num_steps = Int(T_FINAL / dt_est) + 1
    var dt = T_FINAL / Float32(num_steps)
    print("  steps=", num_steps, "  dt=", dt)

    for _ in range(num_steps):
        solver.step_ssprk3(dt, nvtx)
    solver.ctx.synchronize()

    var hbuf_q = solver.ctx.enqueue_create_host_buffer[DType.float32](n_owned_dof)
    solver.ctx.enqueue_copy(
        hbuf_q, solver.d_q.create_sub_buffer[DType.float32](0, n_owned_dof)
    )
    solver.ctx.synchronize()
    var q_ptr = hbuf_q.unsafe_ptr()

    var max_drift: Float64 = 0.0
    for k in range(n_owned_dof):
        var v = q_ptr[k]
        if isnan(v) or isinf(v):
            raise Error("bench_two_fluid_walls_3d_p4: non-finite output")
        var d = Float64(v) - Float64(host_ic[k])
        if d < 0.0: d = -d
        if d > max_drift: max_drift = d

    print("  max |q - q_IC| =", max_drift,
          "  (threshold", DRIFT_TOL, ")")

    if max_drift > DRIFT_TOL:
        raise Error(
            "bench_two_fluid_walls_3d_p4 FAILED: max drift "
            + String(max_drift) + " > " + String(DRIFT_TOL)
        )

    print("=== bench_two_fluid_walls_3d_p4 PASSED ===")
    mpi.finalize()
