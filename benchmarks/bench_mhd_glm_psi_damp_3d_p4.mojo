# ======================================================================
# bench_mhd_glm_psi_damp_3d_p4 -- 3D GLM exp-decay rate gate at P=4
# ======================================================================
#
# P=4 (NP=35) intermediate between bench_mhd_glm_psi_damp_3d_p3
# (NP=20) and bench_mhd_glm_psi_damp_3d_p5 (NP=56).  Same uniform IC
# and same alpha_d>0 / c_h=0 setup, exact-exponential analytic decay
# psi(T) = A0 * exp(-alpha_d * T), routed through Mesh[4] /
# Solver[IdealMHD, 4] / rk_stage_kernel[4].  Mesh 3x3x3 between
# P=3's 4x4x4 and P=5's 2x2x2.
#
# Closes the last 3D GLM damp P-parity hole.  Sweep is now
# P=2/P=3/P=4/P=5 across both 2D and 3D for the GLM damp rate gate.
# Pairs with bench_mhd_glm_psi_transport_3d_p4 for the full set of
# 3D MHD GLM P=4 rate gates.
#
# Pass criteria (P=4, periodic 3x3x3, T=1, alpha_d=1, A0=0.1):
#   * |psi(T) - A0/e| / A0 < 1e-3 at every owned node
#   * non-psi state unchanged from IC to ~Float32 epsilon
#   * no NaN / Inf
# ======================================================================

from std.sys import has_accelerator
from std.gpu import global_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from std.math import sqrt, ceildiv, exp, isnan, isinf

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
comptime NX = 3
comptime NY = 3
comptime NZ = 3
comptime LX = 1.0
comptime LY = 1.0
comptime LZ = 1.0

comptime GAMMA = Float32(5.0 / 3.0)
comptime RHO0 = Float32(1.0)
comptime P0_GAS = Float32(1.0)
comptime B0 = Float32(1.0)
comptime A0 = Float32(0.1)
comptime C_H = Float32(0.0)
comptime ALPHA_D = Float32(1.0)
comptime MIN_DENSITY = Float32(1.0e-6)
comptime MIN_PRESSURE = Float32(1.0e-6)
comptime T_FINAL: Float32 = 1.0
comptime CFL = Float32(0.10)
comptime IC_BLOCK = 256

comptime PSI_TOL_REL: Float64 = 1.0e-3
comptime STATE_TOL: Float64 = 1.0e-4


def damp_ic_kernel_p4(
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

    var rho = RHO0
    var p_gas = P0_GAS
    var bx = B0
    var by = Float32(0.0)
    var bz = Float32(0.0)
    var E = p_gas / (GAMMA - Float32(1.0)) + Float32(0.5) * (
        bx * bx + by * by + bz * bz
    )

    var base = (e * NP + nn) * 9
    q[base + 0] = rho
    q[base + 1] = Float32(0.0)
    q[base + 2] = Float32(0.0)
    q[base + 3] = Float32(0.0)
    q[base + 4] = E
    q[base + 5] = bx
    q[base + 6] = by
    q[base + 7] = bz
    q[base + 8] = A0


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    mpi.init()
    var size = mpi.world_size()
    if size > 1:
        mpi.finalize()
        print("bench_mhd_glm_psi_damp_3d_p4: runs at np=1 only")
        return

    print("bench_mhd_glm_psi_damp_3d_p4 (3D GLM psi damping at P=4)")
    print(
        "  P=",
        P,
        "  NP=",
        NP,
        "  mesh=",
        NX,
        "x",
        NY,
        "x",
        NZ,
        "   alpha_d=",
        ALPHA_D,
        "   T=",
        T_FINAL,
    )

    var rank = mpi.world_rank()
    var nvtx = NvtxContext()
    var ctx = DeviceContext()

    var re = ReferenceElement[P]()
    var D_ref = to_float32(re.D_ref)
    var Lift_ref = to_float32(re.Lift_ref)
    var node_weights = to_float32(re.node_weights)

    var bcs = BoundaryConditions.periodic()
    var mesh = Mesh[P](
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
        nc=IdealMHD.NUM_COMPONENTS,
        d_perm=mesh.d_perm.unsafe_ptr(),
        bcs=bcs,
    )
    var physics = IdealMHD(
        gamma=GAMMA,
        min_density=MIN_DENSITY,
        min_pressure=MIN_PRESSURE,
        c_h=C_H,
        alpha_d=ALPHA_D,
    )
    var solver = Solver[IdealMHD, P](
        ctx=ctx^,
        mesh=mesh^,
        halo=halo^,
        physics=physics^,
        D_ref=D_ref^,
        Lift_ref=Lift_ref^,
        node_weights=node_weights^,
    )

    solver.ctx.enqueue_function[damp_ic_kernel_p4, damp_ic_kernel_p4](
        solver.d_q.unsafe_ptr(),
        solver.mesh.d_owned_elem_ids.unsafe_ptr(),
        solver.mesh.local.d_elem_node_xyz.unsafe_ptr(),
        solver.num_owned_elements,
        grid_dim=ceildiv(solver.num_owned_elements * NP, IC_BLOCK),
        block_dim=IC_BLOCK,
    )
    solver.ctx.synchronize()

    var n_owned_dof = solver.num_owned_elements * NP * IdealMHD.NUM_COMPONENTS
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

    var cs2 = GAMMA * P0_GAS / RHO0
    var ca2 = (B0 * B0) / RHO0
    var cf = sqrt(cs2 + ca2)
    var h = Float32(LX) / Float32(NX)
    var dt_est = CFL * h / (cf * Float32(2 * P + 1))
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
        hbuf_q, solver.d_q.create_sub_buffer[DType.float32](0, n_owned_dof)
    )
    solver.ctx.synchronize()
    var q_ptr = hbuf_q.unsafe_ptr()

    var psi_exact = Float64(A0) * exp(-Float64(ALPHA_D) * Float64(T_FINAL))

    var max_psi_dev: Float64 = 0.0
    var max_state_drift: Float64 = 0.0
    var n_owned_nodes = solver.num_owned_elements * NP
    for i in range(n_owned_nodes):
        for c in range(9):
            var v = q_ptr[i * 9 + c]
            if isnan(v) or isinf(v):
                raise Error("bench_mhd_glm_psi_damp_3d_p4: non-finite output")
            if c == 8:
                var psi_dev = Float64(v) - psi_exact
                if psi_dev < 0.0:
                    psi_dev = -psi_dev
                if psi_dev > max_psi_dev:
                    max_psi_dev = psi_dev
            else:
                var dv = Float64(v - host_ic[i * 9 + c])
                if dv < 0.0:
                    dv = -dv
                if dv > max_state_drift:
                    max_state_drift = dv

    var psi_rel = max_psi_dev / Float64(A0)
    print("  psi exact      =", psi_exact)
    print(
        "  max |psi - exact| / A0 =", psi_rel, "  (threshold", PSI_TOL_REL, ")"
    )
    print(
        "  max state drift (rho, momenta, B, E) =",
        max_state_drift,
        "  (threshold",
        STATE_TOL,
        ")",
    )

    if psi_rel > PSI_TOL_REL:
        raise Error(
            "bench_mhd_glm_psi_damp_3d_p4 FAILED: psi rel err "
            + String(psi_rel)
            + " > "
            + String(PSI_TOL_REL)
        )
    if max_state_drift > STATE_TOL:
        raise Error(
            "bench_mhd_glm_psi_damp_3d_p4 FAILED: state drift "
            + String(max_state_drift)
            + " > "
            + String(STATE_TOL)
        )

    print("=== bench_mhd_glm_psi_damp_3d_p4 PASSED ===")
    mpi.finalize()
