# ======================================================================
# bench_mhd_glm_psi_damp_3d -- 3D GLM psi damping via source_term
# ======================================================================
#
# 3D analog of bench_mhd_glm_psi_damp_2d.  The 2D path uses a
# standalone operator-split damp kernel (`launch_mhd_glm_psi_damp_2d`)
# applied once per timestep AFTER the 3 SSPRK3 stages.  The 3D path
# instead bakes psi-damping into the cooperative `rk_stage_kernel`
# via the `IdealMHD.source_term` hook (src/mhd.mojo line 374), which
# writes `source_out[8] = -alpha_d * q[8]` directly into the RK rhs.
# Different code path -- needs its own gate.
#
# Setup: uniform IC with rho=1, p=1, u=v=w=0, B=(B0, 0, 0)  (so
# div B = 0 identically), psi = A0 uniformly.  All spatial gradients
# vanish, so the GLM transport (c_h^2 div B / -grad psi) contributes
# nothing.  Only the source term acts:
#
#   dpsi/dt = -alpha_d * psi  ->  psi(T) = A0 * exp(-alpha_d * T)
#
# Pass criteria (P=2, periodic 4x4x4, T=1, alpha_d=1, A0=0.1):
#   * |psi(T) - A0/e| / A0 < 1e-3 at every owned node
#   * non-psi state unchanged from IC (rho, momenta, B, E)
#   * no NaN / Inf
#
# Note on 2D vs 3D analytic match:
#   * 2D: damping is exact each step (q[psi] *= exp(-alpha_d*dt));
#     the bench measures the IC * exp(-alpha_d*T) with only Float32
#     roundoff error.
#   * 3D: damping is a continuous source through SSPRK3, so the
#     numerical decay deviates from the exact exponential by O(dt^3)
#     per step.  For alpha_d*dt small the cumulative error is still
#     well below the 1e-3 threshold.
# ======================================================================

from std.sys import has_accelerator
from std.gpu import global_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from std.math import sqrt, ceildiv, exp, isnan, isinf

from src import mpi
from src.partition import build_partition
from src.reference import N_P, build_reference_operators
from src.mesh import Mesh
from src.boundary import BoundaryConditions
from src.halo_exchange import HaloExchange
from src.solver import Solver
from src.mhd import IdealMHD
from src.nvtx import NvtxContext


comptime NX = 4
comptime NY = 4
comptime NZ = 4
comptime LX = 1.0
comptime LY = 1.0
comptime LZ = 1.0

comptime GAMMA     = Float32(5.0 / 3.0)
comptime RHO0      = Float32(1.0)
comptime P0_GAS    = Float32(1.0)
comptime B0        = Float32(1.0)
comptime A0        = Float32(0.1)
comptime C_H       = Float32(0.0)         # no transport
comptime ALPHA_D   = Float32(1.0)         # damping rate
comptime MIN_DENSITY  = Float32(1.0e-6)
comptime MIN_PRESSURE = Float32(1.0e-6)
comptime T_FINAL: Float32 = 1.0           # so psi(T) = A0/e
comptime CFL = Float32(0.10)
comptime IC_BLOCK = 256

# Measured rel err ~4e-5 on current code (SSPRK3 source-term decay
# vs exact exponential, dt ~ 0.005, alpha_d*dt ~ 0.005).  1e-3
# leaves margin for any meaningful regression.
comptime PSI_TOL_REL: Float64 = 1.0e-3
# Non-psi state drift floor (Float32 epsilon over ~200 steps).
comptime STATE_TOL: Float64 = 1.0e-4


def damp_ic_kernel(
    q: UnsafePointer[Float32, MutAnyOrigin],
    owned_elem_ids: UnsafePointer[Int32, MutAnyOrigin],
    elem_node_xyz:  UnsafePointer[Float32, MutAnyOrigin],
    num_owned: Int,
):
    var idx = Int(global_idx.x)
    var total = num_owned * N_P
    if idx >= total:
        return
    var i = idx // N_P
    var nn = idx % N_P
    var e = Int(owned_elem_ids[i])

    # Uniform background: rho=1, u=v=w=0, B=(B0, 0, 0), p=P0_GAS.
    # E_total = p_gas/(gamma-1) + 0.5*rho*v^2 + 0.5*B^2.
    var rho = RHO0
    var p_gas = P0_GAS
    var bx = B0
    var by = Float32(0.0)
    var bz = Float32(0.0)
    var E = (
        p_gas / (GAMMA - Float32(1.0))
        + Float32(0.5) * (bx * bx + by * by + bz * bz)
    )

    var base = (e * N_P + nn) * 9
    q[base + 0] = rho
    q[base + 1] = Float32(0.0)
    q[base + 2] = Float32(0.0)
    q[base + 3] = Float32(0.0)
    q[base + 4] = E
    q[base + 5] = bx
    q[base + 6] = by
    q[base + 7] = bz
    q[base + 8] = A0    # psi


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    mpi.init()
    var size = mpi.world_size()
    if size > 1:
        mpi.finalize()
        print("bench_mhd_glm_psi_damp_3d: runs at np=1 only")
        return

    print("bench_mhd_glm_psi_damp_3d (3D GLM psi damping via source_term)")
    print("  P= 2   mesh=", NX, "x", NY, "x", NZ,
          "   alpha_d=", ALPHA_D, "   T=", T_FINAL)

    var rank = mpi.world_rank()
    var nvtx = NvtxContext()
    var refs = build_reference_operators(nvtx)
    var ctx = DeviceContext()

    var bcs = BoundaryConditions.periodic()
    var mesh = Mesh(
        ctx, build_partition(rank, size, NX, NY, NZ), LX, LY, LZ, bcs,
    )
    var halo = HaloExchange(
        ctx, mesh.part, IdealMHD.NUM_COMPONENTS,
        mesh.d_perm.unsafe_ptr(), bcs,
    )
    var physics = IdealMHD(
        GAMMA, MIN_DENSITY, MIN_PRESSURE, C_H, ALPHA_D,
    )
    var solver = Solver[IdealMHD](
        ctx^, mesh^, halo^, physics^,
        refs.D_ref^, refs.Lift_ref^, refs.node_weights^,
    )

    solver.ctx.enqueue_function[damp_ic_kernel, damp_ic_kernel](
        solver.d_q.unsafe_ptr(),
        solver.mesh.d_owned_elem_ids.unsafe_ptr(),
        solver.mesh.local.d_elem_node_xyz.unsafe_ptr(),
        solver.num_owned_elements,
        grid_dim=ceildiv(solver.num_owned_elements * N_P, IC_BLOCK),
        block_dim=IC_BLOCK,
    )
    solver.ctx.synchronize()

    var n_owned_dof = solver.num_owned_elements * N_P * IdealMHD.NUM_COMPONENTS
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

    # CFL on the magnetosonic speed.  With B0=1 and rho=1 the Alfven
    # speed is 1; sound speed = sqrt(gamma*p/rho) = sqrt(5/3) ~= 1.29.
    # Fast magnetosonic upper bound c_f^2 = c_s^2 + c_a^2.
    var cs2 = GAMMA * P0_GAS / RHO0
    var ca2 = (B0 * B0) / RHO0
    var cf = sqrt(cs2 + ca2)
    var h = Float32(LX) / Float32(NX)
    var dt_est = CFL * h / (cf * Float32(5.0))
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

    # Analytic psi after T: A0 * exp(-alpha_d * T).
    var psi_exact = Float64(A0) * exp(-Float64(ALPHA_D) * Float64(T_FINAL))

    var max_psi_dev: Float64 = 0.0
    var max_state_drift: Float64 = 0.0
    var n_owned_nodes = solver.num_owned_elements * N_P
    for i in range(n_owned_nodes):
        for c in range(9):
            var v = q_ptr[i * 9 + c]
            if isnan(v) or isinf(v):
                raise Error("bench_mhd_glm_psi_damp_3d: non-finite output")
            if c == 8:
                var psi_dev = Float64(v) - psi_exact
                if psi_dev < 0.0: psi_dev = -psi_dev
                if psi_dev > max_psi_dev: max_psi_dev = psi_dev
            else:
                var dv = Float64(v - host_ic[i * 9 + c])
                if dv < 0.0: dv = -dv
                if dv > max_state_drift: max_state_drift = dv

    var psi_rel = max_psi_dev / Float64(A0)
    print("  psi exact      =", psi_exact)
    print("  max |psi - exact| / A0 =", psi_rel,
          "  (threshold", PSI_TOL_REL, ")")
    print("  max state drift (rho, momenta, B, E) =", max_state_drift,
          "  (threshold", STATE_TOL, ")")

    if psi_rel > PSI_TOL_REL:
        raise Error(
            "bench_mhd_glm_psi_damp_3d FAILED: psi rel err "
            + String(psi_rel) + " > " + String(PSI_TOL_REL)
        )
    if max_state_drift > STATE_TOL:
        raise Error(
            "bench_mhd_glm_psi_damp_3d FAILED: state drift "
            + String(max_state_drift) + " > " + String(STATE_TOL)
        )

    print("=== bench_mhd_glm_psi_damp_3d PASSED ===")
    mpi.finalize()
