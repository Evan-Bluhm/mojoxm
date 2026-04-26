# ======================================================================
# bench_euler_vortex_3d_p3 -- 3D vortex (one period) at P=3 (NP=20)
# ======================================================================
#
# P=3 (NP=20) HLLEC counterpart of bench_euler_vortex_3d.  Same
# z-uniform Shu-Erlebacher isentropic vortex IC, same one-period
# horizon (T = LX / U0 = 10), but routed through Mesh[3] /
# Solver[Euler, 3] / rk_stage_kernel[3] so the 3D HLLEC kernel is
# exercised at NP=20 on long-time rotational flow.  Mirrors the
# 2D _p3 vortex bench landed in commit d4e2dda.
#
# NX/NY=24 (vs 32 at P=2) keeps wall-clock comparable: NP=20 vs
# NP=10 is ~4x volume work per element, dt is tighter by 7/5 = 1.4x,
# so the smaller mesh balances the heavier per-step cost.
#
# Pass criteria (P=3, N=24x24x4, periodic, T=10, HLLEC):
#   * rel L2(state) < 10 %% (Rusanov dissipation floor on this setup
#     is roughly P- and flux-type-independent at this long horizon
#     -- the 2D _p3 bench measured 6.81 %% at HLLC NP=10, almost
#     identical to P=2 / Rusanov / NP=6's 6.8 %%; 3D with z-uniform
#     IC should sit in the same ballpark since Fz = 0 on the IC)
#   * no NaN / Inf
# ======================================================================

from std.sys import has_accelerator
from std.gpu import global_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from std.math import sqrt, ceildiv, exp, pi, isnan, isinf

from src import mpi
from src.partition import build_partition
from src.reference import ReferenceElement, to_float32, num_tet_nodes
from src.mesh import Mesh
from src.boundary import BoundaryConditions
from src.halo_exchange import HaloExchange
from src.solver import Solver
from src.euler import Euler, FLUX_HLLEC
from src.nvtx import NvtxContext


comptime P = 3
comptime NP = num_tet_nodes(P)   # 20 at P=3
comptime NX = 24
comptime NY = 24
comptime NZ = 4
comptime LX = 10.0
comptime LY = 10.0
comptime LZ = Float64(NZ) / Float64(NX) * LX

comptime GAMMA: Float32 = 1.4
comptime T_INF: Float32 = 1.0
comptime U0:    Float32 = 1.0
comptime V0:    Float32 = 1.0
comptime BETA:  Float32 = 5.0
comptime CX0:   Float32 = 5.0
comptime CY0:   Float32 = 5.0
comptime T_FINAL: Float32 = 10.0
comptime CFL = Float32(0.15)
comptime IC_BLOCK = 256
comptime PI_F: Float32 = 3.14159265358979323846
comptime TWO_PI_F: Float32 = 2.0 * PI_F
comptime MIN_DENSITY  = Float32(1.0e-6)
comptime MIN_PRESSURE = Float32(1.0e-6)

# Empirical: 6.8 %% in 2D at NX=32; 3D with z-uniform IC and HLLEC
# should sit in the same ballpark.  10 %% gate catches catastrophic
# regressions while accommodating the Rusanov-class dissipation
# floor on this rotational-flow problem.
comptime L2_MAX_REL: Float64 = 0.10


def vortex_ic_kernel(
    q: UnsafePointer[Float32, MutAnyOrigin],
    owned_elem_ids: UnsafePointer[Int32, MutAnyOrigin],
    elem_node_xyz:  UnsafePointer[Float32, MutAnyOrigin],
    num_owned: Int,
):
    var idx = Int(global_idx.x)
    var total = num_owned * NP
    if idx >= total:
        return
    var i = idx // NP
    var nn = idx % NP
    var e = Int(owned_elem_ids[i])
    var px = elem_node_xyz[(e * NP + nn) * 3 + 0]
    var py = elem_node_xyz[(e * NP + nn) * 3 + 1]

    # Periodic delta to vortex center.
    var dx = px - CX0
    if dx >  Float32(LX * 0.5): dx -= Float32(LX)
    if dx < -Float32(LX * 0.5): dx += Float32(LX)
    var dy = py - CY0
    if dy >  Float32(LY * 0.5): dy -= Float32(LY)
    if dy < -Float32(LY * 0.5): dy += Float32(LY)

    var r2 = dx * dx + dy * dy
    var factor = (GAMMA - Float32(1.0)) * BETA * BETA / (
        Float32(8.0) * GAMMA * TWO_PI_F * TWO_PI_F
    )
    var T = T_INF - factor * exp(Float32(1.0) - r2)
    var e_half = exp(Float32(0.5) * (Float32(1.0) - r2))
    var u = U0 - (BETA / TWO_PI_F) * dy * e_half
    var v = V0 + (BETA / TWO_PI_F) * dx * e_half
    var w = Float32(0.0)
    var rho = T ** (Float32(1.0) / (GAMMA - Float32(1.0)))
    var p = rho * T
    var E = (
        p / (GAMMA - Float32(1.0))
        + Float32(0.5) * rho * (u * u + v * v + w * w)
    )

    var base = (e * NP + nn) * 5
    q[base + 0] = rho
    q[base + 1] = rho * u
    q[base + 2] = rho * v
    q[base + 3] = rho * w
    q[base + 4] = E


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    mpi.init()
    var size = mpi.world_size()
    if size > 1:
        mpi.finalize()
        print("bench_euler_vortex_3d_p3: runs at np=1 only")
        return

    print("bench_euler_vortex_3d_p3 (3D Shu-Erlebacher isentropic vortex at P=3)")
    print("  P=", P, "  NP=", NP, "  mesh=", NX, "x", NY, "x", NZ,
          "   T=", T_FINAL, " (one period)")

    var rank = mpi.world_rank()
    var nvtx = NvtxContext()
    var ctx = DeviceContext()

    # Build P=3 reference operators directly: build_reference_operators()
    # in src.reference defaults to P=2 / NP=10, wrong for Solver[Euler, 3].
    var re = ReferenceElement[P]()
    var D_ref = to_float32(re.D_ref)
    var Lift_ref = to_float32(re.Lift_ref)
    var node_weights = to_float32(re.node_weights)

    var bcs = BoundaryConditions.periodic()
    var mesh = Mesh[P](
        ctx, build_partition(rank, size, NX, NY, NZ), LX, LY, LZ, bcs,
    )
    var halo = HaloExchange(
        ctx, mesh.part, Euler.NUM_COMPONENTS,
        mesh.d_perm.unsafe_ptr(), bcs,
    )
    var physics = Euler(
        GAMMA, MIN_DENSITY, MIN_PRESSURE, FLUX_HLLEC, False,
    )
    var solver = Solver[Euler, P](
        ctx^, mesh^, halo^, physics^, D_ref^, Lift_ref^, node_weights^,
    )

    solver.ctx.enqueue_function[vortex_ic_kernel, vortex_ic_kernel](
        solver.d_q.unsafe_ptr(),
        solver.mesh.d_owned_elem_ids.unsafe_ptr(),
        solver.mesh.local.d_elem_node_xyz.unsafe_ptr(),
        solver.num_owned_elements,
        grid_dim=ceildiv(solver.num_owned_elements * NP, IC_BLOCK),
        block_dim=IC_BLOCK,
    )
    solver.ctx.synchronize()

    var n_owned_dof = solver.num_owned_elements * NP * Euler.NUM_COMPONENTS
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

    var c_inf = sqrt(GAMMA * T_INF)
    var wave_max = sqrt(U0 * U0 + V0 * V0) + c_inf + BETA / TWO_PI_F
    var h = Float32(LX) / Float32(NX)
    var dt_est = CFL * h / (wave_max * Float32(2 * P + 1))
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
            raise Error("bench_euler_vortex_3d_p3: non-finite output at "
                        + String(k))
        var err = Float64(v_now - host_ic[k])
        sum_sq += err * err
        var ic = Float64(host_ic[k])
        sum_ic += ic * ic
    var l2 = sqrt(sum_sq / Float64(n_owned_dof))
    var l2_ic = sqrt(sum_ic / Float64(n_owned_dof))
    var rel = l2 / l2_ic
    print("  rel L2(state) =", rel, "  (threshold", L2_MAX_REL, ")")

    if rel > L2_MAX_REL:
        raise Error(
            "bench_euler_vortex_3d_p3 FAILED: rel L2 "
            + String(rel) + " > " + String(L2_MAX_REL)
        )

    print("=== bench_euler_vortex_3d_p3 PASSED ===")
    mpi.finalize()
