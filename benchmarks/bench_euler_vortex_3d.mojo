# ======================================================================
# bench_euler_vortex_3d -- 3D Euler isentropic vortex (one period)
# ======================================================================
#
# 3D extension of bench_euler_vortex_2d.  Same Shu-Erlebacher
# isentropic vortex IC extruded uniformly in z; the analytic
# solution is just the 2D vortex translated by (U0, V0)*T after one
# period.  Tests the rotational dynamics of the full 3D Euler
# pipeline -- different from bench_euler_smooth_wave_3d (linear
# entropy advection under uniform velocity) since the vortex IC
# has spatially-varying density, pressure, and all three velocity
# components.  All flux terms (rho*u^2 + p, rho*u*v, etc.) are
# active.
#
# IC (3D vortex with z-uniform extrusion):
#   T(x,y) = T_inf - factor * exp(1 - r^2)
#   u(x,y) = U0 - (BETA/2pi) * (y - CY0) * exp((1-r^2)/2)
#   v(x,y) = V0 + (BETA/2pi) * (x - CX0) * exp((1-r^2)/2)
#   w      = 0
#   rho    = T^(1/(gamma-1)),  p = rho * T
#   E      = p/(gamma-1) + 0.5*rho*(u^2 + v^2)
# After T = LX / U0 = LY / V0 the analytic solution returns to the
# IC (or shifted by one period under periodic BCs).  Residual L2 is
# pure scheme dissipation in the rotational frame.
#
# Pass criteria (P=2, N=32x32x4, periodic, T=10, HLLEC):
#   * rel L2(state) < 10 %% (Rusanov dissipation floor for an
#     isentropic vortex at this resolution; 2D bench measures ~6.8 %%
#     and 3D should be similar with z-uniform IC since Fz = 0)
#   * no NaN / Inf
# ======================================================================

from std.sys import has_accelerator
from std.gpu import global_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from std.math import sqrt, ceildiv, exp, pi, isnan, isinf

from src import mpi
from src.partition import build_partition
from src.reference import N_P, build_reference_operators
from src.mesh import Mesh
from src.boundary import BoundaryConditions
from src.halo_exchange import HaloExchange
from src.solver import Solver
from src.euler import Euler, FLUX_HLLEC
from src.nvtx import NvtxContext


comptime NX = 32
comptime NY = 32
comptime NZ = 4
comptime LX = 10.0
comptime LY = 10.0
comptime LZ = Float64(NZ) / Float64(NX) * LX

comptime GAMMA: Float32 = 1.4
comptime T_INF: Float32 = 1.0
comptime U0: Float32 = 1.0
comptime V0: Float32 = 1.0
comptime BETA: Float32 = 5.0
comptime CX0: Float32 = 5.0
comptime CY0: Float32 = 5.0
comptime T_FINAL: Float32 = 10.0
comptime CFL = Float32(0.15)
comptime IC_BLOCK = 256
comptime PI_F: Float32 = 3.14159265358979323846
comptime TWO_PI_F: Float32 = 2.0 * PI_F
comptime MIN_DENSITY = Float32(1.0e-6)
comptime MIN_PRESSURE = Float32(1.0e-6)

# Empirical: 6.8 %% in 2D at NX=32; 3D with z-uniform IC and HLLEC
# should sit in the same ballpark.  10 %% gate catches catastrophic
# regressions while accommodating the Rusanov-class dissipation
# floor on this rotational-flow problem.
comptime L2_MAX_REL: Float64 = 0.10


def vortex_ic_kernel(
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
    var px = elem_node_xyz[(e * N_P + nn) * 3 + 0]
    var py = elem_node_xyz[(e * N_P + nn) * 3 + 1]

    # Periodic delta to vortex center.
    var dx = px - CX0
    if dx > Float32(LX * 0.5):
        dx -= Float32(LX)
    if dx < -Float32(LX * 0.5):
        dx += Float32(LX)
    var dy = py - CY0
    if dy > Float32(LY * 0.5):
        dy -= Float32(LY)
    if dy < -Float32(LY * 0.5):
        dy += Float32(LY)

    var r2 = dx * dx + dy * dy
    var factor = (
        (GAMMA - Float32(1.0))
        * BETA
        * BETA
        / (Float32(8.0) * GAMMA * TWO_PI_F * TWO_PI_F)
    )
    var T = T_INF - factor * exp(Float32(1.0) - r2)
    var e_half = exp(Float32(0.5) * (Float32(1.0) - r2))
    var u = U0 - (BETA / TWO_PI_F) * dy * e_half
    var v = V0 + (BETA / TWO_PI_F) * dx * e_half
    var w = Float32(0.0)
    var rho = T ** (Float32(1.0) / (GAMMA - Float32(1.0)))
    var p = rho * T
    var E = p / (GAMMA - Float32(1.0)) + Float32(0.5) * rho * (
        u * u + v * v + w * w
    )

    var base = (e * N_P + nn) * 5
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
        print("bench_euler_vortex_3d: runs at np=1 only")
        return

    print("bench_euler_vortex_3d (3D Shu-Erlebacher isentropic vortex)")
    print(
        "  P= 2   mesh=",
        NX,
        "x",
        NY,
        "x",
        NZ,
        "   T=",
        T_FINAL,
        " (one period)",
    )

    var rank = mpi.world_rank()
    var nvtx = NvtxContext()
    var refs = build_reference_operators(nvtx)
    var ctx = DeviceContext()

    var bcs = BoundaryConditions.periodic()
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
    var physics = Euler(
        GAMMA,
        MIN_DENSITY,
        MIN_PRESSURE,
        FLUX_HLLEC,
        False,
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

    solver.ctx.enqueue_function[vortex_ic_kernel, vortex_ic_kernel](
        solver.d_q.unsafe_ptr(),
        solver.mesh.d_owned_elem_ids.unsafe_ptr(),
        solver.mesh.local.d_elem_node_xyz.unsafe_ptr(),
        solver.num_owned_elements,
        grid_dim=ceildiv(solver.num_owned_elements * N_P, IC_BLOCK),
        block_dim=IC_BLOCK,
    )
    solver.ctx.synchronize()

    var n_owned_dof = solver.num_owned_elements * N_P * Euler.NUM_COMPONENTS
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
    var dt_est = CFL * h / (wave_max * Float32(5.0))
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
            raise Error(
                "bench_euler_vortex_3d: non-finite output at " + String(k)
            )
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
            "bench_euler_vortex_3d FAILED: rel L2 "
            + String(rel)
            + " > "
            + String(L2_MAX_REL)
        )

    print("=== bench_euler_vortex_3d PASSED ===")
    mpi.finalize()
