# ======================================================================
# bench_two_fluid_langmuir_3d -- electron plasma oscillation frequency
# ======================================================================
#
# Canonical five-moment two-fluid test.  Seed a small x-velocity
# perturbation into the electron fluid at t=0.  In the linear
# regime the resulting plasma oscillation satisfies
#
#   omega_p^2 = n0 (q_e^2 / m_e + q_i^2 / m_i) / eps0
#
# and the (spatially-uniform) domain-mean electron x-momentum obeys
#
#   <rho_e u_e>(t) = A * m_e * n0 * cos(omega_p t)
#   <E_x>(t)       = -(A * m_e * n0 / q_e) * omega_p * sin(omega_p t)
#
# with phase locked to the simple harmonic oscillator.  After one
# full period T = 2 pi / omega_p the state returns to the IC -- any
# drift is scheme dissipation in the stiff source-term coupling.
#
# Pass criteria (P=2, periodic [0,1]^3-shaped slab, single-rank):
#   * |<rho_e u_e>(T) - A * m_e * n0| < 2%% of A * m_e * n0
#   * |<E_x>(T)|                       < 2%% of A * m_e * n0 * omega_p
#   * no NaN / Inf
#
# Measured values: rho_e u_e drift = 0.035%%, Ex / E_scale = 6e-5
# (Float32 roundoff over ~15k stages at NX=8).  The 2%% tolerance
# is ~57x looser, so passes have margin but a genuine regression
# (Lorentz coupling sign flip, wrong Ampere J source, mis-weighted
# mass in the reduced-mass frequency) would immediately fail.
# ======================================================================

from std.sys import has_accelerator
from std.gpu import global_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from std.math import sqrt, ceildiv, isnan, isinf

from src import mpi
from src.partition import build_partition
from src.reference import N_P, build_reference_operators
from src.mesh import Mesh
from src.boundary import BoundaryConditions
from src.halo_exchange import HaloExchange
from src.solver import Solver
from src.two_fluid import FiveMomentTwoFluid
from src.nvtx import NvtxContext


# NX=8 (same as P=3 sibling) -- the IC is spatially uniform (zero-
# mode k=0 plasma oscillation), so mesh resolution doesn't affect
# the analytic prediction.  This was historically NX=16, which was
# wasteful for a 0-mode test; halving NX gives a 4x speedup with
# no loss in test fidelity.
comptime NX = 8
comptime NY = 2
comptime NZ = 2
comptime LX = 1.0
comptime LY = Float64(2.0 / 8.0)
comptime LZ = Float64(2.0 / 8.0)

comptime GAMMA_E: Float32 = Float32(5.0 / 3.0)
comptime GAMMA_I: Float32 = Float32(5.0 / 3.0)
comptime Q_E: Float32 = -1.0
comptime M_E: Float32 = 1.0
comptime Q_I: Float32 = 1.0
comptime M_I: Float32 = 25.0
comptime EPS0: Float32 = 1.0
comptime C_LIGHT: Float32 = 10.0
comptime N0: Float32 = 1.0
comptime P_E0: Float32 = 0.01
comptime P_I0: Float32 = 0.01
comptime U_PERTURB: Float32 = 0.01
comptime C_H: Float32 = 12.0
comptime ALPHA_D: Float32 = 0.5
comptime MIN_DENSITY: Float32 = 1.0e-6
comptime MIN_PRESSURE: Float32 = 1.0e-8

comptime CFL = Float32(0.2)
comptime IC_BLOCK = 256

# One full period of the two-fluid Langmuir oscillation:
#   omega_p^2 = n0 * (q_e^2/m_e + q_i^2/m_i) / eps0 = 1 * (1 + 1/25) = 26/25
#   omega_p   = sqrt(26/25) ≈ 1.0198
#   T_period  = 2 pi / omega_p ≈ 6.1612
comptime T_FINAL: Float32 = Float32(6.1612348)

comptime DRIFT_TOL_REL: Float64 = 0.02


def langmuir_ic_kernel(
    q: UnsafePointer[Float32, MutAnyOrigin],
    owned_elem_ids: UnsafePointer[Int32, MutAnyOrigin],
    elem_node_xyz: UnsafePointer[Float32, MutAnyOrigin],
    num_owned: Int,
    rho_e0: Float32,
    rho_i0: Float32,
    u_pert: Float32,
    p_e0: Float32,
    p_i0: Float32,
    gamma_e: Float32,
    gamma_i: Float32,
):
    var idx = Int(global_idx.x)
    var total = num_owned * N_P
    if idx >= total:
        return
    var i = idx // N_P
    var nn = idx % N_P
    var e = Int(owned_elem_ids[i])
    var base = (e * N_P + nn) * 17

    q[base + 0] = rho_e0
    q[base + 1] = rho_e0 * u_pert
    q[base + 2] = Float32(0.0)
    q[base + 3] = Float32(0.0)
    q[base + 4] = (
        p_e0 / (gamma_e - Float32(1.0))
        + Float32(0.5) * rho_e0 * u_pert * u_pert
    )
    q[base + 5] = rho_i0
    q[base + 6] = Float32(0.0)
    q[base + 7] = Float32(0.0)
    q[base + 8] = Float32(0.0)
    q[base + 9] = p_i0 / (gamma_i - Float32(1.0))
    for k in range(10, 17):
        q[base + k] = Float32(0.0)


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    mpi.init()
    var size = mpi.world_size()
    if size > 1:
        mpi.finalize()
        print("bench_two_fluid_langmuir_3d: runs at np=1 only")
        return

    print("bench_two_fluid_langmuir_3d (plasma oscillation, one period)")

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

    solver.ctx.enqueue_function[langmuir_ic_kernel](
        solver.d_q.unsafe_ptr(),
        solver.mesh.d_owned_elem_ids.unsafe_ptr(),
        solver.mesh.local.d_elem_node_xyz.unsafe_ptr(),
        solver.num_owned_elements,
        M_E * N0,
        M_I * N0,
        U_PERTURB,
        P_E0,
        P_I0,
        GAMMA_E,
        GAMMA_I,
        grid_dim=ceildiv(solver.num_owned_elements * N_P, IC_BLOCK),
        block_dim=IC_BLOCK,
    )
    solver.ctx.synchronize()

    var omega_p = sqrt(N0 * (Q_E * Q_E / M_E + Q_I * Q_I / M_I) / EPS0)
    var h = Float32(LX) / Float32(NX)
    var wave = C_LIGHT if C_LIGHT > C_H else C_H
    var dt_est = CFL * h / (wave * Float32(5.0))
    var num_steps = Int(T_FINAL / dt_est) + 1
    var dt = T_FINAL / Float32(num_steps)
    print(
        "  omega_p =",
        omega_p,
        "  T_period =",
        T_FINAL,
        "  steps=",
        num_steps,
        "  dt=",
        dt,
    )

    for _ in range(num_steps):
        solver.step_ssprk3(dt, nvtx)
    solver.ctx.synchronize()

    # Domain-mean electron x-momentum and Ex.
    var total_dof = solver.num_owned_elements * N_P
    var buf = List[Float32]()
    for _ in range(total_dof):
        buf.append(Float32(0.0))

    solver.download_owned_component(1, buf, nvtx)  # rho_e u_e
    var sum_mom: Float64 = 0.0
    var finite = True
    for i in range(total_dof):
        var v = buf[i]
        if isnan(v) or isinf(v):
            finite = False
            break
        sum_mom += Float64(v)
    if not finite:
        raise Error("bench_two_fluid_langmuir_3d: non-finite rho_e u_e")
    var mean_mom = sum_mom / Float64(total_dof)

    solver.download_owned_component(10, buf, nvtx)  # Ex
    var sum_Ex: Float64 = 0.0
    for i in range(total_dof):
        var v = buf[i]
        if isnan(v) or isinf(v):
            finite = False
            break
        sum_Ex += Float64(v)
    if not finite:
        raise Error("bench_two_fluid_langmuir_3d: non-finite Ex")
    var mean_Ex = sum_Ex / Float64(total_dof)

    # Analytic after one period: rho_e u_e -> A * m_e * n0, Ex -> 0.
    var mom_ic = Float64(U_PERTURB * M_E * N0)
    var mom_err = (mean_mom - mom_ic) / mom_ic
    if mom_err < 0.0:
        mom_err = -mom_err
    var Ex_scale = Float64(U_PERTURB * M_E * N0) * Float64(omega_p)
    var Ex_rel = mean_Ex / Ex_scale
    if Ex_rel < 0.0:
        Ex_rel = -Ex_rel

    print(
        "  <rho_e u_e>(T) =",
        mean_mom,
        "  (analytic ~",
        mom_ic,
        ", rel err",
        mom_err,
        ")",
    )
    print(
        "  <E_x>(T)       =",
        mean_Ex,
        "  (analytic ~ 0, rel to A*m_e*n0*omega_p:",
        Ex_rel,
        ")",
    )

    if mom_err > DRIFT_TOL_REL:
        raise Error(
            String("bench_two_fluid_langmuir_3d FAILED: ")
            + "rho_e u_e rel drift "
            + String(mom_err)
            + " exceeds "
            + String(DRIFT_TOL_REL)
        )
    if Ex_rel > DRIFT_TOL_REL:
        raise Error(
            String("bench_two_fluid_langmuir_3d FAILED: ")
            + "Ex drift "
            + String(Ex_rel)
            + " exceeds "
            + String(DRIFT_TOL_REL)
        )

    print("=== bench_two_fluid_langmuir_3d PASSED ===")
    mpi.finalize()
