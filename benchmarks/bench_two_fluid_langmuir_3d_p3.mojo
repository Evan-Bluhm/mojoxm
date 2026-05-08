# ======================================================================
# bench_two_fluid_langmuir_3d_p3 -- plasma oscillation at P=3 (NP=20)
# ======================================================================
#
# P=3 counterpart of bench_two_fluid_langmuir_3d.  Same canonical
# five-moment plasma oscillation (small electron x-velocity
# perturbation -> rho_e u_e and Ex oscillate at omega_p with phase
# locked to the simple harmonic oscillator), but routed through
# Mesh[3] / Solver[FiveMomentTwoFluid, 3] / NP=20 nodes per tet.
#
# Why this gate exists:
# bench_two_fluid_walls_3d_p3 / _p4 / _p5 already exercise the
# 17-component Two-Fluid pipeline at higher NP, but with charge-
# balanced rest IC -- so the Lorentz force, Ampere current source,
# and source_term hook all evaluate to zero at every node and never
# get tested under non-trivial state evolution.  This bench drives
# those source-term paths with a real plasma-frequency oscillation
# at NP=20, catching regressions that the rest-state walls bench
# can't see.
#
# Pass criteria (P=3, periodic [0,1] x thin slab, single-rank):
#   * |<rho_e u_e>(T) - A * m_e * n0| < 2%% of A * m_e * n0
#   * |<E_x>(T)|                       < 2%% of A * m_e * n0 * omega_p
#   * no NaN / Inf
#
# Same 2%% tolerance as the P=2 bench; the higher-order DG scheme
# should be at least as accurate as P=2 on this smooth test, so a
# tighter threshold here would be defensible but conservative
# matches keep the gate's comparison-against-walls-3d-p3-pass
# semantics simple.
# ======================================================================

from std.sys import has_accelerator
from std.gpu import global_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from std.math import sqrt, ceildiv, isnan, isinf

from src import mpi
from src.partition import build_partition
from src.reference import ReferenceElement, to_float32, num_tet_nodes
from src.mesh import Mesh
from src.boundary import BoundaryConditions
from src.halo_exchange import HaloExchange
from src.solver import Solver
from src.two_fluid import FiveMomentTwoFluid
from src.nvtx import NvtxContext


comptime P = 3
comptime NP = num_tet_nodes(P)  # 20 at P=3
comptime NC = 17

# NX=8 (vs P=2's NX=16) is half-resolution but the IC is spatially
# uniform (zero-mode k=0 plasma oscillation), so mesh resolution
# doesn't affect the analytic prediction.  Halving NX ~= 4x speedup
# (steps halve via 2x dt, per-step compute halves via half the
# elements) -- keeps the P=3 bench's contribution to bench-two-fluid
# in the same range as the other Two-Fluid gates rather than
# dominating it 5:1.
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

# omega_p^2 = n0 * (q_e^2/m_e + q_i^2/m_i) / eps0 = 1 * (1 + 1/25) = 26/25
# omega_p   = sqrt(26/25) ~= 1.0198
# T_period  = 2 pi / omega_p ~= 6.1612
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
    var total = num_owned * NP
    if idx >= total:
        return
    var i = idx // NP
    var nn = idx % NP
    var e = Int(owned_elem_ids[i])
    var base = (e * NP + nn) * NC

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
    for k in range(10, NC):
        q[base + k] = Float32(0.0)


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    mpi.init()
    var size = mpi.world_size()
    if size > 1:
        mpi.finalize()
        print("bench_two_fluid_langmuir_3d_p3: runs at np=1 only")
        return

    print(
        "bench_two_fluid_langmuir_3d_p3 (plasma oscillation at P=3, one period)"
    )

    var rank = mpi.world_rank()
    var nvtx = NvtxContext()
    var ctx = DeviceContext()

    # P=3 reference operators (build_reference_operators() defaults
    # to P=2; for P=3 we instantiate ReferenceElement[3] directly).
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
        nc=FiveMomentTwoFluid.NUM_COMPONENTS,
        d_perm=mesh.d_perm.unsafe_ptr(),
        bcs=bcs,
    )
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
    )
    var solver = Solver[FiveMomentTwoFluid, P](
        ctx=ctx^,
        mesh=mesh^,
        halo=halo^,
        physics=physics^,
        D_ref=D_ref^,
        Lift_ref=Lift_ref^,
        node_weights=node_weights^,
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
        grid_dim=ceildiv(solver.num_owned_elements * NP, IC_BLOCK),
        block_dim=IC_BLOCK,
    )
    solver.ctx.synchronize()

    var omega_p = sqrt(N0 * (Q_E * Q_E / M_E + Q_I * Q_I / M_I) / EPS0)
    # CFL on the speed of light (the fastest wave at this c_h <= c).
    # 1/(2P+1) tightens dt for higher-order quadrature.
    var h = Float32(LX) / Float32(NX)
    var wave = C_LIGHT if C_LIGHT > C_H else C_H
    var dt_est = CFL * h / (wave * Float32(2 * P + 1))
    var num_steps = Int(T_FINAL / dt_est) + 1
    var dt = T_FINAL / Float32(num_steps)
    print(
        "  P=",
        P,
        "  NP=",
        NP,
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
    var total_dof = solver.num_owned_elements * NP
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
        raise Error("bench_two_fluid_langmuir_3d_p3: non-finite rho_e u_e")
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
        raise Error("bench_two_fluid_langmuir_3d_p3: non-finite Ex")
    var mean_Ex = sum_Ex / Float64(total_dof)

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
            String("bench_two_fluid_langmuir_3d_p3 FAILED: ")
            + "rho_e u_e rel drift "
            + String(mom_err)
            + " exceeds "
            + String(DRIFT_TOL_REL)
        )
    if Ex_rel > DRIFT_TOL_REL:
        raise Error(
            String("bench_two_fluid_langmuir_3d_p3 FAILED: ")
            + "Ex drift "
            + String(Ex_rel)
            + " exceeds "
            + String(DRIFT_TOL_REL)
        )

    print("=== bench_two_fluid_langmuir_3d_p3 PASSED ===")
    mpi.finalize()
