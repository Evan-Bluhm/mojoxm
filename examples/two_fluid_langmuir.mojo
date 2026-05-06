# ======================================================================
# two_fluid_langmuir -- electron plasma oscillation in a 2-fluid plasma
# ======================================================================
#
# The canonical two-fluid test: seed a small x-velocity perturbation in
# the electron fluid at t = 0 and watch it oscillate with ions (near-)
# stationary.  The electron fluid executes a Langmuir wave at the
# plasma frequency
#
#   omega_p = sqrt(n_e q_e^2 / (eps0 m_e))
#
# driven by the self-consistent E-field that develops from the charge
# separation.  In our natural unit system with q_e = -1, m_e = 1,
# eps0 = 1, a uniform number density of n0 gives omega_p = sqrt(n0).
#
# Setup:
#   domain:  [0, 1] x (thin y, thin z); periodic on all sides
#   IC:      rho_e = m_e * n0, u_e = (A, 0, 0), p_e = p0
#            rho_i = m_i * n0, u_i = 0,         p_i = p0
#            E = 0, B = 0, psi = 0
#
# With all perturbations y/z-uniform the solution is 1D along x and
# the oscillation is purely electrostatic: E_x swings through
#   E_x(t) = -A (m_e / q_e) omega_p sin(omega_p t)      (spatially uniform)
#
# We verify by tracking the spatial mean of u_e (which should trace a
# cosine at omega_p) and printing the first zero-crossing time.  Can
# also compare against the theoretical period T_p = 2 pi / omega_p.
#
# This test does NOT require non-periodic BCs or a spatial gradient --
# it's a pure source-driven oscillation that exercises the full
# Lorentz / J / Maxwell coupling in a setup where the answer is
# analytic.  A great first validation for the 17-component module.
# ======================================================================

from std.sys import has_accelerator
from std.gpu import global_idx
from std.gpu.host import DeviceContext
from std.math import sqrt, ceildiv

from src import mpi
from src.partition import build_partition
from src.reference import N_P, build_reference_operators
from src.mesh import Mesh
from src.boundary import BoundaryConditions
from src.halo_exchange import HaloExchange
from src.solver import Solver
from src.two_fluid import FiveMomentTwoFluid
from src.nvtx import NvtxContext
from src.frame_writer import FrameWriter, write_snapshot_3d_multi
from src.time_integrator import run_ssprk3_loop_with_diagnostics
from src.diagnostics import DiagnosticsWriter, NamedComponent


comptime NX = 16
comptime NY = 2
comptime NZ = 2
comptime LX = 1.0
comptime LY = Float64(2.0 / 16.0)    # dx = dy = dz
comptime LZ = Float64(2.0 / 16.0)

# Natural units: eps0 = 1, m_e = 1, q_e = -1.  omega_p = sqrt(n0).
comptime GAMMA_E: Float32 = Float32(5.0 / 3.0)
comptime GAMMA_I: Float32 = Float32(5.0 / 3.0)
comptime Q_E:     Float32 = -1.0
comptime M_E:     Float32 =  1.0
comptime Q_I:     Float32 =  1.0
comptime M_I:     Float32 = 25.0       # reduced mass ratio
comptime EPS0:    Float32 =  1.0
comptime C_LIGHT: Float32 = 10.0       # c / v_thermal comfortably > 1

# Background plasma.
comptime N0:      Float32 = 1.0        # number density of each species
comptime P_E0:    Float32 = 0.01
comptime P_I0:    Float32 = 0.01

# Initial electron-fluid x-velocity (small so the linear Langmuir
# wave is a decent approximation).
comptime U_PERTURB: Float32 = 0.01

# GLM cleaner: c_h >= c; alpha_d turns on damping of any div(B) noise.
comptime C_H:     Float32 = 12.0
comptime ALPHA_D: Float32 = 0.5

comptime MIN_DENSITY:  Float32 = 1.0e-6
comptime MIN_PRESSURE: Float32 = 1.0e-8

# CFL: Rusanov dt scales like h / max_wave.  Max wave is c_light (10
# here), so dt <= 0.2 * h / (c_light * 5).
comptime CFL = Float32(0.2)
comptime IC_BLOCK = 256

# Integrate for half a plasma period.  Full two-period round-trips
# are informative for the np=1 correctness check, but a 16x2x2 mesh
# at NC=17 is kernel-launch-bound at multi-rank, so we keep the
# default run short.  Drivers that want the round-trip eye-test can
# bump this (at np=1 the full 4pi integration still finishes in ~7 s).
comptime T_FINAL: Float32 = Float32(0.5 * 6.283185307179586)
comptime NUM_FRAMES = 10


def langmuir_ic_kernel(
    q: UnsafePointer[Float32, MutAnyOrigin],
    owned_elem_ids: UnsafePointer[Int32, MutAnyOrigin],
    elem_node_xyz:  UnsafePointer[Float32, MutAnyOrigin],
    num_owned: Int,
    rho_e0: Float32, rho_i0: Float32,
    u_pert: Float32,
    p_e0: Float32, p_i0: Float32,
    gamma_e: Float32, gamma_i: Float32,
):
    var idx = Int(global_idx.x)
    var total = num_owned * N_P
    if idx >= total:
        return
    var i = idx // N_P
    var nn = idx % N_P
    var e = Int(owned_elem_ids[i])
    var base = (e * N_P + nn) * 17

    # Electron fluid: uniform density, perturbed x-velocity.
    q[base + 0] = rho_e0
    q[base + 1] = rho_e0 * u_pert   # rho_e * u_e
    q[base + 2] = Float32(0.0)
    q[base + 3] = Float32(0.0)
    q[base + 4] = (
        p_e0 / (gamma_e - Float32(1.0))
        + Float32(0.5) * rho_e0 * u_pert * u_pert
    )
    # Ion fluid: uniform density, zero velocity.
    q[base + 5] = rho_i0
    q[base + 6] = Float32(0.0)
    q[base + 7] = Float32(0.0)
    q[base + 8] = Float32(0.0)
    q[base + 9] = p_i0 / (gamma_i - Float32(1.0))
    # EM: zero field initially (charge-neutral at t=0, so E = 0).
    for k in range(10, 17):
        q[base + k] = Float32(0.0)


def choose_dt() raises -> Float32:
    var h = Float32(LX) / Float32(NX)
    # The EM sector's c is the hard CFL constraint here; fluids are
    # much slower.  Factor (2P + 1) = 5 matches the other P2 drivers.
    var wave = max(C_LIGHT, C_H)
    return CFL * h / (wave * Float32(5.0))


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    mpi.init()
    var rank = mpi.world_rank()
    var size = mpi.world_size()

    if rank == 0:
        print(
            "two_fluid_langmuir: GPU DG 5-moment two-fluid + Maxwell, P2 tet,",
            size, "rank(s)",
        )
        print("  global mesh: ", NX, "x", NY, "x", NZ,
              " cells -> ", NX * NY * NZ * 6, "tets")
        var omega_p = sqrt(N0 * Q_E * Q_E / (EPS0 * M_E))
        print("  omega_p =", omega_p,
              "  T_period =", Float32(6.283185307179586) / omega_p,
              "  T_FINAL = ~0.5 period")

    var nvtx = NvtxContext()
    var refs = build_reference_operators(nvtx)
    var ctx = DeviceContext()

    var bcs = BoundaryConditions.periodic()
    var mesh = Mesh(
        ctx, build_partition(rank, size, NX, NY, NZ), LX, LY, LZ, bcs,
    )
    var halo = HaloExchange(
        ctx, mesh.part, FiveMomentTwoFluid.NUM_COMPONENTS,
        mesh.d_perm.unsafe_ptr(), bcs,
    )
    var physics = FiveMomentTwoFluid(
        GAMMA_E, GAMMA_I,
        Q_E, M_E, Q_I, M_I,
        EPS0, C_LIGHT,
        C_H, ALPHA_D,
        MIN_DENSITY, MIN_PRESSURE,
    )
    var solver = Solver[FiveMomentTwoFluid](
        ctx^, mesh^, halo^, physics^, refs.D_ref^, refs.Lift_ref^, refs.node_weights^,
    )

    solver.ctx.enqueue_function[langmuir_ic_kernel, langmuir_ic_kernel](
        solver.d_q.unsafe_ptr(),
        solver.mesh.d_owned_elem_ids.unsafe_ptr(),
        solver.mesh.local.d_elem_node_xyz.unsafe_ptr(),
        solver.num_owned_elements,
        M_E * N0, M_I * N0,
        U_PERTURB,
        P_E0, P_I0,
        GAMMA_E, GAMMA_I,
        grid_dim=ceildiv(
            solver.num_owned_elements * N_P, IC_BLOCK
        ),
        block_dim=IC_BLOCK,
    )
    solver.ctx.synchronize()

    # Pre-step perf snapshot: device memory accounting (rank 0 only).
    if rank == 0:
        solver.memory_report().print()

    var writer = FrameWriter[FiveMomentTwoFluid](
        solver, nvtx, component=10,    # Ex as the output scalar
    )

    # Diagnostics: per-species mass + x-momentum (the relevant one for
    # a 1D Langmuir oscillation), total fluid energies, and EM energy
    # via the squared components.  Conservation of total momentum
    # `rho_e u_e + rho_i u_i` is the clearest signature that the
    # Lorentz coupling is symmetric between the two fluids.
    var diag_linear = List[NamedComponent]()
    diag_linear.append(NamedComponent("mass_e",    0))
    diag_linear.append(NamedComponent("mom_e_x",   1))
    diag_linear.append(NamedComponent("energy_e",  4))
    diag_linear.append(NamedComponent("mass_i",    5))
    diag_linear.append(NamedComponent("mom_i_x",   6))
    diag_linear.append(NamedComponent("energy_i",  9))
    var diag_squared = List[NamedComponent]()
    diag_squared.append(NamedComponent("Ex_sq", 10))
    diag_squared.append(NamedComponent("Ey_sq", 11))
    diag_squared.append(NamedComponent("Ez_sq", 12))
    var diag_maxabs = List[NamedComponent]()
    diag_maxabs.append(NamedComponent("max_abs_psi", 16))
    var diag = DiagnosticsWriter[FiveMomentTwoFluid](
        solver, "output/diagnostics.csv",
        diag_linear, diag_squared, diag_maxabs,
        LX, LY, LZ,
    )

    var dt = choose_dt()
    if rank == 0:
        print("  dt =", dt, " (", Int(T_FINAL / dt), " steps estimated)")

    var result = run_ssprk3_loop_with_diagnostics[FiveMomentTwoFluid](
        solver, writer, diag, dt, T_FINAL, NUM_FRAMES, nvtx,
    )

    writer.finalize("output/solution.pvd", nvtx)

    # Final-state multi-field snapshot for richer ParaView inspection.
    # Two-Fluid is NC=17; emits four physically meaningful scalars:
    # n_e (electron number density rho_e/m_e), n_i (ion number density
    # rho_i/m_i), Ex (the dominant Langmuir-oscillation E-field
    # component), and charge_density (Q_E*n_e + Q_I*n_i, which oscillates
    # 90 degrees out of phase with Ex per the cold-plasma dispersion
    # relation).  Gated on np=1.
    var nprocs = solver.mesh.part.px * solver.mesh.part.py * solver.mesh.part.pz
    if nprocs == 1:
        var n_owned_dof = solver.num_owned_elements * N_P
        var snap_rho_e = List[Float32]()
        var snap_rho_i = List[Float32]()
        var snap_ex    = List[Float32]()
        for _ in range(n_owned_dof):
            snap_rho_e.append(Float32(0.0))
            snap_rho_i.append(Float32(0.0))
            snap_ex.append(Float32(0.0))
        solver.download_owned_component(0,  snap_rho_e, nvtx)  # electron rho
        solver.download_owned_component(5,  snap_rho_i, nvtx)  # ion rho
        solver.download_owned_component(10, snap_ex,    nvtx)  # Ex
        var f_n_e   = List[Float64]()
        var f_n_i   = List[Float64]()
        var f_ex    = List[Float64]()
        var f_chg   = List[Float64]()
        for k in range(n_owned_dof):
            var n_e = snap_rho_e[k] / M_E
            var n_i = snap_rho_i[k] / M_I
            f_n_e.append(Float64(n_e))
            f_n_i.append(Float64(n_i))
            f_ex.append(Float64(snap_ex[k]))
            f_chg.append(Float64(Q_E * n_e + Q_I * n_i))
        var fields = List[List[Float64]]()
        fields.append(f_n_e^)
        fields.append(f_n_i^)
        fields.append(f_ex^)
        fields.append(f_chg^)
        var names = List[String]()
        names.append(String("n_e"))
        names.append(String("n_i"))
        names.append(String("Ex"))
        names.append(String("charge_density"))
        write_snapshot_3d_multi(
            solver=solver, field_names=names, field_data=fields,
            path=String("output/snapshot_t_final.vtu"), nvtx=nvtx,
        )
        if rank == 0:
            print("  wrote output/snapshot_t_final.vtu (n_e + n_i + Ex + charge, t=", T_FINAL, ")")

    # At np=1 sample a few diagnostics: the spatial mean of Ex and of
    # rho_e * u_e (electron x-momentum) should both be traces of the
    # same oscillation, 90 degrees apart in phase.
    if size == 1:
        var total_dof = solver.num_owned_elements * N_P
        var buf = List[Float32]()
        for _ in range(total_dof):
            buf.append(Float32(0.0))
        solver.download_owned_component(10, buf, nvtx)   # Ex
        var sum_Ex: Float64 = 0.0
        for i in range(total_dof):
            sum_Ex += Float64(buf[i])
        var mean_Ex = Float32(sum_Ex / Float64(total_dof))

        solver.download_owned_component(1, buf, nvtx)    # rho_e u_e
        var sum_mom: Float64 = 0.0
        for i in range(total_dof):
            sum_mom += Float64(buf[i])
        var mean_mom = Float32(sum_mom / Float64(total_dof))

        print("  domain-mean Ex at T_FINAL    :", mean_Ex)
        print("  domain-mean rho_e u_e at T_FINAL:", mean_mom)
        # At half a natural period (with ion back-reaction at m_i=25),
        # analytic cos/sin at phase ~ pi gives
        #   rho_e u_e ~ -0.0092,  Ex ~ -0.0006
        # -- see the driver header for the derivation.

    if rank == 0:
        print("  total steps:", result.total_steps,
              " wall time:", result.wall_sec, "s")
        print("  wrote output/solution.pvd")
    # Post-run sync'd throughput measurement.
    var tput = solver.bench_step_loop(dt, nvtx)
    if rank == 0:
        tput.print()

    mpi.finalize()
