# ======================================================================
# bench_euler_flux_coverage_3d -- gates Rusanov, Roe, HLLE flux paths
# ======================================================================
#
# Closes a real coverage gap: every Euler benchmark outside this
# file uses `FLUX_HLLEC`, so `FLUX_RUSANOV`, `FLUX_ROE`, and
# `FLUX_HLLE` paths in `src/euler.mojo` -- plus the Harten-Hyman
# entropy-fix branch in the wave-based (Roe / HLLE) solvers -- have
# no other gate.  A regression in any of those three solvers (or in
# `euler_roe_solver` / `euler_hlle_solver` / Roe averages / the
# entropy-fix wavespeed flooring) would never trip the existing
# harness.
#
# This bench runs the same smooth entropy-wave IC as bench_euler_
# smooth_wave_3d under each of the three currently-untested flux
# types and gates each independently.  Roe and HLLE additionally
# get a second pass with `entropy_fix=True` so the Harten-Hyman
# branch in `src/euler.mojo` (lines ~398, ~463) sees test traffic.
# Smooth IC + uniform background means no sonic transitions, so
# entropy_fix on/off should agree to Float32 noise; the gate's job
# is just to catch a finite-output / type-error regression in the
# fix branch, not to test its correctness on near-sonic rarefactions.
# It does NOT replace the tighter HLLEC convergence-rate gate; it
# just gates that the four other flux dispatch arms produce a
# finite, sensible answer.
#
# IC: rho = rho0 + A * sin(2 pi x) * sin(2 pi y) * sin(2 pi z),
#     u = v = w = 1, p = const -- the entropy-wave that all four
#     fluxes should advect almost dissipation-free under uniform flow.
#
# Pass criteria (P=2, N=8, single-rank, periodic):
#   * For each of {Rusanov, Roe, HLLE}:
#       - rel L2(state) < 5e-3 after one period (T = 1)
#       - no NaN / Inf
#
# 5e-3 leaves room for the more dissipative fluxes (Rusanov is
# expected to be looser than HLLEC); the existing HLLEC bench still
# gates at the tighter ~1e-3 level.  This is a regression-detector,
# not an accuracy gate.
# ======================================================================

from std.sys import has_accelerator
from std.gpu import global_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from std.math import sqrt, ceildiv, sin, isnan, isinf

from src import mpi
from src.partition import build_partition
from src.reference import N_P, build_reference_operators
from src.mesh import Mesh
from src.boundary import BoundaryConditions
from src.halo_exchange import HaloExchange
from src.solver import Solver
from src.euler import (
    Euler,
    FLUX_RUSANOV,
    FLUX_ROE,
    FLUX_HLLE,
)
from src.nvtx import NvtxContext


comptime P = 2
comptime N_RES = 8
comptime LX = 1.0
comptime LY = 1.0
comptime LZ = 1.0
comptime GAMMA: Float32 = 1.4
comptime RHO0: Float32 = 1.0
comptime U0: Float32 = 1.0
comptime V0: Float32 = 1.0
comptime W0: Float32 = 1.0
comptime P0: Float32 = 1.0
comptime AMPLITUDE: Float32 = 0.1
comptime T_FINAL: Float32 = 1.0  # one advection period at u0 = 1
comptime CFL = Float32(0.2)
comptime IC_BLOCK = 256
comptime PI_F: Float32 = 3.14159265358979323846

# Loose threshold; the more dissipative fluxes (Rusanov) should
# still pass.  Existing HLLEC bench keeps the tight ~1e-3 gate.
comptime L2_MAX_REL: Float64 = 5.0e-3


def entropy_wave_ic_kernel(
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
    var pz = elem_node_xyz[(e * N_P + nn) * 3 + 2]

    var k = Float32(2.0) * PI_F / Float32(LX)
    var rho = RHO0 + AMPLITUDE * sin(k * px) * sin(k * py) * sin(k * pz)
    var u = U0
    var v = V0
    var w = W0
    var p = P0
    var E = p / (GAMMA - Float32(1.0)) + Float32(0.5) * rho * (
        u * u + v * v + w * w
    )

    var base = (e * N_P + nn) * 5
    q[base + 0] = rho
    q[base + 1] = rho * u
    q[base + 2] = rho * v
    q[base + 3] = rho * w
    q[base + 4] = E


def _run(flux_type: Int, entropy_fix: Bool = False) raises -> Float64:
    """Run one period under the specified flux_type (and optional Harten-
    Hyman entropy fix for the wave-based solvers); return rel L2."""
    var rank = mpi.world_rank()
    var size = mpi.world_size()
    var nvtx = NvtxContext()
    var refs = build_reference_operators(nvtx)
    var ctx = DeviceContext()

    var mesh = Mesh(
        ctx,
        build_partition(rank, size, N_RES, N_RES, N_RES),
        LX,
        LY,
        LZ,
        BoundaryConditions.periodic(),
    )
    var halo = HaloExchange(
        ctx,
        mesh.part,
        Euler.NUM_COMPONENTS,
        mesh.d_perm.unsafe_ptr(),
    )
    var physics = Euler(
        GAMMA,
        Float32(1.0e-6),
        Float32(1.0e-6),
        flux_type,
        entropy_fix,
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

    solver.ctx.enqueue_function[entropy_wave_ic_kernel, entropy_wave_ic_kernel](
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

    var h = Float32(LX) / Float32(N_RES)
    var c_inf = sqrt(GAMMA * P0 / RHO0)
    var wave_max = sqrt(U0 * U0 + V0 * V0 + W0 * W0) + c_inf
    var dt_est = CFL * h / (wave_max * Float32(2 * P + 1))
    var num_steps = Int(T_FINAL / dt_est) + 1
    var dt = T_FINAL / Float32(num_steps)

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
            raise Error("bench_euler_flux_coverage_3d: non-finite output")
        var err = Float64(v_now - host_ic[k])
        sum_sq += err * err
        var ic = Float64(host_ic[k])
        sum_ic += ic * ic
    var l2 = sqrt(sum_sq / Float64(n_owned_dof))
    var l2_ic = sqrt(sum_ic / Float64(n_owned_dof))
    return l2 / l2_ic


def _gate(name: String, rel_l2: Float64) raises:
    print("  ", name, "rel L2 =", rel_l2, "  (threshold", L2_MAX_REL, ")")
    if rel_l2 > L2_MAX_REL:
        raise Error(
            String("bench_euler_flux_coverage_3d FAILED: ")
            + name
            + " rel L2 "
            + String(rel_l2)
            + " exceeds "
            + String(L2_MAX_REL)
        )


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    mpi.init()
    var size = mpi.world_size()
    if size > 1:
        mpi.finalize()
        print("bench_euler_flux_coverage_3d: runs at np=1 only")
        return

    print("bench_euler_flux_coverage_3d (3D entropy wave under 3 fluxes)")
    print(
        "  P=",
        P,
        "  N=",
        N_RES,
        "  exercising FLUX_RUSANOV, FLUX_ROE, FLUX_HLLE",
        " (each x {entropy_fix=False, True} for the wave-based solvers)",
    )

    # Each flux type without entropy fix.  Smooth IC + uniform background
    # flow -> no sonic transitions -> entropy_fix on/off should agree to
    # Float32 noise.  Both arms gated independently so a regression in
    # the entropy-fix path can't be hidden by the smooth IC.
    var err_rusanov = _run(FLUX_RUSANOV)
    _gate("Rusanov                  ", err_rusanov)

    var err_roe = _run(FLUX_ROE)
    _gate("Roe  (entropy_fix=False) ", err_roe)
    var err_roe_efix = _run(FLUX_ROE, entropy_fix=True)
    _gate("Roe  (entropy_fix=True)  ", err_roe_efix)

    var err_hlle = _run(FLUX_HLLE)
    _gate("HLLE (entropy_fix=False) ", err_hlle)
    var err_hlle_efix = _run(FLUX_HLLE, entropy_fix=True)
    _gate("HLLE (entropy_fix=True)  ", err_hlle_efix)

    print("=== bench_euler_flux_coverage_3d PASSED ===")
    mpi.finalize()
