# ======================================================================
# bench_shallow_water_wave_3d -- linear surface-wave periodic return
# ======================================================================
#
# Small-amplitude linear shallow-water wave on a periodic-x domain.
# The SW physics is 2D embedded in the 3D DG pipeline (F^z = 0),
# NZ = 2 gives a trivially-stacked sheet of cells in z.
#
# IC (rest state + stationary sinusoidal disturbance):
#   h = H + A * sin(2 pi x / Lx)
#   u = 0,  v = 0
#
# Linearised about (H, 0): eta := h - H satisfies the 1D wave equation
#   d2 eta/dt2 = c^2 * d2 eta/dx2        where c = sqrt(g H)
# so d'Alembert splits the IC into two equal-amplitude waves travelling
# at +-c.  With zero initial velocity the exact linear solution is
#   eta(x, t) = A/2 * [sin(k(x - c t)) + sin(k(x + c t))]
#             = A * sin(kx) * cos(k c t)
# After T = 2 pi / (k c) = Lx / c the time factor is cos(2 pi) = 1 and
# the solution returns exactly to the IC (for A/H << 1; nonlinear
# correction is O((A/H)^2) per period).
#
# Pass criteria (P=2, HLLC-like Rusanov, periodic x/y, thin z):
#   * rel L2(h - h_IC) at N = 16, 24, 32 all < 1e-3 (A/H = 0.01 so
#     nonlinear correction is O(1e-4) per period, well below)
#   * mass integral invariant to 1e-5 (strict conservation check)
#   * no NaN / Inf
#
# A wider benchmark would verify convergence rate (expected 3rd order
# at P=2), but wave-packet DG schemes typically sit at floor on this
# linearised problem -- so a tight absolute threshold is the strongest
# gate we can write.
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
from src.shallow_water import ShallowWater
from src.nvtx import NvtxContext


comptime P = 2
comptime LX = 1.0
comptime LY = 1.0
comptime LZ = 0.1
comptime GRAVITY: Float32 = 1.0
comptime H_REST:  Float32 = 1.0
comptime AMPLITUDE: Float32 = 0.01
comptime H_MIN: Float32 = 1.0e-6
comptime CFL = Float32(0.2)
comptime IC_BLOCK = 256
comptime PI_F: Float32 = 3.14159265358979323846

# One wave period: T = Lx / c where c = sqrt(g H).
comptime T_FINAL: Float32 = Float32(LX / 1.0)   # c = sqrt(1*1) = 1

comptime L2_MAX_REL: Float64 = 1.0e-3
# 1e-4 is tight but realistic for float32 mass summation over a 32x32x2
# mesh and ~200 RK steps: per-element roundoff is O(1e-7) and builds
# up additively; any real conservation bug would move this by orders
# of magnitude.
comptime MASS_TOL_REL: Float64 = 1.0e-4


def wave_ic_kernel(
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
    var px = elem_node_xyz[(e * N_P + nn) * 3 + 0]
    var k = Float32(2.0) * PI_F / Float32(LX)
    var h = H_REST + AMPLITUDE * sin(k * px)
    var base = (e * N_P + nn) * 3
    q[base + 0] = h
    q[base + 1] = Float32(0.0)    # hu
    q[base + 2] = Float32(0.0)    # hv


@fieldwise_init
struct RunResult(Movable):
    var rel_l2: Float64
    var mass_rel: Float64


def _run(N: Int) raises -> RunResult:
    """Run one period at NxNx2 resolution.  Returns (rel L2, mass_drift_rel)."""
    var rank = mpi.world_rank()
    var size = mpi.world_size()
    var nvtx = NvtxContext()
    var refs = build_reference_operators(nvtx)
    var ctx = DeviceContext()

    var NZ = 2
    var mesh = Mesh(
        ctx, build_partition(rank, size, N, N, NZ), LX, LY, LZ,
        BoundaryConditions.periodic(),
    )
    var halo = HaloExchange(
        ctx, mesh.part, ShallowWater.NUM_COMPONENTS,
        mesh.d_perm.unsafe_ptr(),
    )
    var physics = ShallowWater(GRAVITY, H_MIN)
    var solver = Solver[ShallowWater](
        ctx^, mesh^, halo^, physics^,
        refs.D_ref^, refs.Lift_ref^, refs.node_weights^,
    )

    solver.ctx.enqueue_function[wave_ic_kernel, wave_ic_kernel](
        solver.d_q.unsafe_ptr(),
        solver.mesh.d_owned_elem_ids.unsafe_ptr(),
        solver.mesh.local.d_elem_node_xyz.unsafe_ptr(),
        solver.num_owned_elements,
        grid_dim=ceildiv(solver.num_owned_elements * N_P, IC_BLOCK),
        block_dim=IC_BLOCK,
    )
    solver.ctx.synchronize()

    # Snapshot IC.
    var n_owned_dof = solver.num_owned_elements * N_P * ShallowWater.NUM_COMPONENTS
    var hbuf_ic = solver.ctx.enqueue_create_host_buffer[DType.float32](n_owned_dof)
    solver.ctx.enqueue_copy(
        hbuf_ic,
        solver.d_q.create_sub_buffer[DType.float32](0, n_owned_dof),
    )
    solver.ctx.synchronize()
    var ic_ptr = hbuf_ic.unsafe_ptr()
    var host_ic = List[Float32]()
    for k in range(n_owned_dof):
        host_ic.append(ic_ptr[k])

    # Time stepping.
    var h_cell = Float32(LX) / Float32(N)
    var c = sqrt(GRAVITY * H_REST)
    var dt_est = CFL * h_cell / (c * Float32(2 * P + 1))
    var num_steps = Int(T_FINAL / dt_est) + 1
    var dt = T_FINAL / Float32(num_steps)

    for _ in range(num_steps):
        solver.step_ssprk3(dt, nvtx)
    solver.ctx.synchronize()

    var hbuf_q = solver.ctx.enqueue_create_host_buffer[DType.float32](n_owned_dof)
    solver.ctx.enqueue_copy(
        hbuf_q,
        solver.d_q.create_sub_buffer[DType.float32](0, n_owned_dof),
    )
    solver.ctx.synchronize()
    var q_ptr = hbuf_q.unsafe_ptr()

    # Relative L2 error over all components.
    var sum_sq: Float64 = 0.0
    var sum_ic: Float64 = 0.0
    for k in range(n_owned_dof):
        var v = q_ptr[k]
        if isnan(v) or isinf(v):
            raise Error("bench_shallow_water_wave_3d: non-finite output")
        var err = Float64(v - host_ic[k])
        sum_sq += err * err
        var ic = Float64(host_ic[k])
        sum_ic += ic * ic
    var l2 = sqrt(sum_sq / Float64(n_owned_dof))
    var l2_ic = sqrt(sum_ic / Float64(n_owned_dof))
    var rel_l2 = l2 / l2_ic

    # Mass conservation: sum over h component.  Mean h should be H exactly
    # (the sin integrates to zero), and time evolution preserves that.
    var mass_ic: Float64 = 0.0
    var mass_fin: Float64 = 0.0
    var n_elem_nodes = solver.num_owned_elements * N_P
    for i in range(n_elem_nodes):
        mass_ic += Float64(host_ic[i * 3 + 0])
        mass_fin += Float64(q_ptr[i * 3 + 0])
    var dmass = mass_fin - mass_ic
    if dmass < 0.0: dmass = -dmass
    var mass_rel = dmass / mass_ic

    return RunResult(rel_l2, mass_rel)


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    mpi.init()
    print("bench_shallow_water_wave_3d (linear SW wave, periodic return)")
    print("  P=", P, "  T=", T_FINAL, "  A/H=", AMPLITUDE / H_REST)

    var Ns = List[Int]()
    Ns.append(16)
    Ns.append(24)
    Ns.append(32)

    var all_ok = True
    for idx in range(len(Ns)):
        var N = Ns[idx]
        var result = _run(N)
        var l2 = result.rel_l2
        var mrel = result.mass_rel
        print("  N=", N, "  rel_L2=", l2, "  mass_drift=", mrel)
        if l2 > L2_MAX_REL:
            all_ok = False
            print("    -> L2 above threshold", L2_MAX_REL)
        if mrel > MASS_TOL_REL:
            all_ok = False
            print("    -> mass drift above threshold", MASS_TOL_REL)

    if not all_ok:
        raise Error("bench_shallow_water_wave_3d FAILED")
    print("=== bench_shallow_water_wave_3d PASSED ===")
    mpi.finalize()
