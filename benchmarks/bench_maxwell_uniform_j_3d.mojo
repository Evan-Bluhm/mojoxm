# ======================================================================
# bench_maxwell_uniform_j_3d -- Maxwell J source-term gate
# ======================================================================
#
# Closes a coverage gap parallel to bench_euler_hydrostatic_3d: the
# Maxwell source term in src/maxwell.mojo (lines 237+) accepts a
# uniform current J and magnetization M but every existing Maxwell
# bench / example sets them to zero, so the source code path is
# completely untested.  A regression in the dE/dt += -c^2*J or
# dB/dt += -M coupling would not have tripped any gate.
#
# Cleanest analytic test: uniform J on a periodic box with q = 0
# at t = 0.  Since q is uniform, all flux divergences vanish (no
# spatial gradients) and only the source contributes:
#
#   dEx/dt = -c^2 * Jx  ->  Ex(T) = -c^2 * Jx * T   (exact)
#   dEy/dt = -c^2 * Jy  ->  Ey(T) = -c^2 * Jy * T
#   ...
#   B(T) = 0  identically (M = 0)
#
# After T_FINAL the analytic state has only Ex nonzero (we pick
# J = (Jx, 0, 0)).  The bench verifies Ex matches the linear ramp
# exactly up to Float32 roundoff and that all other components stay
# at machine epsilon.
#
# Pass criteria (P=2, periodic 8x8x8, T=0.5):
#   * |Ex - Ex_exact| / |Ex_exact| < 1e-5 (Float32 roundoff floor)
#   * max |Ey, Ez, Bx, By, Bz| < 1e-5
#   * no NaN / Inf
# ======================================================================

from std.sys import has_accelerator
from std.gpu.host import DeviceContext, DeviceBuffer
from std.math import sqrt, isnan, isinf

from src import mpi
from src.partition import build_partition
from src.reference import N_P, build_reference_operators
from src.mesh import Mesh
from src.boundary import BoundaryConditions
from src.halo_exchange import HaloExchange
from src.solver import Solver
from src.maxwell import Maxwell
from src.nvtx import NvtxContext


comptime NX = 8
comptime NY = 8
comptime NZ = 8
comptime LX = 1.0
comptime LY = 1.0
comptime LZ = 1.0

comptime C_LIGHT: Float32 = 1.0
comptime JX: Float32 = 1.0
comptime T_FINAL: Float32 = 0.5
comptime CFL = Float32(0.2)

# Float32 roundoff floor over ~80 SSPRK3 steps on this problem.
comptime EX_REL_TOL: Float64 = 1.0e-5
comptime ZERO_COMPONENT_TOL: Float64 = 1.0e-5


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    mpi.init()
    var size = mpi.world_size()
    if size > 1:
        mpi.finalize()
        print("bench_maxwell_uniform_j_3d: runs at np=1 only")
        return

    print("bench_maxwell_uniform_j_3d (uniform J source-term gate)")
    print(
        "  P= 2   mesh=", NX, "x", NY, "x", NZ, "   Jx=", JX, "   T=", T_FINAL
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
        Maxwell.NUM_COMPONENTS,
        mesh.d_perm.unsafe_ptr(),
        bcs,
    )
    var physics = Maxwell(
        C_LIGHT,
        JX,
        Float32(0.0),
        Float32(0.0),  # J = (Jx, 0, 0)
        Float32(0.0),
        Float32(0.0),
        Float32(0.0),  # M = 0
    )
    var solver = Solver[Maxwell](
        ctx^,
        mesh^,
        halo^,
        physics^,
        refs.D_ref^,
        refs.Lift_ref^,
        refs.node_weights^,
    )

    # IC: all components zero.  d_q is created zero-initialised by
    # solver construction, so no IC kernel needed.

    var n_owned_dof = solver.num_owned_elements * N_P * Maxwell.NUM_COMPONENTS

    var c2 = C_LIGHT * C_LIGHT
    var h = Float32(LX) / Float32(NX)
    var dt_est = CFL * h / (C_LIGHT * Float32(5.0))
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

    # Expected Ex at every node: -c^2 * Jx * T.
    var ex_exact = -c2 * JX * T_FINAL

    var max_ex_dev: Float64 = 0.0
    var max_other: Float64 = 0.0
    var n_owned_nodes = solver.num_owned_elements * N_P
    for i in range(n_owned_nodes):
        var ex = q_ptr[i * 6 + 0]
        var ey = q_ptr[i * 6 + 1]
        var ez = q_ptr[i * 6 + 2]
        var bx = q_ptr[i * 6 + 3]
        var by = q_ptr[i * 6 + 4]
        var bz = q_ptr[i * 6 + 5]
        if (
            isnan(ex)
            or isinf(ex)
            or isnan(ey)
            or isinf(ey)
            or isnan(ez)
            or isinf(ez)
            or isnan(bx)
            or isinf(bx)
            or isnan(by)
            or isinf(by)
            or isnan(bz)
            or isinf(bz)
        ):
            raise Error("bench_maxwell_uniform_j_3d: non-finite output")
        var ex_dev = Float64(ex - ex_exact)
        if ex_dev < 0.0:
            ex_dev = -ex_dev
        if ex_dev > max_ex_dev:
            max_ex_dev = ex_dev
        var other_max = Float64(0.0)
        var values = [
            Float64(ey),
            Float64(ez),
            Float64(bx),
            Float64(by),
            Float64(bz),
        ]
        for k in range(5):
            var v = values[k]
            if v < 0.0:
                v = -v
            if v > other_max:
                other_max = v
        if other_max > max_other:
            max_other = other_max

    var ex_rel = max_ex_dev / Float64(ex_exact)
    if ex_rel < 0.0:
        ex_rel = -ex_rel
    print("  Ex exact       =", ex_exact)
    print(
        "  max |Ex - Ex_exact| / |Ex_exact| =",
        ex_rel,
        "  (threshold",
        EX_REL_TOL,
        ")",
    )
    print(
        "  max |Ey, Ez, Bx, By, Bz| =",
        max_other,
        "  (threshold",
        ZERO_COMPONENT_TOL,
        ")",
    )

    if ex_rel > EX_REL_TOL:
        raise Error(
            "bench_maxwell_uniform_j_3d FAILED: Ex deviation "
            + String(ex_rel)
            + " > "
            + String(EX_REL_TOL)
        )
    if max_other > ZERO_COMPONENT_TOL:
        raise Error(
            "bench_maxwell_uniform_j_3d FAILED: zero-component leak "
            + String(max_other)
            + " > "
            + String(ZERO_COMPONENT_TOL)
        )

    print("=== bench_maxwell_uniform_j_3d PASSED ===")
    mpi.finalize()
