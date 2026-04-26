# ======================================================================
# bench_maxwell_uniform_j_3d_p3 -- 3D Maxwell J source at P=3 (NP=20)
# ======================================================================
#
# P=3 (NP=20 nodes per tet) counterpart of bench_maxwell_uniform_j_3d.
# Same uniform J source-term gate (q=0 IC, J=(Jx,0,0), Ex(T) =
# -c^2*Jx*T exact linear ramp), but routed through Mesh[3] /
# Solver[Maxwell, 3] / rk_stage_kernel[3] so the 3D Maxwell J source
# path lands at NP=20.  Mirrors bench_maxwell_uniform_j_2d_p3 (NP=10
# in 2D); together with the P=2 variants and the M-arm pair, the
# Maxwell source coverage now has full 4-corner P / dimension parity.
#
# Pass criteria (P=3, periodic 4x4x4, T=0.5):
#   * |Ex - Ex_exact| / |Ex_exact| < 1e-5 (Float32 roundoff floor)
#   * max |Ey, Ez, Bx, By, Bz| < 1e-5
#   * no NaN / Inf
# ======================================================================

from std.sys import has_accelerator
from std.gpu.host import DeviceContext, DeviceBuffer
from std.math import sqrt, isnan, isinf

from src import mpi
from src.partition import build_partition
from src.reference import ReferenceElement, to_float32, num_tet_nodes
from src.mesh import Mesh
from src.boundary import BoundaryConditions
from src.halo_exchange import HaloExchange
from src.solver import Solver
from src.maxwell import Maxwell
from src.nvtx import NvtxContext


comptime P = 3
comptime NP = num_tet_nodes(P)
comptime NX = 4
comptime NY = 4
comptime NZ = 4
comptime LX = 1.0
comptime LY = 1.0
comptime LZ = 1.0

comptime C_LIGHT: Float32 = 1.0
comptime JX:     Float32 = 1.0
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
        print("bench_maxwell_uniform_j_3d_p3: runs at np=1 only")
        return

    print("bench_maxwell_uniform_j_3d_p3 (uniform J source at P=3)")
    print("  P=", P, "  NP=", NP, "  mesh=", NX, "x", NY, "x", NZ,
          "   Jx=", JX, "   T=", T_FINAL)

    var rank = mpi.world_rank()
    var nvtx = NvtxContext()
    var ctx = DeviceContext()

    # Build P=3 reference operators directly: build_reference_operators()
    # in src.reference defaults to P=2 / NP=10, wrong for Solver[Maxwell, 3].
    var re = ReferenceElement[P]()
    var D_ref = to_float32(re.D_ref)
    var Lift_ref = to_float32(re.Lift_ref)
    var node_weights = to_float32(re.node_weights)

    var bcs = BoundaryConditions.periodic()
    var mesh = Mesh[P](
        ctx, build_partition(rank, size, NX, NY, NZ), LX, LY, LZ, bcs,
    )
    var halo = HaloExchange(
        ctx, mesh.part, Maxwell.NUM_COMPONENTS,
        mesh.d_perm.unsafe_ptr(), bcs,
    )
    var physics = Maxwell(
        C_LIGHT,
        JX, Float32(0.0), Float32(0.0),       # J = (Jx, 0, 0)
        Float32(0.0), Float32(0.0), Float32(0.0),  # M = 0
    )
    var solver = Solver[Maxwell, P](
        ctx^, mesh^, halo^, physics^, D_ref^, Lift_ref^, node_weights^,
    )

    # IC: all components zero.  d_q is created zero-initialised by
    # solver construction, so no IC kernel needed.

    var n_owned_dof = solver.num_owned_elements * NP * Maxwell.NUM_COMPONENTS

    var c2 = C_LIGHT * C_LIGHT
    var h = Float32(LX) / Float32(NX)
    var dt_est = CFL * h / (C_LIGHT * Float32(2 * P + 1))
    var num_steps = Int(T_FINAL / dt_est) + 1
    var dt = T_FINAL / Float32(num_steps)
    print("  steps=", num_steps, "  dt=", dt)

    for _ in range(num_steps):
        solver.step_ssprk3(dt, nvtx)
    solver.ctx.synchronize()

    var hbuf_q = solver.ctx.enqueue_create_host_buffer[DType.float32](n_owned_dof)
    solver.ctx.enqueue_copy(
        hbuf_q, solver.d_q.create_sub_buffer[DType.float32](0, n_owned_dof),
    )
    solver.ctx.synchronize()
    var q_ptr = hbuf_q.unsafe_ptr()

    # Expected Ex at every node: -c^2 * Jx * T.
    var ex_exact = -c2 * JX * T_FINAL

    var max_ex_dev: Float64 = 0.0
    var max_other: Float64 = 0.0
    var n_owned_nodes = solver.num_owned_elements * NP
    for i in range(n_owned_nodes):
        var ex = q_ptr[i * 6 + 0]
        var ey = q_ptr[i * 6 + 1]
        var ez = q_ptr[i * 6 + 2]
        var bx = q_ptr[i * 6 + 3]
        var by = q_ptr[i * 6 + 4]
        var bz = q_ptr[i * 6 + 5]
        if (isnan(ex) or isinf(ex) or isnan(ey) or isinf(ey)
            or isnan(ez) or isinf(ez) or isnan(bx) or isinf(bx)
            or isnan(by) or isinf(by) or isnan(bz) or isinf(bz)):
            raise Error("bench_maxwell_uniform_j_3d_p3: non-finite output")
        var ex_dev = Float64(ex - ex_exact)
        if ex_dev < 0.0: ex_dev = -ex_dev
        if ex_dev > max_ex_dev: max_ex_dev = ex_dev
        var other_max = Float64(0.0)
        var values = [Float64(ey), Float64(ez), Float64(bx),
                      Float64(by), Float64(bz)]
        for k in range(5):
            var v = values[k]
            if v < 0.0: v = -v
            if v > other_max: other_max = v
        if other_max > max_other: max_other = other_max

    var ex_rel = max_ex_dev / Float64(ex_exact)
    if ex_rel < 0.0: ex_rel = -ex_rel
    print("  Ex exact       =", ex_exact)
    print("  max |Ex - Ex_exact| / |Ex_exact| =", ex_rel,
          "  (threshold", EX_REL_TOL, ")")
    print("  max |Ey, Ez, Bx, By, Bz| =", max_other,
          "  (threshold", ZERO_COMPONENT_TOL, ")")

    if ex_rel > EX_REL_TOL:
        raise Error(
            "bench_maxwell_uniform_j_3d_p3 FAILED: Ex deviation "
            + String(ex_rel) + " > " + String(EX_REL_TOL)
        )
    if max_other > ZERO_COMPONENT_TOL:
        raise Error(
            "bench_maxwell_uniform_j_3d_p3 FAILED: zero-component leak "
            + String(max_other) + " > " + String(ZERO_COMPONENT_TOL)
        )

    print("=== bench_maxwell_uniform_j_3d_p3 PASSED ===")
    mpi.finalize()
