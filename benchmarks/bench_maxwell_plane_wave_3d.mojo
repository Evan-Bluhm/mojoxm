# ======================================================================
# bench_maxwell_plane_wave_3d -- 3D TM plane wave on a periodic box
# ======================================================================
#
# 3D companion to bench_maxwell_plane_wave_2d.  Periodic [0, 1]^3 box,
# plane-wave TM-mode traveling in +x at speed c:
#
#   Ez(x, t) = cos(2 pi (x - c t))
#   By(x, t) = -(1 / c) cos(2 pi (x - c t))
#   Ex = Ey = Bx = Bz = 0
#
# After T = 1/c (one period at c=1) the exact solution returns to IC;
# residual L2 is pure scheme dissipation.  Pairs with
# bench_maxwell_plane_wave_2d (2D box) and bench_maxwell_cavity_3d
# (PEC standing wave) to fully cover the 3D Maxwell GPU path:
# periodic faces + actual propagation, in addition to wall reflection
# + standing modes.  TE-mode dual lives in
# bench_maxwell_te_plane_wave_3d (Bz / Ey nonzero instead of
# Ez / By); both polarizations together gate every flux path in
# the 3D Maxwell volume + face-flux kernels.
#
# Pass criteria (P=2, NX=16, NY=NZ=4, single-rank):
#   * rel L2(6-component state) < 1e-3
#   * leakage into the analytically-zero components (Ex, Ey, Bx, Bz)
#     bounded < 5e-4 (Kuhn-tet asymmetry under Rusanov leaks O(h)
#     into orthogonal components)
#   * no NaN / Inf
# ======================================================================

from std.sys import has_accelerator
from std.gpu import global_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from std.math import sqrt, ceildiv, cos, isnan, isinf

from src import mpi
from src.partition import build_partition
from src.reference import N_P, build_reference_operators
from src.mesh import Mesh
from src.boundary import BoundaryConditions
from src.halo_exchange import HaloExchange
from src.solver import Solver
from src.maxwell import Maxwell
from src.nvtx import NvtxContext


comptime NX = 16
comptime NY = 4
comptime NZ = 4
comptime LX = 1.0
comptime LY = 1.0
comptime LZ = 1.0

comptime C_LIGHT: Float32 = 1.0
comptime T_FINAL: Float32 = 1.0  # one period at c = 1
comptime CFL = Float32(0.2)
comptime IC_BLOCK = 256
comptime TWO_PI_F: Float32 = 6.28318530717958647692

# Empirical: ~5e-4 at NX=16 NY=NZ=4 P=2; 1e-3 leaves ~2x margin.
comptime L2_MAX_REL: Float64 = 1.0e-3
# Empirical leakage ~3e-4 (Kuhn-tet asymmetry).  5e-4 = ~1.7x margin.
comptime ZERO_COMPONENT_MAX: Float64 = 5.0e-4


def plane_wave_ic_kernel(
    q: UnsafePointer[Float32, MutAnyOrigin],
    owned_elem_ids: UnsafePointer[Int32, MutAnyOrigin],
    elem_node_xyz: UnsafePointer[Float32, MutAnyOrigin],
    num_owned: Int,
    inv_c: Float32,
):
    var idx = Int(global_idx.x)
    var total = num_owned * N_P
    if idx >= total:
        return
    var i = idx // N_P
    var nn = idx % N_P
    var e = Int(owned_elem_ids[i])
    var px = elem_node_xyz[(e * N_P + nn) * 3 + 0]
    var Ez = cos(TWO_PI_F * px)
    var By = -inv_c * cos(TWO_PI_F * px)
    var base = (e * N_P + nn) * 6
    q[base + 0] = Float32(0.0)  # Ex
    q[base + 1] = Float32(0.0)  # Ey
    q[base + 2] = Ez  # Ez
    q[base + 3] = Float32(0.0)  # Bx
    q[base + 4] = By  # By
    q[base + 5] = Float32(0.0)  # Bz


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    mpi.init()
    var size = mpi.world_size()
    if size > 1:
        mpi.finalize()
        print("bench_maxwell_plane_wave_3d: runs at np=1 only")
        return

    print("bench_maxwell_plane_wave_3d (3D TM plane wave, periodic box)")
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
        Maxwell.NUM_COMPONENTS,
        mesh.d_perm.unsafe_ptr(),
        bcs,
    )
    var physics = Maxwell(
        C_LIGHT,
        Float32(0.0),
        Float32(0.0),
        Float32(0.0),
        Float32(0.0),
        Float32(0.0),
        Float32(0.0),
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

    var inv_c = Float32(1.0) / C_LIGHT
    solver.ctx.enqueue_function[plane_wave_ic_kernel, plane_wave_ic_kernel](
        solver.d_q.unsafe_ptr(),
        solver.mesh.d_owned_elem_ids.unsafe_ptr(),
        solver.mesh.local.d_elem_node_xyz.unsafe_ptr(),
        solver.num_owned_elements,
        inv_c,
        grid_dim=ceildiv(solver.num_owned_elements * N_P, IC_BLOCK),
        block_dim=IC_BLOCK,
    )
    solver.ctx.synchronize()

    var n_owned_dof = solver.num_owned_elements * N_P * Maxwell.NUM_COMPONENTS
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

    var sum_sq: Float64 = 0.0
    var sum_ic: Float64 = 0.0
    var max_zero_leak: Float64 = 0.0
    var nc = Maxwell.NUM_COMPONENTS
    for k in range(n_owned_dof):
        var v_now = q_ptr[k]
        if isnan(v_now) or isinf(v_now):
            raise Error("bench_maxwell_plane_wave_3d: non-finite output")
        var e = Float64(v_now - host_ic[k])
        sum_sq += e * e
        var ic = Float64(host_ic[k])
        sum_ic += ic * ic
        # Components Ex (0), Ey (1), Bx (3), Bz (5) are identically
        # zero in the analytic solution.
        var c_idx = k % nc
        if c_idx == 0 or c_idx == 1 or c_idx == 3 or c_idx == 5:
            var av = Float64(v_now)
            if av < 0.0:
                av = -av
            if av > max_zero_leak:
                max_zero_leak = av

    var l2 = sqrt(sum_sq / Float64(n_owned_dof))
    var l2_ic = sqrt(sum_ic / Float64(n_owned_dof))
    var rel_l2 = l2 / l2_ic
    print("  rel L2(state) =", rel_l2, "  (threshold", L2_MAX_REL, ")")
    print(
        "  max |Ex|/|Ey|/|Bx|/|Bz| =",
        max_zero_leak,
        "  (threshold",
        ZERO_COMPONENT_MAX,
        ")",
    )

    if rel_l2 > L2_MAX_REL:
        raise Error(
            "bench_maxwell_plane_wave_3d FAILED: rel L2 "
            + String(rel_l2)
            + " exceeds threshold "
            + String(L2_MAX_REL)
        )
    if max_zero_leak > ZERO_COMPONENT_MAX:
        raise Error(
            String("bench_maxwell_plane_wave_3d FAILED: zero-component ")
            + "leakage "
            + String(max_zero_leak)
            + " > "
            + String(ZERO_COMPONENT_MAX)
        )
    print("=== bench_maxwell_plane_wave_3d PASSED ===")
    mpi.finalize()
