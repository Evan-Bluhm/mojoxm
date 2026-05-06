# ======================================================================
# bench_maxwell_plane_wave_3d_p5 -- 3D TM plane wave at P=5 (NP=56)
# ======================================================================
#
# P=5 counterpart of bench_maxwell_plane_wave_3d_p4.  Same TM-mode
# plane wave on a triply-periodic [0, 1] x [0, NY/NX] x [0, NZ/NX]
# box, but routed through Mesh[5] / Solver[Maxwell, 5] with NP = 56
# nodes per tet.
#
# Pairs with bench_advection_3d_p5, bench_euler_smooth_wave_3d_p5
# (also NP=56 multi-component) to validate Maxwell at NP=56 in 3D.
# Highest-order vacuum-Maxwell gate in the 3D suite, completing the
# P-parity sweep (P=2 / P=3 / P=4 / P=5) for 3D linear-flux EM.
#
# Pass criteria (P=5, NX=8 NY=NZ=4, single-rank, periodic):
#   * rel L2(state, 6 components) < 5e-4 (Float32 floor regime;
#     Kuhn-tet asymmetry under Rusanov + Float32 step accumulation
#     dominate -- the P=5 dispersion error is well below this)
#   * leakage into TM-zero components (Ex, Ey, Bx, Bz) < 5e-4
#   * no NaN / Inf
# ======================================================================

from std.sys import has_accelerator
from std.gpu import global_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from std.math import sqrt, ceildiv, cos, isnan, isinf

from src import mpi
from src.partition import build_partition
from src.reference import (
    ReferenceElement,
    to_float32,
    num_tet_nodes,
)
from src.mesh import Mesh
from src.boundary import BoundaryConditions
from src.halo_exchange import HaloExchange
from src.solver import Solver
from src.maxwell import Maxwell
from src.nvtx import NvtxContext


comptime P = 5
comptime NP = num_tet_nodes(P)  # 56 at P=5
comptime NC = 6  # Maxwell.NUM_COMPONENTS

comptime NX = 8
comptime NY = 4
comptime NZ = 4
comptime LX = 1.0
comptime LY = Float64(NY) / Float64(NX) * LX
comptime LZ = Float64(NZ) / Float64(NX) * LX

comptime C_LIGHT: Float32 = 1.0
comptime T_FINAL: Float32 = 1.0  # one period at c = 1
comptime CFL = Float32(0.08)
comptime IC_BLOCK = 256
comptime TWO_PI_F: Float32 = 6.28318530717958647692

# At P=5 / NP=56 the dispersion error is well below the Kuhn-tet
# asymmetry floor.  5e-4 keeps the same generous margin used at
# P=3/P=4 -- tighter risks false-firing on Float32 wobble across
# hardware while still catching any meaningful Maxwell regression.
comptime L2_MAX_REL: Float64 = 5.0e-4
comptime ZERO_COMPONENT_MAX: Float64 = 5.0e-4


def plane_wave_ic_kernel(
    q: UnsafePointer[Float32, MutAnyOrigin],
    owned_elem_ids: UnsafePointer[Int32, MutAnyOrigin],
    elem_node_xyz: UnsafePointer[Float32, MutAnyOrigin],
    num_owned: Int,
    inv_c: Float32,
):
    var idx = Int(global_idx.x)
    var total = num_owned * NP
    if idx >= total:
        return
    var i = idx // NP
    var nn = idx % NP
    var e = Int(owned_elem_ids[i])
    var px = elem_node_xyz[(e * NP + nn) * 3 + 0]
    var Ez = cos(TWO_PI_F * px)
    var By = -inv_c * cos(TWO_PI_F * px)
    var base = (e * NP + nn) * 6
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
        print("bench_maxwell_plane_wave_3d_p5: runs at np=1 only")
        return

    print("bench_maxwell_plane_wave_3d_p5 (3D TM plane wave, P=5, periodic)")
    print(
        "  P=", P, "  NP=", NP, "  mesh=", NX, "x", NY, "x", NZ, "  T=", T_FINAL
    )

    var rank = mpi.world_rank()
    var nvtx = NvtxContext()
    var ctx = DeviceContext()

    var re = ReferenceElement[P]()
    var D_ref = to_float32(re.D_ref)
    var Lift_ref = to_float32(re.Lift_ref)
    var node_weights = to_float32(re.node_weights)

    var bcs = BoundaryConditions.periodic()
    var mesh = Mesh[P](
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
    var solver = Solver[Maxwell, P](
        ctx^,
        mesh^,
        halo^,
        physics^,
        D_ref^,
        Lift_ref^,
        node_weights^,
    )

    var inv_c = Float32(1.0) / C_LIGHT
    solver.ctx.enqueue_function[plane_wave_ic_kernel, plane_wave_ic_kernel](
        solver.d_q.unsafe_ptr(),
        solver.mesh.d_owned_elem_ids.unsafe_ptr(),
        solver.mesh.local.d_elem_node_xyz.unsafe_ptr(),
        solver.num_owned_elements,
        inv_c,
        grid_dim=ceildiv(solver.num_owned_elements * NP, IC_BLOCK),
        block_dim=IC_BLOCK,
    )
    solver.ctx.synchronize()

    var n_owned_dof = solver.num_owned_elements * NP * NC
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
    # CFL: factor 2P+1 = 11 at P=5.
    var dt_est = CFL * h / (C_LIGHT * Float32(2 * P + 1))
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
    for k in range(n_owned_dof):
        var v_now = q_ptr[k]
        if isnan(v_now) or isinf(v_now):
            raise Error("bench_maxwell_plane_wave_3d_p5: non-finite output")
        var e = Float64(v_now - host_ic[k])
        sum_sq += e * e
        var ic = Float64(host_ic[k])
        sum_ic += ic * ic
        var c_idx = k % NC
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
            "bench_maxwell_plane_wave_3d_p5 FAILED: rel L2 "
            + String(rel_l2)
            + " > "
            + String(L2_MAX_REL)
        )
    if max_zero_leak > ZERO_COMPONENT_MAX:
        raise Error(
            String("bench_maxwell_plane_wave_3d_p5 FAILED: zero-component ")
            + "leakage "
            + String(max_zero_leak)
            + " > "
            + String(ZERO_COMPONENT_MAX)
        )

    print("=== bench_maxwell_plane_wave_3d_p5 PASSED ===")
    mpi.finalize()
