# ======================================================================
# bench_mhd_glm_psi_transport_3d_p3 -- 3D GLM linear psi/Bx wave at P=3
# ======================================================================
#
# P=3 (NP=20 nodes per tet) counterpart of
# bench_mhd_glm_psi_transport_3d.  Same psi=A*sin(2pi x/Lx) IC on a
# rest state, same one-period (T = Lx / c_h) standing-wave horizon,
# same threshold structure, but routed through Mesh[3] /
# Solver[IdealMHD, 3] / rk_stage_kernel[3] so the GLM transport
# coupling in src/mhd.mojo's `internal_flux` is exercised at NP=20.
#
# Closes the P=3 GLM rate-gate gap: the existing 2D path has a P=3
# variant (bench_mhd_glm_psi_transport_2d_p3, NP=10), but in 3D the
# only P=3 MHD bench is bench_mhd_alfven_3d_p3 which exercises the
# wave but not the GLM transport coupling directly.  This bench
# pairs with that one for full 3D MHD P=3 coverage.
#
# Pass criteria (P=3, NX=16, NY=NZ=4, c_h=1, T=1):
#   * rel L2(state) < 5e-4 (Float32 floor at NP=20; same threshold
#     as the P=2 analog -- at higher P the spatial discretization
#     is more accurate, so error is dominated by Float32 round-off
#     accumulated over the SSPRK3 steps, comparable to P=2.)
#   * |psi| <= 1.1 * A throughout (no spurious amplification)
#   * no NaN / Inf
# ======================================================================

from std.sys import has_accelerator
from std.gpu import global_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from std.math import sqrt, ceildiv, sin, pi, isnan, isinf

from src import mpi
from src.partition import build_partition
from src.reference import ReferenceElement, to_float32, num_tet_nodes
from src.mesh import Mesh
from src.boundary import BoundaryConditions
from src.halo_exchange import HaloExchange
from src.solver import Solver
from src.mhd import IdealMHD
from src.nvtx import NvtxContext


comptime P = 3
comptime NP = num_tet_nodes(P)  # 20 at P=3
comptime NX = 16
comptime NY = 4
comptime NZ = 4
comptime LX = 1.0
comptime LY = Float64(NY) / Float64(NX) * LX
comptime LZ = Float64(NZ) / Float64(NX) * LX

comptime GAMMA = Float32(5.0 / 3.0)
comptime RHO0 = Float32(1.0)
comptime P0_GAS = Float32(1.0)
comptime AMPLITUDE = Float32(0.01)
comptime C_H = Float32(1.0)
comptime ALPHA_D = Float32(0.0)
comptime MIN_DENSITY = Float32(1.0e-6)
comptime MIN_PRESSURE = Float32(1.0e-6)
comptime T_FINAL: Float32 = 1.0
comptime CFL = Float32(0.10)
comptime IC_BLOCK = 256
comptime PI_F = Float32(3.14159265358979323846)

comptime L2_MAX_REL: Float64 = 5.0e-4


def transport_ic_kernel_p3(
    q: UnsafePointer[Float32, MutAnyOrigin],
    owned_elem_ids: UnsafePointer[Int32, MutAnyOrigin],
    elem_node_xyz: UnsafePointer[Float32, MutAnyOrigin],
    num_owned: Int,
):
    var idx = Int(global_idx.x)
    var total = num_owned * NP
    if idx >= total:
        return
    var i = idx // NP
    var nn = idx % NP
    var e = Int(owned_elem_ids[i])
    var px = elem_node_xyz[(e * NP + nn) * 3 + 0]

    var k_wave = Float32(2.0) * PI_F / Float32(LX)
    var psi = AMPLITUDE * sin(k_wave * px)

    var E = P0_GAS / (GAMMA - Float32(1.0))

    var base = (e * NP + nn) * 9
    q[base + 0] = RHO0
    q[base + 1] = Float32(0.0)
    q[base + 2] = Float32(0.0)
    q[base + 3] = Float32(0.0)
    q[base + 4] = E
    q[base + 5] = Float32(0.0)
    q[base + 6] = Float32(0.0)
    q[base + 7] = Float32(0.0)
    q[base + 8] = psi


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    mpi.init()
    var size = mpi.world_size()
    if size > 1:
        mpi.finalize()
        print("bench_mhd_glm_psi_transport_3d_p3: runs at np=1 only")
        return

    print(
        "bench_mhd_glm_psi_transport_3d_p3 (3D GLM linear psi/Bx wave at P=3)"
    )
    print(
        "  P=",
        P,
        "  NP=",
        NP,
        "  mesh=",
        NX,
        "x",
        NY,
        "x",
        NZ,
        "   c_h=",
        C_H,
        "   T=",
        T_FINAL,
        "   amplitude=",
        AMPLITUDE,
    )

    var rank = mpi.world_rank()
    var nvtx = NvtxContext()
    var ctx = DeviceContext()

    # Build P=3 reference operators directly: build_reference_operators()
    # in src.reference defaults to P=2 / NP=10, wrong for Solver[IdealMHD, 3].
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
        IdealMHD.NUM_COMPONENTS,
        mesh.d_perm.unsafe_ptr(),
        bcs,
    )
    var physics = IdealMHD(
        GAMMA,
        MIN_DENSITY,
        MIN_PRESSURE,
        C_H,
        ALPHA_D,
    )
    var solver = Solver[IdealMHD, P](
        ctx^,
        mesh^,
        halo^,
        physics^,
        D_ref^,
        Lift_ref^,
        node_weights^,
    )

    solver.ctx.enqueue_function[transport_ic_kernel_p3, transport_ic_kernel_p3](
        solver.d_q.unsafe_ptr(),
        solver.mesh.d_owned_elem_ids.unsafe_ptr(),
        solver.mesh.local.d_elem_node_xyz.unsafe_ptr(),
        solver.num_owned_elements,
        grid_dim=ceildiv(solver.num_owned_elements * NP, IC_BLOCK),
        block_dim=IC_BLOCK,
    )
    solver.ctx.synchronize()

    var n_owned_dof = solver.num_owned_elements * NP * IdealMHD.NUM_COMPONENTS
    var hbuf_ic = solver.ctx.enqueue_create_host_buffer[DType.float32](
        n_owned_dof
    )
    solver.ctx.enqueue_copy(
        hbuf_ic, solver.d_q.create_sub_buffer[DType.float32](0, n_owned_dof)
    )
    solver.ctx.synchronize()
    var ic_ptr = hbuf_ic.unsafe_ptr()
    var host_ic = List[Float32]()
    for k in range(n_owned_dof):
        host_ic.append(ic_ptr[k])

    # CFL: tighter at higher P (factor 2P+1 = 7 vs 5).
    var h = Float32(LX) / Float32(NX)
    var dt_est = CFL * h / (C_H * Float32(2 * P + 1))
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
        hbuf_q, solver.d_q.create_sub_buffer[DType.float32](0, n_owned_dof)
    )
    solver.ctx.synchronize()
    var q_ptr = hbuf_q.unsafe_ptr()

    var sum_sq: Float64 = 0.0
    var sum_ic: Float64 = 0.0
    var psi_max: Float32 = 0.0
    for k in range(n_owned_dof):
        var v = q_ptr[k]
        if isnan(v) or isinf(v):
            raise Error("bench_mhd_glm_psi_transport_3d_p3: non-finite output")
        var err = Float64(v - host_ic[k])
        sum_sq += err * err
        var ic = Float64(host_ic[k])
        sum_ic += ic * ic
    var n_owned_nodes = solver.num_owned_elements * NP
    for i in range(n_owned_nodes):
        var p = q_ptr[i * 9 + 8]
        var a = p
        if a < Float32(0.0):
            a = -a
        if a > psi_max:
            psi_max = a

    var amp_bound = AMPLITUDE * Float32(1.1)
    if psi_max > amp_bound:
        raise Error(
            String("bench_mhd_glm_psi_transport_3d_p3 FAILED: psi_max ")
            + String(psi_max)
            + " exceeds 1.1 * AMPLITUDE "
            + String(amp_bound)
        )

    var l2 = sqrt(sum_sq / Float64(n_owned_dof))
    var l2_ic = sqrt(sum_ic / Float64(n_owned_dof))
    var rel_l2 = l2 / l2_ic
    print("  rel L2(state) =", rel_l2, "  (threshold", L2_MAX_REL, ")")

    if rel_l2 > L2_MAX_REL:
        raise Error(
            "bench_mhd_glm_psi_transport_3d_p3 FAILED: rel L2 "
            + String(rel_l2)
            + " > "
            + String(L2_MAX_REL)
        )

    print("=== bench_mhd_glm_psi_transport_3d_p3 PASSED ===")
    mpi.finalize()
