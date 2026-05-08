# ======================================================================
# bench_mhd_glm_psi_transport_3d -- 3D GLM linear psi/Bx wave
# ======================================================================
#
# 3D analog of bench_mhd_glm_psi_transport_2d.  In 3D, the GLM
# transport coupling lives inside src/mhd.mojo's `internal_flux`
# (F_psi^d = c_h^2 * B_d, F_B^d_d = psi) -- not a separate kernel
# like the 2D path's operator-split psi-damp.  bench_mhd_alfven_3d
# / _p3 set c_h > 0 but don't directly measure the transport rate.
#
# Setup: rest state with rho=1, p=1, B=0; psi = A*sin(2 pi x / Lx)
# IC, all other components at rest.  Linearised GLM on a rest
# state (no Lorentz back-reaction since u=0, B0=0):
#   d psi/dt + c_h^2 d Bx/dx = 0
#   d Bx/dt + d psi/dx        = 0
# Wave equation in (Bx, psi) with characteristic speeds +-c_h.
# After T = Lx / c_h the standing-wave superposition returns to IC.
#
# Pass criteria (P=2, NX=24, NY=NZ=4, c_h=1, T=1):
#   * rel L2(state) < 5e-4
#   * |psi| <= 1.1 * A throughout (no spurious amplification)
#   * no NaN / Inf
# ======================================================================

from std.sys import has_accelerator
from std.gpu import global_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from std.math import sqrt, ceildiv, sin, pi, isnan, isinf

from src import mpi
from src.partition import build_partition
from src.reference import N_P, build_reference_operators
from src.mesh import Mesh
from src.boundary import BoundaryConditions
from src.halo_exchange import HaloExchange
from src.solver import Solver
from src.mhd import IdealMHD
from src.nvtx import NvtxContext


comptime NX = 24
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
comptime ALPHA_D = Float32(0.0)  # transport-only; no damping
comptime MIN_DENSITY = Float32(1.0e-6)
comptime MIN_PRESSURE = Float32(1.0e-6)
comptime T_FINAL: Float32 = 1.0  # one period: T = LX / c_h
comptime CFL = Float32(0.10)
comptime IC_BLOCK = 256
comptime PI_F = Float32(3.14159265358979323846)

comptime L2_MAX_REL: Float64 = 5.0e-4


def transport_ic_kernel(
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

    var k_wave = Float32(2.0) * PI_F / Float32(LX)
    var psi = AMPLITUDE * sin(k_wave * px)

    # Rest state: u=v=w=0, B=0.  E_total = p_gas/(gamma-1) + 0 + 0.
    var E = P0_GAS / (GAMMA - Float32(1.0))

    var base = (e * N_P + nn) * 9
    q[base + 0] = RHO0
    q[base + 1] = Float32(0.0)
    q[base + 2] = Float32(0.0)
    q[base + 3] = Float32(0.0)
    q[base + 4] = E
    q[base + 5] = Float32(0.0)  # Bx
    q[base + 6] = Float32(0.0)  # By
    q[base + 7] = Float32(0.0)  # Bz
    q[base + 8] = psi


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    mpi.init()
    var size = mpi.world_size()
    if size > 1:
        mpi.finalize()
        print("bench_mhd_glm_psi_transport_3d: runs at np=1 only")
        return

    print("bench_mhd_glm_psi_transport_3d (3D GLM linear psi/Bx wave)")
    print(
        "  P= 2   mesh=",
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
    var solver = Solver[IdealMHD](
        ctx^,
        mesh^,
        halo^,
        physics^,
        refs.D_ref^,
        refs.Lift_ref^,
        refs.node_weights^,
    )

    solver.ctx.enqueue_function[transport_ic_kernel, transport_ic_kernel](
        solver.d_q.unsafe_ptr(),
        solver.mesh.d_owned_elem_ids.unsafe_ptr(),
        solver.mesh.local.d_elem_node_xyz.unsafe_ptr(),
        solver.num_owned_elements,
        grid_dim=ceildiv(solver.num_owned_elements * N_P, IC_BLOCK),
        block_dim=IC_BLOCK,
    )
    solver.ctx.synchronize()

    var n_owned_dof = solver.num_owned_elements * N_P * IdealMHD.NUM_COMPONENTS
    var hbuf_ic = solver.ctx.enqueue_create_host_buffer[DType.float32](n_owned_dof)
    solver.ctx.enqueue_copy(hbuf_ic, solver.d_q.create_sub_buffer[DType.float32](0, n_owned_dof))
    solver.ctx.synchronize()
    var ic_ptr = hbuf_ic.unsafe_ptr()
    var host_ic = List[Float32]()
    for k in range(n_owned_dof):
        host_ic.append(ic_ptr[k])

    var h = Float32(LX) / Float32(NX)
    var dt_est = CFL * h / (C_H * Float32(5.0))
    var num_steps = Int(T_FINAL / dt_est) + 1
    var dt = T_FINAL / Float32(num_steps)
    print("  steps=", num_steps, "  dt=", dt)

    for _ in range(num_steps):
        solver.step_ssprk3(dt, nvtx)
    solver.ctx.synchronize()

    var hbuf_q = solver.ctx.enqueue_create_host_buffer[DType.float32](n_owned_dof)
    solver.ctx.enqueue_copy(hbuf_q, solver.d_q.create_sub_buffer[DType.float32](0, n_owned_dof))
    solver.ctx.synchronize()
    var q_ptr = hbuf_q.unsafe_ptr()

    var sum_sq: Float64 = 0.0
    var sum_ic: Float64 = 0.0
    var psi_max: Float32 = 0.0
    for k in range(n_owned_dof):
        var v = q_ptr[k]
        if isnan(v) or isinf(v):
            raise Error("bench_mhd_glm_psi_transport_3d: non-finite output")
        var err = Float64(v - host_ic[k])
        sum_sq += err * err
        var ic = Float64(host_ic[k])
        sum_ic += ic * ic
    var n_owned_nodes = solver.num_owned_elements * N_P
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
            String("bench_mhd_glm_psi_transport_3d FAILED: psi_max ")
            + String(psi_max)
            + " exceeds 1.1 * AMPLITUDE "
            + String(amp_bound)
        )

    var l2 = sqrt(sum_sq / Float64(n_owned_dof))
    var l2_ic = sqrt(sum_ic / Float64(n_owned_dof))
    var rel_l2 = l2 / l2_ic
    print("  rel L2(state) =", rel_l2, "  (threshold", L2_MAX_REL, ")")

    if rel_l2 > L2_MAX_REL:
        raise Error("bench_mhd_glm_psi_transport_3d FAILED: rel L2 " + String(rel_l2) + " > " + String(L2_MAX_REL))

    print("=== bench_mhd_glm_psi_transport_3d PASSED ===")
    mpi.finalize()
