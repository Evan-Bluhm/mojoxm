# ======================================================================
# bench_advection_outflow_3d -- 3D BC_OUTFLOW drainage gate
# ======================================================================
#
# 3D analog of bench_advection_outflow_2d.  Gaussian bump centered at
# (0.3, 0.3, 0.3) with velocity v = (1, 1, 1) and BC_OUTFLOW on all
# six sides of the [0, 1]^3 cube.  By T = 1.2 the bump has translated
# to (1.5, 1.5, 1.5) -- well outside the domain, so the analytic mass
# is zero.
#
# Pass criteria (P=2, NX=NY=NZ=24, T=1.2, single-rank):
#   * residual mass at t = T < 1e-3 of IC mass (mass-matrix-weighted
#     nodal quadrature)
#   * no NaN / Inf
#
# This benchmark also serves as the regression gate for the BC_OUTFLOW
# fix in src/advection.mojo: before that fix, the boundary_flux used a
# zero-gradient ghost (q_ghost = q_int) which gave spurious inflow on
# boundaries where v.n < 0 and grew the solution past the IC value.
# Pre-fix observation: mass at T=1.2 was 42x of IC instead of ~1e-3.
# ======================================================================

from std.sys import has_accelerator
from std.gpu import global_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from std.math import sqrt, ceildiv, exp, isnan, isinf

from src import mpi
from src.partition import build_partition
from src.reference import (
    ReferenceElement,
    to_float32,
    num_tet_nodes,
    build_reference_operators,
    N_P,
)
from src.mesh import Mesh
from src.boundary import BoundaryConditions, BC_OUTFLOW
from src.halo_exchange import HaloExchange
from src.solver import Solver
from src.advection import Advection
from src.nvtx import NvtxContext


comptime LX = 1.0
comptime LY = 1.0
comptime LZ = 1.0
comptime VX: Float32 = 1.0
comptime VY: Float32 = 1.0
comptime VZ: Float32 = 1.0
comptime T_FINAL: Float32 = 1.2
comptime CFL = Float32(0.2)
comptime GAUSS_SIGMA: Float32 = 0.10
comptime CX: Float32 = 0.3
comptime CY: Float32 = 0.3
comptime CZ: Float32 = 0.3
comptime NX = 24
comptime NY = 24
comptime NZ = 24
comptime IC_BLOCK = 256

# Measured residual mass ~1.4e-12 of IC (essentially noise) on
# current code.  1e-9 leaves 3 orders of headroom and would catch
# any regression that leaves more than O(1e-9) mass behind.
comptime DRAIN_TOL_REL: Float64 = 1.0e-9


def gaussian_ic_kernel(
    q: UnsafePointer[Float32, MutAnyOrigin],
    owned_elem_ids: UnsafePointer[Int32, MutAnyOrigin],
    elem_node_xyz: UnsafePointer[Float32, MutAnyOrigin],
    num_owned: Int,
    inv_two_sigma2: Float32,
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
    var dx = px - CX
    var dy = py - CY
    var dz = pz - CZ
    q[e * N_P + nn] = exp(-(dx * dx + dy * dy + dz * dz) * inv_two_sigma2)


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    mpi.init()
    var size = mpi.world_size()
    if size > 1:
        mpi.finalize()
        print("bench_advection_outflow_3d: runs at np=1 only")
        return

    print("bench_advection_outflow_3d (3D BC_OUTFLOW x 6, drainage)")
    print("  P=2  mesh=", NX, "x", NY, "x", NZ, "  T=", T_FINAL)

    var rank = mpi.world_rank()
    var nvtx = NvtxContext()
    var refs = build_reference_operators(nvtx)
    var ctx = DeviceContext()

    var bcs = BoundaryConditions(
        BC_OUTFLOW,
        BC_OUTFLOW,
        BC_OUTFLOW,
        BC_OUTFLOW,
        BC_OUTFLOW,
        BC_OUTFLOW,
    )
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
        Advection.NUM_COMPONENTS,
        mesh.d_perm.unsafe_ptr(),
        bcs,
    )
    var physics = Advection(VX, VY, VZ)
    var solver = Solver[Advection](
        ctx^,
        mesh^,
        halo^,
        physics^,
        refs.D_ref^,
        refs.Lift_ref^,
        refs.node_weights^,
    )

    var inv_two_sigma2 = Float32(1.0) / (
        Float32(2.0) * GAUSS_SIGMA * GAUSS_SIGMA
    )
    solver.ctx.enqueue_function[gaussian_ic_kernel, gaussian_ic_kernel](
        solver.d_q.unsafe_ptr(),
        solver.mesh.d_owned_elem_ids.unsafe_ptr(),
        solver.mesh.local.d_elem_node_xyz.unsafe_ptr(),
        solver.num_owned_elements,
        inv_two_sigma2,
        grid_dim=ceildiv(solver.num_owned_elements * N_P, IC_BLOCK),
        block_dim=IC_BLOCK,
    )
    solver.ctx.synchronize()

    var n_owned = solver.num_owned_elements
    var n_dof = n_owned * N_P
    var rho_ic = List[Float32]()
    for _ in range(n_dof):
        rho_ic.append(Float32(0.0))
    solver.download_owned_component(0, rho_ic, nvtx)

    # Mass-matrix-weighted nodal quadrature.
    var re_host = ReferenceElement[2]()
    var node_w = List[Float32]()
    for k in range(N_P):
        node_w.append(Float32(re_host.node_weights[k]))

    var mass_ic: Float64 = 0.0
    for i in range(n_owned):
        for nn in range(N_P):
            mass_ic += Float64(rho_ic[i * N_P + nn]) * Float64(node_w[nn])

    var v = sqrt(VX * VX + VY * VY + VZ * VZ)
    var h = Float32(LX) / Float32(NX)
    var dt_est = CFL * h / (v * Float32(2 * 2 + 1))
    var num_steps = Int(T_FINAL / dt_est) + 1
    var dt = T_FINAL / Float32(num_steps)

    for _ in range(num_steps):
        solver.step_ssprk3(dt, nvtx)
    solver.ctx.synchronize()

    solver.download_owned_component(0, rho_ic, nvtx)
    var mass_fin: Float64 = 0.0
    var max_abs: Float32 = 0.0
    for i in range(n_owned):
        for nn in range(N_P):
            var v_now = rho_ic[i * N_P + nn]
            if isnan(v_now) or isinf(v_now):
                raise Error("bench_advection_outflow_3d: non-finite output")
            mass_fin += Float64(v_now) * Float64(node_w[nn])
            var av = v_now if v_now >= Float32(0.0) else -v_now
            if av > max_abs:
                max_abs = av

    var rel = mass_fin / mass_ic
    if rel < 0.0:
        rel = -rel
    print(
        "  mass(IC)=",
        mass_ic,
        "  mass(t=T)=",
        mass_fin,
        "  rel=",
        rel,
        "  max |q|=",
        max_abs,
    )
    if rel > DRAIN_TOL_REL:
        raise Error(
            String("bench_advection_outflow_3d FAILED: residual mass ")
            + String(rel)
            + " > "
            + String(DRAIN_TOL_REL)
        )

    print("=== bench_advection_outflow_3d PASSED ===")
    mpi.finalize()
