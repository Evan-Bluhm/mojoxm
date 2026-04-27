# ======================================================================
# bench_shallow_water_dam_break_3d -- 3D closed-pool dam break
# ======================================================================
#
# 3D analog of bench_shallow_water_dam_break_2d.  Same dam-break IC
# (h_L=2 on left half, h_R=1 on right half, zero velocity), same
# closed-pool wall boundary on all 6 faces, same conservation +
# positivity invariants -- but routed through Mesh[2] /
# Solver[ShallowWater, 2] / NP=10 with the 3D BJ limiter enabled
# via `solver.enable_cell_limiter`.
#
# Closes a real gap: 3D SW is currently only validated on smooth
# flow (`bench_shallow_water_wave_3d{,_p3,_p4}`) and uniform-state
# preservation under BC_INFLOW (`bench_shallow_water_inflow_3d`).
# This is the first 3D SW bench that tests truly shocked flow -- it
# exercises the 3D Rusanov flux + reflective walls + BJ limiter
# pipeline on a discontinuous IC.
#
# Pass criteria (P=2, NX=40 NY=NZ=4, T=0.5, BJ + Venkat eps=0.1):
#   * Mass drift < 1e-3 over the run (slightly looser than 2D's
#     1e-4 since 3D conservation suffers ~5x more Float32-summation
#     ops at the larger DOF count, but still tight enough to catch
#     any meaningful regression)
#   * h_min > 0 (positivity preserved by BJ limiter)
#   * h_max < H_L * 1.10 (no spurious overshoot above 10%)
#   * |v|_max < 2 * sqrt(g * h_L) (Riemann waves bounded)
#   * no NaN / Inf
# ======================================================================

from std.sys import has_accelerator
from std.gpu import global_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from std.math import sqrt, ceildiv, isnan, isinf

from src import mpi
from src.partition import build_partition
from src.reference import (
    ReferenceElement, to_float32, num_tet_nodes,
)
from src.mesh import Mesh
from src.boundary import BoundaryConditions, BC_WALL
from src.halo_exchange import HaloExchange
from src.solver import Solver
from src.shallow_water import ShallowWater
from src.nvtx import NvtxContext


comptime P = 2
comptime NP = num_tet_nodes(P)   # 10 at P=2
comptime NC = 3                  # ShallowWater.NUM_COMPONENTS

comptime NX = 40
comptime NY = 4
comptime NZ = 4
comptime LX = 2.0
comptime LY = Float64(4.0 / 40.0) * 2.0
comptime LZ = Float64(4.0 / 40.0) * 2.0
comptime G:   Float32 = 9.81
comptime H_L: Float32 = 2.0
comptime H_R: Float32 = 1.0
comptime H_MIN: Float32 = 1.0e-6

comptime T_FINAL: Float32 = 0.5
comptime CFL = Float32(0.10)
comptime IC_BLOCK = 256

comptime MASS_TOL_REL: Float64 = 1.0e-3
comptime H_MAX_OK: Float32 = Float32(H_L) * Float32(1.10)


def dam_break_ic_kernel(
    q: UnsafePointer[Float32, MutAnyOrigin],
    owned_elem_ids: UnsafePointer[Int32, MutAnyOrigin],
    elem_node_xyz:  UnsafePointer[Float32, MutAnyOrigin],
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
    var h = H_L if px < Float32(LX) * Float32(0.5) else H_R
    var base = (e * NP + nn) * NC
    q[base + 0] = h
    q[base + 1] = Float32(0.0)    # h*u
    q[base + 2] = Float32(0.0)    # h*v


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    mpi.init()
    var size = mpi.world_size()
    if size > 1:
        mpi.finalize()
        print("bench_shallow_water_dam_break_3d: runs at np=1 only")
        return

    print("bench_shallow_water_dam_break_3d (3D dam break, closed pool)")
    print("  P=", P, "  NP=", NP, "  mesh=", NX, "x", NY, "x", NZ,
          "  H_L=", H_L, "  H_R=", H_R, "  T=", T_FINAL)

    var rank = mpi.world_rank()
    var nvtx = NvtxContext()
    var ctx = DeviceContext()

    var re = ReferenceElement[P]()
    var D_ref = to_float32(re.D_ref)
    var Lift_ref = to_float32(re.Lift_ref)
    var node_weights = to_float32(re.node_weights)

    # Closed pool: BC_WALL on all 6 faces.
    var bcs = BoundaryConditions(
        BC_WALL, BC_WALL,
        BC_WALL, BC_WALL,
        BC_WALL, BC_WALL,
    )
    var mesh = Mesh[P](
        ctx, build_partition(rank, size, NX, NY, NZ), LX, LY, LZ, bcs,
    )
    var halo = HaloExchange(
        ctx, mesh.part, ShallowWater.NUM_COMPONENTS,
        mesh.d_perm.unsafe_ptr(), bcs,
    )
    var physics = ShallowWater(G, H_MIN)
    var solver = Solver[ShallowWater, P](
        ctx^, mesh^, halo^, physics^,
        D_ref^, Lift_ref^, node_weights^,
    )
    solver.enable_cell_limiter(True, Float32(0.1))

    solver.ctx.enqueue_function[dam_break_ic_kernel, dam_break_ic_kernel](
        solver.d_q.unsafe_ptr(),
        solver.mesh.d_owned_elem_ids.unsafe_ptr(),
        solver.mesh.local.d_elem_node_xyz.unsafe_ptr(),
        solver.num_owned_elements,
        grid_dim=ceildiv(solver.num_owned_elements * NP, IC_BLOCK),
        block_dim=IC_BLOCK,
    )
    solver.ctx.synchronize()

    var n_owned = solver.num_owned_elements
    var n_dof = n_owned * NP
    var h_buf = List[Float32]()
    for _ in range(n_dof):
        h_buf.append(Float32(0.0))
    solver.download_owned_component(0, h_buf, nvtx)
    # Mass-matrix-weighted IC mass.  All cells have the same volume on
    # uniform Cartesian mesh, so factoring it out leaves a scaled
    # invariant; the relative drift below is unaffected.
    var mass_ic: Float64 = 0.0
    for elem in range(n_owned):
        for nn in range(NP):
            mass_ic += Float64(h_buf[elem * NP + nn]) * Float64(re.node_weights[nn])

    var c_peak = sqrt(G * H_L)
    var h_cell = Float32(LX) / Float32(NX)
    var dt_est = CFL * h_cell / (c_peak * Float32(2 * P + 1))
    var num_steps = Int(T_FINAL / dt_est) + 1
    var dt = T_FINAL / Float32(num_steps)
    print("  steps=", num_steps, "  dt=", dt)

    for _ in range(num_steps):
        solver.step_ssprk3(dt, nvtx)
    solver.ctx.synchronize()

    # h bounds + |v|_max + mass drift.
    solver.download_owned_component(0, h_buf, nvtx)
    var h_max: Float32 = Float32(-1.0e30)
    var h_min: Float32 = Float32( 1.0e30)
    for i in range(n_dof):
        var h = h_buf[i]
        if isnan(h) or isinf(h):
            raise Error("bench_shallow_water_dam_break_3d: non-finite h")
        if h > h_max: h_max = h
        if h < h_min: h_min = h
    var mass_fin: Float64 = 0.0
    for elem in range(n_owned):
        for nn in range(NP):
            mass_fin += Float64(h_buf[elem * NP + nn]) * Float64(re.node_weights[nn])

    var hu_buf = List[Float32]()
    for _ in range(n_dof):
        hu_buf.append(Float32(0.0))
    solver.download_owned_component(1, hu_buf, nvtx)
    var hv_buf = List[Float32]()
    for _ in range(n_dof):
        hv_buf.append(Float32(0.0))
    solver.download_owned_component(2, hv_buf, nvtx)
    var v_max: Float32 = 0.0
    for i in range(n_dof):
        var h_safe = h_buf[i] if h_buf[i] > Float32(H_MIN) else Float32(H_MIN)
        var u = hu_buf[i] / h_safe
        var v = hv_buf[i] / h_safe
        var vm = sqrt(u * u + v * v)
        if vm > v_max: v_max = vm

    var dmass = mass_fin - mass_ic
    if dmass < 0.0: dmass = -dmass
    var rel = dmass / mass_ic

    print("  h in [", h_min, ",", h_max, "]",
          "  |v|_max=", v_max,
          "  mass(IC)=", mass_ic, "  mass(t=T)=", mass_fin,
          "  rel=", rel)

    if rel > MASS_TOL_REL:
        raise Error(
            String("bench_shallow_water_dam_break_3d FAILED: mass drift ")
            + String(rel) + " > " + String(MASS_TOL_REL)
        )
    if h_min < Float32(0.0):
        raise Error(
            String("bench_shallow_water_dam_break_3d FAILED: h_min ")
            + String(h_min) + " negative (positivity lost)"
        )
    if h_max > H_MAX_OK:
        raise Error(
            String("bench_shallow_water_dam_break_3d FAILED: h_max ")
            + String(h_max) + " > " + String(H_MAX_OK)
        )
    var v_bound = Float32(2.0) * Float32(sqrt(G * H_L))
    if v_max > v_bound:
        raise Error(
            String("bench_shallow_water_dam_break_3d FAILED: |v|_max ")
            + String(v_max) + " > " + String(v_bound)
        )

    print("=== bench_shallow_water_dam_break_3d PASSED ===")
    mpi.finalize()
