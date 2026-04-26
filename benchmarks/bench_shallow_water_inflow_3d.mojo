# ======================================================================
# bench_shallow_water_inflow_3d -- 3D Shallow Water BC_INFLOW gate
# ======================================================================
#
# 3D analog of bench_shallow_water_inflow_2d.  Closes the 3D gap:
# `ShallowWater.boundary_flux` accepts (inflow_h, inflow_hu,
# inflow_hv) but no 3D bench passes non-zero values, so the
# BC_INFLOW arm of the 3D ShallowWater path is untested.
#
# Cleanest test: uniform subcritical state matched to the inflow
# ghost.  IC matches BC_INFLOW exactly so the analytic solution is
# the IC unchanged.  Any drift signals a BC bug.
#
# Setup: h0 = 2, u0 = 0.5 (Fr = 0.35 < 1, subcritical so BC_OUTFLOW
# at +x is well-posed), periodic in y/z (the 3D SW physics is the
# 2D system embedded in 3D with F^z = 0).
#
# Pass criteria (P=2, NX=16, NY=NZ=2, T=1):
#   * max |h - h0| / h0           < 2e-3 (Float32 step accumulation)
#   * max |hu - h0*u0| / h0*u0    < 2e-3
#   * max |hv|                    < 1e-3
#   * no NaN / Inf
# ======================================================================

from std.sys import has_accelerator
from std.gpu import global_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from std.math import sqrt, ceildiv, isnan, isinf

from src import mpi
from src.partition import build_partition
from src.reference import N_P, build_reference_operators
from src.mesh import Mesh
from src.boundary import (
    BoundaryConditions, BC_INTERIOR, BC_INFLOW, BC_OUTFLOW,
)
from src.halo_exchange import HaloExchange
from src.solver import Solver
from src.shallow_water import ShallowWater
from src.nvtx import NvtxContext


comptime NX = 16
comptime NY = 2
comptime NZ = 2
comptime LX = 1.0
comptime LY = Float64(NY) / Float64(NX) * LX
comptime LZ = Float64(NZ) / Float64(NX) * LX

comptime GRAVITY: Float32 = 1.0
comptime H0:      Float32 = 2.0
comptime U0:      Float32 = 0.5
comptime H_MIN:   Float32 = 1.0e-6
comptime CFL = Float32(0.2)
comptime IC_BLOCK = 256
comptime T_FINAL: Float32 = 1.0

comptime H_REL_TOL:  Float64 = 2.0e-3
comptime HU_REL_TOL: Float64 = 2.0e-3
comptime HV_TOL:     Float64 = 1.0e-3


def uniform_ic_kernel(
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
    var base = (e * N_P + nn) * 3
    q[base + 0] = H0
    q[base + 1] = H0 * U0
    q[base + 2] = Float32(0.0)


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    mpi.init()
    var size = mpi.world_size()
    if size > 1:
        mpi.finalize()
        print("bench_shallow_water_inflow_3d: runs at np=1 only")
        return

    print("bench_shallow_water_inflow_3d (3D SW BC_INFLOW preservation)")
    print("  P= 2   mesh=", NX, "x", NY, "x", NZ,
          "   H0=", H0, "   U0=", U0, "   T=", T_FINAL)

    var rank = mpi.world_rank()
    var nvtx = NvtxContext()
    var refs = build_reference_operators(nvtx)
    var ctx = DeviceContext()

    var bcs = BoundaryConditions(
        BC_INFLOW, BC_OUTFLOW,         # -x, +x
        BC_INTERIOR, BC_INTERIOR,      # -y, +y
        BC_INTERIOR, BC_INTERIOR,      # -z, +z
    )
    var mesh = Mesh(
        ctx, build_partition(rank, size, NX, NY, NZ), LX, LY, LZ, bcs,
    )
    var halo = HaloExchange(
        ctx, mesh.part, ShallowWater.NUM_COMPONENTS,
        mesh.d_perm.unsafe_ptr(), bcs,
    )
    # Inflow ghost matches the IC exactly.
    var physics = ShallowWater(
        GRAVITY, H_MIN,
        H0, H0 * U0, Float32(0.0),
    )
    var solver = Solver[ShallowWater](
        ctx^, mesh^, halo^, physics^,
        refs.D_ref^, refs.Lift_ref^, refs.node_weights^,
    )

    solver.ctx.enqueue_function[uniform_ic_kernel, uniform_ic_kernel](
        solver.d_q.unsafe_ptr(),
        solver.mesh.d_owned_elem_ids.unsafe_ptr(),
        solver.mesh.local.d_elem_node_xyz.unsafe_ptr(),
        solver.num_owned_elements,
        grid_dim=ceildiv(solver.num_owned_elements * N_P, IC_BLOCK),
        block_dim=IC_BLOCK,
    )
    solver.ctx.synchronize()

    var n_owned_dof = solver.num_owned_elements * N_P * ShallowWater.NUM_COMPONENTS

    var c = sqrt(GRAVITY * H0)
    var wave = c + U0
    var h_cell = Float32(LX) / Float32(NX)
    var dt_est = CFL * h_cell / (wave * Float32(5.0))
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

    var max_h_dev: Float64 = 0.0
    var max_hu_dev: Float64 = 0.0
    var max_hv: Float64 = 0.0
    var n_owned_nodes = solver.num_owned_elements * N_P
    for i in range(n_owned_nodes):
        var h = q_ptr[i * 3 + 0]
        var hu = q_ptr[i * 3 + 1]
        var hv = q_ptr[i * 3 + 2]
        if (isnan(h) or isinf(h) or isnan(hu) or isinf(hu)
            or isnan(hv) or isinf(hv)):
            raise Error("bench_shallow_water_inflow_3d: non-finite output")
        var dh = Float64(h - H0)
        if dh < 0.0: dh = -dh
        if dh > max_h_dev: max_h_dev = dh
        var dhu = Float64(hu - H0 * U0)
        if dhu < 0.0: dhu = -dhu
        if dhu > max_hu_dev: max_hu_dev = dhu
        var ahv = Float64(hv)
        if ahv < 0.0: ahv = -ahv
        if ahv > max_hv: max_hv = ahv

    var h_rel = max_h_dev / Float64(H0)
    var hu_rel = max_hu_dev / Float64(H0 * U0)
    print("  max |h - H0| / H0        =", h_rel,  "  (threshold", H_REL_TOL,  ")")
    print("  max |hu - H0*U0| / H0*U0 =", hu_rel, "  (threshold", HU_REL_TOL, ")")
    print("  max |hv|                 =", max_hv, "  (threshold", HV_TOL,     ")")

    if h_rel > H_REL_TOL:
        raise Error(
            "bench_shallow_water_inflow_3d FAILED: h rel err "
            + String(h_rel) + " > " + String(H_REL_TOL)
        )
    if hu_rel > HU_REL_TOL:
        raise Error(
            "bench_shallow_water_inflow_3d FAILED: hu rel err "
            + String(hu_rel) + " > " + String(HU_REL_TOL)
        )
    if max_hv > HV_TOL:
        raise Error(
            "bench_shallow_water_inflow_3d FAILED: hv drift "
            + String(max_hv) + " > " + String(HV_TOL)
        )

    print("=== bench_shallow_water_inflow_3d PASSED ===")
    mpi.finalize()
