# ======================================================================
# bench_shallow_water_inflow_2d -- BC_INFLOW preservation gate
# ======================================================================
#
# Closes a real coverage gap.  The 2D Shallow Water face-flux kernel
# (`sw_face_flux_kernel_2d` and `sw_face_flux_hll_kernel_2d` in
# src/local_mesh_2d_gpu_sw.mojo) accepts BC_INFLOW with prescribed
# (inflow_h, inflow_hu, inflow_hv) ghost state, but every existing
# SW bench/test uses periodic BCs and passes inflow_h=0.  A
# regression in the SW BC_INFLOW branch would not have tripped any
# gate.
#
# Cleanest test: uniform-state preservation under matched
# BC_INFLOW at the -x face and BC_OUTFLOW at the +x face.  At t=0
# the IC matches the inflow ghost exactly, so the analytic solution
# is the IC unchanged for all time.  Any drift signals a BC bug.
#
# Setup: h0 = 2.0, u0 = 0.5, v = 0  (Froude number 0.35 < 1, so
# subcritical -- BC_OUTFLOW is well-posed).
#
# Pass criteria (P=2, NX=32 NY=4, T=1, HLL flux):
#   * max |h - h0| < 1e-3
#   * max |hu - h0*u0| / |h0*u0| < 1e-3
#   * max |hv| < 1e-3
#   * no NaN / Inf
# ======================================================================

from std.math import sqrt, isnan, isinf
from std.sys import has_accelerator
from std.gpu.host import DeviceContext, DeviceBuffer
from src import mpi
from src.local_mesh_2d import LocalMesh2D
from src.local_mesh_2d_gpu import LocalMesh2DGpu
from src.local_mesh_2d_gpu_sw import sw_rk_stage_hll_2d
from src.reference_2d import (
    ReferenceElement2D, num_tri_nodes_2d, num_edge_nodes,
)
from src.reference_2d_gpu import ReferenceElement2DGpu
from src.boundary import (
    BoundaryConditions2D, BC_INFLOW, BC_OUTFLOW, BC_INTERIOR,
)


comptime P = 2
comptime NX = 32
comptime NY = 4
comptime LX = 1.0
comptime LY = Float64(NY) / Float64(NX) * LX

comptime GRAVITY: Float32 = 1.0
comptime H0:      Float32 = 2.0
comptime U0:      Float32 = 0.5    # Fr = U0/sqrt(g*H0) ~ 0.35 (subcritical)
comptime H_MIN:   Float32 = 1.0e-6
comptime CFL: Float64 = 0.15
comptime T_FINAL: Float64 = 1.0

# Empirical: state drift sits at Float32 floor on this uniform IC
# (~few * 1e-5).  1e-3 leaves ample margin for any meaningful BC-
# coupling regression while staying well above the noise floor.
comptime H_TOL: Float64 = 1.0e-3
comptime HU_REL_TOL: Float64 = 1.0e-3
comptime HV_TOL: Float64 = 1.0e-3


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    mpi.init()
    var size = mpi.world_size()
    if size > 1:
        mpi.finalize()
        print("bench_shallow_water_inflow_2d: runs at np=1 only")
        return

    print("bench_shallow_water_inflow_2d (BC_INFLOW preservation gate)")
    print("  P=", P, "  mesh=", NX, "x", NY,
          "   H0=", H0, "   U0=", U0, "   T=", T_FINAL)

    comptime NP_p = num_tri_nodes_2d(P)
    comptime NFP_e = num_edge_nodes(P)
    comptime NC = 3
    var ctx = DeviceContext()

    # Periodic in y (BC_INTERIOR), inflow on -x, outflow on +x.
    var bcs = BoundaryConditions2D(
        BC_INFLOW, BC_OUTFLOW,         # -x, +x
        BC_INTERIOR, BC_INTERIOR,      # -y, +y (periodic)
    )
    var host_mesh = LocalMesh2D[P](NX, NY, LX, LY, bcs)
    var mesh_coords = LocalMesh2D[P](NX, NY, LX, LY, bcs)
    var host_re = ReferenceElement2D[P]()
    var gpu_mesh = LocalMesh2DGpu[P](ctx, host_mesh^)
    var gpu_re = ReferenceElement2DGpu[P](ctx, host_re)

    # IC: uniform h = H0, hu = H0*U0, hv = 0.  Matches the inflow
    # ghost exactly so the analytic solution is the IC for all time.
    var n_q = gpu_mesh.num_elements * NP_p * NC
    var host_q = List[Float32]()
    for _ in range(gpu_mesh.num_elements * NP_p):
        host_q.append(H0)
        host_q.append(H0 * U0)
        host_q.append(Float32(0.0))

    var d_q  = ctx.enqueue_create_buffer[DType.float32](n_q)
    var d_q1 = ctx.enqueue_create_buffer[DType.float32](n_q)
    var d_q2 = ctx.enqueue_create_buffer[DType.float32](n_q)
    var d_fstar = ctx.enqueue_create_buffer[DType.float32](
        gpu_mesh.num_faces * NFP_e * NC
    )
    var hbuf_q = ctx.enqueue_create_host_buffer[DType.float32](n_q)
    var hptr_q = hbuf_q.unsafe_ptr()
    for k in range(n_q):
        hptr_q[k] = host_q[k]
    ctx.enqueue_copy(d_q, hbuf_q)
    ctx.synchronize()

    # CFL on c + |u|.
    var c = sqrt(GRAVITY * H0)
    var wave = c + U0
    var h_cell = LX / Float64(NX)
    var dt_est = CFL * Float64(h_cell) / (Float64(wave) * Float64(2 * P + 1))
    var num_steps = Int(T_FINAL / dt_est) + 1
    var dt = Float32(T_FINAL / Float64(num_steps))
    print("  steps=", num_steps, "  dt=", dt)

    # Inflow ghost state passed to every RK stage.
    var inflow_h = H0
    var inflow_hu = H0 * U0
    var inflow_hv = Float32(0.0)

    for _ in range(num_steps):
        sw_rk_stage_hll_2d[P](
            ctx, gpu_mesh,
            gpu_re.d_Lift_ref.unsafe_ptr(), gpu_re.d_D_ref.unsafe_ptr(),
            d_q.unsafe_ptr(),
            d_q.unsafe_ptr(), d_q.unsafe_ptr(),
            d_q1.unsafe_ptr(),
            d_fstar.unsafe_ptr(),
            GRAVITY, H_MIN,
            Float32(1.0), Float32(0.0), Float32(1.0), dt,
            inflow_h=inflow_h, inflow_hu=inflow_hu, inflow_hv=inflow_hv,
        )
        sw_rk_stage_hll_2d[P](
            ctx, gpu_mesh,
            gpu_re.d_Lift_ref.unsafe_ptr(), gpu_re.d_D_ref.unsafe_ptr(),
            d_q1.unsafe_ptr(),
            d_q.unsafe_ptr(), d_q1.unsafe_ptr(),
            d_q2.unsafe_ptr(),
            d_fstar.unsafe_ptr(),
            GRAVITY, H_MIN,
            Float32(0.75), Float32(0.25), Float32(0.25), dt,
            inflow_h=inflow_h, inflow_hu=inflow_hu, inflow_hv=inflow_hv,
        )
        sw_rk_stage_hll_2d[P](
            ctx, gpu_mesh,
            gpu_re.d_Lift_ref.unsafe_ptr(), gpu_re.d_D_ref.unsafe_ptr(),
            d_q2.unsafe_ptr(),
            d_q.unsafe_ptr(), d_q2.unsafe_ptr(),
            d_q.unsafe_ptr(),
            d_fstar.unsafe_ptr(),
            GRAVITY, H_MIN,
            Float32(1.0 / 3.0), Float32(2.0 / 3.0),
            Float32(2.0 / 3.0), dt,
            inflow_h=inflow_h, inflow_hu=inflow_hu, inflow_hv=inflow_hv,
        )
    ctx.synchronize()
    ctx.enqueue_copy(hbuf_q, d_q)
    ctx.synchronize()

    var max_h_dev: Float64 = 0.0
    var max_hu_dev: Float64 = 0.0
    var max_hv: Float64 = 0.0
    var n_owned_nodes = gpu_mesh.num_elements * NP_p
    for i in range(n_owned_nodes):
        var h = hptr_q[i * 3 + 0]
        var hu = hptr_q[i * 3 + 1]
        var hv = hptr_q[i * 3 + 2]
        if (isnan(h) or isinf(h) or isnan(hu) or isinf(hu)
            or isnan(hv) or isinf(hv)):
            raise Error("bench_shallow_water_inflow_2d: non-finite output")
        var dh = Float64(h - H0)
        if dh < 0.0: dh = -dh
        if dh > max_h_dev: max_h_dev = dh
        var dhu = Float64(hu - H0 * U0)
        if dhu < 0.0: dhu = -dhu
        if dhu > max_hu_dev: max_hu_dev = dhu
        var ahv = Float64(hv)
        if ahv < 0.0: ahv = -ahv
        if ahv > max_hv: max_hv = ahv

    var hu_rel = max_hu_dev / Float64(H0 * U0)
    print("  max |h - H0|             =", max_h_dev,  "  (threshold", H_TOL,        ")")
    print("  max |hu - H0*U0| / H0*U0 =", hu_rel,     "  (threshold", HU_REL_TOL,   ")")
    print("  max |hv|                 =", max_hv,     "  (threshold", HV_TOL,       ")")

    if max_h_dev > H_TOL:
        raise Error(
            "bench_shallow_water_inflow_2d FAILED: max |h - H0| "
            + String(max_h_dev) + " > " + String(H_TOL)
        )
    if hu_rel > HU_REL_TOL:
        raise Error(
            "bench_shallow_water_inflow_2d FAILED: hu rel err "
            + String(hu_rel) + " > " + String(HU_REL_TOL)
        )
    if max_hv > HV_TOL:
        raise Error(
            "bench_shallow_water_inflow_2d FAILED: max |hv| "
            + String(max_hv) + " > " + String(HV_TOL)
        )

    print("=== bench_shallow_water_inflow_2d PASSED ===")
    mpi.finalize()
