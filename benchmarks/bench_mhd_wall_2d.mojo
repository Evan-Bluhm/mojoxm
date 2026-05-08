# ======================================================================
# bench_mhd_wall_2d -- BC_WALL preservation gate for 2D plain MHD
# ======================================================================
#
# Plain-MHD analog of bench_mhd_wall_2d_glm.  Same setup -- u=0, B
# tangential to the +/-y walls (periodic in x, walls only in y) --
# routed through `mhd_face_flux_kernel_2d` (NC=6) instead of the
# GLM kernel.  Closes a coverage gap: the BC_WALL branch in the
# plain-MHD kernel (lines 287-295 of src/local_mesh_2d_gpu_mhd.mojo)
# was previously untested.
#
# Pass criteria (P=2, NX=8 NY=4, T=1):
#   * max relative drift in any of the 6 components < 1e-3
#   * no NaN / Inf
# ======================================================================

from std.math import sqrt, isnan, isinf
from std.sys import has_accelerator
from std.gpu.host import DeviceContext, DeviceBuffer
from src import mpi
from src.local_mesh_2d import LocalMesh2D
from src.local_mesh_2d_gpu import LocalMesh2DGpu
from src.local_mesh_2d_gpu_mhd import mhd_rk_stage_2d
from src.ssprk3 import ssprk3_stage_plans
from src.reference_2d import (
    ReferenceElement2D,
    num_tri_nodes_2d,
    num_edge_nodes,
)
from src.reference_2d_gpu import ReferenceElement2DGpu
from src.boundary import BoundaryConditions2D, BC_WALL, BC_INTERIOR


comptime P = 2
comptime NX = 8
comptime NY = 4
comptime LX = 1.0
comptime LY = Float64(NY) / Float64(NX) * LX

comptime GAMMA: Float32 = Float32(5.0 / 3.0)
comptime RHO0: Float32 = 1.0
comptime B0: Float32 = 1.0  # B_x; tangential to +/-y walls
comptime P0: Float32 = 0.5
comptime MIN_RHO: Float32 = 1.0e-6
comptime MIN_P: Float32 = 1.0e-6
comptime CFL: Float64 = 0.15
comptime T_FINAL: Float64 = 1.0

comptime REL_TOL: Float64 = 1.0e-3


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    mpi.init()
    var size = mpi.world_size()
    if size > 1:
        mpi.finalize()
        print("bench_mhd_wall_2d: runs at np=1 only")
        return

    print("bench_mhd_wall_2d (BC_WALL preservation, plain MHD NC=6)")
    print(
        "  P=",
        P,
        "  mesh=",
        NX,
        "x",
        NY,
        "   B0=",
        B0,
        "   T=",
        T_FINAL,
        "  (u=0; B tangent to +/-y walls; periodic in x)",
    )

    comptime NP_p = num_tri_nodes_2d(P)
    comptime NFP_e = num_edge_nodes(P)
    comptime NC = 6
    var ctx = DeviceContext()

    var bcs = BoundaryConditions2D(
        BC_INTERIOR,
        BC_INTERIOR,  # -x, +x periodic
        BC_WALL,
        BC_WALL,  # -y, +y reflecting
    )
    var host_mesh = LocalMesh2D[P](Nx=NX, Ny=NY, Lx=LX, Ly=LY, bcs=bcs)
    var host_re = ReferenceElement2D[P]()
    var gpu_mesh = LocalMesh2DGpu[P](ctx=ctx, host=host_mesh^)
    var gpu_re = ReferenceElement2DGpu[P](ctx=ctx, host=host_re)

    var E0 = P0 / (GAMMA - Float32(1.0)) + Float32(0.5) * B0 * B0

    var n_q = gpu_mesh.num_elements * NP_p * NC
    var host_q = List[Float32]()
    for _ in range(gpu_mesh.num_elements * NP_p):
        host_q.append(RHO0)
        host_q.append(Float32(0.0))  # rhou
        host_q.append(Float32(0.0))  # rhov
        host_q.append(B0)  # Bx
        host_q.append(Float32(0.0))  # By
        host_q.append(E0)  # E

    var d_q = ctx.enqueue_create_buffer[DType.float32](n_q)
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

    var cf = sqrt(GAMMA * P0 / RHO0 + B0 * B0 / RHO0)
    var h_cell = LX / Float64(NX)
    var dt_est = CFL * Float64(h_cell) / (Float64(cf) * Float64(2 * P + 1))
    var num_steps = Int(T_FINAL / dt_est) + 1
    var dt = Float32(T_FINAL / Float64(num_steps))
    print("  c_f=", cf, "  steps=", num_steps, "  dt=", dt)

    var stage_plans = ssprk3_stage_plans(
        d_q=d_q.unsafe_ptr(),
        d_q1=d_q1.unsafe_ptr(),
        d_q2=d_q2.unsafe_ptr(),
    )
    for _ in range(num_steps):
        for stage in stage_plans:
            mhd_rk_stage_2d[P](
                ctx=ctx,
                mesh=gpu_mesh,
                Lift_ref=gpu_re.d_Lift_ref.unsafe_ptr(),
                D_ref=gpu_re.d_D_ref.unsafe_ptr(),
                q_in=stage.q_in,
                q_a=stage.q_a,
                q_b=stage.q_b,
                q_out=stage.q_out,
                fstar_scratch=d_fstar.unsafe_ptr(),
                gamma=GAMMA,
                min_density=MIN_RHO,
                min_pressure=MIN_P,
                a=stage.a,
                b=stage.b,
                cc=stage.c,
                dt=dt,
            )
    ctx.synchronize()
    ctx.enqueue_copy(hbuf_q, d_q)
    ctx.synchronize()

    var ref_scale = Float64(E0)
    var max_drift: Float64 = 0.0
    var n_owned_nodes = gpu_mesh.num_elements * NP_p
    for i in range(n_owned_nodes):
        for c in range(NC):
            var qv = hptr_q[i * NC + c]
            if isnan(qv) or isinf(qv):
                raise Error("bench_mhd_wall_2d: non-finite output")
            var qref = host_q[i * NC + c]
            var d = Float64(qv) - Float64(qref)
            if d < 0.0:
                d = -d
            var rel = d / ref_scale
            if rel > max_drift:
                max_drift = rel

    print("  max relative drift   =", max_drift, "  (threshold", REL_TOL, ")")

    if max_drift > REL_TOL:
        raise Error(
            "bench_mhd_wall_2d FAILED: max relative drift "
            + String(max_drift)
            + " > "
            + String(REL_TOL)
        )

    print("=== bench_mhd_wall_2d PASSED ===")
    mpi.finalize()
