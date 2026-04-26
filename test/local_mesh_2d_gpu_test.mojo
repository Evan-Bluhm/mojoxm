# ======================================================================
# local_mesh_2d_gpu_test -- 2D mesh + advection GPU foundation test
# ======================================================================
#
# Verifies that the 2D GPU foundation works end-to-end at P=1/2/3:
#   (a) LocalMesh2DGpu upload is lossless for Int32 tables and within
#       Float32 round-trip precision for geometry tables.
#   (b) cell_mean_kernel_2d (mass-matrix-weighted nodal quadrature)
#       matches an inline Float32 host computation.
#   (c) D_ref / Lift_ref upload via ReferenceElement2DGpu round-trips
#       cleanly.
#   (d) advection_volume_rhs_kernel_2d + advection_face_flux_kernel_2d
#       agree with inline Float32 host formulas (sin+cos IC, periodic
#       mesh, pure upwind).
#   (e) The full advection SSPRK3 step preserves a constant state to
#       Float32 roundoff -- rigorous scheme self-check for the
#       divergence-theorem cancellation.  Needs no external reference.
#
# This file uses no CPU physics reference: the volume / face-flux
# kernels are checked against inline Float32 formulas that mirror the
# kernel code, and the full SSPRK3 step is verified via a
# constant-state preservation invariant (divergence theorem) rather
# than a ground-truth physics solver.
#
# Runs at np=1 only.  Requires an accelerator.
# ======================================================================

from std.sys import has_accelerator
from std.gpu.host import DeviceContext, DeviceBuffer
from std.math import sin, cos, isnan, isinf
from src import mpi
from src.local_mesh_2d import LocalMesh2D
from src.local_mesh_2d_gpu import LocalMesh2DGpu, launch_cell_mean_2d
from src.local_mesh_2d_gpu_advection import (
    launch_advection_volume_rhs_2d, launch_advection_face_flux_2d,
    advection_rk_stage_2d,
)
from src.reference_2d import ReferenceElement2D, num_tri_nodes_2d, num_edge_nodes
from src.reference_2d_gpu import ReferenceElement2DGpu


def _abs32(x: Float32) -> Float32:
    return x if x >= Float32(0.0) else -x


def check[P: Int]() raises:
    print("  P=", P)
    comptime Nx = 5
    comptime Ny = 4
    comptime NP_p = num_tri_nodes_2d(P)
    comptime NFP_e = num_edge_nodes(P)

    var host = LocalMesh2D[P](Nx, Ny, 1.0, 1.0)
    var ctx = DeviceContext()
    var gpu = LocalMesh2DGpu[P](ctx, host^)

    # Compare elem_node_xyz (Float32 after upload).
    var n_elem_xyz = gpu.num_elements * NP_p * 2
    var hbuf_f = ctx.enqueue_create_host_buffer[DType.float32](n_elem_xyz)
    ctx.enqueue_copy(hbuf_f, gpu.d_elem_node_xyz)
    ctx.synchronize()
    var hptr_f = hbuf_f.unsafe_ptr()
    # Rebuild host coords for comparison (host's elem_node_xyz was
    # consumed by LocalMesh2DGpu's __init__; retrieve them by
    # re-running the builder).
    var host2 = LocalMesh2D[P](Nx, Ny, 1.0, 1.0)
    var max_err_xyz: Float32 = 0.0
    for k in range(n_elem_xyz):
        var e = Float32(host2.elem_node_xyz[k]) - hptr_f[k]
        var ae = _abs32(e)
        if ae > max_err_xyz:
            max_err_xyz = ae
    print("    elem_node_xyz max |f64 -> f32 round-trip err| =",
          max_err_xyz)
    if max_err_xyz > Float32(1.0e-6):
        raise Error("elem_node_xyz upload round-trip failed")

    # Int32 tables should be bit-identical.
    var n_ef = gpu.num_elements * 3
    var hbuf_i = ctx.enqueue_create_host_buffer[DType.int32](n_ef)
    ctx.enqueue_copy(hbuf_i, gpu.d_elem_faces)
    ctx.synchronize()
    var hptr_i = hbuf_i.unsafe_ptr()
    var mismatch = 0
    for k in range(n_ef):
        if hptr_i[k] != host2.elem_faces[k]:
            mismatch += 1
    print("    elem_faces Int32 mismatches =", mismatch, "/", n_ef)
    if mismatch != 0:
        raise Error("elem_faces upload mismatch")

    # face_bc_type -- periodic mesh should be all zeros.
    var n_bc = gpu.num_faces
    var hbuf_bc = ctx.enqueue_create_host_buffer[DType.int32](n_bc)
    ctx.enqueue_copy(hbuf_bc, gpu.d_face_bc_type)
    ctx.synchronize()
    var hptr_bc = hbuf_bc.unsafe_ptr()
    for k in range(n_bc):
        if hptr_bc[k] != 0:
            raise Error("periodic mesh has non-zero face_bc_type")
    print("    face_bc_type all zeros (periodic mesh OK)")

    # Reference-element upload round-trip.  Done before cell_mean test
    # because that kernel needs node_weights from re_gpu.
    var re_host = ReferenceElement2D[P]()
    var re_gpu = ReferenceElement2DGpu[P](ctx, re_host)

    # cell_mean kernel on a synthetic scalar (NC=1).  Mass-matrix-weighted
    # nodal quadrature: sum_i q_i * w_i where w_i = node_weights[i].
    comptime NC = 1
    var n_total = gpu.num_elements * NP_p * NC
    var host_q = List[Float32]()
    for elem in range(gpu.num_elements):
        for nn in range(NP_p):
            host_q.append(Float32(elem) + Float32(0.1) * Float32(nn))
    var d_q = ctx.enqueue_create_buffer[DType.float32](n_total)
    var hbuf_q = ctx.enqueue_create_host_buffer[DType.float32](n_total)
    var hptr_q = hbuf_q.unsafe_ptr()
    for k in range(n_total):
        hptr_q[k] = host_q[k]
    ctx.enqueue_copy(d_q, hbuf_q)
    var d_mean = ctx.enqueue_create_buffer[DType.float32](
        gpu.num_elements * NC
    )
    launch_cell_mean_2d[NP_p, NC](
        ctx, d_q.unsafe_ptr(),
        re_gpu.d_node_weights.unsafe_ptr(),
        gpu.num_elements, d_mean.unsafe_ptr(),
    )
    var hbuf_mean = ctx.enqueue_create_host_buffer[DType.float32](
        gpu.num_elements * NC
    )
    ctx.enqueue_copy(hbuf_mean, d_mean)
    ctx.synchronize()
    var hptr_mean = hbuf_mean.unsafe_ptr()
    var max_mean_err: Float32 = 0.0
    for elem in range(gpu.num_elements):
        var s: Float32 = 0.0
        for nn in range(NP_p):
            s += host_q[elem * NP_p + nn] * Float32(re_host.node_weights[nn])
        var diff = s - hptr_mean[elem]
        var adiff = _abs32(diff)
        if adiff > max_mean_err:
            max_mean_err = adiff
    print("    cell_mean GPU vs CPU max err =", max_mean_err)
    if max_mean_err > Float32(1.0e-4):
        raise Error(
            "cell_mean_kernel_2d mismatch: " + String(max_mean_err)
        )

    var d_ref_len = 2 * NP_p * NP_p
    var hbuf_dref = ctx.enqueue_create_host_buffer[DType.float32](d_ref_len)
    ctx.enqueue_copy(hbuf_dref, re_gpu.d_D_ref)
    ctx.synchronize()
    var dptr = hbuf_dref.unsafe_ptr()
    var max_dref_err: Float32 = 0.0
    for k in range(d_ref_len):
        var diff = Float32(re_host.D_ref[k]) - dptr[k]
        var adiff = _abs32(diff)
        if adiff > max_dref_err:
            max_dref_err = adiff
    print("    D_ref max |f64->f32 err| =", max_dref_err)
    if max_dref_err > Float32(1.0e-5):
        raise Error("D_ref upload round-trip failed")

    var lift_len = 3 * NP_p * NFP_e
    var hbuf_lift = ctx.enqueue_create_host_buffer[DType.float32](lift_len)
    ctx.enqueue_copy(hbuf_lift, re_gpu.d_Lift_ref)
    ctx.synchronize()
    var lptr = hbuf_lift.unsafe_ptr()
    var max_lift_err: Float32 = 0.0
    for k in range(lift_len):
        var diff = Float32(re_host.Lift_ref[k]) - lptr[k]
        var adiff = _abs32(diff)
        if adiff > max_lift_err:
            max_lift_err = adiff
    print("    Lift_ref max |f64->f32 err| =", max_lift_err)
    if max_lift_err > Float32(1.0e-5):
        raise Error("Lift_ref upload round-trip failed")

    # GPU volume-only rhs vs inline Float32 host formula (no CPU
    # physics code -- the host reference is hand-coded here to match
    # `advection_volume_rhs_kernel_2d` exactly).
    var vx = Float32(0.7)
    var vy = Float32(-0.4)
    var two_pi = Float64(6.283185307179586)
    var q_host_f32 = List[Float32]()
    for elem in range(gpu.num_elements):
        for nn in range(NP_p):
            var x = host2.elem_node_xyz[(elem * NP_p + nn) * 2 + 0]
            var y = host2.elem_node_xyz[(elem * NP_p + nn) * 2 + 1]
            q_host_f32.append(
                Float32(sin(two_pi * x) + cos(two_pi * y))
            )

    for k in range(gpu.num_elements * NP_p):
        hptr_q[k] = q_host_f32[k]
    ctx.enqueue_copy(d_q, hbuf_q)
    var d_vol = ctx.enqueue_create_buffer[DType.float32](
        gpu.num_elements * NP_p
    )
    launch_advection_volume_rhs_2d[NP_p](
        ctx,
        d_q.unsafe_ptr(),
        gpu.d_elem_invJ.unsafe_ptr(),
        re_gpu.d_D_ref.unsafe_ptr(),
        gpu.num_elements, vx, vy,
        d_vol.unsafe_ptr(),
    )
    var hbuf_vol = ctx.enqueue_create_host_buffer[DType.float32](
        gpu.num_elements * NP_p
    )
    ctx.enqueue_copy(hbuf_vol, d_vol)
    ctx.synchronize()
    var hptr_vol = hbuf_vol.unsafe_ptr()

    var cpu_vol = List[Float32]()
    for elem in range(gpu.num_elements):
        var iJ00 = Float32(host2.elem_invJ[elem * 4 + 0])
        var iJ01 = Float32(host2.elem_invJ[elem * 4 + 1])
        var iJ10 = Float32(host2.elem_invJ[elem * 4 + 2])
        var iJ11 = Float32(host2.elem_invJ[elem * 4 + 3])
        for i in range(NP_p):
            var vol_c = Float32(0.0)
            for j in range(NP_p):
                var qj = q_host_f32[elem * NP_p + j]
                var fx = vx * qj
                var fy = vy * qj
                var fr0 = iJ00 * fx + iJ01 * fy
                var fr1 = iJ10 * fx + iJ11 * fy
                var D_r = Float32(re_host.D_ref[0 * NP_p * NP_p + i * NP_p + j])
                var D_s = Float32(re_host.D_ref[1 * NP_p * NP_p + i * NP_p + j])
                vol_c += fr0 * D_r + fr1 * D_s
            cpu_vol.append(vol_c)

    var max_vol_err: Float32 = 0.0
    for k in range(len(cpu_vol)):
        var diff = cpu_vol[k] - hptr_vol[k]
        var adiff = _abs32(diff)
        if adiff > max_vol_err:
            max_vol_err = adiff
    print("    volume rhs GPU vs CPU max err =", max_vol_err)
    if max_vol_err > Float32(1.0e-4):
        raise Error(
            "advection_volume_rhs_kernel_2d: max err "
            + String(max_vol_err)
        )

    # GPU face-flux kernel vs inline upwind (periodic mesh -> every
    # face is BC_INTERIOR).
    var d_fstar = ctx.enqueue_create_buffer[DType.float32](
        gpu.num_faces * NFP_e
    )
    launch_advection_face_flux_2d[NP_p, NFP_e](
        ctx,
        d_q.unsafe_ptr(),
        gpu.d_face_elem.unsafe_ptr(),
        gpu.d_face_elem_node.unsafe_ptr(),
        gpu.d_face_normal.unsafe_ptr(),
        gpu.d_face_bc_type.unsafe_ptr(),
        gpu.num_faces,
        vx, vy, Float32(0.0),
        d_fstar.unsafe_ptr(),
    )
    var hbuf_fstar = ctx.enqueue_create_host_buffer[DType.float32](
        gpu.num_faces * NFP_e
    )
    ctx.enqueue_copy(hbuf_fstar, d_fstar)
    ctx.synchronize()
    var fstar_ptr = hbuf_fstar.unsafe_ptr()
    var max_fstar_err: Float32 = 0.0
    for fid in range(gpu.num_faces):
        var fnx = Float32(host2.face_normal[fid * 2 + 0])
        var fny = Float32(host2.face_normal[fid * 2 + 1])
        var vn = vx * fnx + vy * fny
        var e_l = Int(host2.face_elem[fid * 2 + 0])
        var e_r = Int(host2.face_elem[fid * 2 + 1])
        for m in range(NFP_e):
            var n_l = Int(host2.face_elem_node[(fid * 2 + 0) * NFP_e + m])
            var n_r = Int(host2.face_elem_node[(fid * 2 + 1) * NFP_e + m])
            var q_l = q_host_f32[e_l * NP_p + n_l]
            var q_r = q_host_f32[e_r * NP_p + n_r]
            var cpu_fstar: Float32
            if vn >= Float32(0.0):
                cpu_fstar = vn * q_l
            else:
                cpu_fstar = vn * q_r
            var diff = cpu_fstar - fstar_ptr[fid * NFP_e + m]
            var adiff = _abs32(diff)
            if adiff > max_fstar_err:
                max_fstar_err = adiff
    print("    face flux GPU vs CPU max err =", max_fstar_err)
    if max_fstar_err > Float32(1.0e-5):
        raise Error(
            "advection_face_flux_kernel_2d: max err "
            + String(max_fstar_err)
        )

    # Full advection SSPRK3 step on a constant IC: the divergence
    # theorem cancels volume + face contributions exactly, so the
    # updated q must equal the IC to Float32 roundoff.  Covers the
    # full lift-combine + rk-update chain without a reference.
    var q_const: Float32 = 3.14
    for k in range(gpu.num_elements * NP_p):
        hptr_q[k] = q_const
    ctx.enqueue_copy(d_q, hbuf_q)
    ctx.synchronize()

    var d_q1 = ctx.enqueue_create_buffer[DType.float32](
        gpu.num_elements * NP_p
    )
    var d_q2 = ctx.enqueue_create_buffer[DType.float32](
        gpu.num_elements * NP_p
    )
    var dt_step = Float32(0.001)
    advection_rk_stage_2d[P](
        ctx, gpu,
        re_gpu.d_Lift_ref.unsafe_ptr(), re_gpu.d_D_ref.unsafe_ptr(),
        d_q.unsafe_ptr(),
        d_q.unsafe_ptr(), d_q.unsafe_ptr(),
        d_q1.unsafe_ptr(),
        d_fstar.unsafe_ptr(),
        vx, vy,
        Float32(1.0), Float32(0.0), Float32(1.0), dt_step,
    )
    advection_rk_stage_2d[P](
        ctx, gpu,
        re_gpu.d_Lift_ref.unsafe_ptr(), re_gpu.d_D_ref.unsafe_ptr(),
        d_q1.unsafe_ptr(),
        d_q.unsafe_ptr(), d_q1.unsafe_ptr(),
        d_q2.unsafe_ptr(),
        d_fstar.unsafe_ptr(),
        vx, vy,
        Float32(0.75), Float32(0.25), Float32(0.25), dt_step,
    )
    advection_rk_stage_2d[P](
        ctx, gpu,
        re_gpu.d_Lift_ref.unsafe_ptr(), re_gpu.d_D_ref.unsafe_ptr(),
        d_q2.unsafe_ptr(),
        d_q.unsafe_ptr(), d_q2.unsafe_ptr(),
        d_q.unsafe_ptr(),
        d_fstar.unsafe_ptr(),
        vx, vy,
        Float32(1.0 / 3.0), Float32(2.0 / 3.0),
        Float32(2.0 / 3.0), dt_step,
    )
    ctx.enqueue_copy(hbuf_q, d_q)
    ctx.synchronize()
    var max_const_err: Float32 = 0.0
    for k in range(gpu.num_elements * NP_p):
        var v = hptr_q[k]
        if isnan(v) or isinf(v):
            raise Error("advection SSPRK3: non-finite at index " + String(k))
        var err = _abs32(v - q_const)
        if err > max_const_err:
            max_const_err = err
    print("    constant-state SSPRK3 step max |q - q_IC| =",
          max_const_err)
    if max_const_err > Float32(1.0e-4):
        raise Error(
            "advection SSPRK3: constant state not preserved (max err "
            + String(max_const_err) + ")"
        )


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    mpi.init()
    var size = mpi.world_size()
    if size > 1:
        mpi.finalize()
        print("local_mesh_2d_gpu_test: runs at np=1 only")
        return
    print("local_mesh_2d_gpu_test -- mesh upload + advection pipeline")
    check[1]()
    check[2]()
    check[3]()
    print("=== local_mesh_2d_gpu_test PASSED ===")
    mpi.finalize()
