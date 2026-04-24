# ======================================================================
# local_mesh_2d_gpu_test -- smoke-test the 2D GPU mesh upload
# ======================================================================
#
# Builds a host LocalMesh2D[P], wraps it in a LocalMesh2DGpu[P], and
# verifies by download-compare on several buffers that
#   (a) the upload roundtrip is lossless for Int32 tables,
#   (b) the Float64 -> Float32 conversion round-trips within Float32's
#       relative precision (< 1e-6) for the geometry tables.
#
# This is the foundational task #19 proof-of-life: the CPU mesh is
# now accessible to GPU kernels in the native format they'll need.
# Kernels themselves (rk_stage_kernel_2d, etc.) follow in future
# iterations.
#
# Runs at np=1 only.  Requires an accelerator.
# ======================================================================

from std.sys import has_accelerator
from std.gpu.host import DeviceContext, DeviceBuffer
from src import mpi
from std.math import sin, cos, pi
from src.local_mesh_2d import LocalMesh2D
from src.local_mesh_2d_gpu import (
    LocalMesh2DGpu, launch_cell_avg_2d,
    launch_advection_volume_rhs_2d, launch_advection_face_flux_2d,
    launch_advection_lift_combine_2d, launch_rk_update_2d,
)
from src.dg_rhs_2d import Advection2D, dg_rhs_2d
from src.reference_2d import ReferenceElement2D, num_tri_nodes_2d, num_edge_nodes
from src.reference_2d_gpu import ReferenceElement2DGpu


def _abs(x: Float64) -> Float64:
    return x if x >= 0.0 else -x


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
    # consumed by LocalMesh2DGpu's __init__; we retrieve them by
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
    print("    elem_faces Int32 mismatches =", mismatch,
          "/", n_ef)
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

    # Compute GPU cell averages on a synthetic q = (elem_idx + 0.1 * nn)
    # and compare to a host computation.  NC=1 (scalar).  Proves the
    # device q buffer + cell_avg_kernel_2d + download all compose.
    comptime NC = 1
    var n_total = gpu.num_elements * NP_p * NC
    var host_q = List[Float32]()
    for elem in range(gpu.num_elements):
        for nn in range(NP_p):
            host_q.append(Float32(elem) + Float32(0.1) * Float32(nn))
    # Upload q to device.
    var d_q = ctx.enqueue_create_buffer[DType.float32](n_total)
    var hbuf_q = ctx.enqueue_create_host_buffer[DType.float32](n_total)
    var hptr_q = hbuf_q.unsafe_ptr()
    for k in range(n_total):
        hptr_q[k] = host_q[k]
    ctx.enqueue_copy(d_q, hbuf_q)
    var d_avg = ctx.enqueue_create_buffer[DType.float32](
        gpu.num_elements * NC
    )
    launch_cell_avg_2d[NP_p, NC](
        ctx, d_q.unsafe_ptr(), gpu.num_elements, d_avg.unsafe_ptr(),
    )
    # Download cell averages.
    var hbuf_avg = ctx.enqueue_create_host_buffer[DType.float32](
        gpu.num_elements * NC
    )
    ctx.enqueue_copy(hbuf_avg, d_avg)
    ctx.synchronize()
    var hptr_avg = hbuf_avg.unsafe_ptr()
    # Host reference.
    var max_avg_err: Float32 = 0.0
    var inv_np = Float32(1.0) / Float32(NP_p)
    for elem in range(gpu.num_elements):
        var s: Float32 = 0.0
        for nn in range(NP_p):
            s += host_q[elem * NP_p + nn]
        var cpu_avg = s * inv_np
        var gpu_val = hptr_avg[elem]
        var diff = cpu_avg - gpu_val
        var adiff = diff if diff >= Float32(0.0) else -diff
        if adiff > max_avg_err:
            max_avg_err = adiff
    print("    cell_avg GPU vs CPU max err =", max_avg_err)
    if max_avg_err > Float32(1.0e-4):
        raise Error(
            "cell_avg_kernel_2d mismatch: " + String(max_avg_err)
        )

    # Reference-element upload: verify D_ref and Lift_ref round-trip
    # Float64 -> Float32 within Float32 precision.
    var re_host = ReferenceElement2D[P]()
    var re_gpu = ReferenceElement2DGpu[P](ctx, re_host)

    var d_ref_len = 2 * NP_p * NP_p
    var hbuf_dref = ctx.enqueue_create_host_buffer[DType.float32](d_ref_len)
    ctx.enqueue_copy(hbuf_dref, re_gpu.d_D_ref)
    ctx.synchronize()
    var dptr = hbuf_dref.unsafe_ptr()
    var max_dref_err: Float32 = 0.0
    for k in range(d_ref_len):
        var diff = Float32(re_host.D_ref[k]) - dptr[k]
        var adiff = diff if diff >= Float32(0.0) else -diff
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
        var adiff = diff if diff >= Float32(0.0) else -diff
        if adiff > max_lift_err:
            max_lift_err = adiff
    print("    Lift_ref max |f64->f32 err| =", max_lift_err)
    if max_lift_err > Float32(1.0e-5):
        raise Error("Lift_ref upload round-trip failed")

    # GPU volume-only rhs for scalar advection, compared to a host
    # reference.  IC: q = sin(2 pi x) + cos(2 pi y) so vol_c is non-
    # trivial (host computes in Float64 then casts -- small FP drift
    # vs the GPU's Float32 math is expected).
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

    # Upload to an already-created d_q; compute vol on device.
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

    # Host reference: pure Float32 (matches GPU path bit-perfectly).
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
        var adiff = diff if diff >= Float32(0.0) else -diff
        if adiff > max_vol_err:
            max_vol_err = adiff
    print("    volume rhs GPU vs CPU max err =", max_vol_err)
    if max_vol_err > Float32(1.0e-4):
        raise Error(
            "advection_volume_rhs_kernel_2d: max err "
            + String(max_vol_err)
        )

    # GPU face-flux kernel for scalar advection (periodic mesh, so
    # every face is BC_INTERIOR and the CPU reference collapses to
    # pure upwind).
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
            var adiff = diff if diff >= Float32(0.0) else -diff
            if adiff > max_fstar_err:
                max_fstar_err = adiff
    print("    face flux GPU vs CPU max err =", max_fstar_err)
    if max_fstar_err > Float32(1.0e-5):
        raise Error(
            "advection_face_flux_kernel_2d: max err "
            + String(max_fstar_err)
        )

    # Full GPU advection rhs: combine vol_c + fstar -> rhs, compare
    # to CPU `dg_rhs_2d(Advection2D(vx, vy), ...)`.
    var d_rhs = ctx.enqueue_create_buffer[DType.float32](
        gpu.num_elements * NP_p
    )
    launch_advection_lift_combine_2d[NP_p, NFP_e](
        ctx,
        d_vol.unsafe_ptr(),
        d_fstar.unsafe_ptr(),
        gpu.d_elem_inv_2A.unsafe_ptr(),
        gpu.d_elem_faces.unsafe_ptr(),
        gpu.d_elem_face_side.unsafe_ptr(),
        gpu.d_elem_canon_to_ref.unsafe_ptr(),
        gpu.d_face_length.unsafe_ptr(),
        re_gpu.d_Lift_ref.unsafe_ptr(),
        gpu.num_elements,
        d_rhs.unsafe_ptr(),
    )
    var hbuf_rhs = ctx.enqueue_create_host_buffer[DType.float32](
        gpu.num_elements * NP_p
    )
    ctx.enqueue_copy(hbuf_rhs, d_rhs)
    ctx.synchronize()
    var rhs_ptr = hbuf_rhs.unsafe_ptr()

    # CPU reference via dg_rhs_2d.  Need Float64 q to match its API;
    # pass through the Float32 IC by casting each entry.
    var q_f64 = List[Float64]()
    for k in range(gpu.num_elements * NP_p):
        q_f64.append(Float64(q_host_f32[k]))
    var rhs_f64 = List[Float64]()
    for _ in range(gpu.num_elements * NP_p):
        rhs_f64.append(0.0)
    var phys = Advection2D(Float64(vx), Float64(vy))
    dg_rhs_2d[P, Advection2D](
        host2, re_host, phys, q_f64, rhs_f64,
    )
    var max_rhs_err: Float32 = 0.0
    for k in range(gpu.num_elements * NP_p):
        var diff = Float32(rhs_f64[k]) - rhs_ptr[k]
        var adiff = diff if diff >= Float32(0.0) else -diff
        if adiff > max_rhs_err:
            max_rhs_err = adiff
    print("    full rhs (CPU f64 vs GPU f32) max err =", max_rhs_err)
    # Tolerance allows Float64->Float32 precision loss on the sin/cos
    # IC and the accumulation ordering difference.  1e-4 is generous.
    if max_rhs_err > Float32(1.0e-4):
        raise Error(
            "GPU advection rhs disagrees with CPU dg_rhs_2d: "
            + String(max_rhs_err)
        )

    # RK-update combiner: reuse d_q (acts as q_a and q_b in a SSPRK3
    # stage-1 call where q_a = q_b = q) with the computed rhs and
    # dt = 0.01.  Expected: q_new[k] = q[k] + 0.01 * rhs[k] when
    # (a, b, cc) = (1, 0, 1).
    var d_qnew = ctx.enqueue_create_buffer[DType.float32](
        gpu.num_elements * NP_p
    )
    launch_rk_update_2d[NP_p, 1](
        ctx,
        d_q.unsafe_ptr(), d_q.unsafe_ptr(), d_rhs.unsafe_ptr(),
        gpu.num_elements,
        Float32(1.0), Float32(0.0), Float32(1.0), Float32(0.01),
        d_qnew.unsafe_ptr(),
    )
    var hbuf_qnew = ctx.enqueue_create_host_buffer[DType.float32](
        gpu.num_elements * NP_p
    )
    ctx.enqueue_copy(hbuf_qnew, d_qnew)
    ctx.synchronize()
    var qnew_ptr = hbuf_qnew.unsafe_ptr()
    var max_update_err: Float32 = 0.0
    for k in range(gpu.num_elements * NP_p):
        var expect = q_host_f32[k] + Float32(0.01) * rhs_ptr[k]
        var diff = expect - qnew_ptr[k]
        var adiff = diff if diff >= Float32(0.0) else -diff
        if adiff > max_update_err:
            max_update_err = adiff
    print("    RK-update kernel err =", max_update_err)
    if max_update_err > Float32(1.0e-6):
        raise Error(
            "rk_update_kernel_2d: max err " + String(max_update_err)
        )


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    mpi.init()
    var size = mpi.world_size()
    if size > 1:
        mpi.finalize()
        print("local_mesh_2d_gpu_test: runs at np=1 only")
        return
    print("local_mesh_2d_gpu_test -- upload roundtrip")
    check[1]()
    check[2]()
    check[3]()
    print("=== local_mesh_2d_gpu_test PASSED ===")
    mpi.finalize()
