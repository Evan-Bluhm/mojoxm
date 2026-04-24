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
from src.local_mesh_2d import LocalMesh2D
from src.local_mesh_2d_gpu import LocalMesh2DGpu, launch_cell_avg_2d
from src.reference_2d import num_tri_nodes_2d, num_edge_nodes


def _abs(x: Float64) -> Float64:
    return x if x >= 0.0 else -x


def _abs32(x: Float32) -> Float32:
    return x if x >= Float32(0.0) else -x


def check[P: Int]() raises:
    print("  P=", P)
    comptime Nx = 5
    comptime Ny = 4
    comptime NP_p = num_tri_nodes_2d(P)
    var NFP_e = num_edge_nodes(P)

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
