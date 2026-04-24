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
from src.local_mesh_2d_gpu import LocalMesh2DGpu
from src.reference_2d import num_tri_nodes_2d, num_edge_nodes


def _abs(x: Float64) -> Float64:
    return x if x >= 0.0 else -x


def _abs32(x: Float32) -> Float32:
    return x if x >= Float32(0.0) else -x


def check[P: Int]() raises:
    print("  P=", P)
    comptime Nx = 5
    comptime Ny = 4
    var NP_p = num_tri_nodes_2d(P)
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
