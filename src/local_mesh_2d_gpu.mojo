# ======================================================================
# local_mesh_2d_gpu.mojo -- 2D GPU mesh wrapper + shared utilities
# ======================================================================
#
# This module holds the foundation of the 2D GPU DG stack:
#
#   * `LocalMesh2DGpu[P]` -- Float32 device mirror of `LocalMesh2D[P]`
#     (`src/local_mesh_2d.mojo` is host-side Float64; the GPU stack
#     runs in Float32 to match the 3D stack's register / shared-mem
#     budget).  Constructor takes a host `LocalMesh2D[P]` +
#     `DeviceContext`, converts Float64 -> Float32 in pinned host
#     buffers, and enqueues the upload.  Int32 tables (face topology,
#     bc_type, etc.) transfer directly.
#
#   * Generic NC-templated kernels usable by every physics path:
#     `cell_avg_kernel_2d`        -- unweighted nodal-mean reduction
#     `cell_mean_kernel_2d`       -- mass-matrix-weighted true cell mean
#     `rk_update_kernel_2d`       -- q_out = a*q_a + b*q_b + cc*dt*rhs
#
# Per-physics kernels live in dedicated sibling modules to keep this
# file focused on the shared infrastructure:
#
#   src/local_mesh_2d_gpu_advection.mojo  -- scalar advection (NC=1)
#   src/local_mesh_2d_gpu_euler.mojo      -- Euler (NC=4): Rusanov + HLLC,
#                                            gravity (gx, gy) source
#   src/local_mesh_2d_gpu_sw.mojo         -- Shallow Water (NC=3): Rusanov + HLL
#   src/local_mesh_2d_gpu_mhd.mojo        -- IdealMHD (NC=6, no GLM)
#   src/local_mesh_2d_gpu_mhd_glm.mojo    -- IdealMHD + Dedner GLM (NC=7)
#   src/local_mesh_2d_gpu_maxwell.mojo    -- Maxwell (NC=6 EM): full BC menu,
#                                            J / M source
#   src/local_mesh_2d_gpu_limiter.mojo    -- Barth-Jespersen slope limiter
#
# Buffer layout (mirrors the host mesh exactly; no reshuffling):
#   d_elem_node_xyz    [num_elements * NP * 2]   Float32
#   d_elem_invJ        [num_elements * 4]        Float32
#   d_elem_inv_2A      [num_elements]            Float32
#   d_elem_faces       [num_elements * 3]        Int32
#   d_elem_face_side   [num_elements * 3]        Int32
#   d_elem_canon_to_ref [num_elements * 3 * (P+1)] Int32
#   d_face_elem        [num_faces * 2]           Int32
#   d_face_elem_node   [num_faces * 2 * (P+1)]   Int32
#   d_face_normal      [num_faces * 2]           Float32
#   d_face_length      [num_faces]               Float32
#   d_face_bc_type     [num_faces]               Int32
# ======================================================================

from src.local_mesh_2d import LocalMesh2D
from src.reference_2d import num_tri_nodes_2d, num_edge_nodes
from src.boundary import BC_INTERIOR, BC_WALL, BC_OUTFLOW, BC_INFLOW
from std.gpu import global_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from std.math import ceildiv, sqrt
from std.memory import memcpy


comptime mesh_f = DType.float32
comptime mesh_i = DType.int32


def _upload_f64_as_f32(
    mut ctx: DeviceContext, src: List[Float64]
) raises -> DeviceBuffer[mesh_f]:
    """Convert a host Float64 list to Float32 in a pinned host buffer,
    then enqueue_copy it to a fresh device buffer.  Used for the mesh
    geometry (coordinates, Jacobians, face normals, etc.) -- the 2D
    GPU solver will run in Float32 like the 3D one does."""
    var n = len(src)
    var hbuf = ctx.enqueue_create_host_buffer[mesh_f](n)
    var hptr = hbuf.unsafe_ptr()
    for k in range(n):
        hptr[k] = Float32(src[k])
    var dbuf = ctx.enqueue_create_buffer[mesh_f](n)
    ctx.enqueue_copy(dbuf, hbuf)
    return dbuf^


def _upload_i32(
    mut ctx: DeviceContext, src: List[Int32]
) raises -> DeviceBuffer[mesh_i]:
    var n = len(src)
    var hbuf = ctx.enqueue_create_host_buffer[mesh_i](n)
    memcpy(dest=hbuf.unsafe_ptr(), src=src.unsafe_ptr(), count=n)
    var dbuf = ctx.enqueue_create_buffer[mesh_i](n)
    ctx.enqueue_copy(dbuf, hbuf)
    return dbuf^


# ----------------------------------------------------------------------
# GPU mesh wrapper
# ----------------------------------------------------------------------

struct LocalMesh2DGpu[P: Int = 2](Movable):
    comptime NP = num_tri_nodes_2d(Self.P)
    comptime NFP_edge = num_edge_nodes(Self.P)

    var num_elements: Int
    var num_faces: Int

    var d_elem_node_xyz:    DeviceBuffer[mesh_f]
    var d_elem_invJ:        DeviceBuffer[mesh_f]
    var d_elem_inv_2A:      DeviceBuffer[mesh_f]
    var d_elem_faces:       DeviceBuffer[mesh_i]
    var d_elem_face_side:   DeviceBuffer[mesh_i]
    var d_elem_canon_to_ref: DeviceBuffer[mesh_i]
    var d_face_elem:        DeviceBuffer[mesh_i]
    var d_face_elem_node:   DeviceBuffer[mesh_i]
    var d_face_normal:      DeviceBuffer[mesh_f]
    var d_face_length:      DeviceBuffer[mesh_f]
    var d_face_bc_type:     DeviceBuffer[mesh_i]

    def __init__(
        out self,
        mut ctx: DeviceContext,
        host: LocalMesh2D[Self.P],
    ) raises:
        self.num_elements = host.num_elements
        self.num_faces    = host.num_faces

        self.d_elem_node_xyz    = _upload_f64_as_f32(ctx, host.elem_node_xyz)
        self.d_elem_invJ        = _upload_f64_as_f32(ctx, host.elem_invJ)
        self.d_elem_inv_2A      = _upload_f64_as_f32(ctx, host.elem_inv_2A)
        self.d_elem_faces       = _upload_i32(ctx, host.elem_faces)
        self.d_elem_face_side   = _upload_i32(ctx, host.elem_face_side)
        self.d_elem_canon_to_ref = _upload_i32(ctx, host.elem_canon_to_ref)
        self.d_face_elem        = _upload_i32(ctx, host.face_elem)
        self.d_face_elem_node   = _upload_i32(ctx, host.face_elem_node)
        self.d_face_normal      = _upload_f64_as_f32(ctx, host.face_normal)
        self.d_face_length      = _upload_f64_as_f32(ctx, host.face_length)
        self.d_face_bc_type     = _upload_i32(ctx, host.face_bc_type)
        ctx.synchronize()


# ----------------------------------------------------------------------
# First real GPU kernel on the 2D mesh: per-element cell averages.
# ----------------------------------------------------------------------
# Simple reduction -- one thread per element, loops over NP nodes and
# NC components, writes the mean to `cell_avg_out[elem * NC + c]`.
# Useful in its own right (cell averages feed the BJ slope limiter's
# neighbour comparison) and serves as the "hello GPU" for the 2D mesh:
# tests can round-trip an arbitrary q through the device and confirm
# the kernel reads the right stride layout.
# ----------------------------------------------------------------------

def cell_avg_kernel_2d[NP: Int, NC: Int](
    q:            UnsafePointer[Float32, MutAnyOrigin],
    num_elements: Int,
    cell_avg:     UnsafePointer[Float32, MutAnyOrigin],
):
    var elem = Int(global_idx.x)
    if elem >= num_elements:
        return
    var base_q = elem * NP * NC
    var base_avg = elem * NC
    var inv_np = Float32(1.0) / Float32(NP)
    for c in range(NC):
        var s: Float32 = 0.0
        for nn in range(NP):
            s += q[base_q + nn * NC + c]
        cell_avg[base_avg + c] = s * inv_np


def launch_cell_avg_2d[NP: Int, NC: Int](
    mut ctx: DeviceContext,
    q:        UnsafePointer[Float32, MutAnyOrigin],
    num_elements: Int,
    cell_avg: UnsafePointer[Float32, MutAnyOrigin],
) raises:
    """Convenience launcher: 256 threads/block, one element per thread."""
    comptime _kernel = cell_avg_kernel_2d[NP, NC]
    ctx.enqueue_function[_kernel, _kernel](
        q, num_elements, cell_avg,
        grid_dim=ceildiv(num_elements, 256),
        block_dim=256,
    )


# ----------------------------------------------------------------------
# Cell mean kernel (mass-matrix-weighted average).
# ----------------------------------------------------------------------
# Unlike `cell_avg_kernel_2d` (unweighted nodal arithmetic mean),
# this computes the true DG cell mean
#
#   cell_mean[elem, c] = sum_i q[elem, i, c] * w_i
#
# where w_i = int phi_i dr ds / A_ref = 2 int phi_i dr ds are the
# mass-matrix quadrature weights (uploaded as `d_node_weights` in
# ReferenceElement2DGpu).  For P=1 Lagrange all w_i = 1/3 and the two
# kernels coincide.  For P>=2 they differ: at P=2 the 3 vertex
# weights are zero and the 3 edge-midpoint weights are 1/3 each, so
# an unweighted "average" over all 6 nodes is not the cell mean.
#
# The BJ slope limiter needs this properly-weighted mean; using the
# unweighted kernel at P>=2 produced a systematic shock-speed drift
# (e.g. ~10 cells on the Sod shock tube at NX=256 / P=2).
# ----------------------------------------------------------------------

def cell_mean_kernel_2d[NP: Int, NC: Int](
    q:            UnsafePointer[Float32, MutAnyOrigin],
    node_weights: UnsafePointer[Float32, MutAnyOrigin],
    num_elements: Int,
    cell_mean:    UnsafePointer[Float32, MutAnyOrigin],
):
    var elem = Int(global_idx.x)
    if elem >= num_elements:
        return
    var base_q = elem * NP * NC
    var base_mean = elem * NC
    for c in range(NC):
        var s: Float32 = 0.0
        for nn in range(NP):
            s += q[base_q + nn * NC + c] * node_weights[nn]
        cell_mean[base_mean + c] = s


def launch_cell_mean_2d[NP: Int, NC: Int](
    mut ctx: DeviceContext,
    q:            UnsafePointer[Float32, MutAnyOrigin],
    node_weights: UnsafePointer[Float32, MutAnyOrigin],
    num_elements: Int,
    cell_mean:    UnsafePointer[Float32, MutAnyOrigin],
) raises:
    comptime _kernel = cell_mean_kernel_2d[NP, NC]
    ctx.enqueue_function[_kernel, _kernel](
        q, node_weights, num_elements, cell_mean,
        grid_dim=ceildiv(num_elements, 256),
        block_dim=256,
    )


# ----------------------------------------------------------------------
# RK-update combiner: q_out = a * q_a + b * q_b + cc * dt * rhs.
# ----------------------------------------------------------------------
# The fused per-physics `*_vol_lift_combine_rk_kernel_2d` paths perform
# this update inline, so this kernel is currently only exercised by
# `test/local_mesh_2d_gpu_test.mojo`, which builds an SSPRK3 stage out
# of the individual primitives to gate them in isolation.
# ----------------------------------------------------------------------

def rk_update_kernel_2d[NP: Int, NC: Int](
    q_a:     UnsafePointer[Float32, MutAnyOrigin],
    q_b:     UnsafePointer[Float32, MutAnyOrigin],
    rhs:     UnsafePointer[Float32, MutAnyOrigin],
    num_elements: Int,
    a: Float32, b: Float32, cc: Float32, dt: Float32,
    q_out:   UnsafePointer[Float32, MutAnyOrigin],
):
    var tid = Int(global_idx.x)
    var total = num_elements * NP * NC
    if tid >= total:
        return
    q_out[tid] = a * q_a[tid] + b * q_b[tid] + cc * dt * rhs[tid]


def launch_rk_update_2d[NP: Int, NC: Int](
    mut ctx: DeviceContext,
    q_a:     UnsafePointer[Float32, MutAnyOrigin],
    q_b:     UnsafePointer[Float32, MutAnyOrigin],
    rhs:     UnsafePointer[Float32, MutAnyOrigin],
    num_elements: Int,
    a: Float32, b: Float32, cc: Float32, dt: Float32,
    q_out:   UnsafePointer[Float32, MutAnyOrigin],
) raises:
    var total = num_elements * NP * NC
    comptime _kernel = rk_update_kernel_2d[NP, NC]
    ctx.enqueue_function[_kernel, _kernel](
        q_a, q_b, rhs, num_elements, a, b, cc, dt, q_out,
        grid_dim=ceildiv(total, 256),
        block_dim=256,
    )


