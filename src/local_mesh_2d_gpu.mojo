# ======================================================================
# Device mirror of LocalMesh2D -- the foundation of task #19
# ======================================================================
#
# The CPU mesh in `src/local_mesh_2d.mojo` is Float64 host-side.  A GPU
# solver wants Float32 device buffers (to match the 3D stack and keep
# shared-memory / register pressure in line).  `LocalMesh2DGpu[P]`
# takes a host `LocalMesh2D[P]` + a `DeviceContext`, converts the
# Float64 tables to Float32, and uploads everything.  Int32 tables
# (face topology, bc_type, etc.) transfer directly.
#
# No GPU kernels are defined here yet -- this module just gets the
# mesh data onto the device.  The follow-on work is a 2D analog of
# `src/solver.mojo::rk_stage_kernel` that reads these buffers and
# writes a Float32 q buffer.  See project_2d_triangles_scope.md for
# the broader roadmap.
#
# Buffer layout mirrors the host mesh exactly; no reshuffling:
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
from std.math import ceildiv
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
# Advection volume-rhs kernel (2D, scalar).
# ----------------------------------------------------------------------
# Computes, for each owned node (elem, i),
#
#   vol_c[elem, i] = sum_j sum_k D_ref[k, i, j] * (invJ[elem, k, :] . F(q_j))
#
# where F(q) = (vx * q, vy * q) for scalar linear advection.  One
# thread per (element, node) = 1 thread per nodal DOF.
#
# This is only the volume half of the DG rhs -- the face-flux half
# requires a two-pass kernel (gather fstar per face, then lift into
# each element).  Landing the volume kernel first validates the
# mesh/reference-operator plumbing end-to-end on device; the face
# pass follows.
# ----------------------------------------------------------------------

def advection_volume_rhs_kernel_2d[NP: Int](
    q:          UnsafePointer[Float32, MutAnyOrigin],
    elem_invJ:  UnsafePointer[Float32, MutAnyOrigin],
    D_ref:      UnsafePointer[Float32, MutAnyOrigin],
    num_elements: Int,
    vx: Float32, vy: Float32,
    vol_out:    UnsafePointer[Float32, MutAnyOrigin],
):
    var tid = Int(global_idx.x)
    var total = num_elements * NP
    if tid >= total:
        return
    var elem = tid // NP
    var i    = tid %  NP

    var iJ00 = elem_invJ[elem * 4 + 0]
    var iJ01 = elem_invJ[elem * 4 + 1]
    var iJ10 = elem_invJ[elem * 4 + 2]
    var iJ11 = elem_invJ[elem * 4 + 3]

    var vol_c: Float32 = 0.0
    for j in range(NP):
        var qj = q[elem * NP + j]
        var fx = vx * qj
        var fy = vy * qj
        var fr0 = iJ00 * fx + iJ01 * fy
        var fr1 = iJ10 * fx + iJ11 * fy
        var D_r = D_ref[0 * NP * NP + i * NP + j]
        var D_s = D_ref[1 * NP * NP + i * NP + j]
        vol_c += fr0 * D_r + fr1 * D_s

    vol_out[elem * NP + i] = vol_c


def launch_advection_volume_rhs_2d[NP: Int](
    mut ctx: DeviceContext,
    q:         UnsafePointer[Float32, MutAnyOrigin],
    elem_invJ: UnsafePointer[Float32, MutAnyOrigin],
    D_ref:     UnsafePointer[Float32, MutAnyOrigin],
    num_elements: Int,
    vx: Float32, vy: Float32,
    vol_out:   UnsafePointer[Float32, MutAnyOrigin],
) raises:
    var total = num_elements * NP
    comptime _kernel = advection_volume_rhs_kernel_2d[NP]
    ctx.enqueue_function[_kernel, _kernel](
        q, elem_invJ, D_ref, num_elements, vx, vy, vol_out,
        grid_dim=ceildiv(total, 256),
        block_dim=256,
    )


# ----------------------------------------------------------------------
# Advection face-flux kernel (2D, scalar).
# ----------------------------------------------------------------------
# Computes upwind fstar at every face-local slot:
#   fstar = vn * q_upwind
# where vn = v . n (side-0's outward normal).  On non-periodic BC
# faces we pick the ghost state by bc_type (zero for WALL/OUTFLOW
# incoming direction; `inflow_q` for BC_INFLOW).
#
# One thread per (face, slot) = num_faces * NFP_edge threads.
# ----------------------------------------------------------------------

def advection_face_flux_kernel_2d[NP: Int, NFP: Int](
    q:               UnsafePointer[Float32, MutAnyOrigin],
    face_elem:       UnsafePointer[Int32,   MutAnyOrigin],
    face_elem_node:  UnsafePointer[Int32,   MutAnyOrigin],
    face_normal:     UnsafePointer[Float32, MutAnyOrigin],
    face_bc_type:    UnsafePointer[Int32,   MutAnyOrigin],
    num_faces:       Int,
    vx: Float32, vy: Float32, inflow_q: Float32,
    fstar_out:       UnsafePointer[Float32, MutAnyOrigin],
):
    var tid = Int(global_idx.x)
    var total = num_faces * NFP
    if tid >= total:
        return
    var fid = tid // NFP
    var m   = tid %  NFP

    var nx = face_normal[fid * 2 + 0]
    var ny = face_normal[fid * 2 + 1]
    var vn = vx * nx + vy * ny
    var e_l = Int(face_elem[fid * 2 + 0])
    var n_l = Int(face_elem_node[(fid * 2 + 0) * NFP + m])
    var q_l = q[e_l * NP + n_l]
    var bc_type = face_bc_type[fid]

    var fstar: Float32
    if bc_type == BC_INTERIOR:
        var e_r = Int(face_elem[fid * 2 + 1])
        var n_r = Int(face_elem_node[(fid * 2 + 1) * NFP + m])
        var q_r = q[e_r * NP + n_r]
        if vn >= Float32(0.0):
            fstar = vn * q_l
        else:
            fstar = vn * q_r
    elif bc_type == BC_INFLOW:
        # Characteristic entering domain (vn < 0) picks user-set
        # inflow; exiting (vn >= 0) still upwinds from interior.
        if vn >= Float32(0.0):
            fstar = vn * q_l
        else:
            fstar = vn * inflow_q
    else:
        # BC_WALL / BC_OUTFLOW / default: zero-Dirichlet ghost.
        if vn >= Float32(0.0):
            fstar = vn * q_l
        else:
            fstar = Float32(0.0)
    fstar_out[fid * NFP + m] = fstar


def launch_advection_face_flux_2d[NP: Int, NFP: Int](
    mut ctx: DeviceContext,
    q:              UnsafePointer[Float32, MutAnyOrigin],
    face_elem:      UnsafePointer[Int32,   MutAnyOrigin],
    face_elem_node: UnsafePointer[Int32,   MutAnyOrigin],
    face_normal:    UnsafePointer[Float32, MutAnyOrigin],
    face_bc_type:   UnsafePointer[Int32,   MutAnyOrigin],
    num_faces:      Int,
    vx: Float32, vy: Float32, inflow_q: Float32,
    fstar_out:      UnsafePointer[Float32, MutAnyOrigin],
) raises:
    var total = num_faces * NFP
    comptime _kernel = advection_face_flux_kernel_2d[NP, NFP]
    ctx.enqueue_function[_kernel, _kernel](
        q, face_elem, face_elem_node, face_normal, face_bc_type,
        num_faces, vx, vy, inflow_q, fstar_out,
        grid_dim=ceildiv(total, 256),
        block_dim=256,
    )


# ----------------------------------------------------------------------
# Lift-combine kernel: finishes the DG rhs.
# ----------------------------------------------------------------------
# Given the volume-integral output `vol_c[elem, i]` and the per-face
# numerical flux `fstar[fid, m]`, write the complete rhs into
# `rhs[elem, i] = vol_c[elem, i] - inv_2A[elem] * face_c[elem, i]`,
# where
#
#   face_c[elem, i] = sum_{lf=0..2} sum_{m=0..P} sign(side)
#                       * face_length[fid(elem, lf)]
#                       * Lift_ref[lf, i, r(elem, lf, m)]
#                       * fstar[fid(elem, lf), m]
#
# `r = elem_canon_to_ref[(elem * 3 + lf) * NFP + m]` remaps the
# canonical face-local slot to the ref-edge slot the Lift_ref entry
# indexes.  One thread per (elem, i) = num_elements * NP threads.
# ----------------------------------------------------------------------

def advection_lift_combine_kernel_2d[NP: Int, NFP: Int](
    vol_c:             UnsafePointer[Float32, MutAnyOrigin],
    fstar:             UnsafePointer[Float32, MutAnyOrigin],
    elem_inv_2A:       UnsafePointer[Float32, MutAnyOrigin],
    elem_faces:        UnsafePointer[Int32,   MutAnyOrigin],
    elem_face_side:    UnsafePointer[Int32,   MutAnyOrigin],
    elem_canon_to_ref: UnsafePointer[Int32,   MutAnyOrigin],
    face_length:       UnsafePointer[Float32, MutAnyOrigin],
    Lift_ref:          UnsafePointer[Float32, MutAnyOrigin],
    num_elements:      Int,
    rhs_out:           UnsafePointer[Float32, MutAnyOrigin],
):
    var tid = Int(global_idx.x)
    var total = num_elements * NP
    if tid >= total:
        return
    var elem = tid // NP
    var i    = tid %  NP

    var inv_2A = elem_inv_2A[elem]
    var face_c: Float32 = 0.0
    for lf in range(3):
        var fid = Int(elem_faces[elem * 3 + lf])
        var side = Int(elem_face_side[elem * 3 + lf])
        var sign: Float32 = Float32(1.0) if side == 0 else Float32(-1.0)
        var flen = face_length[fid]
        for m in range(NFP):
            var r = Int(
                elem_canon_to_ref[(elem * 3 + lf) * NFP + m]
            )
            var Lim = Lift_ref[lf * NP * NFP + i * NFP + r]
            face_c += sign * flen * Lim * fstar[fid * NFP + m]

    rhs_out[elem * NP + i] = vol_c[elem * NP + i] - inv_2A * face_c


def launch_advection_lift_combine_2d[NP: Int, NFP: Int](
    mut ctx: DeviceContext,
    vol_c:             UnsafePointer[Float32, MutAnyOrigin],
    fstar:             UnsafePointer[Float32, MutAnyOrigin],
    elem_inv_2A:       UnsafePointer[Float32, MutAnyOrigin],
    elem_faces:        UnsafePointer[Int32,   MutAnyOrigin],
    elem_face_side:    UnsafePointer[Int32,   MutAnyOrigin],
    elem_canon_to_ref: UnsafePointer[Int32,   MutAnyOrigin],
    face_length:       UnsafePointer[Float32, MutAnyOrigin],
    Lift_ref:          UnsafePointer[Float32, MutAnyOrigin],
    num_elements:      Int,
    rhs_out:           UnsafePointer[Float32, MutAnyOrigin],
) raises:
    var total = num_elements * NP
    comptime _kernel = advection_lift_combine_kernel_2d[NP, NFP]
    ctx.enqueue_function[_kernel, _kernel](
        vol_c, fstar, elem_inv_2A, elem_faces, elem_face_side,
        elem_canon_to_ref, face_length, Lift_ref,
        num_elements, rhs_out,
        grid_dim=ceildiv(total, 256),
        block_dim=256,
    )
