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
# Fused advection volume + lift + RK update (2D, NC=1).
# ----------------------------------------------------------------------
# Per-(elem, node) thread.  Combines what was three separate launches
# into two: the face-flux kernel still runs first (its data layout is
# per-face, not per-node, so different parallelism), but the volume
# RHS computation is no longer materialised to global memory -- it's
# computed locally and immediately fed into the lift + RK update step.
#
# Saves: one kernel launch per RK stage (~10us per launch * 7000+
# instances on the P=3 advection benchmark) and the global-memory
# round-trip of vol_c.
#
# Invariant: the face-flux kernel must complete before this kernel
# starts (same stream serialisation as before).
# ----------------------------------------------------------------------

def advection_vol_lift_combine_rk_kernel_2d[NP: Int, NFP: Int](
    q_in:              UnsafePointer[Float32, MutAnyOrigin],
    elem_invJ:         UnsafePointer[Float32, MutAnyOrigin],
    D_ref:             UnsafePointer[Float32, MutAnyOrigin],
    fstar:             UnsafePointer[Float32, MutAnyOrigin],
    elem_inv_2A:       UnsafePointer[Float32, MutAnyOrigin],
    elem_faces:        UnsafePointer[Int32,   MutAnyOrigin],
    elem_face_side:    UnsafePointer[Int32,   MutAnyOrigin],
    elem_canon_to_ref: UnsafePointer[Int32,   MutAnyOrigin],
    face_length:       UnsafePointer[Float32, MutAnyOrigin],
    Lift_ref:          UnsafePointer[Float32, MutAnyOrigin],
    q_a:               UnsafePointer[Float32, MutAnyOrigin],
    q_b:               UnsafePointer[Float32, MutAnyOrigin],
    num_elements:      Int,
    vx: Float32, vy: Float32,
    a: Float32, b: Float32, cc: Float32, dt: Float32,
    q_out:             UnsafePointer[Float32, MutAnyOrigin],
):
    var tid = Int(global_idx.x)
    var total = num_elements * NP
    if tid >= total:
        return
    var elem = tid // NP
    var i    = tid %  NP

    # ---- Volume RHS contribution (computed locally, no global write).
    var iJ00 = elem_invJ[elem * 4 + 0]
    var iJ01 = elem_invJ[elem * 4 + 1]
    var iJ10 = elem_invJ[elem * 4 + 2]
    var iJ11 = elem_invJ[elem * 4 + 3]
    var vol_c: Float32 = 0.0
    for j in range(NP):
        var qj = q_in[elem * NP + j]
        var fx = vx * qj
        var fy = vy * qj
        var fr0 = iJ00 * fx + iJ01 * fy
        var fr1 = iJ10 * fx + iJ11 * fy
        var D_r = D_ref[0 * NP * NP + i * NP + j]
        var D_s = D_ref[1 * NP * NP + i * NP + j]
        vol_c += fr0 * D_r + fr1 * D_s

    # ---- Lift contribution (NC=1 -> fstar indexed without c).
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

    # ---- Combine + RK update.
    var idx = elem * NP + i
    var rhs_val = vol_c - inv_2A * face_c
    q_out[idx] = a * q_a[idx] + b * q_b[idx] + cc * dt * rhs_val


def launch_advection_vol_lift_2d[NP: Int, NFP: Int](
    mut ctx: DeviceContext,
    q_in:              UnsafePointer[Float32, MutAnyOrigin],
    elem_invJ:         UnsafePointer[Float32, MutAnyOrigin],
    D_ref:             UnsafePointer[Float32, MutAnyOrigin],
    fstar:             UnsafePointer[Float32, MutAnyOrigin],
    elem_inv_2A:       UnsafePointer[Float32, MutAnyOrigin],
    elem_faces:        UnsafePointer[Int32,   MutAnyOrigin],
    elem_face_side:    UnsafePointer[Int32,   MutAnyOrigin],
    elem_canon_to_ref: UnsafePointer[Int32,   MutAnyOrigin],
    face_length:       UnsafePointer[Float32, MutAnyOrigin],
    Lift_ref:          UnsafePointer[Float32, MutAnyOrigin],
    q_a:               UnsafePointer[Float32, MutAnyOrigin],
    q_b:               UnsafePointer[Float32, MutAnyOrigin],
    num_elements:      Int,
    vx: Float32, vy: Float32,
    a: Float32, b: Float32, cc: Float32, dt: Float32,
    q_out:             UnsafePointer[Float32, MutAnyOrigin],
) raises:
    var total = num_elements * NP
    comptime _kernel = advection_vol_lift_combine_rk_kernel_2d[NP, NFP]
    ctx.enqueue_function[_kernel, _kernel](
        q_in, elem_invJ, D_ref, fstar,
        elem_inv_2A, elem_faces, elem_face_side, elem_canon_to_ref,
        face_length, Lift_ref, q_a, q_b,
        num_elements, vx, vy, a, b, cc, dt, q_out,
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


# ----------------------------------------------------------------------
# RK-update combiner: q_out = a * q_a + b * q_b + cc * dt * rhs.
# ----------------------------------------------------------------------
# Matches the SSPRK3 weighted combination pattern used in the 3D
# Solver's rk_stage_kernel (separated here since the 2D pipeline
# doesn't yet fuse rhs + update into a single shared-memory kernel).
# One thread per nodal DOF; stateless, just arithmetic.
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


# ----------------------------------------------------------------------
# Full advection RK-stage orchestration (3 rhs kernels + 1 update).
# ----------------------------------------------------------------------
# Wraps the volume / face-flux / lift-combine / rk-update chain so a
# driver only has to provide the per-stage (a, b, cc) weights, the dt,
# and the q_in / q_a / q_b / q_out buffers.  Uses the caller-supplied
# scratch buffers for vol_c, fstar, and rhs so the orchestration is
# allocation-free on the hot path.
#
# Signature mirrors what the eventual Solver2D struct will expose
# internally.  Calling it three times with the SSPRK3 weights
# reproduces the 2D GPU analog of the 3D Solver.step_ssprk3 loop.
# ----------------------------------------------------------------------

def advection_rk_stage_2d[P: Int](
    mut ctx: DeviceContext,
    mesh: LocalMesh2DGpu[P],
    Lift_ref: UnsafePointer[Float32, MutAnyOrigin],
    D_ref:    UnsafePointer[Float32, MutAnyOrigin],
    q_in:     UnsafePointer[Float32, MutAnyOrigin],
    q_a:      UnsafePointer[Float32, MutAnyOrigin],
    q_b:      UnsafePointer[Float32, MutAnyOrigin],
    q_out:    UnsafePointer[Float32, MutAnyOrigin],
    vol_scratch:   UnsafePointer[Float32, MutAnyOrigin],
    fstar_scratch: UnsafePointer[Float32, MutAnyOrigin],
    rhs_scratch:   UnsafePointer[Float32, MutAnyOrigin],
    vx: Float32, vy: Float32, inflow_q: Float32,
    a: Float32, b: Float32, cc: Float32, dt: Float32,
) raises:
    # Two launches per stage (down from three): face flux, then a fused
    # vol+lift+RK kernel that computes the volume RHS locally without
    # round-tripping through global vol_scratch.  vol_scratch and
    # rhs_scratch are unused on the fused path (kept in the signature
    # for backward compatibility with drivers that allocated them).
    _ = vol_scratch
    _ = rhs_scratch
    comptime NP = num_tri_nodes_2d(P)
    comptime NFP = num_edge_nodes(P)
    launch_advection_face_flux_2d[NP, NFP](
        ctx, q_in,
        mesh.d_face_elem.unsafe_ptr(),
        mesh.d_face_elem_node.unsafe_ptr(),
        mesh.d_face_normal.unsafe_ptr(),
        mesh.d_face_bc_type.unsafe_ptr(),
        mesh.num_faces, vx, vy, inflow_q, fstar_scratch,
    )
    launch_advection_vol_lift_2d[NP, NFP](
        ctx, q_in, mesh.d_elem_invJ.unsafe_ptr(), D_ref,
        fstar_scratch,
        mesh.d_elem_inv_2A.unsafe_ptr(),
        mesh.d_elem_faces.unsafe_ptr(),
        mesh.d_elem_face_side.unsafe_ptr(),
        mesh.d_elem_canon_to_ref.unsafe_ptr(),
        mesh.d_face_length.unsafe_ptr(),
        Lift_ref, q_a, q_b,
        mesh.num_elements, vx, vy, a, b, cc, dt, q_out,
    )


# ----------------------------------------------------------------------
# Multi-component lift-combine.
# ----------------------------------------------------------------------
# Same structure as `advection_lift_combine_kernel_2d` but indexes q /
# vol_c / rhs / fstar with a trailing component axis (NC).  Used by the
# Euler / SW / MHD GPU paths where the DG rhs has >1 conservative
# variable per node.  One thread per (elem, i, c) = num_elements*NP*NC.
# ----------------------------------------------------------------------

def lift_combine_kernel_2d[NP: Int, NFP: Int, NC: Int](
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
    var total = num_elements * NP * NC
    if tid >= total:
        return
    var elem = tid // (NP * NC)
    var rem  = tid %  (NP * NC)
    var i    = rem // NC
    var c    = rem %  NC

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
            face_c += sign * flen * Lim * fstar[(fid * NFP + m) * NC + c]

    var idx = (elem * NP + i) * NC + c
    rhs_out[idx] = vol_c[idx] - inv_2A * face_c


def launch_lift_combine_2d[NP: Int, NFP: Int, NC: Int](
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
    var total = num_elements * NP * NC
    comptime _kernel = lift_combine_kernel_2d[NP, NFP, NC]
    ctx.enqueue_function[_kernel, _kernel](
        vol_c, fstar, elem_inv_2A, elem_faces, elem_face_side,
        elem_canon_to_ref, face_length, Lift_ref,
        num_elements, rhs_out,
        grid_dim=ceildiv(total, 256),
        block_dim=256,
    )


# ----------------------------------------------------------------------
# Fused lift-combine + RK update.
# ----------------------------------------------------------------------
# Same per-(elem, i, c) thread mapping as `lift_combine_kernel_2d`,
# but instead of writing rhs to scratch and launching a separate
# `rk_update_kernel_2d`, this kernel inlines the SSPRK3 weighted
# combination on-thread:
#   q_out[idx] = a*q_a[idx] + b*q_b[idx] + cc*dt*rhs_val
# Each fused launch replaces two prior launches per RK stage,
# cutting the 4-launch-per-stage 2D pipeline to 3 launches per stage.
# Identical numerics to the unfused chain to Float32 precision.
# ----------------------------------------------------------------------

def lift_combine_rk_kernel_2d[NP: Int, NFP: Int, NC: Int](
    vol_c:             UnsafePointer[Float32, MutAnyOrigin],
    fstar:             UnsafePointer[Float32, MutAnyOrigin],
    elem_inv_2A:       UnsafePointer[Float32, MutAnyOrigin],
    elem_faces:        UnsafePointer[Int32,   MutAnyOrigin],
    elem_face_side:    UnsafePointer[Int32,   MutAnyOrigin],
    elem_canon_to_ref: UnsafePointer[Int32,   MutAnyOrigin],
    face_length:       UnsafePointer[Float32, MutAnyOrigin],
    Lift_ref:          UnsafePointer[Float32, MutAnyOrigin],
    q_a:               UnsafePointer[Float32, MutAnyOrigin],
    q_b:               UnsafePointer[Float32, MutAnyOrigin],
    num_elements:      Int,
    a: Float32, b: Float32, cc: Float32, dt: Float32,
    q_out:             UnsafePointer[Float32, MutAnyOrigin],
):
    var tid = Int(global_idx.x)
    var total = num_elements * NP * NC
    if tid >= total:
        return
    var elem = tid // (NP * NC)
    var rem  = tid %  (NP * NC)
    var i    = rem // NC
    var c    = rem %  NC

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
            face_c += sign * flen * Lim * fstar[(fid * NFP + m) * NC + c]

    var idx = (elem * NP + i) * NC + c
    var rhs_val = vol_c[idx] - inv_2A * face_c
    q_out[idx] = a * q_a[idx] + b * q_b[idx] + cc * dt * rhs_val


def launch_lift_combine_rk_2d[NP: Int, NFP: Int, NC: Int](
    mut ctx: DeviceContext,
    vol_c:             UnsafePointer[Float32, MutAnyOrigin],
    fstar:             UnsafePointer[Float32, MutAnyOrigin],
    elem_inv_2A:       UnsafePointer[Float32, MutAnyOrigin],
    elem_faces:        UnsafePointer[Int32,   MutAnyOrigin],
    elem_face_side:    UnsafePointer[Int32,   MutAnyOrigin],
    elem_canon_to_ref: UnsafePointer[Int32,   MutAnyOrigin],
    face_length:       UnsafePointer[Float32, MutAnyOrigin],
    Lift_ref:          UnsafePointer[Float32, MutAnyOrigin],
    q_a:               UnsafePointer[Float32, MutAnyOrigin],
    q_b:               UnsafePointer[Float32, MutAnyOrigin],
    num_elements:      Int,
    a: Float32, b: Float32, cc: Float32, dt: Float32,
    q_out:             UnsafePointer[Float32, MutAnyOrigin],
) raises:
    var total = num_elements * NP * NC
    comptime _kernel = lift_combine_rk_kernel_2d[NP, NFP, NC]
    ctx.enqueue_function[_kernel, _kernel](
        vol_c, fstar, elem_inv_2A, elem_faces, elem_face_side,
        elem_canon_to_ref, face_length, Lift_ref,
        q_a, q_b, num_elements, a, b, cc, dt, q_out,
        grid_dim=ceildiv(total, 256),
        block_dim=256,
    )


# ----------------------------------------------------------------------
# Euler2D volume RHS (4 components).
# ----------------------------------------------------------------------
# For each owned (elem, i) writes vol_c[elem, i, c] for c = 0..3, where
# vol_c = sum_j (D_r[i, j] * (invJ . F(q_j))_0 + D_s[i, j] * (..)_1) and
# F(q) is the 2x4 Euler flux:
#   rho, mx, my, E  ->
#     Fx = (mx, mx*u+p, mx*v,   u*(E+p))
#     Fy = (my, my*u,   my*v+p, v*(E+p))
# with u=mx/rho, v=my/rho, p = (gamma-1)*(E - 0.5*rho*(u^2+v^2)).
# One thread per (elem, i) computes all 4 components together to amortise
# the inner-loop q_j read + flux computation.
# ----------------------------------------------------------------------

def euler_volume_rhs_kernel_2d[NP: Int](
    q:            UnsafePointer[Float32, MutAnyOrigin],
    elem_invJ:    UnsafePointer[Float32, MutAnyOrigin],
    D_ref:        UnsafePointer[Float32, MutAnyOrigin],
    num_elements: Int,
    gamma:        Float32,
    min_density:  Float32,
    min_pressure: Float32,
    vol_out:      UnsafePointer[Float32, MutAnyOrigin],
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

    var acc0: Float32 = 0.0
    var acc1: Float32 = 0.0
    var acc2: Float32 = 0.0
    var acc3: Float32 = 0.0

    for j in range(NP):
        var base = (elem * NP + j) * 4
        var rho = q[base + 0]
        if rho < min_density:
            rho = min_density
        var mx = q[base + 1]
        var my = q[base + 2]
        var E  = q[base + 3]
        var u = mx / rho
        var v = my / rho
        var ke = Float32(0.5) * rho * (u * u + v * v)
        var p = (gamma - Float32(1.0)) * (E - ke)
        if p < min_pressure:
            p = min_pressure

        var Fx0 = mx
        var Fx1 = mx * u + p
        var Fx2 = mx * v
        var Fx3 = u * (E + p)
        var Fy0 = my
        var Fy1 = my * u
        var Fy2 = my * v + p
        var Fy3 = v * (E + p)

        var D_r = D_ref[0 * NP * NP + i * NP + j]
        var D_s = D_ref[1 * NP * NP + i * NP + j]

        # (invJ . (Fx, Fy))_0 = iJ00*Fx + iJ01*Fy
        # (invJ . (Fx, Fy))_1 = iJ10*Fx + iJ11*Fy
        acc0 += (iJ00 * Fx0 + iJ01 * Fy0) * D_r + (iJ10 * Fx0 + iJ11 * Fy0) * D_s
        acc1 += (iJ00 * Fx1 + iJ01 * Fy1) * D_r + (iJ10 * Fx1 + iJ11 * Fy1) * D_s
        acc2 += (iJ00 * Fx2 + iJ01 * Fy2) * D_r + (iJ10 * Fx2 + iJ11 * Fy2) * D_s
        acc3 += (iJ00 * Fx3 + iJ01 * Fy3) * D_r + (iJ10 * Fx3 + iJ11 * Fy3) * D_s

    var out = (elem * NP + i) * 4
    vol_out[out + 0] = acc0
    vol_out[out + 1] = acc1
    vol_out[out + 2] = acc2
    vol_out[out + 3] = acc3


def launch_euler_volume_rhs_2d[NP: Int](
    mut ctx: DeviceContext,
    q:            UnsafePointer[Float32, MutAnyOrigin],
    elem_invJ:    UnsafePointer[Float32, MutAnyOrigin],
    D_ref:        UnsafePointer[Float32, MutAnyOrigin],
    num_elements: Int,
    gamma:        Float32,
    min_density:  Float32,
    min_pressure: Float32,
    vol_out:      UnsafePointer[Float32, MutAnyOrigin],
) raises:
    var total = num_elements * NP
    comptime _kernel = euler_volume_rhs_kernel_2d[NP]
    ctx.enqueue_function[_kernel, _kernel](
        q, elem_invJ, D_ref, num_elements,
        gamma, min_density, min_pressure, vol_out,
        grid_dim=ceildiv(total, 256),
        block_dim=256,
    )


# ----------------------------------------------------------------------
# Fused Euler volume + lift + RK update (2D, NC=4).
# ----------------------------------------------------------------------
# Per-(elem, node) thread.  Same fusion pattern as
# advection_vol_lift_combine_rk_kernel_2d but for the 4-component
# Euler state.  Eliminates the global-memory round-trip through vol_c
# and one of the three kernel launches per RK stage.
# ----------------------------------------------------------------------

def euler_vol_lift_combine_rk_kernel_2d[NP: Int, NFP: Int](
    q_in:              UnsafePointer[Float32, MutAnyOrigin],
    elem_invJ:         UnsafePointer[Float32, MutAnyOrigin],
    D_ref:             UnsafePointer[Float32, MutAnyOrigin],
    fstar:             UnsafePointer[Float32, MutAnyOrigin],
    elem_inv_2A:       UnsafePointer[Float32, MutAnyOrigin],
    elem_faces:        UnsafePointer[Int32,   MutAnyOrigin],
    elem_face_side:    UnsafePointer[Int32,   MutAnyOrigin],
    elem_canon_to_ref: UnsafePointer[Int32,   MutAnyOrigin],
    face_length:       UnsafePointer[Float32, MutAnyOrigin],
    Lift_ref:          UnsafePointer[Float32, MutAnyOrigin],
    q_a:               UnsafePointer[Float32, MutAnyOrigin],
    q_b:               UnsafePointer[Float32, MutAnyOrigin],
    num_elements:      Int,
    gamma:             Float32,
    min_density:       Float32,
    min_pressure:      Float32,
    a: Float32, b: Float32, cc: Float32, dt: Float32,
    q_out:             UnsafePointer[Float32, MutAnyOrigin],
):
    var tid = Int(global_idx.x)
    var total = num_elements * NP
    if tid >= total:
        return
    var elem = tid // NP
    var i    = tid %  NP

    # ---- Volume RHS contribution (4 components, computed locally).
    var iJ00 = elem_invJ[elem * 4 + 0]
    var iJ01 = elem_invJ[elem * 4 + 1]
    var iJ10 = elem_invJ[elem * 4 + 2]
    var iJ11 = elem_invJ[elem * 4 + 3]
    var acc0: Float32 = 0.0
    var acc1: Float32 = 0.0
    var acc2: Float32 = 0.0
    var acc3: Float32 = 0.0
    for j in range(NP):
        var base = (elem * NP + j) * 4
        var rho = q_in[base + 0]
        if rho < min_density:
            rho = min_density
        var mx = q_in[base + 1]
        var my = q_in[base + 2]
        var E  = q_in[base + 3]
        var u = mx / rho
        var v = my / rho
        var ke = Float32(0.5) * rho * (u * u + v * v)
        var p = (gamma - Float32(1.0)) * (E - ke)
        if p < min_pressure:
            p = min_pressure
        var Fx0 = mx
        var Fx1 = mx * u + p
        var Fx2 = mx * v
        var Fx3 = u * (E + p)
        var Fy0 = my
        var Fy1 = my * u
        var Fy2 = my * v + p
        var Fy3 = v * (E + p)
        var D_r = D_ref[0 * NP * NP + i * NP + j]
        var D_s = D_ref[1 * NP * NP + i * NP + j]
        acc0 += (iJ00 * Fx0 + iJ01 * Fy0) * D_r + (iJ10 * Fx0 + iJ11 * Fy0) * D_s
        acc1 += (iJ00 * Fx1 + iJ01 * Fy1) * D_r + (iJ10 * Fx1 + iJ11 * Fy1) * D_s
        acc2 += (iJ00 * Fx2 + iJ01 * Fy2) * D_r + (iJ10 * Fx2 + iJ11 * Fy2) * D_s
        acc3 += (iJ00 * Fx3 + iJ01 * Fy3) * D_r + (iJ10 * Fx3 + iJ11 * Fy3) * D_s

    # ---- Lift contribution (NC=4, expanded inline).
    var inv_2A = elem_inv_2A[elem]
    var face0: Float32 = 0.0
    var face1: Float32 = 0.0
    var face2: Float32 = 0.0
    var face3: Float32 = 0.0
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
            var sLf = sign * flen * Lim
            var fbase = (fid * NFP + m) * 4
            face0 += sLf * fstar[fbase + 0]
            face1 += sLf * fstar[fbase + 1]
            face2 += sLf * fstar[fbase + 2]
            face3 += sLf * fstar[fbase + 3]

    # ---- Combine + RK update for all 4 components.
    var idx = (elem * NP + i) * 4
    var rhs0 = acc0 - inv_2A * face0
    var rhs1 = acc1 - inv_2A * face1
    var rhs2 = acc2 - inv_2A * face2
    var rhs3 = acc3 - inv_2A * face3
    q_out[idx + 0] = a * q_a[idx + 0] + b * q_b[idx + 0] + cc * dt * rhs0
    q_out[idx + 1] = a * q_a[idx + 1] + b * q_b[idx + 1] + cc * dt * rhs1
    q_out[idx + 2] = a * q_a[idx + 2] + b * q_b[idx + 2] + cc * dt * rhs2
    q_out[idx + 3] = a * q_a[idx + 3] + b * q_b[idx + 3] + cc * dt * rhs3


def launch_euler_vol_lift_2d[NP: Int, NFP: Int](
    mut ctx: DeviceContext,
    q_in:              UnsafePointer[Float32, MutAnyOrigin],
    elem_invJ:         UnsafePointer[Float32, MutAnyOrigin],
    D_ref:             UnsafePointer[Float32, MutAnyOrigin],
    fstar:             UnsafePointer[Float32, MutAnyOrigin],
    elem_inv_2A:       UnsafePointer[Float32, MutAnyOrigin],
    elem_faces:        UnsafePointer[Int32,   MutAnyOrigin],
    elem_face_side:    UnsafePointer[Int32,   MutAnyOrigin],
    elem_canon_to_ref: UnsafePointer[Int32,   MutAnyOrigin],
    face_length:       UnsafePointer[Float32, MutAnyOrigin],
    Lift_ref:          UnsafePointer[Float32, MutAnyOrigin],
    q_a:               UnsafePointer[Float32, MutAnyOrigin],
    q_b:               UnsafePointer[Float32, MutAnyOrigin],
    num_elements:      Int,
    gamma:             Float32,
    min_density:       Float32,
    min_pressure:      Float32,
    a: Float32, b: Float32, cc: Float32, dt: Float32,
    q_out:             UnsafePointer[Float32, MutAnyOrigin],
) raises:
    var total = num_elements * NP
    comptime _kernel = euler_vol_lift_combine_rk_kernel_2d[NP, NFP]
    ctx.enqueue_function[_kernel, _kernel](
        q_in, elem_invJ, D_ref, fstar,
        elem_inv_2A, elem_faces, elem_face_side, elem_canon_to_ref,
        face_length, Lift_ref, q_a, q_b,
        num_elements, gamma, min_density, min_pressure,
        a, b, cc, dt, q_out,
        grid_dim=ceildiv(total, 256),
        block_dim=256,
    )


# ----------------------------------------------------------------------
# Euler2D face flux (Rusanov / Lax-Friedrichs, 4 components).
# ----------------------------------------------------------------------
# Computes fstar[fid, m, 0..3] at every face-local slot using the
# Rusanov flux F* = 0.5(F_L.n + F_R.n) - 0.5 alpha (q_R - q_L) with
# alpha = max(|v.n|+c) over the two sides.  Ghost state for non-interior
# faces mirrors the CPU Euler2D.boundary_flux:
#   BC_WALL    -> reflect normal momentum
#   BC_INFLOW  -> user-set (inflow_rho, inflow_rhou, inflow_rhov, inflow_E)
#   BC_OUTFLOW -> zero-gradient ghost (q_g = q_int)
# One thread per (face, slot) = num_faces * NFP threads.
# ----------------------------------------------------------------------

def euler_face_flux_kernel_2d[NP: Int, NFP: Int](
    q:              UnsafePointer[Float32, MutAnyOrigin],
    face_elem:      UnsafePointer[Int32,   MutAnyOrigin],
    face_elem_node: UnsafePointer[Int32,   MutAnyOrigin],
    face_normal:    UnsafePointer[Float32, MutAnyOrigin],
    face_bc_type:   UnsafePointer[Int32,   MutAnyOrigin],
    num_faces:      Int,
    gamma:          Float32,
    min_density:    Float32,
    min_pressure:   Float32,
    inflow_rho:     Float32,
    inflow_rhou:    Float32,
    inflow_rhov:    Float32,
    inflow_E:       Float32,
    fstar_out:      UnsafePointer[Float32, MutAnyOrigin],
):
    var tid = Int(global_idx.x)
    var total = num_faces * NFP
    if tid >= total:
        return
    var fid = tid // NFP
    var m   = tid %  NFP

    var nx = face_normal[fid * 2 + 0]
    var ny = face_normal[fid * 2 + 1]
    var bc_type = face_bc_type[fid]

    var e_l = Int(face_elem[fid * 2 + 0])
    var n_l = Int(face_elem_node[(fid * 2 + 0) * NFP + m])
    var l_off = (e_l * NP + n_l) * 4
    var qL0 = q[l_off + 0]
    var qL1 = q[l_off + 1]
    var qL2 = q[l_off + 2]
    var qL3 = q[l_off + 3]

    var qR0: Float32
    var qR1: Float32
    var qR2: Float32
    var qR3: Float32
    if bc_type == BC_INTERIOR:
        var e_r = Int(face_elem[fid * 2 + 1])
        var n_r = Int(face_elem_node[(fid * 2 + 1) * NFP + m])
        var r_off = (e_r * NP + n_r) * 4
        qR0 = q[r_off + 0]
        qR1 = q[r_off + 1]
        qR2 = q[r_off + 2]
        qR3 = q[r_off + 3]
    elif bc_type == BC_WALL:
        var m_n = qL1 * nx + qL2 * ny
        qR0 = qL0
        qR1 = qL1 - Float32(2.0) * m_n * nx
        qR2 = qL2 - Float32(2.0) * m_n * ny
        qR3 = qL3
    elif bc_type == BC_INFLOW:
        qR0 = inflow_rho
        qR1 = inflow_rhou
        qR2 = inflow_rhov
        qR3 = inflow_E
    else:
        # BC_OUTFLOW default: zero-gradient
        qR0 = qL0
        qR1 = qL1
        qR2 = qL2
        qR3 = qL3

    # Left internal flux + wave speed.
    var rhoL = qL0
    if rhoL < min_density:
        rhoL = min_density
    var uL = qL1 / rhoL
    var vL = qL2 / rhoL
    var keL = Float32(0.5) * rhoL * (uL * uL + vL * vL)
    var pL = (gamma - Float32(1.0)) * (qL3 - keL)
    if pL < min_pressure:
        pL = min_pressure
    var cL = sqrt(gamma * pL / rhoL)
    var speedL = sqrt(uL * uL + vL * vL) + cL

    var FxL0 = qL1
    var FxL1 = qL1 * uL + pL
    var FxL2 = qL1 * vL
    var FxL3 = uL * (qL3 + pL)
    var FyL0 = qL2
    var FyL1 = qL2 * uL
    var FyL2 = qL2 * vL + pL
    var FyL3 = vL * (qL3 + pL)

    # Right internal flux + wave speed.
    var rhoR = qR0
    if rhoR < min_density:
        rhoR = min_density
    var uR = qR1 / rhoR
    var vR = qR2 / rhoR
    var keR = Float32(0.5) * rhoR * (uR * uR + vR * vR)
    var pR = (gamma - Float32(1.0)) * (qR3 - keR)
    if pR < min_pressure:
        pR = min_pressure
    var cR = sqrt(gamma * pR / rhoR)
    var speedR = sqrt(uR * uR + vR * vR) + cR

    var FxR0 = qR1
    var FxR1 = qR1 * uR + pR
    var FxR2 = qR1 * vR
    var FxR3 = uR * (qR3 + pR)
    var FyR0 = qR2
    var FyR1 = qR2 * uR
    var FyR2 = qR2 * vR + pR
    var FyR3 = vR * (qR3 + pR)

    var alpha: Float32 = speedL if speedL > speedR else speedR
    var half = Float32(0.5)
    var out = (fid * NFP + m) * 4
    fstar_out[out + 0] = half * ((FxL0 + FxR0) * nx + (FyL0 + FyR0) * ny) \
                         - half * alpha * (qR0 - qL0)
    fstar_out[out + 1] = half * ((FxL1 + FxR1) * nx + (FyL1 + FyR1) * ny) \
                         - half * alpha * (qR1 - qL1)
    fstar_out[out + 2] = half * ((FxL2 + FxR2) * nx + (FyL2 + FyR2) * ny) \
                         - half * alpha * (qR2 - qL2)
    fstar_out[out + 3] = half * ((FxL3 + FxR3) * nx + (FyL3 + FyR3) * ny) \
                         - half * alpha * (qR3 - qL3)


def launch_euler_face_flux_2d[NP: Int, NFP: Int](
    mut ctx: DeviceContext,
    q:              UnsafePointer[Float32, MutAnyOrigin],
    face_elem:      UnsafePointer[Int32,   MutAnyOrigin],
    face_elem_node: UnsafePointer[Int32,   MutAnyOrigin],
    face_normal:    UnsafePointer[Float32, MutAnyOrigin],
    face_bc_type:   UnsafePointer[Int32,   MutAnyOrigin],
    num_faces:      Int,
    gamma:          Float32,
    min_density:    Float32,
    min_pressure:   Float32,
    inflow_rho:     Float32,
    inflow_rhou:    Float32,
    inflow_rhov:    Float32,
    inflow_E:       Float32,
    fstar_out:      UnsafePointer[Float32, MutAnyOrigin],
) raises:
    var total = num_faces * NFP
    comptime _kernel = euler_face_flux_kernel_2d[NP, NFP]
    ctx.enqueue_function[_kernel, _kernel](
        q, face_elem, face_elem_node, face_normal, face_bc_type,
        num_faces,
        gamma, min_density, min_pressure,
        inflow_rho, inflow_rhou, inflow_rhov, inflow_E,
        fstar_out,
        grid_dim=ceildiv(total, 256),
        block_dim=256,
    )


# ----------------------------------------------------------------------
# Euler2D HLLC face flux (Toro 1994, 4 components).
# ----------------------------------------------------------------------
# Same signature as the Rusanov kernel; resolves contact discontinuities
# exactly via the three-wave Riemann structure with the middle contact
# speed S_star derived from pressure continuity.  Ghost-state plumbing
# (BC_WALL / BC_INFLOW / BC_OUTFLOW) is identical to the Rusanov kernel
# -- only the interior flux differs.
# ----------------------------------------------------------------------

def euler_face_flux_hllc_kernel_2d[NP: Int, NFP: Int](
    q:              UnsafePointer[Float32, MutAnyOrigin],
    face_elem:      UnsafePointer[Int32,   MutAnyOrigin],
    face_elem_node: UnsafePointer[Int32,   MutAnyOrigin],
    face_normal:    UnsafePointer[Float32, MutAnyOrigin],
    face_bc_type:   UnsafePointer[Int32,   MutAnyOrigin],
    num_faces:      Int,
    gamma:          Float32,
    min_density:    Float32,
    min_pressure:   Float32,
    inflow_rho:     Float32,
    inflow_rhou:    Float32,
    inflow_rhov:    Float32,
    inflow_E:       Float32,
    fstar_out:      UnsafePointer[Float32, MutAnyOrigin],
):
    var tid = Int(global_idx.x)
    var total = num_faces * NFP
    if tid >= total:
        return
    var fid = tid // NFP
    var m   = tid %  NFP

    var nx = face_normal[fid * 2 + 0]
    var ny = face_normal[fid * 2 + 1]
    var bc_type = face_bc_type[fid]

    var e_l = Int(face_elem[fid * 2 + 0])
    var n_l = Int(face_elem_node[(fid * 2 + 0) * NFP + m])
    var l_off = (e_l * NP + n_l) * 4
    var qL0 = q[l_off + 0]
    var qL1 = q[l_off + 1]
    var qL2 = q[l_off + 2]
    var qL3 = q[l_off + 3]

    var qR0: Float32
    var qR1: Float32
    var qR2: Float32
    var qR3: Float32
    if bc_type == BC_INTERIOR:
        var e_r = Int(face_elem[fid * 2 + 1])
        var n_r = Int(face_elem_node[(fid * 2 + 1) * NFP + m])
        var r_off = (e_r * NP + n_r) * 4
        qR0 = q[r_off + 0]
        qR1 = q[r_off + 1]
        qR2 = q[r_off + 2]
        qR3 = q[r_off + 3]
    elif bc_type == BC_WALL:
        var m_n = qL1 * nx + qL2 * ny
        qR0 = qL0
        qR1 = qL1 - Float32(2.0) * m_n * nx
        qR2 = qL2 - Float32(2.0) * m_n * ny
        qR3 = qL3
    elif bc_type == BC_INFLOW:
        qR0 = inflow_rho
        qR1 = inflow_rhou
        qR2 = inflow_rhov
        qR3 = inflow_E
    else:
        qR0 = qL0
        qR1 = qL1
        qR2 = qL2
        qR3 = qL3

    # Primitives on each side (with floors).
    var rho_L = qL0
    if rho_L < min_density: rho_L = min_density
    var rho_R = qR0
    if rho_R < min_density: rho_R = min_density
    var uL = qL1 / rho_L
    var vL = qL2 / rho_L
    var uR = qR1 / rho_R
    var vR = qR2 / rho_R
    var keL = Float32(0.5) * rho_L * (uL * uL + vL * vL)
    var keR = Float32(0.5) * rho_R * (uR * uR + vR * vR)
    var pL = (gamma - Float32(1.0)) * (qL3 - keL)
    if pL < min_pressure: pL = min_pressure
    var pR = (gamma - Float32(1.0)) * (qR3 - keR)
    if pR < min_pressure: pR = min_pressure
    var aL = sqrt(gamma * pL / rho_L)
    var aR = sqrt(gamma * pR / rho_R)
    var unL = uL * nx + vL * ny
    var unR = uR * nx + vR * ny

    # Normal-direction fluxes F_K.n for each side (4 components).
    var FnL0 = qL1 * nx + qL2 * ny
    var FnL1 = (qL1 * uL + pL) * nx + (qL1 * vL) * ny
    var FnL2 = (qL1 * vL) * nx + (qL2 * vL + pL) * ny
    var FnL3 = (uL * (qL3 + pL)) * nx + (vL * (qL3 + pL)) * ny
    var FnR0 = qR1 * nx + qR2 * ny
    var FnR1 = (qR1 * uR + pR) * nx + (qR1 * vR) * ny
    var FnR2 = (qR1 * vR) * nx + (qR2 * vR + pR) * ny
    var FnR3 = (uR * (qR3 + pR)) * nx + (vR * (qR3 + pR)) * ny

    # Davis wave-speed estimates.
    var S_L = unL - aL
    var tmp = unR - aR
    if tmp < S_L: S_L = tmp
    var S_R = unL + aL
    tmp = unR + aR
    if tmp > S_R: S_R = tmp

    # Contact wave speed.
    var num = pR - pL + rho_L * unL * (S_L - unL) - rho_R * unR * (S_R - unR)
    var den = rho_L * (S_L - unL) - rho_R * (S_R - unR)
    var S_star = num / den

    var out = (fid * NFP + m) * 4
    if S_L >= Float32(0.0):
        fstar_out[out + 0] = FnL0
        fstar_out[out + 1] = FnL1
        fstar_out[out + 2] = FnL2
        fstar_out[out + 3] = FnL3
    elif S_R <= Float32(0.0):
        fstar_out[out + 0] = FnR0
        fstar_out[out + 1] = FnR1
        fstar_out[out + 2] = FnR2
        fstar_out[out + 3] = FnR3
    elif S_star >= Float32(0.0):
        var coef = (S_L - unL) / (S_L - S_star)
        var rho_s = rho_L * coef
        var u_s = uL + (S_star - unL) * nx
        var v_s = vL + (S_star - unL) * ny
        var E_over_rho_s = (
            qL3 / rho_L
            + (S_star - unL) * (S_star + pL / (rho_L * (S_L - unL)))
        )
        var qs0 = rho_s
        var qs1 = rho_s * u_s
        var qs2 = rho_s * v_s
        var qs3 = rho_s * E_over_rho_s
        fstar_out[out + 0] = FnL0 + S_L * (qs0 - qL0)
        fstar_out[out + 1] = FnL1 + S_L * (qs1 - qL1)
        fstar_out[out + 2] = FnL2 + S_L * (qs2 - qL2)
        fstar_out[out + 3] = FnL3 + S_L * (qs3 - qL3)
    else:
        var coef = (S_R - unR) / (S_R - S_star)
        var rho_s = rho_R * coef
        var u_s = uR + (S_star - unR) * nx
        var v_s = vR + (S_star - unR) * ny
        var E_over_rho_s = (
            qR3 / rho_R
            + (S_star - unR) * (S_star + pR / (rho_R * (S_R - unR)))
        )
        var qs0 = rho_s
        var qs1 = rho_s * u_s
        var qs2 = rho_s * v_s
        var qs3 = rho_s * E_over_rho_s
        fstar_out[out + 0] = FnR0 + S_R * (qs0 - qR0)
        fstar_out[out + 1] = FnR1 + S_R * (qs1 - qR1)
        fstar_out[out + 2] = FnR2 + S_R * (qs2 - qR2)
        fstar_out[out + 3] = FnR3 + S_R * (qs3 - qR3)


def launch_euler_face_flux_hllc_2d[NP: Int, NFP: Int](
    mut ctx: DeviceContext,
    q:              UnsafePointer[Float32, MutAnyOrigin],
    face_elem:      UnsafePointer[Int32,   MutAnyOrigin],
    face_elem_node: UnsafePointer[Int32,   MutAnyOrigin],
    face_normal:    UnsafePointer[Float32, MutAnyOrigin],
    face_bc_type:   UnsafePointer[Int32,   MutAnyOrigin],
    num_faces:      Int,
    gamma:          Float32,
    min_density:    Float32,
    min_pressure:   Float32,
    inflow_rho:     Float32,
    inflow_rhou:    Float32,
    inflow_rhov:    Float32,
    inflow_E:       Float32,
    fstar_out:      UnsafePointer[Float32, MutAnyOrigin],
) raises:
    var total = num_faces * NFP
    comptime _kernel = euler_face_flux_hllc_kernel_2d[NP, NFP]
    ctx.enqueue_function[_kernel, _kernel](
        q, face_elem, face_elem_node, face_normal, face_bc_type,
        num_faces,
        gamma, min_density, min_pressure,
        inflow_rho, inflow_rhou, inflow_rhov, inflow_E,
        fstar_out,
        grid_dim=ceildiv(total, 256),
        block_dim=256,
    )


# ----------------------------------------------------------------------
# Full Euler RK-stage orchestration.
# ----------------------------------------------------------------------
# Mirrors `advection_rk_stage_2d` but with 4 components per node and
# the Euler volume / face-flux kernels.  Callers provide q/q_a/q_b/q_out
# (each sized num_elements*NP*4) plus scratch buffers for vol, fstar,
# and rhs.
# ----------------------------------------------------------------------

def euler_rk_stage_2d[P: Int](
    mut ctx: DeviceContext,
    mesh: LocalMesh2DGpu[P],
    Lift_ref: UnsafePointer[Float32, MutAnyOrigin],
    D_ref:    UnsafePointer[Float32, MutAnyOrigin],
    q_in:     UnsafePointer[Float32, MutAnyOrigin],
    q_a:      UnsafePointer[Float32, MutAnyOrigin],
    q_b:      UnsafePointer[Float32, MutAnyOrigin],
    q_out:    UnsafePointer[Float32, MutAnyOrigin],
    vol_scratch:   UnsafePointer[Float32, MutAnyOrigin],
    fstar_scratch: UnsafePointer[Float32, MutAnyOrigin],
    rhs_scratch:   UnsafePointer[Float32, MutAnyOrigin],
    gamma: Float32, min_density: Float32, min_pressure: Float32,
    inflow_rho: Float32, inflow_rhou: Float32,
    inflow_rhov: Float32, inflow_E: Float32,
    a: Float32, b: Float32, cc: Float32, dt: Float32,
) raises:
    # Two launches per stage (down from three): face flux, then a fused
    # vol+lift+RK kernel.  vol_scratch / rhs_scratch are unused on the
    # fused path; kept in the signature for backward compatibility.
    _ = vol_scratch
    _ = rhs_scratch
    comptime NP = num_tri_nodes_2d(P)
    comptime NFP = num_edge_nodes(P)
    launch_euler_face_flux_2d[NP, NFP](
        ctx, q_in,
        mesh.d_face_elem.unsafe_ptr(),
        mesh.d_face_elem_node.unsafe_ptr(),
        mesh.d_face_normal.unsafe_ptr(),
        mesh.d_face_bc_type.unsafe_ptr(),
        mesh.num_faces,
        gamma, min_density, min_pressure,
        inflow_rho, inflow_rhou, inflow_rhov, inflow_E,
        fstar_scratch,
    )
    launch_euler_vol_lift_2d[NP, NFP](
        ctx, q_in, mesh.d_elem_invJ.unsafe_ptr(), D_ref,
        fstar_scratch,
        mesh.d_elem_inv_2A.unsafe_ptr(),
        mesh.d_elem_faces.unsafe_ptr(),
        mesh.d_elem_face_side.unsafe_ptr(),
        mesh.d_elem_canon_to_ref.unsafe_ptr(),
        mesh.d_face_length.unsafe_ptr(),
        Lift_ref, q_a, q_b,
        mesh.num_elements, gamma, min_density, min_pressure,
        a, b, cc, dt, q_out,
    )


# ----------------------------------------------------------------------
# Same as `euler_rk_stage_2d` but uses HLLC instead of Rusanov for the
# interior numerical flux.  Volume RHS and lift-combine kernels are
# unchanged -- HLLC affects only `fstar`.
# ----------------------------------------------------------------------

def euler_rk_stage_hllc_2d[P: Int](
    mut ctx: DeviceContext,
    mesh: LocalMesh2DGpu[P],
    Lift_ref: UnsafePointer[Float32, MutAnyOrigin],
    D_ref:    UnsafePointer[Float32, MutAnyOrigin],
    q_in:     UnsafePointer[Float32, MutAnyOrigin],
    q_a:      UnsafePointer[Float32, MutAnyOrigin],
    q_b:      UnsafePointer[Float32, MutAnyOrigin],
    q_out:    UnsafePointer[Float32, MutAnyOrigin],
    vol_scratch:   UnsafePointer[Float32, MutAnyOrigin],
    fstar_scratch: UnsafePointer[Float32, MutAnyOrigin],
    rhs_scratch:   UnsafePointer[Float32, MutAnyOrigin],
    gamma: Float32, min_density: Float32, min_pressure: Float32,
    inflow_rho: Float32, inflow_rhou: Float32,
    inflow_rhov: Float32, inflow_E: Float32,
    a: Float32, b: Float32, cc: Float32, dt: Float32,
) raises:
    # Two launches per stage (down from three).  HLLC variant of the
    # face flux + the same fused vol+lift+RK kernel as Rusanov path.
    _ = vol_scratch
    _ = rhs_scratch
    comptime NP = num_tri_nodes_2d(P)
    comptime NFP = num_edge_nodes(P)
    launch_euler_face_flux_hllc_2d[NP, NFP](
        ctx, q_in,
        mesh.d_face_elem.unsafe_ptr(),
        mesh.d_face_elem_node.unsafe_ptr(),
        mesh.d_face_normal.unsafe_ptr(),
        mesh.d_face_bc_type.unsafe_ptr(),
        mesh.num_faces,
        gamma, min_density, min_pressure,
        inflow_rho, inflow_rhou, inflow_rhov, inflow_E,
        fstar_scratch,
    )
    launch_euler_vol_lift_2d[NP, NFP](
        ctx, q_in, mesh.d_elem_invJ.unsafe_ptr(), D_ref,
        fstar_scratch,
        mesh.d_elem_inv_2A.unsafe_ptr(),
        mesh.d_elem_faces.unsafe_ptr(),
        mesh.d_elem_face_side.unsafe_ptr(),
        mesh.d_elem_canon_to_ref.unsafe_ptr(),
        mesh.d_face_length.unsafe_ptr(),
        Lift_ref, q_a, q_b,
        mesh.num_elements, gamma, min_density, min_pressure,
        a, b, cc, dt, q_out,
    )


# ----------------------------------------------------------------------
# ShallowWater2D volume RHS (3 components).
# ----------------------------------------------------------------------
# State (h, mx=h*u, my=h*v); flux (Fx, Fy) with Fx = (mx, mx*u+p, mx*v),
# Fy = (my, my*u, my*v+p), p = 0.5 * g * h * h.  One thread per (elem, i).
# ----------------------------------------------------------------------

def sw_volume_rhs_kernel_2d[NP: Int](
    q:            UnsafePointer[Float32, MutAnyOrigin],
    elem_invJ:    UnsafePointer[Float32, MutAnyOrigin],
    D_ref:        UnsafePointer[Float32, MutAnyOrigin],
    num_elements: Int,
    g:            Float32,
    min_h:        Float32,
    vol_out:      UnsafePointer[Float32, MutAnyOrigin],
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

    var acc0: Float32 = 0.0
    var acc1: Float32 = 0.0
    var acc2: Float32 = 0.0

    for j in range(NP):
        var base = (elem * NP + j) * 3
        var h = q[base + 0]
        if h < min_h:
            h = min_h
        var mx = q[base + 1]
        var my = q[base + 2]
        var u = mx / h
        var v = my / h
        var p = Float32(0.5) * g * h * h

        var Fx0 = mx
        var Fx1 = mx * u + p
        var Fx2 = mx * v
        var Fy0 = my
        var Fy1 = my * u
        var Fy2 = my * v + p

        var D_r = D_ref[0 * NP * NP + i * NP + j]
        var D_s = D_ref[1 * NP * NP + i * NP + j]

        acc0 += (iJ00 * Fx0 + iJ01 * Fy0) * D_r + (iJ10 * Fx0 + iJ11 * Fy0) * D_s
        acc1 += (iJ00 * Fx1 + iJ01 * Fy1) * D_r + (iJ10 * Fx1 + iJ11 * Fy1) * D_s
        acc2 += (iJ00 * Fx2 + iJ01 * Fy2) * D_r + (iJ10 * Fx2 + iJ11 * Fy2) * D_s

    var out = (elem * NP + i) * 3
    vol_out[out + 0] = acc0
    vol_out[out + 1] = acc1
    vol_out[out + 2] = acc2


def launch_sw_volume_rhs_2d[NP: Int](
    mut ctx: DeviceContext,
    q:            UnsafePointer[Float32, MutAnyOrigin],
    elem_invJ:    UnsafePointer[Float32, MutAnyOrigin],
    D_ref:        UnsafePointer[Float32, MutAnyOrigin],
    num_elements: Int,
    g:            Float32,
    min_h:        Float32,
    vol_out:      UnsafePointer[Float32, MutAnyOrigin],
) raises:
    var total = num_elements * NP
    comptime _kernel = sw_volume_rhs_kernel_2d[NP]
    ctx.enqueue_function[_kernel, _kernel](
        q, elem_invJ, D_ref, num_elements, g, min_h, vol_out,
        grid_dim=ceildiv(total, 256),
        block_dim=256,
    )


# ----------------------------------------------------------------------
# Fused ShallowWater volume + lift + RK update (2D, NC=3).
# ----------------------------------------------------------------------
# Same fusion pattern as advection / Euler 2D variants.
# ----------------------------------------------------------------------

def sw_vol_lift_combine_rk_kernel_2d[NP: Int, NFP: Int](
    q_in:              UnsafePointer[Float32, MutAnyOrigin],
    elem_invJ:         UnsafePointer[Float32, MutAnyOrigin],
    D_ref:             UnsafePointer[Float32, MutAnyOrigin],
    fstar:             UnsafePointer[Float32, MutAnyOrigin],
    elem_inv_2A:       UnsafePointer[Float32, MutAnyOrigin],
    elem_faces:        UnsafePointer[Int32,   MutAnyOrigin],
    elem_face_side:    UnsafePointer[Int32,   MutAnyOrigin],
    elem_canon_to_ref: UnsafePointer[Int32,   MutAnyOrigin],
    face_length:       UnsafePointer[Float32, MutAnyOrigin],
    Lift_ref:          UnsafePointer[Float32, MutAnyOrigin],
    q_a:               UnsafePointer[Float32, MutAnyOrigin],
    q_b:               UnsafePointer[Float32, MutAnyOrigin],
    num_elements:      Int,
    g:                 Float32,
    min_h:             Float32,
    a: Float32, b: Float32, cc: Float32, dt: Float32,
    q_out:             UnsafePointer[Float32, MutAnyOrigin],
):
    var tid = Int(global_idx.x)
    var total = num_elements * NP
    if tid >= total:
        return
    var elem = tid // NP
    var i    = tid %  NP

    # ---- Volume RHS contribution.
    var iJ00 = elem_invJ[elem * 4 + 0]
    var iJ01 = elem_invJ[elem * 4 + 1]
    var iJ10 = elem_invJ[elem * 4 + 2]
    var iJ11 = elem_invJ[elem * 4 + 3]
    var acc0: Float32 = 0.0
    var acc1: Float32 = 0.0
    var acc2: Float32 = 0.0
    for j in range(NP):
        var base = (elem * NP + j) * 3
        var h = q_in[base + 0]
        if h < min_h:
            h = min_h
        var mx = q_in[base + 1]
        var my = q_in[base + 2]
        var u = mx / h
        var v = my / h
        var p = Float32(0.5) * g * h * h
        var Fx0 = mx
        var Fx1 = mx * u + p
        var Fx2 = mx * v
        var Fy0 = my
        var Fy1 = my * u
        var Fy2 = my * v + p
        var D_r = D_ref[0 * NP * NP + i * NP + j]
        var D_s = D_ref[1 * NP * NP + i * NP + j]
        acc0 += (iJ00 * Fx0 + iJ01 * Fy0) * D_r + (iJ10 * Fx0 + iJ11 * Fy0) * D_s
        acc1 += (iJ00 * Fx1 + iJ01 * Fy1) * D_r + (iJ10 * Fx1 + iJ11 * Fy1) * D_s
        acc2 += (iJ00 * Fx2 + iJ01 * Fy2) * D_r + (iJ10 * Fx2 + iJ11 * Fy2) * D_s

    # ---- Lift contribution (NC=3).
    var inv_2A = elem_inv_2A[elem]
    var face0: Float32 = 0.0
    var face1: Float32 = 0.0
    var face2: Float32 = 0.0
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
            var sLf = sign * flen * Lim
            var fbase = (fid * NFP + m) * 3
            face0 += sLf * fstar[fbase + 0]
            face1 += sLf * fstar[fbase + 1]
            face2 += sLf * fstar[fbase + 2]

    # ---- Combine + RK update.
    var idx = (elem * NP + i) * 3
    var rhs0 = acc0 - inv_2A * face0
    var rhs1 = acc1 - inv_2A * face1
    var rhs2 = acc2 - inv_2A * face2
    q_out[idx + 0] = a * q_a[idx + 0] + b * q_b[idx + 0] + cc * dt * rhs0
    q_out[idx + 1] = a * q_a[idx + 1] + b * q_b[idx + 1] + cc * dt * rhs1
    q_out[idx + 2] = a * q_a[idx + 2] + b * q_b[idx + 2] + cc * dt * rhs2


def launch_sw_vol_lift_2d[NP: Int, NFP: Int](
    mut ctx: DeviceContext,
    q_in:              UnsafePointer[Float32, MutAnyOrigin],
    elem_invJ:         UnsafePointer[Float32, MutAnyOrigin],
    D_ref:             UnsafePointer[Float32, MutAnyOrigin],
    fstar:             UnsafePointer[Float32, MutAnyOrigin],
    elem_inv_2A:       UnsafePointer[Float32, MutAnyOrigin],
    elem_faces:        UnsafePointer[Int32,   MutAnyOrigin],
    elem_face_side:    UnsafePointer[Int32,   MutAnyOrigin],
    elem_canon_to_ref: UnsafePointer[Int32,   MutAnyOrigin],
    face_length:       UnsafePointer[Float32, MutAnyOrigin],
    Lift_ref:          UnsafePointer[Float32, MutAnyOrigin],
    q_a:               UnsafePointer[Float32, MutAnyOrigin],
    q_b:               UnsafePointer[Float32, MutAnyOrigin],
    num_elements:      Int,
    g:                 Float32,
    min_h:             Float32,
    a: Float32, b: Float32, cc: Float32, dt: Float32,
    q_out:             UnsafePointer[Float32, MutAnyOrigin],
) raises:
    var total = num_elements * NP
    comptime _kernel = sw_vol_lift_combine_rk_kernel_2d[NP, NFP]
    ctx.enqueue_function[_kernel, _kernel](
        q_in, elem_invJ, D_ref, fstar,
        elem_inv_2A, elem_faces, elem_face_side, elem_canon_to_ref,
        face_length, Lift_ref, q_a, q_b,
        num_elements, g, min_h, a, b, cc, dt, q_out,
        grid_dim=ceildiv(total, 256),
        block_dim=256,
    )


# ----------------------------------------------------------------------
# ShallowWater2D face flux (Rusanov, 3 components).
# ----------------------------------------------------------------------
# BC ghost: WALL reflects normal momentum, INFLOW is user-set, OUTFLOW
# is zero-gradient -- matches ShallowWater2D.boundary_flux on CPU.
# ----------------------------------------------------------------------

def sw_face_flux_kernel_2d[NP: Int, NFP: Int](
    q:              UnsafePointer[Float32, MutAnyOrigin],
    face_elem:      UnsafePointer[Int32,   MutAnyOrigin],
    face_elem_node: UnsafePointer[Int32,   MutAnyOrigin],
    face_normal:    UnsafePointer[Float32, MutAnyOrigin],
    face_bc_type:   UnsafePointer[Int32,   MutAnyOrigin],
    num_faces:      Int,
    g:              Float32,
    min_h:          Float32,
    inflow_h:       Float32,
    inflow_hu:      Float32,
    inflow_hv:      Float32,
    fstar_out:      UnsafePointer[Float32, MutAnyOrigin],
):
    var tid = Int(global_idx.x)
    var total = num_faces * NFP
    if tid >= total:
        return
    var fid = tid // NFP
    var m   = tid %  NFP

    var nx = face_normal[fid * 2 + 0]
    var ny = face_normal[fid * 2 + 1]
    var bc_type = face_bc_type[fid]

    var e_l = Int(face_elem[fid * 2 + 0])
    var n_l = Int(face_elem_node[(fid * 2 + 0) * NFP + m])
    var l_off = (e_l * NP + n_l) * 3
    var qL0 = q[l_off + 0]
    var qL1 = q[l_off + 1]
    var qL2 = q[l_off + 2]

    var qR0: Float32
    var qR1: Float32
    var qR2: Float32
    if bc_type == BC_INTERIOR:
        var e_r = Int(face_elem[fid * 2 + 1])
        var n_r = Int(face_elem_node[(fid * 2 + 1) * NFP + m])
        var r_off = (e_r * NP + n_r) * 3
        qR0 = q[r_off + 0]
        qR1 = q[r_off + 1]
        qR2 = q[r_off + 2]
    elif bc_type == BC_WALL:
        var m_n = qL1 * nx + qL2 * ny
        qR0 = qL0
        qR1 = qL1 - Float32(2.0) * m_n * nx
        qR2 = qL2 - Float32(2.0) * m_n * ny
    elif bc_type == BC_INFLOW:
        qR0 = inflow_h
        qR1 = inflow_hu
        qR2 = inflow_hv
    else:
        qR0 = qL0
        qR1 = qL1
        qR2 = qL2

    var hL = qL0
    if hL < min_h:
        hL = min_h
    var uL = qL1 / hL
    var vL = qL2 / hL
    var pL = Float32(0.5) * g * hL * hL
    var cL = sqrt(g * hL)
    var speedL = sqrt(uL * uL + vL * vL) + cL
    var FxL0 = qL1
    var FxL1 = qL1 * uL + pL
    var FxL2 = qL1 * vL
    var FyL0 = qL2
    var FyL1 = qL2 * uL
    var FyL2 = qL2 * vL + pL

    var hR = qR0
    if hR < min_h:
        hR = min_h
    var uR = qR1 / hR
    var vR = qR2 / hR
    var pR = Float32(0.5) * g * hR * hR
    var cR = sqrt(g * hR)
    var speedR = sqrt(uR * uR + vR * vR) + cR
    var FxR0 = qR1
    var FxR1 = qR1 * uR + pR
    var FxR2 = qR1 * vR
    var FyR0 = qR2
    var FyR1 = qR2 * uR
    var FyR2 = qR2 * vR + pR

    var alpha: Float32 = speedL if speedL > speedR else speedR
    var half = Float32(0.5)
    var out = (fid * NFP + m) * 3
    fstar_out[out + 0] = half * ((FxL0 + FxR0) * nx + (FyL0 + FyR0) * ny) \
                         - half * alpha * (qR0 - qL0)
    fstar_out[out + 1] = half * ((FxL1 + FxR1) * nx + (FyL1 + FyR1) * ny) \
                         - half * alpha * (qR1 - qL1)
    fstar_out[out + 2] = half * ((FxL2 + FxR2) * nx + (FyL2 + FyR2) * ny) \
                         - half * alpha * (qR2 - qL2)


def launch_sw_face_flux_2d[NP: Int, NFP: Int](
    mut ctx: DeviceContext,
    q:              UnsafePointer[Float32, MutAnyOrigin],
    face_elem:      UnsafePointer[Int32,   MutAnyOrigin],
    face_elem_node: UnsafePointer[Int32,   MutAnyOrigin],
    face_normal:    UnsafePointer[Float32, MutAnyOrigin],
    face_bc_type:   UnsafePointer[Int32,   MutAnyOrigin],
    num_faces:      Int,
    g:              Float32,
    min_h:          Float32,
    inflow_h:       Float32,
    inflow_hu:      Float32,
    inflow_hv:      Float32,
    fstar_out:      UnsafePointer[Float32, MutAnyOrigin],
) raises:
    var total = num_faces * NFP
    comptime _kernel = sw_face_flux_kernel_2d[NP, NFP]
    ctx.enqueue_function[_kernel, _kernel](
        q, face_elem, face_elem_node, face_normal, face_bc_type,
        num_faces,
        g, min_h, inflow_h, inflow_hu, inflow_hv,
        fstar_out,
        grid_dim=ceildiv(total, 256),
        block_dim=256,
    )


# ----------------------------------------------------------------------
# ShallowWater2D HLL face flux (Einfeldt 1988, 3 components).
# ----------------------------------------------------------------------
# Same signature as the Rusanov kernel; uses Davis wave-speed estimates
# S_L = min(unL - cL, unR - cR), S_R = max(unL + cL, unR + cR), then
# the three-region HLL formula for the middle state.  Less dissipative
# than Rusanov's single-alpha fan, particularly around contacts and
# expansion fans.  Ghost-state plumbing (WALL / INFLOW / OUTFLOW) is
# the same as the Rusanov kernel -- only the interior flux differs.
# ----------------------------------------------------------------------

def sw_face_flux_hll_kernel_2d[NP: Int, NFP: Int](
    q:              UnsafePointer[Float32, MutAnyOrigin],
    face_elem:      UnsafePointer[Int32,   MutAnyOrigin],
    face_elem_node: UnsafePointer[Int32,   MutAnyOrigin],
    face_normal:    UnsafePointer[Float32, MutAnyOrigin],
    face_bc_type:   UnsafePointer[Int32,   MutAnyOrigin],
    num_faces:      Int,
    g:              Float32,
    min_h:          Float32,
    inflow_h:       Float32,
    inflow_hu:      Float32,
    inflow_hv:      Float32,
    fstar_out:      UnsafePointer[Float32, MutAnyOrigin],
):
    var tid = Int(global_idx.x)
    var total = num_faces * NFP
    if tid >= total:
        return
    var fid = tid // NFP
    var m   = tid %  NFP

    var nx = face_normal[fid * 2 + 0]
    var ny = face_normal[fid * 2 + 1]
    var bc_type = face_bc_type[fid]

    var e_l = Int(face_elem[fid * 2 + 0])
    var n_l = Int(face_elem_node[(fid * 2 + 0) * NFP + m])
    var l_off = (e_l * NP + n_l) * 3
    var qL0 = q[l_off + 0]
    var qL1 = q[l_off + 1]
    var qL2 = q[l_off + 2]

    var qR0: Float32
    var qR1: Float32
    var qR2: Float32
    if bc_type == BC_INTERIOR:
        var e_r = Int(face_elem[fid * 2 + 1])
        var n_r = Int(face_elem_node[(fid * 2 + 1) * NFP + m])
        var r_off = (e_r * NP + n_r) * 3
        qR0 = q[r_off + 0]
        qR1 = q[r_off + 1]
        qR2 = q[r_off + 2]
    elif bc_type == BC_WALL:
        var m_n = qL1 * nx + qL2 * ny
        qR0 = qL0
        qR1 = qL1 - Float32(2.0) * m_n * nx
        qR2 = qL2 - Float32(2.0) * m_n * ny
    elif bc_type == BC_INFLOW:
        qR0 = inflow_h
        qR1 = inflow_hu
        qR2 = inflow_hv
    else:
        qR0 = qL0
        qR1 = qL1
        qR2 = qL2

    var hL = qL0
    if hL < min_h: hL = min_h
    var hR = qR0
    if hR < min_h: hR = min_h
    var uL = qL1 / hL
    var vL = qL2 / hL
    var uR = qR1 / hR
    var vR = qR2 / hR
    var pL = Float32(0.5) * g * hL * hL
    var pR = Float32(0.5) * g * hR * hR
    var cL = sqrt(g * hL)
    var cR = sqrt(g * hR)
    var unL = uL * nx + vL * ny
    var unR = uR * nx + vR * ny

    # Davis wave speeds.
    var S_L = unL - cL
    var tmp = unR - cR
    if tmp < S_L: S_L = tmp
    var S_R = unL + cL
    tmp = unR + cR
    if tmp > S_R: S_R = tmp

    # Normal fluxes on each side.
    var FxL0 = qL1
    var FxL1 = qL1 * uL + pL
    var FxL2 = qL1 * vL
    var FyL0 = qL2
    var FyL1 = qL2 * uL
    var FyL2 = qL2 * vL + pL
    var FxR0 = qR1
    var FxR1 = qR1 * uR + pR
    var FxR2 = qR1 * vR
    var FyR0 = qR2
    var FyR1 = qR2 * uR
    var FyR2 = qR2 * vR + pR
    var FnL0 = FxL0 * nx + FyL0 * ny
    var FnL1 = FxL1 * nx + FyL1 * ny
    var FnL2 = FxL2 * nx + FyL2 * ny
    var FnR0 = FxR0 * nx + FyR0 * ny
    var FnR1 = FxR1 * nx + FyR1 * ny
    var FnR2 = FxR2 * nx + FyR2 * ny

    var out = (fid * NFP + m) * 3
    if S_L >= Float32(0.0):
        fstar_out[out + 0] = FnL0
        fstar_out[out + 1] = FnL1
        fstar_out[out + 2] = FnL2
    elif S_R <= Float32(0.0):
        fstar_out[out + 0] = FnR0
        fstar_out[out + 1] = FnR1
        fstar_out[out + 2] = FnR2
    else:
        var inv = Float32(1.0) / (S_R - S_L)
        fstar_out[out + 0] = (S_R * FnL0 - S_L * FnR0 + S_L * S_R * (qR0 - qL0)) * inv
        fstar_out[out + 1] = (S_R * FnL1 - S_L * FnR1 + S_L * S_R * (qR1 - qL1)) * inv
        fstar_out[out + 2] = (S_R * FnL2 - S_L * FnR2 + S_L * S_R * (qR2 - qL2)) * inv


def launch_sw_face_flux_hll_2d[NP: Int, NFP: Int](
    mut ctx: DeviceContext,
    q:              UnsafePointer[Float32, MutAnyOrigin],
    face_elem:      UnsafePointer[Int32,   MutAnyOrigin],
    face_elem_node: UnsafePointer[Int32,   MutAnyOrigin],
    face_normal:    UnsafePointer[Float32, MutAnyOrigin],
    face_bc_type:   UnsafePointer[Int32,   MutAnyOrigin],
    num_faces:      Int,
    g:              Float32,
    min_h:          Float32,
    inflow_h:       Float32,
    inflow_hu:      Float32,
    inflow_hv:      Float32,
    fstar_out:      UnsafePointer[Float32, MutAnyOrigin],
) raises:
    var total = num_faces * NFP
    comptime _kernel = sw_face_flux_hll_kernel_2d[NP, NFP]
    ctx.enqueue_function[_kernel, _kernel](
        q, face_elem, face_elem_node, face_normal, face_bc_type,
        num_faces,
        g, min_h, inflow_h, inflow_hu, inflow_hv,
        fstar_out,
        grid_dim=ceildiv(total, 256),
        block_dim=256,
    )


def sw_rk_stage_2d[P: Int](
    mut ctx: DeviceContext,
    mesh: LocalMesh2DGpu[P],
    Lift_ref: UnsafePointer[Float32, MutAnyOrigin],
    D_ref:    UnsafePointer[Float32, MutAnyOrigin],
    q_in:     UnsafePointer[Float32, MutAnyOrigin],
    q_a:      UnsafePointer[Float32, MutAnyOrigin],
    q_b:      UnsafePointer[Float32, MutAnyOrigin],
    q_out:    UnsafePointer[Float32, MutAnyOrigin],
    vol_scratch:   UnsafePointer[Float32, MutAnyOrigin],
    fstar_scratch: UnsafePointer[Float32, MutAnyOrigin],
    rhs_scratch:   UnsafePointer[Float32, MutAnyOrigin],
    g: Float32, min_h: Float32,
    inflow_h: Float32, inflow_hu: Float32, inflow_hv: Float32,
    a: Float32, b: Float32, cc: Float32, dt: Float32,
) raises:
    # Two launches per stage (down from three).
    _ = vol_scratch
    _ = rhs_scratch
    comptime NP = num_tri_nodes_2d(P)
    comptime NFP = num_edge_nodes(P)
    launch_sw_face_flux_2d[NP, NFP](
        ctx, q_in,
        mesh.d_face_elem.unsafe_ptr(),
        mesh.d_face_elem_node.unsafe_ptr(),
        mesh.d_face_normal.unsafe_ptr(),
        mesh.d_face_bc_type.unsafe_ptr(),
        mesh.num_faces,
        g, min_h, inflow_h, inflow_hu, inflow_hv,
        fstar_scratch,
    )
    launch_sw_vol_lift_2d[NP, NFP](
        ctx, q_in, mesh.d_elem_invJ.unsafe_ptr(), D_ref,
        fstar_scratch,
        mesh.d_elem_inv_2A.unsafe_ptr(),
        mesh.d_elem_faces.unsafe_ptr(),
        mesh.d_elem_face_side.unsafe_ptr(),
        mesh.d_elem_canon_to_ref.unsafe_ptr(),
        mesh.d_face_length.unsafe_ptr(),
        Lift_ref, q_a, q_b,
        mesh.num_elements, g, min_h,
        a, b, cc, dt, q_out,
    )


# ----------------------------------------------------------------------
# ShallowWater RK stage using HLL instead of Rusanov for the interior
# numerical flux.  Drop-in replacement for sw_rk_stage_2d.
# ----------------------------------------------------------------------

def sw_rk_stage_hll_2d[P: Int](
    mut ctx: DeviceContext,
    mesh: LocalMesh2DGpu[P],
    Lift_ref: UnsafePointer[Float32, MutAnyOrigin],
    D_ref:    UnsafePointer[Float32, MutAnyOrigin],
    q_in:     UnsafePointer[Float32, MutAnyOrigin],
    q_a:      UnsafePointer[Float32, MutAnyOrigin],
    q_b:      UnsafePointer[Float32, MutAnyOrigin],
    q_out:    UnsafePointer[Float32, MutAnyOrigin],
    vol_scratch:   UnsafePointer[Float32, MutAnyOrigin],
    fstar_scratch: UnsafePointer[Float32, MutAnyOrigin],
    rhs_scratch:   UnsafePointer[Float32, MutAnyOrigin],
    g: Float32, min_h: Float32,
    inflow_h: Float32, inflow_hu: Float32, inflow_hv: Float32,
    a: Float32, b: Float32, cc: Float32, dt: Float32,
) raises:
    # Two launches per stage (down from three), HLL flux variant.
    _ = vol_scratch
    _ = rhs_scratch
    comptime NP = num_tri_nodes_2d(P)
    comptime NFP = num_edge_nodes(P)
    launch_sw_face_flux_hll_2d[NP, NFP](
        ctx, q_in,
        mesh.d_face_elem.unsafe_ptr(),
        mesh.d_face_elem_node.unsafe_ptr(),
        mesh.d_face_normal.unsafe_ptr(),
        mesh.d_face_bc_type.unsafe_ptr(),
        mesh.num_faces,
        g, min_h, inflow_h, inflow_hu, inflow_hv,
        fstar_scratch,
    )
    launch_sw_vol_lift_2d[NP, NFP](
        ctx, q_in, mesh.d_elem_invJ.unsafe_ptr(), D_ref,
        fstar_scratch,
        mesh.d_elem_inv_2A.unsafe_ptr(),
        mesh.d_elem_faces.unsafe_ptr(),
        mesh.d_elem_face_side.unsafe_ptr(),
        mesh.d_elem_canon_to_ref.unsafe_ptr(),
        mesh.d_face_length.unsafe_ptr(),
        Lift_ref, q_a, q_b,
        mesh.num_elements, g, min_h,
        a, b, cc, dt, q_out,
    )


# ----------------------------------------------------------------------
# IdealMHD2D volume RHS (6 components).
# ----------------------------------------------------------------------
# State (rho, mx, my, Bx, By, E).  Flux per Powell 1999 ideal 2D MHD
# (no div B cleaning); fast-magnetosonic speed bound.  Component order
# matches CPU IdealMHD2D.internal_flux exactly.
# ----------------------------------------------------------------------

def mhd_volume_rhs_kernel_2d[NP: Int](
    q:            UnsafePointer[Float32, MutAnyOrigin],
    elem_invJ:    UnsafePointer[Float32, MutAnyOrigin],
    D_ref:        UnsafePointer[Float32, MutAnyOrigin],
    num_elements: Int,
    gamma:        Float32,
    min_density:  Float32,
    min_pressure: Float32,
    vol_out:      UnsafePointer[Float32, MutAnyOrigin],
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

    var acc0: Float32 = 0.0
    var acc1: Float32 = 0.0
    var acc2: Float32 = 0.0
    var acc3: Float32 = 0.0
    var acc4: Float32 = 0.0
    var acc5: Float32 = 0.0

    for j in range(NP):
        var base = (elem * NP + j) * 6
        var rho = q[base + 0]
        if rho < min_density:
            rho = min_density
        var mx = q[base + 1]
        var my = q[base + 2]
        var Bx = q[base + 3]
        var By = q[base + 4]
        var E  = q[base + 5]
        var u = mx / rho
        var v = my / rho
        var BB = Bx * Bx + By * By
        var ke = Float32(0.5) * (mx * mx + my * my) / rho
        var mp = Float32(0.5) * BB
        var p = (gamma - Float32(1.0)) * (E - ke - mp)
        if p < min_pressure:
            p = min_pressure
        var pstar = p + Float32(0.5) * BB

        var Fx0 = mx
        var Fx1 = mx * u + pstar - Bx * Bx
        var Fx2 = mx * v         - Bx * By
        var Fx3 = Float32(0.0)
        var Fx4 = u * By - v * Bx
        var Fx5 = (E + pstar) * u - Bx * (u * Bx + v * By)
        var Fy0 = my
        var Fy1 = my * u         - By * Bx
        var Fy2 = my * v + pstar - By * By
        var Fy3 = v * Bx - u * By
        var Fy4 = Float32(0.0)
        var Fy5 = (E + pstar) * v - By * (u * Bx + v * By)

        var D_r = D_ref[0 * NP * NP + i * NP + j]
        var D_s = D_ref[1 * NP * NP + i * NP + j]

        acc0 += (iJ00 * Fx0 + iJ01 * Fy0) * D_r + (iJ10 * Fx0 + iJ11 * Fy0) * D_s
        acc1 += (iJ00 * Fx1 + iJ01 * Fy1) * D_r + (iJ10 * Fx1 + iJ11 * Fy1) * D_s
        acc2 += (iJ00 * Fx2 + iJ01 * Fy2) * D_r + (iJ10 * Fx2 + iJ11 * Fy2) * D_s
        acc3 += (iJ00 * Fx3 + iJ01 * Fy3) * D_r + (iJ10 * Fx3 + iJ11 * Fy3) * D_s
        acc4 += (iJ00 * Fx4 + iJ01 * Fy4) * D_r + (iJ10 * Fx4 + iJ11 * Fy4) * D_s
        acc5 += (iJ00 * Fx5 + iJ01 * Fy5) * D_r + (iJ10 * Fx5 + iJ11 * Fy5) * D_s

    var out = (elem * NP + i) * 6
    vol_out[out + 0] = acc0
    vol_out[out + 1] = acc1
    vol_out[out + 2] = acc2
    vol_out[out + 3] = acc3
    vol_out[out + 4] = acc4
    vol_out[out + 5] = acc5


def launch_mhd_volume_rhs_2d[NP: Int](
    mut ctx: DeviceContext,
    q:            UnsafePointer[Float32, MutAnyOrigin],
    elem_invJ:    UnsafePointer[Float32, MutAnyOrigin],
    D_ref:        UnsafePointer[Float32, MutAnyOrigin],
    num_elements: Int,
    gamma:        Float32,
    min_density:  Float32,
    min_pressure: Float32,
    vol_out:      UnsafePointer[Float32, MutAnyOrigin],
) raises:
    var total = num_elements * NP
    comptime _kernel = mhd_volume_rhs_kernel_2d[NP]
    ctx.enqueue_function[_kernel, _kernel](
        q, elem_invJ, D_ref, num_elements,
        gamma, min_density, min_pressure, vol_out,
        grid_dim=ceildiv(total, 256),
        block_dim=256,
    )


# ----------------------------------------------------------------------
# Fused IdealMHD volume + lift + RK update (2D, NC=6).
# ----------------------------------------------------------------------
# Same fusion pattern as advection / Euler / SW 2D variants.
# ----------------------------------------------------------------------

def mhd_vol_lift_combine_rk_kernel_2d[NP: Int, NFP: Int](
    q_in:              UnsafePointer[Float32, MutAnyOrigin],
    elem_invJ:         UnsafePointer[Float32, MutAnyOrigin],
    D_ref:             UnsafePointer[Float32, MutAnyOrigin],
    fstar:             UnsafePointer[Float32, MutAnyOrigin],
    elem_inv_2A:       UnsafePointer[Float32, MutAnyOrigin],
    elem_faces:        UnsafePointer[Int32,   MutAnyOrigin],
    elem_face_side:    UnsafePointer[Int32,   MutAnyOrigin],
    elem_canon_to_ref: UnsafePointer[Int32,   MutAnyOrigin],
    face_length:       UnsafePointer[Float32, MutAnyOrigin],
    Lift_ref:          UnsafePointer[Float32, MutAnyOrigin],
    q_a:               UnsafePointer[Float32, MutAnyOrigin],
    q_b:               UnsafePointer[Float32, MutAnyOrigin],
    num_elements:      Int,
    gamma:             Float32,
    min_density:       Float32,
    min_pressure:      Float32,
    a: Float32, b: Float32, cc: Float32, dt: Float32,
    q_out:             UnsafePointer[Float32, MutAnyOrigin],
):
    var tid = Int(global_idx.x)
    var total = num_elements * NP
    if tid >= total:
        return
    var elem = tid // NP
    var i    = tid %  NP

    # ---- Volume RHS contribution (NC=6).
    var iJ00 = elem_invJ[elem * 4 + 0]
    var iJ01 = elem_invJ[elem * 4 + 1]
    var iJ10 = elem_invJ[elem * 4 + 2]
    var iJ11 = elem_invJ[elem * 4 + 3]
    var acc0: Float32 = 0.0
    var acc1: Float32 = 0.0
    var acc2: Float32 = 0.0
    var acc3: Float32 = 0.0
    var acc4: Float32 = 0.0
    var acc5: Float32 = 0.0
    for j in range(NP):
        var base = (elem * NP + j) * 6
        var rho = q_in[base + 0]
        if rho < min_density:
            rho = min_density
        var mx = q_in[base + 1]
        var my = q_in[base + 2]
        var Bx = q_in[base + 3]
        var By = q_in[base + 4]
        var E  = q_in[base + 5]
        var u = mx / rho
        var v = my / rho
        var BB = Bx * Bx + By * By
        var ke = Float32(0.5) * (mx * mx + my * my) / rho
        var mp = Float32(0.5) * BB
        var p = (gamma - Float32(1.0)) * (E - ke - mp)
        if p < min_pressure:
            p = min_pressure
        var pstar = p + Float32(0.5) * BB
        var Fx0 = mx
        var Fx1 = mx * u + pstar - Bx * Bx
        var Fx2 = mx * v         - Bx * By
        var Fx3 = Float32(0.0)
        var Fx4 = u * By - v * Bx
        var Fx5 = (E + pstar) * u - Bx * (u * Bx + v * By)
        var Fy0 = my
        var Fy1 = my * u         - By * Bx
        var Fy2 = my * v + pstar - By * By
        var Fy3 = v * Bx - u * By
        var Fy4 = Float32(0.0)
        var Fy5 = (E + pstar) * v - By * (u * Bx + v * By)
        var D_r = D_ref[0 * NP * NP + i * NP + j]
        var D_s = D_ref[1 * NP * NP + i * NP + j]
        acc0 += (iJ00 * Fx0 + iJ01 * Fy0) * D_r + (iJ10 * Fx0 + iJ11 * Fy0) * D_s
        acc1 += (iJ00 * Fx1 + iJ01 * Fy1) * D_r + (iJ10 * Fx1 + iJ11 * Fy1) * D_s
        acc2 += (iJ00 * Fx2 + iJ01 * Fy2) * D_r + (iJ10 * Fx2 + iJ11 * Fy2) * D_s
        acc3 += (iJ00 * Fx3 + iJ01 * Fy3) * D_r + (iJ10 * Fx3 + iJ11 * Fy3) * D_s
        acc4 += (iJ00 * Fx4 + iJ01 * Fy4) * D_r + (iJ10 * Fx4 + iJ11 * Fy4) * D_s
        acc5 += (iJ00 * Fx5 + iJ01 * Fy5) * D_r + (iJ10 * Fx5 + iJ11 * Fy5) * D_s

    # ---- Lift contribution (NC=6).
    var inv_2A = elem_inv_2A[elem]
    var face0: Float32 = 0.0
    var face1: Float32 = 0.0
    var face2: Float32 = 0.0
    var face3: Float32 = 0.0
    var face4: Float32 = 0.0
    var face5: Float32 = 0.0
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
            var sLf = sign * flen * Lim
            var fbase = (fid * NFP + m) * 6
            face0 += sLf * fstar[fbase + 0]
            face1 += sLf * fstar[fbase + 1]
            face2 += sLf * fstar[fbase + 2]
            face3 += sLf * fstar[fbase + 3]
            face4 += sLf * fstar[fbase + 4]
            face5 += sLf * fstar[fbase + 5]

    # ---- Combine + RK update.
    var idx = (elem * NP + i) * 6
    var rhs0 = acc0 - inv_2A * face0
    var rhs1 = acc1 - inv_2A * face1
    var rhs2 = acc2 - inv_2A * face2
    var rhs3 = acc3 - inv_2A * face3
    var rhs4 = acc4 - inv_2A * face4
    var rhs5 = acc5 - inv_2A * face5
    q_out[idx + 0] = a * q_a[idx + 0] + b * q_b[idx + 0] + cc * dt * rhs0
    q_out[idx + 1] = a * q_a[idx + 1] + b * q_b[idx + 1] + cc * dt * rhs1
    q_out[idx + 2] = a * q_a[idx + 2] + b * q_b[idx + 2] + cc * dt * rhs2
    q_out[idx + 3] = a * q_a[idx + 3] + b * q_b[idx + 3] + cc * dt * rhs3
    q_out[idx + 4] = a * q_a[idx + 4] + b * q_b[idx + 4] + cc * dt * rhs4
    q_out[idx + 5] = a * q_a[idx + 5] + b * q_b[idx + 5] + cc * dt * rhs5


def launch_mhd_vol_lift_2d[NP: Int, NFP: Int](
    mut ctx: DeviceContext,
    q_in:              UnsafePointer[Float32, MutAnyOrigin],
    elem_invJ:         UnsafePointer[Float32, MutAnyOrigin],
    D_ref:             UnsafePointer[Float32, MutAnyOrigin],
    fstar:             UnsafePointer[Float32, MutAnyOrigin],
    elem_inv_2A:       UnsafePointer[Float32, MutAnyOrigin],
    elem_faces:        UnsafePointer[Int32,   MutAnyOrigin],
    elem_face_side:    UnsafePointer[Int32,   MutAnyOrigin],
    elem_canon_to_ref: UnsafePointer[Int32,   MutAnyOrigin],
    face_length:       UnsafePointer[Float32, MutAnyOrigin],
    Lift_ref:          UnsafePointer[Float32, MutAnyOrigin],
    q_a:               UnsafePointer[Float32, MutAnyOrigin],
    q_b:               UnsafePointer[Float32, MutAnyOrigin],
    num_elements:      Int,
    gamma:             Float32,
    min_density:       Float32,
    min_pressure:      Float32,
    a: Float32, b: Float32, cc: Float32, dt: Float32,
    q_out:             UnsafePointer[Float32, MutAnyOrigin],
) raises:
    var total = num_elements * NP
    comptime _kernel = mhd_vol_lift_combine_rk_kernel_2d[NP, NFP]
    ctx.enqueue_function[_kernel, _kernel](
        q_in, elem_invJ, D_ref, fstar,
        elem_inv_2A, elem_faces, elem_face_side, elem_canon_to_ref,
        face_length, Lift_ref, q_a, q_b,
        num_elements, gamma, min_density, min_pressure,
        a, b, cc, dt, q_out,
        grid_dim=ceildiv(total, 256),
        block_dim=256,
    )


# ----------------------------------------------------------------------
# IdealMHD2D face flux (Rusanov / local Lax-Friedrichs, 6 components).
# ----------------------------------------------------------------------
# Wave-speed bound uses the fast-magnetosonic speed cf, as on the CPU.
# BC ghost: WALL reflects both normal momentum AND normal B (perfectly
# conducting slip wall); all others (OUTFLOW / INFLOW / unhandled) fall
# back to zero-gradient, matching CPU IdealMHD2D.boundary_flux.
# ----------------------------------------------------------------------

def mhd_face_flux_kernel_2d[NP: Int, NFP: Int](
    q:              UnsafePointer[Float32, MutAnyOrigin],
    face_elem:      UnsafePointer[Int32,   MutAnyOrigin],
    face_elem_node: UnsafePointer[Int32,   MutAnyOrigin],
    face_normal:    UnsafePointer[Float32, MutAnyOrigin],
    face_bc_type:   UnsafePointer[Int32,   MutAnyOrigin],
    num_faces:      Int,
    gamma:          Float32,
    min_density:    Float32,
    min_pressure:   Float32,
    fstar_out:      UnsafePointer[Float32, MutAnyOrigin],
):
    var tid = Int(global_idx.x)
    var total = num_faces * NFP
    if tid >= total:
        return
    var fid = tid // NFP
    var m   = tid %  NFP

    var nx = face_normal[fid * 2 + 0]
    var ny = face_normal[fid * 2 + 1]
    var bc_type = face_bc_type[fid]

    var e_l = Int(face_elem[fid * 2 + 0])
    var n_l = Int(face_elem_node[(fid * 2 + 0) * NFP + m])
    var l_off = (e_l * NP + n_l) * 6
    var qL0 = q[l_off + 0]
    var qL1 = q[l_off + 1]
    var qL2 = q[l_off + 2]
    var qL3 = q[l_off + 3]
    var qL4 = q[l_off + 4]
    var qL5 = q[l_off + 5]

    var qR0: Float32
    var qR1: Float32
    var qR2: Float32
    var qR3: Float32
    var qR4: Float32
    var qR5: Float32
    if bc_type == BC_INTERIOR:
        var e_r = Int(face_elem[fid * 2 + 1])
        var n_r = Int(face_elem_node[(fid * 2 + 1) * NFP + m])
        var r_off = (e_r * NP + n_r) * 6
        qR0 = q[r_off + 0]
        qR1 = q[r_off + 1]
        qR2 = q[r_off + 2]
        qR3 = q[r_off + 3]
        qR4 = q[r_off + 4]
        qR5 = q[r_off + 5]
    elif bc_type == BC_WALL:
        var m_n = qL1 * nx + qL2 * ny
        var B_n = qL3 * nx + qL4 * ny
        qR0 = qL0
        qR1 = qL1 - Float32(2.0) * m_n * nx
        qR2 = qL2 - Float32(2.0) * m_n * ny
        qR3 = qL3 - Float32(2.0) * B_n * nx
        qR4 = qL4 - Float32(2.0) * B_n * ny
        qR5 = qL5
    else:
        qR0 = qL0
        qR1 = qL1
        qR2 = qL2
        qR3 = qL3
        qR4 = qL4
        qR5 = qL5

    # Left internal flux + fast-magnetosonic speed.
    var rhoL = qL0
    if rhoL < min_density:
        rhoL = min_density
    var uL = qL1 / rhoL
    var vL = qL2 / rhoL
    var BBL = qL3 * qL3 + qL4 * qL4
    var keL = Float32(0.5) * (qL1 * qL1 + qL2 * qL2) / rhoL
    var pL = (gamma - Float32(1.0)) * (qL5 - keL - Float32(0.5) * BBL)
    if pL < min_pressure:
        pL = min_pressure
    var pstarL = pL + Float32(0.5) * BBL
    var FxL0 = qL1
    var FxL1 = qL1 * uL + pstarL - qL3 * qL3
    var FxL2 = qL1 * vL         - qL3 * qL4
    var FxL3 = Float32(0.0)
    var FxL4 = uL * qL4 - vL * qL3
    var FxL5 = (qL5 + pstarL) * uL - qL3 * (uL * qL3 + vL * qL4)
    var FyL0 = qL2
    var FyL1 = qL2 * uL         - qL4 * qL3
    var FyL2 = qL2 * vL + pstarL - qL4 * qL4
    var FyL3 = vL * qL3 - uL * qL4
    var FyL4 = Float32(0.0)
    var FyL5 = (qL5 + pstarL) * vL - qL4 * (uL * qL3 + vL * qL4)
    var cs2L = gamma * pL / rhoL
    var ca2L = BBL / rhoL
    var sL = cs2L + ca2L
    var discL = sL * sL - Float32(4.0) * cs2L * (qL3 * qL3) / rhoL
    if discL < Float32(0.0):
        discL = Float32(0.0)
    var cf2L = Float32(0.5) * (sL + sqrt(discL))
    var cfL = sqrt(cf2L)
    var speedL = sqrt(uL * uL + vL * vL) + cfL

    # Right internal flux + fast-magnetosonic speed.
    var rhoR = qR0
    if rhoR < min_density:
        rhoR = min_density
    var uR = qR1 / rhoR
    var vR = qR2 / rhoR
    var BBR = qR3 * qR3 + qR4 * qR4
    var keR = Float32(0.5) * (qR1 * qR1 + qR2 * qR2) / rhoR
    var pR = (gamma - Float32(1.0)) * (qR5 - keR - Float32(0.5) * BBR)
    if pR < min_pressure:
        pR = min_pressure
    var pstarR = pR + Float32(0.5) * BBR
    var FxR0 = qR1
    var FxR1 = qR1 * uR + pstarR - qR3 * qR3
    var FxR2 = qR1 * vR         - qR3 * qR4
    var FxR3 = Float32(0.0)
    var FxR4 = uR * qR4 - vR * qR3
    var FxR5 = (qR5 + pstarR) * uR - qR3 * (uR * qR3 + vR * qR4)
    var FyR0 = qR2
    var FyR1 = qR2 * uR         - qR4 * qR3
    var FyR2 = qR2 * vR + pstarR - qR4 * qR4
    var FyR3 = vR * qR3 - uR * qR4
    var FyR4 = Float32(0.0)
    var FyR5 = (qR5 + pstarR) * vR - qR4 * (uR * qR3 + vR * qR4)
    var cs2R = gamma * pR / rhoR
    var ca2R = BBR / rhoR
    var sR = cs2R + ca2R
    var discR = sR * sR - Float32(4.0) * cs2R * (qR3 * qR3) / rhoR
    if discR < Float32(0.0):
        discR = Float32(0.0)
    var cf2R = Float32(0.5) * (sR + sqrt(discR))
    var cfR = sqrt(cf2R)
    var speedR = sqrt(uR * uR + vR * vR) + cfR

    var alpha: Float32 = speedL if speedL > speedR else speedR
    var half = Float32(0.5)
    var out = (fid * NFP + m) * 6
    fstar_out[out + 0] = half * ((FxL0 + FxR0) * nx + (FyL0 + FyR0) * ny) \
                         - half * alpha * (qR0 - qL0)
    fstar_out[out + 1] = half * ((FxL1 + FxR1) * nx + (FyL1 + FyR1) * ny) \
                         - half * alpha * (qR1 - qL1)
    fstar_out[out + 2] = half * ((FxL2 + FxR2) * nx + (FyL2 + FyR2) * ny) \
                         - half * alpha * (qR2 - qL2)
    fstar_out[out + 3] = half * ((FxL3 + FxR3) * nx + (FyL3 + FyR3) * ny) \
                         - half * alpha * (qR3 - qL3)
    fstar_out[out + 4] = half * ((FxL4 + FxR4) * nx + (FyL4 + FyR4) * ny) \
                         - half * alpha * (qR4 - qL4)
    fstar_out[out + 5] = half * ((FxL5 + FxR5) * nx + (FyL5 + FyR5) * ny) \
                         - half * alpha * (qR5 - qL5)


def launch_mhd_face_flux_2d[NP: Int, NFP: Int](
    mut ctx: DeviceContext,
    q:              UnsafePointer[Float32, MutAnyOrigin],
    face_elem:      UnsafePointer[Int32,   MutAnyOrigin],
    face_elem_node: UnsafePointer[Int32,   MutAnyOrigin],
    face_normal:    UnsafePointer[Float32, MutAnyOrigin],
    face_bc_type:   UnsafePointer[Int32,   MutAnyOrigin],
    num_faces:      Int,
    gamma:          Float32,
    min_density:    Float32,
    min_pressure:   Float32,
    fstar_out:      UnsafePointer[Float32, MutAnyOrigin],
) raises:
    var total = num_faces * NFP
    comptime _kernel = mhd_face_flux_kernel_2d[NP, NFP]
    ctx.enqueue_function[_kernel, _kernel](
        q, face_elem, face_elem_node, face_normal, face_bc_type,
        num_faces,
        gamma, min_density, min_pressure,
        fstar_out,
        grid_dim=ceildiv(total, 256),
        block_dim=256,
    )


def mhd_rk_stage_2d[P: Int](
    mut ctx: DeviceContext,
    mesh: LocalMesh2DGpu[P],
    Lift_ref: UnsafePointer[Float32, MutAnyOrigin],
    D_ref:    UnsafePointer[Float32, MutAnyOrigin],
    q_in:     UnsafePointer[Float32, MutAnyOrigin],
    q_a:      UnsafePointer[Float32, MutAnyOrigin],
    q_b:      UnsafePointer[Float32, MutAnyOrigin],
    q_out:    UnsafePointer[Float32, MutAnyOrigin],
    vol_scratch:   UnsafePointer[Float32, MutAnyOrigin],
    fstar_scratch: UnsafePointer[Float32, MutAnyOrigin],
    rhs_scratch:   UnsafePointer[Float32, MutAnyOrigin],
    gamma: Float32, min_density: Float32, min_pressure: Float32,
    a: Float32, b: Float32, cc: Float32, dt: Float32,
) raises:
    # Two launches per stage (down from three).
    _ = vol_scratch
    _ = rhs_scratch
    comptime NP = num_tri_nodes_2d(P)
    comptime NFP = num_edge_nodes(P)
    launch_mhd_face_flux_2d[NP, NFP](
        ctx, q_in,
        mesh.d_face_elem.unsafe_ptr(),
        mesh.d_face_elem_node.unsafe_ptr(),
        mesh.d_face_normal.unsafe_ptr(),
        mesh.d_face_bc_type.unsafe_ptr(),
        mesh.num_faces,
        gamma, min_density, min_pressure,
        fstar_scratch,
    )
    launch_mhd_vol_lift_2d[NP, NFP](
        ctx, q_in, mesh.d_elem_invJ.unsafe_ptr(), D_ref,
        fstar_scratch,
        mesh.d_elem_inv_2A.unsafe_ptr(),
        mesh.d_elem_faces.unsafe_ptr(),
        mesh.d_elem_face_side.unsafe_ptr(),
        mesh.d_elem_canon_to_ref.unsafe_ptr(),
        mesh.d_face_length.unsafe_ptr(),
        Lift_ref, q_a, q_b,
        mesh.num_elements, gamma, min_density, min_pressure,
        a, b, cc, dt, q_out,
    )


# ======================================================================
# Maxwell 2D physics moved to src/local_mesh_2d_gpu_maxwell.mojo.
# Import maxwell_rk_stage_2d (and the underlying kernels if needed)
# directly from that module.
# ======================================================================

# ======================================================================
# GLM-enabled 2D MHD (Dedner div-cleaning).  NC = 7 with component
# layout [rho, rho*u, rho*v, Bx, By, E, psi].
# ======================================================================
#
# Adds the Dedner-Kemm-Kroner-Munz-Schnitzer-Wesenberg generalised
# Lagrange multiplier divergence-cleaning equation:
#
#   dB/dt + div(u B - B u + psi I) = 0           (modified)
#   dpsi/dt + c_h^2 div(B)         = -alpha_d * psi
#
# In 2D the modifications relative to plain ideal MHD are:
#   * Bx-flux x-component += psi
#   * By-flux y-component += psi
#   * psi-flux = c_h^2 * (Bx, By)
#   * Rusanov wave-speed bound: alpha = max(|u_n| + c_f, c_h)
#   * BC_WALL: psi reflects (- psi on the ghost) so the slip wall
#     looks fully insulating to the GLM transport.
#
# psi damping (-alpha_d * psi) is applied via Strang/operator splitting
# AFTER the flux update: q_out[psi] *= exp(-alpha_d * dt).  This is
# semi-discretely exact and stable for any alpha_d, dt.
# ======================================================================

def mhd_glm_volume_rhs_kernel_2d[NP: Int](
    q:            UnsafePointer[Float32, MutAnyOrigin],
    elem_invJ:    UnsafePointer[Float32, MutAnyOrigin],
    D_ref:        UnsafePointer[Float32, MutAnyOrigin],
    num_elements: Int,
    gamma:        Float32,
    min_density:  Float32,
    min_pressure: Float32,
    c_h:          Float32,
    vol_out:      UnsafePointer[Float32, MutAnyOrigin],
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

    var acc0: Float32 = 0.0
    var acc1: Float32 = 0.0
    var acc2: Float32 = 0.0
    var acc3: Float32 = 0.0
    var acc4: Float32 = 0.0
    var acc5: Float32 = 0.0
    var acc6: Float32 = 0.0
    var ch2 = c_h * c_h

    for j in range(NP):
        var base = (elem * NP + j) * 7
        var rho = q[base + 0]
        if rho < min_density:
            rho = min_density
        var mx = q[base + 1]
        var my = q[base + 2]
        var Bx = q[base + 3]
        var By = q[base + 4]
        var E  = q[base + 5]
        var psi = q[base + 6]
        var u = mx / rho
        var v = my / rho
        var BB = Bx * Bx + By * By
        var ke = Float32(0.5) * (mx * mx + my * my) / rho
        var mp = Float32(0.5) * BB
        var p = (gamma - Float32(1.0)) * (E - ke - mp)
        if p < min_pressure:
            p = min_pressure
        var pstar = p + Float32(0.5) * BB

        var Fx0 = mx
        var Fx1 = mx * u + pstar - Bx * Bx
        var Fx2 = mx * v         - Bx * By
        var Fx3 = psi
        var Fx4 = u * By - v * Bx
        var Fx5 = (E + pstar) * u - Bx * (u * Bx + v * By)
        var Fx6 = ch2 * Bx
        var Fy0 = my
        var Fy1 = my * u         - By * Bx
        var Fy2 = my * v + pstar - By * By
        var Fy3 = v * Bx - u * By
        var Fy4 = psi
        var Fy5 = (E + pstar) * v - By * (u * Bx + v * By)
        var Fy6 = ch2 * By

        var D_r = D_ref[0 * NP * NP + i * NP + j]
        var D_s = D_ref[1 * NP * NP + i * NP + j]

        acc0 += (iJ00 * Fx0 + iJ01 * Fy0) * D_r + (iJ10 * Fx0 + iJ11 * Fy0) * D_s
        acc1 += (iJ00 * Fx1 + iJ01 * Fy1) * D_r + (iJ10 * Fx1 + iJ11 * Fy1) * D_s
        acc2 += (iJ00 * Fx2 + iJ01 * Fy2) * D_r + (iJ10 * Fx2 + iJ11 * Fy2) * D_s
        acc3 += (iJ00 * Fx3 + iJ01 * Fy3) * D_r + (iJ10 * Fx3 + iJ11 * Fy3) * D_s
        acc4 += (iJ00 * Fx4 + iJ01 * Fy4) * D_r + (iJ10 * Fx4 + iJ11 * Fy4) * D_s
        acc5 += (iJ00 * Fx5 + iJ01 * Fy5) * D_r + (iJ10 * Fx5 + iJ11 * Fy5) * D_s
        acc6 += (iJ00 * Fx6 + iJ01 * Fy6) * D_r + (iJ10 * Fx6 + iJ11 * Fy6) * D_s

    var out = (elem * NP + i) * 7
    vol_out[out + 0] = acc0
    vol_out[out + 1] = acc1
    vol_out[out + 2] = acc2
    vol_out[out + 3] = acc3
    vol_out[out + 4] = acc4
    vol_out[out + 5] = acc5
    vol_out[out + 6] = acc6


# ----------------------------------------------------------------------
# Fused IdealMHD-GLM volume + lift + RK update (2D, NC=7).
# ----------------------------------------------------------------------
# Same fusion pattern with the GLM divergence-cleaning flux additions:
# Fx_Bx += psi, Fy_By += psi, F_psi = c_h^2 * B.
# ----------------------------------------------------------------------

def mhd_glm_vol_lift_combine_rk_kernel_2d[NP: Int, NFP: Int](
    q_in:              UnsafePointer[Float32, MutAnyOrigin],
    elem_invJ:         UnsafePointer[Float32, MutAnyOrigin],
    D_ref:             UnsafePointer[Float32, MutAnyOrigin],
    fstar:             UnsafePointer[Float32, MutAnyOrigin],
    elem_inv_2A:       UnsafePointer[Float32, MutAnyOrigin],
    elem_faces:        UnsafePointer[Int32,   MutAnyOrigin],
    elem_face_side:    UnsafePointer[Int32,   MutAnyOrigin],
    elem_canon_to_ref: UnsafePointer[Int32,   MutAnyOrigin],
    face_length:       UnsafePointer[Float32, MutAnyOrigin],
    Lift_ref:          UnsafePointer[Float32, MutAnyOrigin],
    q_a:               UnsafePointer[Float32, MutAnyOrigin],
    q_b:               UnsafePointer[Float32, MutAnyOrigin],
    num_elements:      Int,
    gamma:             Float32,
    min_density:       Float32,
    min_pressure:      Float32,
    c_h:               Float32,
    a: Float32, b: Float32, cc: Float32, dt: Float32,
    q_out:             UnsafePointer[Float32, MutAnyOrigin],
):
    var tid = Int(global_idx.x)
    var total = num_elements * NP
    if tid >= total:
        return
    var elem = tid // NP
    var i    = tid %  NP

    # ---- Volume RHS contribution (NC=7, GLM flux additions).
    var iJ00 = elem_invJ[elem * 4 + 0]
    var iJ01 = elem_invJ[elem * 4 + 1]
    var iJ10 = elem_invJ[elem * 4 + 2]
    var iJ11 = elem_invJ[elem * 4 + 3]
    var acc0: Float32 = 0.0
    var acc1: Float32 = 0.0
    var acc2: Float32 = 0.0
    var acc3: Float32 = 0.0
    var acc4: Float32 = 0.0
    var acc5: Float32 = 0.0
    var acc6: Float32 = 0.0
    var ch2 = c_h * c_h
    for j in range(NP):
        var base = (elem * NP + j) * 7
        var rho = q_in[base + 0]
        if rho < min_density:
            rho = min_density
        var mx = q_in[base + 1]
        var my = q_in[base + 2]
        var Bx = q_in[base + 3]
        var By = q_in[base + 4]
        var E  = q_in[base + 5]
        var psi = q_in[base + 6]
        var u = mx / rho
        var v = my / rho
        var BB = Bx * Bx + By * By
        var ke = Float32(0.5) * (mx * mx + my * my) / rho
        var mp = Float32(0.5) * BB
        var p = (gamma - Float32(1.0)) * (E - ke - mp)
        if p < min_pressure:
            p = min_pressure
        var pstar = p + Float32(0.5) * BB
        var Fx0 = mx
        var Fx1 = mx * u + pstar - Bx * Bx
        var Fx2 = mx * v         - Bx * By
        var Fx3 = psi
        var Fx4 = u * By - v * Bx
        var Fx5 = (E + pstar) * u - Bx * (u * Bx + v * By)
        var Fx6 = ch2 * Bx
        var Fy0 = my
        var Fy1 = my * u         - By * Bx
        var Fy2 = my * v + pstar - By * By
        var Fy3 = v * Bx - u * By
        var Fy4 = psi
        var Fy5 = (E + pstar) * v - By * (u * Bx + v * By)
        var Fy6 = ch2 * By
        var D_r = D_ref[0 * NP * NP + i * NP + j]
        var D_s = D_ref[1 * NP * NP + i * NP + j]
        acc0 += (iJ00 * Fx0 + iJ01 * Fy0) * D_r + (iJ10 * Fx0 + iJ11 * Fy0) * D_s
        acc1 += (iJ00 * Fx1 + iJ01 * Fy1) * D_r + (iJ10 * Fx1 + iJ11 * Fy1) * D_s
        acc2 += (iJ00 * Fx2 + iJ01 * Fy2) * D_r + (iJ10 * Fx2 + iJ11 * Fy2) * D_s
        acc3 += (iJ00 * Fx3 + iJ01 * Fy3) * D_r + (iJ10 * Fx3 + iJ11 * Fy3) * D_s
        acc4 += (iJ00 * Fx4 + iJ01 * Fy4) * D_r + (iJ10 * Fx4 + iJ11 * Fy4) * D_s
        acc5 += (iJ00 * Fx5 + iJ01 * Fy5) * D_r + (iJ10 * Fx5 + iJ11 * Fy5) * D_s
        acc6 += (iJ00 * Fx6 + iJ01 * Fy6) * D_r + (iJ10 * Fx6 + iJ11 * Fy6) * D_s

    # ---- Lift contribution (NC=7).
    var inv_2A = elem_inv_2A[elem]
    var face0: Float32 = 0.0
    var face1: Float32 = 0.0
    var face2: Float32 = 0.0
    var face3: Float32 = 0.0
    var face4: Float32 = 0.0
    var face5: Float32 = 0.0
    var face6: Float32 = 0.0
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
            var sLf = sign * flen * Lim
            var fbase = (fid * NFP + m) * 7
            face0 += sLf * fstar[fbase + 0]
            face1 += sLf * fstar[fbase + 1]
            face2 += sLf * fstar[fbase + 2]
            face3 += sLf * fstar[fbase + 3]
            face4 += sLf * fstar[fbase + 4]
            face5 += sLf * fstar[fbase + 5]
            face6 += sLf * fstar[fbase + 6]

    # ---- Combine + RK update.
    var idx = (elem * NP + i) * 7
    var rhs0 = acc0 - inv_2A * face0
    var rhs1 = acc1 - inv_2A * face1
    var rhs2 = acc2 - inv_2A * face2
    var rhs3 = acc3 - inv_2A * face3
    var rhs4 = acc4 - inv_2A * face4
    var rhs5 = acc5 - inv_2A * face5
    var rhs6 = acc6 - inv_2A * face6
    q_out[idx + 0] = a * q_a[idx + 0] + b * q_b[idx + 0] + cc * dt * rhs0
    q_out[idx + 1] = a * q_a[idx + 1] + b * q_b[idx + 1] + cc * dt * rhs1
    q_out[idx + 2] = a * q_a[idx + 2] + b * q_b[idx + 2] + cc * dt * rhs2
    q_out[idx + 3] = a * q_a[idx + 3] + b * q_b[idx + 3] + cc * dt * rhs3
    q_out[idx + 4] = a * q_a[idx + 4] + b * q_b[idx + 4] + cc * dt * rhs4
    q_out[idx + 5] = a * q_a[idx + 5] + b * q_b[idx + 5] + cc * dt * rhs5
    q_out[idx + 6] = a * q_a[idx + 6] + b * q_b[idx + 6] + cc * dt * rhs6


def launch_mhd_glm_vol_lift_2d[NP: Int, NFP: Int](
    mut ctx: DeviceContext,
    q_in:              UnsafePointer[Float32, MutAnyOrigin],
    elem_invJ:         UnsafePointer[Float32, MutAnyOrigin],
    D_ref:             UnsafePointer[Float32, MutAnyOrigin],
    fstar:             UnsafePointer[Float32, MutAnyOrigin],
    elem_inv_2A:       UnsafePointer[Float32, MutAnyOrigin],
    elem_faces:        UnsafePointer[Int32,   MutAnyOrigin],
    elem_face_side:    UnsafePointer[Int32,   MutAnyOrigin],
    elem_canon_to_ref: UnsafePointer[Int32,   MutAnyOrigin],
    face_length:       UnsafePointer[Float32, MutAnyOrigin],
    Lift_ref:          UnsafePointer[Float32, MutAnyOrigin],
    q_a:               UnsafePointer[Float32, MutAnyOrigin],
    q_b:               UnsafePointer[Float32, MutAnyOrigin],
    num_elements:      Int,
    gamma:             Float32,
    min_density:       Float32,
    min_pressure:      Float32,
    c_h:               Float32,
    a: Float32, b: Float32, cc: Float32, dt: Float32,
    q_out:             UnsafePointer[Float32, MutAnyOrigin],
) raises:
    var total = num_elements * NP
    comptime _kernel = mhd_glm_vol_lift_combine_rk_kernel_2d[NP, NFP]
    ctx.enqueue_function[_kernel, _kernel](
        q_in, elem_invJ, D_ref, fstar,
        elem_inv_2A, elem_faces, elem_face_side, elem_canon_to_ref,
        face_length, Lift_ref, q_a, q_b,
        num_elements, gamma, min_density, min_pressure, c_h,
        a, b, cc, dt, q_out,
        grid_dim=ceildiv(total, 256),
        block_dim=256,
    )


def launch_mhd_glm_volume_rhs_2d[NP: Int](
    mut ctx: DeviceContext,
    q:            UnsafePointer[Float32, MutAnyOrigin],
    elem_invJ:    UnsafePointer[Float32, MutAnyOrigin],
    D_ref:        UnsafePointer[Float32, MutAnyOrigin],
    num_elements: Int,
    gamma:        Float32,
    min_density:  Float32,
    min_pressure: Float32,
    c_h:          Float32,
    vol_out:      UnsafePointer[Float32, MutAnyOrigin],
) raises:
    var total = num_elements * NP
    comptime _kernel = mhd_glm_volume_rhs_kernel_2d[NP]
    ctx.enqueue_function[_kernel, _kernel](
        q, elem_invJ, D_ref, num_elements,
        gamma, min_density, min_pressure, c_h, vol_out,
        grid_dim=ceildiv(total, 256),
        block_dim=256,
    )


def mhd_glm_face_flux_kernel_2d[NP: Int, NFP: Int](
    q:              UnsafePointer[Float32, MutAnyOrigin],
    face_elem:      UnsafePointer[Int32,   MutAnyOrigin],
    face_elem_node: UnsafePointer[Int32,   MutAnyOrigin],
    face_normal:    UnsafePointer[Float32, MutAnyOrigin],
    face_bc_type:   UnsafePointer[Int32,   MutAnyOrigin],
    num_faces:      Int,
    gamma:          Float32,
    min_density:    Float32,
    min_pressure:   Float32,
    c_h:            Float32,
    fstar_out:      UnsafePointer[Float32, MutAnyOrigin],
):
    var tid = Int(global_idx.x)
    var total = num_faces * NFP
    if tid >= total:
        return
    var fid = tid // NFP
    var m   = tid %  NFP

    var nx = face_normal[fid * 2 + 0]
    var ny = face_normal[fid * 2 + 1]
    var bc_type = face_bc_type[fid]

    var e_l = Int(face_elem[fid * 2 + 0])
    var n_l = Int(face_elem_node[(fid * 2 + 0) * NFP + m])
    var l_off = (e_l * NP + n_l) * 7
    var qL0 = q[l_off + 0]
    var qL1 = q[l_off + 1]
    var qL2 = q[l_off + 2]
    var qL3 = q[l_off + 3]
    var qL4 = q[l_off + 4]
    var qL5 = q[l_off + 5]
    var qL6 = q[l_off + 6]

    var qR0: Float32
    var qR1: Float32
    var qR2: Float32
    var qR3: Float32
    var qR4: Float32
    var qR5: Float32
    var qR6: Float32
    if bc_type == BC_INTERIOR:
        var e_r = Int(face_elem[fid * 2 + 1])
        var n_r = Int(face_elem_node[(fid * 2 + 1) * NFP + m])
        var r_off = (e_r * NP + n_r) * 7
        qR0 = q[r_off + 0]
        qR1 = q[r_off + 1]
        qR2 = q[r_off + 2]
        qR3 = q[r_off + 3]
        qR4 = q[r_off + 4]
        qR5 = q[r_off + 5]
        qR6 = q[r_off + 6]
    elif bc_type == BC_WALL:
        var m_n = qL1 * nx + qL2 * ny
        var B_n = qL3 * nx + qL4 * ny
        qR0 = qL0
        qR1 = qL1 - Float32(2.0) * m_n * nx
        qR2 = qL2 - Float32(2.0) * m_n * ny
        qR3 = qL3 - Float32(2.0) * B_n * nx
        qR4 = qL4 - Float32(2.0) * B_n * ny
        qR5 = qL5
        qR6 = -qL6
    else:
        qR0 = qL0
        qR1 = qL1
        qR2 = qL2
        qR3 = qL3
        qR4 = qL4
        qR5 = qL5
        qR6 = qL6

    var ch2 = c_h * c_h

    # Left internal flux + fast-magnetosonic speed.
    var rhoL = qL0
    if rhoL < min_density:
        rhoL = min_density
    var uL = qL1 / rhoL
    var vL = qL2 / rhoL
    var BBL = qL3 * qL3 + qL4 * qL4
    var keL = Float32(0.5) * (qL1 * qL1 + qL2 * qL2) / rhoL
    var pL = (gamma - Float32(1.0)) * (qL5 - keL - Float32(0.5) * BBL)
    if pL < min_pressure:
        pL = min_pressure
    var pstarL = pL + Float32(0.5) * BBL
    var FxL0 = qL1
    var FxL1 = qL1 * uL + pstarL - qL3 * qL3
    var FxL2 = qL1 * vL         - qL3 * qL4
    var FxL3 = qL6
    var FxL4 = uL * qL4 - vL * qL3
    var FxL5 = (qL5 + pstarL) * uL - qL3 * (uL * qL3 + vL * qL4)
    var FxL6 = ch2 * qL3
    var FyL0 = qL2
    var FyL1 = qL2 * uL         - qL4 * qL3
    var FyL2 = qL2 * vL + pstarL - qL4 * qL4
    var FyL3 = vL * qL3 - uL * qL4
    var FyL4 = qL6
    var FyL5 = (qL5 + pstarL) * vL - qL4 * (uL * qL3 + vL * qL4)
    var FyL6 = ch2 * qL4
    var cs2L = gamma * pL / rhoL
    var ca2L = BBL / rhoL
    var sL = cs2L + ca2L
    var discL = sL * sL - Float32(4.0) * cs2L * (qL3 * qL3) / rhoL
    if discL < Float32(0.0):
        discL = Float32(0.0)
    var cf2L = Float32(0.5) * (sL + sqrt(discL))
    var cfL = sqrt(cf2L)
    var speedL = sqrt(uL * uL + vL * vL) + cfL
    if c_h > speedL:
        speedL = c_h

    # Right internal flux + fast-magnetosonic speed.
    var rhoR = qR0
    if rhoR < min_density:
        rhoR = min_density
    var uR = qR1 / rhoR
    var vR = qR2 / rhoR
    var BBR = qR3 * qR3 + qR4 * qR4
    var keR = Float32(0.5) * (qR1 * qR1 + qR2 * qR2) / rhoR
    var pR = (gamma - Float32(1.0)) * (qR5 - keR - Float32(0.5) * BBR)
    if pR < min_pressure:
        pR = min_pressure
    var pstarR = pR + Float32(0.5) * BBR
    var FxR0 = qR1
    var FxR1 = qR1 * uR + pstarR - qR3 * qR3
    var FxR2 = qR1 * vR         - qR3 * qR4
    var FxR3 = qR6
    var FxR4 = uR * qR4 - vR * qR3
    var FxR5 = (qR5 + pstarR) * uR - qR3 * (uR * qR3 + vR * qR4)
    var FxR6 = ch2 * qR3
    var FyR0 = qR2
    var FyR1 = qR2 * uR         - qR4 * qR3
    var FyR2 = qR2 * vR + pstarR - qR4 * qR4
    var FyR3 = vR * qR3 - uR * qR4
    var FyR4 = qR6
    var FyR5 = (qR5 + pstarR) * vR - qR4 * (uR * qR3 + vR * qR4)
    var FyR6 = ch2 * qR4
    var cs2R = gamma * pR / rhoR
    var ca2R = BBR / rhoR
    var sR = cs2R + ca2R
    var discR = sR * sR - Float32(4.0) * cs2R * (qR3 * qR3) / rhoR
    if discR < Float32(0.0):
        discR = Float32(0.0)
    var cf2R = Float32(0.5) * (sR + sqrt(discR))
    var cfR = sqrt(cf2R)
    var speedR = sqrt(uR * uR + vR * vR) + cfR
    if c_h > speedR:
        speedR = c_h

    var alpha: Float32 = speedL if speedL > speedR else speedR
    var half = Float32(0.5)
    var out = (fid * NFP + m) * 7
    fstar_out[out + 0] = half * ((FxL0 + FxR0) * nx + (FyL0 + FyR0) * ny) \
                         - half * alpha * (qR0 - qL0)
    fstar_out[out + 1] = half * ((FxL1 + FxR1) * nx + (FyL1 + FyR1) * ny) \
                         - half * alpha * (qR1 - qL1)
    fstar_out[out + 2] = half * ((FxL2 + FxR2) * nx + (FyL2 + FyR2) * ny) \
                         - half * alpha * (qR2 - qL2)
    fstar_out[out + 3] = half * ((FxL3 + FxR3) * nx + (FyL3 + FyR3) * ny) \
                         - half * alpha * (qR3 - qL3)
    fstar_out[out + 4] = half * ((FxL4 + FxR4) * nx + (FyL4 + FyR4) * ny) \
                         - half * alpha * (qR4 - qL4)
    fstar_out[out + 5] = half * ((FxL5 + FxR5) * nx + (FyL5 + FyR5) * ny) \
                         - half * alpha * (qR5 - qL5)
    fstar_out[out + 6] = half * ((FxL6 + FxR6) * nx + (FyL6 + FyR6) * ny) \
                         - half * alpha * (qR6 - qL6)


def launch_mhd_glm_face_flux_2d[NP: Int, NFP: Int](
    mut ctx: DeviceContext,
    q:              UnsafePointer[Float32, MutAnyOrigin],
    face_elem:      UnsafePointer[Int32,   MutAnyOrigin],
    face_elem_node: UnsafePointer[Int32,   MutAnyOrigin],
    face_normal:    UnsafePointer[Float32, MutAnyOrigin],
    face_bc_type:   UnsafePointer[Int32,   MutAnyOrigin],
    num_faces:      Int,
    gamma:          Float32,
    min_density:    Float32,
    min_pressure:   Float32,
    c_h:            Float32,
    fstar_out:      UnsafePointer[Float32, MutAnyOrigin],
) raises:
    var total = num_faces * NFP
    comptime _kernel = mhd_glm_face_flux_kernel_2d[NP, NFP]
    ctx.enqueue_function[_kernel, _kernel](
        q, face_elem, face_elem_node, face_normal, face_bc_type,
        num_faces,
        gamma, min_density, min_pressure, c_h,
        fstar_out,
        grid_dim=ceildiv(total, 256),
        block_dim=256,
    )


# Operator-splitting psi damping: q[psi] *= exp(-alpha_d * dt).
def mhd_glm_psi_damp_kernel_2d[NP: Int](
    q:            UnsafePointer[Float32, MutAnyOrigin],
    num_elements: Int,
    decay:        Float32,
):
    var tid = Int(global_idx.x)
    var total = num_elements * NP
    if tid >= total:
        return
    var idx = tid * 7 + 6
    q[idx] = q[idx] * decay


def launch_mhd_glm_psi_damp_2d[NP: Int](
    mut ctx: DeviceContext,
    q:            UnsafePointer[Float32, MutAnyOrigin],
    num_elements: Int,
    alpha_d:      Float32,
    dt:           Float32,
) raises:
    if alpha_d <= Float32(0.0):
        return
    # exp(-alpha_d * dt) computed on host since the kernel is per-node
    # and the decay factor is the same for every node.
    from std.math import exp
    var decay = Float32(exp(-Float64(alpha_d) * Float64(dt)))
    var total = num_elements * NP
    comptime _kernel = mhd_glm_psi_damp_kernel_2d[NP]
    ctx.enqueue_function[_kernel, _kernel](
        q, num_elements, decay,
        grid_dim=ceildiv(total, 256),
        block_dim=256,
    )


def mhd_glm_rk_stage_2d[P: Int](
    mut ctx: DeviceContext,
    mesh: LocalMesh2DGpu[P],
    Lift_ref: UnsafePointer[Float32, MutAnyOrigin],
    D_ref:    UnsafePointer[Float32, MutAnyOrigin],
    q_in:     UnsafePointer[Float32, MutAnyOrigin],
    q_a:      UnsafePointer[Float32, MutAnyOrigin],
    q_b:      UnsafePointer[Float32, MutAnyOrigin],
    q_out:    UnsafePointer[Float32, MutAnyOrigin],
    vol_scratch:   UnsafePointer[Float32, MutAnyOrigin],
    fstar_scratch: UnsafePointer[Float32, MutAnyOrigin],
    rhs_scratch:   UnsafePointer[Float32, MutAnyOrigin],
    gamma: Float32, min_density: Float32, min_pressure: Float32,
    c_h: Float32, alpha_d: Float32,
    a: Float32, b: Float32, cc: Float32, dt: Float32,
) raises:
    # Two flux launches per stage (down from three) plus psi damping.
    _ = vol_scratch
    _ = rhs_scratch
    comptime NP = num_tri_nodes_2d(P)
    comptime NFP = num_edge_nodes(P)
    launch_mhd_glm_face_flux_2d[NP, NFP](
        ctx, q_in,
        mesh.d_face_elem.unsafe_ptr(),
        mesh.d_face_elem_node.unsafe_ptr(),
        mesh.d_face_normal.unsafe_ptr(),
        mesh.d_face_bc_type.unsafe_ptr(),
        mesh.num_faces,
        gamma, min_density, min_pressure, c_h,
        fstar_scratch,
    )
    launch_mhd_glm_vol_lift_2d[NP, NFP](
        ctx, q_in, mesh.d_elem_invJ.unsafe_ptr(), D_ref,
        fstar_scratch,
        mesh.d_elem_inv_2A.unsafe_ptr(),
        mesh.d_elem_faces.unsafe_ptr(),
        mesh.d_elem_face_side.unsafe_ptr(),
        mesh.d_elem_canon_to_ref.unsafe_ptr(),
        mesh.d_face_length.unsafe_ptr(),
        Lift_ref, q_a, q_b,
        mesh.num_elements, gamma, min_density, min_pressure, c_h,
        a, b, cc, dt, q_out,
    )
    # NOTE: the alpha_d argument is no longer applied here.  Operator-
    # splitting psi damping must be invoked ONCE PER SSPRK3 step by the
    # caller via `launch_mhd_glm_psi_damp_2d`, not once per stage --
    # otherwise the decay gets applied three times per timestep.  The
    # parameter is retained for backward compatibility with existing
    # drivers that pass alpha_d (the GLM Alfven and psi-transport
    # benches use alpha_d = 0 so the change is observably a no-op for
    # them).
    _ = alpha_d


# ----------------------------------------------------------------------
# Barth-Jespersen slope limiter (Venkatakrishnan-smoothed), 2D GPU.
# ----------------------------------------------------------------------
# Post-stage limiter that scales every nodal deviation from the local
# mean by the tightest theta keeping the scaled deviation within the
# (min, max) cell-average range over self + 3 face neighbours, sampled
# on component 0 (density).  The scaling is then applied uniformly to
# every NC component so coupled quantities (e.g. mass + momentum) stay
# consistent.  Venkat smoothing (epsilon>0) avoids over-limiting smooth
# regions -- epsilon=0 recovers raw BJ, which kills P+1 accuracy even
# where the solution is smooth.
#
# Two-pass structure:
#   1. Caller first runs `cell_avg_kernel_2d[NP, NC]` into `d_cell_avg`
#      (num_elements * NC Float32).
#   2. `bj_limit_kernel_2d[NP, NC]` -- one thread per element.  Reads
#      own cell_avg[NC], peeks at 3 neighbours' component-0 averages to
#      build nbr_min / nbr_max, computes Venkat theta on the own node
#      deviations (component 0), and scales all NP*NC values in-place.
#
# Boundary faces have face_elem[fid*2+1] == face_elem[fid*2+0], so
# neighbour-lookup self-matches and contributes nothing to the min/max
# range -- matching the CPU convention for bc-limited cells.
# ----------------------------------------------------------------------

def bj_limit_kernel_2d[NP: Int, NC: Int](
    q:            UnsafePointer[Float32, MutAnyOrigin],
    cell_avg:     UnsafePointer[Float32, MutAnyOrigin],
    elem_faces:   UnsafePointer[Int32,   MutAnyOrigin],
    face_elem:    UnsafePointer[Int32,   MutAnyOrigin],
    num_elements: Int,
    venkat_eps2:  Float32,
):
    var elem = Int(global_idx.x)
    if elem >= num_elements:
        return

    var own_avg = cell_avg[elem * NC + 0]
    var nbr_min = own_avg
    var nbr_max = own_avg
    for lf in range(3):
        var fid = Int(elem_faces[elem * 3 + lf])
        var e_l = Int(face_elem[fid * 2 + 0])
        var e_r = Int(face_elem[fid * 2 + 1])
        var n = e_r if e_l == elem else e_l
        if n == elem:
            continue   # boundary face: don't constrain
        var a = cell_avg[n * NC + 0]
        if a < nbr_min: nbr_min = a
        if a > nbr_max: nbr_max = a

    # Venkat-smoothed theta on the density component.
    var theta: Float32 = 1.0
    var tiny: Float32 = 1.0e-30
    for nn in range(NP):
        var node_val = q[(elem * NP + nn) * NC + 0]
        var delta = node_val - own_avg
        var d_abs = delta if delta >= Float32(0.0) else -delta
        if d_abs <= tiny:
            continue
        var D: Float32
        if delta > Float32(0.0):
            D = nbr_max - own_avg
        else:
            D = own_avg - nbr_min
        if D < Float32(0.0):
            D = Float32(0.0)
        var D2 = D * D
        var d2 = d_abs * d_abs
        var Dd = D * d_abs
        var numer = D2 + Float32(2.0) * Dd + venkat_eps2
        var denom = D2 + Float32(2.0) * d2 + Dd + venkat_eps2
        var alpha = numer / denom
        if alpha < theta:
            theta = alpha

    if not (theta < Float32(1.0)):
        return   # smooth cell, leave it alone

    # Apply theta uniformly to every node, every component.  Base
    # per-component mean is the cell_avg buffer we already have.
    for nn in range(NP):
        for c in range(NC):
            var offset = (elem * NP + nn) * NC + c
            var base = cell_avg[elem * NC + c]
            q[offset] = base + theta * (q[offset] - base)


def launch_bj_limit_2d[NP: Int, NC: Int](
    mut ctx: DeviceContext,
    q:            UnsafePointer[Float32, MutAnyOrigin],
    cell_avg:     UnsafePointer[Float32, MutAnyOrigin],
    elem_faces:   UnsafePointer[Int32,   MutAnyOrigin],
    face_elem:    UnsafePointer[Int32,   MutAnyOrigin],
    num_elements: Int,
    venkat_eps2:  Float32,
) raises:
    comptime _kernel = bj_limit_kernel_2d[NP, NC]
    ctx.enqueue_function[_kernel, _kernel](
        q, cell_avg, elem_faces, face_elem, num_elements, venkat_eps2,
        grid_dim=ceildiv(num_elements, 256),
        block_dim=256,
    )


# Convenience two-pass orchestrator: run the mass-weighted cell mean
# then the BJ limiter on top of a caller-supplied scratch buffer.
# Drivers call this between RK stages to enforce monotonicity on
# shocked problems.  `node_weights` is `ReferenceElement2DGpu.d_node_weights`;
# using the unweighted `cell_avg_kernel_2d` here would systematically
# drift shock speeds at P>=2 because Lagrange-P>=2 node weights aren't
# uniform (at P=2 the 3 vertex weights are 0, the 3 midpoint weights
# are 1/3).

def bj_limit_full_2d[P: Int, NC: Int](
    mut ctx: DeviceContext,
    mesh: LocalMesh2DGpu[P],
    q:             UnsafePointer[Float32, MutAnyOrigin],
    node_weights:  UnsafePointer[Float32, MutAnyOrigin],
    cell_mean_scratch: UnsafePointer[Float32, MutAnyOrigin],
    venkat_eps:    Float32 = Float32(0.1),
) raises:
    comptime NP = num_tri_nodes_2d(P)
    launch_cell_mean_2d[NP, NC](
        ctx, q, node_weights, mesh.num_elements, cell_mean_scratch,
    )
    launch_bj_limit_2d[NP, NC](
        ctx, q, cell_mean_scratch,
        mesh.d_elem_faces.unsafe_ptr(),
        mesh.d_face_elem.unsafe_ptr(),
        mesh.num_elements,
        venkat_eps * venkat_eps,
    )
