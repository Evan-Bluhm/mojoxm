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


# ======================================================================
# 2D Euler (NC=4) moved to src/local_mesh_2d_gpu_euler.mojo.  Import
# euler_rk_stage_2d (Rusanov) or euler_rk_stage_hllc_2d (HLLC) from
# that module.
# ======================================================================

# ======================================================================
# 2D Shallow Water (NC=3) moved to src/local_mesh_2d_gpu_sw.mojo.
# Import sw_rk_stage_2d (Rusanov) or sw_rk_stage_hll_2d (HLL) from
# that module.
# ======================================================================

# ======================================================================
# Plain ideal 2D MHD (NC=6, no GLM) moved to
# src/local_mesh_2d_gpu_mhd.mojo.  Import mhd_rk_stage_2d (and the
# underlying kernels if needed) from that module.  For divergence
# cleaning use the parallel NC=7 GLM stack in
# src/local_mesh_2d_gpu_mhd_glm.mojo.
# ======================================================================

# ======================================================================
# Maxwell 2D physics moved to src/local_mesh_2d_gpu_maxwell.mojo.
# Import maxwell_rk_stage_2d (and the underlying kernels if needed)
# directly from that module.
# ======================================================================

# ======================================================================
# GLM-enabled 2D MHD (Dedner div-cleaning) moved to
# src/local_mesh_2d_gpu_mhd_glm.mojo.  Import mhd_glm_rk_stage_2d /
# launch_mhd_glm_psi_damp_2d from that module.
# ======================================================================

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
