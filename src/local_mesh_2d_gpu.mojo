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
    comptime NP = num_tri_nodes_2d(P)
    comptime NFP = num_edge_nodes(P)
    launch_advection_volume_rhs_2d[NP](
        ctx, q_in, mesh.d_elem_invJ.unsafe_ptr(), D_ref,
        mesh.num_elements, vx, vy, vol_scratch,
    )
    launch_advection_face_flux_2d[NP, NFP](
        ctx, q_in,
        mesh.d_face_elem.unsafe_ptr(),
        mesh.d_face_elem_node.unsafe_ptr(),
        mesh.d_face_normal.unsafe_ptr(),
        mesh.d_face_bc_type.unsafe_ptr(),
        mesh.num_faces, vx, vy, inflow_q, fstar_scratch,
    )
    launch_advection_lift_combine_2d[NP, NFP](
        ctx, vol_scratch, fstar_scratch,
        mesh.d_elem_inv_2A.unsafe_ptr(),
        mesh.d_elem_faces.unsafe_ptr(),
        mesh.d_elem_face_side.unsafe_ptr(),
        mesh.d_elem_canon_to_ref.unsafe_ptr(),
        mesh.d_face_length.unsafe_ptr(),
        Lift_ref,
        mesh.num_elements, rhs_scratch,
    )
    launch_rk_update_2d[NP, 1](
        ctx, q_a, q_b, rhs_scratch,
        mesh.num_elements, a, b, cc, dt, q_out,
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
    comptime NP = num_tri_nodes_2d(P)
    comptime NFP = num_edge_nodes(P)
    launch_euler_volume_rhs_2d[NP](
        ctx, q_in, mesh.d_elem_invJ.unsafe_ptr(), D_ref,
        mesh.num_elements, gamma, min_density, min_pressure, vol_scratch,
    )
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
    launch_lift_combine_2d[NP, NFP, 4](
        ctx, vol_scratch, fstar_scratch,
        mesh.d_elem_inv_2A.unsafe_ptr(),
        mesh.d_elem_faces.unsafe_ptr(),
        mesh.d_elem_face_side.unsafe_ptr(),
        mesh.d_elem_canon_to_ref.unsafe_ptr(),
        mesh.d_face_length.unsafe_ptr(),
        Lift_ref,
        mesh.num_elements, rhs_scratch,
    )
    launch_rk_update_2d[NP, 4](
        ctx, q_a, q_b, rhs_scratch,
        mesh.num_elements, a, b, cc, dt, q_out,
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
    comptime NP = num_tri_nodes_2d(P)
    comptime NFP = num_edge_nodes(P)
    launch_sw_volume_rhs_2d[NP](
        ctx, q_in, mesh.d_elem_invJ.unsafe_ptr(), D_ref,
        mesh.num_elements, g, min_h, vol_scratch,
    )
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
    launch_lift_combine_2d[NP, NFP, 3](
        ctx, vol_scratch, fstar_scratch,
        mesh.d_elem_inv_2A.unsafe_ptr(),
        mesh.d_elem_faces.unsafe_ptr(),
        mesh.d_elem_face_side.unsafe_ptr(),
        mesh.d_elem_canon_to_ref.unsafe_ptr(),
        mesh.d_face_length.unsafe_ptr(),
        Lift_ref,
        mesh.num_elements, rhs_scratch,
    )
    launch_rk_update_2d[NP, 3](
        ctx, q_a, q_b, rhs_scratch,
        mesh.num_elements, a, b, cc, dt, q_out,
    )
