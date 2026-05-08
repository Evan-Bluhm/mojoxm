# ======================================================================
# local_mesh_2d_gpu_advection.mojo -- 2D scalar advection (NC=1)
# ======================================================================
# State: scalar q.  Single hyperbolic eigenvalue, so a single upwind
# flux (no Rusanov / HLLC variants).  Driver entry point:
# `advection_rk_stage_2d[P]` runs one SSPRK3 stage as 2 launches:
# face-flux kernel (`launch_advection_face_flux_2d`) writes fstar to
# global, then a per-(elem, node) fused kernel
# (`launch_advection_vol_lift_2d`) computes the volume RHS locally,
# applies the face-lift, and does the RK update.
#
# Public surface used outside this module:
#   * `advection_rk_stage_2d[P]`            -- driver SSPRK3 stage
#   * `launch_advection_volume_rhs_2d[NP]`  -- standalone volume-rhs
#     and `launch_advection_face_flux_2d[NP, NFP]` are exercised by
#     `local_mesh_2d_gpu_test` to verify the individual stages.
# ======================================================================

from src.local_mesh_2d_gpu import LocalMesh2DGpu
from src.reference_2d import num_tri_nodes_2d, num_edge_nodes
from src.boundary import BC_INTERIOR, BC_INFLOW
from std.gpu import global_idx
from std.gpu.host import DeviceContext
from std.math import ceildiv


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


def advection_volume_rhs_kernel_2d[
    NP: Int
](
    q: UnsafePointer[Float32, MutAnyOrigin],
    elem_invJ: UnsafePointer[Float32, MutAnyOrigin],
    D_ref: UnsafePointer[Float32, MutAnyOrigin],
    num_elements: Int,
    vx: Float32,
    vy: Float32,
    vol_out: UnsafePointer[Float32, MutAnyOrigin],
):
    var tid = Int(global_idx.x)
    var total = num_elements * NP
    if tid >= total:
        return
    var elem = tid // NP
    var i = tid % NP

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


def launch_advection_volume_rhs_2d[
    NP: Int
](
    mut ctx: DeviceContext,
    q: UnsafePointer[Float32, MutAnyOrigin],
    elem_invJ: UnsafePointer[Float32, MutAnyOrigin],
    D_ref: UnsafePointer[Float32, MutAnyOrigin],
    num_elements: Int,
    vx: Float32,
    vy: Float32,
    vol_out: UnsafePointer[Float32, MutAnyOrigin],
) raises:
    var total = num_elements * NP
    comptime _kernel = advection_volume_rhs_kernel_2d[NP]
    ctx.enqueue_function[_kernel](q, elem_invJ, D_ref, num_elements, vx, vy, vol_out, grid_dim=ceildiv(total, 256), block_dim=256)


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


def advection_vol_lift_combine_rk_kernel_2d[
    NP: Int, NFP: Int
](
    q_in: UnsafePointer[Float32, MutAnyOrigin],
    elem_invJ: UnsafePointer[Float32, MutAnyOrigin],
    D_ref: UnsafePointer[Float32, MutAnyOrigin],
    fstar: UnsafePointer[Float32, MutAnyOrigin],
    elem_inv_2A: UnsafePointer[Float32, MutAnyOrigin],
    elem_faces: UnsafePointer[Int32, MutAnyOrigin],
    elem_face_side: UnsafePointer[Int32, MutAnyOrigin],
    elem_canon_to_ref: UnsafePointer[Int32, MutAnyOrigin],
    face_length: UnsafePointer[Float32, MutAnyOrigin],
    Lift_ref: UnsafePointer[Float32, MutAnyOrigin],
    q_a: UnsafePointer[Float32, MutAnyOrigin],
    q_b: UnsafePointer[Float32, MutAnyOrigin],
    num_elements: Int,
    vx: Float32,
    vy: Float32,
    a: Float32,
    b: Float32,
    cc: Float32,
    dt: Float32,
    q_out: UnsafePointer[Float32, MutAnyOrigin],
):
    var tid = Int(global_idx.x)
    var total = num_elements * NP
    if tid >= total:
        return
    var elem = tid // NP
    var i = tid % NP

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
            var r = Int(elem_canon_to_ref[(elem * 3 + lf) * NFP + m])
            var Lim = Lift_ref[lf * NP * NFP + i * NFP + r]
            face_c += sign * flen * Lim * fstar[fid * NFP + m]

    # ---- Combine + RK update.
    var idx = elem * NP + i
    var rhs_val = vol_c - inv_2A * face_c
    q_out[idx] = a * q_a[idx] + b * q_b[idx] + cc * dt * rhs_val


def launch_advection_vol_lift_2d[
    NP: Int, NFP: Int
](
    mut ctx: DeviceContext,
    q_in: UnsafePointer[Float32, MutAnyOrigin],
    elem_invJ: UnsafePointer[Float32, MutAnyOrigin],
    D_ref: UnsafePointer[Float32, MutAnyOrigin],
    fstar: UnsafePointer[Float32, MutAnyOrigin],
    elem_inv_2A: UnsafePointer[Float32, MutAnyOrigin],
    elem_faces: UnsafePointer[Int32, MutAnyOrigin],
    elem_face_side: UnsafePointer[Int32, MutAnyOrigin],
    elem_canon_to_ref: UnsafePointer[Int32, MutAnyOrigin],
    face_length: UnsafePointer[Float32, MutAnyOrigin],
    Lift_ref: UnsafePointer[Float32, MutAnyOrigin],
    q_a: UnsafePointer[Float32, MutAnyOrigin],
    q_b: UnsafePointer[Float32, MutAnyOrigin],
    num_elements: Int,
    vx: Float32,
    vy: Float32,
    a: Float32,
    b: Float32,
    cc: Float32,
    dt: Float32,
    q_out: UnsafePointer[Float32, MutAnyOrigin],
) raises:
    var total = num_elements * NP
    comptime _kernel = advection_vol_lift_combine_rk_kernel_2d[NP, NFP]
    ctx.enqueue_function[_kernel](
        q_in,
        elem_invJ,
        D_ref,
        fstar,
        elem_inv_2A,
        elem_faces,
        elem_face_side,
        elem_canon_to_ref,
        face_length,
        Lift_ref,
        q_a,
        q_b,
        num_elements,
        vx,
        vy,
        a,
        b,
        cc,
        dt,
        q_out,
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


def advection_face_flux_kernel_2d[
    NP: Int, NFP: Int
](
    q: UnsafePointer[Float32, MutAnyOrigin],
    face_elem: UnsafePointer[Int32, MutAnyOrigin],
    face_elem_node: UnsafePointer[Int32, MutAnyOrigin],
    face_normal: UnsafePointer[Float32, MutAnyOrigin],
    face_bc_type: UnsafePointer[Int32, MutAnyOrigin],
    num_faces: Int,
    vx: Float32,
    vy: Float32,
    inflow_q: Float32,
    fstar_out: UnsafePointer[Float32, MutAnyOrigin],
):
    var tid = Int(global_idx.x)
    var total = num_faces * NFP
    if tid >= total:
        return
    var fid = tid // NFP
    var m = tid % NFP

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


def launch_advection_face_flux_2d[
    NP: Int, NFP: Int
](
    mut ctx: DeviceContext,
    q: UnsafePointer[Float32, MutAnyOrigin],
    face_elem: UnsafePointer[Int32, MutAnyOrigin],
    face_elem_node: UnsafePointer[Int32, MutAnyOrigin],
    face_normal: UnsafePointer[Float32, MutAnyOrigin],
    face_bc_type: UnsafePointer[Int32, MutAnyOrigin],
    num_faces: Int,
    vx: Float32,
    vy: Float32,
    inflow_q: Float32,
    fstar_out: UnsafePointer[Float32, MutAnyOrigin],
) raises:
    var total = num_faces * NFP
    comptime _kernel = advection_face_flux_kernel_2d[NP, NFP]
    ctx.enqueue_function[_kernel](q, face_elem, face_elem_node, face_normal, face_bc_type, num_faces, vx, vy, inflow_q, fstar_out, grid_dim=ceildiv(total, 256), block_dim=256)


# ----------------------------------------------------------------------
# Full advection RK-stage orchestration (2 launches per stage).
# ----------------------------------------------------------------------
# Wraps `launch_advection_face_flux_2d` + `launch_advection_vol_lift_2d`
# (the fused per-element vol+lift+RK kernel).  Driver only has to
# provide the per-stage (a, b, cc) weights, the dt, and the
# q_in / q_a / q_b / q_out buffers.
#
# Signature mirrors what the eventual Solver2D struct will expose
# internally.  Calling it three times with the SSPRK3 weights
# reproduces the 2D GPU analog of the 3D Solver.step_ssprk3 loop.
# ----------------------------------------------------------------------


def advection_rk_stage_2d[
    P: Int
](
    mut ctx: DeviceContext,
    mesh: LocalMesh2DGpu[P],
    Lift_ref: UnsafePointer[Float32, MutAnyOrigin],
    D_ref: UnsafePointer[Float32, MutAnyOrigin],
    q_in: UnsafePointer[Float32, MutAnyOrigin],
    q_a: UnsafePointer[Float32, MutAnyOrigin],
    q_b: UnsafePointer[Float32, MutAnyOrigin],
    q_out: UnsafePointer[Float32, MutAnyOrigin],
    fstar_scratch: UnsafePointer[Float32, MutAnyOrigin],
    vx: Float32,
    vy: Float32,
    a: Float32,
    b: Float32,
    cc: Float32,
    dt: Float32,
    inflow_q: Float32 = Float32(0.0),
) raises:
    # Two launches per stage: face flux, then a fused vol+lift+RK kernel
    # that computes the volume RHS locally without round-tripping
    # through a global scratch buffer.  inflow_q defaults to 0 so
    # callers without BC_INFLOW are source-compatible; pass via
    # kwarg (`inflow_q=...`) at sites that do use it.
    comptime NP = num_tri_nodes_2d(P)
    comptime NFP = num_edge_nodes(P)
    launch_advection_face_flux_2d[NP, NFP](
        ctx,
        q_in,
        mesh.d_face_elem.unsafe_ptr(),
        mesh.d_face_elem_node.unsafe_ptr(),
        mesh.d_face_normal.unsafe_ptr(),
        mesh.d_face_bc_type.unsafe_ptr(),
        mesh.num_faces,
        vx,
        vy,
        inflow_q,
        fstar_scratch,
    )
    launch_advection_vol_lift_2d[NP, NFP](
        ctx,
        q_in,
        mesh.d_elem_invJ.unsafe_ptr(),
        D_ref,
        fstar_scratch,
        mesh.d_elem_inv_2A.unsafe_ptr(),
        mesh.d_elem_faces.unsafe_ptr(),
        mesh.d_elem_face_side.unsafe_ptr(),
        mesh.d_elem_canon_to_ref.unsafe_ptr(),
        mesh.d_face_length.unsafe_ptr(),
        Lift_ref,
        q_a,
        q_b,
        mesh.num_elements,
        vx,
        vy,
        a,
        b,
        cc,
        dt,
        q_out,
    )
