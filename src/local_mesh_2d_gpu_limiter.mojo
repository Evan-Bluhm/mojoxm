# ======================================================================
# local_mesh_2d_gpu_limiter.mojo -- 2D Barth-Jespersen slope limiter
# ======================================================================
# Post-stage filter: scales every nodal deviation from the cell mean
# by the tightest theta keeping the scaled deviation within the
# (min, max) cell-average range over self + 3 face neighbours,
# sampled on component 0 (typically density).  The scaling is then
# applied uniformly to every NC component so coupled quantities
# (e.g. mass + momentum) stay consistent.  Venkatakrishnan smoothing
# (epsilon > 0) avoids over-limiting smooth regions; epsilon = 0
# recovers raw BJ which kills P+1 accuracy.
#
# Public entry point: `bj_limit_full_2d[P, NC](ctx, mesh, q,
# node_weights, cell_mean_scratch, venkat_eps)`.  Internally chains
# `launch_cell_mean_2d` (parent module, mass-matrix-weighted mean) +
# `launch_bj_limit_compute_theta_2d` + `launch_bj_limit_apply_2d`.
#
# `cell_mean_scratch` is sized `num_elements * (NC + 1)` Float32: the
# first `num_elements * NC` slots hold the per-component cell means,
# the trailing `num_elements` slots hold the per-element theta from
# the compute pass (read by the apply pass).
# ======================================================================

from src.local_mesh_2d_gpu import LocalMesh2DGpu, launch_cell_mean_2d
from src.reference_2d import num_tri_nodes_2d
from std.gpu import global_idx
from std.gpu.host import DeviceContext
from std.math import ceildiv


# ----------------------------------------------------------------------
# Barth-Jespersen slope limiter (Venkatakrishnan-smoothed), 2D GPU.
# ----------------------------------------------------------------------
# Three-pass structure (the cell_mean pass is the caller's
# `launch_cell_mean_2d` invocation; the limiter contributes the other
# two):
#   1. Caller first runs `cell_mean_kernel_2d[NP, NC]` (mass-matrix-
#      weighted, in `src/local_mesh_2d_gpu.mojo`) into a num_elements*NC
#      Float32 scratch buffer.  An unweighted nodal-arithmetic mean
#      gives a wrong cell mean at P>=2 -- see the docstring on
#      `bj_limit_full_2d` below and `bench_euler_sod_limited_2d` for
#      the regression bug that motivated the mass-weighted formulation.
#   2. `bj_limit_compute_theta_kernel_2d[NP, NC]` -- one thread per
#      element, reads own cell_mean[NC] + 3 neighbour means on
#      component 0, then loops NP nodes to compute Venkat theta on
#      the component-0 deviation.  Writes the per-element theta
#      into `theta_out[elem]`.
#   3. `bj_limit_apply_kernel_2d[NP, NC]` -- one thread per
#      `(elem, nn, c)` triple (NP*NC threads per element, total
#      `num_elements * NP * NC` threads).  Reads theta[elem]; if
#      theta >= 1 the cell is smooth so the thread early-exits
#      (no read/write).  Otherwise applies the uniform scaling
#      `q[i] = bm + theta * (q[i] - bm)` for its single (nn, c)
#      slot.  Adjacent threads in a warp share the same elem (and
#      thus the same theta) for nearly all of the warp, and they
#      hit consecutive q[] memory addresses -> coalesced access.
#
# This split exists because the original kernel did one thread per
# element and one thread did NP*NC consecutive q-stores at stride 1
# WITHIN the element but stride-NP*NC ACROSS the warp -- 32 threads
# in the warp wrote to 32 disjoint NP*NC-float blocks each, which was
# a worst-case uncoalesced pattern (it accounted for >50% of GPU
# time on the shocked-Sod P=3 limited gate).  The split moves the
# apply work into a coalesced kernel with NP*NC = 40x more threads
# (P=3, NC=4) and keeps the (cheap) theta-reduction pass at the old
# parallelism.
#
# Boundary faces have face_elem[fid*2+1] == face_elem[fid*2+0], so
# neighbour-lookup self-matches and contributes nothing to the min/max
# range -- matching the CPU convention for bc-limited cells.
# ----------------------------------------------------------------------

def bj_limit_compute_theta_kernel_2d[NP: Int, NC: Int](
    q:            UnsafePointer[Float32, MutAnyOrigin],
    cell_mean:    UnsafePointer[Float32, MutAnyOrigin],
    elem_faces:   UnsafePointer[Int32,   MutAnyOrigin],
    face_elem:    UnsafePointer[Int32,   MutAnyOrigin],
    num_elements: Int,
    venkat_eps2:  Float32,
    theta_out:    UnsafePointer[Float32, MutAnyOrigin],
):
    var elem = Int(global_idx.x)
    if elem >= num_elements:
        return

    var own_mean = cell_mean[elem * NC + 0]
    var nbr_min = own_mean
    var nbr_max = own_mean
    for lf in range(3):
        var fid = Int(elem_faces[elem * 3 + lf])
        var e_l = Int(face_elem[fid * 2 + 0])
        var e_r = Int(face_elem[fid * 2 + 1])
        var n = e_r if e_l == elem else e_l
        if n == elem:
            continue   # boundary face: don't constrain
        var a = cell_mean[n * NC + 0]
        if a < nbr_min: nbr_min = a
        if a > nbr_max: nbr_max = a

    # Venkat-smoothed theta on the density component.
    var theta: Float32 = 1.0
    var tiny: Float32 = 1.0e-30
    for nn in range(NP):
        var node_val = q[(elem * NP + nn) * NC + 0]
        var delta = node_val - own_mean
        var d_abs = delta if delta >= Float32(0.0) else -delta
        if d_abs <= tiny:
            continue
        var D: Float32
        if delta > Float32(0.0):
            D = nbr_max - own_mean
        else:
            D = own_mean - nbr_min
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

    theta_out[elem] = theta


def bj_limit_apply_kernel_2d[NP: Int, NC: Int](
    q:            UnsafePointer[Float32, MutAnyOrigin],
    cell_mean:    UnsafePointer[Float32, MutAnyOrigin],
    theta_in:     UnsafePointer[Float32, MutAnyOrigin],
    num_elements: Int,
):
    # One thread per (elem, nn, c) -- coalesced apply pass.
    var idx = Int(global_idx.x)
    var total = num_elements * NP * NC
    if idx >= total:
        return

    var elem = idx // (NP * NC)
    var theta = theta_in[elem]
    if not (theta < Float32(1.0)):
        return  # smooth cell: no read/write needed

    var c = idx % NC
    var bm = cell_mean[elem * NC + c]
    var v = q[idx]
    q[idx] = bm + theta * (v - bm)


def launch_bj_limit_compute_theta_2d[NP: Int, NC: Int](
    mut ctx: DeviceContext,
    q:            UnsafePointer[Float32, MutAnyOrigin],
    cell_mean:    UnsafePointer[Float32, MutAnyOrigin],
    elem_faces:   UnsafePointer[Int32,   MutAnyOrigin],
    face_elem:    UnsafePointer[Int32,   MutAnyOrigin],
    num_elements: Int,
    venkat_eps2:  Float32,
    theta_out:    UnsafePointer[Float32, MutAnyOrigin],
) raises:
    comptime _kernel = bj_limit_compute_theta_kernel_2d[NP, NC]
    ctx.enqueue_function[_kernel, _kernel](
        q, cell_mean, elem_faces, face_elem, num_elements,
        venkat_eps2, theta_out,
        grid_dim=ceildiv(num_elements, 256),
        block_dim=256,
    )


def launch_bj_limit_apply_2d[NP: Int, NC: Int](
    mut ctx: DeviceContext,
    q:            UnsafePointer[Float32, MutAnyOrigin],
    cell_mean:    UnsafePointer[Float32, MutAnyOrigin],
    theta_in:     UnsafePointer[Float32, MutAnyOrigin],
    num_elements: Int,
) raises:
    comptime _kernel = bj_limit_apply_kernel_2d[NP, NC]
    var total = num_elements * NP * NC
    ctx.enqueue_function[_kernel, _kernel](
        q, cell_mean, theta_in, num_elements,
        grid_dim=ceildiv(total, 256),
        block_dim=256,
    )


# Convenience three-pass orchestrator: run the mass-weighted cell mean
# then the BJ compute_theta + apply kernels on top of a caller-supplied
# scratch buffer.  Drivers call this between RK stages to enforce
# monotonicity on shocked problems.  `node_weights` is
# `ReferenceElement2DGpu.d_node_weights`; using an unweighted nodal
# arithmetic mean here would systematically drift shock speeds at
# P>=2 because Lagrange-P>=2 node weights aren't uniform (at P=2 the
# 3 vertex weights are 0, the 3 midpoint weights are 1/3).
#
# `cell_mean_scratch` must be `num_elements * (NC + 1)` floats: the
# first num_elements*NC slots hold the per-component cell means
# (written by the cell_mean pass, read by both downstream passes),
# the trailing num_elements slots hold the per-element theta written
# by compute_theta and read by apply.

def bj_limit_full_2d[P: Int, NC: Int](
    mut ctx: DeviceContext,
    mesh: LocalMesh2DGpu[P],
    q:             UnsafePointer[Float32, MutAnyOrigin],
    node_weights:  UnsafePointer[Float32, MutAnyOrigin],
    cell_mean_scratch: UnsafePointer[Float32, MutAnyOrigin],
    venkat_eps:    Float32 = Float32(0.1),
) raises:
    comptime NP = num_tri_nodes_2d(P)
    var theta_scratch = cell_mean_scratch + mesh.num_elements * NC
    launch_cell_mean_2d[NP, NC](
        ctx, q, node_weights, mesh.num_elements, cell_mean_scratch,
    )
    launch_bj_limit_compute_theta_2d[NP, NC](
        ctx, q, cell_mean_scratch,
        mesh.d_elem_faces.unsafe_ptr(),
        mesh.d_face_elem.unsafe_ptr(),
        mesh.num_elements,
        venkat_eps * venkat_eps,
        theta_scratch,
    )
    launch_bj_limit_apply_2d[NP, NC](
        ctx, q, cell_mean_scratch, theta_scratch,
        mesh.num_elements,
    )
