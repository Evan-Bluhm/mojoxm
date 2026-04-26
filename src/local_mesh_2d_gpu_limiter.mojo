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
# `launch_cell_mean_2d` (parent module, mass-matrix-weighted mean)
# with the BJ scaling pass.
# ======================================================================

from src.local_mesh_2d_gpu import LocalMesh2DGpu, launch_cell_mean_2d
from src.reference_2d import num_tri_nodes_2d
from std.gpu import global_idx
from std.gpu.host import DeviceContext
from std.math import ceildiv


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
#   1. Caller first runs `cell_mean_kernel_2d[NP, NC]` (mass-matrix-
#      weighted, in `src/local_mesh_2d_gpu.mojo`) into a num_elements*NC
#      Float32 scratch buffer.  The unweighted `cell_avg_kernel_2d`
#      gives a wrong cell mean at P>=2 -- see the comment on line 135
#      below and `bench_euler_sod_limited_2d` for the regression bug it
#      caused.
#   2. `bj_limit_kernel_2d[NP, NC]` -- one thread per element.  Reads
#      own cell_mean[NC], peeks at 3 neighbours' component-0 means to
#      build nbr_min / nbr_max, computes Venkat theta on the own node
#      deviations (component 0), and scales all NP*NC values in-place.
# `bj_limit_full_2d` below orchestrates both passes; that's the public
# entry point.
#
# Boundary faces have face_elem[fid*2+1] == face_elem[fid*2+0], so
# neighbour-lookup self-matches and contributes nothing to the min/max
# range -- matching the CPU convention for bc-limited cells.
# ----------------------------------------------------------------------

def bj_limit_kernel_2d[NP: Int, NC: Int](
    q:            UnsafePointer[Float32, MutAnyOrigin],
    cell_mean:    UnsafePointer[Float32, MutAnyOrigin],
    elem_faces:   UnsafePointer[Int32,   MutAnyOrigin],
    face_elem:    UnsafePointer[Int32,   MutAnyOrigin],
    num_elements: Int,
    venkat_eps2:  Float32,
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

    if not (theta < Float32(1.0)):
        return   # smooth cell, leave it alone

    # Apply theta uniformly to every node, every component.  Base
    # per-component mean is the cell_mean buffer we already have.
    for nn in range(NP):
        for c in range(NC):
            var offset = (elem * NP + nn) * NC + c
            var base = cell_mean[elem * NC + c]
            q[offset] = base + theta * (q[offset] - base)


def launch_bj_limit_2d[NP: Int, NC: Int](
    mut ctx: DeviceContext,
    q:            UnsafePointer[Float32, MutAnyOrigin],
    cell_mean:    UnsafePointer[Float32, MutAnyOrigin],
    elem_faces:   UnsafePointer[Int32,   MutAnyOrigin],
    face_elem:    UnsafePointer[Int32,   MutAnyOrigin],
    num_elements: Int,
    venkat_eps2:  Float32,
) raises:
    comptime _kernel = bj_limit_kernel_2d[NP, NC]
    ctx.enqueue_function[_kernel, _kernel](
        q, cell_mean, elem_faces, face_elem, num_elements, venkat_eps2,
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
