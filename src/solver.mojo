# ======================================================================
# Generic DG solver (unified single- and multi-rank)
# ======================================================================
#
# Parameterized by `PhysT`, a physics type implementing `Physics` (see
# below).  Built on top of `Mesh` + `HaloExchange` -- the same user-
# facing API in both single-rank (np=1) and multi-rank (np>1) modes.
#
# The RK-stage kernel iterates over owned elements via an
# `owned_elem_ids` list.  At np=1 that list is the identity (no ghost
# elements to skip), so the indirection collapses to one extra
# coalesced i32 load per thread.  At np>1 it points at the
# `num_owned_elements` scattered IDs in the patch-local mesh, and the
# stepper splits work into interior + halo passes bracketing a
# non-blocking halo exchange.
#
# Data layout
# -----------
#   * q / q1 / q2 : [num_local_elements * N_P * NC] Float32 on device
#     (at np=1, num_local = num_owned; at np>1, includes ghost slots)
#   * d_owned_elem_ids : indices into the flat local-mesh element space
#     (built by Mesh).
#
# Kernel dispatch
# ---------------
#   Per block:
#       elem_in_block = thread // N_P
#       i             = thread %  N_P
#       owned_idx     = block_idx * EPB + elem_in_block
#       e             = owned_elem_ids[owned_idx]   (local elem id)
# ======================================================================

from src.reference import num_tet_nodes, num_tri_nodes
from src.mesh import Mesh
from src.halo_exchange import HaloExchange
from src.nvtx import NvtxContext
from std.gpu import thread_idx, block_idx, barrier, global_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu.host.device_context import DevicePassable
from std.gpu.memory import AddressSpace
from std.math import ceildiv
from std.memory import memcpy, stack_allocation

comptime dtype = DType.float32

# One block handles EPB elements cooperatively.  Each
# element gets N_P threads (one per nodal DOF).  At N_P = 10 and 16
# elements the block is 160 threads = 5 warps.  The block size is a
# multiple of N_P so the (element, node-in-element) mapping is
# contiguous and there's no cross-block element split.
#
# Shared-memory usage scales linearly with NC: the volume-flux slab is
# `EPB * N_P * N_D * NC` floats and the face-flux slab is
# `EPB * N_F * N_FP * NC`.  At NC=17 (two-fluid plasma) a
# block of 16 elements overruns the default 48 KB cap, so we pick the
# block size per-NC at comptime from a small cascade.
comptime EPB_DEFAULT = 16

def elems_per_block_for(NC: Int, P: Int = 2) -> Int:
    # Keep the per-block shared slab below ~40 KB (leaves headroom for
    # register spills and locals).  Shared slab per element per block
    # = (NP * 3 + 4 * NFP) * NC * 4 bytes, where NP = num_tet_nodes(P)
    # and NFP = num_tri_nodes(P).  At P=2 that's (10*3 + 4*6) * NC * 4
    # = 216 * NC bytes per elem, matching the hand-computed P=2 value.
    var NP = num_tet_nodes(P)
    var NFP = num_tri_nodes(P)
    var bytes_per_elem = (NP * 3 + 4 * NFP) * NC * 4
    var budget = 40 * 1024
    var max_elems = budget // bytes_per_elem
    if max_elems >= 16: return 16
    if max_elems >= 8:  return 8
    if max_elems >= 4:  return 4
    if max_elems >= 2:  return 2
    return 1


# ----------------------------------------------------------------------
# Physics trait
# ----------------------------------------------------------------------
# A hyperbolic-system physics type that the generic Solver specializes
# on.  Concrete implementations (Advection, Euler) define the three
# entries below.  `internal_flux` writes the [d * NC + c] physical flux
# tensor; `numerical_flux` writes the NC-vector upwind flux at a face
# given a (nx, ny, nz) outward unit normal.  Both return a signal speed
# (used by CFL-controlling callers); `numerical_flux`'s return is the
# max |wave speed| at the interface.
# ----------------------------------------------------------------------

trait Physics(Copyable, Movable, ImplicitlyDestructible, DevicePassable):
    comptime NUM_COMPONENTS: Int

    def internal_flux(
        self,
        q:    UnsafePointer[Float32, MutAnyOrigin],
        flux: UnsafePointer[Float32, MutAnyOrigin],
    ) -> Float32:
        ...

    def numerical_flux(
        self,
        q_l:  UnsafePointer[Float32, MutAnyOrigin],
        q_r:  UnsafePointer[Float32, MutAnyOrigin],
        nx: Float32, ny: Float32, nz: Float32,
        flux: UnsafePointer[Float32, MutAnyOrigin],
    ) -> Float32:
        ...

    # Boundary-face flux.  Called on any face whose `face_bc_type` is
    # non-zero, i.e. a face sitting on a non-periodic global boundary.
    # `q_int` is the interior state at the face node; `bc_type` is one
    # of the `BC_*` constants from src.boundary that the physics module
    # chooses how to interpret.  The normal (nx, ny, nz) points OUTWARD
    # from the interior element into the (non-existent) ghost.  Returns
    # the max |wave speed| at the interface, same semantics as
    # `numerical_flux`'s return.
    def boundary_flux(
        self,
        q_int: UnsafePointer[Float32, MutAnyOrigin],
        bc_type: Int32,
        nx: Float32, ny: Float32, nz: Float32,
        flux: UnsafePointer[Float32, MutAnyOrigin],
    ) -> Float32:
        ...

    # Pointwise source term S(q, x).  Evaluated per nodal DOF and added
    # directly to the semi-discrete RHS at that node (strong-form nodal
    # collocation -- the DG mass matrix is diagonal under nodal P2 on
    # the reference element at the chosen quadrature, so M^-1 * (M * S)
    # reduces to S evaluated at the node).  Writes NC values to
    # `source_out`.  Physics types with no source term (pure
    # conservation law) can just fill with zeros; the compiler elides
    # the resulting zero-adds in the RK kernel.
    def source_term(
        self,
        q: UnsafePointer[Float32, MutAnyOrigin],
        x: Float32, y: Float32, z: Float32,
        source_out: UnsafePointer[Float32, MutAnyOrigin],
    ):
        ...

    # Post-stage positivity / bound limiter.  Called on `q_out` at each
    # nodal DOF after the RK update writes that node, before the next
    # stage reads it.  Physics types with no positivity requirement
    # (scalar advection, Maxwell) leave this as a no-op; Euler / MHD /
    # two-fluid clamp density and pressure to their configured floors.
    # This is a crude but robust shock-stabilization step: without it,
    # euler_sod NaNs out around t~0.15 due to Gibbs oscillations pushing
    # density below zero at the shock.
    def limit_state(
        self,
        q: UnsafePointer[Float32, MutAnyOrigin],
    ):
        ...


# ----------------------------------------------------------------------
# Patch-aware RK-stage kernel.
#
# Identical to `src.solver.rk_stage_kernel` except:
#   * iterates over a `owned_elem_ids` list instead of [0, num_elements);
#   * the block/grid sizing therefore comes from num_owned_elements.
#
# The cooperative shared-memory scheme (per-element internal_flux +
# per-element numerical_flux + Lift/volume-sum + RK update) is
# preserved verbatim, just using the redirected element id `e`.
# ----------------------------------------------------------------------

def rk_stage_kernel[
    NC: Int, EPB: Int, P: Int, PhysT: Physics,
](
    physics: PhysT,
    q_in:  UnsafePointer[Float32, MutAnyOrigin],
    q_a:   UnsafePointer[Float32, MutAnyOrigin],
    q_b:   UnsafePointer[Float32, MutAnyOrigin],
    q_out: UnsafePointer[Float32, MutAnyOrigin],
    elem_invJ:         UnsafePointer[Float32, MutAnyOrigin],
    elem_inv_6V:       UnsafePointer[Float32, MutAnyOrigin],
    elem_node_xyz:     UnsafePointer[Float32, MutAnyOrigin],
    elem_faces:        UnsafePointer[Int32,   MutAnyOrigin],
    elem_face_side:    UnsafePointer[Int32,   MutAnyOrigin],
    elem_canon_to_ref: UnsafePointer[Int32,   MutAnyOrigin],
    face_elem:         UnsafePointer[Int32,   MutAnyOrigin],
    face_elem_node:    UnsafePointer[Int32,   MutAnyOrigin],
    face_normal:       UnsafePointer[Float32, MutAnyOrigin],
    face_area:         UnsafePointer[Float32, MutAnyOrigin],
    face_bc_type:      UnsafePointer[Int32,   MutAnyOrigin],
    D_ref:             UnsafePointer[Float32, MutAnyOrigin],
    Lift_ref:          UnsafePointer[Float32, MutAnyOrigin],
    elem_base: Int,
    num_elems: Int,
    a: Float32, b: Float32, cc: Float32, dt: Float32,
):
    # Per-P sizes.  At P=2 these match the legacy module-level
    # N_P=10 / N_FP=6 exactly; for higher P they grow via the
    # Lagrange-tet counting formulas in src.reference.
    comptime NP = num_tet_nodes(P)
    comptime NFP = num_tri_nodes(P)
    comptime NF = 4  # tet faces, P-independent
    comptime ND = 3  # spatial dims
    # The Mesh constructor rearranges per-element arrays so that every
    # subset the solver dispatches over is CONTIGUOUS in element-id
    # space:
    #   np=1 / owned (full mesh):   [0, num_owned)
    #   np>1 / interior:            [0, num_interior)
    #   np>1 / halo:                [num_interior, num_interior+num_halo)
    # So we don't need an indirection buffer at all -- the caller
    # passes a base offset and a count, and `e = elem_base + owned_idx`
    # is the right local element id.  Skipping the load recovers ~4%
    # of step-loop time versus an indirect `owned_elem_ids[i]` read on
    # the M1 48^3 advection benchmark.
    var tid = Int(thread_idx.x)
    var bid = Int(block_idx.x)
    var elem_in_block = tid // NP
    var i = tid % NP
    var owned_idx = bid * EPB + elem_in_block
    var valid = owned_idx < num_elems
    var e: Int = elem_base + owned_idx

    # Shared memory for cooperative flux computation.  Layout mirrors
    # src.solver.rk_stage_kernel exactly -- see that file for the
    # invariant and block-size rationale.
    var shared_vol_flux = stack_allocation[
        EPB * NP * ND * NC,
        Scalar[DType.float32],
        address_space=AddressSpace.SHARED,
    ]()
    var shared_face_flux = stack_allocation[
        EPB * NF * NFP * NC,
        Scalar[DType.float32],
        address_space=AddressSpace.SHARED,
    ]()

    # ---- Phase 1: one internal_flux per (element, node) ------------
    if valid:
        var q_my_ptr = q_in + (e * NP + i) * NC
        var my_flux_dc = InlineArray[Float32, NC * 3](fill=0.0)
        var my_flux_dc_p = rebind[UnsafePointer[Float32, MutAnyOrigin]](
            my_flux_dc.unsafe_ptr()
        )
        _ = physics.internal_flux(
            rebind[UnsafePointer[Float32, MutAnyOrigin]](q_my_ptr),
            my_flux_dc_p,
        )
        var vol_base = (elem_in_block * NP + i) * ND * NC
        for k in range(ND * NC):
            shared_vol_flux[vol_base + k] = my_flux_dc[k]

    # ---- Phase 2: cooperative numerical_flux across element faces --
    if valid:
        comptime FN_TOTAL = NF * NFP
        for k in range(3):
            var fn_idx = i + k * NP
            if fn_idx < FN_TOTAL:
                var lf = fn_idx // NFP
                var m_canon = fn_idx % NFP
                var fid = Int(elem_faces[e * NF + lf])
                var nx = face_normal[fid * 3 + 0]
                var ny = face_normal[fid * 3 + 1]
                var nz = face_normal[fid * 3 + 2]
                var e_l = Int(face_elem[fid * 2 + 0])
                var e_r = Int(face_elem[fid * 2 + 1])
                var n_l = Int(face_elem_node[fid * 2 * NFP + 0 * NFP + m_canon])
                var n_r = Int(face_elem_node[fid * 2 * NFP + 1 * NFP + m_canon])

                var q_l_ptr = q_in + (e_l * NP + n_l) * NC
                var q_r_ptr = q_in + (e_r * NP + n_r) * NC
                var fstar = InlineArray[Float32, NC](fill=0.0)
                var fstar_p = rebind[UnsafePointer[Float32, MutAnyOrigin]](
                    fstar.unsafe_ptr()
                )
                # Interior faces take the two-sided numerical flux;
                # boundary faces (bc_type != 0) use the physics type's
                # boundary_flux on the interior state only, with the
                # outward normal pointing from interior to ghost side.
                # For BC faces the mesh builder guarantees: (a) this
                # element is on side 0, (b) face_normal points outward
                # from this element, (c) face_elem[*,1] is safe to
                # dereference as `q_l_ptr` (either identical to side 0
                # or a no-op ghost slot).
                var bc_type = face_bc_type[fid]
                if bc_type != Int32(0):
                    _ = physics.boundary_flux(
                        rebind[UnsafePointer[Float32, MutAnyOrigin]](q_l_ptr),
                        bc_type, nx, ny, nz, fstar_p,
                    )
                else:
                    _ = physics.numerical_flux(
                        rebind[UnsafePointer[Float32, MutAnyOrigin]](q_l_ptr),
                        rebind[UnsafePointer[Float32, MutAnyOrigin]](q_r_ptr),
                        nx, ny, nz, fstar_p,
                    )
                var face_base = (
                    (elem_in_block * NF + lf) * NFP + m_canon
                ) * NC
                for c in range(NC):
                    shared_face_flux[face_base + c] = fstar[c]

    barrier()
    if not valid:
        return

    # ---- Phase 3 + 4: per-component sum and RK update ---------------
    var invJ_base = e * 9
    var iJ00 = elem_invJ[invJ_base + 0]
    var iJ01 = elem_invJ[invJ_base + 1]
    var iJ02 = elem_invJ[invJ_base + 2]
    var iJ10 = elem_invJ[invJ_base + 3]
    var iJ11 = elem_invJ[invJ_base + 4]
    var iJ12 = elem_invJ[invJ_base + 5]
    var iJ20 = elem_invJ[invJ_base + 6]
    var iJ21 = elem_invJ[invJ_base + 7]
    var iJ22 = elem_invJ[invJ_base + 8]
    var inv_6V = elem_inv_6V[e]
    var out_base = (e * NP + i) * NC

    # Pointwise source term evaluated once per (element, node) thread.
    # Nodal collocation: the source contribution at node i is just
    # S(q_i, x_i), added directly to the RHS.  Physics types with no
    # sources zero-fill here and the compiler elides the add.
    var my_q_ptr = q_in + (e * NP + i) * NC
    var my_x = elem_node_xyz[(e * NP + i) * 3 + 0]
    var my_y = elem_node_xyz[(e * NP + i) * 3 + 1]
    var my_z = elem_node_xyz[(e * NP + i) * 3 + 2]
    var source = InlineArray[Float32, NC](fill=0.0)
    var source_p = rebind[UnsafePointer[Float32, MutAnyOrigin]](
        source.unsafe_ptr()
    )
    physics.source_term(
        rebind[UnsafePointer[Float32, MutAnyOrigin]](my_q_ptr),
        my_x, my_y, my_z, source_p,
    )

    for c in range(NC):
        var vol_c: Float32 = 0.0
        for j in range(NP):
            var shared_base = (elem_in_block * NP + j) * ND * NC
            var fx = shared_vol_flux[shared_base + 0 * NC + c]
            var fy = shared_vol_flux[shared_base + 1 * NC + c]
            var fz = shared_vol_flux[shared_base + 2 * NC + c]
            var fr0 = iJ00 * fx + iJ01 * fy + iJ02 * fz
            var fr1 = iJ10 * fx + iJ11 * fy + iJ12 * fz
            var fr2 = iJ20 * fx + iJ21 * fy + iJ22 * fz
            var d0 = D_ref[0 * NP * NP + i * NP + j]
            var d1 = D_ref[1 * NP * NP + i * NP + j]
            var d2 = D_ref[2 * NP * NP + i * NP + j]
            vol_c += fr0 * d0 + fr1 * d1 + fr2 * d2

        var face_c: Float32 = 0.0
        for lf in range(NF):
            var side = Int(elem_face_side[e * NF + lf])
            var sign = Float32(1.0) if side == 0 else Float32(-1.0)
            var fid = Int(elem_faces[e * NF + lf])
            var area = face_area[fid]
            for m_canon in range(NFP):
                var r = Int(
                    elem_canon_to_ref[(e * NF + lf) * NFP + m_canon]
                )
                var Lim = Lift_ref[lf * NP * NFP + i * NFP + r]
                var face_base = (
                    (elem_in_block * NF + lf) * NFP + m_canon
                ) * NC
                face_c += sign * area * Lim * shared_face_flux[
                    face_base + c
                ]

        var rhs_val = vol_c - inv_6V * face_c + source[c]
        q_out[out_base + c] = (
            a * q_a[out_base + c]
            + b * q_b[out_base + c]
            + cc * dt * rhs_val
        )

    # Post-stage limiter.  Runs once per (element, node) thread on
    # q_out, after the NC RK writes complete.  Physics types without
    # positivity requirements no-op; Euler / MHD / two-fluid clamp
    # density + pressure to their configured floors.
    physics.limit_state(
        rebind[UnsafePointer[Float32, MutAnyOrigin]](q_out + out_base)
    )


# ----------------------------------------------------------------------
# Barth-Jespersen (BJ) slope limiter
# ----------------------------------------------------------------------
# Three-pass conservation-preserving slope limiter, run once per RK stage.
#
#   Pass 1 (`compute_cell_averages_kernel`): one thread per (local element,
#   component) pair -- writes the NC-vector cell mean to `d_cell_avg` for
#   every local cell (owned + ghost; ghost means are read as face-neighbour
#   references in pass 2).
#
#   Pass 2 (`bj_limiter_compute_theta_kernel`): one thread per *owned*
#   element.  Reads own cell average + the 4 face-neighbour cell
#   averages; for every component independently finds the BJ scaling
#   factor alpha that keeps every nodal value within
#       [nbr_min_avg, nbr_max_avg]
#   where nbr_min/max_avg is the min/max cell average across the owning
#   cell + its 4 face-neighbours.  theta = min over nodes * components of
#   alpha (bounded to [0, 1]).  Writes one Float32 theta per owned
#   element to `d_bj_theta`.
#
#   Pass 3 (`bj_limiter_apply_kernel`): one thread per
#   (owned-element, node, component) triple (NP*NC = 100x more
#   parallelism at P=3, NC=5 vs the per-element pass).  Reads
#   theta[owned_idx]; if theta == 1 early-exits, otherwise applies
#       q_new[node, c] = own_avg[c] + theta * (q_old[node, c] - own_avg[c])
#   to its single (nn, c) slot.  Adjacent threads in a warp share the
#   same elem so theta-broadcast costs 1 transaction and q[] writes
#   are stride-1 within the element -> fully coalesced.  Conservation
#   of mass / momentum / energy is exact because own_avg is preserved.
#
# Why the split: with one thread per element doing NP*NC stores at
# stride 1 within the element but stride NP*NC across the warp, the
# apply phase was the worst-case uncoalesced pattern -- 20.8% of GPU
# time on shocked Sod 3D P=3, the same hot path the 2D limiter
# bottlenecked on (commit cdc3210 split that one in half too).
#
# Smooth regions: theta ~= 1, the limiter is a near no-op.  Shocks:
# theta << 1, the high-order modes get dampened proportionally while the
# cell average is untouched.  This is a textbook TVD-in-the-means limiter
# -- sharper than flatten-to-mean on shocks, and invisibly no-op on smooth
# flow.
# ----------------------------------------------------------------------

def compute_cell_averages_kernel[NP: Int, NC: Int](
    q:            UnsafePointer[Float32, MutAnyOrigin],
    node_weights: UnsafePointer[Float32, MutAnyOrigin],
    num_local:    Int,
    cell_avg_out: UnsafePointer[Float32, MutAnyOrigin],
):
    # One thread per (element, component) pair (NC-fold parallelism
    # vs the original 1-thread-per-element design).  Adjacent threads
    # in a warp access q[base_q + nn*NC + c] at consecutive c values
    # for the same (elem, nn) -- stride-1 inside a warp, full
    # coalescing on the q reads.  Same refactor applied to the 2D
    # `cell_mean_kernel_2d` in commit eff0cba (2.3x speedup at NP=10).
    var tid = Int(global_idx.x)
    var total = num_local * NC
    if tid >= total:
        return
    var elem = tid // NC
    var c = tid % NC
    var base_q = elem * NP * NC
    # Mass-matrix-weighted nodal quadrature for the exact P=P Lagrange
    # cell mean.  node_weights[] is normalised so sum == 1 over the
    # reference element; weights can be negative (e.g. -1/20 at P=2
    # tet vertex nodes), so the naive unweighted average is wrong.
    var s: Float32 = 0.0
    for nn in range(NP):
        s += node_weights[nn] * q[base_q + nn * NC + c]
    cell_avg_out[elem * NC + c] = s


def bj_limiter_compute_theta_kernel[NP: Int, NC: Int](
    q:              UnsafePointer[Float32, MutAnyOrigin],
    owned_elem_ids: UnsafePointer[Int32,   MutAnyOrigin],
    num_owned:      Int,
    cell_avg:       UnsafePointer[Float32, MutAnyOrigin],
    elem_faces:     UnsafePointer[Int32,   MutAnyOrigin],
    face_elem:      UnsafePointer[Int32,   MutAnyOrigin],
    venkat_eps:     Float32,
    theta_out:      UnsafePointer[Float32, MutAnyOrigin],
):
    var idx = Int(global_idx.x)
    if idx >= num_owned:
        return
    var elem = Int(owned_elem_ids[idx])

    # Load own cell average for every component.
    var base_avg = elem * NC
    var own_avg = InlineArray[Float32, NC](fill=Float32(0.0))
    for c in range(NC):
        own_avg[c] = cell_avg[base_avg + c]

    # Compute min/max of neighbour cell averages (including self so the
    # range is never empty, and so BC faces -- where "neighbour" resolves
    # to self -- don't introduce spurious bounds).
    var nbr_min = InlineArray[Float32, NC](fill=Float32(0.0))
    var nbr_max = InlineArray[Float32, NC](fill=Float32(0.0))
    for c in range(NC):
        nbr_min[c] = own_avg[c]
        nbr_max[c] = own_avg[c]

    for lf in range(4):
        var fid = Int(elem_faces[elem * 4 + lf])
        var e_l = Int(face_elem[fid * 2 + 0])
        var e_r = Int(face_elem[fid * 2 + 1])
        var n = e_r if e_l == elem else e_l
        var base_n = n * NC
        for c in range(NC):
            var a = cell_avg[base_n + c]
            if a < nbr_min[c]:
                nbr_min[c] = a
            if a > nbr_max[c]:
                nbr_max[c] = a

    # Compute theta: the tightest alpha scaling factor across every
    # node x component pair.  Venkatakrishnan's smoothing:
    #   alpha(d, D) = (D^2 + 2*D*d + eps^2) / (D^2 + 2*d^2 + D*d + eps^2)
    # where d = |delta| = |node - own_avg| and D = |allowed| (the signed
    # allowed deviation towards that side, which is nbr_max - own_avg if
    # delta > 0, else own_avg - nbr_min).  eps is the smoothness
    # tolerance: for d << D (smooth flow), alpha -> 1; for d >> D
    # (shock), alpha -> D/d (classical Barth-Jespersen).  eps=0
    # recovers pure BJ, which over-limits P2 DG even in smooth regions.
    var eps2 = venkat_eps * venkat_eps
    var base_q = elem * NP * NC
    var theta: Float32 = 1.0
    var tiny: Float32 = 1.0e-30
    for nn in range(NP):
        for c in range(NC):
            var node_val = q[base_q + nn * NC + c]
            var delta_s = node_val - own_avg[c]
            var d_abs = delta_s if delta_s >= Float32(0.0) else -delta_s
            if d_abs <= tiny:
                continue
            var D: Float32
            if delta_s > Float32(0.0):
                D = nbr_max[c] - own_avg[c]
            else:
                D = own_avg[c] - nbr_min[c]
            if D < Float32(0.0):
                D = Float32(0.0)   # shouldn't happen, but be safe
            var D2 = D * D
            var d2 = d_abs * d_abs
            var Dd = D * d_abs
            var numer = D2 + Float32(2.0) * Dd + eps2
            var denom = D2 + Float32(2.0) * d2 + Dd + eps2
            var alpha = numer / denom
            if alpha < theta:
                theta = alpha

    theta_out[idx] = theta


def bj_limiter_apply_kernel[NP: Int, NC: Int](
    q:              UnsafePointer[Float32, MutAnyOrigin],
    owned_elem_ids: UnsafePointer[Int32,   MutAnyOrigin],
    num_owned:      Int,
    cell_avg:       UnsafePointer[Float32, MutAnyOrigin],
    theta_in:       UnsafePointer[Float32, MutAnyOrigin],
):
    # One thread per (owned_idx, nn, c) -- coalesced apply pass.
    var tid = Int(global_idx.x)
    var total = num_owned * NP * NC
    if tid >= total:
        return

    var owned_idx = tid // (NP * NC)
    var theta = theta_in[owned_idx]
    if not (theta < Float32(1.0)):
        return  # smooth cell: no read/write needed

    var rem = tid - owned_idx * NP * NC
    var nn = rem // NC
    var c = rem % NC
    var elem = Int(owned_elem_ids[owned_idx])
    var base_q = elem * NP * NC
    var bm = cell_avg[elem * NC + c]
    var v = q[base_q + nn * NC + c]
    q[base_q + nn * NC + c] = bm + theta * (v - bm)


# ----------------------------------------------------------------------
# Solver
# ----------------------------------------------------------------------

struct Solver[PhysT: Physics, P: Int = 2](Movable):
    comptime NC = Self.PhysT.NUM_COMPONENTS
    # Nodal-DOF count for the chosen spatial order.  At P=2 this is 10
    # (the historical hand-coded value).  Every shape computation below
    # goes through `NP` / `NFP` rather than the module-level N_P / N_FP
    # aliases so that a higher-P Solver is a drop-in once the mesh
    # side catches up.
    comptime NP = num_tet_nodes(Self.P)
    comptime NFP = num_tri_nodes(Self.P)

    var ctx: DeviceContext
    var physics: Self.PhysT

    var mesh: Mesh[Self.P]
    var halo: HaloExchange

    var num_local_elements: Int
    var num_owned_elements: Int
    # All DOF counts below are in element-nodes (not times NC).
    var total_local_dof: Int
    var total_owned_dof: Int
    var total_q_len: Int      # total_local_dof * NC

    # RK-stage buffers sized for the full local mesh (owned + ghost).
    var d_q:  DeviceBuffer[dtype]
    var d_q1: DeviceBuffer[dtype]
    var d_q2: DeviceBuffer[dtype]

    # Reference DG operators (uploaded once).
    var d_D_ref:    DeviceBuffer[dtype]
    var d_Lift_ref: DeviceBuffer[dtype]
    # Nodal cell-mean weights (length NP, sum=1) for the BJ limiter.
    var d_node_weights: DeviceBuffer[dtype]

    # Cell-level Barth-Jespersen slope-limiter toggle.  False (default)
    # disables the limiter entirely -- it is a no-op add to the kernel
    # graph.  True runs `compute_cell_averages_kernel` +
    # `bj_limiter_compute_theta_kernel` + `bj_limiter_apply_kernel`
    # after every RK stage: BJ damps every element's nodal deviations
    # from its cell average by the tightest factor that keeps each
    # nodal value within [min, max] of the cell + face-neighbour
    # averages.  Conservation is exact; smooth flow is untouched
    # (theta ~= 1); shocks get dampened proportionally.
    var cell_limiter_enabled: Bool
    # Venkatakrishnan smoothness parameter for the BJ limiter.  Larger
    # values preserve more smooth variation (theta -> 1) at the cost of
    # slightly looser shock capture.  Default is 0.1 which works for
    # density in O(1) range.  Set to 0 to recover raw Barth-Jespersen
    # (over-limits P2 DG on smooth flow).
    var cell_limiter_venkat_eps: Float32
    # Scratch per-element mean buffer sized for the full local mesh
    # (owned + ghost).  Populated fresh on each limiter launch.
    var d_cell_avg: DeviceBuffer[dtype]
    # Per-owned-element theta scratch.  compute_theta writes one
    # Float32 per owned element; apply reads it.  Sized num_owned.
    var d_bj_theta: DeviceBuffer[dtype]

    def __init__(
        out self,
        var ctx: DeviceContext,
        var mesh: Mesh[Self.P],
        var halo: HaloExchange,
        var physics: Self.PhysT,
        D_ref: List[Float32],
        Lift_ref: List[Float32],
        node_weights: List[Float32],
    ) raises:
        self.ctx = ctx^
        self.mesh = mesh^
        self.halo = halo^
        self.physics = physics^
        self.num_local_elements = self.mesh.local.num_elements
        self.num_owned_elements = self.mesh.num_owned_elements
        self.total_local_dof = self.num_local_elements * Self.NP
        self.total_owned_dof = self.num_owned_elements * Self.NP
        self.total_q_len = self.total_local_dof * Self.NC
        self.cell_limiter_enabled = False
        self.cell_limiter_venkat_eps = Float32(0.1)
        # d_cell_avg is only read when the limiter is enabled, but we
        # allocate it up-front (cheap) so `enable_cell_limiter()` doesn't
        # need to be raised and `_launch_cell_limiter` can just branch on
        # the bool.  Sizing: NC floats per LOCAL element (owned + ghost)
        # so the limiter can read ghost cell averages as face-neighbour
        # references without out-of-bounds access.
        self.d_cell_avg = self.ctx.enqueue_create_buffer[dtype](
            self.num_local_elements * Self.NC,
        )
        # One theta per OWNED element (not local).  Size 1 if the
        # owner has no owned elements, since DeviceBuffer creation
        # is unhappy with size 0.
        var theta_size = self.num_owned_elements
        if theta_size == 0:
            theta_size = 1
        self.d_bj_theta = self.ctx.enqueue_create_buffer[dtype](theta_size)

        self.d_q  = self.ctx.enqueue_create_buffer[dtype](self.total_q_len)
        self.d_q1 = self.ctx.enqueue_create_buffer[dtype](self.total_q_len)
        self.d_q2 = self.ctx.enqueue_create_buffer[dtype](self.total_q_len)
        # Zero-initialise so ghost slots are sane until the first halo
        # exchange populates them.
        self.d_q.enqueue_fill(Float32(0.0))
        self.d_q1.enqueue_fill(Float32(0.0))
        self.d_q2.enqueue_fill(Float32(0.0))

        self.d_D_ref = _upload_f32(self.ctx, D_ref)
        self.d_Lift_ref = _upload_f32(self.ctx, Lift_ref)
        self.d_node_weights = _upload_f32(self.ctx, node_weights)
        self.ctx.synchronize()

    # --- Download helpers --------------------------------------------
    def download_owned_component_with_ids(
        mut self, c: Int,
        mut scalar_host: List[Float32],
        mut global_elem_ids_host: List[Int32],
        nx_global: Int, ny_global: Int, nz_global: Int,
        mut nvtx: NvtxContext,
    ) raises:
        """Download component `c` for owned elements AND compute each
        element's global-mesh ID so a downstream test harness can
        reassemble a whole-domain field from multiple ranks' dumps.

        `scalar_host`          length >= num_owned_elements * N_P
        `global_elem_ids_host`  length >= num_owned_elements
        """
        nvtx.push_range("download_owned_component_with_ids")
        var hbuf = self.ctx.enqueue_create_host_buffer[dtype](
            self.total_q_len
        )
        var h_ids = self.ctx.enqueue_create_host_buffer[DType.int32](
            self.num_owned_elements
        )
        # inv_perm[new_id] -> original build-time id.  After the
        # Mesh element reordering, owned_elem_ids entries are
        # new-numbering ids; we need the original id to decode cube
        # coordinates from the simple (cube, tet) formula.
        var h_invperm = self.ctx.enqueue_create_host_buffer[DType.int32](
            self.mesh.local.num_elements
        )
        self.ctx.enqueue_copy(hbuf,  self.d_q)
        self.ctx.enqueue_copy(h_ids, self.mesh.d_owned_elem_ids)
        self.ctx.enqueue_copy(h_invperm, self.mesh.d_inv_perm)
        self.ctx.synchronize()
        var q_p    = hbuf.unsafe_ptr()
        var ids_p  = h_ids.unsafe_ptr()
        var inv_p  = h_invperm.unsafe_ptr()
        var stride = Self.NC

        # Local grid dimensions used to decode cube coords from element
        # id.  `ghost_width` is 1 on the multi-patch path and 0 on the
        # single-patch fast path, so loc_nx / offsets compute correctly
        # in both.
        var gw = self.mesh.ghost_width
        var loc_nx = self.mesh.part.nx + 2 * gw
        var loc_ny = self.mesh.part.ny + 2 * gw
        var cx0 = self.mesh.part.cx0
        var cy0 = self.mesh.part.cy0
        var cz0 = self.mesh.part.cz0

        for i in range(self.num_owned_elements):
            var e_new = Int(ids_p[i])          # post-permutation id
            var e_old = Int(inv_p[e_new])       # build-time id (decodable)
            var cube = e_old // 6
            var tet = e_old - cube * 6
            var lcz = cube // (loc_nx * loc_ny)
            var rem = cube - lcz * loc_nx * loc_ny
            var lcy = rem // loc_nx
            var lcx = rem - lcy * loc_nx
            var gcx = cx0 + (lcx - gw)
            var gcy = cy0 + (lcy - gw)
            var gcz = cz0 + (lcz - gw)
            var gcube = gcx + nx_global * (gcy + ny_global * gcz)
            global_elem_ids_host[i] = Int32(gcube * 6 + tet)
            # q is stored under the NEW id (that's how the permuted
            # mesh addresses it).
            for nn in range(Self.NP):
                scalar_host[i * Self.NP + nn] = q_p[
                    (e_new * Self.NP + nn) * stride + c
                ]
        nvtx.pop_range()

    def download_owned_component(
        mut self, c: Int, mut scalar_host: List[Float32],
        mut nvtx: NvtxContext,
    ) raises:
        """Download component `c` of q for *owned* elements only into
        `scalar_host` (length >= num_owned_elements * N_P)."""
        nvtx.push_range("download_owned_component")
        # Full-buffer download, then host-side gather through
        # owned_elem_ids.  Ghost-slot values are simply skipped.
        var hbuf = self.ctx.enqueue_create_host_buffer[dtype](
            self.total_q_len
        )
        var h_ids = self.ctx.enqueue_create_host_buffer[DType.int32](
            self.num_owned_elements
        )
        self.ctx.enqueue_copy(hbuf,  self.d_q)
        self.ctx.enqueue_copy(h_ids, self.mesh.d_owned_elem_ids)
        self.ctx.synchronize()
        var q_p   = hbuf.unsafe_ptr()
        var ids_p = h_ids.unsafe_ptr()
        var stride = Self.NC
        for i in range(self.num_owned_elements):
            var e = Int(ids_p[i])
            for nn in range(Self.NP):
                scalar_host[i * Self.NP + nn] = q_p[
                    (e * Self.NP + nn) * stride + c
                ]
        nvtx.pop_range()

    # --- Internal: one RK stage kernel launch -----------------------
    # `elem_base` + `num_elems` select the contiguous block of element
    # ids the kernel iterates over:
    #    full owned (np=1):  base=0, num_elems=num_owned
    #    interior (np>1):    base=0, num_elems=num_interior
    #    halo (np>1):        base=num_interior, num_elems=num_halo
    # The Mesh constructor guarantees these three ranges are
    # contiguous in element-id space, so the kernel computes the local
    # element id as `e = elem_base + owned_idx` without an indirection
    # buffer.
    def enable_cell_limiter(
        mut self,
        enabled: Bool = True,
        venkat_eps: Float32 = Float32(0.1),
    ):
        """Turn the Barth-Jespersen slope limiter on or off.  When on,
        a two-pass post-RK-stage limiter runs after every SSPRK3 stage:
        `compute_cell_averages_kernel` writes per-element means and
        `bj_limiter_kernel` damps nodal deviations via the Venkat-
        smoothed Barth-Jespersen formula.  `venkat_eps` controls the
        smoothness tolerance: small values approach classical BJ
        (over-limits smooth P2 flow); larger values preserve smooth
        variation.  Default 0.1 is good for density in O(1) range;
        larger scales should bump eps proportionally."""
        self.cell_limiter_enabled = enabled
        self.cell_limiter_venkat_eps = venkat_eps

    def _launch_cell_limiter(
        mut self,
        q_ptr: UnsafePointer[Float32, MutAnyOrigin],
    ) raises:
        if not self.cell_limiter_enabled:
            return
        var num_local = self.num_local_elements
        var num_owned = self.num_owned_elements
        if num_owned == 0:
            return

        # Pass 1: cell averages over every local element.  One thread
        # per (element, component) pair for NC-fold parallelism +
        # coalesced q reads (see kernel comment).
        comptime _avg_kernel = compute_cell_averages_kernel[Self.NP, Self.NC]
        self.ctx.enqueue_function[_avg_kernel, _avg_kernel](
            q_ptr,
            self.d_node_weights.unsafe_ptr(),
            num_local,
            self.d_cell_avg.unsafe_ptr(),
            grid_dim=ceildiv(num_local * Self.NC, 256),
            block_dim=256,
        )

        # Pass 2: BJ theta computation over owned elements (one
        # thread per owned element; same parallelism as the old
        # single-kernel limiter).
        comptime _theta_kernel = bj_limiter_compute_theta_kernel[Self.NP, Self.NC]
        self.ctx.enqueue_function[_theta_kernel, _theta_kernel](
            q_ptr,
            self.mesh.d_owned_elem_ids.unsafe_ptr(),
            num_owned,
            self.d_cell_avg.unsafe_ptr(),
            self.mesh.local.d_elem_faces.unsafe_ptr(),
            self.mesh.local.d_face_elem.unsafe_ptr(),
            self.cell_limiter_venkat_eps,
            self.d_bj_theta.unsafe_ptr(),
            grid_dim=ceildiv(num_owned, 256),
            block_dim=256,
        )

        # Pass 3: theta apply over (owned_elem, node, component)
        # triples (NP*NC = 100x more parallelism at P=3, NC=5).
        comptime _apply_kernel = bj_limiter_apply_kernel[Self.NP, Self.NC]
        var apply_total = num_owned * Self.NP * Self.NC
        self.ctx.enqueue_function[_apply_kernel, _apply_kernel](
            q_ptr,
            self.mesh.d_owned_elem_ids.unsafe_ptr(),
            num_owned,
            self.d_cell_avg.unsafe_ptr(),
            self.d_bj_theta.unsafe_ptr(),
            grid_dim=ceildiv(apply_total, 256),
            block_dim=256,
        )

    def _launch_rk_stage(
        mut self,
        elem_base: Int,
        num_elems: Int,
        q_in_ptr:  UnsafePointer[Float32, MutAnyOrigin],
        q_a_ptr:   UnsafePointer[Float32, MutAnyOrigin],
        q_b_ptr:   UnsafePointer[Float32, MutAnyOrigin],
        q_out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        a: Float32, b: Float32, cc: Float32, dt: Float32,
    ) raises:
        if num_elems == 0:
            return
        comptime EPB = elems_per_block_for(Self.NC, Self.P)
        comptime kernel = rk_stage_kernel[Self.NC, EPB, Self.P, Self.PhysT]
        self.ctx.enqueue_function[kernel, kernel](
            self.physics,
            q_in_ptr, q_a_ptr, q_b_ptr, q_out_ptr,
            self.mesh.local.d_elem_invJ.unsafe_ptr(),
            self.mesh.local.d_elem_inv_6V.unsafe_ptr(),
            self.mesh.local.d_elem_node_xyz.unsafe_ptr(),
            self.mesh.local.d_elem_faces.unsafe_ptr(),
            self.mesh.local.d_elem_face_side.unsafe_ptr(),
            self.mesh.local.d_elem_canon_to_ref.unsafe_ptr(),
            self.mesh.local.d_face_elem.unsafe_ptr(),
            self.mesh.local.d_face_elem_node.unsafe_ptr(),
            self.mesh.local.d_face_normal.unsafe_ptr(),
            self.mesh.local.d_face_area.unsafe_ptr(),
            self.mesh.local.d_face_bc_type.unsafe_ptr(),
            self.d_D_ref.unsafe_ptr(),
            self.d_Lift_ref.unsafe_ptr(),
            elem_base, num_elems,
            a, b, cc, dt,
            grid_dim=ceildiv(num_elems, EPB),
            block_dim=EPB * Self.NP,
        )

    # Run one RK stage.  Two paths:
    #
    #  * Multi-rank (num_halo > 0): split interior / halo kernels
    #    bracketing a non-blocking halo exchange, so interior compute
    #    overlaps with MPI progress on the host.
    #  * Single-rank (num_halo == 0): no ghost elements exist, so there
    #    is nothing to exchange and nothing to split.  One kernel
    #    launch over all owned elements (`num_interior == num_owned`).
    def _step_stage_overlapped(
        mut self,
        q_in_ptr:  UnsafePointer[Float32, MutAnyOrigin],
        q_a_ptr:   UnsafePointer[Float32, MutAnyOrigin],
        q_b_ptr:   UnsafePointer[Float32, MutAnyOrigin],
        q_out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        a: Float32, b: Float32, cc: Float32, dt: Float32,
        mut nvtx: NvtxContext,
    ) raises:
        if self.mesh.num_halo_elements == 0:
            # Single-patch fast path: every owned element is interior.
            # One kernel, no MPI, no split.
            nvtx.push_range("rk_stage")
            self._launch_rk_stage(
                0, self.mesh.num_owned_elements,
                q_in_ptr, q_a_ptr, q_b_ptr, q_out_ptr,
                a, b, cc, dt,
            )
            nvtx.pop_range()
            nvtx.push_range("cell_limiter")
            self._launch_cell_limiter(q_out_ptr)
            nvtx.pop_range()
            return

        # Kick off pack + D->H + MPI_Isend/Irecv; returns while MPI
        # progresses on the host.
        nvtx.push_range("submit_pack")
        self.halo.submit_pack(self.ctx, q_in_ptr)
        nvtx.pop_range()

        # Interior compute: needs only owned q, no ghost data.  Runs
        # on the default stream concurrently with MPI.  Post-permutation
        # interior ids are [0, num_interior).
        nvtx.push_range("rk_stage_interior")
        self._launch_rk_stage(
            0, self.mesh.num_interior_elements,
            q_in_ptr, q_a_ptr, q_b_ptr, q_out_ptr,
            a, b, cc, dt,
        )
        nvtx.pop_range()

        # Waitall + H->D + unpack.  After this returns, ghost q is up
        # to date on device.
        nvtx.push_range("complete_exchange")
        self.halo.complete_exchange(self.ctx, q_in_ptr)
        nvtx.pop_range()

        # Halo compute: needs ghost q.  Post-permutation halo ids are
        # [num_interior, num_interior + num_halo).
        nvtx.push_range("rk_stage_halo")
        self._launch_rk_stage(
            self.mesh.num_interior_elements, self.mesh.num_halo_elements,
            q_in_ptr, q_a_ptr, q_b_ptr, q_out_ptr,
            a, b, cc, dt,
        )
        nvtx.pop_range()

        # Cell-level limiter: operates on the full owned set (interior +
        # halo now complete).  No-op when the threshold is 0.  Must run
        # before the next stage re-reads this q_out as q_in.
        nvtx.push_range("cell_limiter")
        self._launch_cell_limiter(q_out_ptr)
        nvtx.pop_range()

    # --- SSPRK3 time step with comm-compute overlap ------------------
    def step_ssprk3(
        mut self, dt: Float32, mut nvtx: NvtxContext,
    ) raises:
        nvtx.push_range("ssprk3_step")
        var p_q  = self.d_q.unsafe_ptr()
        var p_q1 = self.d_q1.unsafe_ptr()
        var p_q2 = self.d_q2.unsafe_ptr()

        # Stage 1: rhs(q) -> q1
        nvtx.push_range("rk_stage_1")
        self._step_stage_overlapped(
            p_q, p_q, p_q, p_q1,
            Float32(1.0), Float32(0.0), Float32(1.0), dt,
            nvtx,
        )
        nvtx.pop_range()

        # Stage 2: rhs(q1) -> q2
        nvtx.push_range("rk_stage_2")
        self._step_stage_overlapped(
            p_q1, p_q, p_q1, p_q2,
            Float32(0.75), Float32(0.25), Float32(0.25), dt,
            nvtx,
        )
        nvtx.pop_range()

        # Stage 3: rhs(q2) -> q
        nvtx.push_range("rk_stage_3")
        self._step_stage_overlapped(
            p_q2, p_q, p_q2, p_q,
            Float32(1.0 / 3.0), Float32(2.0 / 3.0),
            Float32(2.0 / 3.0), dt,
            nvtx,
        )
        nvtx.pop_range()

        nvtx.pop_range()


def _upload_f32(
    mut ctx: DeviceContext, src: List[Float32]
) raises -> DeviceBuffer[dtype]:
    var n = len(src)
    var hbuf = ctx.enqueue_create_host_buffer[dtype](n)
    memcpy(dest=hbuf.unsafe_ptr(), src=src.unsafe_ptr(), count=n)
    var dbuf = ctx.enqueue_create_buffer[dtype](n)
    ctx.enqueue_copy(dbuf, hbuf)
    return dbuf^
