# ======================================================================
# Mesh: per-rank Kuhn-tet mesh with optional ghost ring
# ======================================================================
#
# Given a `Partition` describing this rank's owned sub-box of the
# global cube grid, builds a Kuhn-tet mesh that covers:
#
#   * the nx x ny x nz owned cubes, plus
#   * a 1-cube-thick ghost ring on every face (when np > 1; for DG the
#     halo is always exactly one element thick).
#
# At np=1 the ghost ring is omitted: `Mesh` then covers the entire
# global domain as a single periodic Kuhn-tet mesh and the patch-aware
# metadata (owned_elem_ids, interior_elem_ids, perm) collapses to the
# identity.  This keeps a single user-facing type for both single-rank
# and multi-rank runs -- solver code uniformly iterates owned_elem_ids
# and never branches on whether MPI is in use.
#
# Internal layout
# ---------------
# `Mesh` wraps a `LocalMesh` (src/local_mesh.mojo) that implements the
# raw Kuhn-tet generator.  For multi-patch the underlying grid is
# (nx+2, ny+2, nz+2) and every stored node coordinate is shifted by
# (cx0-1, cy0-1, cz0-1) * (dx, dy, dz) so the local mesh sits at the
# right physical location inside the global domain.  For single-patch
# it's just (nx, ny, nz) with no shift.
#
# Element ordering in the local mesh
# ----------------------------------
# `LocalMesh` numbers cells as `cell = lcx + Lx * (lcy + Ly * lcz)` with
# (Lx, Ly, Lz) = (nx+2*gw, ny+2*gw, nz+2*gw) where gw = ghost_width.
# Element IDs are `cell * 6 + tet`.
#
# The **owned** elements are the ones whose cell lies in the inner box
# `lcx in [gw, nx+gw)`, `lcy in [gw, ny+gw)`, `lcz in [gw, nz+gw)`.
# For single-patch (gw=0) every element is owned.  For multi-patch
# (gw=1) owned IDs are scattered through the flat element range, so
# we publish a `d_owned_elem_ids` list for the solver to dispatch
# against.
#
# Ghost cubes (`lcx = 0`, `lcx = nx+1`, etc., only present when gw=1)
# have invalid `elem_faces` entries (the underlying build uses periodic
# wrapping on the local grid, which wraps ghosts onto each other
# rather than onto the real neighbour patch).  That's harmless as
# long as the solver never iterates over ghost elements -- their `q`
# values are populated via MPI halo exchange.
# ======================================================================

from src.reference import N_F, num_tet_nodes
from src.local_mesh import LocalMesh, KUHN_TETS_PER_CELL
from src.partition import Partition
from src.boundary import BoundaryConditions, BC_INTERIOR
from std.gpu import global_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from std.math import ceildiv

comptime patch_f = DType.float32
comptime patch_i = DType.int32
comptime PATCH_BLOCK = 256


# Helper: upload an identity permutation [0, 1, 2, ..., n-1] as Int32.
# Used by the single-patch fast path where owned_elem_ids, the perm,
# and its inverse are all the identity.
def _upload_identity_i32(
    mut ctx: DeviceContext,
    n: Int,
) raises -> DeviceBuffer[patch_i]:
    var hbuf = ctx.enqueue_create_host_buffer[patch_i](n)
    var p = hbuf.unsafe_ptr()
    for i in range(n):
        p[i] = Int32(i)
    var dbuf = ctx.enqueue_create_buffer[patch_i](n)
    ctx.enqueue_copy(dbuf, hbuf)
    return dbuf^


# ----------------------------------------------------------------------
# Kernel: shift every stored node coordinate by (ox, oy, oz).
# ----------------------------------------------------------------------


def offset_nodes_kernel(
    elem_node_xyz: UnsafePointer[Float32, MutAnyOrigin],
    num_points: Int,
    ox: Float32,
    oy: Float32,
    oz: Float32,
):
    var idx = Int(global_idx.x)
    if idx >= num_points:
        return
    elem_node_xyz[idx * 3 + 0] += ox
    elem_node_xyz[idx * 3 + 1] += oy
    elem_node_xyz[idx * 3 + 2] += oz


# ----------------------------------------------------------------------
# Kernel: enumerate owned element IDs in the local mesh.
#
# Owned cubes span (lcx, lcy, lcz) in [1, nx+1) x [1, ny+1) x [1, nz+1).
# Each cube has `KUHN_TETS_PER_CELL` tets.  Output is a flat list of
# num_owned = nx * ny * nz * KUHN_TETS_PER_CELL element IDs.
# ----------------------------------------------------------------------


def build_owned_elem_ids_kernel(
    o_owned_ids: UnsafePointer[Int32, MutAnyOrigin],  # [num_owned]
    nx: Int,
    ny: Int,
    nz: Int,  # owned-cube counts per axis
    loc_nx: Int,
    loc_ny: Int,  # local-mesh nx, ny (= owned + 2)
    num_owned: Int,
):
    var idx = Int(global_idx.x)
    if idx >= num_owned:
        return
    var tet = idx % KUHN_TETS_PER_CELL
    var cube = idx // KUHN_TETS_PER_CELL
    # (owned cube index) -> (lcx-1, lcy-1, lcz-1) with z-fastest ordering
    # to keep owned IDs contiguous in the order the solver iterates them.
    var ocx = cube // (ny * nz)
    var rem = cube - ocx * ny * nz
    var ocy = rem // nz
    var ocz = rem - ocy * nz
    var lcx = ocx + 1
    var lcy = ocy + 1
    var lcz = ocz + 1
    var cell = lcx + loc_nx * (lcy + loc_ny * lcz)
    o_owned_ids[idx] = Int32(cell * KUHN_TETS_PER_CELL + tet)


# ----------------------------------------------------------------------
# Kernel: gather node coordinates for just the owned elements into a
# contiguous buffer, so the VTU writer can emit a per-rank file without
# ghost geometry.
# ----------------------------------------------------------------------


def gather_owned_nodes_kernel[
    NP: Int
](
    o_owned_node_xyz: UnsafePointer[Float32, MutAnyOrigin],  # [num_owned*NP*3]
    owned_ids: UnsafePointer[Int32, MutAnyOrigin],  # [num_owned]
    local_node_xyz: UnsafePointer[Float32, MutAnyOrigin],  # [num_local_elements*NP*3]
    num_owned: Int,
):
    var tid = Int(global_idx.x)
    var total = num_owned * NP
    if tid >= total:
        return
    var owned_idx = tid // NP
    var nn = tid % NP
    var src_elem = Int(owned_ids[owned_idx])
    for d in range(3):
        o_owned_node_xyz[(owned_idx * NP + nn) * 3 + d] = local_node_xyz[(src_elem * NP + nn) * 3 + d]


# ----------------------------------------------------------------------
# Kernel: classify every owned element as halo (at least one face
# neighbour is a ghost) or interior (all 4 face neighbours are owned).
#
# We decide "is neighbour owned?" by looking at the neighbour's cube
# coordinates in the local grid and checking that it lies inside the
# inner owned box [1, nx+1) x [1, ny+1) x [1, nz+1).  An element's
# halo flag is written into `o_halo_flag` as 0 (interior) or 1 (halo).
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# Kernels: apply an element permutation to the per-element mesh arrays
# and remap element-id values inside face_elem.
#
# The permutation is stored as two parallel Int32 arrays:
#   perm[old_id]     = new_id   (forward, used to remap face_elem values)
#   inv_perm[new_id] = old_id   (inverse, used to gather-into-new-layout)
#
# Every per-element array has a stride of (sizeof entry / sizeof Float32
# or Int32) per element.  `gather_stride_*_kernel` reads from the old
# layout using inv_perm and writes to the new layout at contiguous
# positions.
# ----------------------------------------------------------------------


def gather_stride_f32_kernel(
    new_arr: UnsafePointer[Float32, MutAnyOrigin],
    old_arr: UnsafePointer[Float32, MutAnyOrigin],
    inv_perm: UnsafePointer[Int32, MutAnyOrigin],  # [num_local] new -> old
    num_local: Int,
    stride: Int,
):
    var idx = Int(global_idx.x)
    var total = num_local * stride
    if idx >= total:
        return
    var new_id = idx // stride
    var offset = idx % stride
    var old_id = Int(inv_perm[new_id])
    new_arr[new_id * stride + offset] = old_arr[old_id * stride + offset]


def gather_stride_i32_kernel(
    new_arr: UnsafePointer[Int32, MutAnyOrigin],
    old_arr: UnsafePointer[Int32, MutAnyOrigin],
    inv_perm: UnsafePointer[Int32, MutAnyOrigin],
    num_local: Int,
    stride: Int,
):
    var idx = Int(global_idx.x)
    var total = num_local * stride
    if idx >= total:
        return
    var new_id = idx // stride
    var offset = idx % stride
    var old_id = Int(inv_perm[new_id])
    new_arr[new_id * stride + offset] = old_arr[old_id * stride + offset]


def remap_face_elem_values_kernel(
    face_elem: UnsafePointer[Int32, MutAnyOrigin],  # [num_faces * 2]
    perm: UnsafePointer[Int32, MutAnyOrigin],  # [num_local] old -> new
    num_entries: Int,  # = num_faces * 2
):
    var idx = Int(global_idx.x)
    if idx >= num_entries:
        return
    var old_v = Int(face_elem[idx])
    # Element ids in face_elem should always be valid indices into the
    # local element array, but guard defensively.
    if old_v >= 0:
        face_elem[idx] = perm[old_v]


def classify_owned_kernel(
    o_halo_flag: UnsafePointer[Int32, MutAnyOrigin],  # [num_owned]
    o_primary_ring: UnsafePointer[Int32, MutAnyOrigin],  # [num_owned], -1 if interior
    owned_elem_ids: UnsafePointer[Int32, MutAnyOrigin],  # [num_owned]
    elem_faces: UnsafePointer[Int32, MutAnyOrigin],  # [num_local*N_F]
    face_elem: UnsafePointer[Int32, MutAnyOrigin],  # [num_faces*2]
    num_owned: Int,
    loc_nx: Int,
    loc_ny: Int,
    nx: Int,
    ny: Int,
    nz: Int,
):
    var idx = Int(global_idx.x)
    if idx >= num_owned:
        return
    var elem = Int(owned_elem_ids[idx])
    var is_halo: Int32 = 0
    # primary_ring: the lowest-priority (direction index) ghost
    # neighbour.  Directions are 0=-x, 1=+x, 2=-y, 3=+y, 4=-z, 5=+z.
    # -1 sentinel means interior (no ghost neighbours).
    var primary: Int32 = -1
    # Decode this element's own cube coords so we know which direction
    # each neighbour lies in.
    var my_cube = elem // KUHN_TETS_PER_CELL
    var my_lcz = my_cube // (loc_nx * loc_ny)
    var my_rem = my_cube - my_lcz * loc_nx * loc_ny
    var my_lcy = my_rem // loc_nx
    var my_lcx = my_rem - my_lcy * loc_nx
    for lf in range(N_F):
        var face_id = Int(elem_faces[elem * N_F + lf])
        var s0 = Int(face_elem[face_id * 2 + 0])
        var s1 = Int(face_elem[face_id * 2 + 1])
        var neighbour = s1 if s0 == elem else s0
        var n_cube = neighbour // KUHN_TETS_PER_CELL
        var n_lcz = n_cube // (loc_nx * loc_ny)
        var rem = n_cube - n_lcz * loc_nx * loc_ny
        var n_lcy = rem // loc_nx
        var n_lcx = rem - n_lcy * loc_nx
        var n_owned = n_lcx >= 1 and n_lcx <= nx and n_lcy >= 1 and n_lcy <= ny and n_lcz >= 1 and n_lcz <= nz
        if not n_owned:
            is_halo = 1
            # Translate the cube offset to a direction index.
            var dx = n_lcx - my_lcx
            var dy = n_lcy - my_lcy
            var dz = n_lcz - my_lcz
            var this_dir: Int32 = -1
            if dx == -1 and dy == 0 and dz == 0:
                this_dir = 0  # -x
            elif dx == 1 and dy == 0 and dz == 0:
                this_dir = 1  # +x
            elif dx == 0 and dy == -1 and dz == 0:
                this_dir = 2  # -y
            elif dx == 0 and dy == 1 and dz == 0:
                this_dir = 3  # +y
            elif dx == 0 and dy == 0 and dz == -1:
                this_dir = 4  # -z
            elif dx == 0 and dy == 0 and dz == 1:
                this_dir = 5  # +z
            # First-match by direction index wins (i.e. lowest index).
            if this_dir != -1 and (primary == -1 or this_dir < primary):
                primary = this_dir
    o_halo_flag[idx] = is_halo
    o_primary_ring[idx] = primary


# ----------------------------------------------------------------------
# Mesh
# ----------------------------------------------------------------------


struct Mesh[P: Int = 2](Movable):
    # Per-P nodal-DOF count.  Kept as a comptime alias so any Mesh
    # method can multiply / divide by `Self.NP` without rebuilding
    # it from `num_tet_nodes(Self.P)` every time.
    comptime NP = num_tet_nodes(Self.P)

    var part: Partition

    # Underlying Kuhn-tet mesh (defined in src/local_mesh.mojo).  For
    # multi-patch (np>1) this is built for (nx+2) x (ny+2) x (nz+2)
    # cubes with the origin shifted to match this rank's patch.  For
    # single-patch (np=1) it covers just the owned (nx, ny, nz) cubes
    # with no ghost ring -- see ghost_width.
    var local: LocalMesh[Self.P]

    # Width of the ghost ring around the owned region, in cube units.
    # 1 for the multi-patch path (the DG halo is always one element
    # deep); 0 for the single-patch fast path (no ghost ring at all).
    # Downstream code that needs to decode local cube coordinates uses
    # this to compute loc_nx = part.nx + 2*ghost_width etc.
    var ghost_width: Int

    # Owned-element index list.  Length = num_owned_elements.
    # Values are indices into mesh.d_* arrays.
    var num_owned_elements: Int
    var d_owned_elem_ids: DeviceBuffer[patch_i]

    # Host-side (owned-only) node coordinates for the per-rank VTU
    # writer.  Layout: [owned_elem][node][xyz].  Length in floats =
    # num_owned_elements * N_P * 3.
    var owned_node_xyz_f32_ptr: UnsafePointer[Float32, MutExternalOrigin]
    var owned_node_xyz_f32_len: Int

    # Interior / halo partition of the owned element set.  Interior
    # elements have all 4 face neighbours in the owned set; halo
    # elements touch at least one ghost face neighbour.  The element
    # reordering below lays them out contiguously so the solver can
    # dispatch over [0, num_interior) and
    # [num_interior, num_interior + num_halo) without a scatter buffer
    # -- interior compute can then overlap with MPI halo exchange.
    var num_halo_elements: Int
    var num_interior_elements: Int

    # Per-direction (primary-ring-ordered) halo element counts.  For
    # each of the 6 face directions [-x, +x, -y, +y, -z, +z] this is
    # the number of halo tets whose PRIMARY (lowest-priority-wins)
    # ghost neighbour lies in that direction.
    var halo_primary_count: List[Int]
    # Ring counts per direction on the ghost side (equals the owned
    # halo ring counts for face-adjacent rings -- each ghost cube on
    # a face ring has exactly one primary sender).
    var ghost_ring_count: List[Int]

    # Permutation of local element IDs.  After Mesh construction
    # the mesh arrays have been rearranged so that:
    #
    #   [0, num_interior)                             owned interior
    #   [num_interior, num_interior + halo_primary_count[0]) halo -x
    #   next block: halo +x,  then -y, +y, -z, +z
    #   [num_owned, num_owned + ghost_ring_count[0])  ghost -x
    #   next 5 ghost blocks by direction
    #   [end_of_ghost_rings, num_local_elements)      unused edge/corner
    #                                                  ghost cubes (not
    #                                                  exchanged; never
    #                                                  read by the solver)
    #
    # `d_perm` maps the BUILD-time (cube_id * 6 + tet) element id to
    # the post-permutation id; `d_inv_perm` is the reverse.  Kept on
    # the device so HaloExchange can remap its pack-index lists.
    var d_perm: DeviceBuffer[patch_i]
    var d_inv_perm: DeviceBuffer[patch_i]

    def __init__(
        out self,
        mut ctx: DeviceContext,
        var part: Partition,
        Lx: Float64,
        Ly: Float64,
        Lz: Float64,
        bcs: BoundaryConditions,
    ) raises:
        # Single-patch fast path: at np=1 the entire global domain lives
        # on this rank, so we drop the ghost ring, the shift origin, the
        # halo classification, and the element reordering.  The
        # underlying Mesh is built exactly as a pre-refactor single-rank
        # Mesh would be, and patch-aware metadata (owned_elem_ids, perm,
        # per-ring counts) is trivial/identity so downstream solver code
        # that iterates over owned_elem_ids keeps working uniformly.
        #
        # Inlined here rather than factored into a helper because Mojo's
        # flow-sensitive init analysis can't verify across a method
        # call that every struct field gets initialised.
        if part.px * part.py * part.pz == 1:
            self.local = LocalMesh[Self.P](
                ctx,
                part.nx,
                part.ny,
                part.nz,
                Lx,
                Ly,
                Lz,
                bcs,
            )
            self.ghost_width = 0

            var n = self.local.num_elements
            self.num_owned_elements = n
            self.num_halo_elements = 0
            self.num_interior_elements = n

            # Identity owned / perm buffers.  d_owned_elem_ids is kept
            # for driver IC kernels that iterate owned DOFs by index
            # (at np=1 this is just [0, num_owned)).
            self.d_owned_elem_ids = _upload_identity_i32(ctx, n)
            self.d_perm = _upload_identity_i32(ctx, n)
            self.d_inv_perm = _upload_identity_i32(ctx, n)

            self.halo_primary_count = List[Int]()
            self.ghost_ring_count = List[Int]()
            for _ in range(6):
                self.halo_primary_count.append(0)
                self.ghost_ring_count.append(0)

            # Owned-only host node coordinates for the per-rank VTU
            # writer.  At np=1 "owned" is the full mesh; allocate a
            # fresh copy so the pointer's ownership model matches the
            # multi-patch path (the writer takes ownership of a
            # dedicated buffer in both cases).
            var owned_points = n * Self.NP
            self.owned_node_xyz_f32_len = owned_points * 3
            self.owned_node_xyz_f32_ptr = alloc[Float32](self.owned_node_xyz_f32_len)
            self.local.d_elem_node_xyz.enqueue_copy_to(self.owned_node_xyz_f32_ptr)

            self.part = part^
            ctx.synchronize()
            return

        var nx_loc = part.nx + 2
        var ny_loc = part.ny + 2
        var nz_loc = part.nz + 2
        var dx = Lx / Float64(part.global_nx)
        var dy = Ly / Float64(part.global_ny)
        var dz = Lz / Float64(part.global_nz)
        var Lx_loc = Float64(nx_loc) * dx
        var Ly_loc = Float64(ny_loc) * dy
        var Lz_loc = Float64(nz_loc) * dz

        # Filter the user's BC config down to only the sides where
        # THIS rank sits on the global boundary.  For interior ranks
        # every side stays BC_INTERIOR so LocalMesh's periodic-wrap
        # face builder handles them normally (with halo exchange
        # feeding the ghost ring).
        var rank_bcs = BoundaryConditions(
            bcs.bc_x_lo if part.rx == 0 else BC_INTERIOR,
            bcs.bc_x_hi if part.rx == part.px - 1 else BC_INTERIOR,
            bcs.bc_y_lo if part.ry == 0 else BC_INTERIOR,
            bcs.bc_y_hi if part.ry == part.py - 1 else BC_INTERIOR,
            bcs.bc_z_lo if part.rz == 0 else BC_INTERIOR,
            bcs.bc_z_hi if part.rz == part.pz - 1 else BC_INTERIOR,
        )

        # Build the underlying mesh.  Its coordinates run from
        # (0, 0, 0) to (Lx_loc, Ly_loc, Lz_loc); we offset them below.
        # bnd_off=1 tells the BC overlay kernels to target the first
        # OWNED cube layer (local lcx=1, lcy=1, lcz=1 etc.), skipping
        # the 1-cube ghost ring.
        self.local = LocalMesh[Self.P](
            ctx,
            nx_loc,
            ny_loc,
            nz_loc,
            Lx_loc,
            Ly_loc,
            Lz_loc,
            rank_bcs,
            1,
        )
        self.ghost_width = 1

        # Shift every stored node coordinate so the mesh sits in the
        # correct global-coordinate location.  The ghost ring is one
        # cube-width *outside* the patch's owned region, so the shift
        # origin is at (cx0 - 1, cy0 - 1, cz0 - 1) * d*.
        var ox = Float32(Float64(part.cx0 - 1) * dx)
        var oy = Float32(Float64(part.cy0 - 1) * dy)
        var oz = Float32(Float64(part.cz0 - 1) * dz)
        var num_points = self.local.num_elements * Self.NP
        ctx.enqueue_function[offset_nodes_kernel, offset_nodes_kernel](
            self.local.d_elem_node_xyz.unsafe_ptr(),
            num_points,
            ox,
            oy,
            oz,
            grid_dim=ceildiv(num_points, PATCH_BLOCK),
            block_dim=PATCH_BLOCK,
        )

        # Build owned-element ID list.
        self.num_owned_elements = part.nx * part.ny * part.nz * KUHN_TETS_PER_CELL
        self.d_owned_elem_ids = ctx.enqueue_create_buffer[patch_i](self.num_owned_elements)
        ctx.enqueue_function[
            build_owned_elem_ids_kernel,
            build_owned_elem_ids_kernel,
        ](
            self.d_owned_elem_ids.unsafe_ptr(),
            part.nx,
            part.ny,
            part.nz,
            nx_loc,
            ny_loc,
            self.num_owned_elements,
            grid_dim=ceildiv(self.num_owned_elements, PATCH_BLOCK),
            block_dim=PATCH_BLOCK,
        )

        # Gather owned-element node coordinates into a contiguous
        # device buffer, then download to host for the VTU writer.
        var owned_points = self.num_owned_elements * Self.NP
        var d_owned_nodes = ctx.enqueue_create_buffer[patch_f](owned_points * 3)
        comptime _gather_owned_initial = gather_owned_nodes_kernel[Self.NP]
        ctx.enqueue_function[_gather_owned_initial, _gather_owned_initial](
            d_owned_nodes.unsafe_ptr(),
            self.d_owned_elem_ids.unsafe_ptr(),
            self.local.d_elem_node_xyz.unsafe_ptr(),
            self.num_owned_elements,
            grid_dim=ceildiv(owned_points, PATCH_BLOCK),
            block_dim=PATCH_BLOCK,
        )
        self.owned_node_xyz_f32_len = owned_points * 3
        self.owned_node_xyz_f32_ptr = alloc[Float32](self.owned_node_xyz_f32_len)
        d_owned_nodes.enqueue_copy_to(self.owned_node_xyz_f32_ptr)

        # -------- Interior / halo classification ---------------------
        # Flag every owned element as 0 (interior) or 1 (halo), and
        # record its primary-ring direction (-1 for interior).
        var d_halo_flag = ctx.enqueue_create_buffer[patch_i](self.num_owned_elements)
        var d_primary_ring = ctx.enqueue_create_buffer[patch_i](self.num_owned_elements)
        ctx.enqueue_function[classify_owned_kernel, classify_owned_kernel](
            d_halo_flag.unsafe_ptr(),
            d_primary_ring.unsafe_ptr(),
            self.d_owned_elem_ids.unsafe_ptr(),
            self.local.d_elem_faces.unsafe_ptr(),
            self.local.d_face_elem.unsafe_ptr(),
            self.num_owned_elements,
            nx_loc,
            ny_loc,
            part.nx,
            part.ny,
            part.nz,
            grid_dim=ceildiv(self.num_owned_elements, PATCH_BLOCK),
            block_dim=PATCH_BLOCK,
        )

        # Download everything we need for host-side partitioning.
        var h_flag = ctx.enqueue_create_host_buffer[patch_i](self.num_owned_elements)
        var h_primary = ctx.enqueue_create_host_buffer[patch_i](self.num_owned_elements)
        var h_owned = ctx.enqueue_create_host_buffer[patch_i](self.num_owned_elements)
        ctx.enqueue_copy(h_flag, d_halo_flag)
        ctx.enqueue_copy(h_primary, d_primary_ring)
        ctx.enqueue_copy(h_owned, self.d_owned_elem_ids)
        ctx.synchronize()

        var pflag = h_flag.unsafe_ptr()
        var pprimary = h_primary.unsafe_ptr()
        var powned = h_owned.unsafe_ptr()

        # Count per-bucket sizes: [interior, halo_-x, halo_+x, halo_-y,
        # halo_+y, halo_-z, halo_+z].
        var halo_per_dir = InlineArray[Int, 6](fill=0)
        var halo_count = 0
        for i in range(self.num_owned_elements):
            if pflag[i] != 0:
                halo_count += 1
                var d = Int(pprimary[i])
                if d >= 0 and d < 6:
                    halo_per_dir[d] += 1
        var interior_count = self.num_owned_elements - halo_count
        self.num_halo_elements = halo_count
        self.num_interior_elements = interior_count

        # Publish primary counts.  The list is canonical [-x,+x,-y,+y,-z,+z].
        self.halo_primary_count = List[Int]()
        for d in range(6):
            self.halo_primary_count.append(halo_per_dir[d])
        # Ghost ring counts mirror the owned halo RINGS (every cube on
        # the -x ring contributes 6 tets, regardless of primary-ring
        # picks; the face-ring size is nx*ny*6 at the -z ring etc.).
        self.ghost_ring_count = List[Int]()
        self.ghost_ring_count.append(part.ny * part.nz * KUHN_TETS_PER_CELL)  # -x
        self.ghost_ring_count.append(part.ny * part.nz * KUHN_TETS_PER_CELL)  # +x
        self.ghost_ring_count.append(part.nx * part.nz * KUHN_TETS_PER_CELL)  # -y
        self.ghost_ring_count.append(part.nx * part.nz * KUHN_TETS_PER_CELL)  # +y
        self.ghost_ring_count.append(part.nx * part.ny * KUHN_TETS_PER_CELL)  # -z
        self.ghost_ring_count.append(part.nx * part.ny * KUHN_TETS_PER_CELL)  # +z

        # Host-side interior / halo ID lists.  Not uploaded to device
        # (the solver dispatches over contiguous [base, base+count)
        # ranges after the permutation below, without needing a
        # scatter buffer).  We build them here only to drive the
        # permutation itself -- they tell us which old element ids
        # should go into the "first interior_count" slots and which
        # into the "halo" block.
        var h_halo = ctx.enqueue_create_host_buffer[patch_i](halo_count)
        var h_int = ctx.enqueue_create_host_buffer[patch_i](interior_count)
        var p_halo = h_halo.unsafe_ptr()
        var p_int = h_int.unsafe_ptr()

        # Compute cumulative primary offsets.
        var halo_offset = InlineArray[Int, 6](fill=0)
        for d in range(1, 6):
            halo_offset[d] = halo_offset[d - 1] + halo_per_dir[d - 1]
        # Mutable working cursors.
        var halo_cursor = InlineArray[Int, 6](fill=0)
        for d in range(6):
            halo_cursor[d] = halo_offset[d]

        var ic = 0
        for i in range(self.num_owned_elements):
            if pflag[i] == 0:
                p_int[ic] = powned[i]
                ic += 1
            else:
                var d = Int(pprimary[i])
                if d < 0 or d >= 6:
                    # Should not happen if classify kernel is correct;
                    # fall back to bucket 0.
                    d = 0
                p_halo[halo_cursor[d]] = powned[i]
                halo_cursor[d] += 1

        # Move the partition into place *before* the element reordering
        # step, because _permute_mesh_arrays passes self around and
        # Mojo requires every struct field initialised before that.
        self.part = part^

        # -------- Element reordering (phase 5) ----------------------
        # Compute a permutation of ALL local elements (owned + ghost)
        # so the mesh arrays layout becomes:
        #   [interior] [halo_-x] [halo_+x] [halo_-y] [halo_+y] [halo_-z] [halo_+z]
        #   [ghost_-x] [ghost_+x] [ghost_-y] [ghost_+y] [ghost_-z] [ghost_+z]
        #   [unused ghost: edge/corner cubes that no rank exchanges]
        # This is the structural foundation for pack-free halo
        # exchange (future work); the current HaloExchange still
        # uses pack kernels.  Correctness is preserved.
        var num_local = self.local.num_elements
        var h_perm = ctx.enqueue_create_host_buffer[patch_i](num_local)
        var h_inv = ctx.enqueue_create_host_buffer[patch_i](num_local)

        # Initialise perm to -1 (sentinel), then fill in bucket-by-bucket.
        var pp = h_perm.unsafe_ptr()
        var pi = h_inv.unsafe_ptr()
        for i in range(num_local):
            pp[i] = Int32(-1)
            pi[i] = Int32(-1)

        # Compute per-direction owned-halo offsets.
        var halo_offset_abs = InlineArray[Int, 6](fill=0)
        halo_offset_abs[0] = interior_count
        for d in range(1, 6):
            halo_offset_abs[d] = halo_offset_abs[d - 1] + halo_per_dir[d - 1]
        # Cumulative ghost-ring offsets after owned region.
        var ghost_offset_abs = InlineArray[Int, 7](fill=0)
        ghost_offset_abs[0] = self.num_owned_elements
        for d in range(1, 7):
            ghost_offset_abs[d] = ghost_offset_abs[d - 1] + self.ghost_ring_count[d - 1]
        var unused_start = ghost_offset_abs[6]

        # --- Owned bucket assignment (from the already-ordered
        # d_halo_elem_ids / d_interior_elem_ids above). ---
        for i in range(interior_count):
            var old_id = Int(p_int[i])
            pp[old_id] = Int32(i)
        for i in range(halo_count):
            var old_id = Int(p_halo[i])
            pp[old_id] = Int32(interior_count + i)

        # --- Ghost bucket assignment (walk the 6 face rings on the
        # local grid). ---
        # Face-ring tuples describe the lcx/lcy/lcz constants and
        # iterate-ranges in the canonical order used by HaloExchange
        # (y-inner, z-outer for x-axis rings; x-inner, z-outer for
        # y-axis; x-inner, y-outer for z-axis -- matching what
        # _pack_list_for_dir in halo_exchange.mojo uses).
        var ghost_cursor = List[Int]()
        for d in range(6):
            ghost_cursor.append(ghost_offset_abs[d])

        def _assign_ghost_ring(
            mut perm_p: UnsafePointer[Int32, MutAnyOrigin],
            mut cursor: List[Int],
            d: Int,
            fx: Int,
            fy: Int,
            fz: Int,
            iter_a_lo: Int,
            iter_a_hi: Int,
            iter_b_lo: Int,
            iter_b_hi: Int,
            axis: Int,
            loc_nx_cap: Int,
            loc_ny_cap: Int,
        ) capturing:
            # `axis` == 0 means x-axis ring: fx set, iterate (lcy, lcz).
            # `axis` == 1 means y-axis ring: fy set, iterate (lcx, lcz).
            # `axis` == 2 means z-axis ring: fz set, iterate (lcx, lcy).
            for a in range(iter_a_lo, iter_a_hi):
                for b in range(iter_b_lo, iter_b_hi):
                    var lcx: Int
                    var lcy: Int
                    var lcz: Int
                    if axis == 0:
                        lcx = fx
                        lcy = a
                        lcz = b
                    elif axis == 1:
                        lcx = a
                        lcy = fy
                        lcz = b
                    else:
                        lcx = a
                        lcy = b
                        lcz = fz
                    var cell = lcx + loc_nx_cap * (lcy + loc_ny_cap * lcz)
                    for t in range(KUHN_TETS_PER_CELL):
                        var old_id = cell * KUHN_TETS_PER_CELL + t
                        perm_p[old_id] = Int32(cursor[d])
                        cursor[d] = cursor[d] + 1

        var nx_p = self.part.nx
        var ny_p = self.part.ny
        var nz_p = self.part.nz
        # -x ghost ring (lcx=0), iterate (lcy, lcz).
        _assign_ghost_ring(
            pp,
            ghost_cursor,
            0,
            0,
            -1,
            -1,
            1,
            ny_p + 1,
            1,
            nz_p + 1,
            0,
            nx_loc,
            ny_loc,
        )
        # +x ghost ring (lcx=nx+1).
        _assign_ghost_ring(
            pp,
            ghost_cursor,
            1,
            nx_p + 1,
            -1,
            -1,
            1,
            ny_p + 1,
            1,
            nz_p + 1,
            0,
            nx_loc,
            ny_loc,
        )
        # -y ghost ring (lcy=0), iterate (lcx, lcz).
        _assign_ghost_ring(
            pp,
            ghost_cursor,
            2,
            -1,
            0,
            -1,
            1,
            nx_p + 1,
            1,
            nz_p + 1,
            1,
            nx_loc,
            ny_loc,
        )
        # +y ghost ring (lcy=ny+1).
        _assign_ghost_ring(
            pp,
            ghost_cursor,
            3,
            -1,
            ny_p + 1,
            -1,
            1,
            nx_p + 1,
            1,
            nz_p + 1,
            1,
            nx_loc,
            ny_loc,
        )
        # -z ghost ring (lcz=0), iterate (lcx, lcy).
        _assign_ghost_ring(
            pp,
            ghost_cursor,
            4,
            -1,
            -1,
            0,
            1,
            nx_p + 1,
            1,
            ny_p + 1,
            2,
            nx_loc,
            ny_loc,
        )
        # +z ghost ring (lcz=nz+1).
        _assign_ghost_ring(
            pp,
            ghost_cursor,
            5,
            -1,
            -1,
            nz_p + 1,
            1,
            nx_p + 1,
            1,
            ny_p + 1,
            2,
            nx_loc,
            ny_loc,
        )

        # --- Unused buckets: everything still at -1 (edge/corner
        # ghost cubes and any interior gaps). ---
        var leftover_cursor = unused_start
        for old_id in range(num_local):
            if pp[old_id] < 0:
                pp[old_id] = Int32(leftover_cursor)
                leftover_cursor += 1

        # Build inverse permutation.
        for old_id in range(num_local):
            var nid = Int(pp[old_id])
            pi[nid] = Int32(old_id)

        # Upload perm / inv_perm.
        self.d_perm = ctx.enqueue_create_buffer[patch_i](num_local)
        self.d_inv_perm = ctx.enqueue_create_buffer[patch_i](num_local)
        ctx.enqueue_copy(self.d_perm, h_perm)
        ctx.enqueue_copy(self.d_inv_perm, h_inv)
        ctx.synchronize()

        # --- Apply permutation to per-element mesh arrays by
        # allocating new buffers and gathering through inv_perm. ---
        self._permute_mesh_arrays(ctx, num_local)

        # After the permutation, owned_elem_ids is simply [0, num_owned)
        # (the permutation brought every owned element into that
        # contiguous range, with interior in [0, num_interior) and halo
        # in [num_interior, num_owned)).  Uploaded for IC kernels that
        # still iterate owned DOFs by index.
        self.d_owned_elem_ids = _upload_identity_i32(
            ctx,
            self.num_owned_elements,
        )
        ctx.synchronize()

        # --- Regenerate the host-side owned node coordinates now
        # that element ids have been permuted. ---
        var d_owned_nodes_new = ctx.enqueue_create_buffer[patch_f](owned_points * 3)
        comptime _gather_owned_post = gather_owned_nodes_kernel[Self.NP]
        ctx.enqueue_function[_gather_owned_post, _gather_owned_post](
            d_owned_nodes_new.unsafe_ptr(),
            self.d_owned_elem_ids.unsafe_ptr(),
            self.local.d_elem_node_xyz.unsafe_ptr(),
            self.num_owned_elements,
            grid_dim=ceildiv(owned_points, PATCH_BLOCK),
            block_dim=PATCH_BLOCK,
        )
        d_owned_nodes_new.enqueue_copy_to(self.owned_node_xyz_f32_ptr)
        ctx.synchronize()

    def _permute_mesh_arrays(
        mut self,
        mut ctx: DeviceContext,
        num_local: Int,
    ) raises:
        """Apply `d_inv_perm` to every per-element mesh array and
        remap element-id values in `face_elem`.  Each per-element
        array is gathered into a fresh buffer, which then replaces
        the original via move-assignment."""
        var inv = self.d_inv_perm.unsafe_ptr()
        var perm = self.d_perm.unsafe_ptr()

        # Per-element Float32 arrays (elem_node_xyz, elem_invJ,
        # elem_inv_6V).
        @parameter
        def _gather_f32(
            mut old: DeviceBuffer[patch_f],
            stride: Int,
        ) raises -> DeviceBuffer[patch_f]:
            var new = ctx.enqueue_create_buffer[patch_f](num_local * stride)
            var total = num_local * stride
            ctx.enqueue_function[
                gather_stride_f32_kernel,
                gather_stride_f32_kernel,
            ](
                new.unsafe_ptr(),
                old.unsafe_ptr(),
                inv,
                num_local,
                stride,
                grid_dim=ceildiv(total, PATCH_BLOCK),
                block_dim=PATCH_BLOCK,
            )
            return new^

        @parameter
        def _gather_i32(
            mut old: DeviceBuffer[patch_i],
            stride: Int,
        ) raises -> DeviceBuffer[patch_i]:
            var new = ctx.enqueue_create_buffer[patch_i](num_local * stride)
            var total = num_local * stride
            ctx.enqueue_function[
                gather_stride_i32_kernel,
                gather_stride_i32_kernel,
            ](
                new.unsafe_ptr(),
                old.unsafe_ptr(),
                inv,
                num_local,
                stride,
                grid_dim=ceildiv(total, PATCH_BLOCK),
                block_dim=PATCH_BLOCK,
            )
            return new^

        self.local.d_elem_node_xyz = _gather_f32(self.local.d_elem_node_xyz, Self.NP * 3)
        self.local.d_elem_invJ = _gather_f32(self.local.d_elem_invJ, 9)
        self.local.d_elem_inv_6V = _gather_f32(self.local.d_elem_inv_6V, 1)
        self.local.d_elem_faces = _gather_i32(self.local.d_elem_faces, N_F)
        self.local.d_elem_face_side = _gather_i32(self.local.d_elem_face_side, N_F)
        self.local.d_elem_canon_to_ref = _gather_i32(
            self.local.d_elem_canon_to_ref,
            N_F * 6,
        )

        # Remap element-id VALUES inside face_elem (face entries
        # reference elements by id).
        var n_entries = self.local.num_faces * 2
        ctx.enqueue_function[
            remap_face_elem_values_kernel,
            remap_face_elem_values_kernel,
        ](
            self.local.d_face_elem.unsafe_ptr(),
            perm,
            n_entries,
            grid_dim=ceildiv(n_entries, PATCH_BLOCK),
            block_dim=PATCH_BLOCK,
        )
        ctx.synchronize()
