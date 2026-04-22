# ======================================================================
# Per-rank (patch-local) Kuhn-tet mesh
# ======================================================================
#
# Given a `Partition` describing this rank's owned sub-box of the
# global cube grid, builds a local Kuhn-tet mesh that covers:
#
#   * the nx x ny x nz owned cubes, AND
#   * a 1-cube-thick ghost ring on every face (for DG, the halo is
#     always exactly one element thick).
#
# Internally this reuses `Mesh` with a local grid of size
# (nx+2, ny+2, nz+2) and physical size (nx+2)*dx etc., then offsets
# every node coordinate by (cx0-1, cy0-1, cz0-1) * (dx, dy, dz) so the
# local mesh sits in the correct physical location within the global
# domain.
#
# Element ordering in the local mesh
# ----------------------------------
# `Mesh` numbers cells as `cell = lcx + Lx * (lcy + Ly * lcz)` with
# Lx = nx+2, Ly = ny+2, Lz = nz+2.  Element IDs are `cell * 6 + tet`.
#
# This rank's **owned** elements are the ones whose cell lies in the
# inner box `lcx in [1, nx+1)`, `lcy in [1, ny+1)`, `lcz in [1, nz+1)`.
# Their IDs are scattered through the flat element range, not
# contiguous, so we publish a `d_owned_elem_ids` list for the solver
# to dispatch against.
#
# Ghost cubes (`lcx = 0`, `lcx = nx+1`, etc.) have invalid `elem_faces`
# entries (the underlying build uses periodic wrapping on the local
# grid, which wraps ghosts onto each other rather than to the real
# neighbour patch).  That's harmless as long as the solver never
# iterates over ghost elements -- their `q` values are populated via
# MPI halo exchange, and only owned elements read them via the
# `face_elem` refs of faces owned by owned cubes.
# ======================================================================

from src.reference import N_P
from src.mesh import Mesh, KUHN_TETS_PER_CELL
from src.partition import Partition
from std.gpu import global_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from std.math import ceildiv

comptime patch_f = DType.float32
comptime patch_i = DType.int32
comptime PATCH_BLOCK = 256


# ----------------------------------------------------------------------
# Kernel: shift every stored node coordinate by (ox, oy, oz).
# ----------------------------------------------------------------------

def offset_nodes_kernel(
    elem_node_xyz: UnsafePointer[Float32, MutAnyOrigin],
    num_points: Int,
    ox: Float32, oy: Float32, oz: Float32,
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
    nx: Int, ny: Int, nz: Int,      # owned-cube counts per axis
    loc_nx: Int, loc_ny: Int,        # local-mesh nx, ny (= owned + 2)
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

def gather_owned_nodes_kernel(
    o_owned_node_xyz: UnsafePointer[Float32, MutAnyOrigin],  # [num_owned*N_P*3]
    owned_ids:  UnsafePointer[Int32,   MutAnyOrigin],  # [num_owned]
    local_node_xyz: UnsafePointer[Float32, MutAnyOrigin],    # [num_local_elements*N_P*3]
    num_owned: Int,
):
    var tid = Int(global_idx.x)
    var total = num_owned * N_P
    if tid >= total:
        return
    var owned_idx = tid // N_P
    var nn = tid % N_P
    var src_elem = Int(owned_ids[owned_idx])
    for d in range(3):
        o_owned_node_xyz[(owned_idx * N_P + nn) * 3 + d] = (
            local_node_xyz[(src_elem * N_P + nn) * 3 + d]
        )


# ----------------------------------------------------------------------
# PatchMesh
# ----------------------------------------------------------------------

struct PatchMesh(Movable):
    var part: Partition

    # Underlying local mesh.  Built for (nx+2) x (ny+2) x (nz+2) cubes
    # with physical origin shifted to match this rank's patch.
    var mesh: Mesh

    # Owned-element index list.  Length = num_owned_elements.
    # Values are indices into mesh.d_* arrays.
    var num_owned_elements: Int
    var d_owned_elem_ids: DeviceBuffer[patch_i]

    # Host-side (owned-only) node coordinates for the per-rank VTU
    # writer.  Layout: [owned_elem][node][xyz].  Length in floats =
    # num_owned_elements * N_P * 3.
    var owned_node_xyz_f32_ptr: UnsafePointer[Float32, MutExternalOrigin]
    var owned_node_xyz_f32_len: Int

    def __init__(
        out self,
        mut ctx: DeviceContext,
        var part: Partition,
        Lx: Float64, Ly: Float64, Lz: Float64,
    ) raises:
        var nx_loc = part.nx + 2
        var ny_loc = part.ny + 2
        var nz_loc = part.nz + 2
        var dx = Lx / Float64(part.global_nx)
        var dy = Ly / Float64(part.global_ny)
        var dz = Lz / Float64(part.global_nz)
        var Lx_loc = Float64(nx_loc) * dx
        var Ly_loc = Float64(ny_loc) * dy
        var Lz_loc = Float64(nz_loc) * dz

        # Build the underlying mesh.  Its coordinates run from
        # (0, 0, 0) to (Lx_loc, Ly_loc, Lz_loc); we offset them below.
        self.mesh = Mesh(ctx, nx_loc, ny_loc, nz_loc, Lx_loc, Ly_loc, Lz_loc)

        # Shift every stored node coordinate so the mesh sits in the
        # correct global-coordinate location.  The ghost ring is one
        # cube-width *outside* the patch's owned region, so the shift
        # origin is at (cx0 - 1, cy0 - 1, cz0 - 1) * d*.
        var ox = Float32(Float64(part.cx0 - 1) * dx)
        var oy = Float32(Float64(part.cy0 - 1) * dy)
        var oz = Float32(Float64(part.cz0 - 1) * dz)
        var num_points = self.mesh.num_elements * N_P
        ctx.enqueue_function[offset_nodes_kernel, offset_nodes_kernel](
            self.mesh.d_elem_node_xyz.unsafe_ptr(),
            num_points, ox, oy, oz,
            grid_dim=ceildiv(num_points, PATCH_BLOCK),
            block_dim=PATCH_BLOCK,
        )

        # Build owned-element ID list.
        self.num_owned_elements = (
            part.nx * part.ny * part.nz * KUHN_TETS_PER_CELL
        )
        self.d_owned_elem_ids = ctx.enqueue_create_buffer[patch_i](
            self.num_owned_elements
        )
        ctx.enqueue_function[
            build_owned_elem_ids_kernel, build_owned_elem_ids_kernel,
        ](
            self.d_owned_elem_ids.unsafe_ptr(),
            part.nx, part.ny, part.nz,
            nx_loc, ny_loc,
            self.num_owned_elements,
            grid_dim=ceildiv(self.num_owned_elements, PATCH_BLOCK),
            block_dim=PATCH_BLOCK,
        )

        # Gather owned-element node coordinates into a contiguous
        # device buffer, then download to host for the VTU writer.
        var owned_points = self.num_owned_elements * N_P
        var d_owned_nodes = ctx.enqueue_create_buffer[patch_f](
            owned_points * 3
        )
        ctx.enqueue_function[
            gather_owned_nodes_kernel, gather_owned_nodes_kernel,
        ](
            d_owned_nodes.unsafe_ptr(),
            self.d_owned_elem_ids.unsafe_ptr(),
            self.mesh.d_elem_node_xyz.unsafe_ptr(),
            self.num_owned_elements,
            grid_dim=ceildiv(owned_points, PATCH_BLOCK),
            block_dim=PATCH_BLOCK,
        )
        self.owned_node_xyz_f32_len = owned_points * 3
        self.owned_node_xyz_f32_ptr = alloc[Float32](
            self.owned_node_xyz_f32_len
        )
        d_owned_nodes.enqueue_copy_to(self.owned_node_xyz_f32_ptr)
        ctx.synchronize()

        # Finally, move the partition into place.  We had to keep it
        # readable above for coordinate computations.
        self.part = part^
