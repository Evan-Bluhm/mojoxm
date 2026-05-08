# ======================================================================
# VTU XML writer for arbitrary-P tetrahedral DG fields (binary-appended)
# ======================================================================
#
# Outputs a VTK UnstructuredGrid file with the "appended raw binary"
# payload convention.  Each DataArray points into the appended blob
# via an offset; the blob starts after a literal '_' marker.  Each
# array is preceded by a 32-bit unsigned byte-count header (standard
# VTK/ParaView convention).
#
# Optimization notes
# ------------------
#   * Mesh data (points, connectivity, offsets, types) does not change
#     between frames, so the `VtuWriter` caches it as a single
#     pre-serialized blob at construction time.  Per frame we only
#     format the density array and then do three bulk writes (header,
#     density, static-mesh-blob + tail).
#   * Everything below uses raw UnsafePointer + memcpy for the hot
#     path; List[UInt8].append was previously dominating wall time.
#
# Each element contributes its own NP nodes (NP = (P+1)(P+2)(P+3)/6:
# 10/20/35/56 at P=2/3/4/5) -- no sharing between elements so DG
# discontinuities are preserved in ParaView.
#
# Cell type:
#   * P=2 (NP=10): emits VTK_QUADRATIC_TETRA (24).  Node ordering is
#     4 vertices then edge midpoints 0-1, 1-2, 2-0, 0-3, 1-3, 2-3.
#   * P>=3 (NP=20/35/56): emits VTK_LAGRANGE_TETRAHEDRON (71).  Node
#     ordering follows VTK's arbitrary-order Lagrange spec via
#     `_lagrange_tet_exponents(P)` in src.local_mesh; gated end-to-
#     end by `make test-vtu-meshio` (meshio-roundtrip + spec
#     validation in scripts/validate_vtu.py).
# ======================================================================

from std.pathlib import Path
from std.memory import memcpy, memset, alloc
from src.async_writer import WriteSegment

# At P=2 each tet has 10 nodes laid out as 4 vertices + 6 edge midpoints,
# which matches VTK_QUADRATIC_TETRA's node ordering bit-for-bit.  For
# P != 2 we emit VTK_LAGRANGE_TETRAHEDRON (arbitrary-order Lagrange tet)
# and rely on the canonical node ordering produced by
# `_lagrange_tet_exponents(P)` in src.local_mesh matching VTK's
# expected Lagrange ordering.
comptime VTK_QUADRATIC_TETRA = 24
comptime VTK_LAGRANGE_TETRAHEDRON = 71

# ----------------------------------------------------------------------
# Bulk writer: owns a malloc'd byte buffer + write cursor.
# ----------------------------------------------------------------------


struct ByteBuf(Movable):
    var ptr: UnsafePointer[UInt8, MutExternalOrigin]
    var cap: Int
    var len: Int

    def __init__(out self, capacity: Int):
        self.ptr = alloc[UInt8](capacity)
        self.cap = capacity
        self.len = 0

    # Note: no __del__ -- in Mojo 0.26.2 we were seeing spurious
    # destructor invocations on ByteBuf values that were still live in
    # another context (see git history).  Since the ByteBufs we keep
    # alive for the duration of the simulation are process-lifetime
    # (VtuWriter's xml/static/tail), we accept the memory leak at exit
    # rather than risk freeing data still referenced by writer threads.

    def write_u32_le(mut self, v: UInt32):
        var p = self.ptr + self.len
        p[0] = UInt8(v & 0xFF)
        p[1] = UInt8((v >> 8) & 0xFF)
        p[2] = UInt8((v >> 16) & 0xFF)
        p[3] = UInt8((v >> 24) & 0xFF)
        self.len += 4

    def write_bytes[mut: Bool, //, origin: Origin[mut=mut]](mut self, src: UnsafePointer[UInt8, origin], n: Int):
        memcpy(dest=self.ptr + self.len, src=src, count=n)
        self.len += n

    def write_str(mut self, s: String):
        var n = s.byte_length()
        memcpy(
            dest=self.ptr + self.len,
            src=s.unsafe_ptr().bitcast[UInt8](),
            count=n,
        )
        self.len += n


# ----------------------------------------------------------------------
# Static (mesh-geometry) blob - computed once, reused per frame.
# ----------------------------------------------------------------------
#
# Layout (within the `AppendedData` section after '_'):
#   [u32] density byte count | density bytes            <-- per-frame
#   [u32] points byte count  | points bytes             \
#   [u32] conn byte count    | conn bytes                |
#   [u32] offsets byte count | offsets bytes             | static
#   [u32] types byte count   | types bytes              /
#
# We build the static (post-density) suffix once, including its own
# leading u32 headers.


struct VtuWriter(Movable):
    var num_elements: Int
    var nodes_per_elem: Int
    var total_points: Int

    # We keep three static pieces on the host:
    #   * xml_header  : text up through the AppendedData '_' marker.
    #   * static_pre  : 4-byte u32 byte-count for the points array.
    #   * static_post : [u32 conn][conn][u32 off][off][u32 typ][typ]
    # The points array itself is NOT copied here.  We reference the
    # caller-owned elem_node_xyz buffer directly in build_segments(),
    # so the 80 MB mesh coords live in exactly one host-side allocation
    # (owned by Mesh) and are scatter-gather-written straight out to
    # disk.
    var static_pre: ByteBuf
    var static_post: ByteBuf
    var pts_ptr: UnsafePointer[UInt8, MutAnyOrigin]
    var pts_byte_len: Int

    var xml_header: String
    var xml_tail: String

    # Offsets (relative to the '_' marker) passed into the XML header.
    var off_density: Int
    var off_points: Int
    var off_conn: Int
    var off_offsets: Int
    var off_types: Int

    def __init__[
        mut: Bool, //, origin: Origin[mut=mut]
    ](out self, num_elements: Int, elem_node_xyz_ptr: UnsafePointer[Float32, origin], nodes_per_elem: Int = 10,) raises:
        # `elem_node_xyz_ptr` points to num_elements * nodes_per_elem * 3
        # Float32s, laid out (elem, node, component).  We only read it.
        # `nodes_per_elem` defaults to 10 (Lagrange P=2), but any
        # num_tet_nodes(P) value is accepted.
        self.num_elements = num_elements
        self.nodes_per_elem = nodes_per_elem
        self.total_points = num_elements * nodes_per_elem
        var total_points = self.total_points
        var cell_type = VTK_QUADRATIC_TETRA if nodes_per_elem == 10 else VTK_LAGRANGE_TETRAHEDRON

        var pts_bytes = total_points * 3 * 4
        var conn_bytes = total_points * 4
        var off_bytes = num_elements * 4
        var typ_bytes = num_elements

        # Tiny 4-byte prefix: the points array's byte-count u32.
        self.static_pre = ByteBuf(4)
        self.static_pre.write_u32_le(UInt32(pts_bytes))

        # Reference to the caller's points buffer (not copied).
        self.pts_ptr = rebind[UnsafePointer[UInt8, MutAnyOrigin]](elem_node_xyz_ptr.bitcast[UInt8]())
        self.pts_byte_len = pts_bytes

        # Post-points static blob: conn + offsets + types, each with
        # its u32 byte-count header.  ~29 MB at 48^3 instead of 109 MB,
        # so the ByteBuf allocation alone is 4x smaller.
        var post_size = 4 + conn_bytes + 4 + off_bytes + 4 + typ_bytes
        self.static_post = ByteBuf(post_size)

        # Connectivity (Int32): [0, 1, 2, ..., total_points-1]
        self.static_post.write_u32_le(UInt32(conn_bytes))
        var conn_ptr = (self.static_post.ptr + self.static_post.len).bitcast[Int32]()
        for i in range(total_points):
            conn_ptr[i] = Int32(i)
        self.static_post.len += conn_bytes

        # Offsets (Int32): (k+1) * nodes_per_elem
        self.static_post.write_u32_le(UInt32(off_bytes))
        var off_ptr = (self.static_post.ptr + self.static_post.len).bitcast[Int32]()
        for e in range(num_elements):
            off_ptr[e] = Int32((e + 1) * nodes_per_elem)
        self.static_post.len += off_bytes

        # Types (UInt8): memset with the chosen cell type.
        self.static_post.write_u32_le(UInt32(typ_bytes))
        memset(
            ptr=self.static_post.ptr + self.static_post.len,
            value=UInt8(cell_type),
            count=num_elements,
        )
        self.static_post.len += typ_bytes

        # Offsets (relative to '_'): density is first, static follows.
        self.off_density = 0
        self.off_points = 4 + total_points * 4
        self.off_conn = self.off_points + 4 + pts_bytes
        self.off_offsets = self.off_conn + 4 + conn_bytes
        self.off_types = self.off_offsets + 4 + off_bytes

        # XML header (up to and including '_').
        var hdr = String()
        hdr += '<?xml version="1.0"?>\n'
        hdr += '<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian" header_type="UInt32">\n'
        hdr += "<UnstructuredGrid>\n"
        hdr += '<Piece NumberOfPoints="' + String(total_points) + '" NumberOfCells="' + String(num_elements) + '">\n'
        hdr += '<PointData Scalars="density">\n'
        hdr += (
            '<DataArray type="Float32" Name="density" format="appended" offset="' + String(self.off_density) + '"/>\n'
        )
        hdr += "</PointData>\n"
        hdr += "<Points>\n"
        hdr += (
            '<DataArray type="Float32" NumberOfComponents="3" format="appended" offset="'
            + String(self.off_points)
            + '"/>\n'
        )
        hdr += "</Points>\n"
        hdr += "<Cells>\n"
        hdr += (
            '<DataArray type="Int32" Name="connectivity" format="appended" offset="' + String(self.off_conn) + '"/>\n'
        )
        hdr += '<DataArray type="Int32" Name="offsets" format="appended" offset="' + String(self.off_offsets) + '"/>\n'
        hdr += '<DataArray type="UInt8" Name="types" format="appended" offset="' + String(self.off_types) + '"/>\n'
        hdr += "</Cells>\n"
        hdr += "</Piece>\n"
        hdr += "</UnstructuredGrid>\n"
        hdr += '<AppendedData encoding="raw">\n_'
        self.xml_header = hdr^
        self.xml_tail = String("\n</AppendedData>\n</VTKFile>\n")

    # Build a scatter-gather list of six segments:
    #   1. xml_header         (owned by VtuWriter, not freed)
    #   2. density (u32 + data, per-frame, owned)
    #   3. static_pre  (owned by VtuWriter, not freed)  -- 4-byte points count
    #   4. points ptr   (owned by Mesh, not freed)       -- 80 MB zero-copy
    #   5. static_post (owned by VtuWriter, not freed)  -- conn+offsets+types
    #   6. xml_tail            (owned by VtuWriter, not freed)
    # Only the density segment is copied per frame (~26 MB at 48^3).
    # The 80 MB mesh coords are referenced once from Mesh's host buffer
    # and fed to writev() in place -- no copy anywhere in the pipeline.
    def build_segments(mut self, q: List[Float32]) raises -> List[WriteSegment]:
        var total_points = self.total_points
        var density_bytes = total_points * 4

        var hdr_seg = WriteSegment(
            rebind[UnsafePointer[UInt8, MutAnyOrigin]](self.xml_header.unsafe_ptr()),
            self.xml_header.byte_length(),
            False,
        )

        # Density -- owned; 4-byte u32 count prefix + density data
        var density_seg_len = 4 + density_bytes
        var density_seg_ptr = alloc[UInt8](density_seg_len)
        density_seg_ptr[0] = UInt8(UInt32(density_bytes) & 0xFF)
        density_seg_ptr[1] = UInt8((UInt32(density_bytes) >> 8) & 0xFF)
        density_seg_ptr[2] = UInt8((UInt32(density_bytes) >> 16) & 0xFF)
        density_seg_ptr[3] = UInt8((UInt32(density_bytes) >> 24) & 0xFF)
        memcpy(
            dest=density_seg_ptr + 4,
            src=q.unsafe_ptr().bitcast[UInt8](),
            count=density_bytes,
        )
        var density_seg = WriteSegment(
            rebind[UnsafePointer[UInt8, MutAnyOrigin]](density_seg_ptr),
            density_seg_len,
            True,
        )

        var pre_seg = WriteSegment(
            self.static_pre.ptr,
            self.static_pre.len,
            False,
        )
        var pts_seg = WriteSegment(
            self.pts_ptr,
            self.pts_byte_len,
            False,
        )
        var post_seg = WriteSegment(
            self.static_post.ptr,
            self.static_post.len,
            False,
        )
        var tail_seg = WriteSegment(
            rebind[UnsafePointer[UInt8, MutAnyOrigin]](self.xml_tail.unsafe_ptr()),
            self.xml_tail.byte_length(),
            False,
        )

        var segs = [hdr_seg, density_seg, pre_seg, pts_seg, post_seg, tail_seg]
        return segs^


# ----------------------------------------------------------------------
# ParaView collection (.pvd) file
# ----------------------------------------------------------------------


def write_pvd(path: String, vtu_paths: List[String], times: List[Float64]) raises:
    """Emit a ParaView `.pvd` collection that ties a sequence of VTU
    frames to their physical timestamps.  Used by both the 3D
    `FrameWriter.finalize` and the 2D example drivers (via
    `src.vtu_2d.dump_pvd_collection`, which re-exports this).
    `vtu_paths` are relative to the `.pvd`'s directory."""
    if len(vtu_paths) != len(times):
        raise Error(
            "write_pvd: vtu_paths/times length mismatch (" + String(len(vtu_paths)) + " vs " + String(len(times)) + ")"
        )
    var out = String()
    out += '<?xml version="1.0"?>\n'
    out += '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n'
    out += "<Collection>\n"
    for i in range(len(vtu_paths)):
        out += '<DataSet timestep="'
        out += String(times[i])
        out += '" group="" part="0" file="'
        out += vtu_paths[i]
        out += '"/>\n'
    out += "</Collection>\n"
    out += "</VTKFile>\n"
    var p = Path(path)
    p.write_text(out)


# ----------------------------------------------------------------------
# Multi-field 3D VTU writer (synchronous)
# ----------------------------------------------------------------------
# Mirror of `src.vtu_2d.dump_vtu_2d_frame_multi` for the 3D side.  The
# production `FrameWriter` / `VtuWriter` / `AsyncWriter` path emits a
# single hardcoded "density" field per frame using a scatter-gather
# writev for performance.  This synchronous helper accepts N named
# scalar fields, does one host-side serialisation pass, and writes
# the whole VTU in a single `write_bytes`.  Slower per frame than
# the async path but it unblocks 3D drivers that want to dump
# multiple physically meaningful fields per frame (e.g. rho + p +
# |v| for Euler) without refactoring the async pipeline.
#
# Layout:
#   - Points:     num_elements * NP, each stored as 3 Float32 (x, y, z).
#   - Cells:      num_elements tets.
#   - Cell type:  24 (VTK_QUADRATIC_TETRA) at NP=10, 71 (VTK_LAGRANGE_
#                 TETRAHEDRON) elsewhere.
#   - PointData:  N scalar fields (Float32) at every nodal DOF, in
#                 the order passed.  The first field is exposed as
#                 the PointData `Scalars` default (ParaView opens to
#                 this).
# ----------------------------------------------------------------------


def _u32_le(
    buf: UnsafePointer[UInt8, MutAnyOrigin],
    offset: Int,
    v: UInt32,
):
    buf[offset + 0] = UInt8(v & 0xFF)
    buf[offset + 1] = UInt8((v >> 8) & 0xFF)
    buf[offset + 2] = UInt8((v >> 16) & 0xFF)
    buf[offset + 3] = UInt8((v >> 24) & 0xFF)


def dump_vtu_3d_frame_multi(
    num_elements: Int,
    nodes_per_elem: Int,
    elem_node_xyz: UnsafePointer[Float32, MutAnyOrigin],
    field_names: List[String],
    field_data: List[List[Float64]],
    path: String,
) raises:
    """Serialise one 3D frame to `path` as a standalone VTU with N
    scalar PointData fields.  All `field_data[i]` arrays must be flat
    `num_elements * nodes_per_elem` Float64.  `elem_node_xyz` is the
    per-node x/y/z layout already used by `VtuWriter` / `FrameWriter`
    (length `num_elements * nodes_per_elem * 3`).  The first field is
    exposed as the PointData `Scalars` default."""
    if len(field_names) != len(field_data):
        raise Error(
            "dump_vtu_3d_frame_multi: field_names/field_data length mismatch ("
            + String(len(field_names))
            + " vs "
            + String(len(field_data))
            + ")"
        )
    if len(field_names) == 0:
        raise Error("dump_vtu_3d_frame_multi: at least one field required")

    var total_points = num_elements * nodes_per_elem
    for i in range(len(field_data)):
        if len(field_data[i]) != total_points:
            raise Error(
                String("dump_vtu_3d_frame_multi: field_data[")
                + String(i)
                + "] size "
                + String(len(field_data[i]))
                + " != "
                + String(total_points)
            )
    var n_fields = len(field_names)

    var cell_type = VTK_QUADRATIC_TETRA if nodes_per_elem == 10 else VTK_LAGRANGE_TETRAHEDRON

    var field_bytes = total_points * 4
    var points_bytes = total_points * 3 * 4
    var conn_bytes = total_points * 4
    var off_bytes = num_elements * 4
    var typ_bytes = num_elements

    var off_fields = List[Int]()
    for i in range(n_fields):
        off_fields.append(i * (4 + field_bytes))
    var off_points = n_fields * (4 + field_bytes)
    var off_conn = off_points + 4 + points_bytes
    var off_offsets = off_conn + 4 + conn_bytes
    var off_types = off_offsets + 4 + off_bytes

    var hdr = String()
    hdr += '<?xml version="1.0"?>\n'
    hdr += '<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian" header_type="UInt32">\n'
    hdr += "<UnstructuredGrid>\n"
    hdr += '<Piece NumberOfPoints="' + String(total_points) + '" NumberOfCells="' + String(num_elements) + '">\n'
    hdr += '<PointData Scalars="' + field_names[0] + '">\n'
    for i in range(n_fields):
        hdr += (
            '<DataArray type="Float32" Name="'
            + field_names[i]
            + '" format="appended" offset="'
            + String(off_fields[i])
            + '"/>\n'
        )
    hdr += "</PointData>\n"
    hdr += "<Points>\n"
    hdr += '<DataArray type="Float32" NumberOfComponents="3" format="appended" offset="' + String(off_points) + '"/>\n'
    hdr += "</Points>\n"
    hdr += "<Cells>\n"
    hdr += '<DataArray type="Int32" Name="connectivity" format="appended" offset="' + String(off_conn) + '"/>\n'
    hdr += '<DataArray type="Int32" Name="offsets" format="appended" offset="' + String(off_offsets) + '"/>\n'
    hdr += '<DataArray type="UInt8" Name="types" format="appended" offset="' + String(off_types) + '"/>\n'
    hdr += "</Cells>\n"
    hdr += "</Piece>\n"
    hdr += "</UnstructuredGrid>\n"
    hdr += '<AppendedData encoding="raw">\n_'

    var tail = String("\n</AppendedData>\n</VTKFile>\n")

    var blob_size = n_fields * (4 + field_bytes) + 4 + points_bytes + 4 + conn_bytes + 4 + off_bytes + 4 + typ_bytes
    var total_size = hdr.byte_length() + blob_size + tail.byte_length()
    var out = alloc[UInt8](total_size)
    var cur = 0

    memcpy(
        dest=out + cur,
        src=hdr.unsafe_ptr().bitcast[UInt8](),
        count=hdr.byte_length(),
    )
    cur += hdr.byte_length()

    for i in range(n_fields):
        _u32_le(
            rebind[UnsafePointer[UInt8, MutAnyOrigin]](out),
            cur,
            UInt32(field_bytes),
        )
        cur += 4
        var fi_f32 = (out + cur).bitcast[Float32]()
        for k in range(total_points):
            fi_f32[k] = Float32(field_data[i][k])
        cur += field_bytes

    # Points: copy bulk x/y/z directly (already 3-component Float32).
    _u32_le(
        rebind[UnsafePointer[UInt8, MutAnyOrigin]](out),
        cur,
        UInt32(points_bytes),
    )
    cur += 4
    memcpy(
        dest=out + cur,
        src=elem_node_xyz.bitcast[UInt8](),
        count=points_bytes,
    )
    cur += points_bytes

    _u32_le(rebind[UnsafePointer[UInt8, MutAnyOrigin]](out), cur, UInt32(conn_bytes))
    cur += 4
    var conn_i32 = (out + cur).bitcast[Int32]()
    for k in range(total_points):
        conn_i32[k] = Int32(k)
    cur += conn_bytes

    _u32_le(rebind[UnsafePointer[UInt8, MutAnyOrigin]](out), cur, UInt32(off_bytes))
    cur += 4
    var off_i32 = (out + cur).bitcast[Int32]()
    for e in range(num_elements):
        off_i32[e] = Int32((e + 1) * nodes_per_elem)
    cur += off_bytes

    _u32_le(rebind[UnsafePointer[UInt8, MutAnyOrigin]](out), cur, UInt32(typ_bytes))
    cur += 4
    memset(ptr=out + cur, value=UInt8(cell_type), count=num_elements)
    cur += typ_bytes

    memcpy(
        dest=out + cur,
        src=tail.unsafe_ptr().bitcast[UInt8](),
        count=tail.byte_length(),
    )
    cur += tail.byte_length()

    var p = Path(path)
    var span = Span(ptr=out, length=total_size)
    p.write_bytes(span)
    out.free()
