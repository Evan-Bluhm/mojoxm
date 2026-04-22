# ======================================================================
# VTU XML writer for a P2 tetrahedral DG field (binary-appended format)
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
# Each element contributes its own 10 nodes (no sharing between
# elements) so DG discontinuities are preserved in ParaView.
# Cell type: 24 (VTK_QUADRATIC_TETRA).  Node ordering: 4 vertices
# then edge midpoints 0-1, 1-2, 2-0, 0-3, 1-3, 2-3 -- matches the
# element-node ordering used throughout this project.
# ======================================================================

from std.pathlib import Path
from std.memory import memcpy, memset, alloc
from std.os import FileDescriptor, open
from src.reference import N_P, N_F, N_FP
from src.nvtx import NvtxContext
from src.async_writer import WriteSegment

comptime VTK_QUADRATIC_TETRA = 24

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

    def write_bytes[
        mut: Bool, //, origin: Origin[mut=mut]
    ](mut self, src: UnsafePointer[UInt8, origin], n: Int):
        memcpy(dest=self.ptr + self.len, src=src, count=n)
        self.len += n

    def write_str(mut self, s: String):
        var n = s.byte_length()
        memcpy(dest=self.ptr + self.len,
               src=s.unsafe_ptr().bitcast[UInt8](),
               count=n)
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

    var _last_size: Int

    def __init__[
        mut: Bool, //, origin: Origin[mut=mut]
    ](out self,
        num_elements: Int,
        elem_node_xyz_ptr: UnsafePointer[Float32, origin],
    ) raises:
        # `elem_node_xyz_ptr` points to num_elements * N_P * 3 Float32s,
        # laid out (elem, node, component).  We only read it.
        self.num_elements = num_elements
        self.total_points = num_elements * N_P
        var total_points = self.total_points

        var pts_bytes = total_points * 3 * 4
        var conn_bytes = total_points * 4
        var off_bytes = num_elements * 4
        var typ_bytes = num_elements

        # Tiny 4-byte prefix: the points array's byte-count u32.
        self.static_pre = ByteBuf(4)
        self.static_pre.write_u32_le(UInt32(pts_bytes))

        # Reference to the caller's points buffer (not copied).
        self.pts_ptr = rebind[UnsafePointer[UInt8, MutAnyOrigin]](
            elem_node_xyz_ptr.bitcast[UInt8]()
        )
        self.pts_byte_len = pts_bytes

        # Post-points static blob: conn + offsets + types, each with
        # its u32 byte-count header.  ~29 MB at 48^3 instead of 109 MB,
        # so the ByteBuf allocation alone is 4x smaller.
        var post_size = 4 + conn_bytes + 4 + off_bytes + 4 + typ_bytes
        self.static_post = ByteBuf(post_size)

        # Connectivity (Int32): [0, 1, 2, ..., total_points-1]
        self.static_post.write_u32_le(UInt32(conn_bytes))
        var conn_ptr = (
            self.static_post.ptr + self.static_post.len
        ).bitcast[Int32]()
        for i in range(total_points):
            conn_ptr[i] = Int32(i)
        self.static_post.len += conn_bytes

        # Offsets (Int32): (k+1) * N_P
        self.static_post.write_u32_le(UInt32(off_bytes))
        var off_ptr = (
            self.static_post.ptr + self.static_post.len
        ).bitcast[Int32]()
        for e in range(num_elements):
            off_ptr[e] = Int32((e + 1) * N_P)
        self.static_post.len += off_bytes

        # Types (UInt8): all VTK_QUADRATIC_TETRA -> memset.
        self.static_post.write_u32_le(UInt32(typ_bytes))
        memset(
            ptr=self.static_post.ptr + self.static_post.len,
            value=UInt8(VTK_QUADRATIC_TETRA),
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
        hdr += '<UnstructuredGrid>\n'
        hdr += '<Piece NumberOfPoints="' + String(total_points) + '" NumberOfCells="' + String(num_elements) + '">\n'
        hdr += '<PointData Scalars="density">\n'
        hdr += '<DataArray type="Float32" Name="density" format="appended" offset="' + String(self.off_density) + '"/>\n'
        hdr += '</PointData>\n'
        hdr += '<Points>\n'
        hdr += '<DataArray type="Float32" NumberOfComponents="3" format="appended" offset="' + String(self.off_points) + '"/>\n'
        hdr += '</Points>\n'
        hdr += '<Cells>\n'
        hdr += '<DataArray type="Int32" Name="connectivity" format="appended" offset="' + String(self.off_conn) + '"/>\n'
        hdr += '<DataArray type="Int32" Name="offsets" format="appended" offset="' + String(self.off_offsets) + '"/>\n'
        hdr += '<DataArray type="UInt8" Name="types" format="appended" offset="' + String(self.off_types) + '"/>\n'
        hdr += '</Cells>\n'
        hdr += '</Piece>\n'
        hdr += '</UnstructuredGrid>\n'
        hdr += '<AppendedData encoding="raw">\n_'
        self.xml_header = hdr^
        self.xml_tail = String('\n</AppendedData>\n</VTKFile>\n')
        self._last_size = 0

    def write_frame(mut self, path: String, q: List[Float32],
                   mut nvtx: NvtxContext) raises:
        """Synchronous write -- deprecated, kept for compatibility.
        Prefer `build_segments()` + AsyncWriter.submit() on the hot path."""
        nvtx.push_range("vtu_serialize")
        var buf_ptr = self._serialize(q)
        var buf_len = self._last_size
        nvtx.pop_range()

        nvtx.push_range("vtu_file_write")
        var p = Path(path)
        var span = Span(ptr=buf_ptr, length=buf_len)
        p.write_bytes(span)
        nvtx.pop_range()
        buf_ptr.free()

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
    def build_segments(
        mut self, q: List[Float32]
    ) raises -> List[WriteSegment]:
        var total_points = self.total_points
        var density_bytes = total_points * 4

        var hdr_seg = WriteSegment(
            rebind[UnsafePointer[UInt8, MutAnyOrigin]](
                self.xml_header.unsafe_ptr()
            ),
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
            rebind[UnsafePointer[UInt8, MutAnyOrigin]](
                self.xml_tail.unsafe_ptr()
            ),
            self.xml_tail.byte_length(),
            False,
        )

        var segs = [hdr_seg, density_seg, pre_seg, pts_seg, post_seg, tail_seg]
        return segs^

    def last_size(self) -> Int:
        return self._last_size

    def _serialize(
        mut self, q: List[Float32]
    ) raises -> UnsafePointer[UInt8, MutAnyOrigin]:
        var total_points = self.total_points
        var density_bytes = total_points * 4
        var total_size = (
            self.xml_header.byte_length()
            + 4 + density_bytes
            + self.static_pre.len
            + self.pts_byte_len
            + self.static_post.len
            + self.xml_tail.byte_length()
        )
        var out_ptr = alloc[UInt8](total_size)
        var cursor = 0

        var hn = self.xml_header.byte_length()
        memcpy(dest=out_ptr + cursor,
               src=self.xml_header.unsafe_ptr().bitcast[UInt8](), count=hn)
        cursor += hn

        out_ptr[cursor + 0] = UInt8(UInt32(density_bytes) & 0xFF)
        out_ptr[cursor + 1] = UInt8((UInt32(density_bytes) >> 8) & 0xFF)
        out_ptr[cursor + 2] = UInt8((UInt32(density_bytes) >> 16) & 0xFF)
        out_ptr[cursor + 3] = UInt8((UInt32(density_bytes) >> 24) & 0xFF)
        cursor += 4

        memcpy(dest=out_ptr + cursor,
               src=q.unsafe_ptr().bitcast[UInt8](), count=density_bytes)
        cursor += density_bytes

        memcpy(dest=out_ptr + cursor,
               src=self.static_pre.ptr, count=self.static_pre.len)
        cursor += self.static_pre.len

        memcpy(dest=out_ptr + cursor,
               src=self.pts_ptr, count=self.pts_byte_len)
        cursor += self.pts_byte_len

        memcpy(dest=out_ptr + cursor,
               src=self.static_post.ptr, count=self.static_post.len)
        cursor += self.static_post.len

        var tn = self.xml_tail.byte_length()
        memcpy(dest=out_ptr + cursor,
               src=self.xml_tail.unsafe_ptr().bitcast[UInt8](), count=tn)
        cursor += tn

        self._last_size = cursor
        return rebind[UnsafePointer[UInt8, MutAnyOrigin]](out_ptr)


# ----------------------------------------------------------------------
# ParaView collection (.pvd) file
# ----------------------------------------------------------------------

def write_pvd(path: String, vtu_paths: List[String], times: List[Float64]) raises:
    var out = String()
    out += '<?xml version="1.0"?>\n'
    out += '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n'
    out += '<Collection>\n'
    for i in range(len(vtu_paths)):
        out += '<DataSet timestep="'
        out += String(times[i])
        out += '" group="" part="0" file="'
        out += vtu_paths[i]
        out += '"/>\n'
    out += '</Collection>\n'
    out += '</VTKFile>\n'
    var p = Path(path)
    p.write_text(out)
