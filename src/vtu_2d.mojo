# ======================================================================
# 2D VTU frame writer
# ======================================================================
#
# One-shot dump of a 2D triangulated scalar field to a VTK
# UnstructuredGrid file (appended-raw binary).  Host-only; suitable for
# CPU-stepped advection demos before the 2D GPU solver lands.
#
#   - Points:     num_elements * NP points, each stored 3D with z=0
#                 so ParaView opens them in a 3D viewer naturally.
#   - Cells:      num_elements cells, one per triangle.
#   - Cell type:  VTK_TRIANGLE (5) at P=1, VTK_QUADRATIC_TRIANGLE (22)
#                 at P=2, VTK_LAGRANGE_TRIANGLE (69) at P >= 3.  The
#                 first two use our canonical 3- and 6-node orderings
#                 (matching VTK); P >= 3 assumes the `_tri_node_exponents`
#                 ordering matches VTK Lagrange (validated up to P=2;
#                 may need remap for P >= 3).
#   - PointData:  one scalar field at every nodal DOF.
#
# Unlike the 3D `VtuWriter`, this writer does not cache per-frame
# blobs -- 2D meshes are small enough that the cost is negligible.
# ======================================================================

from std.pathlib import Path
from std.memory import alloc, memcpy, memset
from src.local_mesh_2d import LocalMesh2D
from src.reference_2d import num_tri_nodes_2d


comptime VTK_TRIANGLE = 5
comptime VTK_QUADRATIC_TRIANGLE = 22
comptime VTK_LAGRANGE_TRIANGLE = 69


def _cell_type_for(nodes_per_elem: Int) -> Int:
    if nodes_per_elem == 3:
        return VTK_TRIANGLE
    if nodes_per_elem == 6:
        return VTK_QUADRATIC_TRIANGLE
    return VTK_LAGRANGE_TRIANGLE


def _u32_le(buf: UnsafePointer[UInt8, MutAnyOrigin], offset: Int, v: UInt32):
    buf[offset + 0] = UInt8(v & 0xFF)
    buf[offset + 1] = UInt8((v >> 8) & 0xFF)
    buf[offset + 2] = UInt8((v >> 16) & 0xFF)
    buf[offset + 3] = UInt8((v >> 24) & 0xFF)


def dump_vtu_2d_frame[P: Int](
    mesh: LocalMesh2D[P],
    q: List[Float64],
    path: String,
    field_name: String = String("q"),
) raises:
    """Serialise one frame to `path` as a standalone VTU.  `q` is flat
    `num_elements * NP` Float64, each component becomes a scalar at one
    nodal DOF in PointData."""
    var NP_p = num_tri_nodes_2d(P)
    var num_elements = mesh.num_elements
    var total_points = num_elements * NP_p
    if len(q) != total_points:
        raise Error("dump_vtu_2d_frame: q size mismatch")

    var cell_type = _cell_type_for(NP_p)

    # Binary appended blob layout (after the '_' marker):
    #   [u32 bytes][field Float32 * total_points]           -- scalar
    #   [u32 bytes][points Float32 * total_points * 3]       -- x, y, z=0
    #   [u32 bytes][connectivity Int32 * total_points]       -- 0..total-1
    #   [u32 bytes][offsets Int32 * num_elements]            -- (k+1) * NP
    #   [u32 bytes][types UInt8 * num_elements]              -- cell_type
    var field_bytes = total_points * 4
    var points_bytes = total_points * 3 * 4
    var conn_bytes = total_points * 4
    var off_bytes = num_elements * 4
    var typ_bytes = num_elements

    var off_field   = 0
    var off_points  = off_field + 4 + field_bytes
    var off_conn    = off_points + 4 + points_bytes
    var off_offsets = off_conn + 4 + conn_bytes
    var off_types   = off_offsets + 4 + off_bytes

    var hdr = String()
    hdr += '<?xml version="1.0"?>\n'
    hdr += ('<VTKFile type="UnstructuredGrid" version="0.1"'
            ' byte_order="LittleEndian" header_type="UInt32">\n')
    hdr += '<UnstructuredGrid>\n'
    hdr += ('<Piece NumberOfPoints="' + String(total_points)
            + '" NumberOfCells="' + String(num_elements) + '">\n')
    hdr += '<PointData Scalars="' + field_name + '">\n'
    hdr += ('<DataArray type="Float32" Name="' + field_name
            + '" format="appended" offset="' + String(off_field) + '"/>\n')
    hdr += '</PointData>\n'
    hdr += '<Points>\n'
    hdr += ('<DataArray type="Float32" NumberOfComponents="3"'
            ' format="appended" offset="' + String(off_points) + '"/>\n')
    hdr += '</Points>\n'
    hdr += '<Cells>\n'
    hdr += ('<DataArray type="Int32" Name="connectivity"'
            ' format="appended" offset="' + String(off_conn) + '"/>\n')
    hdr += ('<DataArray type="Int32" Name="offsets"'
            ' format="appended" offset="' + String(off_offsets) + '"/>\n')
    hdr += ('<DataArray type="UInt8" Name="types"'
            ' format="appended" offset="' + String(off_types) + '"/>\n')
    hdr += '</Cells>\n'
    hdr += '</Piece>\n'
    hdr += '</UnstructuredGrid>\n'
    hdr += '<AppendedData encoding="raw">\n_'

    var tail = String('\n</AppendedData>\n</VTKFile>\n')

    # Total buffer size.
    var blob_size = (
        4 + field_bytes
        + 4 + points_bytes
        + 4 + conn_bytes
        + 4 + off_bytes
        + 4 + typ_bytes
    )
    var total_size = hdr.byte_length() + blob_size + tail.byte_length()
    var out = alloc[UInt8](total_size)
    var cur = 0

    # XML header.
    memcpy(
        dest=out + cur,
        src=hdr.unsafe_ptr().bitcast[UInt8](),
        count=hdr.byte_length(),
    )
    cur += hdr.byte_length()

    # Field data.
    _u32_le(
        rebind[UnsafePointer[UInt8, MutAnyOrigin]](out), cur, UInt32(field_bytes)
    )
    cur += 4
    var field_f32 = (out + cur).bitcast[Float32]()
    for k in range(total_points):
        field_f32[k] = Float32(q[k])
    cur += field_bytes

    # Points: pad to 3D with z=0.
    _u32_le(
        rebind[UnsafePointer[UInt8, MutAnyOrigin]](out), cur, UInt32(points_bytes)
    )
    cur += 4
    var pts_f32 = (out + cur).bitcast[Float32]()
    for elem in range(num_elements):
        for nn in range(NP_p):
            var x = mesh.elem_node_xyz[(elem * NP_p + nn) * 2 + 0]
            var y = mesh.elem_node_xyz[(elem * NP_p + nn) * 2 + 1]
            pts_f32[(elem * NP_p + nn) * 3 + 0] = Float32(x)
            pts_f32[(elem * NP_p + nn) * 3 + 1] = Float32(y)
            pts_f32[(elem * NP_p + nn) * 3 + 2] = Float32(0.0)
    cur += points_bytes

    # Connectivity: identity (each element owns its own NP points).
    _u32_le(
        rebind[UnsafePointer[UInt8, MutAnyOrigin]](out), cur, UInt32(conn_bytes)
    )
    cur += 4
    var conn_i32 = (out + cur).bitcast[Int32]()
    for k in range(total_points):
        conn_i32[k] = Int32(k)
    cur += conn_bytes

    # Offsets.
    _u32_le(
        rebind[UnsafePointer[UInt8, MutAnyOrigin]](out), cur, UInt32(off_bytes)
    )
    cur += 4
    var off_i32 = (out + cur).bitcast[Int32]()
    for e in range(num_elements):
        off_i32[e] = Int32((e + 1) * NP_p)
    cur += off_bytes

    # Types.
    _u32_le(
        rebind[UnsafePointer[UInt8, MutAnyOrigin]](out), cur, UInt32(typ_bytes)
    )
    cur += 4
    memset(ptr=out + cur, value=UInt8(cell_type), count=num_elements)
    cur += typ_bytes

    # XML tail.
    memcpy(
        dest=out + cur,
        src=tail.unsafe_ptr().bitcast[UInt8](),
        count=tail.byte_length(),
    )
    cur += tail.byte_length()

    # Write to disk via pathlib (auto-mkdir parent dirs).
    var p = Path(path)
    var span = Span(ptr=out, length=total_size)
    p.write_bytes(span)
    out.free()


# ----------------------------------------------------------------------
# PVD time-series collection emitter
# ----------------------------------------------------------------------
# After dumping a sequence of `dump_vtu_2d_frame` files, callers want a
# `.pvd` (ParaView XML) collection that ties them together with their
# physical timestamps.  Format:
#
#   <?xml version="1.0"?>
#   <VTKFile type="Collection" version="0.1" byte_order="LittleEndian">
#   <Collection>
#     <DataSet timestep="0.0"   group="" part="0" file="frame_NNNNN.vtu"/>
#     <DataSet timestep="0.05"  group="" part="0" file="frame_NNNNN.vtu"/>
#     ...
#   </Collection>
#   </VTKFile>
#
# This helper centralises the XML emission that the 2D example drivers
# all duplicated -- 14-line blocks repeated 9x with no per-driver
# variation.  `paths` are written into the PVD verbatim (relative to
# the PVD's directory), `times` are the corresponding physical
# timestamps in the same order.  Lengths must match.
# ----------------------------------------------------------------------

def dump_pvd_collection(
    pvd_path: String,
    paths: List[String],
    times: List[Float64],
) raises:
    if len(paths) != len(times):
        raise Error(
            "dump_pvd_collection: paths/times length mismatch ("
            + String(len(paths)) + " vs " + String(len(times)) + ")"
        )
    var pvd = String()
    pvd += '<?xml version="1.0"?>\n'
    pvd += ('<VTKFile type="Collection" version="0.1"'
            ' byte_order="LittleEndian">\n')
    pvd += '<Collection>\n'
    for i in range(len(paths)):
        pvd += '<DataSet timestep="'
        pvd += String(times[i])
        pvd += '" group="" part="0" file="'
        pvd += paths[i]
        pvd += '"/>\n'
    pvd += '</Collection>\n'
    pvd += '</VTKFile>\n'
    Path(pvd_path).write_text(pvd)
