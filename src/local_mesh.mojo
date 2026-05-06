# ======================================================================
# GPU-resident periodic Cartesian Kuhn tetrahedral mesh (internal)
# ======================================================================
#
# This module provides `LocalMesh`, the lower-level mesh builder used
# internally by `Mesh` (defined in src/mesh.mojo).  A `LocalMesh` is
# just a periodic Kuhn-tet mesh covering an (Nx, Ny, Nz) cube grid --
# it knows nothing about rank partitioning or ghost elements.
#
# User-facing code should import `Mesh` from `src.mesh` (which may
# wrap a LocalMesh with patch-aware metadata).  This module is exposed
# directly only so `src.mesh.Mesh.__init__` can compose it.
# ======================================================================
#
# Each (Nx x Ny x Nz) cube is split into 6 Kuhn tets sharing the main
# diagonal 0-7.  Each cube "owns" 12 faces:
#
#   face_type 0..5 : six internal diagonal faces within the cube
#   face_type 6,7  : two triangles on the +x cube boundary
#   face_type 8,9  : two triangles on the +y cube boundary
#   face_type 10,11: two triangles on the +z cube boundary
#
# A global face index is computed directly:
#
#   face_id = owner_cell_id * 12 + face_type
#
# The canonical face-node ordering follows the owner-cell cube-corner
# indices (ascending).  Because Kuhn tetrahedra are translation-
# invariant, the per-(tet, local_face, side) canonical-to-tet-node
# mapping and all the geometric Jacobians are constants across the
# whole mesh.  The mesh build therefore reduces to:
#
#   1. Compute a small set of tables on the host (< 1 KB total).
#   2. Upload those tables to the GPU.
#   3. Launch two kernels that populate the device-side arrays in
#      parallel, one thread per element and one thread per face.
#   4. Download just `elem_node_xyz` to the host for the VTU writer
#      (the writer's static mesh blob needs a host copy once).
#
# Everything the DG solver needs lives in DeviceBuffers owned by this
# struct -- no post-build host-to-device upload pass.
# ======================================================================

from src.reference import (
    N_F,
    num_tet_nodes,
    num_tri_nodes,
    ReferenceElement,
)
from src.boundary import BoundaryConditions, BC_INTERIOR
from std.math import sqrt, ceildiv
from std.gpu import global_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from std.memory import memcpy

comptime mesh_f = DType.float32
comptime mesh_i = DType.int32

comptime KUHN_TETS_PER_CELL = 6
comptime FACES_PER_CELL = 12  # 6 internal + 6 owned external (+x/+y/+z)

comptime MESH_BLOCK = 256

# ======================================================================
# Host-side helpers (fast, small tables)
# ======================================================================


def kuhn_vertex(t: Int, k: Int) -> Int:
    # Cube-corner index of each tet-local vertex (0..3).  Corner layout:
    #   0=(0,0,0) 1=(1,0,0) 2=(0,1,0) 3=(1,1,0)
    #   4=(0,0,1) 5=(1,0,1) 6=(0,1,1) 7=(1,1,1)
    if t == 0:
        if k == 0:
            return 0
        if k == 1:
            return 1
        if k == 2:
            return 3
        return 7
    if t == 1:
        if k == 0:
            return 0
        if k == 1:
            return 3
        if k == 2:
            return 2
        return 7
    if t == 2:
        if k == 0:
            return 0
        if k == 1:
            return 2
        if k == 2:
            return 6
        return 7
    if t == 3:
        if k == 0:
            return 0
        if k == 1:
            return 6
        if k == 2:
            return 4
        return 7
    if t == 4:
        if k == 0:
            return 0
        if k == 1:
            return 4
        if k == 2:
            return 5
        return 7
    if k == 0:
        return 0
    if k == 1:
        return 5
    if k == 2:
        return 1
    return 7


def corner_dx(c: Int) -> Int:
    if c == 1 or c == 3 or c == 5 or c == 7:
        return 1
    return 0


def corner_dy(c: Int) -> Int:
    if c == 2 or c == 3 or c == 6 or c == 7:
        return 1
    return 0


def corner_dz(c: Int) -> Int:
    if c >= 4:
        return 1
    return 0


def edge_mid_node(a: Int, b: Int) raises -> Int:
    # Element-local P2 node index for the edge between two tet-local
    # vertex indices (unordered pair).
    var x = a
    var y = b
    if x > y:
        var t = x
        x = y
        y = t
    if x == 0 and y == 1:
        return 4
    if x == 1 and y == 2:
        return 5
    if x == 0 and y == 2:
        return 6
    if x == 0 and y == 3:
        return 7
    if x == 1 and y == 3:
        return 8
    if x == 2 and y == 3:
        return 9
    raise Error("edge_mid_node: invalid edge")


@fieldwise_init
struct TetFaceInfo(ImplicitlyCopyable, Movable):
    var face_type: Int
    var di: Int
    var dj: Int
    var dk: Int
    var side: Int


def tet_face_info(t: Int, f: Int) -> TetFaceInfo:
    # (tet, local_face) -> (face_type, owner_offset, side).
    if t == 0:
        if f == 0:
            return TetFaceInfo(6, 0, 0, 0, 0)
        if f == 1:
            return TetFaceInfo(1, 0, 0, 0, 0)
        if f == 2:
            return TetFaceInfo(0, 0, 0, 0, 0)
        return TetFaceInfo(11, 0, 0, -1, 1)
    if t == 1:
        if f == 0:
            return TetFaceInfo(8, 0, 0, 0, 0)
        if f == 1:
            return TetFaceInfo(2, 0, 0, 0, 0)
        if f == 2:
            return TetFaceInfo(1, 0, 0, 0, 1)
        return TetFaceInfo(10, 0, 0, -1, 1)
    if t == 2:
        if f == 0:
            return TetFaceInfo(9, 0, 0, 0, 0)
        if f == 1:
            return TetFaceInfo(3, 0, 0, 0, 0)
        if f == 2:
            return TetFaceInfo(2, 0, 0, 0, 1)
        return TetFaceInfo(6, -1, 0, 0, 1)
    if t == 3:
        if f == 0:
            return TetFaceInfo(10, 0, 0, 0, 0)
        if f == 1:
            return TetFaceInfo(4, 0, 0, 0, 0)
        if f == 2:
            return TetFaceInfo(3, 0, 0, 0, 1)
        return TetFaceInfo(7, -1, 0, 0, 1)
    if t == 4:
        if f == 0:
            return TetFaceInfo(11, 0, 0, 0, 0)
        if f == 1:
            return TetFaceInfo(5, 0, 0, 0, 0)
        if f == 2:
            return TetFaceInfo(4, 0, 0, 0, 1)
        return TetFaceInfo(9, 0, -1, 0, 1)
    # t == 5
    if f == 0:
        return TetFaceInfo(7, 0, 0, 0, 0)
    if f == 1:
        return TetFaceInfo(0, 0, 0, 0, 1)
    if f == 2:
        return TetFaceInfo(5, 0, 0, 0, 1)
    return TetFaceInfo(8, 0, -1, 0, 1)


def face_type_corner(ft: Int, idx: Int) -> Int:
    # Owner-cell cube-corner index at canonical position (0..2).
    if ft == 0:
        if idx == 0:
            return 0
        if idx == 1:
            return 1
        return 7
    if ft == 1:
        if idx == 0:
            return 0
        if idx == 1:
            return 3
        return 7
    if ft == 2:
        if idx == 0:
            return 0
        if idx == 1:
            return 2
        return 7
    if ft == 3:
        if idx == 0:
            return 0
        if idx == 1:
            return 6
        return 7
    if ft == 4:
        if idx == 0:
            return 0
        if idx == 1:
            return 4
        return 7
    if ft == 5:
        if idx == 0:
            return 0
        if idx == 1:
            return 5
        return 7
    if ft == 6:
        if idx == 0:
            return 1
        if idx == 1:
            return 3
        return 7
    if ft == 7:
        if idx == 0:
            return 1
        if idx == 1:
            return 5
        return 7
    if ft == 8:
        if idx == 0:
            return 2
        if idx == 1:
            return 3
        return 7
    if ft == 9:
        if idx == 0:
            return 2
        if idx == 1:
            return 6
        return 7
    if ft == 10:
        if idx == 0:
            return 4
        if idx == 1:
            return 6
        return 7
    if idx == 0:
        return 4
    if idx == 1:
        return 5
    return 7


# ======================================================================
# Pre-computed tables on the host (feed into the two GPU kernels)
# ======================================================================
#
# Layout:
#   tet_invJ           [6 * 9]     Float32    per-tet-type inverse Jacobian
#   tet_inv_6V         [6]         Float32    per-tet-type 1/(6V)
#   tet_node_rel_xyz   [6 * 10*3]  Float32    P2 node positions in cell-unit coords
#   tet_face_type      [6 * 4]     Int32      face_type 0..11 per (tet, local_face)
#   tet_face_off       [6 * 4 * 3] Int32      owner-cell offset (di, dj, dk)
#   tet_face_side      [6 * 4]     Int32      0 or 1
#   tet_canon_to_ref   [6 * 4 * 6] Int32      per-(tet, face) canon -> reference perm
#   face_type_side0_tet[12]        Int32      tet index on side 0
#   face_type_side1_tet[12]        Int32      tet index on side 1
#   face_type_side1_off[12 * 3]    Int32      cell offset to side-1's cell
#   face_type_elem_node[12 * 2 * 6] Int32     per-side element-local node per canon
#   face_type_normal   [12 * 3]    Float32    unit outward normal (side 0 -> side 1)
#   face_type_area     [12]        Float32    face area

# ----------------------------------------------------------------------
# Order-P helpers: Lagrange node exponents and face-canonical
# barycentric coords.  Host-side only; called once per LocalMesh init.
# ----------------------------------------------------------------------


@fieldwise_init
struct _NodeExp(ImplicitlyCopyable, Movable):
    var a0: Int
    var a1: Int
    var a2: Int
    var a3: Int


def _lagrange_tet_exponents(P: Int) raises -> List[_NodeExp]:
    """Same ordering as `src.reference._tet_node_exponents`: vertex,
    edge, face-interior, volume-interior.  Duplicated here because
    that helper lives behind the ReferenceElement internals and we
    need a plain tuple here rather than a Monomial."""
    var out = List[_NodeExp]()
    # Vertex nodes.
    out.append(_NodeExp(P, 0, 0, 0))
    out.append(_NodeExp(0, P, 0, 0))
    out.append(_NodeExp(0, 0, P, 0))
    out.append(_NodeExp(0, 0, 0, P))
    if P < 2:
        return out^
    # Edges in VTK Lagrange tet order: (0,1), (1,2), (2,0), (0,3), (1,3), (2,3).
    var edges_a = List[Int]()
    edges_a.append(0)
    edges_a.append(1)
    edges_a.append(2)
    edges_a.append(0)
    edges_a.append(1)
    edges_a.append(2)
    var edges_b = List[Int]()
    edges_b.append(1)
    edges_b.append(2)
    edges_b.append(0)
    edges_b.append(3)
    edges_b.append(3)
    edges_b.append(3)
    for e in range(6):
        var a_idx = edges_a[e]
        var b_idx = edges_b[e]
        for k in range(1, P):
            var e0 = P - k if a_idx == 0 else (k if b_idx == 0 else 0)
            var e1 = P - k if a_idx == 1 else (k if b_idx == 1 else 0)
            var e2 = P - k if a_idx == 2 else (k if b_idx == 2 else 0)
            var e3 = P - k if a_idx == 3 else (k if b_idx == 3 else 0)
            out.append(_NodeExp(e0, e1, e2, e3))
    if P < 3:
        return out^
    # Face interior: a_f = 0, others >= 1, sum = P.
    for f in range(4):
        var ni = List[Int]()
        for i in range(4):
            if i != f:
                ni.append(i)
        for a0v in range(P - 2, 0, -1):
            for a1v in range(P - a0v - 1, 0, -1):
                var a2v = P - a0v - a1v
                var e0 = a0v if ni[0] == 0 else (
                    a1v if ni[1] == 0 else (a2v if ni[2] == 0 else 0)
                )
                var e1 = a0v if ni[0] == 1 else (
                    a1v if ni[1] == 1 else (a2v if ni[2] == 1 else 0)
                )
                var e2 = a0v if ni[0] == 2 else (
                    a1v if ni[1] == 2 else (a2v if ni[2] == 2 else 0)
                )
                var e3 = a0v if ni[0] == 3 else (
                    a1v if ni[1] == 3 else (a2v if ni[2] == 3 else 0)
                )
                out.append(_NodeExp(e0, e1, e2, e3))
    if P < 4:
        return out^
    # Volume interior: all a_i >= 1, sum = P.
    for a0v in range(P - 2, 0, -1):
        for a1v in range(P - a0v - 1, 0, -1):
            for a2v in range(P - a0v - a1v - 1, 0, -1):
                var a3v = P - a0v - a1v - a2v
                out.append(_NodeExp(a0v, a1v, a2v, a3v))
    return out^


def _find_tet_node(
    tet_nodes: List[_NodeExp],
    a0: Int,
    a1: Int,
    a2: Int,
    a3: Int,
) raises -> Int:
    """Linear scan for the tet-local node whose exponents match (a0..a3)."""
    for i in range(len(tet_nodes)):
        var n = tet_nodes[i]
        if n.a0 == a0 and n.a1 == a1 and n.a2 == a2 and n.a3 == a3:
            return i
    raise Error(
        "_find_tet_node: no match for ("
        + String(a0)
        + ","
        + String(a1)
        + ","
        + String(a2)
        + ","
        + String(a3)
        + ")"
    )


def _canon_face_barycentric(m: Int, P: Int) raises -> List[Int]:
    """For face-canonical index m at order P, return barycentric
    weights (b0, b1, b2) relative to the 3 canon face-vertices
    (summing to P).  Ordering matches `src.reference._face_to_element_node`:
    vertex 0, vertex 1, vertex 2, edge 01, edge 12, edge 20, face-interior."""
    var out = List[Int]()
    if m == 0:
        out.append(P)
        out.append(0)
        out.append(0)
        return out^
    if m == 1:
        out.append(0)
        out.append(P)
        out.append(0)
        return out^
    if m == 2:
        out.append(0)
        out.append(0)
        out.append(P)
        return out^
    # Edge interior, three edges of (P-1) each.
    var edge_len = P - 1
    if m < 3 + 3 * edge_len:
        var which = (m - 3) // edge_len
        var k = (m - 3) % edge_len + 1  # 1..P-1
        if which == 0:
            out.append(P - k)
            out.append(k)
            out.append(0)
        elif which == 1:
            out.append(0)
            out.append(P - k)
            out.append(k)
        else:
            # Edge 2: canon 2 -> canon 0
            out.append(k)
            out.append(0)
            out.append(P - k)
        return out^
    # Face interior: iterate (b0, b1, b2) >= 1, sum = P, lex-descending
    # on (b0, b1).  Matches the convention used in the reference module
    # for face_interior generation (with (b0, b1, b2) playing the role
    # of (a_ni0, a_ni1, a_ni2)).
    var idx = 3 + 3 * edge_len
    for b0 in range(P - 2, 0, -1):
        for b1 in range(P - b0 - 1, 0, -1):
            var b2 = P - b0 - b1
            if idx == m:
                out.append(b0)
                out.append(b1)
                out.append(b2)
                return out^
            idx += 1
    raise Error(
        "_canon_face_barycentric: index "
        + String(m)
        + " out of range for P="
        + String(P)
    )


def _reference_face_to_elem(P: Int) raises -> List[Int32]:
    """Raw (non-parametric) face-to-element-node table for order P.
    Dispatches a comptime-parameterised `ReferenceElement[P]`; we only
    need the face_to_elem list out so downstream callers can stay in a
    single non-parametric code path.  Supports P=1..5."""
    if P == 1:
        var r1 = ReferenceElement[1]()
        return r1.face_to_elem.copy()
    if P == 2:
        var r2 = ReferenceElement[2]()
        return r2.face_to_elem.copy()
    if P == 3:
        var r3 = ReferenceElement[3]()
        return r3.face_to_elem.copy()
    if P == 4:
        var r4 = ReferenceElement[4]()
        return r4.face_to_elem.copy()
    if P == 5:
        var r5 = ReferenceElement[5]()
        return r5.face_to_elem.copy()
    raise Error("unsupported Lagrange order P=" + String(P))


@fieldwise_init
struct _MeshTables(Movable):
    var tet_invJ: List[Float32]
    var tet_inv_6V: List[Float32]
    var tet_node_rel_xyz: List[Float32]
    var tet_face_type: List[Int32]
    var tet_face_off: List[Int32]
    var tet_face_side: List[Int32]
    var tet_canon_to_ref: List[Int32]
    var face_type_side0_tet: List[Int32]
    var face_type_side1_tet: List[Int32]
    var face_type_side1_off: List[Int32]
    var face_type_elem_node: List[Int32]
    var face_type_normal: List[Float32]
    var face_type_area: List[Float32]


def _compute_tables(
    dx: Float32,
    dy: Float32,
    dz: Float32,
    P: Int,
) raises -> _MeshTables:
    """Generic Kuhn-tet mesh table builder for order-P Lagrange.
    At P=2 produces the same tables as the old hand-coded P=2
    builder (bit-identical to within Float32 roundoff)."""

    var N_P_p = num_tet_nodes(P)
    var N_FP_p = num_tri_nodes(P)

    # Face-local (ref ordering) -> tet-local node index, flat [4*N_FP].
    # The reference element is parametric on P but we only need the
    # face_to_elem table here; pull it via a small runtime dispatch
    # rather than templating all of _compute_tables on P.
    var ref_face_node_map = _reference_face_to_elem(P)

    # For each tet-local node n, record its (a0,a1,a2,a3) barycentric
    # exponents (summing to P).  These identify face membership and
    # the (edge-param, face-interior) offset used when mapping from
    # canonical face-local indices to ref face-local indices.
    var tet_node_exp = _lagrange_tet_exponents(P)

    # ---- Per-tet-type geometry and reference node layout -----------
    var tet_invJ = _zeros_f32(6 * 9)
    var tet_inv_6V = _zeros_f32(6)
    var tet_node_rel_xyz = _zeros_f32(6 * N_P_p * 3)

    for t in range(6):
        # Cube-corner coords of each tet-local vertex (0..3).
        var vpx = List[Float32]()
        var vpy = List[Float32]()
        var vpz = List[Float32]()
        for k in range(4):
            var c = kuhn_vertex(t, k)
            vpx.append(Float32(corner_dx(c)))
            vpy.append(Float32(corner_dy(c)))
            vpz.append(Float32(corner_dz(c)))

        # Jacobian (physical): columns = (v1-v0)*scale, (v2-v0)*scale, (v3-v0)*scale
        var Jm = _zeros_f32(9)
        Jm[0 * 3 + 0] = (vpx[1] - vpx[0]) * dx
        Jm[0 * 3 + 1] = (vpx[2] - vpx[0]) * dx
        Jm[0 * 3 + 2] = (vpx[3] - vpx[0]) * dx
        Jm[1 * 3 + 0] = (vpy[1] - vpy[0]) * dy
        Jm[1 * 3 + 1] = (vpy[2] - vpy[0]) * dy
        Jm[1 * 3 + 2] = (vpy[3] - vpy[0]) * dy
        Jm[2 * 3 + 0] = (vpz[1] - vpz[0]) * dz
        Jm[2 * 3 + 1] = (vpz[2] - vpz[0]) * dz
        Jm[2 * 3 + 2] = (vpz[3] - vpz[0]) * dz

        var det = (
            Jm[0] * (Jm[4] * Jm[8] - Jm[5] * Jm[7])
            - Jm[1] * (Jm[3] * Jm[8] - Jm[5] * Jm[6])
            + Jm[2] * (Jm[3] * Jm[7] - Jm[4] * Jm[6])
        )
        if det <= 0.0:
            raise Error("non-positive Jacobian in tet template")
        var V = det / 6.0
        tet_inv_6V[t] = 1.0 / (6.0 * V)

        tet_invJ[t * 9 + 0] = (Jm[4] * Jm[8] - Jm[5] * Jm[7]) / det
        tet_invJ[t * 9 + 1] = (Jm[2] * Jm[7] - Jm[1] * Jm[8]) / det
        tet_invJ[t * 9 + 2] = (Jm[1] * Jm[5] - Jm[2] * Jm[4]) / det
        tet_invJ[t * 9 + 3] = (Jm[5] * Jm[6] - Jm[3] * Jm[8]) / det
        tet_invJ[t * 9 + 4] = (Jm[0] * Jm[8] - Jm[2] * Jm[6]) / det
        tet_invJ[t * 9 + 5] = (Jm[2] * Jm[3] - Jm[0] * Jm[5]) / det
        tet_invJ[t * 9 + 6] = (Jm[3] * Jm[7] - Jm[4] * Jm[6]) / det
        tet_invJ[t * 9 + 7] = (Jm[1] * Jm[6] - Jm[0] * Jm[7]) / det
        tet_invJ[t * 9 + 8] = (Jm[0] * Jm[4] - Jm[1] * Jm[3]) / det

        # Node positions in cell-unit coords: each tet-local Lagrange
        # node lives at the affine image of the reference-tet node
        # through the tet's own (v0, v1, v2, v3) physical vertices.
        # For node n with exponents (a0,a1,a2,a3), the barycentric
        # weights in tet-local space are (a_k/P) at tet-vertex k, and
        # the cell-unit position is sum_k (a_k/P) * v_k.
        var Pf = Float32(P)
        for n in range(N_P_p):
            var a0 = tet_node_exp[n].a0
            var a1 = tet_node_exp[n].a1
            var a2 = tet_node_exp[n].a2
            var a3 = tet_node_exp[n].a3
            var w0 = Float32(a0) / Pf
            var w1 = Float32(a1) / Pf
            var w2 = Float32(a2) / Pf
            var w3 = Float32(a3) / Pf
            tet_node_rel_xyz[(t * N_P_p + n) * 3 + 0] = (
                w0 * vpx[0] + w1 * vpx[1] + w2 * vpx[2] + w3 * vpx[3]
            )
            tet_node_rel_xyz[(t * N_P_p + n) * 3 + 1] = (
                w0 * vpy[0] + w1 * vpy[1] + w2 * vpy[2] + w3 * vpy[3]
            )
            tet_node_rel_xyz[(t * N_P_p + n) * 3 + 2] = (
                w0 * vpz[0] + w1 * vpz[1] + w2 * vpz[2] + w3 * vpz[3]
            )

    # ---- Per-(tet, local_face) tables ------------------------------
    var tet_face_type = _zeros_i32(6 * 4)
    var tet_face_off = _zeros_i32(6 * 4 * 3)
    var tet_face_side = _zeros_i32(6 * 4)
    var tet_canon_to_ref = _zeros_i32(6 * 4 * N_FP_p)
    # per-(tet, face, canon) -> element-local node index
    var tet_face_cnode_to_elemnode = _zeros_i32(6 * 4 * N_FP_p)

    for t in range(6):
        for f in range(4):
            var info = tet_face_info(t, f)
            var ft = info.face_type
            var di = info.di
            var dj = info.dj
            var dk = info.dk
            tet_face_type[t * 4 + f] = Int32(ft)
            tet_face_off[(t * 4 + f) * 3 + 0] = Int32(di)
            tet_face_off[(t * 4 + f) * 3 + 1] = Int32(dj)
            tet_face_off[(t * 4 + f) * 3 + 2] = Int32(dk)
            tet_face_side[t * 4 + f] = Int32(info.side)

            # Tet-local vertex indices on this face, ascending.
            var fv = List[Int]()
            for k in range(4):
                if k != f:
                    fv.append(k)
            var fv_cube = List[Int]()
            for kk in range(3):
                fv_cube.append(kuhn_vertex(t, fv[kk]))
            # Convert this cell's cube-corners to owner's (if owner offset != 0).
            var owner_cube = List[Int]()
            for kk in range(3):
                var cc = fv_cube[kk]
                if di == -1:
                    cc = cc ^ 1
                if dj == -1:
                    cc = cc ^ 2
                if dk == -1:
                    cc = cc ^ 4
                owner_cube.append(cc)

            var canon = List[Int]()
            canon.append(face_type_corner(ft, 0))
            canon.append(face_type_corner(ft, 1))
            canon.append(face_type_corner(ft, 2))

            # c2r_vertex[k]: reference fv index (0,1, or 2) matching canon[k].
            var c2r_vertex = List[Int]()
            for k in range(3):
                for ii in range(3):
                    if owner_cube[ii] == canon[k]:
                        c2r_vertex.append(ii)
                        break

            # Build the per-canon-face-local node maps algorithmically
            # from the barycentric coord of the node in face-canonical
            # coords (b0, b1, b2), mapping them through c2r_vertex to
            # tet-local exponents and looking up the matching node.
            for m in range(N_FP_p):
                # Canon face-local barycentric (b0, b1, b2) summing to P.
                var b = _canon_face_barycentric(m, P)
                # Map to tet-local face barycentric via the vertex
                # permutation: weight at fv[ii] = b_k where
                # c2r_vertex[k] = ii.
                var w_at_fv = List[Int]()
                w_at_fv.append(0)
                w_at_fv.append(0)
                w_at_fv.append(0)
                for k in range(3):
                    w_at_fv[c2r_vertex[k]] = b[k]
                # Tet-local node exponents: a_f = 0 on this face, and
                # a_{fv[ii]} = w_at_fv[ii] for the three face vertices.
                var a = List[Int]()
                a.append(0)
                a.append(0)
                a.append(0)
                a.append(0)
                for ii in range(3):
                    a[fv[ii]] = w_at_fv[ii]
                # a[f] stays 0.  Find the tet-local node with these
                # exponents.
                var tet_node = _find_tet_node(
                    tet_node_exp, a[0], a[1], a[2], a[3]
                )
                tet_face_cnode_to_elemnode[(t * 4 + f) * N_FP_p + m] = Int32(
                    tet_node
                )
                # And find the ref face-local index by matching the
                # tet_node against ref_face_node_map[f, ...].
                var ref_m = -1
                for mm in range(N_FP_p):
                    if Int(ref_face_node_map[f * N_FP_p + mm]) == tet_node:
                        ref_m = mm
                        break
                if ref_m < 0:
                    raise Error(
                        "canon-to-ref lookup failed for tet="
                        + String(t)
                        + " f="
                        + String(f)
                        + " canon m="
                        + String(m)
                    )
                tet_canon_to_ref[(t * 4 + f) * N_FP_p + m] = Int32(ref_m)

    # ---- Per-face-type tables --------------------------------------
    var face_type_side0_tet = _zeros_i32(12)
    var face_type_side1_tet = _zeros_i32(12)
    var face_type_side1_off = _zeros_i32(12 * 3)
    var face_type_side0_lf = _zeros_i32(12)
    var face_type_side1_lf = _zeros_i32(12)
    for t in range(6):
        for f in range(4):
            var info = tet_face_info(t, f)
            var ft = info.face_type
            if info.side == 0:
                face_type_side0_tet[ft] = Int32(t)
                face_type_side0_lf[ft] = Int32(f)
            else:
                face_type_side1_tet[ft] = Int32(t)
                face_type_side1_lf[ft] = Int32(f)
                face_type_side1_off[ft * 3 + 0] = Int32(-info.di)
                face_type_side1_off[ft * 3 + 1] = Int32(-info.dj)
                face_type_side1_off[ft * 3 + 2] = Int32(-info.dk)

    var face_type_normal = _zeros_f32(12 * 3)
    var face_type_area = _zeros_f32(12)
    for ft in range(12):
        var c0 = face_type_corner(ft, 0)
        var c1 = face_type_corner(ft, 1)
        var c2 = face_type_corner(ft, 2)
        var p0x = Float32(corner_dx(c0)) * dx
        var p0y = Float32(corner_dy(c0)) * dy
        var p0z = Float32(corner_dz(c0)) * dz
        var p1x = Float32(corner_dx(c1)) * dx
        var p1y = Float32(corner_dy(c1)) * dy
        var p1z = Float32(corner_dz(c1)) * dz
        var p2x = Float32(corner_dx(c2)) * dx
        var p2y = Float32(corner_dy(c2)) * dy
        var p2z = Float32(corner_dz(c2)) * dz
        var e01x = p1x - p0x
        var e01y = p1y - p0y
        var e01z = p1z - p0z
        var e02x = p2x - p0x
        var e02y = p2y - p0y
        var e02z = p2z - p0z
        var nx = e01y * e02z - e01z * e02y
        var ny = e01z * e02x - e01x * e02z
        var nz = e01x * e02y - e01y * e02x
        var nlen = sqrt(nx * nx + ny * ny + nz * nz)
        var area = Float32(0.5) * nlen
        nx = nx / nlen
        ny = ny / nlen
        nz = nz / nlen
        # Orient away from the side-0 tet's 4th vertex.
        var t0 = Int(face_type_side0_tet[ft])
        var f0 = Int(face_type_side0_lf[ft])
        var v4 = kuhn_vertex(t0, f0)
        var v4x = Float32(corner_dx(v4)) * dx
        var v4y = Float32(corner_dy(v4)) * dy
        var v4z = Float32(corner_dz(v4)) * dz
        var dot_n = (v4x - p0x) * nx + (v4y - p0y) * ny + (v4z - p0z) * nz
        if dot_n > 0.0:
            nx = -nx
            ny = -ny
            nz = -nz
        face_type_normal[ft * 3 + 0] = nx
        face_type_normal[ft * 3 + 1] = ny
        face_type_normal[ft * 3 + 2] = nz
        face_type_area[ft] = area

    # Per-(face_type, side, canon) element-local node.  Sized by the
    # per-order face-node count so higher-P meshes allocate the right
    # amount of scratch for the build kernel.
    var face_type_elem_node = _zeros_i32(12 * 2 * N_FP_p)
    for ft in range(12):
        var t0 = Int(face_type_side0_tet[ft])
        var f0 = Int(face_type_side0_lf[ft])
        var t1 = Int(face_type_side1_tet[ft])
        var f1 = Int(face_type_side1_lf[ft])
        for m in range(N_FP_p):
            face_type_elem_node[
                (ft * 2 + 0) * N_FP_p + m
            ] = tet_face_cnode_to_elemnode[(t0 * 4 + f0) * N_FP_p + m]
            face_type_elem_node[
                (ft * 2 + 1) * N_FP_p + m
            ] = tet_face_cnode_to_elemnode[(t1 * 4 + f1) * N_FP_p + m]

    return _MeshTables(
        tet_invJ^,
        tet_inv_6V^,
        tet_node_rel_xyz^,
        tet_face_type^,
        tet_face_off^,
        tet_face_side^,
        tet_canon_to_ref^,
        face_type_side0_tet^,
        face_type_side1_tet^,
        face_type_side1_off^,
        face_type_elem_node^,
        face_type_normal^,
        face_type_area^,
    )


# ======================================================================
# GPU kernels
# ======================================================================


def build_elements_kernel[
    NP: Int, NFP: Int
](
    # Outputs
    o_elem_node_xyz: UnsafePointer[Float32, MutAnyOrigin],  # [Ne * NP * 3]
    o_elem_invJ: UnsafePointer[Float32, MutAnyOrigin],  # [Ne * 9]
    o_elem_inv_6V: UnsafePointer[Float32, MutAnyOrigin],  # [Ne]
    o_elem_faces: UnsafePointer[Int32, MutAnyOrigin],  # [Ne * 4]
    o_elem_face_side: UnsafePointer[Int32, MutAnyOrigin],  # [Ne * 4]
    o_elem_canon_to_ref: UnsafePointer[Int32, MutAnyOrigin],  # [Ne * 4 * NFP]
    # Tables (small, uploaded once)
    tet_invJ: UnsafePointer[Float32, MutAnyOrigin],
    tet_inv_6V: UnsafePointer[Float32, MutAnyOrigin],
    tet_node_rel_xyz: UnsafePointer[Float32, MutAnyOrigin],
    tet_face_type: UnsafePointer[Int32, MutAnyOrigin],
    tet_face_off: UnsafePointer[Int32, MutAnyOrigin],
    tet_face_side: UnsafePointer[Int32, MutAnyOrigin],
    tet_canon_to_ref: UnsafePointer[Int32, MutAnyOrigin],
    # Dimensions
    Nx: Int,
    Ny: Int,
    Nz: Int,
    dx: Float32,
    dy: Float32,
    dz: Float32,
    num_elements: Int,
):
    var elem = Int(global_idx.x)
    if elem >= num_elements:
        return
    var tet_t = elem % KUHN_TETS_PER_CELL
    var cell = elem // KUHN_TETS_PER_CELL
    var k = cell // (Nx * Ny)
    var rem = cell - k * Nx * Ny
    var j = rem // Nx
    var i = rem - j * Nx

    var cell_ox = Float32(i) * dx
    var cell_oy = Float32(j) * dy
    var cell_oz = Float32(k) * dz

    # NP Lagrange-P node positions: cell origin + unit-coord rel * physical
    # cell size.  `tet_node_rel_xyz` is laid out [6 tets * NP nodes * 3].
    var node_base = tet_t * NP * 3
    for nn in range(NP):
        var rx = tet_node_rel_xyz[node_base + nn * 3 + 0]
        var ry = tet_node_rel_xyz[node_base + nn * 3 + 1]
        var rz = tet_node_rel_xyz[node_base + nn * 3 + 2]
        o_elem_node_xyz[(elem * NP + nn) * 3 + 0] = cell_ox + rx * dx
        o_elem_node_xyz[(elem * NP + nn) * 3 + 1] = cell_oy + ry * dy
        o_elem_node_xyz[(elem * NP + nn) * 3 + 2] = cell_oz + rz * dz

    for idx in range(9):
        o_elem_invJ[elem * 9 + idx] = tet_invJ[tet_t * 9 + idx]
    o_elem_inv_6V[elem] = tet_inv_6V[tet_t]

    for lf in range(N_F):
        var ft = Int(tet_face_type[tet_t * N_F + lf])
        var di = Int(tet_face_off[(tet_t * N_F + lf) * 3 + 0])
        var dj = Int(tet_face_off[(tet_t * N_F + lf) * 3 + 1])
        var dk = Int(tet_face_off[(tet_t * N_F + lf) * 3 + 2])
        var own_i = (i + di + Nx) % Nx
        var own_j = (j + dj + Ny) % Ny
        var own_k = (k + dk + Nz) % Nz
        var owner_cell = own_i + Nx * (own_j + Ny * own_k)
        o_elem_faces[elem * N_F + lf] = Int32(owner_cell * FACES_PER_CELL + ft)
        o_elem_face_side[elem * N_F + lf] = tet_face_side[tet_t * N_F + lf]
        for m in range(NFP):
            o_elem_canon_to_ref[(elem * N_F + lf) * NFP + m] = tet_canon_to_ref[
                (tet_t * N_F + lf) * NFP + m
            ]


def build_faces_kernel[
    NFP: Int
](
    # Outputs
    o_face_elem: UnsafePointer[Int32, MutAnyOrigin],  # [Nf * 2]
    o_face_elem_node: UnsafePointer[Int32, MutAnyOrigin],  # [Nf * 2 * NFP]
    o_face_normal: UnsafePointer[Float32, MutAnyOrigin],  # [Nf * 3]
    o_face_area: UnsafePointer[Float32, MutAnyOrigin],  # [Nf]
    # Tables
    face_type_side0_tet: UnsafePointer[Int32, MutAnyOrigin],
    face_type_side1_tet: UnsafePointer[Int32, MutAnyOrigin],
    face_type_side1_off: UnsafePointer[Int32, MutAnyOrigin],
    face_type_elem_node: UnsafePointer[Int32, MutAnyOrigin],
    face_type_normal: UnsafePointer[Float32, MutAnyOrigin],
    face_type_area: UnsafePointer[Float32, MutAnyOrigin],
    # Dimensions
    Nx: Int,
    Ny: Int,
    Nz: Int,
    num_faces: Int,
):
    var face = Int(global_idx.x)
    if face >= num_faces:
        return
    var ft = face % FACES_PER_CELL
    var cell = face // FACES_PER_CELL
    var k = cell // (Nx * Ny)
    var rem = cell - k * Nx * Ny
    var j = rem // Nx
    var i = rem - j * Nx

    var t0 = Int(face_type_side0_tet[ft])
    o_face_elem[face * 2 + 0] = Int32(cell * KUHN_TETS_PER_CELL + t0)

    var off_i = Int(face_type_side1_off[ft * 3 + 0])
    var off_j = Int(face_type_side1_off[ft * 3 + 1])
    var off_k = Int(face_type_side1_off[ft * 3 + 2])
    var s1_i = (i + off_i + Nx) % Nx
    var s1_j = (j + off_j + Ny) % Ny
    var s1_k = (k + off_k + Nz) % Nz
    var s1_cell = s1_i + Nx * (s1_j + Ny * s1_k)
    var t1 = Int(face_type_side1_tet[ft])
    o_face_elem[face * 2 + 1] = Int32(s1_cell * KUHN_TETS_PER_CELL + t1)

    for d in range(3):
        o_face_normal[face * 3 + d] = face_type_normal[ft * 3 + d]
    o_face_area[face] = face_type_area[ft]

    for m in range(NFP):
        o_face_elem_node[(face * 2 + 0) * NFP + m] = face_type_elem_node[
            (ft * 2 + 0) * NFP + m
        ]
        o_face_elem_node[(face * 2 + 1) * NFP + m] = face_type_elem_node[
            (ft * 2 + 1) * NFP + m
        ]


# ======================================================================
# Boundary-condition overlay kernels
# ======================================================================
#
# After the default periodic mesh build, these kernels flip specific
# faces to "boundary" status.  There are two flavours:
#
#  * `apply_plus_axis_bc_kernel`: for +x/+y/+z, the existing face
#    already has the interior element on side 0 (the cube at i=Nx-1
#    etc. owns this face in the periodic scheme).  We just set
#    face_elem[*, 1] = face_elem[*, 0] (safe no-op dereference for
#    the kernel's q_r_ptr even though the BC branch never uses it)
#    and stamp the face's bc_type.
#
#  * `apply_minus_axis_bc_kernel`: for -x/-y/-z, the existing periodic
#    face shared between cubes at i=Nx-1 and i=0 is already being
#    repurposed for the +x BC above, so the interior-at-i=0 view needs
#    its OWN face.  We allocate a fresh block of face IDs
#    [base_fid, base_fid + 2 * Ndir_a * Ndir_b) and populate each new
#    face as a one-sided mirror of the periodic face: side-0 element
#    is the boundary tet, face_normal is negated to point outward from
#    the interior, face_area is copied.  The boundary tet's
#    elem_faces[lf=3] entry is retargeted at its new face ID and
#    elem_face_side is forced to 0 (the interior is now canonically
#    side-0 of the new face).
#
# The (tet, face_type) picks come from tet_face_info(t, 3) for each
# of the 6 per-cube tets:
#   +x: tets 0, 5 at face types 6, 7 (their f=0 side-0 faces).
#   -x: tets 2, 3 at face types 6, 7 (their f=3 side-1 faces).
#   +y: tets 1, 2 at face types 8, 9.
#   -y: tets 5, 4 at face types 8, 9.
#   +z: tets 3, 4 at face types 10, 11.
#   -z: tets 1, 0 at face types 10, 11.
# The two face types per axis always enumerate the two triangles the
# axis-normal face splits into, so `face_offset` in [0, 2) picks one.
# ======================================================================


# `bnd_off` is the number of cube layers between the target boundary
# layer and the edge of the local mesh.  0 at single-patch (apply BC
# on the outermost cubes); 1 at multi-patch with a 1-cube ghost ring
# (apply BC on the first OWNED cube layer, skipping the ghost ring).


def apply_plus_x_bc_kernel(
    face_elem: UnsafePointer[Int32, MutAnyOrigin],
    face_bc_type: UnsafePointer[Int32, MutAnyOrigin],
    Nx: Int,
    Ny: Int,
    Nz: Int,
    bnd_off: Int,
    bc_value: Int32,
):
    # Two faces per (j, k) pair at cube (Nx-1-bnd_off, j, k), iterating
    # j in [bnd_off, Ny - bnd_off) and k in [bnd_off, Nz - bnd_off) so
    # we only touch the owned face ring (and skip the ghost ring at
    # multi-patch).
    var tid = Int(global_idx.x)
    var yw = Ny - 2 * bnd_off
    var zw = Nz - 2 * bnd_off
    var total = 2 * yw * zw
    if tid >= total:
        return
    var face_offset = tid % 2
    var jk = tid // 2
    var k_rel = jk % zw
    var j_rel = jk // zw
    var j = bnd_off + j_rel
    var k = bnd_off + k_rel
    var cube = (Nx - 1 - bnd_off) + Nx * (j + Ny * k)
    var ft = 6 + face_offset
    var fid = cube * FACES_PER_CELL + ft
    face_elem[fid * 2 + 1] = face_elem[fid * 2 + 0]
    face_bc_type[fid] = bc_value


def apply_plus_y_bc_kernel(
    face_elem: UnsafePointer[Int32, MutAnyOrigin],
    face_bc_type: UnsafePointer[Int32, MutAnyOrigin],
    Nx: Int,
    Ny: Int,
    Nz: Int,
    bnd_off: Int,
    bc_value: Int32,
):
    var tid = Int(global_idx.x)
    var xw = Nx - 2 * bnd_off
    var zw = Nz - 2 * bnd_off
    var total = 2 * xw * zw
    if tid >= total:
        return
    var face_offset = tid % 2
    var ik = tid // 2
    var k_rel = ik % zw
    var i_rel = ik // zw
    var i = bnd_off + i_rel
    var k = bnd_off + k_rel
    var cube = i + Nx * ((Ny - 1 - bnd_off) + Ny * k)
    var ft = 8 + face_offset
    var fid = cube * FACES_PER_CELL + ft
    face_elem[fid * 2 + 1] = face_elem[fid * 2 + 0]
    face_bc_type[fid] = bc_value


def apply_plus_z_bc_kernel(
    face_elem: UnsafePointer[Int32, MutAnyOrigin],
    face_bc_type: UnsafePointer[Int32, MutAnyOrigin],
    Nx: Int,
    Ny: Int,
    Nz: Int,
    bnd_off: Int,
    bc_value: Int32,
):
    var tid = Int(global_idx.x)
    var xw = Nx - 2 * bnd_off
    var yw = Ny - 2 * bnd_off
    var total = 2 * xw * yw
    if tid >= total:
        return
    var face_offset = tid % 2
    var ij = tid // 2
    var j_rel = ij % yw
    var i_rel = ij // yw
    var i = bnd_off + i_rel
    var j = bnd_off + j_rel
    var cube = i + Nx * (j + Ny * (Nz - 1 - bnd_off))
    var ft = 10 + face_offset
    var fid = cube * FACES_PER_CELL + ft
    face_elem[fid * 2 + 1] = face_elem[fid * 2 + 0]
    face_bc_type[fid] = bc_value


# For each -axis kernel we look up:
#   * the tet index on the boundary cube whose f=3 face is the boundary
#     face (from tet_face_info's offset == -1 entries),
#   * the face-type that tet shared with the +axis cube under periodic
#     wrap (so we can reuse the face_type_* tables for normal/area/
#     face-node pattern).
#
# -x at cube (0, j, k): tet 2 f=3 -> face_type 6, tet 3 f=3 -> face_type 7.
# -y at cube (i, 0, k): tet 5 f=3 -> face_type 8, tet 4 f=3 -> face_type 9.
# -z at cube (i, j, 0): tet 1 f=3 -> face_type 10, tet 0 f=3 -> face_type 11.


def apply_minus_x_bc_kernel[
    NFP: Int
](
    face_elem: UnsafePointer[Int32, MutAnyOrigin],
    face_elem_node: UnsafePointer[Int32, MutAnyOrigin],
    face_normal: UnsafePointer[Float32, MutAnyOrigin],
    face_area: UnsafePointer[Float32, MutAnyOrigin],
    face_bc_type: UnsafePointer[Int32, MutAnyOrigin],
    elem_faces: UnsafePointer[Int32, MutAnyOrigin],
    elem_face_side: UnsafePointer[Int32, MutAnyOrigin],
    ft_elem_node: UnsafePointer[Int32, MutAnyOrigin],  # [12 * 2 * NFP]
    ft_normal: UnsafePointer[Float32, MutAnyOrigin],  # [12 * 3]
    ft_area: UnsafePointer[Float32, MutAnyOrigin],  # [12]
    base_fid: Int,
    Nx: Int,
    Ny: Int,
    Nz: Int,
    bnd_off: Int,
    bc_value: Int32,
):
    var tid = Int(global_idx.x)
    var yw = Ny - 2 * bnd_off
    var zw = Nz - 2 * bnd_off
    var total = 2 * yw * zw
    if tid >= total:
        return
    var face_offset = tid % 2  # 0 -> tet 2 / ft 6,   1 -> tet 3 / ft 7
    var jk = tid // 2
    var k_rel = jk % zw
    var j_rel = jk // zw
    var j = bnd_off + j_rel
    var k = bnd_off + k_rel
    var cube = bnd_off + Nx * (j + Ny * k)
    var tet = 2 + face_offset
    var ft = 6 + face_offset
    var elem = cube * KUHN_TETS_PER_CELL + tet
    var new_fid = base_fid + tid

    face_elem[new_fid * 2 + 0] = Int32(elem)
    face_elem[new_fid * 2 + 1] = Int32(elem)  # safe no-op dereference
    for m in range(NFP):
        # Side-1 face-node pattern of the original face-type IS the
        # tet-f=3 pattern on the interior cube -- exactly what we want
        # for the interior on side 0 of the new face.
        var pat = ft_elem_node[(ft * 2 + 1) * NFP + m]
        face_elem_node[(new_fid * 2 + 0) * NFP + m] = pat
        face_elem_node[(new_fid * 2 + 1) * NFP + m] = pat
    # Normal flips sign: we want outward from the interior (-x), not
    # the periodic face's +x-pointing normal.
    face_normal[new_fid * 3 + 0] = -ft_normal[ft * 3 + 0]
    face_normal[new_fid * 3 + 1] = -ft_normal[ft * 3 + 1]
    face_normal[new_fid * 3 + 2] = -ft_normal[ft * 3 + 2]
    face_area[new_fid] = ft_area[ft]
    face_bc_type[new_fid] = bc_value

    # Retarget the interior tet's f=3 entry at the new face, side 0.
    elem_faces[elem * N_F + 3] = Int32(new_fid)
    elem_face_side[elem * N_F + 3] = Int32(0)


def apply_minus_y_bc_kernel[
    NFP: Int
](
    face_elem: UnsafePointer[Int32, MutAnyOrigin],
    face_elem_node: UnsafePointer[Int32, MutAnyOrigin],
    face_normal: UnsafePointer[Float32, MutAnyOrigin],
    face_area: UnsafePointer[Float32, MutAnyOrigin],
    face_bc_type: UnsafePointer[Int32, MutAnyOrigin],
    elem_faces: UnsafePointer[Int32, MutAnyOrigin],
    elem_face_side: UnsafePointer[Int32, MutAnyOrigin],
    ft_elem_node: UnsafePointer[Int32, MutAnyOrigin],
    ft_normal: UnsafePointer[Float32, MutAnyOrigin],
    ft_area: UnsafePointer[Float32, MutAnyOrigin],
    base_fid: Int,
    Nx: Int,
    Ny: Int,
    Nz: Int,
    bnd_off: Int,
    bc_value: Int32,
):
    var tid = Int(global_idx.x)
    var xw = Nx - 2 * bnd_off
    var zw = Nz - 2 * bnd_off
    var total = 2 * xw * zw
    if tid >= total:
        return
    var face_offset = tid % 2  # 0 -> tet 5 / ft 8,   1 -> tet 4 / ft 9
    var ik = tid // 2
    var k_rel = ik % zw
    var i_rel = ik // zw
    var i = bnd_off + i_rel
    var k = bnd_off + k_rel
    var cube = i + Nx * (bnd_off + Ny * k)
    var tet = 5 - face_offset
    var ft = 8 + face_offset
    var elem = cube * KUHN_TETS_PER_CELL + tet
    var new_fid = base_fid + tid

    face_elem[new_fid * 2 + 0] = Int32(elem)
    face_elem[new_fid * 2 + 1] = Int32(elem)
    for m in range(NFP):
        var pat = ft_elem_node[(ft * 2 + 1) * NFP + m]
        face_elem_node[(new_fid * 2 + 0) * NFP + m] = pat
        face_elem_node[(new_fid * 2 + 1) * NFP + m] = pat
    face_normal[new_fid * 3 + 0] = -ft_normal[ft * 3 + 0]
    face_normal[new_fid * 3 + 1] = -ft_normal[ft * 3 + 1]
    face_normal[new_fid * 3 + 2] = -ft_normal[ft * 3 + 2]
    face_area[new_fid] = ft_area[ft]
    face_bc_type[new_fid] = bc_value

    elem_faces[elem * N_F + 3] = Int32(new_fid)
    elem_face_side[elem * N_F + 3] = Int32(0)


def apply_minus_z_bc_kernel[
    NFP: Int
](
    face_elem: UnsafePointer[Int32, MutAnyOrigin],
    face_elem_node: UnsafePointer[Int32, MutAnyOrigin],
    face_normal: UnsafePointer[Float32, MutAnyOrigin],
    face_area: UnsafePointer[Float32, MutAnyOrigin],
    face_bc_type: UnsafePointer[Int32, MutAnyOrigin],
    elem_faces: UnsafePointer[Int32, MutAnyOrigin],
    elem_face_side: UnsafePointer[Int32, MutAnyOrigin],
    ft_elem_node: UnsafePointer[Int32, MutAnyOrigin],
    ft_normal: UnsafePointer[Float32, MutAnyOrigin],
    ft_area: UnsafePointer[Float32, MutAnyOrigin],
    base_fid: Int,
    Nx: Int,
    Ny: Int,
    Nz: Int,
    bnd_off: Int,
    bc_value: Int32,
):
    var tid = Int(global_idx.x)
    var xw = Nx - 2 * bnd_off
    var yw = Ny - 2 * bnd_off
    var total = 2 * xw * yw
    if tid >= total:
        return
    var face_offset = tid % 2  # 0 -> tet 1 / ft 10,  1 -> tet 0 / ft 11
    var ij = tid // 2
    var j_rel = ij % yw
    var i_rel = ij // yw
    var i = bnd_off + i_rel
    var j = bnd_off + j_rel
    var cube = i + Nx * (j + Ny * bnd_off)
    var tet = 1 - face_offset
    var ft = 10 + face_offset
    var elem = cube * KUHN_TETS_PER_CELL + tet
    var new_fid = base_fid + tid

    face_elem[new_fid * 2 + 0] = Int32(elem)
    face_elem[new_fid * 2 + 1] = Int32(elem)
    for m in range(NFP):
        var pat = ft_elem_node[(ft * 2 + 1) * NFP + m]
        face_elem_node[(new_fid * 2 + 0) * NFP + m] = pat
        face_elem_node[(new_fid * 2 + 1) * NFP + m] = pat
    face_normal[new_fid * 3 + 0] = -ft_normal[ft * 3 + 0]
    face_normal[new_fid * 3 + 1] = -ft_normal[ft * 3 + 1]
    face_normal[new_fid * 3 + 2] = -ft_normal[ft * 3 + 2]
    face_area[new_fid] = ft_area[ft]
    face_bc_type[new_fid] = bc_value

    elem_faces[elem * N_F + 3] = Int32(new_fid)
    elem_face_side[elem * N_F + 3] = Int32(0)


# ======================================================================
# LocalMesh struct -- owns device buffers; built entirely on the GPU.
# ======================================================================


struct LocalMesh[P: Int = 2](Movable):
    """Order-P Lagrange Kuhn-tet mesh.  The GPU build kernels are
    parameterized on `NP = num_tet_nodes(P)` and `NFP = num_tri_nodes(P)`
    as comptime template parameters, and the host-side table build is
    already P-agnostic.  P=2 remains the only order wired through the
    rest of the stack (VTU emits VTK_QUADRATIC_TETRA, drivers default
    `ReferenceElement()` to P=2); LocalMesh at higher P now builds a
    correct buffer layout for later consumers."""

    comptime NP = num_tet_nodes(Self.P)
    comptime NFP = num_tri_nodes(Self.P)

    var Nx: Int
    var Ny: Int
    var Nz: Int
    var Lx: Float64
    var Ly: Float64
    var Lz: Float64
    var num_elements: Int
    var num_faces: Int

    # Device-resident output buffers (all the data the solver needs).
    var d_elem_node_xyz: DeviceBuffer[mesh_f]
    var d_elem_invJ: DeviceBuffer[mesh_f]
    var d_elem_inv_6V: DeviceBuffer[mesh_f]
    var d_elem_faces: DeviceBuffer[mesh_i]
    var d_elem_face_side: DeviceBuffer[mesh_i]
    var d_elem_canon_to_ref: DeviceBuffer[mesh_i]
    var d_face_elem: DeviceBuffer[mesh_i]
    var d_face_elem_node: DeviceBuffer[mesh_i]
    var d_face_normal: DeviceBuffer[mesh_f]
    var d_face_area: DeviceBuffer[mesh_f]
    # Per-face BC kind (BC_INTERIOR == 0 for regular two-sided faces,
    # non-zero for a boundary face dispatched to `physics.boundary_flux`).
    # Populated by the `BoundaryConditions` overlay in `Mesh.__init__`;
    # zero-initialised here so a default-periodic build is a no-op.
    var d_face_bc_type: DeviceBuffer[mesh_i]

    # Host copy of elem_node_xyz for the VTU writer's static mesh blob.
    # Stored as a raw pointer + length rather than a List[Float32] so
    # that the 80 MB device->host download doesn't need a pinned
    # staging buffer (cuMemAllocHost is slow for large sizes) and we
    # also avoid the 6.6M-iteration List.append pre-sizing pass.
    var elem_node_xyz_f32_ptr: UnsafePointer[Float32, MutExternalOrigin]
    var elem_node_xyz_f32_len: Int

    def __init__(
        out self,
        mut ctx: DeviceContext,
        Nx: Int,
        Ny: Int,
        Nz: Int,
        Lx: Float64,
        Ly: Float64,
        Lz: Float64,
        bcs: BoundaryConditions,
        bnd_off: Int = 0,
    ) raises:
        self.Nx = Nx
        self.Ny = Ny
        self.Nz = Nz
        self.Lx = Lx
        self.Ly = Ly
        self.Lz = Lz
        self.num_elements = Nx * Ny * Nz * KUHN_TETS_PER_CELL
        var nf_periodic = Nx * Ny * Nz * FACES_PER_CELL

        # Each non-periodic "-" axis spawns a fresh block of face IDs
        # for the interior-on-side-0 mirror faces (see BC overlay doc
        # block above).  The "+" axis just rewrites existing face
        # entries in place, so it doesn't grow the face count.
        # With bnd_off > 0 (multi-patch with ghost ring) we only allocate
        # extras for the owned face ring, not the ghost ring.
        var xw_bc = Nx - 2 * bnd_off
        var yw_bc = Ny - 2 * bnd_off
        var zw_bc = Nz - 2 * bnd_off
        var extra_mx = 2 * yw_bc * zw_bc if bcs.bc_x_lo != BC_INTERIOR else 0
        var extra_my = 2 * xw_bc * zw_bc if bcs.bc_y_lo != BC_INTERIOR else 0
        var extra_mz = 2 * xw_bc * yw_bc if bcs.bc_z_lo != BC_INTERIOR else 0
        var nf_extra = extra_mx + extra_my + extra_mz
        self.num_faces = nf_periodic + nf_extra

        var ne = self.num_elements
        var nf = self.num_faces
        var dx = Float32(Lx / Float64(Nx))
        var dy = Float32(Ly / Float64(Ny))
        var dz = Float32(Lz / Float64(Nz))

        # 1. Compute tables (~1 KB total) on host.  Order-parametric --
        # all generated tables are sized by NP / NFP rather than the
        # hand-coded P=2 constants.
        var tables = _compute_tables(dx, dy, dz, Self.P)

        # 2. Allocate device buffers for outputs.
        self.d_elem_node_xyz = ctx.enqueue_create_buffer[mesh_f](
            ne * Self.NP * 3
        )
        self.d_elem_invJ = ctx.enqueue_create_buffer[mesh_f](ne * 9)
        self.d_elem_inv_6V = ctx.enqueue_create_buffer[mesh_f](ne)
        self.d_elem_faces = ctx.enqueue_create_buffer[mesh_i](ne * N_F)
        self.d_elem_face_side = ctx.enqueue_create_buffer[mesh_i](ne * N_F)
        self.d_elem_canon_to_ref = ctx.enqueue_create_buffer[mesh_i](
            ne * N_F * Self.NFP
        )
        self.d_face_elem = ctx.enqueue_create_buffer[mesh_i](nf * 2)
        self.d_face_elem_node = ctx.enqueue_create_buffer[mesh_i](
            nf * 2 * Self.NFP
        )
        self.d_face_normal = ctx.enqueue_create_buffer[mesh_f](nf * 3)
        self.d_face_area = ctx.enqueue_create_buffer[mesh_f](nf)
        # Zero-initialise the BC type buffer so every face is "interior"
        # by default; the BC overlay in Mesh flips specific face ids to
        # non-zero values once BoundaryConditions plumbing lands.
        self.d_face_bc_type = ctx.enqueue_create_buffer[mesh_i](nf)
        self.d_face_bc_type.enqueue_fill(Int32(0))

        # 3. Upload small tables to device.
        var d_tet_invJ = _upload_f32_small(ctx, tables.tet_invJ)
        var d_tet_inv_6V = _upload_f32_small(ctx, tables.tet_inv_6V)
        var d_tet_node_rel = _upload_f32_small(ctx, tables.tet_node_rel_xyz)
        var d_tet_face_type = _upload_i32_small(ctx, tables.tet_face_type)
        var d_tet_face_off = _upload_i32_small(ctx, tables.tet_face_off)
        var d_tet_face_side = _upload_i32_small(ctx, tables.tet_face_side)
        var d_tet_canon_to_ref = _upload_i32_small(ctx, tables.tet_canon_to_ref)
        var d_ft_side0_tet = _upload_i32_small(ctx, tables.face_type_side0_tet)
        var d_ft_side1_tet = _upload_i32_small(ctx, tables.face_type_side1_tet)
        var d_ft_side1_off = _upload_i32_small(ctx, tables.face_type_side1_off)
        var d_ft_elem_node = _upload_i32_small(ctx, tables.face_type_elem_node)
        var d_ft_normal = _upload_f32_small(ctx, tables.face_type_normal)
        var d_ft_area = _upload_f32_small(ctx, tables.face_type_area)

        # 4. Launch build_elements kernel.
        comptime _build_elems = build_elements_kernel[Self.NP, Self.NFP]
        ctx.enqueue_function[_build_elems, _build_elems](
            self.d_elem_node_xyz.unsafe_ptr(),
            self.d_elem_invJ.unsafe_ptr(),
            self.d_elem_inv_6V.unsafe_ptr(),
            self.d_elem_faces.unsafe_ptr(),
            self.d_elem_face_side.unsafe_ptr(),
            self.d_elem_canon_to_ref.unsafe_ptr(),
            d_tet_invJ.unsafe_ptr(),
            d_tet_inv_6V.unsafe_ptr(),
            d_tet_node_rel.unsafe_ptr(),
            d_tet_face_type.unsafe_ptr(),
            d_tet_face_off.unsafe_ptr(),
            d_tet_face_side.unsafe_ptr(),
            d_tet_canon_to_ref.unsafe_ptr(),
            Nx,
            Ny,
            Nz,
            dx,
            dy,
            dz,
            ne,
            grid_dim=ceildiv(ne, MESH_BLOCK),
            block_dim=MESH_BLOCK,
        )

        # 5. Launch build_faces kernel over the periodic face block;
        # any extra BC face ids get populated by the overlay below.
        comptime _build_faces = build_faces_kernel[Self.NFP]
        ctx.enqueue_function[_build_faces, _build_faces](
            self.d_face_elem.unsafe_ptr(),
            self.d_face_elem_node.unsafe_ptr(),
            self.d_face_normal.unsafe_ptr(),
            self.d_face_area.unsafe_ptr(),
            d_ft_side0_tet.unsafe_ptr(),
            d_ft_side1_tet.unsafe_ptr(),
            d_ft_side1_off.unsafe_ptr(),
            d_ft_elem_node.unsafe_ptr(),
            d_ft_normal.unsafe_ptr(),
            d_ft_area.unsafe_ptr(),
            Nx,
            Ny,
            Nz,
            nf_periodic,
            grid_dim=ceildiv(nf_periodic, MESH_BLOCK),
            block_dim=MESH_BLOCK,
        )

        # 5b. BC overlay.  "+" kernels rewrite existing periodic face
        # entries in place; "-" kernels populate the fresh face IDs
        # allocated above.  The base_fid pointers are assigned in the
        # same order as the extra_* accumulation so each kernel's block
        # is non-overlapping.
        if not bcs.all_periodic():
            var mx_base = nf_periodic
            var my_base = mx_base + extra_mx
            var mz_base = my_base + extra_my

            if bcs.bc_x_hi != BC_INTERIOR:
                var n_plus_x = 2 * yw_bc * zw_bc
                ctx.enqueue_function[
                    apply_plus_x_bc_kernel,
                    apply_plus_x_bc_kernel,
                ](
                    self.d_face_elem.unsafe_ptr(),
                    self.d_face_bc_type.unsafe_ptr(),
                    Nx,
                    Ny,
                    Nz,
                    bnd_off,
                    bcs.bc_x_hi,
                    grid_dim=ceildiv(n_plus_x, MESH_BLOCK),
                    block_dim=MESH_BLOCK,
                )
            if bcs.bc_y_hi != BC_INTERIOR:
                var n_plus_y = 2 * xw_bc * zw_bc
                ctx.enqueue_function[
                    apply_plus_y_bc_kernel,
                    apply_plus_y_bc_kernel,
                ](
                    self.d_face_elem.unsafe_ptr(),
                    self.d_face_bc_type.unsafe_ptr(),
                    Nx,
                    Ny,
                    Nz,
                    bnd_off,
                    bcs.bc_y_hi,
                    grid_dim=ceildiv(n_plus_y, MESH_BLOCK),
                    block_dim=MESH_BLOCK,
                )
            if bcs.bc_z_hi != BC_INTERIOR:
                var n_plus_z = 2 * xw_bc * yw_bc
                ctx.enqueue_function[
                    apply_plus_z_bc_kernel,
                    apply_plus_z_bc_kernel,
                ](
                    self.d_face_elem.unsafe_ptr(),
                    self.d_face_bc_type.unsafe_ptr(),
                    Nx,
                    Ny,
                    Nz,
                    bnd_off,
                    bcs.bc_z_hi,
                    grid_dim=ceildiv(n_plus_z, MESH_BLOCK),
                    block_dim=MESH_BLOCK,
                )

            if bcs.bc_x_lo != BC_INTERIOR:
                comptime _bc_mx = apply_minus_x_bc_kernel[Self.NFP]
                ctx.enqueue_function[_bc_mx, _bc_mx](
                    self.d_face_elem.unsafe_ptr(),
                    self.d_face_elem_node.unsafe_ptr(),
                    self.d_face_normal.unsafe_ptr(),
                    self.d_face_area.unsafe_ptr(),
                    self.d_face_bc_type.unsafe_ptr(),
                    self.d_elem_faces.unsafe_ptr(),
                    self.d_elem_face_side.unsafe_ptr(),
                    d_ft_elem_node.unsafe_ptr(),
                    d_ft_normal.unsafe_ptr(),
                    d_ft_area.unsafe_ptr(),
                    mx_base,
                    Nx,
                    Ny,
                    Nz,
                    bnd_off,
                    bcs.bc_x_lo,
                    grid_dim=ceildiv(extra_mx, MESH_BLOCK),
                    block_dim=MESH_BLOCK,
                )
            if bcs.bc_y_lo != BC_INTERIOR:
                comptime _bc_my = apply_minus_y_bc_kernel[Self.NFP]
                ctx.enqueue_function[_bc_my, _bc_my](
                    self.d_face_elem.unsafe_ptr(),
                    self.d_face_elem_node.unsafe_ptr(),
                    self.d_face_normal.unsafe_ptr(),
                    self.d_face_area.unsafe_ptr(),
                    self.d_face_bc_type.unsafe_ptr(),
                    self.d_elem_faces.unsafe_ptr(),
                    self.d_elem_face_side.unsafe_ptr(),
                    d_ft_elem_node.unsafe_ptr(),
                    d_ft_normal.unsafe_ptr(),
                    d_ft_area.unsafe_ptr(),
                    my_base,
                    Nx,
                    Ny,
                    Nz,
                    bnd_off,
                    bcs.bc_y_lo,
                    grid_dim=ceildiv(extra_my, MESH_BLOCK),
                    block_dim=MESH_BLOCK,
                )
            if bcs.bc_z_lo != BC_INTERIOR:
                comptime _bc_mz = apply_minus_z_bc_kernel[Self.NFP]
                ctx.enqueue_function[_bc_mz, _bc_mz](
                    self.d_face_elem.unsafe_ptr(),
                    self.d_face_elem_node.unsafe_ptr(),
                    self.d_face_normal.unsafe_ptr(),
                    self.d_face_area.unsafe_ptr(),
                    self.d_face_bc_type.unsafe_ptr(),
                    self.d_elem_faces.unsafe_ptr(),
                    self.d_elem_face_side.unsafe_ptr(),
                    d_ft_elem_node.unsafe_ptr(),
                    d_ft_normal.unsafe_ptr(),
                    d_ft_area.unsafe_ptr(),
                    mz_base,
                    Nx,
                    Ny,
                    Nz,
                    bnd_off,
                    bcs.bc_z_lo,
                    grid_dim=ceildiv(extra_mz, MESH_BLOCK),
                    block_dim=MESH_BLOCK,
                )

        # 6. Download elem_node_xyz straight into a plain (unpinned)
        # heap buffer.  The VTU writer takes an UnsafePointer, not a
        # List, so there's no List.append pre-sizing pass either.
        # Transferring into unpinned host memory is slightly slower per
        # byte than going through a pinned staging buffer, but
        # cuMemAllocHost of 80 MB is much slower still than the savings.
        var points_n = ne * Self.NP * 3
        self.elem_node_xyz_f32_len = points_n
        self.elem_node_xyz_f32_ptr = alloc[Float32](points_n)
        self.d_elem_node_xyz.enqueue_copy_to(self.elem_node_xyz_f32_ptr)
        ctx.synchronize()

        # Intermediate table device buffers go out of scope here and
        # are freed by their DeviceBuffer destructors.


# ======================================================================
# Allocation & upload helpers
# ======================================================================


def _zeros_f32(n: Int) raises -> List[Float32]:
    var out = List[Float32]()
    for _ in range(n):
        out.append(Float32(0.0))
    return out^


def _zeros_i32(n: Int) raises -> List[Int32]:
    var out = List[Int32]()
    for _ in range(n):
        out.append(Int32(0))
    return out^


def _upload_f32_small(
    mut ctx: DeviceContext, src: List[Float32]
) raises -> DeviceBuffer[mesh_f]:
    var n = len(src)
    var hbuf = ctx.enqueue_create_host_buffer[mesh_f](n)
    memcpy(dest=hbuf.unsafe_ptr(), src=src.unsafe_ptr(), count=n)
    var dbuf = ctx.enqueue_create_buffer[mesh_f](n)
    ctx.enqueue_copy(dbuf, hbuf)
    return dbuf^


def _upload_i32_small(
    mut ctx: DeviceContext, src: List[Int32]
) raises -> DeviceBuffer[mesh_i]:
    var n = len(src)
    var hbuf = ctx.enqueue_create_host_buffer[mesh_i](n)
    memcpy(dest=hbuf.unsafe_ptr(), src=src.unsafe_ptr(), count=n)
    var dbuf = ctx.enqueue_create_buffer[mesh_i](n)
    ctx.enqueue_copy(dbuf, hbuf)
    return dbuf^
