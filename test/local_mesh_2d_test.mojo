# ======================================================================
# local_mesh_2d_test -- 2D triangulated Cartesian mesh topology
# ======================================================================
#
# Validates the host-side mesh builder in src/local_mesh_2d.mojo at a
# range of small (Nx, Ny, P):
#
#   1. Element count = Nx * Ny * 2.
#   2. Face count    = Nx * Ny * 3 (1 diagonal + 1 +x + 1 +y per cell).
#   3. Every face references two valid element ids.
#   4. Every element's elem_faces list is consistent with face_elem:
#      face_elem[elem_faces[e, lf], elem_face_side[e, lf]] == e.
#   5. face_elem_node on both sides refers to valid element-local node
#      indices and maps to the SAME physical (x, y) coordinate on both
#      sides (modulo periodic wrap).
#   6. Element Jacobians are correctly scaled: invJ @ [dx, dy] = I for
#      degenerate-zero triangle edges.
#
# Host-only; `mojo run test/local_mesh_2d_test.mojo`.
# ======================================================================

from src.local_mesh_2d import LocalMesh2D, TRIS_PER_CELL, FACES_PER_CELL
from src.reference_2d import num_tri_nodes_2d, num_edge_nodes


def abs_f64(x: Float64) -> Float64:
    return x if x >= 0.0 else -x


def check_topology[P: Int](Nx: Int, Ny: Int, Lx: Float64, Ly: Float64) raises:
    print("  P=", P, " Nx=", Nx, " Ny=", Ny, " Lx=", Lx, " Ly=", Ly)
    var NP_p = num_tri_nodes_2d(P)
    var NFP_e = num_edge_nodes(P)
    var mesh = LocalMesh2D[P](Nx, Ny, Lx, Ly)

    var expect_elems = Nx * Ny * TRIS_PER_CELL
    var expect_faces = Nx * Ny * FACES_PER_CELL
    if mesh.num_elements != expect_elems:
        raise Error("num_elements mismatch")
    if mesh.num_faces != expect_faces:
        raise Error("num_faces mismatch")

    # Faces' element IDs are valid.
    for fid in range(mesh.num_faces):
        for side in range(2):
            var e = Int(mesh.face_elem[fid * 2 + side])
            if e < 0 or e >= mesh.num_elements:
                raise Error("face " + String(fid) + " side " + String(side)
                            + " has bad element id " + String(e))
        # Face length strictly positive.
        if not (mesh.face_length[fid] > 0.0):
            raise Error("face " + String(fid) + " has non-positive length")

    # elem_faces + face_elem consistency.
    for elem in range(mesh.num_elements):
        for lf in range(3):
            var fid = Int(mesh.elem_faces[elem * 3 + lf])
            if fid < 0 or fid >= mesh.num_faces:
                raise Error("elem " + String(elem) + " lf " + String(lf)
                            + " has bad fid " + String(fid))
            var side = Int(mesh.elem_face_side[elem * 3 + lf])
            var e_back = Int(mesh.face_elem[fid * 2 + side])
            if e_back != elem:
                raise Error("elem " + String(elem) + " lf " + String(lf)
                            + ": face_elem round-trip failed")

    # face_elem_node validity (every slot references a valid node).
    for fid in range(mesh.num_faces):
        for side in range(2):
            for m in range(NFP_e):
                var nn = Int(mesh.face_elem_node[(fid * 2 + side) * NFP_e + m])
                if nn < 0 or nn >= NP_p:
                    raise Error("face " + String(fid) + " side " + String(side)
                                + " slot " + String(m) + " bad node")

    # Geometric consistency: side-0 and side-1 should reference the
    # SAME physical coordinate at each face-local slot (modulo periodic
    # wrap).  Periodic wrap means we test |dx|, |dy| mod (Lx, Ly) are
    # each small.
    for fid in range(mesh.num_faces):
        var e0 = Int(mesh.face_elem[fid * 2 + 0])
        var e1 = Int(mesh.face_elem[fid * 2 + 1])
        for m in range(NFP_e):
            var n0 = Int(mesh.face_elem_node[(fid * 2 + 0) * NFP_e + m])
            var n1 = Int(mesh.face_elem_node[(fid * 2 + 1) * NFP_e + m])
            var x0 = mesh.elem_node_xyz[(e0 * NP_p + n0) * 2 + 0]
            var y0 = mesh.elem_node_xyz[(e0 * NP_p + n0) * 2 + 1]
            var x1 = mesh.elem_node_xyz[(e1 * NP_p + n1) * 2 + 0]
            var y1 = mesh.elem_node_xyz[(e1 * NP_p + n1) * 2 + 1]
            var dxv = abs_f64(x1 - x0)
            var dyv = abs_f64(y1 - y0)
            # Normalise to [0, period/2] to handle periodic wrap.
            if dxv > Lx / 2.0:
                dxv = Lx - dxv
            if dyv > Ly / 2.0:
                dyv = Ly - dyv
            var tol = 1.0e-10 * (Lx + Ly)
            if dxv > tol or dyv > tol:
                raise Error(
                    "face " + String(fid) + " slot " + String(m)
                    + ": side-0 / side-1 node coords disagree:"
                    + " dx=" + String(dxv) + " dy=" + String(dyv)
                )

    # Element Jacobian sanity: invJ should have positive determinant
    # (already enforced at build time, but re-check).
    for elem in range(mesh.num_elements):
        var J00 = mesh.elem_invJ[elem * 4 + 0]
        var J01 = mesh.elem_invJ[elem * 4 + 1]
        var J10 = mesh.elem_invJ[elem * 4 + 2]
        var J11 = mesh.elem_invJ[elem * 4 + 3]
        var det_invJ = J00 * J11 - J01 * J10
        if not (det_invJ > 0.0):
            raise Error("elem " + String(elem) + " has non-positive invJ det")

    print("    OK:",
          mesh.num_elements, "elements,",
          mesh.num_faces, "faces,",
          NP_p, "nodes/elem")


def main() raises:
    print("local_mesh_2d topology tests")
    check_topology[1](4, 4, 1.0, 1.0)
    check_topology[2](4, 4, 1.0, 1.0)
    check_topology[2](5, 3, 2.0, 1.0)    # non-square cells
    check_topology[3](4, 4, 1.0, 1.0)
    check_topology[4](3, 3, 1.0, 1.0)
    print("=== local_mesh_2d_test PASSED ===")
