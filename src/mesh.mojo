# ======================================================================
# GPU-resident periodic Cartesian Kuhn tetrahedral mesh
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

from reference import N_P, N_F, N_FP, N_D
from std.math import sqrt, ceildiv
from std.gpu import global_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from std.memory import memcpy

comptime mesh_f = DType.float32
comptime mesh_i = DType.int32

comptime KUHN_TETS_PER_CELL = 6
comptime FACES_PER_CELL = 12     # 6 internal + 6 owned external (+x/+y/+z)

comptime MESH_BLOCK = 256

# ======================================================================
# Host-side helpers (fast, small tables)
# ======================================================================

def kuhn_vertex(t: Int, k: Int) -> Int:
    # Cube-corner index of each tet-local vertex (0..3).  Corner layout:
    #   0=(0,0,0) 1=(1,0,0) 2=(0,1,0) 3=(1,1,0)
    #   4=(0,0,1) 5=(1,0,1) 6=(0,1,1) 7=(1,1,1)
    if t == 0:
        if k == 0: return 0
        if k == 1: return 1
        if k == 2: return 3
        return 7
    if t == 1:
        if k == 0: return 0
        if k == 1: return 3
        if k == 2: return 2
        return 7
    if t == 2:
        if k == 0: return 0
        if k == 1: return 2
        if k == 2: return 6
        return 7
    if t == 3:
        if k == 0: return 0
        if k == 1: return 6
        if k == 2: return 4
        return 7
    if t == 4:
        if k == 0: return 0
        if k == 1: return 4
        if k == 2: return 5
        return 7
    if k == 0: return 0
    if k == 1: return 5
    if k == 2: return 1
    return 7

def corner_dx(c: Int) -> Int:
    if c == 1 or c == 3 or c == 5 or c == 7: return 1
    return 0

def corner_dy(c: Int) -> Int:
    if c == 2 or c == 3 or c == 6 or c == 7: return 1
    return 0

def corner_dz(c: Int) -> Int:
    if c >= 4: return 1
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
    if x == 0 and y == 1: return 4
    if x == 1 and y == 2: return 5
    if x == 0 and y == 2: return 6
    if x == 0 and y == 3: return 7
    if x == 1 and y == 3: return 8
    if x == 2 and y == 3: return 9
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
        if f == 0: return TetFaceInfo(6, 0, 0, 0, 0)
        if f == 1: return TetFaceInfo(1, 0, 0, 0, 0)
        if f == 2: return TetFaceInfo(0, 0, 0, 0, 0)
        return TetFaceInfo(11, 0, 0, -1, 1)
    if t == 1:
        if f == 0: return TetFaceInfo(8, 0, 0, 0, 0)
        if f == 1: return TetFaceInfo(2, 0, 0, 0, 0)
        if f == 2: return TetFaceInfo(1, 0, 0, 0, 1)
        return TetFaceInfo(10, 0, 0, -1, 1)
    if t == 2:
        if f == 0: return TetFaceInfo(9, 0, 0, 0, 0)
        if f == 1: return TetFaceInfo(3, 0, 0, 0, 0)
        if f == 2: return TetFaceInfo(2, 0, 0, 0, 1)
        return TetFaceInfo(6, -1, 0, 0, 1)
    if t == 3:
        if f == 0: return TetFaceInfo(10, 0, 0, 0, 0)
        if f == 1: return TetFaceInfo(4, 0, 0, 0, 0)
        if f == 2: return TetFaceInfo(3, 0, 0, 0, 1)
        return TetFaceInfo(7, -1, 0, 0, 1)
    if t == 4:
        if f == 0: return TetFaceInfo(11, 0, 0, 0, 0)
        if f == 1: return TetFaceInfo(5, 0, 0, 0, 0)
        if f == 2: return TetFaceInfo(4, 0, 0, 0, 1)
        return TetFaceInfo(9, 0, -1, 0, 1)
    # t == 5
    if f == 0: return TetFaceInfo(7, 0, 0, 0, 0)
    if f == 1: return TetFaceInfo(0, 0, 0, 0, 1)
    if f == 2: return TetFaceInfo(5, 0, 0, 0, 1)
    return TetFaceInfo(8, 0, -1, 0, 1)

def face_type_corner(ft: Int, idx: Int) -> Int:
    # Owner-cell cube-corner index at canonical position (0..2).
    if ft == 0:
        if idx == 0: return 0
        if idx == 1: return 1
        return 7
    if ft == 1:
        if idx == 0: return 0
        if idx == 1: return 3
        return 7
    if ft == 2:
        if idx == 0: return 0
        if idx == 1: return 2
        return 7
    if ft == 3:
        if idx == 0: return 0
        if idx == 1: return 6
        return 7
    if ft == 4:
        if idx == 0: return 0
        if idx == 1: return 4
        return 7
    if ft == 5:
        if idx == 0: return 0
        if idx == 1: return 5
        return 7
    if ft == 6:
        if idx == 0: return 1
        if idx == 1: return 3
        return 7
    if ft == 7:
        if idx == 0: return 1
        if idx == 1: return 5
        return 7
    if ft == 8:
        if idx == 0: return 2
        if idx == 1: return 3
        return 7
    if ft == 9:
        if idx == 0: return 2
        if idx == 1: return 6
        return 7
    if ft == 10:
        if idx == 0: return 4
        if idx == 1: return 6
        return 7
    if idx == 0: return 4
    if idx == 1: return 5
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
    dx: Float32, dy: Float32, dz: Float32
) raises -> _MeshTables:
    # ---- Per-tet-type geometry and reference node layout -----------
    var tet_invJ = _zeros_f32(6 * 9)
    var tet_inv_6V = _zeros_f32(6)
    var tet_node_rel_xyz = _zeros_f32(6 * 10 * 3)

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
        Jm[0*3+0] = (vpx[1] - vpx[0]) * dx
        Jm[0*3+1] = (vpx[2] - vpx[0]) * dx
        Jm[0*3+2] = (vpx[3] - vpx[0]) * dx
        Jm[1*3+0] = (vpy[1] - vpy[0]) * dy
        Jm[1*3+1] = (vpy[2] - vpy[0]) * dy
        Jm[1*3+2] = (vpy[3] - vpy[0]) * dy
        Jm[2*3+0] = (vpz[1] - vpz[0]) * dz
        Jm[2*3+1] = (vpz[2] - vpz[0]) * dz
        Jm[2*3+2] = (vpz[3] - vpz[0]) * dz

        var det = (
              Jm[0]*(Jm[4]*Jm[8] - Jm[5]*Jm[7])
            - Jm[1]*(Jm[3]*Jm[8] - Jm[5]*Jm[6])
            + Jm[2]*(Jm[3]*Jm[7] - Jm[4]*Jm[6])
        )
        if det <= 0.0:
            raise Error("non-positive Jacobian in tet template")
        var V = det / 6.0
        tet_inv_6V[t] = 1.0 / (6.0 * V)

        tet_invJ[t*9 + 0] = (Jm[4]*Jm[8] - Jm[5]*Jm[7]) / det
        tet_invJ[t*9 + 1] = (Jm[2]*Jm[7] - Jm[1]*Jm[8]) / det
        tet_invJ[t*9 + 2] = (Jm[1]*Jm[5] - Jm[2]*Jm[4]) / det
        tet_invJ[t*9 + 3] = (Jm[5]*Jm[6] - Jm[3]*Jm[8]) / det
        tet_invJ[t*9 + 4] = (Jm[0]*Jm[8] - Jm[2]*Jm[6]) / det
        tet_invJ[t*9 + 5] = (Jm[2]*Jm[3] - Jm[0]*Jm[5]) / det
        tet_invJ[t*9 + 6] = (Jm[3]*Jm[7] - Jm[4]*Jm[6]) / det
        tet_invJ[t*9 + 7] = (Jm[1]*Jm[6] - Jm[0]*Jm[7]) / det
        tet_invJ[t*9 + 8] = (Jm[0]*Jm[4] - Jm[1]*Jm[3]) / det

        # P2 node positions in cell-unit coords (0..1 per axis).
        for nn in range(4):
            tet_node_rel_xyz[t*30 + nn*3 + 0] = vpx[nn]
            tet_node_rel_xyz[t*30 + nn*3 + 1] = vpy[nn]
            tet_node_rel_xyz[t*30 + nn*3 + 2] = vpz[nn]
        var ea = [0, 1, 0, 0, 1, 2]
        var eb = [1, 2, 2, 3, 3, 3]
        var en = [4, 5, 6, 7, 8, 9]
        for e in range(6):
            var a = ea[e]; var b = eb[e]; var n = en[e]
            tet_node_rel_xyz[t*30 + n*3 + 0] = 0.5 * (vpx[a] + vpx[b])
            tet_node_rel_xyz[t*30 + n*3 + 1] = 0.5 * (vpy[a] + vpy[b])
            tet_node_rel_xyz[t*30 + n*3 + 2] = 0.5 * (vpz[a] + vpz[b])

    # ---- Per-(tet, local_face) tables ------------------------------
    var tet_face_type = _zeros_i32(6 * 4)
    var tet_face_off = _zeros_i32(6 * 4 * 3)
    var tet_face_side = _zeros_i32(6 * 4)
    var tet_canon_to_ref = _zeros_i32(6 * 4 * 6)
    # per-(tet, face, canon) -> element-local P2 node index
    var tet_face_cnode_to_elemnode = _zeros_i32(6 * 4 * 6)

    for t in range(6):
        for f in range(4):
            var info = tet_face_info(t, f)
            var ft = info.face_type
            var di = info.di
            var dj = info.dj
            var dk = info.dk
            tet_face_type[t*4 + f] = Int32(ft)
            tet_face_off[(t*4 + f)*3 + 0] = Int32(di)
            tet_face_off[(t*4 + f)*3 + 1] = Int32(dj)
            tet_face_off[(t*4 + f)*3 + 2] = Int32(dk)
            tet_face_side[t*4 + f] = Int32(info.side)

            # Tet-local vertex indices on this face (fv ordering):
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

            # c2r_vertex[k]: reference fv index matching canon[k].
            var c2r_vertex = List[Int]()
            for k in range(3):
                for ii in range(3):
                    if owner_cube[ii] == canon[k]:
                        c2r_vertex.append(ii)
                        break
            for k in range(3):
                tet_canon_to_ref[(t*4 + f)*6 + k] = Int32(c2r_vertex[k])
                tet_face_cnode_to_elemnode[(t*4 + f)*6 + k] = Int32(
                    fv[c2r_vertex[k]]
                )
            for k in range(3):
                var a_r = c2r_vertex[k]
                var b_r = c2r_vertex[(k+1) % 3]
                var lo = a_r if a_r < b_r else b_r
                var hi = a_r if a_r >= b_r else b_r
                var ref_edge: Int32
                if lo == 0 and hi == 1: ref_edge = 3
                elif lo == 1 and hi == 2: ref_edge = 4
                else: ref_edge = 5
                tet_canon_to_ref[(t*4 + f)*6 + 3 + k] = ref_edge
                var tet_a = fv[a_r]
                var tet_b = fv[b_r]
                tet_face_cnode_to_elemnode[(t*4 + f)*6 + 3 + k] = Int32(
                    edge_mid_node(tet_a, tet_b)
                )

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
                face_type_side1_off[ft*3 + 0] = Int32(-info.di)
                face_type_side1_off[ft*3 + 1] = Int32(-info.dj)
                face_type_side1_off[ft*3 + 2] = Int32(-info.dk)

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
        var e01x = p1x - p0x; var e01y = p1y - p0y; var e01z = p1z - p0z
        var e02x = p2x - p0x; var e02y = p2y - p0y; var e02z = p2z - p0z
        var nx = e01y*e02z - e01z*e02y
        var ny = e01z*e02x - e01x*e02z
        var nz = e01x*e02y - e01y*e02x
        var nlen = sqrt(nx*nx + ny*ny + nz*nz)
        var area = Float32(0.5) * nlen
        nx = nx / nlen; ny = ny / nlen; nz = nz / nlen
        # Orient away from the side-0 tet's 4th vertex.
        var t0 = Int(face_type_side0_tet[ft])
        var f0 = Int(face_type_side0_lf[ft])
        var v4 = kuhn_vertex(t0, f0)
        var v4x = Float32(corner_dx(v4)) * dx
        var v4y = Float32(corner_dy(v4)) * dy
        var v4z = Float32(corner_dz(v4)) * dz
        var dot_n = (v4x - p0x)*nx + (v4y - p0y)*ny + (v4z - p0z)*nz
        if dot_n > 0.0:
            nx = -nx; ny = -ny; nz = -nz
        face_type_normal[ft*3 + 0] = nx
        face_type_normal[ft*3 + 1] = ny
        face_type_normal[ft*3 + 2] = nz
        face_type_area[ft] = area

    # Per-(face_type, side, canon) element-local node.
    var face_type_elem_node = _zeros_i32(12 * 2 * 6)
    for ft in range(12):
        var t0 = Int(face_type_side0_tet[ft])
        var f0 = Int(face_type_side0_lf[ft])
        var t1 = Int(face_type_side1_tet[ft])
        var f1 = Int(face_type_side1_lf[ft])
        for m in range(6):
            face_type_elem_node[(ft*2 + 0)*6 + m] = tet_face_cnode_to_elemnode[(t0*4 + f0)*6 + m]
            face_type_elem_node[(ft*2 + 1)*6 + m] = tet_face_cnode_to_elemnode[(t1*4 + f1)*6 + m]

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

def build_elements_kernel(
    # Outputs
    o_elem_node_xyz: UnsafePointer[Float32, MutAnyOrigin],   # [Ne * 30]
    o_elem_invJ: UnsafePointer[Float32, MutAnyOrigin],        # [Ne * 9]
    o_elem_inv_6V: UnsafePointer[Float32, MutAnyOrigin],      # [Ne]
    o_elem_faces: UnsafePointer[Int32, MutAnyOrigin],         # [Ne * 4]
    o_elem_face_side: UnsafePointer[Int32, MutAnyOrigin],     # [Ne * 4]
    o_elem_canon_to_ref: UnsafePointer[Int32, MutAnyOrigin],  # [Ne * 24]
    # Tables (small, uploaded once)
    tet_invJ: UnsafePointer[Float32, MutAnyOrigin],
    tet_inv_6V: UnsafePointer[Float32, MutAnyOrigin],
    tet_node_rel_xyz: UnsafePointer[Float32, MutAnyOrigin],
    tet_face_type: UnsafePointer[Int32, MutAnyOrigin],
    tet_face_off: UnsafePointer[Int32, MutAnyOrigin],
    tet_face_side: UnsafePointer[Int32, MutAnyOrigin],
    tet_canon_to_ref: UnsafePointer[Int32, MutAnyOrigin],
    # Dimensions
    Nx: Int, Ny: Int, Nz: Int,
    dx: Float32, dy: Float32, dz: Float32,
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

    # 10 P2 node positions: cell origin + unit-coord rel * physical cell size.
    var node_base = tet_t * 30
    for nn in range(N_P):
        var rx = tet_node_rel_xyz[node_base + nn*3 + 0]
        var ry = tet_node_rel_xyz[node_base + nn*3 + 1]
        var rz = tet_node_rel_xyz[node_base + nn*3 + 2]
        o_elem_node_xyz[(elem*N_P + nn)*3 + 0] = cell_ox + rx * dx
        o_elem_node_xyz[(elem*N_P + nn)*3 + 1] = cell_oy + ry * dy
        o_elem_node_xyz[(elem*N_P + nn)*3 + 2] = cell_oz + rz * dz

    for idx in range(9):
        o_elem_invJ[elem*9 + idx] = tet_invJ[tet_t*9 + idx]
    o_elem_inv_6V[elem] = tet_inv_6V[tet_t]

    for lf in range(N_F):
        var ft = Int(tet_face_type[tet_t*N_F + lf])
        var di = Int(tet_face_off[(tet_t*N_F + lf)*3 + 0])
        var dj = Int(tet_face_off[(tet_t*N_F + lf)*3 + 1])
        var dk = Int(tet_face_off[(tet_t*N_F + lf)*3 + 2])
        var own_i = (i + di + Nx) % Nx
        var own_j = (j + dj + Ny) % Ny
        var own_k = (k + dk + Nz) % Nz
        var owner_cell = own_i + Nx * (own_j + Ny * own_k)
        o_elem_faces[elem*N_F + lf] = Int32(owner_cell * FACES_PER_CELL + ft)
        o_elem_face_side[elem*N_F + lf] = tet_face_side[tet_t*N_F + lf]
        for m in range(N_FP):
            o_elem_canon_to_ref[
                (elem*N_F + lf)*N_FP + m
            ] = tet_canon_to_ref[(tet_t*N_F + lf)*N_FP + m]


def build_faces_kernel(
    # Outputs
    o_face_elem: UnsafePointer[Int32, MutAnyOrigin],          # [Nf * 2]
    o_face_elem_node: UnsafePointer[Int32, MutAnyOrigin],     # [Nf * 12]
    o_face_normal: UnsafePointer[Float32, MutAnyOrigin],      # [Nf * 3]
    o_face_area: UnsafePointer[Float32, MutAnyOrigin],        # [Nf]
    # Tables
    face_type_side0_tet: UnsafePointer[Int32, MutAnyOrigin],
    face_type_side1_tet: UnsafePointer[Int32, MutAnyOrigin],
    face_type_side1_off: UnsafePointer[Int32, MutAnyOrigin],
    face_type_elem_node: UnsafePointer[Int32, MutAnyOrigin],
    face_type_normal: UnsafePointer[Float32, MutAnyOrigin],
    face_type_area: UnsafePointer[Float32, MutAnyOrigin],
    # Dimensions
    Nx: Int, Ny: Int, Nz: Int,
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
    o_face_elem[face*2 + 0] = Int32(cell * KUHN_TETS_PER_CELL + t0)

    var off_i = Int(face_type_side1_off[ft*3 + 0])
    var off_j = Int(face_type_side1_off[ft*3 + 1])
    var off_k = Int(face_type_side1_off[ft*3 + 2])
    var s1_i = (i + off_i + Nx) % Nx
    var s1_j = (j + off_j + Ny) % Ny
    var s1_k = (k + off_k + Nz) % Nz
    var s1_cell = s1_i + Nx * (s1_j + Ny * s1_k)
    var t1 = Int(face_type_side1_tet[ft])
    o_face_elem[face*2 + 1] = Int32(s1_cell * KUHN_TETS_PER_CELL + t1)

    for d in range(3):
        o_face_normal[face*3 + d] = face_type_normal[ft*3 + d]
    o_face_area[face] = face_type_area[ft]

    for m in range(N_FP):
        o_face_elem_node[(face*2 + 0)*N_FP + m] = face_type_elem_node[(ft*2 + 0)*N_FP + m]
        o_face_elem_node[(face*2 + 1)*N_FP + m] = face_type_elem_node[(ft*2 + 1)*N_FP + m]


# ======================================================================
# Mesh struct -- owns device buffers; built entirely on the GPU.
# ======================================================================

struct Mesh(Movable):
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
        Nx: Int, Ny: Int, Nz: Int,
        Lx: Float64, Ly: Float64, Lz: Float64,
    ) raises:
        self.Nx = Nx
        self.Ny = Ny
        self.Nz = Nz
        self.Lx = Lx
        self.Ly = Ly
        self.Lz = Lz
        self.num_elements = Nx * Ny * Nz * KUHN_TETS_PER_CELL
        self.num_faces = Nx * Ny * Nz * FACES_PER_CELL

        var ne = self.num_elements
        var nf = self.num_faces
        var dx = Float32(Lx / Float64(Nx))
        var dy = Float32(Ly / Float64(Ny))
        var dz = Float32(Lz / Float64(Nz))

        # 1. Compute tables (~1 KB total) on host.
        var tables = _compute_tables(dx, dy, dz)

        # 2. Allocate device buffers for outputs.
        self.d_elem_node_xyz = ctx.enqueue_create_buffer[mesh_f](ne * N_P * 3)
        self.d_elem_invJ = ctx.enqueue_create_buffer[mesh_f](ne * 9)
        self.d_elem_inv_6V = ctx.enqueue_create_buffer[mesh_f](ne)
        self.d_elem_faces = ctx.enqueue_create_buffer[mesh_i](ne * N_F)
        self.d_elem_face_side = ctx.enqueue_create_buffer[mesh_i](ne * N_F)
        self.d_elem_canon_to_ref = ctx.enqueue_create_buffer[mesh_i](ne * N_F * N_FP)
        self.d_face_elem = ctx.enqueue_create_buffer[mesh_i](nf * 2)
        self.d_face_elem_node = ctx.enqueue_create_buffer[mesh_i](nf * 2 * N_FP)
        self.d_face_normal = ctx.enqueue_create_buffer[mesh_f](nf * 3)
        self.d_face_area = ctx.enqueue_create_buffer[mesh_f](nf)

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
        ctx.enqueue_function[build_elements_kernel, build_elements_kernel](
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
            Nx, Ny, Nz, dx, dy, dz, ne,
            grid_dim=ceildiv(ne, MESH_BLOCK),
            block_dim=MESH_BLOCK,
        )

        # 5. Launch build_faces kernel.
        ctx.enqueue_function[build_faces_kernel, build_faces_kernel](
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
            Nx, Ny, Nz, nf,
            grid_dim=ceildiv(nf, MESH_BLOCK),
            block_dim=MESH_BLOCK,
        )

        # 6. Download elem_node_xyz straight into a plain (unpinned)
        # heap buffer.  The VTU writer takes an UnsafePointer, not a
        # List, so there's no List.append pre-sizing pass either.
        # Transferring into unpinned host memory is slightly slower per
        # byte than going through a pinned staging buffer, but
        # cuMemAllocHost of 80 MB is much slower still than the savings.
        var points_n = ne * N_P * 3
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
