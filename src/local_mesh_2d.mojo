# ======================================================================
# 2D triangular Cartesian-split mesh (host-side tables)
# ======================================================================
#
# Splits an Nx x Ny Cartesian grid into 2*Nx*Ny triangles using the
# /-diagonal from (i, j) to (i+1, j+1):
#
#     Triangle 0 (LOWER-RIGHT): vertices (i,j), (i+1,j),   (i+1,j+1)
#     Triangle 1 (UPPER-LEFT):  vertices (i,j), (i+1,j+1), (i,j+1)
#
# Periodic boundaries are the default.  Face count per cell:
#   1 internal face (the diagonal) + 1 +x face + 1 +y face = 3
# so TOTAL_FACES = 3 * Nx * Ny on a periodic grid.
#
# This module is host-only Float64 -- it builds NumPy-scale tables on
# the host that get uploaded to the device (in Float32) by
# `LocalMesh2DGpu` in `src/local_mesh_2d_gpu.mojo`.  Keeping the host
# tables in Float64 lets the topology-construction math be exact; the
# per-physics 2D GPU kernels then run in Float32 to match the 3D
# stack's register / shared-mem budget.
#
# Table layouts (all flat 1D lists):
#
#   elem_node_rel_xyz : [num_elements * NP * 2]  Float64, (r, s) per node
#   elem_invJ         : [num_elements * 4]       Float64, 2x2 per element
#   elem_inv_2A       : [num_elements]           Float64 (1 / (2 * triangle area))
#   elem_faces        : [num_elements * 3]       Int32 (face ids for 3 edges)
#   elem_face_side    : [num_elements * 3]       Int32 (0 or 1; this elem's
#                                                  side of the face)
#   elem_canon_to_ref : [num_elements * 3 * (P+1)] Int32 (face-local slot -> ref-
#                                                  element edge slot)
#   face_elem         : [num_faces * 2]          Int32
#   face_elem_node    : [num_faces * 2 * (P+1)]  Int32 (elem-local node for
#                                                  each face-local slot)
#   face_normal       : [num_faces * 2]          Float64 (unit outward normal
#                                                  in the plane; side 0's view)
#   face_length       : [num_faces]              Float64
# ======================================================================

from src.reference_2d import (
    ReferenceElement2D,
    num_tri_nodes_2d,
    num_edge_nodes,
)
from src.boundary import (
    BoundaryConditions2D,
    BC_INTERIOR,
    BC_WALL,
    BC_OUTFLOW,
)
from std.math import sqrt


comptime TRIS_PER_CELL = 2
comptime FACES_PER_CELL = 3  # 1 diagonal + 1 +x + 1 +y


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _zeros_f64(n: Int) raises -> List[Float64]:
    var out = List[Float64]()
    for _ in range(n):
        out.append(Float64(0.0))
    return out^


def _zeros_i32(n: Int) raises -> List[Int32]:
    var out = List[Int32]()
    for _ in range(n):
        out.append(Int32(0))
    return out^


# ----------------------------------------------------------------------
# Mesh tables container
# ----------------------------------------------------------------------


struct LocalMesh2D[P: Int = 2](Movable):
    """Order-P Lagrange triangle mesh on a Cartesian grid with /-diagonal
    split.  Host-side only -- all tables live in Float64/Int32 lists."""

    comptime NP = num_tri_nodes_2d(Self.P)
    comptime NFP_edge = num_edge_nodes(Self.P)

    var Nx: Int
    var Ny: Int
    var Lx: Float64
    var Ly: Float64
    var dx: Float64
    var dy: Float64
    var num_elements: Int
    var num_faces: Int

    # Per-element tables (see module docstring for shapes).
    var elem_node_xyz: List[Float64]
    var elem_invJ: List[Float64]
    var elem_inv_2A: List[Float64]
    var elem_faces: List[Int32]
    var elem_face_side: List[Int32]
    var elem_canon_to_ref: List[Int32]

    # Per-face tables.
    var face_elem: List[Int32]
    var face_elem_node: List[Int32]
    var face_normal: List[Float64]
    var face_length: List[Float64]
    # BC kind per face (BC_INTERIOR = 0 for all interior / periodic-wrap
    # faces; non-zero for boundary faces dispatched to physics.boundary_flux).
    var face_bc_type: List[Int32]

    def __init__(
        out self,
        Nx: Int,
        Ny: Int,
        Lx: Float64,
        Ly: Float64,
        bcs: BoundaryConditions2D = BoundaryConditions2D.periodic(),
    ) raises:
        self.Nx = Nx
        self.Ny = Ny
        self.Lx = Lx
        self.Ly = Ly
        self.dx = Lx / Float64(Nx)
        self.dy = Ly / Float64(Ny)
        self.num_elements = Nx * Ny * TRIS_PER_CELL
        var nf_periodic = Nx * Ny * FACES_PER_CELL
        # Each non-periodic "-" axis needs an extra face per boundary
        # cell (the +axis face is already allocated in the periodic
        # ring; we just override its bc_type).
        var extra_mx = Ny if bcs.bc_x_lo != BC_INTERIOR else 0
        var extra_my = Nx if bcs.bc_y_lo != BC_INTERIOR else 0
        self.num_faces = nf_periodic + extra_mx + extra_my

        var NP_p = Self.NP
        var NFP_e = Self.NFP_edge

        # Reference element for node positions + edge-to-element map.
        var re = ReferenceElement2D[Self.P]()

        # 1. Per-triangle node positions in physical coords.
        # Triangle 0 (LR): vertices v0=(i,j), v1=(i+1,j), v2=(i+1,j+1).
        # Triangle 1 (UL): vertices v0=(i,j), v1=(i+1,j+1), v2=(i,j+1).
        #
        # A node with barycentric (l0, l1, l2) sits at
        #   pos = l0 * v0 + l1 * v1 + l2 * v2
        # where (l0, l1, l2) = (1 - r - s, r, s).
        self.elem_node_xyz = _zeros_f64(self.num_elements * NP_p * 2)
        self.elem_invJ = _zeros_f64(self.num_elements * 4)
        self.elem_inv_2A = _zeros_f64(self.num_elements)
        for j in range(Ny):
            for i in range(Nx):
                var base_x = Float64(i) * self.dx
                var base_y = Float64(j) * self.dy
                for t in range(TRIS_PER_CELL):
                    var elem = (j * Nx + i) * TRIS_PER_CELL + t
                    var v0x: Float64
                    var v0y: Float64
                    var v1x: Float64
                    var v1y: Float64
                    var v2x: Float64
                    var v2y: Float64
                    if t == 0:
                        v0x = base_x
                        v0y = base_y
                        v1x = base_x + self.dx
                        v1y = base_y
                        v2x = base_x + self.dx
                        v2y = base_y + self.dy
                    else:
                        v0x = base_x
                        v0y = base_y
                        v1x = base_x + self.dx
                        v1y = base_y + self.dy
                        v2x = base_x
                        v2y = base_y + self.dy

                    # Node positions: interpolate in barycentric coords.
                    for nn in range(NP_p):
                        var r = re.node_pos[nn * 2 + 0]
                        var s = re.node_pos[nn * 2 + 1]
                        var l0 = 1.0 - r - s
                        var px = l0 * v0x + r * v1x + s * v2x
                        var py = l0 * v0y + r * v1y + s * v2y
                        self.elem_node_xyz[(elem * NP_p + nn) * 2 + 0] = px
                        self.elem_node_xyz[(elem * NP_p + nn) * 2 + 1] = py

                    # Jacobian of physical -> reference mapping.
                    # Physical x = v0 + r (v1 - v0) + s (v2 - v0), so
                    #   dx/dr = v1 - v0,  dx/ds = v2 - v0
                    #   J = [[dx/dr, dx/ds], [dy/dr, dy/ds]]
                    var J00 = v1x - v0x
                    var J01 = v2x - v0x
                    var J10 = v1y - v0y
                    var J11 = v2y - v0y
                    var detJ = J00 * J11 - J01 * J10
                    if detJ <= 0.0:
                        raise Error(
                            "LocalMesh2D: triangle has non-positive area"
                        )
                    var inv_det = 1.0 / detJ
                    # J^-1 = (1/detJ) * [[J11, -J01], [-J10, J00]]
                    self.elem_invJ[elem * 4 + 0] = J11 * inv_det
                    self.elem_invJ[elem * 4 + 1] = -J01 * inv_det
                    self.elem_invJ[elem * 4 + 2] = -J10 * inv_det
                    self.elem_invJ[elem * 4 + 3] = J00 * inv_det
                    # 1 / (2 * area) = 1 / detJ because detJ = 2 * area for
                    # the reference simplex (area 1/2).
                    self.elem_inv_2A[elem] = inv_det

        # 2. Face topology.
        # Per cell we own 3 faces, numbered by "face type":
        #   ft 0 = diagonal (between T0 and T1 within this cell)
        #   ft 1 = +x edge  (T0's edge 1; between this cell and (i+1, j))
        #   ft 2 = +y edge  (T1's edge 1; between this cell and (i, j+1))
        # The face-type indices of triangle edges (ref_face_2d below)
        # map tri-local edge 0, 1, 2 to the face-type + which cell owns it.
        self.face_elem = _zeros_i32(self.num_faces * 2)
        self.face_elem_node = _zeros_i32(self.num_faces * 2 * NFP_e)
        self.face_normal = _zeros_f64(self.num_faces * 2)
        self.face_length = _zeros_f64(self.num_faces)
        self.face_bc_type = _zeros_i32(self.num_faces)
        self.elem_faces = _zeros_i32(self.num_elements * 3)
        self.elem_face_side = _zeros_i32(self.num_elements * 3)
        self.elem_canon_to_ref = _zeros_i32(self.num_elements * 3 * NFP_e)

        # Face metadata per face-type.  For each (ft, side):
        #   which triangle (0 or 1) owns this side of the face?
        #   which tri-local edge is it?
        #   what (di, dj) offset from the cell to find side-1 owner cell?
        # Tri-local edges:
        #   T0 edges: e0 = v0->v1 (bottom),  e1 = v1->v2 (+x),  e2 = v2->v0 (diagonal)
        #   T1 edges: e0 = v0->v1 (diagonal),e1 = v1->v2 (+y),  e2 = v2->v0 (left)
        # So:
        #   ft 0 (diagonal): side 0 = T0 e2, side 1 = T1 e0, same cell, di/dj = 0/0.
        #   ft 1 (+x):       side 0 = T0 e1, side 1 = T1 e2 of (i+1, j), di = +1.
        #   ft 2 (+y):       side 0 = T1 e1, side 1 = T0 e0 of (i, j+1), dj = +1.
        var ft_side0_tri = List[Int]()
        ft_side0_tri.append(0)
        ft_side0_tri.append(0)
        ft_side0_tri.append(1)
        var ft_side0_edge = List[Int]()
        ft_side0_edge.append(2)
        ft_side0_edge.append(1)
        ft_side0_edge.append(1)
        var ft_side1_tri = List[Int]()
        ft_side1_tri.append(1)
        ft_side1_tri.append(1)
        ft_side1_tri.append(0)
        var ft_side1_edge = List[Int]()
        ft_side1_edge.append(0)
        ft_side1_edge.append(2)
        ft_side1_edge.append(0)
        var ft_di = List[Int]()
        ft_di.append(0)
        ft_di.append(1)
        ft_di.append(0)
        var ft_dj = List[Int]()
        ft_dj.append(0)
        ft_dj.append(0)
        ft_dj.append(1)

        # Face normal + length for each face type, computed from the
        # unit cell (dx, dy) with diagonal length sqrt(dx^2 + dy^2).
        # ft 0 (diagonal): normal points from T0 toward T1, i.e. in
        # direction (-dy, +dx) normalised; length = sqrt(dx^2 + dy^2).
        # ft 1 (+x): normal (+1, 0), length = dy.
        # ft 2 (+y): normal (0, +1), length = dx.
        var diag_len = sqrt(self.dx * self.dx + self.dy * self.dy)
        var ft_nx_list = List[Float64]()
        ft_nx_list.append(-self.dy / diag_len)
        ft_nx_list.append(1.0)
        ft_nx_list.append(0.0)
        var ft_ny_list = List[Float64]()
        ft_ny_list.append(self.dx / diag_len)
        ft_ny_list.append(0.0)
        ft_ny_list.append(1.0)
        var ft_len_list = List[Float64]()
        ft_len_list.append(diag_len)
        ft_len_list.append(self.dy)
        ft_len_list.append(self.dx)

        for j in range(Ny):
            for i in range(Nx):
                var cell = j * Nx + i
                for ft in range(FACES_PER_CELL):
                    var fid = cell * FACES_PER_CELL + ft
                    # Side 0 element = this cell's side-0 triangle.
                    var t0 = ft_side0_tri[ft]
                    var e0_owner = cell * TRIS_PER_CELL + t0
                    # Side 1 element = neighbour cell (wrapped) + side-1
                    # triangle there.
                    var ni = (i + ft_di[ft] + Nx) % Nx
                    var nj = (j + ft_dj[ft] + Ny) % Ny
                    var nbr_cell = nj * Nx + ni
                    var t1 = ft_side1_tri[ft]
                    var e1_owner = nbr_cell * TRIS_PER_CELL + t1
                    self.face_elem[fid * 2 + 0] = Int32(e0_owner)
                    self.face_elem[fid * 2 + 1] = Int32(e1_owner)
                    self.face_normal[fid * 2 + 0] = ft_nx_list[ft]
                    self.face_normal[fid * 2 + 1] = ft_ny_list[ft]
                    self.face_length[fid] = ft_len_list[ft]

                    # Register this face on both neighbouring elements
                    # (side 0 and side 1) using the tri-local edge they
                    # own.
                    var e0_lf = ft_side0_edge[ft]
                    var e1_lf = ft_side1_edge[ft]
                    self.elem_faces[e0_owner * 3 + e0_lf] = Int32(fid)
                    self.elem_face_side[e0_owner * 3 + e0_lf] = Int32(0)
                    self.elem_faces[e1_owner * 3 + e1_lf] = Int32(fid)
                    self.elem_face_side[e1_owner * 3 + e1_lf] = Int32(1)

                    # face_elem_node + canon_to_ref: for each face-local
                    # slot m in [0, NFP_e), which element-local node does
                    # it map to on each side?  The ReferenceElement2D
                    # `edge_to_elem` map does exactly this for each
                    # (tri-local edge, edge-local slot) pair.
                    for m in range(NFP_e):
                        # Side 0: node = re.edge_to_elem[e0_lf * NFP_e + m]
                        var n0 = Int(re.edge_to_elem[e0_lf * NFP_e + m])
                        self.face_elem_node[(fid * 2 + 0) * NFP_e + m] = Int32(
                            n0
                        )
                        # Side 1: same edge but walked in reverse, because
                        # the two elements share this face with opposite
                        # orientation.  Side 1's edge-local index P-m
                        # maps to this slot.
                        var n1 = Int(
                            re.edge_to_elem[e1_lf * NFP_e + (NFP_e - 1 - m)]
                        )
                        self.face_elem_node[(fid * 2 + 1) * NFP_e + m] = Int32(
                            n1
                        )
                        # canon_to_ref: the canonical (side-0-ordering)
                        # slot m maps to ref-edge slot m on side 0 and
                        # (P-m) on side 1, in the tri-local edge index
                        # e0_lf / e1_lf respectively.  Caller consumes
                        # this when indexing Lift_ref[e, i, ref_slot].
                        self.elem_canon_to_ref[
                            (e0_owner * 3 + e0_lf) * NFP_e + m
                        ] = Int32(m)
                        self.elem_canon_to_ref[
                            (e1_owner * 3 + e1_lf) * NFP_e + m
                        ] = Int32(NFP_e - 1 - m)

        # --- BC overlay ------------------------------------------------
        # For non-periodic axes we stamp a bc_type onto the existing
        # outer-ring faces (+x, +y) and allocate fresh faces for the
        # -x / -y boundaries (whose interior triangles would otherwise
        # point at the wrap-around neighbour in the periodic mesh).
        # The "+" overlay is easy: the existing face's side 0 already
        # points at the outgoing boundary triangle (cell Nx-1 for +x,
        # cell Ny-1 for +y); we just set bc_type so the solver
        # dispatches to physics.boundary_flux instead of numerical_flux.
        if bcs.bc_x_hi != BC_INTERIOR:
            for j in range(Ny):
                var cell = j * Nx + (Nx - 1)
                var fid = cell * FACES_PER_CELL + 1  # ft 1 = +x
                # Safe no-op dereference of side 1 (points at cell 0 T1;
                # the solver's boundary_flux ignores q_r).
                self.face_elem[fid * 2 + 1] = self.face_elem[fid * 2 + 0]
                self.face_bc_type[fid] = bcs.bc_x_hi
        if bcs.bc_y_hi != BC_INTERIOR:
            for i in range(Nx):
                var cell = (Ny - 1) * Nx + i
                var fid = cell * FACES_PER_CELL + 2  # ft 2 = +y
                self.face_elem[fid * 2 + 1] = self.face_elem[fid * 2 + 0]
                self.face_bc_type[fid] = bcs.bc_y_hi

        # For the "-" overlay we allocate fresh face IDs after the
        # periodic ring.  For each boundary cell's T1 (-x side) or T0
        # (-y side) we create a new face whose side 0 is that triangle,
        # normal negated to point outward from the interior, and
        # re-target the triangle's edge-2 / edge-0 slot at the new
        # face.  This mirrors the 3D apply_minus_{x,y,z}_bc_kernel
        # overlay in src/local_mesh.mojo.
        if bcs.bc_x_lo != BC_INTERIOR:
            var mx_base = nf_periodic
            for j in range(Ny):
                var cell = j * Nx + 0
                var elem = cell * TRIS_PER_CELL + 1  # T1 owns -x edge
                var new_fid = mx_base + j
                var lf = 2  # T1's edge 2 = -x
                self.face_elem[new_fid * 2 + 0] = Int32(elem)
                self.face_elem[new_fid * 2 + 1] = Int32(elem)
                # Normal points outward from the interior: (-1, 0) for
                # -x.  face_length same as +x.
                self.face_normal[new_fid * 2 + 0] = -1.0
                self.face_normal[new_fid * 2 + 1] = 0.0
                self.face_length[new_fid] = self.dy
                self.face_bc_type[new_fid] = bcs.bc_x_lo
                # Retarget T1's edge-2 slot at the new face, side 0.
                self.elem_faces[elem * 3 + lf] = Int32(new_fid)
                self.elem_face_side[elem * 3 + lf] = Int32(0)
                for m in range(NFP_e):
                    var nn = Int(re.edge_to_elem[lf * NFP_e + m])
                    self.face_elem_node[(new_fid * 2 + 0) * NFP_e + m] = Int32(
                        nn
                    )
                    self.face_elem_node[(new_fid * 2 + 1) * NFP_e + m] = Int32(
                        nn
                    )
                    self.elem_canon_to_ref[(elem * 3 + lf) * NFP_e + m] = Int32(
                        m
                    )
        if bcs.bc_y_lo != BC_INTERIOR:
            var my_base = nf_periodic + extra_mx
            for i in range(Nx):
                var cell = 0 * Nx + i
                var elem = cell * TRIS_PER_CELL + 0  # T0 owns -y edge
                var new_fid = my_base + i
                var lf = 0  # T0's edge 0 = -y
                self.face_elem[new_fid * 2 + 0] = Int32(elem)
                self.face_elem[new_fid * 2 + 1] = Int32(elem)
                self.face_normal[new_fid * 2 + 0] = 0.0
                self.face_normal[new_fid * 2 + 1] = -1.0
                self.face_length[new_fid] = self.dx
                self.face_bc_type[new_fid] = bcs.bc_y_lo
                self.elem_faces[elem * 3 + lf] = Int32(new_fid)
                self.elem_face_side[elem * 3 + lf] = Int32(0)
                for m in range(NFP_e):
                    var nn = Int(re.edge_to_elem[lf * NFP_e + m])
                    self.face_elem_node[(new_fid * 2 + 0) * NFP_e + m] = Int32(
                        nn
                    )
                    self.face_elem_node[(new_fid * 2 + 1) * NFP_e + m] = Int32(
                        nn
                    )
                    self.elem_canon_to_ref[(elem * 3 + lf) * NFP_e + m] = Int32(
                        m
                    )
