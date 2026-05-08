# ======================================================================
# Tetrahedral reference element, equispaced Lagrange, order P
# ======================================================================
#
# Generalises the P2 hand-coded reference element to arbitrary order P
# with equispaced Lagrange nodes on the reference simplex -- the same
# node pattern used by WARPXM and by VTK_LAGRANGE_TETRAHEDRON (cell
# type 71).  Node count scales as
#
#   N_P(P)  = (P+1)(P+2)(P+3)/6        (tetrahedron: 1, 4, 10, 20, 35, ...)
#   N_FP(P) = (P+1)(P+2)/2              (triangle:    1, 3,  6, 10, 15, ...)
#
# Reference tet: vertices (0,0,0), (1,0,0), (0,1,0), (0,0,1), barycentric
# coords (l0, l1, l2, l3) with l0 = 1 - r - s - t.
#
# Node ordering matches VTK Lagrange tet (back-compatible with the
# existing P=2 VTK_QUADRATIC_TETRA layout):
#
#   1. Four vertex nodes: (P,0,0,0), (0,P,0,0), (0,0,P,0), (0,0,0,P).
#   2. (P-1) nodes per edge, for 6 edges in the canonical VTK order
#      (0-1, 1-2, 2-0, 0-3, 1-3, 2-3) from first vertex toward second.
#   3. (P-1)(P-2)/2 nodes per face, for 4 faces (opp v0..v3).
#   4. (P-1)(P-2)(P-3)/6 interior-of-volume nodes.
#
# Each basis function phi_i is the Lagrange nodal polynomial on this
# node set.  We represent it as a sum of homogeneous degree-P monomials
#   sum_m c_m  l0^{a0} l1^{a1} l2^{a2} l3^{a3},    a0+a1+a2+a3 = P
# and solve for the coefficients via Vandermonde inversion.  Analytic
# integration over the reference tet uses the Dirichlet formula
#   integral l0^a l1^b l2^c l3^d dV = a! b! c! d! / (a+b+c+d+3)!
# (and the 2D analogue for the reference triangle's mass matrix, used
# to build the face-lift operator Lift_ref).
#
# The same machinery produces triangle-side Lagrange operators for
# building the face-lift operator at matching order.
#
# Backward compatibility: module-level constants N_P = 10, N_F = 4,
# N_FP = 6, N_D = 3 still refer to the P=2 case so every current
# consumer of this module keeps working.  `ReferenceElement()` with no
# argument is an alias for `ReferenceElement[2]()`.
# ======================================================================

from std.math import sqrt
from src.nvtx import NvtxContext


# --- Static (P=2) constants, for modules that haven't been
# parameterised by P yet. ----------------------------------------------
comptime N_P = 10
comptime N_F = 4
comptime N_FP = 6
comptime N_D = 3


# ----------------------------------------------------------------------
# Compile-time size helpers
# ----------------------------------------------------------------------


def num_tet_nodes(P: Int) -> Int:
    return (P + 1) * (P + 2) * (P + 3) // 6


def num_tri_nodes(P: Int) -> Int:
    return (P + 1) * (P + 2) // 2


# ----------------------------------------------------------------------
# Monomial (generic multi-index with a Float64 coefficient).  Used both
# for the list-of-monomials representation of polynomials and for
# labelling Lagrange nodes (in which case the `c` field is unused).
# ----------------------------------------------------------------------


@fieldwise_init
struct Monomial(ImplicitlyCopyable, Movable):
    var a0: Int
    var a1: Int
    var a2: Int
    var a3: Int
    var c: Float64


def poly_mul(a: List[Monomial], b: List[Monomial]) raises -> List[Monomial]:
    var out = List[Monomial]()
    for ai in range(len(a)):
        for bi in range(len(b)):
            var m = Monomial(a[ai].a0 + b[bi].a0, a[ai].a1 + b[bi].a1, a[ai].a2 + b[bi].a2, a[ai].a3 + b[bi].a3, a[ai].c * b[bi].c)
            out.append(m)
    return out^


def factorial(n: Int) raises -> Float64:
    var f: Float64 = 1.0
    for k in range(2, n + 1):
        f *= Float64(k)
    return f


# Exact integral of prod l_i^{a_i} over the reference tet (volume 1/6).
def integrate_ref_tet(p: List[Monomial]) raises -> Float64:
    var s: Float64 = 0.0
    for i in range(len(p)):
        var m = p[i]
        var denom_n = m.a0 + m.a1 + m.a2 + m.a3 + 3
        var num = factorial(m.a0) * factorial(m.a1) * factorial(m.a2) * factorial(m.a3)
        s += m.c * num / factorial(denom_n)
    return s


# Exact integral over the reference triangle (unit simplex, area 1/2).
# Uses the Monomial's a0, a1, a2 fields only (a3 ignored).
def integrate_ref_tri(p: List[Monomial]) raises -> Float64:
    var s: Float64 = 0.0
    for i in range(len(p)):
        var m = p[i]
        var denom_n = m.a0 + m.a1 + m.a2 + 2
        var num = factorial(m.a0) * factorial(m.a1) * factorial(m.a2)
        s += m.c * num / factorial(denom_n)
    return s


# ----------------------------------------------------------------------
# Node ordering (canonical: vertex -> edge -> face -> volume)
# ----------------------------------------------------------------------
#
# Each tet node is labelled by (a0, a1, a2, a3) with sum = P.  Barycentric
# coordinates are (l_i = a_i / P).  Physical reference coords are
# (r, s, t) = (l1, l2, l3).
#
# Edges in VTK Lagrange order: (0,1), (1,2), (2,0), (0,3), (1,3), (2,3).
# Faces opposite vertex f, with the OTHER three vertices kept in
# ascending order: (1,2,3), (0,2,3), (0,1,3), (0,1,2).


def _tet_node_exponents(P: Int) raises -> List[Monomial]:
    """Canonical-ordered Lagrange tet nodes at order P.  The returned
    `Monomial` objects carry the (a0, a1, a2, a3) exponents; the `c`
    field is unused (set to 0)."""
    # VTK Lagrange tet edge order: (0,1), (1,2), (2,0), (0,3), (1,3), (2,3).
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

    var out = List[Monomial]()
    # Vertex nodes.
    out.append(Monomial(P, 0, 0, 0, 0.0))
    out.append(Monomial(0, P, 0, 0, 0.0))
    out.append(Monomial(0, 0, P, 0, 0.0))
    out.append(Monomial(0, 0, 0, P, 0.0))
    if P < 2:
        return out^
    # Edge interior nodes.
    for e in range(6):
        var a_idx = edges_a[e]
        var b_idx = edges_b[e]
        for k in range(1, P):
            var exp0 = P - k if a_idx == 0 else (k if b_idx == 0 else 0)
            var exp1 = P - k if a_idx == 1 else (k if b_idx == 1 else 0)
            var exp2 = P - k if a_idx == 2 else (k if b_idx == 2 else 0)
            var exp3 = P - k if a_idx == 3 else (k if b_idx == 3 else 0)
            out.append(Monomial(exp0, exp1, exp2, exp3, 0.0))
    if P < 3:
        return out^
    # Face interior nodes, for each face opp v_f.
    for f in range(4):
        var ni = List[Int]()
        for i in range(4):
            if i != f:
                ni.append(i)
        # Iterate (a_ni0, a_ni1, a_ni2) with sum=P, each >= 1,
        # lex-descending on (a_ni0, a_ni1).
        for a0v in range(P - 2, 0, -1):
            for a1v in range(P - a0v - 1, 0, -1):
                var a2v = P - a0v - a1v
                var exp0 = a0v if ni[0] == 0 else (a1v if ni[1] == 0 else (a2v if ni[2] == 0 else 0))
                var exp1 = a0v if ni[0] == 1 else (a1v if ni[1] == 1 else (a2v if ni[2] == 1 else 0))
                var exp2 = a0v if ni[0] == 2 else (a1v if ni[1] == 2 else (a2v if ni[2] == 2 else 0))
                var exp3 = a0v if ni[0] == 3 else (a1v if ni[1] == 3 else (a2v if ni[2] == 3 else 0))
                out.append(Monomial(exp0, exp1, exp2, exp3, 0.0))
    if P < 4:
        return out^
    # Volume interior nodes.
    for a0v in range(P - 2, 0, -1):
        for a1v in range(P - a0v - 1, 0, -1):
            for a2v in range(P - a0v - a1v - 1, 0, -1):
                var a3v = P - a0v - a1v - a2v
                out.append(Monomial(a0v, a1v, a2v, a3v, 0.0))
    return out^


def _tri_node_exponents(P: Int) raises -> List[Monomial]:
    """Canonical-ordered Lagrange ref-triangle nodes at order P.
    Only the a0, a1, a2 fields are used; a3 is always 0.  Order:
    vertex 0, vertex 1, vertex 2, edges (0,1) (1,2) (2,0), interior."""
    var out = List[Monomial]()
    out.append(Monomial(P, 0, 0, 0, 0.0))
    out.append(Monomial(0, P, 0, 0, 0.0))
    out.append(Monomial(0, 0, P, 0, 0.0))
    if P < 2:
        return out^
    # Edge (0,1), (1,2), (2,0) in face-local order.
    var edges_a = List[Int]()
    edges_a.append(0)
    edges_a.append(1)
    edges_a.append(2)
    var edges_b = List[Int]()
    edges_b.append(1)
    edges_b.append(2)
    edges_b.append(0)
    for e in range(3):
        var a_idx = edges_a[e]
        var b_idx = edges_b[e]
        for k in range(1, P):
            var exp0 = P - k if a_idx == 0 else (k if b_idx == 0 else 0)
            var exp1 = P - k if a_idx == 1 else (k if b_idx == 1 else 0)
            var exp2 = P - k if a_idx == 2 else (k if b_idx == 2 else 0)
            out.append(Monomial(exp0, exp1, exp2, 0, 0.0))
    if P < 3:
        return out^
    # Interior of triangle: all entries >= 1.
    for a0v in range(P - 2, 0, -1):
        for a1v in range(P - a0v - 1, 0, -1):
            var a2v = P - a0v - a1v
            out.append(Monomial(a0v, a1v, a2v, 0, 0.0))
    return out^


# ----------------------------------------------------------------------
# Monomial basis of homogeneous degree P.
# ----------------------------------------------------------------------
#
# For the tet we enumerate (a0, a1, a2, a3) with a0 + a1 + a2 + a3 = P
# in lexicographic order.  The count is N_P(P), matching the number of
# Lagrange nodes.  This is the basis in which we expand each Lagrange
# nodal polynomial.


def _tet_degP_monomials(P: Int) raises -> List[Monomial]:
    var out = List[Monomial]()
    for a0v in range(P + 1):
        for a1v in range(P + 1 - a0v):
            for a2v in range(P + 1 - a0v - a1v):
                var a3v = P - a0v - a1v - a2v
                out.append(Monomial(a0v, a1v, a2v, a3v, 1.0))
    return out^


def _tri_degP_monomials(P: Int) raises -> List[Monomial]:
    var out = List[Monomial]()
    for a0v in range(P + 1):
        for a1v in range(P + 1 - a0v):
            var a2v = P - a0v - a1v
            out.append(Monomial(a0v, a1v, a2v, 0, 1.0))
    return out^


# ----------------------------------------------------------------------
# Linear algebra helpers (host-side, Float64).
# ----------------------------------------------------------------------


def mat_zero(n: Int) raises -> List[Float64]:
    var m = List[Float64]()
    for _ in range(n * n):
        m.append(0.0)
    return m^


def mat_get(m: List[Float64], n: Int, i: Int, j: Int) raises -> Float64:
    return m[i * n + j]


def mat_set(mut m: List[Float64], n: Int, i: Int, j: Int, v: Float64):
    m[i * n + j] = v


def mat_inv(m_in: List[Float64], n: Int) raises -> List[Float64]:
    """Gauss-Jordan inversion.  Precision is Float64; n is typically
    small (<= 84 for up to P=6), so no LAPACK needed."""
    var a = List[Float64]()
    for i in range(n):
        for j in range(n):
            a.append(m_in[i * n + j])
        for j in range(n):
            a.append(1.0 if i == j else 0.0)
    var cols = 2 * n
    for i in range(n):
        var pivot = i
        var best: Float64 = 0.0
        for r in range(i, n):
            var v = a[r * cols + i]
            if v < 0.0:
                v = -v
            if v > best:
                best = v
                pivot = r
        if best == 0.0:
            raise Error("mat_inv: singular matrix")
        if pivot != i:
            for c in range(cols):
                var tmp = a[i * cols + c]
                a[i * cols + c] = a[pivot * cols + c]
                a[pivot * cols + c] = tmp
        var piv_val = a[i * cols + i]
        for c in range(cols):
            a[i * cols + c] /= piv_val
        for r in range(n):
            if r == i:
                continue
            var f = a[r * cols + i]
            if f == 0.0:
                continue
            for c in range(cols):
                a[r * cols + c] -= f * a[i * cols + c]
    var inv = List[Float64]()
    for i in range(n):
        for j in range(n):
            inv.append(a[i * cols + n + j])
    return inv^


# ----------------------------------------------------------------------
# Lagrange basis coefficients via Vandermonde inversion.
# ----------------------------------------------------------------------
#
# Given the Lagrange nodes {x_j} and the homogeneous monomial basis
# {m_k(l)} (both of cardinality N_P), the Vandermonde matrix
#
#   V[j, k] = m_k(x_j)
#
# solves  V * C^T = I  for the coefficient matrix  C[i, k]  such that
#   phi_i(l) = sum_k C[i, k] * m_k(l)
# satisfies  phi_i(x_j) = delta_ij.  So  C^T = V^{-1},  i.e.
# C[i, k] = V^{-1}[k, i].


def _build_basis_coefs(nodes: List[Monomial], monos: List[Monomial], P: Int) raises -> List[Float64]:
    """Return coefficients C[i, k] flattened as row-major (N_P x N_P).
    Works for either tet (a0..a3) or tri (a0..a2 only; a3=0)."""
    var N = len(nodes)
    if N != len(monos):
        raise Error("_build_basis_coefs: node/monomial count mismatch")
    var V = List[Float64]()
    for _ in range(N * N):
        V.append(0.0)
    var Pf = Float64(P)
    for j in range(N):
        # Barycentric coords at node j: l_i = a_i / P.
        var l0 = Float64(nodes[j].a0) / Pf
        var l1 = Float64(nodes[j].a1) / Pf
        var l2 = Float64(nodes[j].a2) / Pf
        var l3 = Float64(nodes[j].a3) / Pf
        for k in range(N):
            var mk = monos[k]
            # l_i ^ a_i factor.  We hand-roll int-power via repeated
            # multiply: a_i <= P <= 6 or so, so this is a tight loop.
            var v: Float64 = 1.0
            for _ in range(mk.a0):
                v *= l0
            for _ in range(mk.a1):
                v *= l1
            for _ in range(mk.a2):
                v *= l2
            for _ in range(mk.a3):
                v *= l3
            V[j * N + k] = v
    var Vinv = mat_inv(V, N)
    # C[i, k] = Vinv[k, i]
    var C = List[Float64]()
    for _ in range(N * N):
        C.append(0.0)
    for i in range(N):
        for k in range(N):
            C[i * N + k] = Vinv[k * N + i]
    return C^


def _coefs_to_monomial_list(coefs: List[Float64], basis_idx: Int, monos: List[Monomial]) raises -> List[Monomial]:
    """Extract basis function i as a List[Monomial] (c = coefficient)."""
    var N = len(monos)
    var out = List[Monomial]()
    for k in range(N):
        var c = coefs[basis_idx * N + k]
        if c == 0.0:
            continue
        var mk = monos[k]
        out.append(Monomial(mk.a0, mk.a1, mk.a2, mk.a3, c))
    return out^


# ----------------------------------------------------------------------
# Derivative of a polynomial w.r.t. r_k (k = 0 for r, 1 for s, 2 for t)
# ----------------------------------------------------------------------
#
# Since l0 = 1 - r - s - t, dl0/dr_k = -1.  Otherwise dl_{k+1}/dr_k = 1,
# dl_{other}/dr_k = 0.


def _poly_drdx(p: List[Monomial], k: Int) raises -> List[Monomial]:
    var dl0: Float64 = -1.0
    var dl1: Float64 = 1.0 if k == 0 else 0.0
    var dl2: Float64 = 1.0 if k == 1 else 0.0
    var dl3: Float64 = 1.0 if k == 2 else 0.0
    var out = List[Monomial]()
    for idx in range(len(p)):
        var m = p[idx]
        if m.a0 > 0:
            out.append(Monomial(m.a0 - 1, m.a1, m.a2, m.a3, m.c * Float64(m.a0) * dl0))
        if m.a1 > 0 and dl1 != 0.0:
            out.append(Monomial(m.a0, m.a1 - 1, m.a2, m.a3, m.c * Float64(m.a1) * dl1))
        if m.a2 > 0 and dl2 != 0.0:
            out.append(Monomial(m.a0, m.a1, m.a2 - 1, m.a3, m.c * Float64(m.a2) * dl2))
        if m.a3 > 0 and dl3 != 0.0:
            out.append(Monomial(m.a0, m.a1, m.a2, m.a3 - 1, m.c * Float64(m.a3) * dl3))
    return out^


# ----------------------------------------------------------------------
# Face-to-element node map
# ----------------------------------------------------------------------
#
# For each tet face f (opp vertex f), the three other vertices v_a,
# v_b, v_c in ASCENDING order make up the triangle.  The face-local
# ordering mirrors the triangle's own canonical order: 3 vertex nodes,
# then 3 edge blocks (v_a-v_b, v_b-v_c, v_c-v_a) with P-1 nodes each,
# then face-interior nodes.
#
# Returns a flat list of shape [N_F * N_FP] where entry [f * N_FP + l]
# is the element-local index of the l-th face-local node on face f.


def _lookup_tet_node(tet_nodes: List[Monomial], a0: Int, a1: Int, a2: Int, a3: Int) raises -> Int:
    """Linear-scan reverse lookup (a0..a3) -> element-node index."""
    for i in range(len(tet_nodes)):
        var n = tet_nodes[i]
        if n.a0 == a0 and n.a1 == a1 and n.a2 == a2 and n.a3 == a3:
            return i
    return -1


def _face_to_element_node(P: Int, tet_nodes: List[Monomial]) raises -> List[Int32]:
    var N_FP_P = num_tri_nodes(P)
    var out = List[Int32]()
    for _ in range(4 * N_FP_P):
        out.append(Int32(-1))

    for f in range(4):
        # Other three vertices in ascending order: whichever values in
        # {0, 1, 2, 3} are not equal to f.  Each index shifts down by
        # one past f so the result is strictly increasing.
        var vA = 1 if f == 0 else 0
        var vB = 2 if f <= 1 else 1
        var vC = 3 if f <= 2 else 2

        # Face-local node 0, 1, 2: the three vertices.
        for l_idx, v_idx in [(0, vA), (1, vB), (2, vC)]:
            var e0 = P if v_idx == 0 else 0
            var e1 = P if v_idx == 1 else 0
            var e2 = P if v_idx == 2 else 0
            var e3 = P if v_idx == 3 else 0
            out[f * N_FP_P + l_idx] = Int32(_lookup_tet_node(tet_nodes, e0, e1, e2, e3))

        if P < 2:
            continue
        # Edges in face-local order: vA->vB, vB->vC, vC->vA.
        var pos = 3
        var e_a = List[Int]()
        var e_b = List[Int]()
        e_a.append(vA)
        e_b.append(vB)
        e_a.append(vB)
        e_b.append(vC)
        e_a.append(vC)
        e_b.append(vA)
        for e in range(3):
            var a_i = e_a[e]
            var b_i = e_b[e]
            for k in range(1, P):
                var exp0 = P - k if a_i == 0 else (k if b_i == 0 else 0)
                var exp1 = P - k if a_i == 1 else (k if b_i == 1 else 0)
                var exp2 = P - k if a_i == 2 else (k if b_i == 2 else 0)
                var exp3 = P - k if a_i == 3 else (k if b_i == 3 else 0)
                out[f * N_FP_P + pos] = Int32(_lookup_tet_node(tet_nodes, exp0, exp1, exp2, exp3))
                pos += 1

        if P < 3:
            continue
        # Face-interior nodes, iterated in the same lex-descending
        # scheme as `_tet_node_exponents` generates them.
        for a0v in range(P - 2, 0, -1):
            for a1v in range(P - a0v - 1, 0, -1):
                var a2v = P - a0v - a1v
                var exp0 = a0v if vA == 0 else (a1v if vB == 0 else (a2v if vC == 0 else 0))
                var exp1 = a0v if vA == 1 else (a1v if vB == 1 else (a2v if vC == 1 else 0))
                var exp2 = a0v if vA == 2 else (a1v if vB == 2 else (a2v if vC == 2 else 0))
                var exp3 = a0v if vA == 3 else (a1v if vB == 3 else (a2v if vC == 3 else 0))
                out[f * N_FP_P + pos] = Int32(_lookup_tet_node(tet_nodes, exp0, exp1, exp2, exp3))
                pos += 1
    return out^


# ======================================================================
# ReferenceElement[P] -- pre-computed operators for order P
# ======================================================================


struct ReferenceElement[P: Int = 2](Copyable, Movable):
    # Reference-space coordinates of each node, flattened as [N_P * 3].
    var node_pos: List[Float64]
    # Differentiation operators: D_ref[k, i, j] = (M_ref^-1 @ S_ref^k)[i, j].
    # Flattened row-major as [N_D * N_P * N_P].
    var D_ref: List[Float64]
    # Face lift operators: Lift_ref[f, i, m] = (M_ref^-1 @ L_raw^f)[i, m].
    # Flattened as [N_F * N_P * N_FP].
    var Lift_ref: List[Float64]
    # Inverse mass matrix (diagnostic only).
    var M_ref_inv: List[Float64]
    # Element-local node index for each face-local node.  Flattened as
    # [N_F * N_FP], same layout as the P=2 `ref_face_node` table.
    var face_to_elem: List[Int32]
    # Mass-matrix-weighted nodal quadrature weights for computing the
    # exact cell mean of a P=P Lagrange expansion: cell_mean =
    # sum_i node_weights[i] * q_i.  Defined so sum_i w_i = 1 (partition
    # of unity over the reference tet of volume 1/6).  At P=2 these are
    # -1/20 at the 4 vertex nodes and 1/5 at the 6 edge-midpoint nodes
    # (sum: -4/20 + 6/5 = 1).  The unweighted 1/N_P average that DG
    # codes often use is wrong beyond P=1 and destroys conservation of
    # the slope limiter (the vertex weights flip sign at P=2).
    var node_weights: List[Float64]

    def __init__(out self) raises:
        comptime Pval = Self.P
        var N_P_P = num_tet_nodes(Pval)
        var N_FP_P = num_tri_nodes(Pval)

        # 1. Lagrange node set for this order, canonically ordered.
        var tet_nodes = _tet_node_exponents(Pval)
        var tri_nodes = _tri_node_exponents(Pval)
        if len(tet_nodes) != N_P_P:
            raise Error("ReferenceElement: tet node count mismatch")
        if len(tri_nodes) != N_FP_P:
            raise Error("ReferenceElement: tri node count mismatch")

        # 2. Node physical positions (r, s, t) = (l1, l2, l3).
        self.node_pos = List[Float64]()
        var Pf = Float64(Pval)
        for i in range(N_P_P):
            var n = tet_nodes[i]
            self.node_pos.append(Float64(n.a1) / Pf)
            self.node_pos.append(Float64(n.a2) / Pf)
            self.node_pos.append(Float64(n.a3) / Pf)

        # 3. Basis coefficients via Vandermonde inverse.
        var tet_monos = _tet_degP_monomials(Pval)
        var tri_monos = _tri_degP_monomials(Pval)
        var tet_coefs = _build_basis_coefs(tet_nodes, tet_monos, Pval)
        var tri_coefs = _build_basis_coefs(tri_nodes, tri_monos, Pval)

        # Cache each basis function as a List[Monomial] for
        # multiply-and-integrate below.
        var tet_phi = List[List[Monomial]]()
        for i in range(N_P_P):
            tet_phi.append(_coefs_to_monomial_list(tet_coefs, i, tet_monos))
        var tri_phi = List[List[Monomial]]()
        for i in range(N_FP_P):
            tri_phi.append(_coefs_to_monomial_list(tri_coefs, i, tri_monos))

        # 4. Mass matrix on reference tet.
        var M = mat_zero(N_P_P)
        for i in range(N_P_P):
            for j in range(N_P_P):
                var prod = poly_mul(tet_phi[i], tet_phi[j])
                mat_set(M, N_P_P, i, j, integrate_ref_tet(prod))
        var M_inv = mat_inv(M, N_P_P)
        self.M_ref_inv = M_inv.copy()

        # 5. Stiffness matrix S[k, i, j] = integral (dphi_i/dr_k) * phi_j.
        var S = List[Float64]()
        for _ in range(3 * N_P_P * N_P_P):
            S.append(0.0)
        for k in range(3):
            for i in range(N_P_P):
                var dpi = _poly_drdx(tet_phi[i], k)
                for j in range(N_P_P):
                    var prod = poly_mul(dpi, tet_phi[j])
                    S[k * N_P_P * N_P_P + i * N_P_P + j] = integrate_ref_tet(prod)

        # D_ref[k] = M_inv @ S[k].
        self.D_ref = List[Float64]()
        for _ in range(3 * N_P_P * N_P_P):
            self.D_ref.append(0.0)
        for k in range(3):
            for i in range(N_P_P):
                for j in range(N_P_P):
                    var sum: Float64 = 0.0
                    for ip in range(N_P_P):
                        sum += M_inv[i * N_P_P + ip] * S[k * N_P_P * N_P_P + ip * N_P_P + j]
                    self.D_ref[k * N_P_P * N_P_P + i * N_P_P + j] = sum

        # 6. Reference-triangle mass matrix M_tri (size N_FP x N_FP).
        var M_tri = mat_zero(N_FP_P)
        for i in range(N_FP_P):
            for j in range(N_FP_P):
                var prod = poly_mul(tri_phi[i], tri_phi[j])
                mat_set(M_tri, N_FP_P, i, j, integrate_ref_tri(prod))

        # 7. Face-to-element node map (element-local node per face-local
        # index on each of the 4 faces).
        self.face_to_elem = _face_to_element_node(Pval, tet_nodes)

        # 8. Face lift operator.  L_raw[f, i, m] = 2 * M_tri[l_i(f), m]
        # if element node i is on face f at face-local index l_i(f),
        # else 0.  Factor of 2 converts from reference-triangle
        # integration (area 1/2) to per-unit-area form.  The caller
        # multiplies by the physical face area at runtime.
        var L_raw = List[Float64]()
        for _ in range(4 * N_P_P * N_FP_P):
            L_raw.append(0.0)
        for f in range(4):
            for l in range(N_FP_P):
                var i = Int(self.face_to_elem[f * N_FP_P + l])
                if i < 0:
                    raise Error("ReferenceElement: face_to_elem lookup failed")
                for m in range(N_FP_P):
                    L_raw[f * N_P_P * N_FP_P + i * N_FP_P + m] = 2.0 * M_tri[l * N_FP_P + m]

        # Lift_ref[f, i, m] = M_inv @ L_raw[f].
        self.Lift_ref = List[Float64]()
        for _ in range(4 * N_P_P * N_FP_P):
            self.Lift_ref.append(0.0)
        for f in range(4):
            for i in range(N_P_P):
                for m in range(N_FP_P):
                    var sum: Float64 = 0.0
                    for ip in range(N_P_P):
                        sum += M_inv[i * N_P_P + ip] * L_raw[f * N_P_P * N_FP_P + ip * N_FP_P + m]
                    self.Lift_ref[f * N_P_P * N_FP_P + i * N_FP_P + m] = sum

        # 9. Node weights = integral of each basis function over the
        # reference tet, normalised by reference volume 1/6 so they sum
        # to 1.  Used by the BJ limiter to form the exact cell mean of
        # a P=P Lagrange expansion (the naive unweighted nodal average
        # is wrong at P>=2 and breaks limiter conservation).
        self.node_weights = List[Float64]()
        var v_ref = 1.0 / 6.0
        for i in range(N_P_P):
            self.node_weights.append(integrate_ref_tet(tet_phi[i]) / v_ref)


# ----------------------------------------------------------------------
# Back-compat helpers -- existing drivers still call this name-shape.
# ----------------------------------------------------------------------


def to_float32(src: List[Float64]) raises -> List[Float32]:
    var out = List[Float32]()
    for i in range(len(src)):
        out.append(Float32(src[i]))
    return out^


@fieldwise_init
struct ReferenceOperators(Movable):
    var D_ref: List[Float32]
    var Lift_ref: List[Float32]
    var node_weights: List[Float32]


def build_reference_operators(mut nvtx: NvtxContext) raises -> ReferenceOperators:
    """Build the default (P=2) reference operators used by every
    existing driver.  A parameterised `build_reference_operators_p`
    will land once the mesh + solver + VTU stack takes P as a comptime
    parameter."""
    nvtx.push_range("reference_element")
    var re = ReferenceElement[2]()
    var out = ReferenceOperators(D_ref=to_float32(re.D_ref), Lift_ref=to_float32(re.Lift_ref), node_weights=to_float32(re.node_weights))
    nvtx.pop_range()
    return out^


# ----------------------------------------------------------------------
# Face-local node lookup (back-compat for existing mesh-builder code
# that still uses the hand-coded P=2 ref_face_node function).
# ----------------------------------------------------------------------


def ref_face_node(f: Int, l: Int) raises -> Int:
    """Element-local node index for face-local position l on face f
    at order P=2.  Kept alongside the new `ReferenceElement.face_to_elem`
    generic-order table so that `local_mesh.mojo` (still P=2-specific)
    keeps compiling."""
    # Hand-unrolled P=2 table for speed.
    if f == 0:
        if l == 0:
            return 1
        if l == 1:
            return 2
        if l == 2:
            return 3
        if l == 3:
            return 5
        if l == 4:
            return 9
        return 8
    if f == 1:
        if l == 0:
            return 0
        if l == 1:
            return 2
        if l == 2:
            return 3
        if l == 3:
            return 6
        if l == 4:
            return 9
        return 7
    if f == 2:
        if l == 0:
            return 0
        if l == 1:
            return 1
        if l == 2:
            return 3
        if l == 3:
            return 4
        if l == 4:
            return 8
        return 7
    # f == 3
    if l == 0:
        return 0
    if l == 1:
        return 1
    if l == 2:
        return 2
    if l == 3:
        return 4
    if l == 4:
        return 5
    return 6
