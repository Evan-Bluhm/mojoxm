# ======================================================================
# 2D triangular reference element, equispaced Lagrange, order P
# ======================================================================
#
# A 2D companion to src/reference.mojo.  Produces the Lagrange basis,
# mass matrix, differentiation matrices, and edge-lift operators on
# the unit reference triangle
#
#     (r, s) in { (l1, l2) : 0 <= r, 0 <= s, r + s <= 1 }
#
# with barycentric coordinates (l0, l1, l2) = (1-r-s, r, s).
#
#   NP_2d(P)  = (P+1)(P+2)/2      nodes per triangle      (1, 3, 6, 10, 15, ...)
#   NE        = 3                  edges per triangle
#   NFP_edge  = P + 1              nodes per edge including endpoints
#
# Node order matches the `_tri_node_exponents` ordering already used
# elsewhere: 3 vertex nodes, then 3 edge-interior blocks ((0,1), (1,2),
# (2,0)), then face-interior nodes (for P >= 3).
#
# This module only exposes the math; meshing, solver, and VTU output
# in 2D are future work.  `make test-reference-2d` validates the math
# foundation at P=1..4 (SPD mass matrix, nodes inside the reference
# simplex, edge-to-element coverage).
# ======================================================================

from src.reference import (
    Monomial, factorial, integrate_ref_tri, poly_mul,
    mat_zero, mat_get, mat_set, mat_inv,
    num_tri_nodes, _tri_node_exponents, _tri_degP_monomials,
    _build_basis_coefs, _coefs_to_monomial_list, _poly_drdx,
)


# ----------------------------------------------------------------------
# Size helpers
# ----------------------------------------------------------------------

def num_tri_nodes_2d(P: Int) -> Int:
    """Number of Lagrange nodes on a 2D triangular element at order P.
    Identical to num_tri_nodes; aliased here for clarity in 2D contexts."""
    return num_tri_nodes(P)


def num_edge_nodes(P: Int) -> Int:
    """Number of Lagrange nodes on a 1D edge at order P, including the
    two endpoints (i.e. P+1)."""
    return P + 1


# ----------------------------------------------------------------------
# Edge-to-element node map
# ----------------------------------------------------------------------
# For each of the three edges ((v0, v1), (v1, v2), (v2, v0)), build the
# list of element-local node indices in edge-local order:
#   edge_local 0       -> vertex va (the first endpoint)
#   edge_local 1..P-1  -> interior-of-edge, running from va toward vb
#   edge_local P       -> vertex vb
#
# Returns a flat List[Int32] of shape [3 * (P + 1)].

def _lookup_tri_node(
    tri_nodes: List[Monomial],
    a0: Int, a1: Int, a2: Int,
) raises -> Int:
    for i in range(len(tri_nodes)):
        var n = tri_nodes[i]
        if n.a0 == a0 and n.a1 == a1 and n.a2 == a2:
            return i
    return -1


def _edge_to_element_node(
    P: Int, tri_nodes: List[Monomial],
) raises -> List[Int32]:
    var nfp_edge = P + 1
    var out = List[Int32]()
    for _ in range(3 * nfp_edge):
        out.append(Int32(-1))

    # Edges (va, vb).
    var va_list = List[Int]()
    va_list.append(0); va_list.append(1); va_list.append(2)
    var vb_list = List[Int]()
    vb_list.append(1); vb_list.append(2); vb_list.append(0)

    for e in range(3):
        var va = va_list[e]
        var vb = vb_list[e]
        for k in range(nfp_edge):
            # edge-local node k corresponds to barycentric weights with
            # (P - k) mass on va, (k) mass on vb, 0 elsewhere.
            var exp0 = (P - k) if va == 0 else (k if vb == 0 else 0)
            var exp1 = (P - k) if va == 1 else (k if vb == 1 else 0)
            var exp2 = (P - k) if va == 2 else (k if vb == 2 else 0)
            var i = _lookup_tri_node(tri_nodes, exp0, exp1, exp2)
            if i < 0:
                raise Error(
                    "_edge_to_element_node: failed to locate node for "
                    "edge " + String(e) + " slot " + String(k)
                )
            out[e * nfp_edge + k] = Int32(i)
    return out^


# ----------------------------------------------------------------------
# ReferenceElement2D[P]
# ----------------------------------------------------------------------
# Mirrors ReferenceElement[P] from src/reference.mojo but restricted to
# the 2D triangle: two differentiation directions (r, s), three edges
# (each with P+1 nodes), and the reference triangle's mass matrix
# doubles as both the volume and the edge-lift raw integral once paired
# with a 1D edge mass.
# ----------------------------------------------------------------------

struct ReferenceElement2D[P: Int = 2](Copyable, Movable):
    # Reference-space coordinates of each node, flattened as [NP * 2].
    # Each entry is (r, s) = (l1, l2); the third barycentric coord l0
    # is implicitly 1 - r - s.
    var node_pos: List[Float64]
    # Differentiation operators: D_ref[k, i, j] = (M_ref^-1 @ S_ref^k)[i, j]
    # with k in {0 (d/dr), 1 (d/ds)}.  Flattened as [2 * NP * NP].
    var D_ref: List[Float64]
    # Edge-lift operators: Lift_ref[e, i, m] = (M_ref^-1 @ L_raw^e)[i, m]
    # where e indexes the 3 edges, i the NP nodal DOFs, m the (P+1)
    # edge-local DOFs.  Flattened as [3 * NP * (P+1)].
    var Lift_ref: List[Float64]
    # Inverse 2D mass matrix (diagnostic).
    var M_ref_inv: List[Float64]
    # Per-node weights for the mass-matrix-weighted cell mean.
    # node_weights[i] = 2 * int phi_i(r, s) dr ds on the reference
    # triangle (A_ref = 1/2, so the factor of 2 normalises so
    # sum(node_weights) = 1 and cell_mean = sum_i q_i * node_weights[i]).
    # For P=1 (3 vertices) every weight is 1/3.  For P=2 the 3 vertex
    # weights are 0 and the 3 edge-midpoint weights are 1/3.  Using
    # the unweighted nodal average instead would be a correct cell
    # mean only at P=1; at P>=2 it breaks scheme conservation in
    # anything that needs a true cell mean (e.g. the BJ limiter).
    var node_weights: List[Float64]
    # Element-local node index for each edge-local node.  [3 * (P+1)].
    var edge_to_elem: List[Int32]

    def __init__(out self) raises:
        comptime Pval = Self.P
        var NP_P = num_tri_nodes_2d(Pval)
        var NFP_edge = num_edge_nodes(Pval)

        # 1. Lagrange node set for this order, canonically ordered.
        var tri_nodes = _tri_node_exponents(Pval)
        if len(tri_nodes) != NP_P:
            raise Error("ReferenceElement2D: node count mismatch")

        # 2. Node physical positions (r, s) = (l1, l2).
        self.node_pos = List[Float64]()
        var Pf = Float64(Pval)
        for i in range(NP_P):
            var n = tri_nodes[i]
            self.node_pos.append(Float64(n.a1) / Pf)
            self.node_pos.append(Float64(n.a2) / Pf)

        # 3. Basis coefficients via Vandermonde inverse.
        var tri_monos = _tri_degP_monomials(Pval)
        var tri_coefs = _build_basis_coefs(tri_nodes, tri_monos, Pval)
        var tri_phi = List[List[Monomial]]()
        for i in range(NP_P):
            tri_phi.append(_coefs_to_monomial_list(tri_coefs, i, tri_monos))

        # 4. Mass matrix on the reference triangle.
        var M = mat_zero(NP_P)
        for i in range(NP_P):
            for j in range(NP_P):
                var prod = poly_mul(tri_phi[i], tri_phi[j])
                mat_set(M, NP_P, i, j, integrate_ref_tri(prod))
        var M_inv = mat_inv(M, NP_P)
        self.M_ref_inv = M_inv.copy()

        # 4b. Per-node cell-mean weights.  cell_mean = sum_i q_i * w_i
        # where w_i = int phi_i dA_phys / A_phys = 2 * int phi_i dr ds
        # (reference integral; A_ref = 1/2).  Since partition-of-unity
        # gives sum_i phi_i = 1, sum_i w_i = 1.
        self.node_weights = List[Float64]()
        for i in range(NP_P):
            var w = 2.0 * integrate_ref_tri(tri_phi[i])
            self.node_weights.append(w)

        # 5. Stiffness matrices S[k, i, j] = integral (dphi_i/dr_k) * phi_j
        # on the reference triangle; k in {0: d/dr, 1: d/ds}.
        var S = List[Float64]()
        for _ in range(2 * NP_P * NP_P):
            S.append(0.0)
        for k in range(2):
            for i in range(NP_P):
                var dpi = _poly_drdx(tri_phi[i], k)
                for j in range(NP_P):
                    var prod = poly_mul(dpi, tri_phi[j])
                    S[k * NP_P * NP_P + i * NP_P + j] = integrate_ref_tri(prod)

        # D_ref[k] = M_inv @ S[k].
        self.D_ref = List[Float64]()
        for _ in range(2 * NP_P * NP_P):
            self.D_ref.append(0.0)
        for k in range(2):
            for i in range(NP_P):
                for j in range(NP_P):
                    var sum: Float64 = 0.0
                    for ip in range(NP_P):
                        sum += (
                            M_inv[i * NP_P + ip]
                            * S[k * NP_P * NP_P + ip * NP_P + j]
                        )
                    self.D_ref[k * NP_P * NP_P + i * NP_P + j] = sum

        # 6. 1D edge mass matrix M_edge (size (P+1) x (P+1)).  Uses the
        # standard Gauss-Lobatto-like equispaced Lagrange basis on the
        # reference edge [0, 1]; the entry M_edge[l, m] is
        #   integral_0^1 psi_l(xi) psi_m(xi) dxi
        # where psi_l is the Lagrange polynomial at equispaced node
        # l / P.  Computed in closed form by evaluating the integrand
        # polynomial symbolically.
        var M_edge = _build_edge_mass_matrix(Pval)

        # 7. Edge-to-element map.
        self.edge_to_elem = _edge_to_element_node(Pval, tri_nodes)

        # 8. Edge-lift operator.  L_raw[e, i, m] = M_edge[l_i(e), m] if
        # element node i is on edge e at edge-local index l_i(e), else 0.
        # Then Lift_ref = M_inv @ L_raw.  Caller multiplies by physical
        # edge length at runtime to recover the per-element boundary
        # contribution.
        var L_raw = List[Float64]()
        for _ in range(3 * NP_P * NFP_edge):
            L_raw.append(0.0)
        for e in range(3):
            for l in range(NFP_edge):
                var i = Int(self.edge_to_elem[e * NFP_edge + l])
                if i < 0:
                    raise Error("ReferenceElement2D: edge_to_elem lookup failed")
                for m in range(NFP_edge):
                    L_raw[e * NP_P * NFP_edge + i * NFP_edge + m] = (
                        M_edge[l * NFP_edge + m]
                    )

        self.Lift_ref = List[Float64]()
        for _ in range(3 * NP_P * NFP_edge):
            self.Lift_ref.append(0.0)
        for e in range(3):
            for i in range(NP_P):
                for m in range(NFP_edge):
                    var sum: Float64 = 0.0
                    for ip in range(NP_P):
                        sum += (
                            M_inv[i * NP_P + ip]
                            * L_raw[e * NP_P * NFP_edge + ip * NFP_edge + m]
                        )
                    self.Lift_ref[e * NP_P * NFP_edge + i * NFP_edge + m] = sum


# ----------------------------------------------------------------------
# 1D edge mass matrix
# ----------------------------------------------------------------------
# On the reference edge xi in [0, 1] with equispaced nodes xi_k = k / P
# (k = 0..P), each Lagrange polynomial psi_k satisfies psi_k(xi_l) =
# delta_{k,l} and has degree P.  Writing psi_k(xi) = sum_j A[k, j] xi^j
# and requiring psi_k(xi_l) = delta_kl gives A . V^T = I, so A =
# (V^{-1})^T, i.e. A[k, j] = V^{-1}[j, k].  The edge mass matrix is
#   M_edge[k, l] = integral_0^1 psi_k(xi) psi_l(xi) dxi
#                = sum_{i, j} A[k, i] A[l, j] / (i + j + 1)
#                = sum_{i, j} V^{-1}[i, k] V^{-1}[j, l] / (i + j + 1)
# since integral_0^1 xi^{i+j} dxi = 1 / (i + j + 1).

def _build_edge_mass_matrix(P: Int) raises -> List[Float64]:
    var NE = P + 1

    # Vandermonde V[k, j] = xi_k^j.
    var V = List[Float64]()
    for _ in range(NE * NE):
        V.append(0.0)
    for k in range(NE):
        var xik = Float64(k) / Float64(P)
        var pow: Float64 = 1.0
        for j in range(NE):
            V[k * NE + j] = pow
            pow *= xik

    # C = V^{-1}.  psi_k's j-th coefficient is A[k, j] = C[j, k].
    var C = mat_inv(V, NE)

    # M_edge[k, l] = sum_{i, j} C[i, k] C[j, l] / (i + j + 1).
    var M = List[Float64]()
    for _ in range(NE * NE):
        M.append(0.0)
    for k in range(NE):
        for l in range(NE):
            var s: Float64 = 0.0
            for i in range(NE):
                for j in range(NE):
                    s += C[i * NE + k] * C[j * NE + l] / Float64(i + j + 1)
            M[k * NE + l] = s
    return M^
