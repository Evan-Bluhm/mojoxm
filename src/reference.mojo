# ======================================================================
# P2 Tetrahedral Reference Element
# ======================================================================
# 10-node P2 Lagrange basis on the reference tet with vertices
# (0,0,0), (1,0,0), (0,1,0), (0,0,1).  Node ordering matches the VTK
# VTK_QUADRATIC_TETRA cell type (cell type 24) so we can emit the
# solution to ParaView without reordering.
#
# Barycentric coordinates: l0 = 1-r-s-t, l1 = r, l2 = s, l3 = t.
#
# Node   Position (r,s,t)        Basis function
#   0    (0,0,0)                 l0 (2 l0 - 1)    vertex 0
#   1    (1,0,0)                 l1 (2 l1 - 1)    vertex 1
#   2    (0,1,0)                 l2 (2 l2 - 1)    vertex 2
#   3    (0,0,1)                 l3 (2 l3 - 1)    vertex 3
#   4    (0.5,0,0)               4 l0 l1          edge 0-1
#   5    (0.5,0.5,0)             4 l1 l2          edge 1-2
#   6    (0,0.5,0)               4 l2 l0          edge 2-0
#   7    (0,0,0.5)               4 l0 l3          edge 0-3
#   8    (0.5,0,0.5)             4 l1 l3          edge 1-3
#   9    (0,0.5,0.5)             4 l2 l3          edge 2-3
#
# Faces of the tet (opposite-vertex convention, face f opposite vertex f):
#   Face 0 (opp v0): vertex nodes {1,2,3}, edge-midpoint nodes {5,9,8}
#   Face 1 (opp v1): vertex nodes {0,2,3}, edge-midpoint nodes {6,9,7}
#   Face 2 (opp v2): vertex nodes {0,1,3}, edge-midpoint nodes {4,8,7}
#   Face 3 (opp v3): vertex nodes {0,1,2}, edge-midpoint nodes {4,5,6}
#
# For each face, local face-node ordering is (v_a, v_b, v_c, mid_ab,
# mid_bc, mid_ca) where (v_a, v_b, v_c) are the 3 other vertex indices
# in ascending order (0<1<2, 0<1<3, etc.).
# ======================================================================

from std.math import sqrt

comptime N_P = 10        # nodes per element (P2 tet)
comptime N_F = 4         # faces per element
comptime N_FP = 6        # face nodes per face (P2 tri)
comptime N_D = 3         # spatial dimensions

# ----------------------------------------------------------------------
# Element-node indices on each face (in face-local ordering used below)
# face_nodes[f][l]: element-local node index for face-local node l
# Order per face: 3 vertex nodes (ascending), then 3 edge midpoints
# following the 01, 12, 20 ordering in face-local indexing.
# ----------------------------------------------------------------------
# Face 0 opp vertex 0: face vertices (v1,v2,v3); edges 1-2, 2-3, 3-1
#   element-node idxs: vertices = (1,2,3), edge mids = (5,9,8)
# Face 1 opp vertex 1: face vertices (v0,v2,v3); edges 0-2, 2-3, 3-0
#   element-node idxs: vertices = (0,2,3), edge mids = (6,9,7)
# Face 2 opp vertex 2: face vertices (v0,v1,v3); edges 0-1, 1-3, 3-0
#   element-node idxs: vertices = (0,1,3), edge mids = (4,8,7)
# Face 3 opp vertex 3: face vertices (v0,v1,v2); edges 0-1, 1-2, 2-0
#   element-node idxs: vertices = (0,1,2), edge mids = (4,5,6)
#
# These are "reference" face-node orderings.  The mesh builder will
# permute these to match a canonical globally-agreed ordering (based on
# sorted vertex ids) so that two elements sharing a face agree on which
# face-node index refers to which physical location.

# ----------------------------------------------------------------------
# P2 basis polynomial representation.
#
# Each basis function phi_i(l0,l1,l2,l3) is a sum of monomial terms
#   coef * l0^a0 * l1^a1 * l2^a2 * l3^a3
# We store each term as (a0,a1,a2,a3,coef).
# ----------------------------------------------------------------------

@fieldwise_init
struct Monomial(ImplicitlyCopyable, Movable):
    var a0: Int
    var a1: Int
    var a2: Int
    var a3: Int
    var c: Float64

def poly_phi(i: Int) raises -> List[Monomial]:
    # Vertex bases: phi_i = l_i (2 l_i - 1) = 2 l_i^2 - l_i
    # Edge bases:  phi_ij = 4 l_i l_j
    if i == 0:
        return [Monomial(2,0,0,0, 2.0), Monomial(1,0,0,0, -1.0)]
    if i == 1:
        return [Monomial(0,2,0,0, 2.0), Monomial(0,1,0,0, -1.0)]
    if i == 2:
        return [Monomial(0,0,2,0, 2.0), Monomial(0,0,1,0, -1.0)]
    if i == 3:
        return [Monomial(0,0,0,2, 2.0), Monomial(0,0,0,1, -1.0)]
    if i == 4:
        return [Monomial(1,1,0,0, 4.0)]     # edge 0-1
    if i == 5:
        return [Monomial(0,1,1,0, 4.0)]     # edge 1-2
    if i == 6:
        return [Monomial(1,0,1,0, 4.0)]     # edge 2-0
    if i == 7:
        return [Monomial(1,0,0,1, 4.0)]     # edge 0-3
    if i == 8:
        return [Monomial(0,1,0,1, 4.0)]     # edge 1-3
    if i == 9:
        return [Monomial(0,0,1,1, 4.0)]     # edge 2-3
    raise Error("poly_phi: i out of range")

def poly_mul(
    a: List[Monomial], b: List[Monomial]
) raises -> List[Monomial]:
    var out = List[Monomial]()
    for ai in range(len(a)):
        for bi in range(len(b)):
            var m = Monomial(
                a[ai].a0 + b[bi].a0,
                a[ai].a1 + b[bi].a1,
                a[ai].a2 + b[bi].a2,
                a[ai].a3 + b[bi].a3,
                a[ai].c  * b[bi].c,
            )
            out.append(m)
    return out^

def factorial(n: Int) raises -> Float64:
    var f: Float64 = 1.0
    for k in range(2, n + 1):
        f *= Float64(k)
    return f

# Exact integral of prod l_i^a_i over reference tet (volume 1/6):
#   integral = (a0! a1! a2! a3!) / (|a| + 3)!
def integrate_ref_tet(p: List[Monomial]) raises -> Float64:
    var s: Float64 = 0.0
    for i in range(len(p)):
        var m = p[i]
        var denom_n = m.a0 + m.a1 + m.a2 + m.a3 + 3
        var num = factorial(m.a0) * factorial(m.a1) * factorial(m.a2) * factorial(m.a3)
        s += m.c * num / factorial(denom_n)
    return s

# grad_r basis: returns list of list (one per reference-coord direction)
# poly_dphi[k] = dphi_i / dr_k as a Monomial list.
# dl0/dr = -1, dl0/ds = -1, dl0/dt = -1
# dl1/dr = 1, else 0
# dl2/ds = 1, else 0
# dl3/dt = 1, else 0
def poly_dphi(i: Int, k: Int) raises -> List[Monomial]:
    # k = 0:r, 1:s, 2:t
    # d(prod l_j^a_j)/dr_k = sum_j a_j * l_j^(a_j-1) * (dl_j/dr_k) * (rest)
    # Only l_0 has a non-trivial gradient in every direction (-1), and one of
    # l_1, l_2, l_3 contributes +1 (the one whose index matches k+1).
    var dl0: Float64 = -1.0
    var dl1: Float64 = 1.0 if k == 0 else 0.0
    var dl2: Float64 = 1.0 if k == 1 else 0.0
    var dl3: Float64 = 1.0 if k == 2 else 0.0
    var p = poly_phi(i)
    var out = List[Monomial]()
    for idx in range(len(p)):
        var m = p[idx]
        # derivative w.r.t. l0
        if m.a0 > 0 and dl0 != 0.0:
            out.append(Monomial(m.a0 - 1, m.a1, m.a2, m.a3,
                                m.c * Float64(m.a0) * dl0))
        if m.a1 > 0 and dl1 != 0.0:
            out.append(Monomial(m.a0, m.a1 - 1, m.a2, m.a3,
                                m.c * Float64(m.a1) * dl1))
        if m.a2 > 0 and dl2 != 0.0:
            out.append(Monomial(m.a0, m.a1, m.a2 - 1, m.a3,
                                m.c * Float64(m.a2) * dl2))
        if m.a3 > 0 and dl3 != 0.0:
            out.append(Monomial(m.a0, m.a1, m.a2, m.a3 - 1,
                                m.c * Float64(m.a3) * dl3))
    return out^

# P2 basis on reference triangle (unit simplex u>=0, v>=0, u+v<=1).
# Barycentric: u0 = 1-u-v, u1 = u, u2 = v.
# 6 nodes: 3 vertices + 3 edge midpoints.  Face-local ordering:
#   0 = vertex 0, 1 = vertex 1, 2 = vertex 2
#   3 = mid 0-1, 4 = mid 1-2, 5 = mid 2-0
#
# Monomials here use a 3-tuple (u0_exp, u1_exp, u2_exp) encoded as
# (a0,a1,a2) with a3 unused.
def poly_psi_tri(l: Int) raises -> List[Monomial]:
    if l == 0:
        return [Monomial(2,0,0,0, 2.0), Monomial(1,0,0,0, -1.0)]
    if l == 1:
        return [Monomial(0,2,0,0, 2.0), Monomial(0,1,0,0, -1.0)]
    if l == 2:
        return [Monomial(0,0,2,0, 2.0), Monomial(0,0,1,0, -1.0)]
    if l == 3:
        return [Monomial(1,1,0,0, 4.0)]    # mid 0-1
    if l == 4:
        return [Monomial(0,1,1,0, 4.0)]    # mid 1-2
    if l == 5:
        return [Monomial(1,0,1,0, 4.0)]    # mid 2-0
    raise Error("poly_psi_tri: l out of range")

# Exact integral of prod u_i^a_i over unit reference triangle
# (vertices (0,0), (1,0), (0,1), area = 1/2):
#   integral = (a0! a1! a2!) / (|a| + 2)!
def integrate_ref_tri(p: List[Monomial]) raises -> Float64:
    var s: Float64 = 0.0
    for i in range(len(p)):
        var m = p[i]
        var denom_n = m.a0 + m.a1 + m.a2 + 2
        var num = factorial(m.a0) * factorial(m.a1) * factorial(m.a2)
        s += m.c * num / factorial(denom_n)
    return s

# ----------------------------------------------------------------------
# 10x10 matrix helpers (host-side only).  We do these on the CPU at
# startup since the matrices are tiny.
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

# Invert n x n matrix via Gauss-Jordan, in place.  Returns the inverse.
def mat_inv(m_in: List[Float64], n: Int) raises -> List[Float64]:
    # Augment with identity.
    var a = List[Float64]()
    for i in range(n):
        for j in range(n):
            a.append(m_in[i * n + j])
        for j in range(n):
            a.append(1.0 if i == j else 0.0)
    var cols = 2 * n
    for i in range(n):
        # Partial pivot
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
# ReferenceElement: pre-computed operators
# ----------------------------------------------------------------------
# We compute:
#   M_ref[N_P, N_P]               mass matrix on reference tet
#   M_ref_inv[N_P, N_P]           inverse mass matrix
#   S_ref[N_D, N_P, N_P]          stiffness (drphi_i * phi_j integrals)
#   M_tri_ref[N_FP, N_FP]         mass matrix on reference triangle
#   D_ref[N_D, N_P, N_P]          = M_ref_inv @ S_ref[k]
#   L_ref[N_F, N_P, N_FP]         raw face-lift operators (scaled by 2)
#   Lift_ref[N_F, N_P, N_FP]      = M_ref_inv @ L_ref[f]
#
# All stored flattened as row-major arrays (Float64).
# ----------------------------------------------------------------------

# Per-face, the element-local node indices on that face in
# "local face ordering": vertex0, vertex1, vertex2, mid01, mid12, mid20
# in face-local indexing (not to be confused with global canonical).
#
# Face 0 (opp v0): face verts (v1,v2,v3) -> element nodes (1,2,3)
#   edge mids 1-2, 2-3, 3-1 -> element nodes 5, 9, 8
# Face 1 (opp v1): (v0,v2,v3) -> (0,2,3); edges 0-2,2-3,3-0 -> 6,9,7
# Face 2 (opp v2): (v0,v1,v3) -> (0,1,3); edges 0-1,1-3,3-0 -> 4,8,7
# Face 3 (opp v3): (v0,v1,v2) -> (0,1,2); edges 0-1,1-2,2-0 -> 4,5,6

def ref_face_node(f: Int, l: Int) -> Int:
    # face f, face-local index l in (0..5)
    # flat table of 4*6 = 24 entries
    if f == 0:
        if l == 0: return 1
        if l == 1: return 2
        if l == 2: return 3
        if l == 3: return 5
        if l == 4: return 9
        return 8
    if f == 1:
        if l == 0: return 0
        if l == 1: return 2
        if l == 2: return 3
        if l == 3: return 6
        if l == 4: return 9
        return 7
    if f == 2:
        if l == 0: return 0
        if l == 1: return 1
        if l == 2: return 3
        if l == 3: return 4
        if l == 4: return 8
        return 7
    # f == 3
    if l == 0: return 0
    if l == 1: return 1
    if l == 2: return 2
    if l == 3: return 4
    if l == 4: return 5
    return 6

struct ReferenceElement(Copyable, Movable):
    var node_pos: List[Float64]      # [N_P * 3] reference-coord node positions
    var D_ref: List[Float64]         # [N_D * N_P * N_P]
    var Lift_ref: List[Float64]      # [N_F * N_P * N_FP]
    var M_ref_inv: List[Float64]     # [N_P * N_P]  (diag info for diagnostics)

    def __init__(out self) raises:
        # Reference node positions (matching VTK_QUADRATIC_TETRA ordering).
        self.node_pos = List[Float64]()
        var positions = [
            0.0, 0.0, 0.0,   # 0
            1.0, 0.0, 0.0,   # 1
            0.0, 1.0, 0.0,   # 2
            0.0, 0.0, 1.0,   # 3
            0.5, 0.0, 0.0,   # 4 mid 0-1
            0.5, 0.5, 0.0,   # 5 mid 1-2
            0.0, 0.5, 0.0,   # 6 mid 2-0
            0.0, 0.0, 0.5,   # 7 mid 0-3
            0.5, 0.0, 0.5,   # 8 mid 1-3
            0.0, 0.5, 0.5,   # 9 mid 2-3
        ]
        for p in positions:
            self.node_pos.append(p)

        # Build 10x10 mass matrix on reference tet.
        var M = mat_zero(N_P)
        for i in range(N_P):
            var pi = poly_phi(i)
            for j in range(N_P):
                var pj = poly_phi(j)
                var prod = poly_mul(pi, pj)
                mat_set(M, N_P, i, j, integrate_ref_tet(prod))

        var M_inv = mat_inv(M, N_P)
        self.M_ref_inv = M_inv.copy()

        # Build S_ref[k, i, j] = integral (dphi_i / dr_k) * phi_j  dV_ref
        # Store flat [N_D * N_P * N_P].
        var S = List[Float64]()
        for _ in range(N_D * N_P * N_P):
            S.append(0.0)
        for k in range(N_D):
            for i in range(N_P):
                var dpi = poly_dphi(i, k)
                for j in range(N_P):
                    var pj = poly_phi(j)
                    var prod = poly_mul(dpi, pj)
                    S[k * N_P * N_P + i * N_P + j] = integrate_ref_tet(prod)

        # D_ref[k][i][j] = sum_ip M_inv[i][ip] * S[k][ip][j]
        self.D_ref = List[Float64]()
        for _ in range(N_D * N_P * N_P):
            self.D_ref.append(0.0)
        for k in range(N_D):
            for i in range(N_P):
                for j in range(N_P):
                    var s: Float64 = 0.0
                    for ip in range(N_P):
                        s += M_inv[i * N_P + ip] * S[k * N_P * N_P + ip * N_P + j]
                    self.D_ref[k * N_P * N_P + i * N_P + j] = s

        # Build reference triangle mass matrix M_tri_ref (6x6).
        var M_tri = mat_zero(N_FP)
        for i in range(N_FP):
            var pi = poly_psi_tri(i)
            for j in range(N_FP):
                var pj = poly_psi_tri(j)
                var prod = poly_mul(pi, pj)
                mat_set(M_tri, N_FP, i, j, integrate_ref_tri(prod))

        # Build L_ref[f][i][m]:
        # integral_face[f] of phi_i * psi_m dS  (on reference face)
        # Using the Jacobian of the affine face mapping:
        #   ds_physical_face = (A_phys / A_ref_face) * du dv on ref triangle
        # But since we're building a reference-element operator (before
        # applying physical area), we want
        #   L_ref[f][i][m] = 2 * M_tri[l_i(f), m] if i is on face f, else 0
        # The factor 2 = 1 / (area of unit triangle = 1/2) converts
        # reference-triangle integration to a per-unit-area quantity.
        #
        # In the RHS this is multiplied by A_f (physical face area):
        #   face_contribution_i = A_f * sum_m L_ref[f][i][m] * F*_f[m]
        var L_raw = List[Float64]()
        for _ in range(N_F * N_P * N_FP):
            L_raw.append(0.0)
        for f in range(N_F):
            for l in range(N_FP):
                var i = ref_face_node(f, l)
                for m in range(N_FP):
                    L_raw[f * N_P * N_FP + i * N_FP + m] = 2.0 * M_tri[l * N_FP + m]

        # Lift_ref[f][i][m] = sum_ip M_inv[i][ip] * L_raw[f][ip][m]
        self.Lift_ref = List[Float64]()
        for _ in range(N_F * N_P * N_FP):
            self.Lift_ref.append(0.0)
        for f in range(N_F):
            for i in range(N_P):
                for m in range(N_FP):
                    var s: Float64 = 0.0
                    for ip in range(N_P):
                        s += M_inv[i * N_P + ip] * L_raw[f * N_P * N_FP + ip * N_FP + m]
                    self.Lift_ref[f * N_P * N_FP + i * N_FP + m] = s

# Flatten into Float32 arrays for GPU upload.
def to_float32(src: List[Float64]) raises -> List[Float32]:
    var out = List[Float32]()
    for i in range(len(src)):
        out.append(Float32(src[i]))
    return out^


# One-shot helper for drivers: build the P2 reference element, extract the
# two Float32-quantized operator arrays needed by the Solver, and
# push the `reference_element` NVTX range around it.  Drivers previously
# called ReferenceElement() and to_float32 twice each; this collapses that
# to a single call.
from src.nvtx import NvtxContext

@fieldwise_init
struct ReferenceOperators(Movable):
    var D_ref: List[Float32]
    var Lift_ref: List[Float32]

def build_reference_operators(
    mut nvtx: NvtxContext,
) raises -> ReferenceOperators:
    nvtx.push_range("reference_element")
    var re = ReferenceElement()
    var out = ReferenceOperators(
        D_ref=to_float32(re.D_ref),
        Lift_ref=to_float32(re.Lift_ref),
    )
    nvtx.pop_range()
    return out^


