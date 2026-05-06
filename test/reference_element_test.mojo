# ======================================================================
# reference_element_test -- validate higher-order Lagrange operators
# ======================================================================
#
# Three things we want to know about the general-order
# `ReferenceElement[P]` machinery:
#
#   1. P=2 matches the hand-coded P=2 operators we had before -- if
#      it doesn't, the existing drivers' bit-identical test suites
#      would already have caught it, but we test explicitly here for
#      fail-fast signal during reference-element edits.
#
#   2. Mass matrix is symmetric positive-definite at every supported
#      order.  A singular or indefinite M means the Vandermonde
#      inversion blew up or the node placement is degenerate.
#
#   3. Node positions sanity: node count matches N_P, positions are
#      in [0, 1], and every pair of distinct nodes is at a distance
#      >= 1 / (2P).
#
# Run with: `make test-reference` (hooked into the top-level Makefile).
# ======================================================================

from src.reference import (
    ReferenceElement,
    num_tet_nodes,
    num_tri_nodes,
)
from std.math import sqrt


def _is_spd(M: List[Float64], n: Int) raises -> Bool:
    """Cholesky factorization: succeeds iff M is SPD."""
    var L = List[Float64]()
    for _ in range(n * n):
        L.append(0.0)
    for j in range(n):
        var d = M[j * n + j]
        for k in range(j):
            d -= L[j * n + k] * L[j * n + k]
        if d <= 0.0:
            return False
        var ljj = sqrt(d)
        L[j * n + j] = ljj
        for i in range(j + 1, n):
            var s = M[i * n + j]
            for k in range(j):
                s -= L[i * n + k] * L[j * n + k]
            L[i * n + j] = s / ljj
    return True


def _test_sizes(P: Int) raises:
    var np = num_tet_nodes(P)
    var nfp = num_tri_nodes(P)
    var expected_np = (P + 1) * (P + 2) * (P + 3) // 6
    var expected_nfp = (P + 1) * (P + 2) // 2
    if np != expected_np:
        raise Error(
            "P="
            + String(P)
            + ": num_tet_nodes mismatch ("
            + String(np)
            + " vs "
            + String(expected_np)
            + ")"
        )
    if nfp != expected_nfp:
        raise Error("P=" + String(P) + ": num_tri_nodes mismatch")
    print("  P=", P, " sizes OK (N_P=", np, ", N_FP=", nfp, ")")


def _test_spd_mass_inverse(re: ReferenceElement) raises:
    var np = len(re.M_ref_inv)
    var n = 0
    # recover n from n*n = length
    while n * n < np:
        n += 1
    if n * n != np:
        raise Error("M_ref_inv not square")
    # Invert M_ref_inv back to M_ref (M is SPD iff its inverse is).
    # Cheaper: test that M_ref_inv itself is SPD, which it must be.
    if not _is_spd(re.M_ref_inv, n):
        raise Error("mass matrix inverse is not SPD")
    print("  M_ref_inv is SPD at N_P=", n)


def _test_node_positions(re: ReferenceElement, P: Int) raises:
    """Node count matches N_P; positions are on/in the reference tet
    (barycentric sum <= 1 + eps with all components >= -eps); distinct
    nodes are at least half a Lagrange grid spacing apart."""
    var n_nodes = len(re.node_pos) // 3
    var expected = (P + 1) * (P + 2) * (P + 3) // 6
    if n_nodes != expected:
        raise Error("node count mismatch")
    var eps = 1.0e-9
    for i in range(n_nodes):
        var r = re.node_pos[i * 3 + 0]
        var s = re.node_pos[i * 3 + 1]
        var t = re.node_pos[i * 3 + 2]
        if r < -eps or s < -eps or t < -eps or r + s + t > 1.0 + eps:
            raise Error("node " + String(i) + " outside reference tet")
    # Minimum pairwise distance.
    var min_d2: Float64 = 1.0e30
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            var dx = re.node_pos[i * 3 + 0] - re.node_pos[j * 3 + 0]
            var dy = re.node_pos[i * 3 + 1] - re.node_pos[j * 3 + 1]
            var dz = re.node_pos[i * 3 + 2] - re.node_pos[j * 3 + 2]
            var d2 = dx * dx + dy * dy + dz * dz
            if d2 < min_d2:
                min_d2 = d2
    var min_d = sqrt(min_d2)
    var expected_d = 0.5 / Float64(P)  # half a grid spacing lower bound
    if min_d < expected_d * 0.99:
        raise Error(
            "min node distance "
            + String(min_d)
            + " < expected "
            + String(expected_d)
        )
    print(
        "  node positions OK (",
        n_nodes,
        " nodes, min sep=",
        Float32(min_d),
        ")",
    )


def _test_face_to_elem_coverage(re: ReferenceElement) raises:
    """Every face-local index must map to a valid element-local node
    (>= 0) and within [0, N_P)."""
    var n_nodes = len(re.node_pos) // 3
    var n_fp = len(re.face_to_elem) // 4
    for f in range(4):
        for l in range(n_fp):
            var e = Int(re.face_to_elem[f * n_fp + l])
            if e < 0 or e >= n_nodes:
                raise Error(
                    "face_to_elem["
                    + String(f)
                    + ","
                    + String(l)
                    + "] = "
                    + String(e)
                    + " is out of range"
                )
    print("  face_to_elem coverage OK")


def _test_node_weights_partition(re: ReferenceElement, P: Int, np: Int) raises:
    """The mass-matrix-weighted cell-mean weights must sum to 1
    (partition of unity over the reference tet of volume 1/6,
    normalised so cell_mean = sum_i w_i q_i is exact for any
    P=P Lagrange expansion).  At P=2 this is a load-bearing
    invariant -- the BJ limiter relies on it for conservation,
    and the vertex weights are -1/20 (NEGATIVE) while the
    edge-midpoint weights are 1/5; a regression that swapped to
    a uniform 1/N_P nodal average would silently break
    conservation but pass every existing rate gate.

    P=2 spot check uses the closed-form: w[vertex] = -1/20,
    w[edge_midpoint] = 1/5; vertices are nodes 0..3, edge
    midpoints are nodes 4..9.  At P>=2 we don't require
    positivity (vertex weight is negative even at P=2)."""
    if len(re.node_weights) != np:
        raise Error("node_weights length mismatch")
    var wsum: Float64 = 0.0
    for i in range(np):
        wsum += re.node_weights[i]
    var wsum_err = wsum - 1.0
    var a_wsum_err = wsum_err if wsum_err >= 0.0 else -wsum_err
    if a_wsum_err > 1.0e-12:
        raise Error(
            "P="
            + String(P)
            + ": node_weights sum "
            + String(wsum)
            + " != 1 (partition of unity broken)"
        )
    if P == 2:
        var v_target = -1.0 / 20.0
        var e_target = 1.0 / 5.0
        for i in range(4):
            var w = re.node_weights[i]
            var err = w - v_target
            var a_err = err if err >= 0.0 else -err
            if a_err > 1.0e-12:
                raise Error(
                    "P=2 vertex "
                    + String(i)
                    + " weight "
                    + String(w)
                    + " expected -1/20"
                )
        for i in range(4, 10):
            var w = re.node_weights[i]
            var err = w - e_target
            var a_err = err if err >= 0.0 else -err
            if a_err > 1.0e-12:
                raise Error(
                    "P=2 edge-midpoint "
                    + String(i)
                    + " weight "
                    + String(w)
                    + " expected 1/5"
                )
    print("  node_weights partition-of-unity OK (sum=", wsum, ")")


def main() raises:
    # Size sanity.
    for P in [1, 2, 3, 4, 5]:
        _test_sizes(P)

    # Operator sanity for each supported order.
    print("P=1 reference element...")
    var re1 = ReferenceElement[1]()
    _test_spd_mass_inverse(re1)
    _test_node_positions(re1, 1)
    _test_face_to_elem_coverage(re1)
    _test_node_weights_partition(re1, 1, num_tet_nodes(1))

    print("P=2 reference element...")
    var re2 = ReferenceElement[2]()
    _test_spd_mass_inverse(re2)
    _test_node_positions(re2, 2)
    _test_face_to_elem_coverage(re2)
    _test_node_weights_partition(re2, 2, num_tet_nodes(2))

    print("P=3 reference element...")
    var re3 = ReferenceElement[3]()
    _test_spd_mass_inverse(re3)
    _test_node_positions(re3, 3)
    _test_face_to_elem_coverage(re3)
    _test_node_weights_partition(re3, 3, num_tet_nodes(3))

    print("P=4 reference element...")
    var re4 = ReferenceElement[4]()
    _test_spd_mass_inverse(re4)
    _test_node_positions(re4, 4)
    _test_face_to_elem_coverage(re4)
    _test_node_weights_partition(re4, 4, num_tet_nodes(4))

    print("P=5 reference element...")
    var re5 = ReferenceElement[5]()
    _test_spd_mass_inverse(re5)
    _test_node_positions(re5, 5)
    _test_face_to_elem_coverage(re5)
    _test_node_weights_partition(re5, 5, num_tet_nodes(5))

    print("=== reference_element_test PASSED ===")
