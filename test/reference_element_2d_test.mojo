# ======================================================================
# reference_element_2d_test -- ReferenceElement2D math validation
# ======================================================================
#
# What we check at each P in {1, 2, 3, 4}:
#
#   1. node count matches num_tri_nodes_2d(P) = (P+1)(P+2)/2.
#   2. every node has barycentric coordinates in [0, 1]: each (r, s)
#      is inside the reference simplex r + s <= 1.
#   3. Mass matrix M_ref = (M_ref_inv)^-1 is symmetric positive-definite
#      via Cholesky.
#   4. Edge-to-element map covers every edge node: 3 edges * (P+1)
#      entries all in [0, NP).
#
# Host-only: no GPU, no MPI.  Runs as `mojo run test/reference_element_2d_test.mojo`.
# ======================================================================

from src.reference import mat_inv
from src.reference_2d import (
    ReferenceElement2D,
    num_tri_nodes_2d,
    num_edge_nodes,
)


def _cholesky_succeeds(m: List[Float64], n: Int) raises -> Bool:
    """In-place Cholesky of an SPD matrix -- returns True if it completes
    without encountering a non-positive pivot."""
    var L = List[Float64]()
    for _ in range(n * n):
        L.append(0.0)
    for i in range(n):
        for j in range(i + 1):
            var s: Float64 = m[i * n + j]
            for k in range(j):
                s -= L[i * n + k] * L[j * n + k]
            if i == j:
                if s <= 0.0:
                    return False
                L[i * n + j] = s**0.5
            else:
                L[i * n + j] = s / L[j * n + j]
    return True


def check[P: Int]() raises:
    print("P=", P, " reference element (2D triangle)...")
    var NP_P = num_tri_nodes_2d(P)
    var NFP_edge = num_edge_nodes(P)
    var re = ReferenceElement2D[P]()

    # 1. Node count.
    if len(re.node_pos) != NP_P * 2:
        raise Error("node_pos length mismatch")
    if len(re.M_ref_inv) != NP_P * NP_P:
        raise Error("M_ref_inv length mismatch")
    if len(re.D_ref) != 2 * NP_P * NP_P:
        raise Error("D_ref length mismatch")
    if len(re.Lift_ref) != 3 * NP_P * NFP_edge:
        raise Error("Lift_ref length mismatch")
    if len(re.edge_to_elem) != 3 * NFP_edge:
        raise Error("edge_to_elem length mismatch")

    # 2. Nodes inside reference simplex.
    for i in range(NP_P):
        var r = re.node_pos[i * 2 + 0]
        var s = re.node_pos[i * 2 + 1]
        if r < -1.0e-12 or s < -1.0e-12 or r + s > 1.0 + 1.0e-12:
            raise Error(
                "node "
                + String(i)
                + " outside reference simplex: r="
                + String(r)
                + " s="
                + String(s)
            )
    print("  node positions OK (", NP_P, "nodes)")

    # 3. M_ref is SPD.  We have M_ref_inv; M_ref = (M_ref_inv)^-1.
    var M_ref = mat_inv(re.M_ref_inv, NP_P)
    if not _cholesky_succeeds(M_ref, NP_P):
        raise Error("M_ref is not SPD at this P")
    print("  mass matrix SPD at NP=", NP_P)

    # 4. Edge-to-element map coverage: every entry in [0, NP).
    for i in range(len(re.edge_to_elem)):
        var v = Int(re.edge_to_elem[i])
        if v < 0 or v >= NP_P:
            raise Error(
                "edge_to_elem["
                + String(i)
                + "] = "
                + String(v)
                + " out of range [0, "
                + String(NP_P)
                + ")"
            )
    # Spot-check vertex entries: edge 0 (v0, v1) must start at node 0,
    # end at node 1.  Edge 1 (v1, v2) starts at 1, ends at 2.  Edge 2
    # (v2, v0) starts at 2, ends at 0.
    var vstart = List[Int]()
    vstart.append(0)
    vstart.append(1)
    vstart.append(2)
    var vend = List[Int]()
    vend.append(1)
    vend.append(2)
    vend.append(0)
    for e in range(3):
        if Int(re.edge_to_elem[e * NFP_edge + 0]) != vstart[e]:
            raise Error(
                "edge " + String(e) + " start != vertex " + String(vstart[e])
            )
        if Int(re.edge_to_elem[e * NFP_edge + (NFP_edge - 1)]) != vend[e]:
            raise Error(
                "edge " + String(e) + " end != vertex " + String(vend[e])
            )
    print("  edge-to-element map OK (", 3 * NFP_edge, "entries)")

    # 5. Cell-mean node weights must sum to 1 (partition of unity)
    # and for P=2 match the standard equispaced-Lagrange rule
    # exactly (vertex weights 0, edge-midpoint weights 1/3).  At
    # P=4+ equispaced Lagrange bases have some negative weights --
    # a known property, not a bug -- so we only require partition
    # of unity universally and leave positivity as a P<=3 property.
    if len(re.node_weights) != NP_P:
        raise Error("node_weights length mismatch")
    var wsum: Float64 = 0.0
    for i in range(NP_P):
        wsum += re.node_weights[i]
    var wsum_err = wsum - 1.0
    var a_wsum_err = wsum_err if wsum_err >= 0.0 else -wsum_err
    if a_wsum_err > 1.0e-12:
        raise Error(
            "node_weights sum " + String(wsum) + " != 1 (partition of unity)"
        )
    if P <= 3:
        for i in range(NP_P):
            var w = re.node_weights[i]
            if w < -1.0e-12:
                raise Error(
                    "P<=3 node_weights["
                    + String(i)
                    + "] = "
                    + String(w)
                    + " is negative"
                )
    # P=2-specific spot check -- exact Lagrange-P=2 quadrature.
    if P == 2:
        for i in range(3):
            var w = re.node_weights[i]
            var aw = w if w >= 0.0 else -w
            if aw > 1.0e-12:
                raise Error(
                    "P=2 vertex "
                    + String(i)
                    + " weight "
                    + String(w)
                    + " expected 0"
                )
        var third = 1.0 / 3.0
        for i in range(3, 6):
            var w = re.node_weights[i]
            var err = w - third
            var a_err = err if err >= 0.0 else -err
            if a_err > 1.0e-12:
                raise Error(
                    "P=2 midpoint "
                    + String(i)
                    + " weight "
                    + String(w)
                    + " expected 1/3"
                )
    print("  node_weights OK (sum=", wsum, ")")


def main() raises:
    print("sizes: num_tri_nodes_2d, num_edge_nodes")
    for P in range(1, 6):
        var NP_P = num_tri_nodes_2d(P)
        var NFP_edge = num_edge_nodes(P)
        print("  P=", P, " NP=", NP_P, " NFP_edge=", NFP_edge)

    check[1]()
    check[2]()
    check[3]()
    check[4]()
    check[5]()
    print("=== reference_element_2d_test PASSED ===")
