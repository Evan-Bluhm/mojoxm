# ======================================================================
# shallow_water_2d_test -- 2D shallow water physics sanity
# ======================================================================
#
#   1. Constant-state preservation.  A lake-at-rest (h = 1, u = v = 0)
#      is the trivial steady solution for flat-bed shallow water.  DG
#      rhs must vanish to roundoff.
#
#   2. Momentum-conservation under a uniform flow.  (h, u, v) = (1,
#      0.5, -0.3) with constant state -- divergence of every flux is
#      zero and numerical fluxes cancel around each cell.  rhs ~ 0.
#
# Host-only (CPU Float64).
# ======================================================================

from src.local_mesh_2d import LocalMesh2D
from src.reference_2d import ReferenceElement2D, num_tri_nodes_2d
from src.dg_rhs_2d import ShallowWater2D, dg_rhs_2d


def abs_f(x: Float64) -> Float64:
    return x if x >= 0.0 else -x


def check_const[P: Int](
    Nx: Int, Ny: Int, h: Float64, u: Float64, v: Float64,
) raises:
    print("  P=", P, " (h, u, v) = (", h, ",", u, ",", v, ")")
    var NP_p = num_tri_nodes_2d(P)
    var mesh = LocalMesh2D[P](Nx, Ny, 1.0, 1.0)
    var re = ReferenceElement2D[P]()

    var q = List[Float64]()
    for _ in range(mesh.num_elements * NP_p):
        q.append(h)
        q.append(h * u)
        q.append(h * v)

    var rhs = List[Float64]()
    for _ in range(len(q)):
        rhs.append(0.0)

    var physics = ShallowWater2D(9.81, 1.0e-6)
    dg_rhs_2d[P, ShallowWater2D](mesh, re, physics, q, rhs)

    var max_err: Float64 = 0.0
    for k in range(len(rhs)):
        var a = abs_f(rhs[k])
        if a > max_err:
            max_err = a
    var tol = 1.0e-8
    if not (max_err < tol):
        raise Error(
            "SW2D const state: max |rhs| = " + String(max_err)
            + " (tol " + String(tol) + ")"
        )
    print("    max |rhs| =", max_err)


def main() raises:
    print("shallow_water_2d_test")
    check_const[1](4, 4, 1.0, 0.0, 0.0)         # lake at rest
    check_const[2](6, 6, 1.5, 0.5, -0.3)        # uniform flow
    check_const[3](4, 4, 2.0, -0.8, 0.2)        # reversed flow
    print("=== shallow_water_2d_test PASSED ===")
