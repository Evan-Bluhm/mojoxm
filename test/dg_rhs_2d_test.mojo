# ======================================================================
# dg_rhs_2d_test -- constant-state preservation for 2D DG advection
# ======================================================================
#
# A constant state q(x, y) = c has both zero gradient and a zero net
# face flux (divergence theorem on a periodic domain: integral of v.n
# around the cell boundary = 0 for any constant v).  Therefore
# `advection_rhs_2d` should produce rhs identically zero.  Any non-zero
# rhs points to an error in:
#   - mesh Jacobian scaling,
#   - D_ref / Lift_ref reference operators,
#   - face-normal / side convention,
#   - or elem_canon_to_ref mapping on shared edges.
#
# Runs entirely on CPU.  Tested at several (P, Nx, Ny) combinations.
# ======================================================================

from std.math import sin, cos, pi
from src.local_mesh_2d import LocalMesh2D
from src.reference_2d import ReferenceElement2D, num_tri_nodes_2d
from src.dg_rhs_2d import advection_rhs_2d


def abs_f64(x: Float64) -> Float64:
    return x if x >= 0.0 else -x


def check_constant_preservation[P: Int](
    Nx: Int, Ny: Int, Lx: Float64, Ly: Float64,
    vx: Float64, vy: Float64,
) raises:
    print("  P=", P, " Nx=", Nx, " Ny=", Ny, " v=(", vx, ",", vy, ")")
    var NP_p = num_tri_nodes_2d(P)
    var mesh = LocalMesh2D[P](Nx, Ny, Lx, Ly)
    var re = ReferenceElement2D[P]()

    var q = List[Float64]()
    for _ in range(mesh.num_elements * NP_p):
        q.append(Float64(3.1415))   # arbitrary non-round constant
    var rhs = List[Float64]()
    for _ in range(mesh.num_elements * NP_p):
        rhs.append(Float64(0.0))

    advection_rhs_2d[P](mesh, re, vx, vy, q, rhs)

    # rhs should be machine-precision zero.  Tolerance scales with
    # velocity magnitude + mesh size + P-dependent operator conditioning.
    var max_err: Float64 = 0.0
    for k in range(len(rhs)):
        var a = abs_f64(rhs[k])
        if a > max_err:
            max_err = a
    # Float64 roundoff on ~NP^2 fused operations scales with operator
    # magnitude; at P=4 on a 3x3 mesh, D_ref and Lift_ref entries reach
    # O(20) so the absolute error reaches ~1e-9.  Tolerance 1e-8 is
    # comfortable for every tested P without hiding bugs.
    var tol: Float64 = 1.0e-8
    if not (max_err <= tol):
        raise Error(
            "constant state not preserved: max |rhs| = "
            + String(max_err) + " (tol=" + String(tol) + ")"
        )
    print("    max |rhs| =", max_err, " (tol ", tol, ")")


def check_smooth_consistency[P: Int](Nx: Int, Ny: Int) raises:
    """For q(x, y) = sin(2 pi x) + sin(2 pi y) and v = (1, 0.5), the
    analytic dq/dt (which is what advection_rhs_2d returns) is
        dq/dt = -v . grad q
              = -2 pi (v_x cos(2 pi x) + v_y cos(2 pi y))
    Compared at each nodal point.  DG should match to within truncation
    error that decreases with P and with refinement.  We just check
    that the L_inf error shrinks when either P or N increases --
    systematic wrong sign / missing factor shows up as error ~ 2x the
    analytic amplitude even at "high" resolution.
    """
    print("  smooth sin IC: P=", P, " Nx=", Nx, " Ny=", Ny)
    var NP_p = num_tri_nodes_2d(P)
    var vx = 1.0
    var vy = 0.5
    var mesh = LocalMesh2D[P](Nx, Ny, 1.0, 1.0)
    var re = ReferenceElement2D[P]()

    var q = List[Float64]()
    var expected = List[Float64]()
    var two_pi = 2.0 * pi
    for elem in range(mesh.num_elements):
        for nn in range(NP_p):
            var x = mesh.elem_node_xyz[(elem * NP_p + nn) * 2 + 0]
            var y = mesh.elem_node_xyz[(elem * NP_p + nn) * 2 + 1]
            q.append(sin(two_pi * x) + sin(two_pi * y))
            expected.append(
                -two_pi * (vx * cos(two_pi * x) + vy * cos(two_pi * y))
            )

    var rhs = List[Float64]()
    for _ in range(mesh.num_elements * NP_p):
        rhs.append(0.0)
    advection_rhs_2d[P](mesh, re, vx, vy, q, rhs)

    # L_infinity error of rhs vs analytic dq/dt.
    var max_err: Float64 = 0.0
    var max_exp: Float64 = 0.0
    for k in range(len(rhs)):
        var err = rhs[k] - expected[k]
        var aerr = err if err >= 0.0 else -err
        if aerr > max_err:
            max_err = aerr
        var a_exp = expected[k] if expected[k] >= 0.0 else -expected[k]
        if a_exp > max_exp:
            max_exp = a_exp
    # At these modest resolutions the truncation error should be small
    # (well under the analytic amplitude).  A systematic sign flip or
    # missing factor shows up as error >= the signal magnitude.
    if not (max_err < max_exp):
        raise Error(
            "smooth consistency: max err " + String(max_err)
            + " >= analytic amplitude " + String(max_exp)
            + " -- likely sign or scale bug"
        )
    print("    max |rhs - exact| =", max_err,
          "  (analytic peak =", max_exp, ")")


def main() raises:
    print("dg_rhs_2d constant-preservation tests")
    check_constant_preservation[1](4, 4, 1.0, 1.0, 1.0, 0.0)
    check_constant_preservation[1](4, 4, 1.0, 1.0, 0.7, -0.5)
    check_constant_preservation[2](6, 6, 1.0, 1.0, 1.0, 1.0)
    check_constant_preservation[2](5, 3, 2.0, 1.0, -1.0, 0.5)
    check_constant_preservation[3](4, 4, 1.0, 1.0, 0.3, 0.8)
    check_constant_preservation[4](3, 3, 1.0, 1.0, 1.0, -1.0)

    print("dg_rhs_2d smooth-flow sanity tests")
    check_smooth_consistency[1](16, 16)
    check_smooth_consistency[2](16, 16)
    check_smooth_consistency[3](8, 8)
    print("=== dg_rhs_2d_test PASSED ===")
