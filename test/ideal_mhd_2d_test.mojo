# ======================================================================
# ideal_mhd_2d_test -- 2D ideal MHD sanity
# ======================================================================
#
#   1. Constant uniform state preservation.  (rho, u, v, Bx, By, p) =
#      (1, 0, 0, 0.5, 0.3, 1) has zero spatial gradient, so on a
#      periodic mesh the DG rhs must vanish to roundoff.  Same at-rest
#      property the Euler / SW tests check.
#
#   2. Uniform flow with a transverse magnetic field: (rho, u, v, Bx,
#      By) = (1, 0.3, 0.0, 0.0, 0.2), p = 1.  B_n = 0 at every face
#      perpendicular to the flow, so there's no Lorentz force; the
#      state advects without distortion.  rhs ~ 0 to roundoff.
#
#   3. MHD wall: set Bx initially -0.3 everywhere, u=v=0; with BC_WALL
#      reflecting both velocity AND B along the normal, the
#      configuration must remain at rest (rhs = 0 everywhere).
#
# Host-only; runs under `mojo run` via `make test-ideal-mhd-2d`.
# ======================================================================

from src.local_mesh_2d import LocalMesh2D
from src.reference_2d import ReferenceElement2D, num_tri_nodes_2d
from src.boundary import BoundaryConditions2D, BC_WALL
from src.dg_rhs_2d import IdealMHD2D, dg_rhs_2d


def abs_f(x: Float64) -> Float64:
    return x if x >= 0.0 else -x


def check_uniform[P: Int](
    Nx: Int, Ny: Int,
    rho: Float64, u: Float64, v: Float64,
    Bx: Float64, By: Float64, p: Float64,
    walls: Bool = False,
) raises:
    var NP_p = num_tri_nodes_2d(P)
    var mesh: LocalMesh2D[P]
    if walls:
        var bcs = BoundaryConditions2D(BC_WALL, BC_WALL, BC_WALL, BC_WALL)
        mesh = LocalMesh2D[P](Nx, Ny, 1.0, 1.0, bcs)
    else:
        mesh = LocalMesh2D[P](Nx, Ny, 1.0, 1.0)
    var re = ReferenceElement2D[P]()

    var gamma = 5.0 / 3.0
    var E = p / (gamma - 1.0) + 0.5 * rho * (u*u + v*v) + 0.5 * (Bx*Bx + By*By)
    var q = List[Float64]()
    for _ in range(mesh.num_elements * NP_p):
        q.append(rho)
        q.append(rho * u)
        q.append(rho * v)
        q.append(Bx)
        q.append(By)
        q.append(E)

    var rhs = List[Float64]()
    for _ in range(len(q)):
        rhs.append(0.0)

    var physics = IdealMHD2D(gamma, 1.0e-6, 1.0e-6)
    dg_rhs_2d[P, IdealMHD2D](mesh, re, physics, q, rhs)

    var max_err: Float64 = 0.0
    for k in range(len(rhs)):
        var a = abs_f(rhs[k])
        if a > max_err:
            max_err = a
    print("    (rho, u, v, Bx, By, p, walls) = (",
          rho, ",", u, ",", v, ",", Bx, ",", By, ",", p, ",", walls, ")",
          "max |rhs| =", max_err)
    if not (max_err < 1.0e-8):
        raise Error("at-rest MHD: max |rhs| = " + String(max_err))


def main() raises:
    print("ideal_mhd_2d_test")
    print("  uniform state (static field)")
    check_uniform[2](6, 6, 1.0, 0.0, 0.0, 0.5, 0.3, 1.0)
    print("  uniform flow with aligned B")
    check_uniform[2](6, 6, 1.0, 0.3, 0.0, 0.2, 0.0, 1.0)
    print("  weak-B uniform flow")
    check_uniform[3](4, 4, 1.0, 0.1, 0.1, 0.05, 0.05, 1.0)
    # Walled MHD with fully-consistent IC (uniform rho, p, zero velocity,
    # zero B) -- B = 0 is the only configuration that satisfies
    # "tangential at every wall of a rectangle" on both axes.
    print("  quiescent plasma with walls (B = 0)")
    check_uniform[2](4, 4, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, True)
    print("=== ideal_mhd_2d_test PASSED ===")
