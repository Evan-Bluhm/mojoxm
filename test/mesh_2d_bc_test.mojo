# ======================================================================
# mesh_2d_bc_test -- non-periodic LocalMesh2D + Physics2D.boundary_flux
# ======================================================================
#
# Two invariants that must hold on a closed non-periodic 2D domain:
#
#   1. At rest under shallow-water walls: (h, u, v) = (1, 0, 0) with all
#      four boundaries set to BC_WALL.  For any gravity and any mesh,
#      rhs must be identically zero (gradient of everything is zero,
#      wall reflection makes face contributions cancel pair-wise).
#
#   2. At rest under Euler walls: (rho, u, v, p) = (1.2, 0, 0, 1.5)
#      in a box with BC_WALL everywhere.  Same argument -- rhs = 0.
#
# Additionally, we count the extra faces allocated by the overlay:
# the BC_WALL-on-all-sides mesh should have 3*Nx*Ny + Nx + Ny faces
# (periodic ring + Ny extras for -x + Nx extras for -y).
# ======================================================================

from src.local_mesh_2d import LocalMesh2D
from src.reference_2d import ReferenceElement2D, num_tri_nodes_2d
from src.boundary import BoundaryConditions2D, BC_WALL
from src.dg_rhs_2d import Euler2D, ShallowWater2D, dg_rhs_2d


def abs_f(x: Float64) -> Float64:
    return x if x >= 0.0 else -x


def check_face_count[P: Int](Nx: Int, Ny: Int) raises:
    print("  face count: P=", P, " Nx=", Nx, " Ny=", Ny)
    var bcs = BoundaryConditions2D(BC_WALL, BC_WALL, BC_WALL, BC_WALL)
    var mesh = LocalMesh2D[P](Nx, Ny, 1.0, 1.0, bcs)
    var expect = 3 * Nx * Ny + Nx + Ny
    if mesh.num_faces != expect:
        raise Error(
            "num_faces = " + String(mesh.num_faces)
            + " expected " + String(expect)
        )
    # Every BC face's bc_type must match BC_WALL.
    var n_wall_faces = 0
    for fid in range(mesh.num_faces):
        if mesh.face_bc_type[fid] == BC_WALL:
            n_wall_faces += 1
    # Walls: +x ring (Ny) + +y ring (Nx) + -x ring (Ny) + -y ring (Nx)
    # = 2 (Nx + Ny).
    var expect_walls = 2 * (Nx + Ny)
    if n_wall_faces != expect_walls:
        raise Error(
            "wall-face count = " + String(n_wall_faces)
            + " expected " + String(expect_walls)
        )
    print("    num_faces =", mesh.num_faces, " wall faces =",
          n_wall_faces)


def check_sw_rest[P: Int](Nx: Int, Ny: Int) raises:
    print("  SW at rest in a closed box: P=", P, " Nx=", Nx, " Ny=", Ny)
    var NP_p = num_tri_nodes_2d(P)
    var bcs = BoundaryConditions2D(BC_WALL, BC_WALL, BC_WALL, BC_WALL)
    var mesh = LocalMesh2D[P](Nx, Ny, 1.0, 1.0, bcs)
    var re = ReferenceElement2D[P]()
    var q = List[Float64]()
    for _ in range(mesh.num_elements * NP_p):
        q.append(1.0)  # h
        q.append(0.0)  # h u
        q.append(0.0)  # h v
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
    if not (max_err < 1.0e-8):
        raise Error(
            "SW rest in box: max |rhs| = " + String(max_err)
        )
    print("    max |rhs| =", max_err)


def check_euler_rest[P: Int](Nx: Int, Ny: Int) raises:
    print("  Euler at rest in a closed box: P=", P, " Nx=", Nx, " Ny=", Ny)
    var NP_p = num_tri_nodes_2d(P)
    var bcs = BoundaryConditions2D(BC_WALL, BC_WALL, BC_WALL, BC_WALL)
    var mesh = LocalMesh2D[P](Nx, Ny, 1.0, 1.0, bcs)
    var re = ReferenceElement2D[P]()
    var rho = 1.2
    var p = 1.5
    var gamma = 1.4
    var E = p / (gamma - 1.0)
    var q = List[Float64]()
    for _ in range(mesh.num_elements * NP_p):
        q.append(rho)
        q.append(0.0)
        q.append(0.0)
        q.append(E)
    var rhs = List[Float64]()
    for _ in range(len(q)):
        rhs.append(0.0)
    var physics = Euler2D(gamma, 1.0e-6, 1.0e-6)
    dg_rhs_2d[P, Euler2D](mesh, re, physics, q, rhs)
    var max_err: Float64 = 0.0
    for k in range(len(rhs)):
        var a = abs_f(rhs[k])
        if a > max_err:
            max_err = a
    if not (max_err < 1.0e-8):
        raise Error(
            "Euler rest in box: max |rhs| = " + String(max_err)
        )
    print("    max |rhs| =", max_err)


def main() raises:
    print("mesh_2d_bc_test")
    check_face_count[1](4, 4)
    check_face_count[2](5, 3)
    check_sw_rest[2](6, 6)
    check_euler_rest[2](6, 6)
    check_euler_rest[3](4, 4)
    print("=== mesh_2d_bc_test PASSED ===")
