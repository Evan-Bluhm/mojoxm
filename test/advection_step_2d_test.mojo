# ======================================================================
# advection_step_2d_test -- end-to-end 2D DG advection on CPU
# ======================================================================
#
# Drives `ssprk3_step_2d` on a 2D triangulated mesh and checks that a
# periodic Gaussian advection cycles back to itself after one full
# period of travel.  Concretely:
#
#   domain:     [0, 1]^2 with periodic wrap in both x and y
#   velocity:   (v_x, v_y) = (1, 1)
#   IC:         q(x, y, 0) = exp(-((x-cx)^2 + (y-cy)^2) / (2 sigma^2))
#               (with cx, cy chosen to stay away from seams for clean IC)
#   t_final:    1.0 (one full period in both axes simultaneously)
#   exact:      q(x, y, 1) = q(x, y, 0)
#
# Checks L2 norm of (q_final - q_initial).  DG errors shrink with P
# and N; at P=2 on a 16x16 mesh we expect ~1% L2 error.
#
# Host-only (CPU Float64); runs via `mojo run`.
# ======================================================================

from std.math import sqrt, exp, pi
from src.local_mesh_2d import LocalMesh2D
from src.reference_2d import ReferenceElement2D, num_tri_nodes_2d
from src.dg_rhs_2d import Advection2D, ssprk3_step_2d


comptime LX = 1.0
comptime LY = 1.0
comptime VX = 1.0
comptime VY = 1.0
comptime T_FINAL = 1.0
comptime CX = 0.5
comptime CY = 0.5
comptime SIGMA = 0.12


def _gauss(x: Float64, y: Float64) -> Float64:
    # Periodic nearest-image distance to (CX, CY).
    var dx = x - CX
    if dx >  LX * 0.5: dx -= LX
    if dx < -LX * 0.5: dx += LX
    var dy = y - CY
    if dy >  LY * 0.5: dy -= LY
    if dy < -LY * 0.5: dy += LY
    var sig2 = SIGMA * SIGMA
    return exp(-(dx * dx + dy * dy) / (2.0 * sig2))


def run[P: Int](Nx: Int, Ny: Int, cfl: Float64) raises:
    print("  P=", P, " Nx=", Nx, " Ny=", Ny, " cfl=", cfl)
    var NP_p = num_tri_nodes_2d(P)
    var mesh = LocalMesh2D[P](Nx, Ny, LX, LY)
    var re = ReferenceElement2D[P]()

    var n = mesh.num_elements * NP_p
    var q = List[Float64]()
    var q_ic = List[Float64]()
    for elem in range(mesh.num_elements):
        for nn in range(NP_p):
            var x = mesh.elem_node_xyz[(elem * NP_p + nn) * 2 + 0]
            var y = mesh.elem_node_xyz[(elem * NP_p + nn) * 2 + 1]
            var v = _gauss(x, y)
            q.append(v)
            q_ic.append(v)

    # Scratch buffers
    var s_q1 = List[Float64]()
    var s_q2 = List[Float64]()
    var s_rhs = List[Float64]()
    for _ in range(n):
        s_q1.append(0.0)
        s_q2.append(0.0)
        s_rhs.append(0.0)

    # Pick dt from CFL: dt = cfl * h / (|v| * (2P + 1)).  2P+1 scaling
    # mirrors the 3D drivers' empirical DG-on-tet CFL dependence.
    var h = LX / Float64(Nx)
    var vmag = sqrt(VX * VX + VY * VY)
    var dt = cfl * h / (vmag * Float64(2 * P + 1))
    var num_steps = Int(T_FINAL / dt) + 1
    var used_dt = T_FINAL / Float64(num_steps)
    print("    dt=", used_dt, " num_steps=", num_steps)

    var physics = Advection2D(VX, VY)
    for _ in range(num_steps):
        ssprk3_step_2d[P, Advection2D](
            mesh, re, physics, used_dt,
            q, s_q1, s_q2, s_rhs,
        )

    # L2 error against the IC.
    var sum_sq: Float64 = 0.0
    var sum_ic_sq: Float64 = 0.0
    for k in range(n):
        var e = q[k] - q_ic[k]
        sum_sq += e * e
        sum_ic_sq += q_ic[k] * q_ic[k]
    var l2_err = sqrt(sum_sq / Float64(n))
    var l2_ic = sqrt(sum_ic_sq / Float64(n))
    var rel = l2_err / l2_ic
    print("    L2 err =", l2_err, " rel =", rel,
          " (IC L2 =", l2_ic, ")")

    # Accept up to 10% relative L2 error on coarse meshes (this test is
    # about "does the pipeline work", not "how accurate is it").  A
    # broken sign / scale bug would yield errors near the IC magnitude.
    if not (rel < 0.1):
        raise Error(
            "L2 error " + String(rel) + " exceeds 10% at P=" + String(P)
            + " Nx=" + String(Nx)
        )


def main() raises:
    print("advection_step_2d_test -- one-period Gaussian translation")
    run[1](32, 32, 0.3)
    run[2](16, 16, 0.3)
    run[3](12, 12, 0.25)
    print("=== advection_step_2d_test PASSED ===")
