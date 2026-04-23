# ======================================================================
# euler2d_test -- 2D Euler physics end-to-end sanity
# ======================================================================
#
# Two checks against `Euler2D`:
#
#   1. Constant-state preservation.  A uniform (rho, u, v, p) = (1.2,
#      0.3, -0.4, 1.5) state has zero conservative divergence and
#      numerical fluxes cancel around every cell (divergence theorem).
#      The DG rhs must therefore produce ~0 to within roundoff, same
#      guarantee as the scalar advection test.
#
#   2. Translation of a Gaussian density bump by a uniform flow.  Runs
#      SSPRK3 for a short time under v = (1, 0.5) and checks that the
#      bump's centroid moves by (vx * t, vy * t) to within a small
#      tolerance.  Crude but catches gross physics bugs (wrong sign of
#      flux, wrong gamma handling, etc.).
#
# Host-only (CPU Float64).
# ======================================================================

from std.math import sqrt, exp
from src.local_mesh_2d import LocalMesh2D
from src.reference_2d import ReferenceElement2D, num_tri_nodes_2d
from src.dg_rhs_2d import Euler2D, dg_rhs_2d, ssprk3_step_2d


def abs_f(x: Float64) -> Float64:
    return x if x >= 0.0 else -x


def check_constant[P: Int](Nx: Int, Ny: Int) raises:
    print("  constant state: P=", P, " Nx=", Nx, " Ny=", Ny)
    var NP_p = num_tri_nodes_2d(P)
    var mesh = LocalMesh2D[P](Nx, Ny, 1.0, 1.0)
    var re = ReferenceElement2D[P]()

    # Fill with (rho, rho u, rho v, E) corresponding to rho=1.2, u=0.3,
    # v=-0.4, p=1.5 at gamma=1.4.
    var rho = 1.2
    var u = 0.3
    var v = -0.4
    var p = 1.5
    var gamma = 1.4
    var ke = 0.5 * rho * (u * u + v * v)
    var E = p / (gamma - 1.0) + ke
    var cons = List[Float64]()
    cons.append(rho)
    cons.append(rho * u)
    cons.append(rho * v)
    cons.append(E)

    var n_total = mesh.num_elements * NP_p * 4
    var q = List[Float64]()
    var rhs = List[Float64]()
    for _ in range(mesh.num_elements * NP_p):
        for c in range(4):
            q.append(cons[c])
    for _ in range(n_total):
        rhs.append(0.0)

    var physics = Euler2D(gamma, 1.0e-6, 1.0e-6)
    dg_rhs_2d[P, Euler2D](mesh, re, physics, q, rhs)

    var max_err: Float64 = 0.0
    for k in range(len(rhs)):
        var a = abs_f(rhs[k])
        if a > max_err:
            max_err = a
    # Rusanov adds no dissipation to a constant state (q_r == q_l so
    # the jump term is zero); expect ~ Float64 roundoff scaled by
    # operator magnitude.
    var tol = 1.0e-8
    if not (max_err < tol):
        raise Error(
            "Euler2D const state: max |rhs| = " + String(max_err)
            + " (tol " + String(tol) + ")"
        )
    print("    max |rhs| =", max_err)


def check_translation[P: Int](Nx: Int, Ny: Int) raises:
    print("  Gaussian translation: P=", P, " Nx=", Nx, " Ny=", Ny)
    var NP_p = num_tri_nodes_2d(P)
    var mesh = LocalMesh2D[P](Nx, Ny, 1.0, 1.0)
    var re = ReferenceElement2D[P]()

    var rho_bg = 1.0
    var rho_amp = 0.1
    var u_bg = 1.0
    var v_bg = 0.5
    var p_bg = 1.0
    var gamma = 1.4
    var sigma = 0.1
    var cx_0 = 0.3
    var cy_0 = 0.3
    var T = 0.1    # short time to stay within linear regime

    # Fill IC: rho = rho_bg + rho_amp * gauss; u = u_bg; v = v_bg; p = p_bg.
    # Energy from ideal gas.
    var q = List[Float64]()
    for elem in range(mesh.num_elements):
        for nn in range(NP_p):
            var x = mesh.elem_node_xyz[(elem * NP_p + nn) * 2 + 0]
            var y = mesh.elem_node_xyz[(elem * NP_p + nn) * 2 + 1]
            var dx = x - cx_0
            var dy = y - cy_0
            var g = exp(-(dx * dx + dy * dy) / (2.0 * sigma * sigma))
            var rho = rho_bg + rho_amp * g
            var E = p_bg / (gamma - 1.0) + 0.5 * rho * (
                u_bg * u_bg + v_bg * v_bg
            )
            q.append(rho)
            q.append(rho * u_bg)
            q.append(rho * v_bg)
            q.append(E)

    var n = len(q)
    var s_q1 = List[Float64]()
    var s_q2 = List[Float64]()
    var s_rhs = List[Float64]()
    for _ in range(n):
        s_q1.append(0.0)
        s_q2.append(0.0)
        s_rhs.append(0.0)

    var physics = Euler2D(gamma, 1.0e-6, 1.0e-6)
    # CFL limit for Euler: dt * (|v| + c) / h <= C_P.  c ~ sqrt(1.4) ~ 1.18;
    # |v| ~ 1.12.  Use conservative CFL 0.15.
    var h = 1.0 / Float64(Nx)
    var c_sound = sqrt(gamma * p_bg / rho_bg)
    var wave_max = sqrt(u_bg * u_bg + v_bg * v_bg) + c_sound
    var cfl = 0.15
    var dt = cfl * h / (wave_max * Float64(2 * P + 1))
    var num_steps = Int(T / dt) + 1
    var used_dt = T / Float64(num_steps)

    for _ in range(num_steps):
        ssprk3_step_2d[P, Euler2D](
            mesh, re, physics, used_dt,
            q, s_q1, s_q2, s_rhs,
        )

    # Find the (x, y) centroid of the density perturbation (rho - rho_bg).
    var sum_w: Float64 = 0.0
    var sum_x: Float64 = 0.0
    var sum_y: Float64 = 0.0
    for elem in range(mesh.num_elements):
        for nn in range(NP_p):
            var x = mesh.elem_node_xyz[(elem * NP_p + nn) * 2 + 0]
            var y = mesh.elem_node_xyz[(elem * NP_p + nn) * 2 + 1]
            var rho = q[(elem * NP_p + nn) * 4 + 0]
            var w = rho - rho_bg
            if w > 0.0:
                sum_w += w
                sum_x += w * x
                sum_y += w * y
    var cx_final = sum_x / sum_w
    var cy_final = sum_y / sum_w
    var cx_expect = cx_0 + u_bg * T
    var cy_expect = cy_0 + v_bg * T
    var err_x = abs_f(cx_final - cx_expect)
    var err_y = abs_f(cy_final - cy_expect)
    print("    centroid got (", cx_final, ",", cy_final,
          "), expected (", cx_expect, ",", cy_expect,
          "), err (", err_x, ",", err_y, ")")
    # 3% of the domain size is a generous tolerance given Rusanov's
    # first-order dissipation on a coarse mesh.
    if not (err_x < 0.03 and err_y < 0.03):
        raise Error(
            "centroid translation off by more than 3% of L"
        )


def main() raises:
    print("euler2d_test")
    check_constant[1](4, 4)
    check_constant[2](5, 3)
    check_constant[3](4, 4)

    check_translation[2](16, 16)
    check_translation[3](12, 12)

    print("=== euler2d_test PASSED ===")
