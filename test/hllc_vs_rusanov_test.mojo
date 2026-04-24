# ======================================================================
# hllc_vs_rusanov_test -- Euler2D HLLC vs Rusanov L2 comparison
# ======================================================================
#
# Runs the Shu-Erlebacher isentropic vortex for one full period on a
# periodic [0, 10]^2 mesh with both Euler2D.use_hllc = False (Rusanov
# default) and True (HLLC).  After one period the exact solution
# equals the IC, so the L2 error versus the IC is pure scheme
# dissipation.  HLLC must come in tighter than Rusanov at equal
# resolution / order -- that's the whole point of adopting it.
#
# Host-only (Float64).  Runs in `make test-hllc`.
# ======================================================================

from std.math import sqrt, exp, pi
from src.local_mesh_2d import LocalMesh2D
from src.reference_2d import ReferenceElement2D, num_tri_nodes_2d
from src.dg_rhs_2d import Euler2D, ssprk3_step_2d


# Low order + modest resolution so flux dissipation dominates the
# error budget (at P=2 / 24x24 the two fluxes agree to ~0.1% on the
# smooth vortex -- scheme dissipation is mostly projection error, not
# flux dissipation).  At P=1 / 16x16 the mesh is underresolved enough
# that Rusanov's wider wave fan genuinely hurts.
comptime P = 1
comptime NX = 16
comptime NY = 16
comptime LX = 10.0
comptime LY = 10.0
comptime T_FINAL = 10.0
comptime CFL = 0.15

comptime GAMMA = 1.4
comptime T_INF = 1.0
comptime U0 = 1.0
comptime V0 = 1.0
comptime BETA = 5.0
comptime CX0 = 5.0
comptime CY0 = 5.0


def _pdelta(a: Float64, b: Float64, L: Float64) -> Float64:
    var d = a - b
    if d >  L * 0.5: d -= L
    if d < -L * 0.5: d += L
    return d


def _run_one_period(use_hllc: Bool) raises -> Float64:
    var NP_p = num_tri_nodes_2d(P)
    var mesh = LocalMesh2D[P](NX, NY, LX, LY)
    var re = ReferenceElement2D[P]()

    var n = mesh.num_elements * NP_p * 4
    var q = List[Float64]()
    var q_ic = List[Float64]()
    var two_pi = 2.0 * pi
    var factor = (GAMMA - 1.0) * BETA * BETA / (8.0 * GAMMA * two_pi * two_pi)

    for elem in range(mesh.num_elements):
        for nn in range(NP_p):
            var x = mesh.elem_node_xyz[(elem * NP_p + nn) * 2 + 0]
            var y = mesh.elem_node_xyz[(elem * NP_p + nn) * 2 + 1]
            var dx = _pdelta(x, CX0, LX)
            var dy = _pdelta(y, CY0, LY)
            var r2 = dx * dx + dy * dy
            var T = T_INF - factor * exp(1.0 - r2)
            var e_half = exp(0.5 * (1.0 - r2))
            var u = U0 - (BETA / two_pi) * dy * e_half
            var v = V0 + (BETA / two_pi) * dx * e_half
            var rho = T ** (1.0 / (GAMMA - 1.0))
            var p = rho * T
            var E = p / (GAMMA - 1.0) + 0.5 * rho * (u * u + v * v)
            q.append(rho);      q_ic.append(rho)
            q.append(rho * u);  q_ic.append(rho * u)
            q.append(rho * v);  q_ic.append(rho * v)
            q.append(E);        q_ic.append(E)

    var s_q1 = List[Float64]()
    var s_q2 = List[Float64]()
    var s_rhs = List[Float64]()
    for _ in range(n):
        s_q1.append(0.0)
        s_q2.append(0.0)
        s_rhs.append(0.0)

    var physics = Euler2D(
        GAMMA, 1.0e-6, 1.0e-6,
        0.0, 0.0, 0.0, 0.0,    # inflow unused on periodic mesh
        use_hllc,
    )
    var c_inf = sqrt(GAMMA * T_INF)
    var wave_max = sqrt(U0 * U0 + V0 * V0) + c_inf + BETA / two_pi
    var h_cell = LX / Float64(NX)
    var dt_est = CFL * h_cell / (wave_max * Float64(2 * P + 1))
    var num_steps = Int(T_FINAL / dt_est) + 1
    var dt = T_FINAL / Float64(num_steps)

    for _ in range(num_steps):
        ssprk3_step_2d[P, Euler2D](
            mesh, re, physics, dt, q, s_q1, s_q2, s_rhs,
        )

    var sum_sq: Float64 = 0.0
    var sum_ic: Float64 = 0.0
    for k in range(n):
        var e = q[k] - q_ic[k]
        sum_sq += e * e
        sum_ic += q_ic[k] * q_ic[k]
    return sqrt(sum_sq / Float64(n)) / sqrt(sum_ic / Float64(n))


def main() raises:
    print("hllc_vs_rusanov_test -- vortex one-period rel L2")
    print("  P=", P, "  ", NX, "x", NY, "  T=", T_FINAL)

    var rel_rusanov = _run_one_period(False)
    print("  Rusanov rel L2 =", rel_rusanov)
    var rel_hllc = _run_one_period(True)
    print("  HLLC    rel L2 =", rel_hllc)
    print("  ratio HLLC / Rusanov =", rel_hllc / rel_rusanov)

    if rel_hllc >= rel_rusanov:
        raise Error(
            "HLLC error "
            + String(rel_hllc)
            + " is not lower than Rusanov "
            + String(rel_rusanov)
        )
    print("=== hllc_vs_rusanov_test PASSED ===")
