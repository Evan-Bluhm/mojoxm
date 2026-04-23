# ======================================================================
# convergence_2d_test -- spatial-convergence rate for 2D DG advection
# ======================================================================
#
# Integrates a smooth sinusoidal initial condition under periodic
# advection and compares the numerical L2 error to the exact solution
# at two mesh resolutions.  For a P-th order DG scheme on smooth data,
# the L2 error is expected to scale as h^(P+1); doubling N should
# shrink L2 by roughly 2^(P+1).  We don't require exact rates (pre-
# asymptotic regime, Rusanov dissipation, temporal error all pollute)
# but DO require a meaningful rate of convergence (> 1.5 between P=1
# successive refinements; > 2.5 for P=2).
#
# The test is intentionally short on steps (T=0.05) so the temporal
# error from SSPRK3 (3rd-order in time) stays below the spatial error
# for every P tested.
# ======================================================================

from std.math import sqrt, log2, sin, pi
from src.local_mesh_2d import LocalMesh2D
from src.reference_2d import ReferenceElement2D, num_tri_nodes_2d
from src.dg_rhs_2d import Advection2D, ssprk3_step_2d


def abs_f(x: Float64) -> Float64:
    return x if x >= 0.0 else -x


def _sin_ic(mesh: LocalMesh2D, n_nodes: Int, P: Int,
            shift_x: Float64, shift_y: Float64) raises -> List[Float64]:
    var q = List[Float64]()
    var NP_p = num_tri_nodes_2d(P)
    var two_pi = 2.0 * pi
    for elem in range(mesh.num_elements):
        for nn in range(NP_p):
            var x = mesh.elem_node_xyz[(elem * NP_p + nn) * 2 + 0]
            var y = mesh.elem_node_xyz[(elem * NP_p + nn) * 2 + 1]
            q.append(
                sin(two_pi * (x - shift_x))
                + sin(two_pi * (y - shift_y))
            )
    return q^


def _l2(a: List[Float64], b: List[Float64]) -> Float64:
    var s: Float64 = 0.0
    for k in range(len(a)):
        var d = a[k] - b[k]
        s += d * d
    return sqrt(s / Float64(len(a)))


def run_one[P: Int](Nx: Int, vx: Float64, vy: Float64, T: Float64,
                   cfl: Float64) raises -> Float64:
    var NP_p = num_tri_nodes_2d(P)
    var mesh = LocalMesh2D[P](Nx, Nx, 1.0, 1.0)
    var re = ReferenceElement2D[P]()

    var q = _sin_ic(mesh, mesh.num_elements * NP_p, P, 0.0, 0.0)
    var n = len(q)
    var s1 = List[Float64]()
    var s2 = List[Float64]()
    var s3 = List[Float64]()
    for _ in range(n):
        s1.append(0.0)
        s2.append(0.0)
        s3.append(0.0)

    var physics = Advection2D(vx, vy)
    var h = 1.0 / Float64(Nx)
    var dt_est = cfl * h / (sqrt(vx * vx + vy * vy) * Float64(2 * P + 1))
    var num_steps = Int(T / dt_est) + 1
    var dt = T / Float64(num_steps)

    for _ in range(num_steps):
        ssprk3_step_2d[P, Advection2D](
            mesh, re, physics, dt,
            q, s1, s2, s3,
        )

    # Exact solution: initial sin shifted by (vx*T, vy*T).
    var q_exact = _sin_ic(mesh, n, P, vx * T, vy * T)
    return _l2(q, q_exact)


def main() raises:
    print("convergence_2d_test -- spatial DG-P convergence")
    comptime vx = 1.0
    comptime vy = 0.5
    comptime T = 0.05
    comptime cfl = 0.15

    # P=1 across refinements.
    print("  P=1:")
    var e1_16 = run_one[1](16, vx, vy, T, cfl)
    var e1_32 = run_one[1](32, vx, vy, T, cfl)
    var rate1 = log2(e1_16 / e1_32)
    print("    N=16 -> L2=", e1_16)
    print("    N=32 -> L2=", e1_32)
    print("    observed rate (expect ~2):", rate1)
    if not (rate1 > 1.5):
        raise Error(
            "P=1 convergence rate " + String(rate1) + " below 1.5"
        )

    # P=2.
    print("  P=2:")
    var e2_16 = run_one[2](16, vx, vy, T, cfl)
    var e2_32 = run_one[2](32, vx, vy, T, cfl)
    var rate2 = log2(e2_16 / e2_32)
    print("    N=16 -> L2=", e2_16)
    print("    N=32 -> L2=", e2_32)
    print("    observed rate (expect ~3):", rate2)
    if not (rate2 > 2.5):
        raise Error(
            "P=2 convergence rate " + String(rate2) + " below 2.5"
        )

    # P=3 (only two moderate sizes -- pre-asymptotic concerns above).
    print("  P=3:")
    var e3_8 = run_one[3](8, vx, vy, T, cfl)
    var e3_16 = run_one[3](16, vx, vy, T, cfl)
    var rate3 = log2(e3_8 / e3_16)
    print("    N=8  -> L2=", e3_8)
    print("    N=16 -> L2=", e3_16)
    print("    observed rate (expect ~4):", rate3)
    if not (rate3 > 3.0):
        raise Error(
            "P=3 convergence rate " + String(rate3) + " below 3.0"
        )

    print("=== convergence_2d_test PASSED ===")
