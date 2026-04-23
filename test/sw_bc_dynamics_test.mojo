# ======================================================================
# sw_bc_dynamics_test -- 2D shallow water with walls under evolution
# ======================================================================
#
# Integrates a shallow-water Gaussian bump on a closed (all-walls)
# basin for many RK steps.  At-rest preservation is already tested in
# mesh_2d_bc_test; this test stresses the dynamic path:
#
#   1. h stays strictly positive throughout (no dry patch / NaN from
#      bad wall flux).
#   2. h stays bounded (< 2 * h_max_IC) -- Gibbs overshoot is mild, no
#      runaway instability.
#   3. Volume (nodal-mean h) doesn't drift more than 1% -- a broad
#      tolerance since the limiter isn't on and nodal mean is an
#      approximate diagnostic, but enough to catch a gross conservation
#      bug in the BC-overlay face bookkeeping.
# ======================================================================

from std.math import sqrt, exp
from src.local_mesh_2d import LocalMesh2D
from src.reference_2d import ReferenceElement2D, num_tri_nodes_2d
from src.boundary import BoundaryConditions2D, BC_WALL
from src.dg_rhs_2d import ShallowWater2D, ssprk3_step_2d


def abs_f(x: Float64) -> Float64:
    return x if x >= 0.0 else -x


def run[P: Int]() raises:
    print("  P=", P)
    comptime Nx = 24
    comptime Ny = 24
    comptime Lx = 1.0
    comptime Ly = 1.0
    comptime T_final = 0.2
    comptime g = 9.81
    comptime h_bg = 1.0
    comptime h_amp = 0.3
    comptime sigma = 0.1
    comptime cx = 0.3   # placed off-centre so all four walls see it
    comptime cy = 0.3
    comptime cfl = 0.2

    var NP_p = num_tri_nodes_2d(P)
    var bcs = BoundaryConditions2D(BC_WALL, BC_WALL, BC_WALL, BC_WALL)
    var mesh = LocalMesh2D[P](Nx, Ny, Lx, Ly, bcs)
    var re = ReferenceElement2D[P]()

    var q = List[Float64]()
    var sum0: Float64 = 0.0
    for elem in range(mesh.num_elements):
        for nn in range(NP_p):
            var x = mesh.elem_node_xyz[(elem * NP_p + nn) * 2 + 0]
            var y = mesh.elem_node_xyz[(elem * NP_p + nn) * 2 + 1]
            var dx = x - cx
            var dy = y - cy
            var r2 = dx * dx + dy * dy
            var h = h_bg + h_amp * exp(-r2 / (2.0 * sigma * sigma))
            q.append(h)
            q.append(0.0)
            q.append(0.0)
            sum0 += h
    var h_max_ic = h_bg + h_amp
    var mean0 = sum0 / Float64(mesh.num_elements * NP_p)

    var n = len(q)
    var s_q1 = List[Float64]()
    var s_q2 = List[Float64]()
    var s_rhs = List[Float64]()
    for _ in range(n):
        s_q1.append(0.0)
        s_q2.append(0.0)
        s_rhs.append(0.0)

    var physics = ShallowWater2D(g, 1.0e-6)
    var h_cell = Lx / Float64(Nx)
    var c_peak = sqrt(g * h_max_ic)
    var dt_est = cfl * h_cell / (c_peak * Float64(2 * P + 1))
    var num_steps = Int(T_final / dt_est) + 1
    var dt_used = T_final / Float64(num_steps)

    # Integrate.
    var min_h_seen: Float64 = h_bg
    var max_h_seen: Float64 = h_max_ic
    for _ in range(num_steps):
        ssprk3_step_2d[P, ShallowWater2D](
            mesh, re, physics, dt_used,
            q, s_q1, s_q2, s_rhs,
        )
        # Track extremes mid-run so we don't miss a transient.
        for k in range(mesh.num_elements * NP_p):
            var h = q[k * 3 + 0]
            if h < min_h_seen: min_h_seen = h
            if h > max_h_seen: max_h_seen = h

    # Assertions.
    if not (min_h_seen > 0.0):
        raise Error(
            "h went non-positive (min_h_seen=" + String(min_h_seen) + ")"
        )
    var peak_tol = 2.0 * h_max_ic
    if not (max_h_seen < peak_tol):
        raise Error(
            "h overshoot: " + String(max_h_seen)
            + " >= 2 * h_max_IC = " + String(peak_tol)
        )

    var sum_f: Float64 = 0.0
    for k in range(mesh.num_elements * NP_p):
        sum_f += q[k * 3 + 0]
    var mean_f = sum_f / Float64(mesh.num_elements * NP_p)
    var drift = abs_f((mean_f - mean0) / mean0)
    if not (drift < 0.01):
        raise Error(
            "mass drift " + String(drift)
            + " exceeds 1% (mean0=" + String(mean0)
            + ", mean_f=" + String(mean_f) + ")"
        )
    print("    num_steps=", num_steps,
          " min h=", min_h_seen, " max h=", max_h_seen,
          " mass drift=", drift)


def main() raises:
    print("sw_bc_dynamics_test -- Gaussian bump in a walled box")
    run[1]()
    run[2]()
    run[3]()
    print("=== sw_bc_dynamics_test PASSED ===")
