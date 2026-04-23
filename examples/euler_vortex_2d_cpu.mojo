# ======================================================================
# euler_vortex_2d_cpu -- CPU-only 2D isentropic vortex advection
# ======================================================================
#
# The classical DG validation problem (Shu 1998, Erlebacher et al.):
# an isentropic Euler vortex overlaid on a uniform background flow
# should translate across a periodic domain without distortion.  After
# one full period the exact solution equals the initial condition, so
# any L1/L2 discrepancy isolates the DG scheme's numerical dissipation.
#
# IC formulas (gamma = 1.4):
#   r^2      = (x - cx)^2 + (y - cy)^2       (periodic nearest image)
#   T(0)     = T_inf - (gamma - 1) beta^2 / (8 gamma pi^2) * exp(1 - r^2)
#   u(0)     = u0 - beta / (2 pi) * (y - cy) * exp((1 - r^2) / 2)
#   v(0)     = v0 + beta / (2 pi) * (x - cx) * exp((1 - r^2) / 2)
#   rho(0)   = T(0)^(1/(gamma - 1))                 (isentropic closure)
#   p(0)     = rho(0) * T(0)
#   E(0)     = p(0) / (gamma - 1) + 0.5 * rho(0) * (u^2 + v^2)
#
# Writes NUM_FRAMES snapshots to output/frame_euler2d_NNNNN.vtu + a
# .pvd collection.  Open output/solution_euler2d.pvd in ParaView.
# ======================================================================

from std.math import sqrt, exp, pi
from std.pathlib import Path
from src.local_mesh_2d import LocalMesh2D
from src.reference_2d import ReferenceElement2D, num_tri_nodes_2d
from src.dg_rhs_2d import Euler2D, ssprk3_step_2d
from src.vtu_2d import dump_vtu_2d_frame


comptime P = 2
comptime NX = 32
comptime NY = 32
comptime LX = 10.0
comptime LY = 10.0
comptime T_FINAL = 10.0
comptime NUM_FRAMES = 20
comptime CFL = 0.15

comptime GAMMA = 1.4
comptime T_INF = 1.0
comptime U0 = 1.0
comptime V0 = 1.0
comptime BETA = 5.0
comptime CX0 = 5.0
comptime CY0 = 5.0


def _periodic_delta(a: Float64, b: Float64, L: Float64) -> Float64:
    var d = a - b
    if d >  L * 0.5: d -= L
    if d < -L * 0.5: d += L
    return d


def main() raises:
    print("euler_vortex_2d_cpu (P=", P, ",", NX, "x", NY, ")")
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
            var dx = _periodic_delta(x, CX0, LX)
            var dy = _periodic_delta(y, CY0, LY)
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

    var physics = Euler2D(GAMMA, 1.0e-6, 1.0e-6)
    var h = LX / Float64(NX)
    var c_inf = sqrt(GAMMA * T_INF)
    var wave_max = sqrt(U0 * U0 + V0 * V0) + c_inf + BETA / two_pi
    var dt_est = CFL * h / (wave_max * Float64(2 * P + 1))
    var steps_per_frame = Int(T_FINAL / (Float64(NUM_FRAMES) * dt_est)) + 1
    var dt_used = T_FINAL / Float64(NUM_FRAMES * steps_per_frame)
    print("  dt=", dt_used, " steps/frame=", steps_per_frame)

    def frame_path(i: Int) raises -> String:
        var s = String("output/frame_euler2d_")
        var idx = String(i)
        for _ in range(5 - idx.byte_length()):
            s += "0"
        s += idx
        s += ".vtu"
        return s^

    # Density is what we visualise; pull it out per frame.
    var density = List[Float64]()
    for _ in range(mesh.num_elements * NP_p):
        density.append(0.0)

    var paths = List[String]()
    var times = List[Float64]()
    for k in range(mesh.num_elements * NP_p):
        density[k] = q[k * 4 + 0]
    dump_vtu_2d_frame[P](mesh, density, frame_path(0), String("rho"))
    paths.append(String("frame_euler2d_00000.vtu"))
    times.append(0.0)

    for fi in range(1, NUM_FRAMES + 1):
        for _ in range(steps_per_frame):
            ssprk3_step_2d[P, Euler2D](
                mesh, re, physics, dt_used,
                q, s_q1, s_q2, s_rhs,
            )
        var t = Float64(fi) * Float64(steps_per_frame) * dt_used
        for k in range(mesh.num_elements * NP_p):
            density[k] = q[k * 4 + 0]
        dump_vtu_2d_frame[P](mesh, density, frame_path(fi), String("rho"))
        var pname = String("frame_euler2d_")
        var sidx = String(fi)
        for _ in range(5 - sidx.byte_length()):
            pname += "0"
        pname += sidx
        pname += ".vtu"
        paths.append(pname^)
        times.append(t)
        print("    frame", fi, "/", NUM_FRAMES, " t=", t)

    # Final L2 error against the IC (one full period -> same state).
    var sum_sq: Float64 = 0.0
    var sum_ic_sq: Float64 = 0.0
    for k in range(n):
        var e = q[k] - q_ic[k]
        sum_sq += e * e
        sum_ic_sq += q_ic[k] * q_ic[k]
    var l2_err = sqrt(sum_sq / Float64(n))
    var l2_ic = sqrt(sum_ic_sq / Float64(n))
    print("  final L2 err =", l2_err,
          " rel =", l2_err / l2_ic, " (IC =", l2_ic, ")")

    var pvd = String()
    pvd += '<?xml version="1.0"?>\n'
    pvd += ('<VTKFile type="Collection" version="0.1"'
            ' byte_order="LittleEndian">\n')
    pvd += '<Collection>\n'
    for i in range(len(paths)):
        pvd += '<DataSet timestep="'
        pvd += String(times[i])
        pvd += '" group="" part="0" file="'
        pvd += paths[i]
        pvd += '"/>\n'
    pvd += '</Collection>\n'
    pvd += '</VTKFile>\n'
    Path("output/solution_euler2d.pvd").write_text(pvd)
    print("  wrote output/solution_euler2d.pvd +",
          NUM_FRAMES + 1, "VTU frames")
