# ======================================================================
# euler_channel_2d_cpu -- supersonic Euler channel with BC_INFLOW
# ======================================================================
#
# A uniform supersonic stream enters through the -x boundary and
# exits through the +x boundary, between two slip walls in y.  The
# IC is the inflow state everywhere, so the physical answer is the
# steady uniform flow -- any drift from it is numerical noise.
#
#   inflow state: rho = 1, u = 2 * c_inf (Mach 2), v = 0, p = 1
#                 where c_inf = sqrt(gamma * p / rho).
#
#   BCs: -x: BC_INFLOW    (supersonic inflow; all 4 eigenvalues
#                         point into the domain)
#        +x: BC_OUTFLOW   (supersonic outflow; all 4 eigenvalues
#                         point out, zero-gradient ghost is fine)
#        -y / +y: BC_WALL (slip wall; normal momentum reflected)
#
# Writes 21 density snapshots to output/.  The initial and final
# frames should be visually indistinguishable since the analytic
# answer is steady.  Runs for T = 0.2 on [0, 1] x [0.25] at P=2.
# ======================================================================

from std.math import sqrt
from std.pathlib import Path
from src.local_mesh_2d import LocalMesh2D
from src.reference_2d import ReferenceElement2D, num_tri_nodes_2d
from src.boundary import BoundaryConditions2D, BC_WALL, BC_OUTFLOW, BC_INFLOW
from src.dg_rhs_2d import Euler2D, ssprk3_step_2d
from src.vtu_2d import dump_vtu_2d_frame


comptime P = 2
comptime NX = 64
comptime NY = 16
comptime LX = 1.0
comptime LY = 0.25
comptime GAMMA = 1.4
comptime RHO_0 = 1.0
comptime P_0 = 1.0
comptime MACH = 2.0
comptime T_FINAL = 0.2
comptime NUM_FRAMES = 20
comptime CFL = 0.15


def main() raises:
    print("euler_channel_2d_cpu (Mach", MACH, ", P=", P, ")")
    print("  BCs: -x INFLOW, +x OUTFLOW, y WALL")

    var c_inf = sqrt(GAMMA * P_0 / RHO_0)
    var u_inf = MACH * c_inf
    print("  c_inf =", c_inf, " u_inf =", u_inf)

    # Conservative state for the inflow.
    var rhou_inf = RHO_0 * u_inf
    var E_inf = P_0 / (GAMMA - 1.0) + 0.5 * RHO_0 * u_inf * u_inf

    var NP_p = num_tri_nodes_2d(P)
    var bcs = BoundaryConditions2D(
        BC_INFLOW, BC_OUTFLOW, BC_WALL, BC_WALL,
    )
    var mesh = LocalMesh2D[P](NX, NY, LX, LY, bcs)
    var re = ReferenceElement2D[P]()

    # IC: fill the whole domain with the inflow state.  The analytic
    # solution is stationary in this configuration.
    var n = mesh.num_elements * NP_p * 4
    var q = List[Float64]()
    for _ in range(mesh.num_elements * NP_p):
        q.append(RHO_0)
        q.append(rhou_inf)
        q.append(0.0)
        q.append(E_inf)

    var s_q1 = List[Float64]()
    var s_q2 = List[Float64]()
    var s_rhs = List[Float64]()
    for _ in range(n):
        s_q1.append(0.0)
        s_q2.append(0.0)
        s_rhs.append(0.0)

    var physics = Euler2D(
        GAMMA, 1.0e-6, 1.0e-6,
        RHO_0, rhou_inf, 0.0, E_inf,   # BC_INFLOW state
    )

    var h_cell = LX / Float64(NX)
    var wave_max = u_inf + c_inf
    var dt_est = CFL * h_cell / (wave_max * Float64(2 * P + 1))
    var steps_per_frame = Int(T_FINAL / (Float64(NUM_FRAMES) * dt_est)) + 1
    var dt_used = T_FINAL / Float64(NUM_FRAMES * steps_per_frame)
    print("  dt=", dt_used, " steps/frame=", steps_per_frame)

    def frame_path(i: Int) raises -> String:
        var s = String("output/frame_channel_")
        var idx = String(i)
        for _ in range(5 - idx.byte_length()):
            s += "0"
        s += idx
        s += ".vtu"
        return s^

    var density = List[Float64]()
    for _ in range(mesh.num_elements * NP_p):
        density.append(0.0)

    var paths = List[String]()
    var times = List[Float64]()
    for k in range(mesh.num_elements * NP_p):
        density[k] = q[k * 4 + 0]
    dump_vtu_2d_frame[P](mesh, density, frame_path(0), String("rho"))
    paths.append(String("frame_channel_00000.vtu"))
    times.append(0.0)

    var rho_max_drift: Float64 = 0.0
    for fi in range(1, NUM_FRAMES + 1):
        for _ in range(steps_per_frame):
            ssprk3_step_2d[P, Euler2D](
                mesh, re, physics, dt_used,
                q, s_q1, s_q2, s_rhs,
            )
        var t = Float64(fi) * Float64(steps_per_frame) * dt_used
        for k in range(mesh.num_elements * NP_p):
            density[k] = q[k * 4 + 0]
            var drift = density[k] - RHO_0
            var adr = drift if drift >= 0.0 else -drift
            if adr > rho_max_drift:
                rho_max_drift = adr
        dump_vtu_2d_frame[P](mesh, density, frame_path(fi), String("rho"))
        var pname = String("frame_channel_")
        var sidx = String(fi)
        for _ in range(5 - sidx.byte_length()):
            pname += "0"
        pname += sidx
        pname += ".vtu"
        paths.append(pname^)
        times.append(t)

    print("  rho max |drift| vs IC:", rho_max_drift,
          " (expected ~ 0 for exact steady solution)")

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
    Path("output/solution_channel.pvd").write_text(pvd)
    print("  wrote output/solution_channel.pvd +",
          NUM_FRAMES + 1, "VTU frames")
