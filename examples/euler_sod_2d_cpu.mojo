# ======================================================================
# euler_sod_2d_cpu -- 2D analog of the classical 1D Sod shock tube
# ======================================================================
#
# Initial condition: the 1D Sod problem lifted to a 2D strip
#
#   left state  (x < 0.5 Lx):  rho=1.0, u=v=0, p=1.0
#   right state (x > 0.5 Lx):  rho=0.125, u=v=0, p=0.1
#   gamma = 1.4
#
# Boundary conditions: x ends transmissive outflow (BC_OUTFLOW), y ends
# slip walls (BC_WALL).  Runs to T=0.20 (the classical Sod horizon).
#
# Without the 2D Barth-Jespersen limiter (currently off-by-default
# because of an unresolved mass-drift on discontinuous ICs), the
# Gibbs overshoot at the shock grows large but the sim does not NaN
# -- the Euler Rusanov flux is just dissipative enough on its own to
# keep the solution bounded at this resolution.
#
# Writes 21 VTU frames of density.  Load output/solution_sod2d.pvd in
# ParaView to animate; or `scripts/animate_2d.py output/solution_sod2d.pvd`
# for an MP4.
# ======================================================================

from std.math import sqrt, tanh
from std.pathlib import Path
from src.local_mesh_2d import LocalMesh2D
from src.reference_2d import ReferenceElement2D, num_tri_nodes_2d
from src.boundary import BoundaryConditions2D, BC_WALL, BC_OUTFLOW
from src.dg_rhs_2d import Euler2D, ssprk3_step_2d
from src.vtu_2d import dump_vtu_2d_frame


comptime P = 2
comptime NX = 128
comptime NY = 16
comptime LX = 1.0
comptime LY = 0.125
comptime GAMMA = 1.4
comptime RHO_L = 1.0
comptime P_L = 1.0
comptime RHO_R = 0.125
comptime P_R = 0.1
comptime T_FINAL = 0.20
comptime NUM_FRAMES = 20
comptime CFL = 0.15


def main() raises:
    print("euler_sod_2d_cpu (Sod shock tube lifted to 2D, P=", P, ")")
    print("  mesh:", NX, "x", NY, " domain:", LX, "x", LY)
    print("  BCs: x-ends OUTFLOW, y-ends WALL")

    var NP_p = num_tri_nodes_2d(P)
    var bcs = BoundaryConditions2D(
        BC_OUTFLOW, BC_OUTFLOW,   # -x, +x
        BC_WALL,    BC_WALL,       # -y, +y
    )
    var mesh = LocalMesh2D[P](NX, NY, LX, LY, bcs)
    var re = ReferenceElement2D[P]()
    print("  elements:", mesh.num_elements, " faces:", mesh.num_faces)

    # IC: left / right states, with a tanh smoothing across ~8 cells
    # so the initial jump isn't an unresolved discontinuity.  Classical
    # Sod at this resolution blows up immediately without either this
    # smoothing or a slope limiter; smoothing is the simpler knob.
    var smooth_width = 8.0 * (LX / Float64(NX))
    var n = mesh.num_elements * NP_p * 4
    var q = List[Float64]()
    for elem in range(mesh.num_elements):
        for nn in range(NP_p):
            var x = mesh.elem_node_xyz[(elem * NP_p + nn) * 2 + 0]
            # s in [0, 1]: 0 on far left, 1 on far right.
            var s = 0.5 * (tanh((x - 0.5 * LX) / smooth_width) + 1.0)
            var rho = RHO_L + s * (RHO_R - RHO_L)
            var p   = P_L   + s * (P_R   - P_L)
            var E = p / (GAMMA - 1.0)   # zero velocity
            q.append(rho)
            q.append(0.0)
            q.append(0.0)
            q.append(E)

    var s_q1 = List[Float64]()
    var s_q2 = List[Float64]()
    var s_rhs = List[Float64]()
    for _ in range(n):
        s_q1.append(0.0)
        s_q2.append(0.0)
        s_rhs.append(0.0)

    var physics = Euler2D(GAMMA, 1.0e-6, 1.0e-6)
    var c_peak = sqrt(GAMMA * P_L / RHO_L)    # speed of sound in left state
    var h_cell = LX / Float64(NX)
    var dt_est = CFL * h_cell / (2.0 * c_peak * Float64(2 * P + 1))
    var steps_per_frame = Int(T_FINAL / (Float64(NUM_FRAMES) * dt_est)) + 1
    var dt_used = T_FINAL / Float64(NUM_FRAMES * steps_per_frame)
    print("  dt=", dt_used, " steps/frame=", steps_per_frame,
          " total=", NUM_FRAMES * steps_per_frame)

    def frame_path(i: Int) raises -> String:
        var s = String("output/frame_sod2d_")
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
    paths.append(String("frame_sod2d_00000.vtu"))
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
        var pname = String("frame_sod2d_")
        var sidx = String(fi)
        for _ in range(5 - sidx.byte_length()):
            pname += "0"
        pname += sidx
        pname += ".vtu"
        paths.append(pname^)
        times.append(t)
        print("    frame", fi, "/", NUM_FRAMES, " t=", t)

    # Summary: densities at the boundaries should be close to the
    # original states (shock / rarefaction still inside the domain at
    # T=0.20).
    var rho_left: Float64 = 0.0
    var rho_right: Float64 = 0.0
    var n_left = 0
    var n_right = 0
    for elem in range(mesh.num_elements):
        for nn in range(NP_p):
            var x = mesh.elem_node_xyz[(elem * NP_p + nn) * 2 + 0]
            var rho = q[(elem * NP_p + nn) * 4 + 0]
            if x < 0.02:
                rho_left += rho
                n_left += 1
            if x > LX - 0.02:
                rho_right += rho
                n_right += 1
    if n_left > 0:
        rho_left /= Float64(n_left)
    if n_right > 0:
        rho_right /= Float64(n_right)
    print("  density @ -x boundary  =", rho_left, " (expected ~1.0)")
    print("  density @ +x boundary  =", rho_right, " (expected ~0.125)")

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
    Path("output/solution_sod2d.pvd").write_text(pvd)
    print("  wrote output/solution_sod2d.pvd +",
          NUM_FRAMES + 1, "VTU frames")
