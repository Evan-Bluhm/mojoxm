# ======================================================================
# advection_outflow_2d_cpu -- 2D advection with transmissive outflow
# ======================================================================
#
# Counterpoint to the periodic `advection_gaussian_2d_cpu` driver:
# sets BC_OUTFLOW on all four domain boundaries so a passing Gaussian
# bump *leaves* the domain through the downstream edges.
#
# Physical interpretation: tracer advected through an open channel.
# Total mass decreases monotonically once the bump reaches the
# outflow boundary (here at t ~ 0.4 given v=(1, 1) and the bump
# centred at (0.3, 0.3) on [0, 1]^2).  This driver makes that
# visible both in the VTU frames and in the reported mass trace.
#
# Output: output/solution_advout.pvd plus 21 VTU frames.  Animate via
#   scripts/animate_2d.py output/solution_advout.pvd
# ======================================================================

from std.math import sqrt, exp
from std.pathlib import Path
from src.local_mesh_2d import LocalMesh2D
from src.reference_2d import ReferenceElement2D, num_tri_nodes_2d
from src.boundary import BoundaryConditions2D, BC_OUTFLOW
from src.dg_rhs_2d import Advection2D, ssprk3_step_2d
from src.vtu_2d import dump_vtu_2d_frame


comptime P = 2
comptime NX = 32
comptime NY = 32
comptime LX = 1.0
comptime LY = 1.0
comptime VX = 1.0
comptime VY = 1.0
comptime T_FINAL = 1.0
comptime NUM_FRAMES = 20
comptime CFL = 0.3

comptime CX = 0.3
comptime CY = 0.3
comptime SIGMA = 0.1


def main() raises:
    print("advection_outflow_2d_cpu (BC_OUTFLOW on all 4 sides, P=", P, ")")
    var NP_p = num_tri_nodes_2d(P)
    var bcs = BoundaryConditions2D(
        BC_OUTFLOW, BC_OUTFLOW, BC_OUTFLOW, BC_OUTFLOW,
    )
    var mesh = LocalMesh2D[P](NX, NY, LX, LY, bcs)
    var re = ReferenceElement2D[P]()
    print("  elements:", mesh.num_elements, " faces:", mesh.num_faces,
          " (+", mesh.num_faces - 3 * NX * NY, "boundary faces)")

    var n = mesh.num_elements * NP_p
    var q = List[Float64]()
    var mass_ic: Float64 = 0.0
    for elem in range(mesh.num_elements):
        for nn in range(NP_p):
            var x = mesh.elem_node_xyz[(elem * NP_p + nn) * 2 + 0]
            var y = mesh.elem_node_xyz[(elem * NP_p + nn) * 2 + 1]
            var dx = x - CX
            var dy = y - CY
            var r2 = dx * dx + dy * dy
            var v = exp(-r2 / (2.0 * SIGMA * SIGMA))
            q.append(v)
            mass_ic += v

    var s_q1 = List[Float64]()
    var s_q2 = List[Float64]()
    var s_rhs = List[Float64]()
    for _ in range(n):
        s_q1.append(0.0)
        s_q2.append(0.0)
        s_rhs.append(0.0)

    var physics = Advection2D(VX, VY)
    var h = LX / Float64(NX)
    var dt_est = CFL * h / (sqrt(VX * VX + VY * VY) * Float64(2 * P + 1))
    var steps_per_frame = Int(T_FINAL / (Float64(NUM_FRAMES) * dt_est)) + 1
    var dt_used = T_FINAL / Float64(NUM_FRAMES * steps_per_frame)
    print("  dt=", dt_used, " steps/frame=", steps_per_frame)

    def frame_path(i: Int) raises -> String:
        var s = String("output/frame_advout_")
        var idx = String(i)
        for _ in range(5 - idx.byte_length()):
            s += "0"
        s += idx
        s += ".vtu"
        return s^

    var paths = List[String]()
    var times = List[Float64]()
    dump_vtu_2d_frame[P](mesh, q, frame_path(0), String("q"))
    paths.append(String("frame_advout_00000.vtu"))
    times.append(0.0)

    print("  mass trace (expected to decrease once bump leaves domain):")
    print("    t=0.0       mass =", mass_ic)

    for fi in range(1, NUM_FRAMES + 1):
        for _ in range(steps_per_frame):
            ssprk3_step_2d[P, Advection2D](
                mesh, re, physics, dt_used,
                q, s_q1, s_q2, s_rhs,
            )
        var t = Float64(fi) * Float64(steps_per_frame) * dt_used
        var mass_now: Float64 = 0.0
        for k in range(n):
            mass_now += q[k]
        dump_vtu_2d_frame[P](mesh, q, frame_path(fi), String("q"))
        var pname = String("frame_advout_")
        var sidx = String(fi)
        for _ in range(5 - sidx.byte_length()):
            pname += "0"
        pname += sidx
        pname += ".vtu"
        paths.append(pname^)
        times.append(t)
        # Print mass at select frames.
        if fi % 4 == 0 or fi == NUM_FRAMES:
            print("    t=", t, " mass =", mass_now,
                  " frac =", mass_now / mass_ic)

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
    Path("output/solution_advout.pvd").write_text(pvd)
    print("  wrote output/solution_advout.pvd +",
          NUM_FRAMES + 1, "VTU frames")
