# ======================================================================
# shallow_water_dam_break_2d_cpu -- 2D dam break in a closed basin
# ======================================================================
#
# Classical shallow-water Riemann problem lifted to 2D:
#
#   IC:   h(x, y, 0) = h_L for x < Lx/2,  h_R for x > Lx/2
#         u = v = 0 everywhere
#   BCs:  WALL on all four sides of the [0, Lx] x [0, Ly] basin
#
# The dam collapses into a rightward-propagating bore + leftward
# rarefaction; the bore eventually reflects off the +x wall, the
# rarefaction off the -x wall.  Mass and y-momentum are exactly
# conserved.  Writes 21 VTU frames + a .pvd collection so ParaView
# animates h(x, y, t).  Runs on CPU in Float64.
# ======================================================================

from std.math import sqrt
from std.pathlib import Path
from src.local_mesh_2d import LocalMesh2D
from src.reference_2d import ReferenceElement2D, num_tri_nodes_2d
from src.boundary import BoundaryConditions2D, BC_WALL
from src.dg_rhs_2d import ShallowWater2D, ssprk3_step_2d
from src.vtu_2d import dump_vtu_2d_frame


comptime P = 2
comptime NX = 64
comptime NY = 16
comptime LX = 2.0
comptime LY = 0.5
comptime G = 9.81
comptime H_L = 2.0
comptime H_R = 1.0
comptime T_FINAL = 0.5
comptime NUM_FRAMES = 20
comptime CFL = 0.2


def main() raises:
    print("shallow_water_dam_break_2d_cpu (walls on all 4 sides)")
    print("  mesh:", NX, "x", NY, " domain:", LX, "x", LY,
          " (h_L=", H_L, ", h_R=", H_R, ")")

    var NP_p = num_tri_nodes_2d(P)
    var bcs = BoundaryConditions2D(BC_WALL, BC_WALL, BC_WALL, BC_WALL)
    var mesh = LocalMesh2D[P](NX, NY, LX, LY, bcs)
    var re = ReferenceElement2D[P]()
    print("  elements:", mesh.num_elements,
          "  faces:", mesh.num_faces, "  (", NP_p, "nodes/elem)")

    var n = mesh.num_elements * NP_p * 3
    var q = List[Float64]()
    var initial_mass: Float64 = 0.0
    for elem in range(mesh.num_elements):
        for nn in range(NP_p):
            var x = mesh.elem_node_xyz[(elem * NP_p + nn) * 2 + 0]
            var h: Float64
            if x < LX * 0.5:
                h = H_L
            else:
                h = H_R
            q.append(h)
            q.append(0.0)
            q.append(0.0)
            initial_mass += h
    var mean_h0 = initial_mass / Float64(mesh.num_elements * NP_p)
    print("  initial mean h =", mean_h0)

    var s_q1 = List[Float64]()
    var s_q2 = List[Float64]()
    var s_rhs = List[Float64]()
    for _ in range(n):
        s_q1.append(0.0)
        s_q2.append(0.0)
        s_rhs.append(0.0)

    var physics = ShallowWater2D(G, 1.0e-6)
    var c_peak = sqrt(G * H_L)
    var h_cell = LX / Float64(NX)
    var dt_est = CFL * h_cell / (c_peak * Float64(2 * P + 1))
    var steps_per_frame = Int(T_FINAL / (Float64(NUM_FRAMES) * dt_est)) + 1
    var dt_used = T_FINAL / Float64(NUM_FRAMES * steps_per_frame)
    print("  dt=", dt_used, " steps/frame=", steps_per_frame)

    def frame_path(i: Int) raises -> String:
        var s = String("output/frame_dam_")
        var idx = String(i)
        for _ in range(5 - idx.byte_length()):
            s += "0"
        s += idx
        s += ".vtu"
        return s^

    var depth = List[Float64]()
    for _ in range(mesh.num_elements * NP_p):
        depth.append(0.0)

    var paths = List[String]()
    var times = List[Float64]()

    for k in range(mesh.num_elements * NP_p):
        depth[k] = q[k * 3 + 0]
    dump_vtu_2d_frame[P](mesh, depth, frame_path(0), String("h"))
    paths.append(String("frame_dam_00000.vtu"))
    times.append(0.0)

    for fi in range(1, NUM_FRAMES + 1):
        for _ in range(steps_per_frame):
            ssprk3_step_2d[P, ShallowWater2D](
                mesh, re, physics, dt_used,
                q, s_q1, s_q2, s_rhs,
            )
        var t = Float64(fi) * Float64(steps_per_frame) * dt_used
        for k in range(mesh.num_elements * NP_p):
            depth[k] = q[k * 3 + 0]
        dump_vtu_2d_frame[P](mesh, depth, frame_path(fi), String("h"))
        var pname = String("frame_dam_")
        var sidx = String(fi)
        for _ in range(5 - sidx.byte_length()):
            pname += "0"
        pname += sidx
        pname += ".vtu"
        paths.append(pname^)
        times.append(t)
        print("    frame", fi, "/", NUM_FRAMES, " t=", t)

    # Mass conservation check.
    var final_mass: Float64 = 0.0
    var max_h: Float64 = 0.0
    var min_h: Float64 = 1.0e30
    for k in range(mesh.num_elements * NP_p):
        var h = q[k * 3 + 0]
        final_mass += h
        if h > max_h: max_h = h
        if h < min_h: min_h = h
    var mean_hf = final_mass / Float64(mesh.num_elements * NP_p)
    var mass_err = (mean_hf - mean_h0) / mean_h0
    print("  final mean h =", mean_hf,
          " relative drift =", mass_err,
          " (h range [", min_h, ",", max_h, "])")

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
    Path("output/solution_dam.pvd").write_text(pvd)
    print("  wrote output/solution_dam.pvd +",
          NUM_FRAMES + 1, "VTU frames")
