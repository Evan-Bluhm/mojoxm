# ======================================================================
# shallow_water_drop_2d_cpu -- CPU-only 2D shallow water "drop"
# ======================================================================
#
# Releases a circular elevated water column in an otherwise quiescent
# basin.  The column collapses and radially propagating waves travel
# outward, reflecting off the periodic boundaries (so effectively the
# demo is "drop in an infinite grid of drops").
#
# Initial condition on a [0, 1]^2 periodic domain at rest:
#   h(x, y, 0) = h_bg + h_amp * exp(-r^2 / (2 sigma^2))
#   u(x, y, 0) = 0
#   v(x, y, 0) = 0
# where r is the nearest-image distance to the drop centre.
#
# Writes 21 VTU frames of depth h + a .pvd collection.  Open
# output/solution_sw2d.pvd in ParaView.
# ======================================================================

from std.math import sqrt, exp
from std.pathlib import Path
from src.local_mesh_2d import LocalMesh2D
from src.reference_2d import ReferenceElement2D, num_tri_nodes_2d
from src.dg_rhs_2d import ShallowWater2D, ssprk3_step_2d
from src.vtu_2d import dump_vtu_2d_frame


comptime P = 2
comptime NX = 48
comptime NY = 48
comptime LX = 1.0
comptime LY = 1.0
comptime T_FINAL = 0.4
comptime NUM_FRAMES = 20
comptime CFL = 0.2

comptime G = 9.81
comptime H_BG = 1.0       # background depth
comptime H_AMP = 0.4       # drop amplitude
comptime SIGMA = 0.08
comptime DROP_CX = 0.5
comptime DROP_CY = 0.5


def _periodic_delta(a: Float64, b: Float64, L: Float64) -> Float64:
    var d = a - b
    if d >  L * 0.5: d -= L
    if d < -L * 0.5: d += L
    return d


def main() raises:
    print("shallow_water_drop_2d_cpu (P=", P, ",", NX, "x", NY, ")")
    var NP_p = num_tri_nodes_2d(P)
    var mesh = LocalMesh2D[P](NX, NY, LX, LY)
    var re = ReferenceElement2D[P]()

    var n = mesh.num_elements * NP_p * 3
    var q = List[Float64]()
    for elem in range(mesh.num_elements):
        for nn in range(NP_p):
            var x = mesh.elem_node_xyz[(elem * NP_p + nn) * 2 + 0]
            var y = mesh.elem_node_xyz[(elem * NP_p + nn) * 2 + 1]
            var dx = _periodic_delta(x, DROP_CX, LX)
            var dy = _periodic_delta(y, DROP_CY, LY)
            var r2 = dx * dx + dy * dy
            var h = H_BG + H_AMP * exp(-r2 / (2.0 * SIGMA * SIGMA))
            q.append(h)
            q.append(0.0)   # h * u
            q.append(0.0)   # h * v

    var s_q1 = List[Float64]()
    var s_q2 = List[Float64]()
    var s_rhs = List[Float64]()
    for _ in range(n):
        s_q1.append(0.0)
        s_q2.append(0.0)
        s_rhs.append(0.0)

    var physics = ShallowWater2D(G, 1.0e-6)
    # CFL: dt * (|u| + sqrt(g h)) / h_cell <= C_P.  At rest |u| = 0;
    # peak wave speed sqrt(g * h_max) = sqrt(9.81 * 1.4) ~ 3.70.
    var h_cell = LX / Float64(NX)
    var c_peak = sqrt(G * (H_BG + H_AMP))
    var dt_est = CFL * h_cell / (c_peak * Float64(2 * P + 1))
    var steps_per_frame = Int(T_FINAL / (Float64(NUM_FRAMES) * dt_est)) + 1
    var dt_used = T_FINAL / Float64(NUM_FRAMES * steps_per_frame)
    print("  dt=", dt_used, " steps/frame=", steps_per_frame,
          " total steps=", NUM_FRAMES * steps_per_frame)

    def frame_path(i: Int) raises -> String:
        var s = String("output/frame_sw2d_")
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
    paths.append(String("frame_sw2d_00000.vtu"))
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
        var pname = String("frame_sw2d_")
        var sidx = String(fi)
        for _ in range(5 - sidx.byte_length()):
            pname += "0"
        pname += sidx
        pname += ".vtu"
        paths.append(pname^)
        times.append(t)
        print("    frame", fi, "/", NUM_FRAMES, " t=", t)

    # Mass (integral of h) is conserved by the discretisation.  The
    # naive node-average cross-checks it.
    var sum_h: Float64 = 0.0
    for k in range(mesh.num_elements * NP_p):
        sum_h += q[k * 3 + 0]
    var mean_h = sum_h / Float64(mesh.num_elements * NP_p)
    print("  final mean h =", mean_h, " (expected ~", H_BG,
          " + ", H_AMP, " * sigma^2 * 2pi / (L^2)",
          " =", H_BG + H_AMP * 2.0 * 3.14159265 * SIGMA * SIGMA / (LX * LY),
          ")")

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
    Path("output/solution_sw2d.pvd").write_text(pvd)
    print("  wrote output/solution_sw2d.pvd +",
          NUM_FRAMES + 1, "VTU frames")
