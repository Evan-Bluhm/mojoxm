# ======================================================================
# advection_gaussian_2d_cpu -- CPU-only 2D DG advection demo
# ======================================================================
#
# Integrates a 2D Gaussian bump under linear advection (v = (1, 1)) on
# a periodic [0, 1]^2 domain for one full period.  Uses the CPU Float64
# DG pipeline:
#
#   src/reference_2d.mojo  ->  src/local_mesh_2d.mojo  ->  src/dg_rhs_2d.mojo
#
# Writes NUM_FRAMES VTU snapshots to output/ plus a ParaView collection
# (.pvd) that pins each frame to a time so they animate in order.
# This is the same pattern the 3D drivers use for frame output, just
# simplified for 2D and moved onto the CPU so we can validate the 2D
# stack before the GPU port lands.
#
# Usage:
#   mojo run examples/advection_gaussian_2d_cpu.mojo
#   open output/solution_2d.pvd in ParaView
# ======================================================================

from std.math import sqrt, exp, pi
from std.pathlib import Path
from src.local_mesh_2d import LocalMesh2D
from src.reference_2d import ReferenceElement2D, num_tri_nodes_2d
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

comptime CX = 0.5
comptime CY = 0.5
comptime SIGMA = 0.12


def _gauss(x: Float64, y: Float64) -> Float64:
    var dx = x - CX
    if dx >  LX * 0.5: dx -= LX
    if dx < -LX * 0.5: dx += LX
    var dy = y - CY
    if dy >  LY * 0.5: dy -= LY
    if dy < -LY * 0.5: dy += LY
    var sig2 = SIGMA * SIGMA
    return exp(-(dx * dx + dy * dy) / (2.0 * sig2))


def main() raises:
    print("advection_gaussian_2d_cpu (CPU, P=", P, ")")
    print("  mesh:", NX, "x", NY, " domain: [0,", LX, "] x [0,", LY, "]")

    var NP_p = num_tri_nodes_2d(P)
    var mesh = LocalMesh2D[P](NX, NY, LX, LY)
    var re = ReferenceElement2D[P]()
    print("  elements:", mesh.num_elements,
          "  nodes/elem:", NP_p,
          "  total DOF:", mesh.num_elements * NP_p)

    # Initial condition.
    var n = mesh.num_elements * NP_p
    var q = List[Float64]()
    for elem in range(mesh.num_elements):
        for nn in range(NP_p):
            var x = mesh.elem_node_xyz[(elem * NP_p + nn) * 2 + 0]
            var y = mesh.elem_node_xyz[(elem * NP_p + nn) * 2 + 1]
            q.append(_gauss(x, y))

    # Scratch.
    var s_q1 = List[Float64]()
    var s_q2 = List[Float64]()
    var s_rhs = List[Float64]()
    for _ in range(n):
        s_q1.append(0.0)
        s_q2.append(0.0)
        s_rhs.append(0.0)

    # Frame scheduling.
    var h = LX / Float64(NX)
    var vmag = sqrt(VX * VX + VY * VY)
    var dt_est = CFL * h / (vmag * Float64(2 * P + 1))
    var steps_per_frame = Int(T_FINAL / (Float64(NUM_FRAMES) * dt_est)) + 1
    var dt_used = T_FINAL / Float64(NUM_FRAMES * steps_per_frame)
    print("  dt=", dt_used, " steps/frame=", steps_per_frame,
          " total steps=", NUM_FRAMES * steps_per_frame)

    # Emit the t=0 frame.
    var paths = List[String]()
    var times = List[Float64]()

    def frame_path(i: Int) raises -> String:
        var s = String("output/frame_2d_")
        var idx = String(i)
        for _ in range(5 - idx.byte_length()):
            s += "0"
        s += idx
        s += ".vtu"
        return s^

    dump_vtu_2d_frame[P](mesh, q, frame_path(0))
    paths.append(String("frame_2d_00000.vtu"))
    times.append(0.0)

    var physics = Advection2D(VX, VY)
    for fi in range(1, NUM_FRAMES + 1):
        for _ in range(steps_per_frame):
            ssprk3_step_2d[P, Advection2D](
                mesh, re, physics, dt_used,
                q, s_q1, s_q2, s_rhs,
            )
        var t = Float64(fi) * Float64(steps_per_frame) * dt_used
        dump_vtu_2d_frame[P](mesh, q, frame_path(fi))
        var pname = String("frame_2d_")
        var sidx = String(fi)
        for _ in range(5 - sidx.byte_length()):
            pname += "0"
        pname += sidx
        pname += ".vtu"
        paths.append(pname^)
        times.append(t)
        print("    frame", fi, "/", NUM_FRAMES, " t=", t)

    # Write .pvd collection so ParaView animates the frames.
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
    Path("output/solution_2d.pvd").write_text(pvd)

    print("  wrote output/solution_2d.pvd +", NUM_FRAMES + 1, "VTU frames")
