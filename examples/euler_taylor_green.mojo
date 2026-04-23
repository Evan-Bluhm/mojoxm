# ======================================================================
# euler_taylor_green -- compressible Taylor-Green vortex
# ======================================================================
#
# The canonical smooth test for high-order compressible solvers.  Two
# counter-rotating vortex sheets on a 2 pi cube evolve under the Euler
# equations, stretch, tangle, and cascade energy into smaller scales.
# The flow is fully smooth and triply periodic, so it works cleanly
# inside our current framework (no limiter, no wall BCs).
#
# Initial condition (constant-density form):
#   u   =  U0 * sin(x) cos(y) cos(z)
#   v   = -U0 * cos(x) sin(y) cos(z)
#   w   =  0
#   rho =  rho0
#   p   =  p0 + (rho0 U0^2 / 16) * (cos(2x) + cos(2y)) * (cos(2z) + 2)
#   E   =  p/(gamma-1) + 0.5 rho (u^2 + v^2 + w^2)
#
# Mach number is set by the ratio U0 / sqrt(gamma p0 / rho0).  At
# Ma ~ 0.3 the flow develops visible density oscillations (~10% of rho0)
# and remains shocklet-free for the first few eddy turnover times.
#
# Reference: Taylor & Green, Proc. Roy. Soc. A 158 (1937); and the
# "Taylor-Green Vortex" as popularised by Brachet et al. (1983).  Used
# as the AIAA high-order CFD workshop benchmark.
# ======================================================================

from std.sys import has_accelerator
from std.gpu import global_idx
from std.gpu.host import DeviceContext
from std.math import sqrt, ceildiv, sin, cos

from src.reference import N_P, build_reference_operators
from src.mesh import Mesh
from src.solver import Solver
from src.euler import Euler, FLUX_HLLEC
from src.nvtx import NvtxContext
from src.frame_writer import FrameWriter
from src.time_integrator import run_ssprk3_loop

# Domain is the natural 2 pi cube so sin/cos of coordinates are periodic
# without any wrap-around algebra.
comptime TWO_PI: Float32 = 6.283185307179586
comptime LX = 6.283185307179586
comptime LY = 6.283185307179586
comptime LZ = 6.283185307179586

comptime NX = 32
comptime NY = 32
comptime NZ = 32

# Flow parameters.  Ma = U0 / sqrt(gamma p0 / rho0) ~ 0.3.
comptime GAMMA: Float32 = 1.4
comptime RHO0:  Float32 = 1.0
comptime U0:    Float32 = 1.0
comptime P0:    Float32 = 7.9365079    # chosen so c0 = sqrt(1.4 * p0) ~ 3.33 -> Ma = 0.3

# Numerical-flux floors.
comptime MIN_DENSITY:  Float32 = 1.0e-6
comptime MIN_PRESSURE: Float32 = 1.0e-6

# Integration window.  ~3 eddy turnover times is enough to see the
# large counter-rotating cells stretch, merge, and begin to cascade.
comptime T_FINAL: Float32 = 0.1
comptime NUM_FRAMES = 30

# SSPRK3 safety factor for P2 DG on tets.
comptime CFL = Float32(0.2)

comptime IC_BLOCK = 256


# ----------------------------------------------------------------------
# Initial-condition kernel (Taylor-Green vortex)
# ----------------------------------------------------------------------
# One thread per solution DOF (num_elements * N_P).  Reads node
# coordinates from the already-resident mesh buffer and writes the
# 5-component conserved state into q.  Layout: q[(idx * 5) + c].
# ----------------------------------------------------------------------

def taylor_green_ic_kernel(
    q: UnsafePointer[Float32, MutAnyOrigin],
    elem_node_xyz: UnsafePointer[Float32, MutAnyOrigin],
    total: Int,
    u0: Float32, rho0: Float32, p0: Float32, gamma: Float32,
):
    var idx = Int(global_idx.x)
    if idx >= total:
        return
    var x = elem_node_xyz[idx * 3 + 0]
    var y = elem_node_xyz[idx * 3 + 1]
    var z = elem_node_xyz[idx * 3 + 2]

    var sx = sin(x); var cx = cos(x)
    var sy = sin(y); var cy = cos(y)
    var cz = cos(z)
    var c2x = cos(Float32(2.0) * x)
    var c2y = cos(Float32(2.0) * y)
    var c2z = cos(Float32(2.0) * z)

    var u =  u0 * sx * cy * cz
    var v = -u0 * cx * sy * cz
    var w =  Float32(0.0)

    var rho = rho0
    var p = p0 + (rho0 * u0 * u0 / Float32(16.0)) * (c2x + c2y) * (c2z + Float32(2.0))
    var E = p / (gamma - Float32(1.0)) + Float32(0.5) * rho * (u * u + v * v + w * w)

    var base = idx * 5
    q[base + 0] = rho
    q[base + 1] = rho * u
    q[base + 2] = rho * v
    q[base + 3] = rho * w
    q[base + 4] = E


# Estimated dt bound.  Max wave speed is |U| + c; pick |U| <= U0 and
# c ~ c0 since density/pressure don't move much at Ma ~ 0.3.
def choose_dt() raises -> Float32:
    var h = Float32(LX) / Float32(NX)
    var c0 = sqrt(GAMMA * P0 / RHO0)
    var wave = U0 + c0
    var dt_est = CFL * h / (wave * Float32(2 * 2 + 1))
    return dt_est

def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    print("euler_taylor_green: GPU DG Euler, P2 tet elements, HLLEC flux")
    print("  mesh: ", NX, "x", NY, "x", NZ,
          " cells -> ", NX * NY * NZ * 6, "tets")
    print("  nodes per element:", N_P, " total DOF:",
          NX * NY * NZ * 6 * N_P)

    var nvtx = NvtxContext()
    print("  NVTX:", "enabled" if nvtx.is_enabled() else "unavailable")

    var refs = build_reference_operators(nvtx)

    nvtx.push_range("device_context_create")
    var ctx = DeviceContext()
    nvtx.pop_range()

    nvtx.push_range("build_mesh")
    var mesh = Mesh(ctx, NX, NY, NZ, LX, LY, LZ)
    nvtx.pop_range()
    print("  num elements:", mesh.num_elements,
          " num faces:", mesh.num_faces)

    var physics = Euler(
        GAMMA, MIN_DENSITY, MIN_PRESSURE, FLUX_HLLEC, True
    )

    nvtx.push_range("solver_setup")
    var solver = Solver[Euler](
        ctx^, mesh^, physics^, refs.D_ref^, refs.Lift_ref^,
    )
    nvtx.pop_range()

    nvtx.push_range("initial_condition")
    solver.ctx.enqueue_function[taylor_green_ic_kernel, taylor_green_ic_kernel](
        solver.d_q.unsafe_ptr(),
        solver.mesh.d_elem_node_xyz.unsafe_ptr(),
        solver.total_dof,
        U0, RHO0, P0, GAMMA,
        grid_dim=ceildiv(solver.total_dof, IC_BLOCK),
        block_dim=IC_BLOCK,
    )
    solver.ctx.synchronize()
    nvtx.pop_range()

    # Frame output (density is component 0 of the 5-component Euler state).
    var writer = FrameWriter[Euler](solver, nvtx, component=0)

    var dt = choose_dt()
    print("  dt =", dt, " (", Int(T_FINAL / dt), " steps estimated)")

    var result = run_ssprk3_loop[Euler](
        solver, writer, dt, T_FINAL, NUM_FRAMES, nvtx,
    )
    print("  final sync:", result.final_sync_sec, "s")

    writer.finalize("output/solution.pvd", nvtx)

    print("  total steps:", result.total_steps,
          " wall time:", result.wall_sec, "s")
    print("    step-loop time (enqueue only, no sync):",
          result.step_loop_sec, "s")
    print("    frame-write time (download + VTU):",
          result.frame_write_sec, "s")
    print("  wrote output/solution.pvd")
