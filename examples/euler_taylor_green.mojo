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
from std.math import sqrt, ceildiv, sin, cos, pi
from std.time import perf_counter_ns

from reference import ReferenceElement, N_P, N_F, N_FP, N_D, to_float32
from mesh import Mesh
from solver import Solver
from euler import Euler, FLUX_HLLEC
from vtu import VtuWriter, write_pvd
from nvtx import NvtxContext
from async_writer import AsyncWriter

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

    nvtx.push_range("reference_element")
    var re = ReferenceElement()
    var D_ref = to_float32(re.D_ref)
    var Lift_ref = to_float32(re.Lift_ref)
    nvtx.pop_range()

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
    var solver = Solver[Euler](ctx^, mesh^, physics^, D_ref^, Lift_ref^)
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

    nvtx.push_range("init_vtu_writer")
    var vtu = VtuWriter(
        solver.mesh.num_elements, solver.mesh.elem_node_xyz_f32_ptr
    )
    nvtx.pop_range()
    nvtx.push_range("init_async_writer")
    var async_writer = AsyncWriter(max_concurrent=8)
    nvtx.pop_range()

    var vtu_paths = List[String]()
    var times = List[Float64]()

    nvtx.push_range("init_snapshot_buffer")
    var q_snapshot = List[Float32]()
    for _ in range(solver.total_dof):
        q_snapshot.append(Float32(0.0))
    nvtx.pop_range()

    def do_write_frame(
        frame_id: Int, t: Float64,
        mut solver_ref: Solver[Euler], mut writer: VtuWriter,
        mut aw: AsyncWriter, mut q_buf: List[Float32],
        mut paths: List[String], mut ts: List[Float64],
        mut n: NvtxContext,
    ) raises:
        n.push_range("write_frame")
        solver_ref.download_component(0, q_buf, n)   # rho is component 0
        var fname = "frame_"
        var sid = String(frame_id)
        for _ in range(5 - len(sid)):
            fname += "0"
        fname += sid
        fname += ".vtu"
        n.push_range("vtu_build_segments")
        var segs = writer.build_segments(q_buf)
        n.pop_range()
        n.push_range("vtu_submit")
        aw.submit("output/" + fname, segs)
        n.pop_range()
        paths.append(fname)
        ts.append(t)
        n.pop_range()

    do_write_frame(0, 0.0, solver, vtu, async_writer,
                    q_snapshot, vtu_paths, times, nvtx)
    print("  wrote initial frame")

    var dt = choose_dt()
    var t: Float32 = 0.0
    var frame_dt = T_FINAL / Float32(NUM_FRAMES)
    var next_frame_t = frame_dt
    var frame_id = 1
    var step = 0

    print("  dt =", dt, " (", Int(T_FINAL / dt), " steps estimated)")

    var tloop: Float64 = 0.0
    var twrite: Float64 = 0.0
    var wall_start = perf_counter_ns()
    while t < T_FINAL:
        var step_dt = dt
        if t + step_dt > next_frame_t:
            step_dt = next_frame_t - t
        if step_dt <= 0.0:
            step_dt = dt
        if t + step_dt > T_FINAL:
            step_dt = T_FINAL - t
        var s0 = perf_counter_ns()
        solver.step_ssprk3(step_dt, nvtx)
        var s1 = perf_counter_ns()
        tloop += Float64(s1 - s0) * 1e-9
        t += step_dt
        step += 1
        if t >= next_frame_t - Float32(1e-12) and frame_id < NUM_FRAMES + 1:
            var w0 = perf_counter_ns()
            nvtx.push_range("frame_boundary_sync")
            solver.ctx.synchronize()
            nvtx.pop_range()
            do_write_frame(frame_id, Float64(t),
                           solver, vtu, async_writer, q_snapshot,
                           vtu_paths, times, nvtx)
            nvtx.mark("frame_submitted")
            var w1 = perf_counter_ns()
            twrite += Float64(w1 - w0) * 1e-9
            frame_id += 1
            next_frame_t += frame_dt
    var sync_start = perf_counter_ns()
    solver.ctx.synchronize()
    var sync_end = perf_counter_ns()
    print("  final sync:", Float64(sync_end - sync_start) * 1e-9, "s")

    nvtx.push_range("wait_async_writes")
    var wait_start = perf_counter_ns()
    async_writer.wait_all()
    var wait_end = perf_counter_ns()
    nvtx.pop_range()
    print("  wait for async writes:",
          Float64(wait_end - wait_start) * 1e-9, "s")

    var wall_end = perf_counter_ns()
    var wall_sec = Float64(wall_end - wall_start) * 1e-9
    print("  total steps:", step, " wall time:", wall_sec, "s")
    print("    step-loop time (enqueue only, no sync):", tloop, "s")
    print("    frame-write time (download + VTU):", twrite, "s")

    write_pvd("output/solution.pvd", vtu_paths, times)
    print("  wrote output/solution.pvd")
