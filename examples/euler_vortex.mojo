# ======================================================================
# euler_vortex -- isentropic Euler vortex, triply periodic
# ======================================================================
#
# Classic smooth-periodic test problem for the compressible Euler
# equations.  A 2D vortex in the xy plane, constant along z, superposed
# on a uniform background flow v_bg = (u_bg, v_bg, w_bg).  The exact
# solution is the vortex translated by v_bg; with periodic BCs and
# T_final chosen so v_bg * T_final is a lattice vector, the solution
# returns to the initial condition.
#
# Perturbations (Shu 1998 form):
#   du = -eps * (y - y0) * exp((1 - r^2)/2) / (2 pi)
#   dv = +eps * (x - x0) * exp((1 - r^2)/2) / (2 pi)
#   dw = 0
#   T  = 1 - (gamma - 1) eps^2 / (8 gamma pi^2) * exp(1 - r^2)
#   rho = T^(1/(gamma - 1))
#   p   = rho^gamma
#
# where r^2 = (x-x0)^2 + (y-y0)^2.  The vortex is isentropic so
# p/rho^gamma = const = 1.
#
# Single-compilation-unit DG driver using:
#   * `Euler` physics (5 components) with HLLEC numerical flux
#   * Kuhn-tet Cartesian mesh, triply periodic
#   * SSPRK3 time integration
#   * VTU output of the density field
# ======================================================================

from std.sys import has_accelerator
from std.gpu import global_idx
from std.gpu.host import DeviceContext
from std.math import sqrt, ceildiv, exp, log, pi, pow
from std.time import perf_counter_ns

from src.reference import ReferenceElement, N_P, N_F, N_FP, N_D, to_float32
from src.mesh import Mesh
from src.solver import Solver
from src.euler import Euler, FLUX_HLLEC
from src.vtu import VtuWriter, write_pvd
from src.nvtx import NvtxContext
from src.async_writer import AsyncWriter

comptime NX = 32
comptime NY = 32
comptime NZ = 32
comptime LX = 10.0
comptime LY = 10.0
comptime LZ = 10.0

# Background velocity (unit translation of the vortex per t).
comptime UBG: Float32 = 1.0
comptime VBG: Float32 = 1.0
comptime WBG: Float32 = 0.0

# Vortex parameters.
comptime GAMMA: Float32 = 1.4
comptime VORTEX_EPS: Float32 = 5.0    # peak tangential speed
comptime VORTEX_X0: Float32 = 5.0
comptime VORTEX_Y0: Float32 = 5.0

# Numerical-flux choices.
comptime MIN_DENSITY: Float32 = 1.0e-6
comptime MIN_PRESSURE: Float32 = 1.0e-6

comptime T_FINAL: Float32 = 1.0    # one period for demonstration
comptime NUM_FRAMES = 10

# SSPRK3 safety factor for P2 DG on tets.
comptime CFL = Float32(0.2)

comptime IC_BLOCK = 256


# ----------------------------------------------------------------------
# Initial-condition kernel (isentropic vortex)
# ----------------------------------------------------------------------
# One thread per solution DOF (num_elements * N_P).  Reads node
# coordinates from the already-resident mesh buffer and writes the
# 5-component conserved state into q.  Layout: q[(idx * 5) + c].
# ----------------------------------------------------------------------

def vortex_ic_kernel(
    q: UnsafePointer[Float32, MutAnyOrigin],
    elem_node_xyz: UnsafePointer[Float32, MutAnyOrigin],
    total: Int,
    x0: Float32, y0: Float32,
    ubg: Float32, vbg: Float32, wbg: Float32,
    gamma: Float32, eps: Float32,
):
    var idx = Int(global_idx.x)
    if idx >= total:
        return
    var px = elem_node_xyz[idx * 3 + 0]
    var py = elem_node_xyz[idx * 3 + 1]
    var dx = px - x0
    var dy = py - y0
    var r2 = dx * dx + dy * dy
    var expf = exp((Float32(1.0) - r2) * Float32(0.5))
    var eps_over_2pi = eps / (Float32(2.0) * Float32(3.14159265358979323846))
    var du = -eps_over_2pi * dy * expf
    var dv =  eps_over_2pi * dx * expf
    var u = ubg + du
    var v = vbg + dv
    var w = wbg
    var g1 = gamma - Float32(1.0)
    var T = Float32(1.0) - g1 * eps * eps
            / (Float32(8.0) * gamma
               * Float32(3.14159265358979323846) * Float32(3.14159265358979323846)) \
            * exp(Float32(1.0) - r2)
    var rho = pow(T, Float32(1.0) / g1)
    var p   = pow(rho, gamma)
    var E   = p / g1 + Float32(0.5) * rho * (u * u + v * v + w * w)
    var base = idx * 5
    q[base + 0] = rho
    q[base + 1] = rho * u
    q[base + 2] = rho * v
    q[base + 3] = rho * w
    q[base + 4] = E


# Estimated dt bound based on the background + peak tangential speed
# plus the sound speed at rest (c ~ sqrt(gamma)).
def choose_dt() raises -> Float32:
    var h = Float32(LX) / Float32(NX)
    var v_bg_mag = sqrt(UBG * UBG + VBG * VBG + WBG * WBG)
    var v_peak = VORTEX_EPS / Float32(2.0 * 3.14159265358979323846)  # rough
    var c_ref = sqrt(GAMMA)
    var wave = v_bg_mag + v_peak + c_ref
    var dt_est = CFL * h / (wave * Float32(2 * 2 + 1))
    return dt_est

def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    print("euler_vortex: GPU DG Euler, P2 tet elements, HLLEC flux")
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
    solver.ctx.enqueue_function[vortex_ic_kernel, vortex_ic_kernel](
        solver.d_q.unsafe_ptr(),
        solver.mesh.d_elem_node_xyz.unsafe_ptr(),
        solver.total_dof,
        VORTEX_X0, VORTEX_Y0,
        UBG, VBG, WBG,
        GAMMA, VORTEX_EPS,
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

    # Persistent host buffer for the scalar density snapshot.
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
        for _ in range(5 - sid.byte_length()):
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
