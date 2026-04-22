# ======================================================================
# advection_gaussian -- 3D periodic advection of a Gaussian pulse
# ======================================================================
#
# Single-compilation-unit DG driver using:
#   * `Advection` physics (scalar, constant velocity)
#   * Kuhn-tet Cartesian mesh, triply periodic
#   * Gaussian initial condition, integrated over one period so the
#     exact solution returns to the IC.
#
# Hard-coded setup:
#   domain     : [0, 1]^3
#   mesh       : NX x NY x NZ Cartesian cells, each split into 6 tets
#   velocity   : v = (1, 1, 1)
#   time       : integrate 0 -> 1
#   frames     : NUM_FRAMES evenly-spaced checkpoint VTU files
#
# Output:
#   output/frame_00000.vtu ... output/frame_NNNNN.vtu
#   output/solution.pvd   (ParaView collection file)
# ======================================================================

from std.sys import has_accelerator
from std.gpu import global_idx
from std.gpu.host import DeviceContext
from std.math import sqrt, ceildiv, exp
from std.time import perf_counter_ns

from src.reference import ReferenceElement, N_P, N_F, N_FP, N_D, to_float32
from src.mesh import Mesh
from src.solver import Solver
from src.advection import Advection
from src.vtu import VtuWriter, write_pvd
from src.nvtx import NvtxContext
from src.async_writer import AsyncWriter

comptime NX = 48
comptime NY = 48
comptime NZ = 48
comptime LX = 1.0
comptime LY = 1.0
comptime LZ = 1.0

comptime VX: Float32 = 1.0
comptime VY: Float32 = 1.0
comptime VZ: Float32 = 1.0

comptime T_FINAL: Float32 = 1.0
comptime NUM_FRAMES = 20

# CFL safety factor (explicit P2 DG on tet typically requires CFL ~ 0.1
# times the element-scale / wave-speed).
comptime CFL = Float32(0.2)

# Gaussian pulse parameters -- evaluated by `gaussian_ic_kernel` directly
# on the device at every DG node position.
comptime GAUSS_CX: Float32 = 0.5
comptime GAUSS_CY: Float32 = 0.5
comptime GAUSS_CZ: Float32 = 0.5
comptime GAUSS_SIGMA: Float32 = 0.12

comptime IC_BLOCK = 256


# ----------------------------------------------------------------------
# Initial-condition kernel (periodic Gaussian, single-component)
# ----------------------------------------------------------------------
# Evaluates
#   q(x, y, z) = exp( - (dx^2 + dy^2 + dz^2) / (2 sigma^2) )
# where (dx, dy, dz) is the nearest-image displacement from the pulse
# center (cx, cy, cz) under periodic identification with box size
# (Lx, Ly, Lz).  One thread per solution DOF (num_elements * N_P).
# NC=1 so component layout collapses to a plain scalar write.
# ----------------------------------------------------------------------

def gaussian_ic_kernel(
    q: UnsafePointer[Float32, MutAnyOrigin],
    elem_node_xyz: UnsafePointer[Float32, MutAnyOrigin],
    total: Int,
    cx: Float32, cy: Float32, cz: Float32,
    Lx: Float32, Ly: Float32, Lz: Float32,
    inv_two_sigma2: Float32,
):
    var idx = Int(global_idx.x)
    if idx >= total:
        return
    var px = elem_node_xyz[idx * 3 + 0]
    var py = elem_node_xyz[idx * 3 + 1]
    var pz = elem_node_xyz[idx * 3 + 2]
    var dx = px - cx
    if dx >  Lx * Float32(0.5): dx -= Lx
    if dx < -Lx * Float32(0.5): dx += Lx
    var dy = py - cy
    if dy >  Ly * Float32(0.5): dy -= Ly
    if dy < -Ly * Float32(0.5): dy += Ly
    var dz = pz - cz
    if dz >  Lz * Float32(0.5): dz -= Lz
    if dz < -Lz * Float32(0.5): dz += Lz
    q[idx] = exp(-(dx*dx + dy*dy + dz*dz) * inv_two_sigma2)


# Characteristic element length: ~ cube_side / NX = 1/NX.  Further
# shrink by 1/p^2 for P2 DG stability (explicit RK).  SSPRK3 cfl_max
# ~ 1 on a "standard" DG basis; we include CFL factor for safety.
def choose_dt() raises -> Float32:
    var h = Float32(LX) / Float32(NX)
    var v = sqrt(VX * VX + VY * VY + VZ * VZ)
    # P2 tet empirical CFL factor ~ 1 / (2 p + 1)
    var dt_est = CFL * h / (v * Float32(2 * 2 + 1))
    return dt_est

def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    print("advection_gaussian: GPU DG advection, P2 tet elements")
    print("  mesh: ", NX, "x", NY, "x", NZ,
          " cells -> ", NX * NY * NZ * 6, "tets")
    print("  nodes per element:", N_P, " total DOF:",
          NX * NY * NZ * 6 * N_P)

    var nvtx = NvtxContext()
    print("  NVTX:", "enabled" if nvtx.is_enabled() else "unavailable")

    # Build reference element operators.
    nvtx.push_range("reference_element")
    var re = ReferenceElement()
    var D_ref = to_float32(re.D_ref)
    var Lift_ref = to_float32(re.Lift_ref)
    nvtx.pop_range()

    # Create DeviceContext first so the mesh can build on-device.
    nvtx.push_range("device_context_create")
    var ctx = DeviceContext()
    nvtx.pop_range()

    # Build mesh directly on the GPU.
    nvtx.push_range("build_mesh")
    var mesh = Mesh(ctx, NX, NY, NZ, LX, LY, LZ)
    nvtx.pop_range()
    print("  num elements:", mesh.num_elements,
          " num faces:", mesh.num_faces)

    # Physics: constant-velocity scalar advection.
    var physics = Advection(VX, VY, VZ)

    # Solver takes ownership of ctx + mesh.
    nvtx.push_range("solver_setup")
    var solver = Solver[Advection](ctx^, mesh^, physics^, D_ref^, Lift_ref^)
    nvtx.pop_range()

    # Initial condition -- computed directly on the device from the
    # already-resident mesh node coordinates.  No host buffer, no
    # host-to-device transfer.
    nvtx.push_range("initial_condition")
    var inv_two_sigma2 = Float32(1.0) / (
        Float32(2.0) * GAUSS_SIGMA * GAUSS_SIGMA
    )
    solver.ctx.enqueue_function[gaussian_ic_kernel, gaussian_ic_kernel](
        solver.d_q.unsafe_ptr(),
        solver.mesh.d_elem_node_xyz.unsafe_ptr(),
        solver.total_dof,
        Float32(GAUSS_CX), Float32(GAUSS_CY), Float32(GAUSS_CZ),
        Float32(LX), Float32(LY), Float32(LZ),
        inv_two_sigma2,
        grid_dim=ceildiv(solver.total_dof, IC_BLOCK),
        block_dim=IC_BLOCK,
    )
    solver.ctx.synchronize()
    nvtx.pop_range()

    # VTU writer with pre-serialized static mesh data.
    nvtx.push_range("init_vtu_writer")
    var vtu = VtuWriter(
        solver.mesh.num_elements, solver.mesh.elem_node_xyz_f32_ptr
    )
    nvtx.pop_range()
    nvtx.push_range("init_async_writer")
    var async_writer = AsyncWriter(max_concurrent=8)
    nvtx.pop_range()

    # Write frame 0.
    var vtu_paths = List[String]()
    var times = List[Float64]()

    # Persistent host buffer for scalar density downloads, reused each
    # frame.  For scalar advection this is the full solution.
    nvtx.push_range("init_snapshot_buffer")
    var q_snapshot = List[Float32]()
    for _ in range(solver.total_dof):
        q_snapshot.append(Float32(0.0))
    nvtx.pop_range()

    def do_write_frame(
        frame_id: Int, t: Float64,
        mut solver_ref: Solver[Advection], mut writer: VtuWriter,
        mut aw: AsyncWriter, mut q_buf: List[Float32],
        mut paths: List[String], mut ts: List[Float64],
        mut n: NvtxContext,
    ) raises:
        n.push_range("write_frame")
        solver_ref.download_component(0, q_buf, n)
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

    # Time loop.
    var dt = choose_dt()
    var t: Float32 = 0.0
    var frame_dt = T_FINAL / Float32(NUM_FRAMES)
    var next_frame_t = frame_dt
    var frame_id = 1
    var step = 0

    print("  dt =", dt, " (", Int(T_FINAL / dt), " steps estimated)")

    # Time the main loop excluding write_frame overhead.
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

    # Wait for all outstanding writes to flush before exiting so the
    # .pvd collection file references complete files.
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
