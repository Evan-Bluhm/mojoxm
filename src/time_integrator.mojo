# ======================================================================
# Time integrator: SSPRK3 loop with frame output
# ======================================================================
#
# Advances a Solver[PhysT] with a fixed dt from t=0 to t_final, clamping
# at frame boundaries and at t_final, writing num_frames evenly-spaced
# frames (plus the initial t=0 frame) via a FrameWriter, and returning
# the aggregate timings the drivers used to print by hand.
#
# Usage:
#   var result = run_ssprk3_loop[Euler](
#       solver, writer, dt=dt, t_final=T_FINAL, num_frames=NUM_FRAMES, nvtx,
#   )
#   print("  total steps:", result.total_steps,
#         "  wall time:", result.wall_sec, "s")
#
# NVTX ranges owned here: frame_boundary_sync.
# Marks owned here:      frame_submitted.
# ======================================================================

from src.solver import Solver, Physics
from src.frame_writer import FrameWriter
from src.nvtx import NvtxContext
from std.time import perf_counter_ns


@fieldwise_init
struct TimeLoopResult(Copyable, Movable):
    """Aggregate timings for one run_ssprk3_loop call.  All *_sec fields
    are seconds (Float64).  step_loop_sec counts only the time spent in
    `solver.step_ssprk3` calls (kernel enqueue, no synchronize); actual
    GPU work runs asynchronously, so this number is usually much smaller
    than wall_sec on anything beyond a trivial problem."""
    var total_steps: Int
    var wall_sec: Float64
    var step_loop_sec: Float64
    var frame_write_sec: Float64
    var final_sync_sec: Float64


def run_ssprk3_loop[
    PhysT: Physics,
](
    mut solver: Solver[PhysT],
    mut writer: FrameWriter[PhysT],
    dt: Float32,
    t_final: Float32,
    num_frames: Int,
    mut nvtx: NvtxContext,
) raises -> TimeLoopResult:
    # Write the t=0 frame before stepping.  FrameWriter tracks its own
    # frame counter, so subsequent calls auto-increment.
    writer.write_frame(solver, 0.0, nvtx)

    var t: Float32 = 0.0
    var frame_dt = t_final / Float32(num_frames)
    var next_frame_t = frame_dt
    var frame_id = 1
    var step = 0
    var tloop: Float64 = 0.0
    var twrite: Float64 = 0.0

    var wall_start = perf_counter_ns()
    while t < t_final:
        # Clamp the step to land exactly on the next frame boundary or
        # on t_final, whichever comes first.  If clamping zeros the
        # step (float drift right at the boundary), fall back to the
        # full dt.
        var step_dt = dt
        if t + step_dt > next_frame_t:
            step_dt = next_frame_t - t
        if step_dt <= 0.0:
            step_dt = dt
        if t + step_dt > t_final:
            step_dt = t_final - t
        var s0 = perf_counter_ns()
        solver.step_ssprk3(step_dt, nvtx)
        var s1 = perf_counter_ns()
        tloop += Float64(s1 - s0) * 1e-9
        t += step_dt
        step += 1
        if t >= next_frame_t - Float32(1e-12) and frame_id < num_frames + 1:
            var w0 = perf_counter_ns()
            nvtx.push_range("frame_boundary_sync")
            solver.ctx.synchronize()
            nvtx.pop_range()
            writer.write_frame(solver, Float64(t), nvtx)
            nvtx.mark("frame_submitted")
            var w1 = perf_counter_ns()
            twrite += Float64(w1 - w0) * 1e-9
            frame_id += 1
            next_frame_t += frame_dt

    # Final sync so wall time accounts for any work still queued after
    # the last frame write.
    var sync_start = perf_counter_ns()
    solver.ctx.synchronize()
    var sync_end = perf_counter_ns()
    var wall_end = perf_counter_ns()

    return TimeLoopResult(
        total_steps=step,
        wall_sec=Float64(wall_end - wall_start) * 1e-9,
        step_loop_sec=tloop,
        frame_write_sec=twrite,
        final_sync_sec=Float64(sync_end - sync_start) * 1e-9,
    )
