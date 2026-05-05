# ======================================================================
# MemoryReport -- categorised device memory usage summary
# ======================================================================
#
# Pre-allocation accounting for the GPU buffers a Solver instance owns
# (directly or via mesh + halo).  Mojo doesn't currently expose an
# easy "total VRAM in use" introspection on `DeviceContext`, but every
# allocation in this codebase has a known shape at allocation time --
# so we compute the byte counts from that shape and roll them up into
# the categories WARPXM / Kokkos report at startup.
#
# Categories (mirroring WARPXM-Kokkos's startup table):
#   1. RK stage buffers       -- Solver.d_q + d_q1 + d_q2
#   2. Reference DG operators -- Solver.d_D_ref + d_Lift_ref
#                                + d_node_weights
#   3. Limiter scratch        -- Solver.d_cell_avg + d_bj_theta
#   4. Mesh connectivity      -- LocalMesh d_* buffers + Mesh d_*
#                                buffers (owned_elem_ids, perm, ...)
#   5. Halo exchange (device) -- per-direction pack/unpack indices
#                                + send/recv buffers across 6
#                                neighbours
#   6. Halo exchange (pinned) -- pinned host staging for non-CUDA-
#                                aware MPI; zero when MPI is CUDA-aware
#                                or np=1
#
# Usage:
#   var rep = solver.memory_report()
#   rep.print()
#
# All sizes are reported in bytes; the print method auto-scales to
# B / KB / MB / GB.
# ======================================================================


@fieldwise_init
struct MemoryReport(Movable):
    var rk_stage_bytes:        Int
    var dg_operators_bytes:    Int
    var limiter_bytes:         Int
    var mesh_connectivity_bytes: Int
    var halo_device_bytes:     Int
    var halo_pinned_bytes:     Int

    def total_device_bytes(self) -> Int:
        # Pinned host buffers don't consume device memory.
        return (
            self.rk_stage_bytes
            + self.dg_operators_bytes
            + self.limiter_bytes
            + self.mesh_connectivity_bytes
            + self.halo_device_bytes
        )

    def total_bytes(self) -> Int:
        return self.total_device_bytes() + self.halo_pinned_bytes

    def print(self) raises:
        print("=== Device memory usage ===")
        _print_row("Temporal solver (RK stages):", self.rk_stage_bytes)
        _print_row("Spatial solvers (DG ops):   ", self.dg_operators_bytes)
        _print_row("Cell limiter scratch:       ", self.limiter_bytes)
        _print_row("Mesh connectivity:          ", self.mesh_connectivity_bytes)
        _print_row("Ghost cell sync (device):   ", self.halo_device_bytes)
        _print_row("Ghost cell sync (pinned):   ", self.halo_pinned_bytes)
        print("  ----------------------------------")
        _print_row("Total device memory:        ", self.total_device_bytes())
        if self.halo_pinned_bytes > 0:
            _print_row("Total incl. pinned host:    ", self.total_bytes())
        print("===========================")


def _format_bytes(n: Int) raises -> String:
    """Auto-scale byte counts to a readable unit.  Below 1 KB we
    print exact bytes (e.g. `512 B`); otherwise we pick KB / MB / GB
    so the leading digit is between 1 and 999, matching the WARPXM
    convention."""
    var k = Float64(1024.0)
    var v = Float64(n)
    if v < k:
        return String(n) + " B"
    var kb = v / k
    if kb < k:
        return _round1(kb) + " KB"
    var mb = kb / k
    if mb < k:
        return _round1(mb) + " MB"
    var gb = mb / k
    return _round1(gb) + " GB"


def _round1(x: Float64) raises -> String:
    """One-decimal-place rendering, no exponent, no trailing zeros
    beyond the single decimal (matches `12.3` / `999` style)."""
    var sign: String = ""
    var v = x
    if v < 0.0:
        sign = "-"; v = -v
    var ten_v = v * 10.0
    var int10 = Int(ten_v + 0.5)
    var whole = int10 // 10
    var frac = int10 - whole * 10
    return sign + String(whole) + "." + String(frac)


def _print_row(label: String, n: Int) raises:
    print("  " + label + " " + _format_bytes(n))


# ======================================================================
# ThroughputReport -- per-step wall-time + DOF/s throughput
# ======================================================================
#
# Companion to MemoryReport for the runtime side of perf
# introspection.  After running a step loop the driver constructs one
# of these from:
#   - num_steps        : how many SSPRK3 steps the loop ran
#   - wall_seconds     : measured wall-time for the loop (driver
#                        responsibility -- it knows when to start /
#                        stop the timer)
#   - dof_count        : DOFs being updated per step (one rank's owned
#                        DOF total at np=1; for np>1 the driver should
#                        MPI_Allreduce it before constructing this)
#
# Computed:
#   - per_step_seconds : wall_seconds / num_steps
#   - dof_per_second   : (dof_count * num_steps) / wall_seconds
#                        i.e. throughput in updated-DOF / s.  WARPXM
#                        prints this same metric as the canonical
#                        DG-stack throughput indicator.
#
# Usage:
#   var t0 = perf_counter_ns()
#   for _ in range(num_steps):
#       solver.step_ssprk3(dt, nvtx)
#   solver.ctx.synchronize()
#   var t1 = perf_counter_ns()
#   var dofs = solver.dof_count()
#   var tput = ThroughputReport(num_steps, Float64(t1 - t0) * 1e-9, dofs)
#   tput.print()
# ======================================================================


@fieldwise_init
struct ThroughputReport(Movable):
    var num_steps:    Int
    var wall_seconds: Float64
    var dof_count:    Int

    def per_step_seconds(self) -> Float64:
        if self.num_steps == 0:
            return 0.0
        return self.wall_seconds / Float64(self.num_steps)

    def dof_per_second(self) -> Float64:
        if self.wall_seconds <= 0.0:
            return 0.0
        return Float64(self.dof_count) * Float64(self.num_steps) / self.wall_seconds

    def print(self) raises:
        print("=== Step-loop throughput ===")
        print("  steps:                " + String(self.num_steps))
        print("  DOF / step:           " + String(self.dof_count))
        print("  wall time:            " + _format_seconds(self.wall_seconds))
        print("  per-step wall:        "
              + _format_seconds(self.per_step_seconds()))
        print("  throughput:           "
              + _format_dof_per_s(self.dof_per_second()))
        print("============================")


def _format_seconds(s: Float64) raises -> String:
    """Auto-scale wall-time to a readable unit: ns / us / ms / s."""
    if s < 1.0e-6:
        return _round1(s * 1.0e9) + " ns"
    if s < 1.0e-3:
        return _round1(s * 1.0e6) + " us"
    if s < 1.0:
        return _round1(s * 1.0e3) + " ms"
    return _round1(s) + " s"


def _format_dof_per_s(dps: Float64) raises -> String:
    """Auto-scale DOF/s to readable units: DOF/s, k, M, G."""
    if dps < 1.0e3:
        return _round1(dps) + " DOF/s"
    if dps < 1.0e6:
        return _round1(dps / 1.0e3) + " kDOF/s"
    if dps < 1.0e9:
        return _round1(dps / 1.0e6) + " MDOF/s"
    return _round1(dps / 1.0e9) + " GDOF/s"
