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
