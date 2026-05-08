# ======================================================================
# Domain-integrated diagnostics for conserved quantities
# ======================================================================
#
# Each driver that wants to track conservation of mass / momentum /
# energy etc. over time constructs a `DiagnosticsWriter` with three
# lists of components (each labelled, each referring to a conserved
# slot in q), and calls `record(t)` once per output frame.  The three
# kinds:
#
#   * `linear`         -> int(q[c]) dV     (mass, momentum, total energy)
#   * `squared`        -> int(q[c]^2) dV   (L2^2 norms, EM/magnetic energy
#                                           components, kinetic-energy
#                                           trackers)
#   * `max_abs`        -> max_x |q[c]|     (peak-value tracker; useful
#                                           for div(B) cleaning, shock
#                                           peaks, overshoots)
#
# Everything goes into one CSV row per frame in the order
# [time, <linear>..., <squared>..., <max_abs>...].
#
# Internally each column uses the same "download owned, accumulate in
# Float64, allreduce, scale by (domain_volume / global_total_dof)"
# pattern (except max_abs, which uses allreduce_float_max of Float32s).
# Keeping the integrals in Float64 on the host is deliberate -- summing
# 1M+ Float32 node values would lose a few digits of precision.
#
# Rank 0 owns the CSV; other ranks still allreduce so everyone's
# `last_row` stays synchronised.
# ======================================================================

from src import mpi
from src.solver import Solver, Physics
from src.reference import num_tet_nodes
from src.nvtx import NvtxContext
from std.pathlib import Path


@fieldwise_init
struct NamedComponent(Copyable, Movable):
    """A (label, conserved-component index) pair to record per frame.
    `label` becomes the CSV column header."""

    var name: String
    var component: Int


# DiagComponents -- chainable builder for the three NamedComponent lists
# that DiagnosticsWriter's constructor expects.  Replaces:
#
#   var diag_linear = List[NamedComponent]()
#   diag_linear.append(NamedComponent("mass", 0))
#   diag_linear.append(NamedComponent("mom_x", 1))
#   ...
#   var diag_squared = List[NamedComponent]()
#   var diag_maxabs = List[NamedComponent]()
#   diag_maxabs.append(NamedComponent("max_density", 0))
#
# with the more compact:
#
#   var components = (DiagComponents()
#       .linear("mass", 0)
#       .linear("mom_x", 1)
#       .maxabs("max_density", 0))
#
# and then:
#
#   var diag = DiagnosticsWriter[Euler](
#       solver, "output/diagnostics.csv",
#       components.linear_list, components.squared_list, components.maxabs_list,
#       LX, LY, LZ,
#   )
struct DiagComponents(Movable):
    var linear_list: List[NamedComponent]
    var squared_list: List[NamedComponent]
    var maxabs_list: List[NamedComponent]

    def __init__(out self):
        self.linear_list = List[NamedComponent]()
        self.squared_list = List[NamedComponent]()
        self.maxabs_list = List[NamedComponent]()

    def linear(var self, name: String, c: Int) -> Self:
        self.linear_list.append(NamedComponent(name, c))
        return self^

    def squared(var self, name: String, c: Int) -> Self:
        self.squared_list.append(NamedComponent(name, c))
        return self^

    def maxabs(var self, name: String, c: Int) -> Self:
        self.maxabs_list.append(NamedComponent(name, c))
        return self^


struct DiagnosticsWriter[PhysT: Physics, P: Int = 2](Movable):
    comptime NP = num_tet_nodes(Self.P)
    # Cross-rank total number of owned DOFs (= global_nx*global_ny*
    # global_nz*6*NP); the `volume / N` factor in the nodal-average
    # integral.
    var global_dof_total: Int
    # Physical volume of the full domain -- multiplies the nodal mean
    # to yield an integral.
    var domain_volume: Float64
    # The three reduction kinds, each a list of (label, q-component).
    var linear: List[NamedComponent]  # int(q[c]) dV
    var squared: List[NamedComponent]  # int(q[c]^2) dV
    var max_abs: List[NamedComponent]  # max_x |q[c]|
    # Rank 0 owns the CSV file.  Other ranks hold an empty string;
    # everyone still participates in the allreduces.
    var csv_path: String
    var is_rank_zero: Bool
    # Scratch host buffer that every record() call writes into; sized
    # once up-front to avoid per-call allocation.
    var scratch: List[Float32]
    # Last-written CSV row cache so run_ssprk3_loop can reuse between
    # frames without the user plumbing it through.
    var last_row: List[Float64]

    def __init__(
        out self,
        mut solver: Solver[Self.PhysT, Self.P],
        csv_path: String,
        linear: List[NamedComponent],
        squared: List[NamedComponent],
        max_abs: List[NamedComponent],
        domain_lx: Float64,
        domain_ly: Float64,
        domain_lz: Float64,
    ) raises:
        var part = solver.mesh.part.copy()
        # At np=1 the mesh IS the global grid.  At np>1 we reconstruct
        # global_nx*global_ny*global_nz from the partition metadata.
        self.global_dof_total = part.global_nx * part.global_ny * part.global_nz * 6 * Self.NP
        self.domain_volume = domain_lx * domain_ly * domain_lz
        self.linear = linear.copy()
        self.squared = squared.copy()
        self.max_abs = max_abs.copy()
        self.is_rank_zero = part.rx == 0 and part.ry == 0 and part.rz == 0
        self.csv_path = csv_path
        self.scratch = List[Float32]()
        for _ in range(solver.num_owned_elements * Self.NP):
            self.scratch.append(Float32(0.0))
        self.last_row = List[Float64]()

        # Rank 0 writes the CSV header on construction.  Use Path.write_text
        # so the parent directory is auto-created just like the VTU writer.
        if self.is_rank_zero:
            var header = String("time")
            for i in range(len(self.linear)):
                header += ","
                header += self.linear[i].name
            for i in range(len(self.squared)):
                header += ","
                header += self.squared[i].name
            for i in range(len(self.max_abs)):
                header += ","
                header += self.max_abs[i].name
            header += "\n"
            Path(csv_path).write_text(header)

    def record(mut self, t: Float64, mut solver: Solver[Self.PhysT, Self.P], mut nvtx: NvtxContext) raises:
        """Compute every configured diagnostic, allreduce, and append
        one row to the CSV (rank 0 only).  Every rank gets the
        aggregated values back in `self.last_row`."""
        nvtx.push_range("diagnostics_record")
        self.last_row.clear()

        # --- Linear integrals int(q[c]) dV -------------------------------
        var scale = self.domain_volume / Float64(self.global_dof_total)
        for i in range(len(self.linear)):
            var c = self.linear[i].component
            solver.download_owned_component(c, self.scratch, nvtx)
            var local: Float64 = 0.0
            for k in range(len(self.scratch)):
                local += Float64(self.scratch[k])
            var global_val: Float64 = 0.0
            _allreduce_one_double(local, global_val)
            self.last_row.append(global_val * scale)

        # --- Squared integrals int(q[c]^2) dV ----------------------------
        for i in range(len(self.squared)):
            var c = self.squared[i].component
            solver.download_owned_component(c, self.scratch, nvtx)
            var local: Float64 = 0.0
            for k in range(len(self.scratch)):
                var v = Float64(self.scratch[k])
                local += v * v
            var global_val: Float64 = 0.0
            _allreduce_one_double(local, global_val)
            self.last_row.append(global_val * scale)

        # --- Max |q[c]| across ranks -------------------------------------
        for i in range(len(self.max_abs)):
            var c = self.max_abs[i].component
            solver.download_owned_component(c, self.scratch, nvtx)
            var mloc: Float32 = 0.0
            for k in range(len(self.scratch)):
                var v = self.scratch[k]
                var av = v if v >= Float32(0.0) else -v
                if av > mloc:
                    mloc = av
            var lbuf = InlineArray[Float32, 1](fill=0.0)
            var rbuf = InlineArray[Float32, 1](fill=0.0)
            lbuf[0] = mloc
            var lp = rebind[UnsafePointer[Float32, MutAnyOrigin]](lbuf.unsafe_ptr())
            var rp = rebind[UnsafePointer[Float32, MutAnyOrigin]](rbuf.unsafe_ptr())
            mpi.allreduce_float_max(lp, rp, 1)
            self.last_row.append(Float64(rbuf[0]))

        if self.is_rank_zero:
            # Read-modify-write the file: load current contents, append
            # the new row, write back.  Cheap at frame cadence; avoids
            # keeping a file handle open through the (asynchronous) VTU
            # frame writer's tenure.
            var existing = String(Path(self.csv_path).read_text())
            var row = String(t)
            for i in range(len(self.last_row)):
                row += ","
                row += String(self.last_row[i])
            row += "\n"
            Path(self.csv_path).write_text(existing + row)
        nvtx.pop_range()


# ----------------------------------------------------------------------
# One-shot Float64 allreduce-sum (no shim for scalar-only; just use the
# 1-element buffer form).
# ----------------------------------------------------------------------


def _allreduce_one_double(local: Float64, mut out: Float64) raises:
    var lbuf = InlineArray[Float64, 1](fill=0.0)
    var rbuf = InlineArray[Float64, 1](fill=0.0)
    lbuf[0] = local
    var lp = rebind[UnsafePointer[Float64, MutAnyOrigin]](lbuf.unsafe_ptr())
    var rp = rebind[UnsafePointer[Float64, MutAnyOrigin]](rbuf.unsafe_ptr())
    mpi.allreduce_double_sum(lp, rp, 1)
    out = rbuf[0]
