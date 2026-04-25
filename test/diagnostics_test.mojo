# ======================================================================
# diagnostics_test -- sanity checks for DiagnosticsWriter
# ======================================================================
#
# What we test (all at np=1 to keep the harness simple; the multi-rank
# allreduce path is validated transitively by make test-bc, which
# produces bit-identical output with diagnostics enabled):
#
#   1. `linear` integrals on a uniform field recover (value * domain
#      volume) to better than roundoff.
#   2. `squared` integrals on a uniform field recover (value^2 * vol).
#   3. `max_abs` reports the global peak.
#   4. Zero-component writer (no columns configured) writes just the
#      time column and doesn't crash.
#
# Runs end to end: builds a 4x4x4 mesh, a trivial Advection solver,
# fills q with a known constant, calls record(), checks the produced
# `last_row` values.
# ======================================================================

from src import mpi
from src.partition import build_partition
from src.mesh import Mesh
from src.boundary import BoundaryConditions
from src.halo_exchange import HaloExchange
from src.solver import Solver
from src.reference import ReferenceElement, N_P, to_float32
from src.advection import Advection
from src.diagnostics import DiagnosticsWriter, NamedComponent
from src.nvtx import NvtxContext
from std.sys import has_accelerator
from std.gpu import global_idx
from std.gpu.host import DeviceContext
from std.math import ceildiv


comptime NX = 4
comptime NY = 4
comptime NZ = 4
comptime LX = 1.0
comptime LY = 1.0
comptime LZ = 1.0
comptime VX: Float32 = 0.0
comptime VY: Float32 = 0.0
comptime VZ: Float32 = 0.0
comptime IC_BLOCK = 256


def fill_constant_kernel(
    q: UnsafePointer[Float32, MutAnyOrigin],
    owned_elem_ids: UnsafePointer[Int32, MutAnyOrigin],
    num_owned: Int,
    value: Float32,
):
    var idx = Int(global_idx.x)
    var total = num_owned * N_P
    if idx >= total:
        return
    var i = idx // N_P
    var nn = idx % N_P
    var e = Int(owned_elem_ids[i])
    q[e * N_P + nn] = value


def fill_checkerboard_kernel(
    q: UnsafePointer[Float32, MutAnyOrigin],
    owned_elem_ids: UnsafePointer[Int32, MutAnyOrigin],
    num_owned: Int,
    amplitude: Float32,
):
    var idx = Int(global_idx.x)
    var total = num_owned * N_P
    if idx >= total:
        return
    var i = idx // N_P
    var nn = idx % N_P
    var e = Int(owned_elem_ids[i])
    # Alternate +amplitude / -amplitude by element parity.  Gives a
    # well-defined max|q| = amplitude and int q = ~0 (modulo node
    # aliasing) for testing.
    var sign: Float32 = 1.0 if (e % 2) == 0 else -1.0
    q[e * N_P + nn] = sign * amplitude


def assert_close(name: String, got: Float64, expected: Float64,
                 tol: Float64) raises:
    var d = got - expected
    var ad = d if d >= 0.0 else -d
    if ad > tol:
        raise Error(
            name + ": got=" + String(got) + " expected=" + String(expected)
            + " |diff|=" + String(ad) + " tol=" + String(tol)
        )


def test_uniform_field(
    mut solver: Solver[Advection],
    mut nvtx: NvtxContext,
) raises:
    print("uniform-field integrals...")
    # Fill q with 2.5 everywhere.
    var value: Float32 = 2.5
    var num_owned = solver.num_owned_elements
    solver.ctx.enqueue_function[fill_constant_kernel, fill_constant_kernel](
        solver.d_q.unsafe_ptr(),
        solver.mesh.d_owned_elem_ids.unsafe_ptr(),
        num_owned, value,
        grid_dim=ceildiv(num_owned * N_P, IC_BLOCK),
        block_dim=IC_BLOCK,
    )
    solver.ctx.synchronize()

    # Configure one column of each kind.
    var lin = List[NamedComponent]()
    lin.append(NamedComponent("mass", 0))
    var sq = List[NamedComponent]()
    sq.append(NamedComponent("l2_sq", 0))
    var ma = List[NamedComponent]()
    ma.append(NamedComponent("max_abs_q", 0))
    var diag = DiagnosticsWriter[Advection](
        solver, "output/_test_uniform.csv",
        lin, sq, ma, LX, LY, LZ,
    )
    diag.record(0.0, solver, nvtx)

    # Expected: int(value) = value * Vol, int(value^2) = value^2 * Vol,
    # max|value| = value.  The nodal-sum approximation is *exact* for
    # a constant function (every node contributes `value`, N nodes
    # total, multiplied by Vol/N recovers value * Vol).
    var vol = Float64(LX) * Float64(LY) * Float64(LZ)
    var v = Float64(value)
    assert_close("linear(q)", diag.last_row[0], v * vol, 1.0e-5)
    assert_close("squared(q)", diag.last_row[1], v * v * vol, 1.0e-5)
    assert_close("max_abs(q)", diag.last_row[2], v, 1.0e-6)
    print("  linear  =", diag.last_row[0], " expected=", v * vol)
    print("  squared =", diag.last_row[1], " expected=", v * v * vol)
    print("  max_abs =", diag.last_row[2], " expected=", v)


def test_checkerboard(
    mut solver: Solver[Advection],
    mut nvtx: NvtxContext,
) raises:
    print("checkerboard peak-value test...")
    var amp: Float32 = 7.0
    var num_owned = solver.num_owned_elements
    solver.ctx.enqueue_function[
        fill_checkerboard_kernel, fill_checkerboard_kernel,
    ](
        solver.d_q.unsafe_ptr(),
        solver.mesh.d_owned_elem_ids.unsafe_ptr(),
        num_owned, amp,
        grid_dim=ceildiv(num_owned * N_P, IC_BLOCK),
        block_dim=IC_BLOCK,
    )
    solver.ctx.synchronize()

    var ma = List[NamedComponent]()
    ma.append(NamedComponent("max_abs_q", 0))
    var sq = List[NamedComponent]()
    sq.append(NamedComponent("l2_sq", 0))
    var diag = DiagnosticsWriter[Advection](
        solver, "output/_test_checker.csv",
        List[NamedComponent](), sq, ma, LX, LY, LZ,
    )
    diag.record(0.0, solver, nvtx)

    # squared should be amp^2 * Vol (independent of sign); max_abs = amp.
    var vol = Float64(LX) * Float64(LY) * Float64(LZ)
    var a = Float64(amp)
    assert_close("squared(checker)", diag.last_row[0], a * a * vol, 1.0e-5)
    assert_close("max_abs(checker)", diag.last_row[1], a, 1.0e-6)
    print("  squared =", diag.last_row[0], " (symmetric between +/- amp)")
    print("  max_abs =", diag.last_row[1])


def test_empty_writer(
    mut solver: Solver[Advection],
    mut nvtx: NvtxContext,
) raises:
    print("zero-column writer...")
    var diag = DiagnosticsWriter[Advection](
        solver, "output/_test_empty.csv",
        List[NamedComponent](),
        List[NamedComponent](),
        List[NamedComponent](),
        LX, LY, LZ,
    )
    diag.record(1.5, solver, nvtx)
    if len(diag.last_row) != 0:
        raise Error("empty writer produced non-empty row")
    print("  OK (no columns -> no entries in last_row)")


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    mpi.init()
    var size = mpi.world_size()
    if size > 1:
        mpi.finalize()
        print("diagnostics_test: runs at np=1 only")
        return
    var nvtx = NvtxContext()
    var ctx = DeviceContext()
    var re = ReferenceElement[2]()
    var D_ref = to_float32(re.D_ref)
    var Lift_ref = to_float32(re.Lift_ref)
    var node_weights = to_float32(re.node_weights)

    var mesh = Mesh(
        ctx, build_partition(0, 1, NX, NY, NZ), LX, LY, LZ,
        BoundaryConditions.periodic(),
    )
    var halo = HaloExchange(
        ctx, mesh.part, Advection.NUM_COMPONENTS,
        mesh.d_perm.unsafe_ptr(),
    )
    var physics = Advection(VX, VY, VZ)
    var solver = Solver[Advection](
        ctx^, mesh^, halo^, physics^, D_ref^, Lift_ref^, node_weights^,
    )

    test_uniform_field(solver, nvtx)
    test_checkerboard(solver, nvtx)
    test_empty_writer(solver, nvtx)

    print("=== diagnostics_test PASSED ===")
    mpi.finalize()
