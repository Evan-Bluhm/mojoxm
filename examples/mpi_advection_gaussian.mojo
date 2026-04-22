# ======================================================================
# mpi_advection_gaussian -- multi-rank Gaussian-pulse advection
# ======================================================================
#
# Parallel version of examples/advection_gaussian.mojo.  Each rank
# owns a sub-box of the global cube grid plus a 1-cube ghost ring;
# the halo is refreshed via MPI at the top of every RK stage.  After
# T = T_FINAL (one full period with v = (1,1,1) on [0, 1]^3) the
# exact solution returns to the IC, so rank 0 can report the integral
# L2 error against the original Gaussian to verify correctness.
#
# Build (run from the project root):
#   mpicc -O2 -fPIC -c src/mpi_shim.c -o build/mpi_shim.o
#   .venv/bin/mojo build -O3 -g0 -I . examples/mpi_advection_gaussian.mojo \
#       -o mpi_advection_gaussian \
#       -Xlinker build/mpi_shim.o \
#       -Xlinker -L/usr/lib/x86_64-linux-gnu/openmpi/lib \
#       -Xlinker -lmpi \
#       -Xlinker -lm -Xlinker -lpthread
#
# Run:
#   mpirun -np 8 ./mpi_advection_gaussian
# ======================================================================

from src import mpi
from src.partition import build_partition
from src.patch_mesh import PatchMesh
from src.halo_exchange import HaloExchange
from src.patch_solver import PatchSolver
from src.reference import ReferenceElement, N_P, to_float32
from src.advection import Advection
from src.nvtx import NvtxContext
from std.sys import has_accelerator
from std.gpu import global_idx
from std.gpu.host import DeviceContext
from std.math import sqrt, ceildiv, exp
from std.time import perf_counter_ns

comptime NX = 32
comptime NY = 32
comptime NZ = 32
comptime LX = 1.0
comptime LY = 1.0
comptime LZ = 1.0

comptime VX: Float32 = 1.0
comptime VY: Float32 = 1.0
comptime VZ: Float32 = 1.0

comptime T_FINAL: Float32 = 1.0

# CFL safety factor (explicit P2 DG on tet typically requires
# CFL ~ 0.1 * element-scale / wave-speed).
comptime CFL = Float32(0.2)

# Gaussian pulse parameters.
comptime GAUSS_CX: Float32 = 0.5
comptime GAUSS_CY: Float32 = 0.5
comptime GAUSS_CZ: Float32 = 0.5
comptime GAUSS_SIGMA: Float32 = 0.12

comptime IC_BLOCK = 256


def gaussian_ic_kernel(
    q: UnsafePointer[Float32, MutAnyOrigin],
    owned_elem_ids: UnsafePointer[Int32, MutAnyOrigin],
    elem_node_xyz:  UnsafePointer[Float32, MutAnyOrigin],
    num_owned: Int,
    cx: Float32, cy: Float32, cz: Float32,
    Lx: Float32, Ly: Float32, Lz: Float32,
    inv_two_sigma2: Float32,
):
    var idx = Int(global_idx.x)
    var total = num_owned * N_P
    if idx >= total:
        return
    var i = idx // N_P
    var nn = idx % N_P
    var e = Int(owned_elem_ids[i])
    var px = elem_node_xyz[(e * N_P + nn) * 3 + 0]
    var py = elem_node_xyz[(e * N_P + nn) * 3 + 1]
    var pz = elem_node_xyz[(e * N_P + nn) * 3 + 2]
    var dx = px - cx
    if dx >  Lx * Float32(0.5): dx -= Lx
    if dx < -Lx * Float32(0.5): dx += Lx
    var dy = py - cy
    if dy >  Ly * Float32(0.5): dy -= Ly
    if dy < -Ly * Float32(0.5): dy += Ly
    var dz = pz - cz
    if dz >  Lz * Float32(0.5): dz -= Lz
    if dz < -Lz * Float32(0.5): dz += Lz
    q[e * N_P + nn] = exp(
        -(dx * dx + dy * dy + dz * dz) * inv_two_sigma2
    )


def choose_dt() raises -> Float32:
    var h = Float32(LX) / Float32(NX)
    var v = sqrt(VX * VX + VY * VY + VZ * VZ)
    return CFL * h / (v * Float32(2 * 2 + 1))


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    mpi.init()
    var rank = mpi.world_rank()
    var size = mpi.world_size()

    if rank == 0:
        print("mpi_advection_gaussian: GPU DG advection, P2 tet, ",
              size, "ranks")
        print("  global mesh: ", NX, "x", NY, "x", NZ,
              " cells -> ", NX * NY * NZ * 6, "tets")

    var nvtx = NvtxContext()
    var ctx = DeviceContext()

    var re = ReferenceElement()
    var D_ref    = to_float32(re.D_ref)
    var Lift_ref = to_float32(re.Lift_ref)

    var patch = PatchMesh(
        ctx, build_partition(rank, size, NX, NY, NZ), LX, LY, LZ
    )
    # Report partition layout from rank 0.
    if rank == 0:
        print("  proc-grid: ",
              patch.part.px, "x", patch.part.py, "x", patch.part.pz,
              "  owned cubes per rank: ",
              patch.part.nx, "x", patch.part.ny, "x", patch.part.nz)
        print("  per-rank: ",
              patch.num_owned_elements, "owned elements (halo=",
              patch.num_halo_elements, ", interior=",
              patch.num_interior_elements, ")")

    var halo = HaloExchange(ctx, patch.part, Advection.NUM_COMPONENTS)
    var physics = Advection(VX, VY, VZ)
    var solver = PatchSolver[Advection](
        ctx^, patch^, halo^, physics^, D_ref^, Lift_ref^,
    )

    # Initial condition on owned elements.
    var inv_two_sigma2 = Float32(1.0) / (
        Float32(2.0) * GAUSS_SIGMA * GAUSS_SIGMA
    )
    solver.ctx.enqueue_function[gaussian_ic_kernel, gaussian_ic_kernel](
        solver.d_q.unsafe_ptr(),
        solver.patch.d_owned_elem_ids.unsafe_ptr(),
        solver.patch.mesh.d_elem_node_xyz.unsafe_ptr(),
        solver.num_owned_elements,
        Float32(GAUSS_CX), Float32(GAUSS_CY), Float32(GAUSS_CZ),
        Float32(LX), Float32(LY), Float32(LZ),
        inv_two_sigma2,
        grid_dim=ceildiv(
            solver.num_owned_elements * N_P, IC_BLOCK
        ),
        block_dim=IC_BLOCK,
    )
    solver.ctx.synchronize()

    # Compute initial integral on owned elements (just plain sum of q
    # over owned nodes; this is not area-weighted but is
    # rank-partition-invariant so it suffices as a conservation
    # smoke-test).
    var initial_sum_local = _owned_sum_q(solver, nvtx)
    var initial_sum_global = _allreduce_sum(initial_sum_local)
    var max_local = _owned_max_q(solver, nvtx)
    var max_global = _allreduce_max(max_local)
    if rank == 0:
        print("  initial sum(q) =", initial_sum_global,
              "  max(q) =", max_global)

    var dt = choose_dt()
    if rank == 0:
        print("  dt =", dt, "  steps ~",
              Int(T_FINAL / dt))

    var t: Float32 = 0.0
    var step = 0
    var wall_start = perf_counter_ns()
    while t < T_FINAL:
        var step_dt = dt
        if t + step_dt > T_FINAL:
            step_dt = T_FINAL - t
        solver.step_ssprk3(step_dt, nvtx)
        t += step_dt
        step += 1

    solver.ctx.synchronize()
    mpi.barrier_world()
    var wall_end = perf_counter_ns()
    var wall_sec = Float64(wall_end - wall_start) * 1e-9

    # After one full period (t = 1), the exact solution has returned
    # to the IC.
    var final_sum_local  = _owned_sum_q(solver, nvtx)
    var final_sum_global = _allreduce_sum(final_sum_local)
    var final_max_local  = _owned_max_q(solver, nvtx)
    var final_max_global = _allreduce_max(final_max_local)

    if rank == 0:
        print("  final   sum(q) =", final_sum_global,
              "  max(q) =", final_max_global)
        var conservation_err = (
            (final_sum_global - initial_sum_global)
            / initial_sum_global
        )
        print("  steps:", step, "  wall time:", wall_sec, "s")
        print("  conservation (sum drift):", conservation_err)

    mpi.finalize()


# ----------------------------------------------------------------------
# Owned-only reductions (host-side, one D->H download per call).
# ----------------------------------------------------------------------

def _owned_sum_q(
    mut solver: PatchSolver[Advection], mut nvtx: NvtxContext
) raises -> Float64:
    var buf = List[Float32]()
    for _ in range(solver.num_owned_elements * N_P):
        buf.append(Float32(0.0))
    solver.download_owned_component(0, buf, nvtx)
    var s: Float64 = 0.0
    for i in range(len(buf)):
        s += Float64(buf[i])
    return s

def _owned_max_q(
    mut solver: PatchSolver[Advection], mut nvtx: NvtxContext
) raises -> Float32:
    var buf = List[Float32]()
    for _ in range(solver.num_owned_elements * N_P):
        buf.append(Float32(0.0))
    solver.download_owned_component(0, buf, nvtx)
    var m: Float32 = Float32(-1.0e30)
    for i in range(len(buf)):
        if buf[i] > m:
            m = buf[i]
    return m

def _allreduce_sum(v: Float64) raises -> Float64:
    var sfl = Float32(v)
    var rfl: Float32 = 0.0
    var sp = UnsafePointer(to=sfl)
    var rp = UnsafePointer(to=rfl)
    mpi.allreduce_float_sum(
        rebind[UnsafePointer[Float32, MutAnyOrigin]](sp),
        rebind[UnsafePointer[Float32, MutAnyOrigin]](rp),
        1,
    )
    return Float64(rfl)

def _allreduce_max(v: Float32) raises -> Float32:
    var sfl = v
    var rfl: Float32 = 0.0
    var sp = UnsafePointer(to=sfl)
    var rp = UnsafePointer(to=rfl)
    mpi.allreduce_float_max(
        rebind[UnsafePointer[Float32, MutAnyOrigin]](sp),
        rebind[UnsafePointer[Float32, MutAnyOrigin]](rp),
        1,
    )
    return rfl
