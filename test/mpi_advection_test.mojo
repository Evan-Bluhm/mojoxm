# ======================================================================
# mpi_advection_test -- 50-step MPI correctness driver
# ======================================================================
#
# Runs scalar advection for exactly 50 SSPRK3 steps and dumps every
# rank's owned q field (plus each element's global-mesh id) to a
# per-rank binary file.  test/test_mpi_correctness.sh builds this
# driver, runs it at np=1 and np=4, and diffs the two dumps to
# confirm identical physics across rank counts.
#
# Output (per rank):
#     output/final_q_rank_<rank>.bin
#
# Binary format:
#     u32     magic  = 0x514D584D  ('MXMQ' little-endian)
#     u32     version = 1
#     u32     num_owned_elements
#     u32     num_components (= 1 for advection)
#     u32     nodes_per_elem (= N_P = 10)
#     u32[num_owned] global_elem_ids  -- order matches q values below
#     f32[num_owned * N_P * NC] q values
#
# ======================================================================

from src import mpi
from src.partition import build_partition
from src.mesh import Mesh
from src.boundary import BoundaryConditions
from src.halo_exchange import HaloExchange
from src.solver import Solver
from src.reference import ReferenceElement, N_P, to_float32
from src.advection import Advection
from src.nvtx import NvtxContext
from std.ffi import external_call, c_int, c_size_t, c_ssize_t
from std.memory import alloc
from std.sys import has_accelerator
from std.gpu import global_idx
from std.gpu.host import DeviceContext
from std.math import sqrt, ceildiv, exp

comptime NX = 32
comptime NY = 32
comptime NZ = 32
comptime LX = 1.0
comptime LY = 1.0
comptime LZ = 1.0

comptime VX: Float32 = 1.0
comptime VY: Float32 = 1.0
comptime VZ: Float32 = 1.0

comptime CFL = Float32(0.2)

comptime GAUSS_CX: Float32 = 0.5
comptime GAUSS_CY: Float32 = 0.5
comptime GAUSS_CZ: Float32 = 0.5
comptime GAUSS_SIGMA: Float32 = 0.12

comptime IC_BLOCK = 256

# Target: exactly NUM_TEST_STEPS SSPRK3 steps, short enough that np=1
# and np=4 agree to machine precision yet long enough (150 kernel
# launches, 150 halo exchanges) to catch any comm / ordering bug.
comptime NUM_TEST_STEPS = 50

# We use creat(path, mode) -- equivalent to open(path, O_WRONLY | O_CREAT
# | O_TRUNC, mode) -- instead of a 3-arg open() because open() is
# variadic (int open(const char *, int, ...)).  On ARM64 Apple Darwin,
# variadic args follow a different ABI from fixed args (variadic args
# are promoted to the stack), and Mojo's external_call signature is
# fixed-arity, so passing the mode through a 3-arg open() call delivers
# garbage to the kernel and the file ends up with bits like 0o300
# instead of the intended 0o644.  creat() is POSIX and has a fixed
# 2-arg signature that round-trips cleanly through external_call.
comptime _OPEN_MODE = c_int(0o644)


def gaussian_ic_kernel(
    q: UnsafePointer[Float32, MutAnyOrigin],
    owned_elem_ids: UnsafePointer[Int32, MutAnyOrigin],
    elem_node_xyz: UnsafePointer[Float32, MutAnyOrigin],
    num_owned: Int,
    cx: Float32,
    cy: Float32,
    cz: Float32,
    Lx: Float32,
    Ly: Float32,
    Lz: Float32,
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
    if dx > Lx * Float32(0.5):
        dx -= Lx
    if dx < -Lx * Float32(0.5):
        dx += Lx
    var dy = py - cy
    if dy > Ly * Float32(0.5):
        dy -= Ly
    if dy < -Ly * Float32(0.5):
        dy += Ly
    var dz = pz - cz
    if dz > Lz * Float32(0.5):
        dz -= Lz
    if dz < -Lz * Float32(0.5):
        dz += Lz
    q[e * N_P + nn] = exp(-(dx * dx + dy * dy + dz * dz) * inv_two_sigma2)


def choose_dt() raises -> Float32:
    var h = Float32(LX) / Float32(NX)
    var v = sqrt(VX * VX + VY * VY + VZ * VZ)
    return CFL * h / (v * Float32(2 * 2 + 1))


def _write_bytes(
    fd: Int,
    buf: UnsafePointer[UInt8, MutAnyOrigin],
    n: Int,
) raises:
    var remaining = n
    var p = buf
    while remaining > 0:
        var wrote = Int(
            external_call["write", c_ssize_t](fd, p, c_size_t(remaining))
        )
        if wrote <= 0:
            raise Error("write() failed while dumping final q")
        remaining -= wrote
        p = p + wrote


def dump_final_q(
    mut solver: Solver[Advection],
    rank: Int,
    nx_global: Int,
    ny_global: Int,
    nz_global: Int,
    mut nvtx: NvtxContext,
) raises:
    # Gather per-rank q + global element ids into host buffers.
    var num_owned = solver.num_owned_elements
    var q_buf = List[Float32]()
    for _ in range(num_owned * N_P):
        q_buf.append(Float32(0.0))
    var id_buf = List[Int32]()
    for _ in range(num_owned):
        id_buf.append(Int32(0))
    solver.download_owned_component_with_ids(
        0,
        q_buf,
        id_buf,
        nx_global,
        ny_global,
        nz_global,
        nvtx,
    )

    # Build path `output/final_q_rank_<rank>.bin`.
    var path_s = String("output/final_q_rank_")
    path_s += String(rank)
    path_s += String(".bin")

    # Null-terminated C string for open().
    var pn = path_s.byte_length()
    var path_c = alloc[UInt8](pn + 1)
    for i in range(pn):
        path_c[i] = UInt8(path_s.unsafe_ptr()[i])
    path_c[pn] = 0

    var fd = Int(
        external_call["creat", c_int](
            path_c,
            _OPEN_MODE,
        )
    )
    if fd < 0:
        path_c.free()
        raise Error("creat() failed for " + path_s)

    # Write the file header.
    var header = InlineArray[UInt32, 5](fill=UInt32(0))
    header[0] = UInt32(0x514D584D)  # "MXMQ" in little-endian
    header[1] = UInt32(1)  # version
    header[2] = UInt32(num_owned)
    header[3] = UInt32(1)  # NC for scalar advection
    header[4] = UInt32(N_P)
    var header_ptr = rebind[UnsafePointer[UInt8, MutAnyOrigin]](
        header.unsafe_ptr()
    )
    _write_bytes(fd, header_ptr, 5 * 4)

    # Global element IDs.
    var ids_ptr = rebind[UnsafePointer[UInt8, MutAnyOrigin]](
        id_buf.unsafe_ptr().bitcast[UInt8]()
    )
    _write_bytes(fd, ids_ptr, num_owned * 4)

    # q values (Float32).
    var q_ptr = rebind[UnsafePointer[UInt8, MutAnyOrigin]](
        q_buf.unsafe_ptr().bitcast[UInt8]()
    )
    _write_bytes(fd, q_ptr, num_owned * N_P * 4)

    _ = external_call["close", c_int](c_int(fd))
    path_c.free()


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    mpi.init()
    var rank = mpi.world_rank()
    var size = mpi.world_size()

    if rank == 0:
        print(
            "mpi_advection_test:",
            NUM_TEST_STEPS,
            "step dump for correctness check, ",
            size,
            "ranks",
        )

    var nvtx = NvtxContext()
    var ctx = DeviceContext()
    var re = ReferenceElement()
    var D_ref = to_float32(re.D_ref)
    var Lift_ref = to_float32(re.Lift_ref)
    var node_weights = to_float32(re.node_weights)

    var mesh = Mesh(
        ctx,
        build_partition(rank, size, NX, NY, NZ),
        LX,
        LY,
        LZ,
        BoundaryConditions.periodic(),
    )
    var halo = HaloExchange(
        ctx,
        mesh.part,
        Advection.NUM_COMPONENTS,
        mesh.d_perm.unsafe_ptr(),
    )
    var physics = Advection(VX, VY, VZ)
    var solver = Solver[Advection](
        ctx^,
        mesh^,
        halo^,
        physics^,
        D_ref^,
        Lift_ref^,
        node_weights^,
    )

    var inv_two_sigma2 = Float32(1.0) / (
        Float32(2.0) * GAUSS_SIGMA * GAUSS_SIGMA
    )
    solver.ctx.enqueue_function[gaussian_ic_kernel, gaussian_ic_kernel](
        solver.d_q.unsafe_ptr(),
        solver.mesh.d_owned_elem_ids.unsafe_ptr(),
        solver.mesh.local.d_elem_node_xyz.unsafe_ptr(),
        solver.num_owned_elements,
        Float32(GAUSS_CX),
        Float32(GAUSS_CY),
        Float32(GAUSS_CZ),
        Float32(LX),
        Float32(LY),
        Float32(LZ),
        inv_two_sigma2,
        grid_dim=ceildiv(solver.num_owned_elements * N_P, IC_BLOCK),
        block_dim=IC_BLOCK,
    )
    solver.ctx.synchronize()

    var dt = choose_dt()
    if rank == 0:
        print("  dt =", dt, "  running", NUM_TEST_STEPS, "steps")
    for _ in range(NUM_TEST_STEPS):
        solver.step_ssprk3(dt, nvtx)
    solver.ctx.synchronize()
    mpi.barrier_world()

    dump_final_q(solver, rank, NX, NY, NZ, nvtx)
    mpi.barrier_world()
    if rank == 0:
        print("  wrote output/final_q_rank_*.bin")

    mpi.finalize()
