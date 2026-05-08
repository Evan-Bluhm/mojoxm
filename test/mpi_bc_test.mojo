# ======================================================================
# mpi_bc_test -- MPI correctness for non-periodic boundary conditions
# ======================================================================
#
# Same structural idea as mpi_advection_test (50-step integration +
# per-rank binary dump) but with BC_OUTFLOW on all 6 domain faces
# instead of a fully periodic torus.  The point is to exercise
# boundary_flux + the per-rank BC filtering in Mesh + the
# skip_mpi flag in HaloExchange, and verify that np=1 and np>1 produce
# identical owned-element q fields to FP precision.
#
# Output: same binary format as mpi_advection_test (see that file's
# header for the layout).  Uses a separate file-name prefix so the
# dumps don't collide with a parallel periodic run.
# ======================================================================

from src import mpi
from src.partition import build_partition
from src.mesh import Mesh
from src.boundary import BoundaryConditions, BC_OUTFLOW
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

comptime NUM_TEST_STEPS = 50
comptime _OPEN_MODE = c_int(0o644)


def gaussian_ic_kernel(
    q: UnsafePointer[Float32, MutAnyOrigin],
    owned_elem_ids: UnsafePointer[Int32, MutAnyOrigin],
    elem_node_xyz: UnsafePointer[Float32, MutAnyOrigin],
    num_owned: Int,
    cx: Float32,
    cy: Float32,
    cz: Float32,
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
    var dy = py - cy
    var dz = pz - cz
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

    var path_s = String("output/final_q_rank_")
    path_s += String(rank)
    path_s += String(".bin")

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

    var header = InlineArray[UInt32, 5](fill=UInt32(0))
    header[0] = UInt32(0x514D584D)
    header[1] = UInt32(1)
    header[2] = UInt32(num_owned)
    header[3] = UInt32(1)
    header[4] = UInt32(N_P)
    var header_ptr = rebind[UnsafePointer[UInt8, MutAnyOrigin]](
        header.unsafe_ptr()
    )
    _write_bytes(fd, header_ptr, 5 * 4)

    var ids_ptr = rebind[UnsafePointer[UInt8, MutAnyOrigin]](
        id_buf.unsafe_ptr().bitcast[UInt8]()
    )
    _write_bytes(fd, ids_ptr, num_owned * 4)

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
            "mpi_bc_test:",
            NUM_TEST_STEPS,
            "step dump with BC_OUTFLOW on all 6 sides, ",
            size,
            "ranks",
        )

    var nvtx = NvtxContext()
    var ctx = DeviceContext()
    var re = ReferenceElement()
    var D_ref = to_float32(re.D_ref)
    var Lift_ref = to_float32(re.Lift_ref)
    var node_weights = to_float32(re.node_weights)

    # Outflow on all 6 domain faces.  At np=1 the Mesh applies BC
    # overlays on every external face; at np>1 the Mesh filter keeps
    # only the sides where THIS rank sits on the global boundary, and
    # the HaloExchange skip_mpi flag suppresses pack/Isend/Irecv on
    # those same rings.
    var bcs = BoundaryConditions(
        BC_OUTFLOW,
        BC_OUTFLOW,
        BC_OUTFLOW,
        BC_OUTFLOW,
        BC_OUTFLOW,
        BC_OUTFLOW,
    )

    var mesh = Mesh(
        ctx,
        build_partition(rank, size, NX, NY, NZ),
        LX,
        LY,
        LZ,
        bcs,
    )
    var halo = HaloExchange(
        ctx,
        mesh.part,
        Advection.NUM_COMPONENTS,
        mesh.d_perm.unsafe_ptr(),
        bcs,
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
