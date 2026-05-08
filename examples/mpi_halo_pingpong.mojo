# ======================================================================
# mpi_halo_pingpong -- correctness test for the halo exchange
# ======================================================================
#
# Each rank initialises its owned q values to a recognisable pattern
# (owner_rank * 1e6 + owned_elem_index_in_rank) and zeros its ghost
# ring.  After one halo exchange, each ghost cube should hold its
# *neighbour's* pattern, which we check by comparing the first few
# values per face-ring and confirming they decode to the expected
# (neighbour_rank, owned_elem_index_in_neighbour).
#
# Run:
#   mpirun -np 8 ./mpi_halo_pingpong
# ======================================================================

from src import mpi
from src.partition import build_partition
from src.mesh import Mesh
from src.boundary import BoundaryConditions
from src.halo_exchange import HaloExchange
from std.gpu.host import DeviceContext
from std.sys import has_accelerator
from std.gpu import global_idx
from std.math import ceildiv

comptime NX = 32
comptime NY = 32
comptime NZ = 32
comptime LX = 1.0
comptime NC = 1  # scalar field for this test

comptime IC_BLOCK = 256


def init_q_kernel(
    q: UnsafePointer[Float32, MutAnyOrigin],
    owned_ids: UnsafePointer[Int32, MutAnyOrigin],
    num_owned: Int,
    total_dof: Int,
    rank_f: Float32,
):
    var tid = Int(global_idx.x)
    if tid >= total_dof:
        return
    # Default: zero for any DOF (ghost).  Owned DOFs get overwritten.
    q[tid] = Float32(0.0)


def fill_owned_kernel(
    q: UnsafePointer[Float32, MutAnyOrigin],
    owned_ids: UnsafePointer[Int32, MutAnyOrigin],
    num_owned: Int,
    rank_f: Float32,
):
    var idx = Int(global_idx.x)
    if idx >= num_owned * 10:
        return
    var i = idx // 10
    var nn = idx % 10
    var elem = Int(owned_ids[i])
    q[(elem * 10 + nn)] = rank_f * Float32(1.0e6) + Float32(i)


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    mpi.init()
    var rank = mpi.world_rank()
    var size = mpi.world_size()

    var ctx = DeviceContext()
    var patch = Mesh(
        ctx,
        build_partition(rank, size, NX, NY, NZ),
        LX,
        LX,
        LX,
        BoundaryConditions.periodic(),
    )
    var halo = HaloExchange(
        ctx,
        patch.part,
        NC,
        patch.d_perm.unsafe_ptr(),
    )

    # Allocate scalar q on the local mesh (NC=1).
    var total_dof = patch.local.num_elements * 10 * NC
    var d_q = ctx.enqueue_create_buffer[DType.float32](total_dof)
    ctx.enqueue_function[init_q_kernel](
        d_q.unsafe_ptr(),
        patch.d_owned_elem_ids.unsafe_ptr(),
        patch.num_owned_elements,
        total_dof,
        Float32(rank),
        grid_dim=ceildiv(total_dof, IC_BLOCK),
        block_dim=IC_BLOCK,
    )
    # Then fill owned DOFs with the rank-identifying pattern.
    ctx.enqueue_function[fill_owned_kernel](
        d_q.unsafe_ptr(),
        patch.d_owned_elem_ids.unsafe_ptr(),
        patch.num_owned_elements,
        Float32(rank),
        grid_dim=ceildiv(patch.num_owned_elements * 10, IC_BLOCK),
        block_dim=IC_BLOCK,
    )
    ctx.synchronize()

    # Do one halo exchange.
    halo.exchange(ctx, d_q.unsafe_ptr())

    # Download q and spot-check the ghost values for each face ring.
    # For direction d, the first ghost element in the unpack list is
    # at (lcx=0 or nx+1, lcy=1, lcz=1) depending on axis.  The
    # corresponding sender's first packed element is at the
    # neighbour's *opposite* boundary cube (e.g. our -x ghost came from
    # the neighbour's +x owned boundary at (nx, 1, 1)).  The packed
    # ordering uses the same (lcy, lcz) walk on both sides, so our
    # ghost's DOF[0] should hold neigh_rank * 1e6 + 0.

    var h_q = ctx.enqueue_create_host_buffer[DType.float32](total_dof)
    ctx.enqueue_copy(h_q, d_q)
    ctx.synchronize()
    var p = h_q.unsafe_ptr()

    var nx = patch.part.nx
    var ny = patch.part.ny
    var nz = patch.part.nz
    var loc_nx = nx + 2
    var loc_ny = ny + 2

    def _dof_at(cube_cell: Int, tet: Int, nn: Int) capturing -> Int:
        return (cube_cell * 6 + tet) * 10 + nn

    var neighbours = [
        patch.part.neighbour_minus_x,
        patch.part.neighbour_plus_x,
        patch.part.neighbour_minus_y,
        patch.part.neighbour_plus_y,
        patch.part.neighbour_minus_z,
        patch.part.neighbour_plus_z,
    ]
    var names = [
        String("-x"),
        String("+x"),
        String("-y"),
        String("+y"),
        String("-z"),
        String("+z"),
    ]

    # Ghost-ring first-cube local cell index per direction.
    # -x: (lcx=0, lcy=1, lcz=1) -> cell = 0 + loc_nx*(1 + loc_ny*1)
    # +x: (lcx=nx+1, 1, 1)
    # -y: (1, 0, 1)
    # +y: (1, ny+1, 1)
    # -z: (1, 1, 0)
    # +z: (1, 1, nz+1)
    var cells = [
        0 + loc_nx * (1 + loc_ny * 1),
        (nx + 1) + loc_nx * (1 + loc_ny * 1),
        1 + loc_nx * (0 + loc_ny * 1),
        1 + loc_nx * ((ny + 1) + loc_ny * 1),
        1 + loc_nx * (1 + loc_ny * 0),
        1 + loc_nx * (1 + loc_ny * (nz + 1)),
    ]

    # Check every ghost DOF: extract the encoded (sender_rank) via
    # integer division by 1e6.  It must equal the expected neighbour
    # rank for that direction.  Also track mismatches.
    for r in range(size):
        if r == rank:
            var any_fail = False
            for d in range(6):
                var got = Float32(p[_dof_at(cells[d], 0, 0)])
                var encoded_rank = Int(got / Float32(1.0e6))
                var encoded_offset = Int(got) - encoded_rank * 1000000
                var expected_rank = neighbours[d]
                var ok = encoded_rank == expected_rank
                if not ok:
                    any_fail = True
                print(
                    "[rank",
                    rank,
                    "]",
                    names[d],
                    " ghost_dof0=",
                    got,
                    " -> sender_rank=",
                    encoded_rank,
                    " (expected ",
                    expected_rank,
                    ")",
                    " offset=",
                    encoded_offset,
                    " ok=",
                    ok,
                )
            if not any_fail:
                print("[rank", rank, "] all 6 directions OK")
        mpi.barrier_world()

    mpi.finalize()
