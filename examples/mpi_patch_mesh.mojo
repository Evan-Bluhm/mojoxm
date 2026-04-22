# ======================================================================
# mpi_patch_mesh -- bring-up test for per-rank PatchMesh
# ======================================================================
#
# Each rank builds its own PatchMesh for a chosen global grid and
# prints:
#   * the partition's owned cube box,
#   * the local mesh's (nx+2, ny+2, nz+2) dimensions,
#   * the owned element count (should equal nx*ny*nz*6),
#   * a sample of owned-element node coordinates to spot-check that
#     each patch lies in the right physical region.
#
# Run:
#   mpirun -np 8 ./mpi_patch_mesh
# ======================================================================

from src import mpi
from src.partition import build_partition
from src.patch_mesh import PatchMesh
from std.gpu.host import DeviceContext
from std.sys import has_accelerator

comptime NX = 32
comptime NY = 32
comptime NZ = 32
comptime LX = 1.0
comptime LY = 1.0
comptime LZ = 1.0


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    mpi.init()
    var rank = mpi.world_rank()
    var size = mpi.world_size()

    var ctx = DeviceContext()
    var patch = PatchMesh(
        ctx, build_partition(rank, size, NX, NY, NZ), LX, LY, LZ
    )

    for r in range(size):
        if r == rank:
            print(
                "[rank", rank, "/", size, "]",
                " owned cubes (", patch.part.nx, "x",
                patch.part.ny, "x", patch.part.nz, ")",
                " -> local mesh (", patch.part.nx + 2, "x",
                patch.part.ny + 2, "x", patch.part.nz + 2, ")",
                " num_owned=", patch.num_owned_elements,
                " (halo=", patch.num_halo_elements,
                ", interior=", patch.num_interior_elements, ")",
            )
        mpi.barrier_world()

    mpi.finalize()
