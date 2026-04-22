# ======================================================================
# mpi_partition -- exercise the cube-grid Partition logic
# ======================================================================
#
# Each rank prints its assigned cube box and the 6 face neighbours.
# Run e.g. `mpirun -np 8 ./mpi_partition` and confirm:
#   * cube-box ranges tile the global (NX, NY, NZ) grid exactly,
#   * neighbour ranks form a periodic 3D torus.
# ======================================================================

from src import mpi
from src.partition import build_partition

comptime NX = 32
comptime NY = 32
comptime NZ = 32


def main() raises:
    mpi.init()
    var rank = mpi.world_rank()
    var size = mpi.world_size()

    var part = build_partition(rank, size, NX, NY, NZ)

    for r in range(size):
        if r == rank:
            print(
                "[rank", rank, "/", size, "]",
                " proc-grid (", part.px, ",", part.py, ",", part.pz, ")",
                " coords (", part.rx, ",", part.ry, ",", part.rz, ")",
                " cubes [", part.cx0, ",", part.cx1, ") x [",
                part.cy0, ",", part.cy1, ") x [",
                part.cz0, ",", part.cz1, ")",
                " owned=", part.num_owned_cubes(),
            )
            print(
                "  neighbours -x/+x/-y/+y/-z/+z =",
                part.neighbour_minus_x, part.neighbour_plus_x,
                part.neighbour_minus_y, part.neighbour_plus_y,
                part.neighbour_minus_z, part.neighbour_plus_z,
            )
        mpi.barrier_world()

    mpi.finalize()
