# ======================================================================
# mpi_hello -- minimal MPI bring-up smoke test
# ======================================================================
#
# Verifies the libmpi FFI: each rank initialises MPI, prints its rank
# and the world size, hits a barrier so the prints don't interleave
# completely arbitrarily, and finalises.
#
# Build:
#   mpicc -O2 -fPIC -c src/mpi_shim.c -o build/mpi_shim.o
#   .venv/bin/mojo build -O3 -g0 examples/mpi_hello.mojo -o mpi_hello \
#       -Xlinker build/mpi_shim.o \
#       -Xlinker -L/usr/lib/x86_64-linux-gnu/openmpi/lib \
#       -Xlinker -lmpi \
#       -Xlinker -lm -Xlinker -lpthread
#
# Run:
#   mpirun -np 4 ./mpi_hello
# ======================================================================

from src import mpi


def main() raises:
    mpi.init()
    var rank = mpi.world_rank()
    var size = mpi.world_size()
    # Ordered prints by barrier-walking ranks.
    for r in range(size):
        if r == rank:
            print("[rank", rank, "of", size, "] hello from mojoxm")
        mpi.barrier_world()
    mpi.finalize()
