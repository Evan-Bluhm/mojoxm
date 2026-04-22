# ======================================================================
# mojoxm build
# ======================================================================
#
# Local (WSL2 / Linux workstation with system OpenMPI):
#     make              # build every driver
#     make nonmpi       # just the non-MPI drivers
#     make mpi          # just the MPI drivers
#     make mpi_hello    # just one driver by name
#     make clean
#
# Klone (Rocky 8 cluster + Apptainer; requires `mojo.sif` built via
# `apptainer build --fakeroot mojo.sif mojo.def`).  First build the
# MPI shim with the host toolchain, then invoke make with overrides:
#
#     LD_LIBRARY_PATH=/sw/gcc/13.2.0/lib64:/sw/gcc/13.2.0/lib:/sw/ompi/4.1.6-2/lib \
#     PATH=/sw/gcc/13.2.0/bin:/sw/ompi/4.1.6-2/bin:$PATH \
#     make \
#         MOJO='apptainer exec --nv --bind /sw --bind /gscratch mojo.sif mojo' \
#         MPICC=/sw/ompi/4.1.6-2/bin/mpicc \
#         MPI_LIBDIR=/sw/ompi/4.1.6-2/lib \
#         <target>
#
# On Klone, GPU-using drivers (advection_gaussian, euler_vortex,
# euler_taylor_green, mpi_advection_gaussian) must be built on a GPU
# compute node because Mojo elaborates GPU kernels at build time --
# wrap the `make` call in `srun -A ... --gres=gpu:... ...`.  See
# README for the exact srun recipe.
# ======================================================================

SHELL := /bin/bash

# Compiler invocations.  Override on the command line for cluster use.
MOJO        ?= .venv/bin/mojo
MPICC       ?= mpicc

# Default MPI_LIBDIR picks up the system convention for each platform:
#   * Linux (WSL2, Klone): Debian/Ubuntu multiarch path for OpenMPI.
#   * macOS: Homebrew's `open-mpi` cellar under /opt/homebrew on Apple
#     Silicon, /usr/local on Intel.  `brew --prefix open-mpi` returns
#     the right directory on either Homebrew layout.
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    MPI_LIBDIR ?= $(shell brew --prefix open-mpi 2>/dev/null)/lib
else
    MPI_LIBDIR ?= /usr/lib/x86_64-linux-gnu/openmpi/lib
endif

BUILD_DIR    = build

# Flags applied to every Mojo build.  `-g0` is load-bearing -- default
# debug info bloats the RK kernel's register count and kills L1/L2
# throughput.
MOJO_FLAGS   = -O3 -g0 -I .
LINK_BASE    = -Xlinker -lm -Xlinker -lpthread
LINK_MPI     = -Xlinker $(BUILD_DIR)/mpi_shim.o \
               -Xlinker -L$(MPI_LIBDIR) -Xlinker -lmpi

# Any src/ change rebuilds every driver.  Too coarse for granular
# incremental builds, but Mojo compiles a whole import graph per
# invocation anyway, so finer-grained deps wouldn't help.
SRC_MOJO    := $(wildcard src/*.mojo)

# Driver groupings.
#   NONMPI_GPU  -- single-rank drivers that use the GPU (need a GPU to
#                  build because Mojo elaborates the kernel at compile
#                  time).
#   MPI_CPU     -- MPI drivers that don't touch the GPU at all.  Safe
#                  to build on login nodes.
#   MPI_GPU     -- MPI drivers that instantiate PatchMesh / Solver.
#                  Need a GPU at build time.
NONMPI_GPU   = advection_gaussian euler_vortex euler_taylor_green
MPI_CPU      = mpi_hello mpi_partition
MPI_GPU      = mpi_patch_mesh mpi_halo_pingpong mpi_advection_gaussian
MPI          = $(MPI_CPU) $(MPI_GPU)
ALL_DRIVERS  = $(NONMPI_GPU) $(MPI)

# Test drivers live under test/; they use the same Physics / PatchSolver
# machinery as the examples/ drivers but emit per-rank binary dumps that
# the test harness diffs across rank counts.
TEST_DRIVERS = mpi_advection_test

.PHONY: all cpu gpu nonmpi mpi clean help test test-klone

help:
	@echo 'mojoxm build targets'
	@echo '  make                     build every driver (needs GPU for most)'
	@echo '  make cpu                 CPU-only drivers (login-node safe): $(MPI_CPU)'
	@echo '  make gpu                 GPU-using drivers (need GPU node)'
	@echo '  make nonmpi              single-rank GPU drivers: $(NONMPI_GPU)'
	@echo '  make mpi                 all MPI drivers: $(MPI)'
	@echo '  make <driver>            build a single driver by name'
	@echo '  make shim                build just build/mpi_shim.o'
	@echo '  make test                run MPI correctness test (np=1 vs np=4)'
	@echo '  make test-klone          run MPI correctness test on Klone'
	@echo '  make clean               remove driver binaries + $(BUILD_DIR)/'
	@echo ''
	@echo 'Overrides (use for Klone + Apptainer):'
	@echo "  MOJO='apptainer exec --nv --bind /sw --bind /gscratch mojo.sif mojo'"
	@echo '  MPICC=/sw/ompi/4.1.6-2/bin/mpicc'
	@echo '  MPI_LIBDIR=/sw/ompi/4.1.6-2/lib'

all: $(ALL_DRIVERS)
cpu: $(MPI_CPU)
gpu: $(NONMPI_GPU) $(MPI_GPU)
nonmpi: $(NONMPI_GPU)
mpi: $(MPI)
shim: $(BUILD_DIR)/mpi_shim.o

# Static pattern rule: each non-MPI driver is built from the matching
# examples/<name>.mojo file plus anything in src/.
$(NONMPI_GPU): %: examples/%.mojo $(SRC_MOJO)
	$(MOJO) build $(MOJO_FLAGS) $< -o $@ $(LINK_BASE)

# MPI drivers additionally depend on the shim object file.
$(MPI): %: examples/%.mojo $(BUILD_DIR)/mpi_shim.o $(SRC_MOJO)
	$(MOJO) build $(MOJO_FLAGS) $< -o $@ $(LINK_MPI) $(LINK_BASE)

# Test drivers use the same link line as MPI drivers but source from test/.
$(TEST_DRIVERS): %: test/%.mojo $(BUILD_DIR)/mpi_shim.o $(SRC_MOJO)
	$(MOJO) build $(MOJO_FLAGS) $< -o $@ $(LINK_MPI) $(LINK_BASE)

# `make test` builds the test driver locally and runs the np=1 vs np=4
# correctness script.  `make test-klone` reroutes through scripts/klone-run
# so the build happens inside the GPU srun allocation that will run it
# (no up-front build dependency: klone-run rebuilds each invocation).
test: $(TEST_DRIVERS)
	test/test_mpi_correctness.sh

test-klone:
	test/test_mpi_correctness.sh --klone

$(BUILD_DIR)/mpi_shim.o: src/mpi_shim.c | $(BUILD_DIR)
	$(MPICC) -O2 -fPIC -c $< -o $@

$(BUILD_DIR):
	mkdir -p $@

clean:
	rm -f $(ALL_DRIVERS) $(TEST_DRIVERS)
	rm -rf $(BUILD_DIR) test/dumps_np1 test/dumps_np4
