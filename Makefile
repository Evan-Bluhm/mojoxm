# ======================================================================
# mojoxm build
# ======================================================================
#
# Every driver links libmpi now -- a single-rank run is just np=1 under
# MPI, not a separate code path.  So the classical "MPI vs non-MPI"
# distinction is gone; what remains is whether the driver touches the
# GPU (every DG solver driver does) or is a pure CPU smoke test
# (mpi_hello, mpi_partition).
#
# Local (WSL2 / Linux workstation with system OpenMPI):
#     make              # build every driver
#     make gpu          # just the GPU-using drivers
#     make cpu          # just the CPU-only smoke tests (login-node safe)
#     make <driver>     # just one driver by name
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
# On Klone, GPU-using drivers must be built on a GPU compute node
# because Mojo elaborates GPU kernels at build time -- wrap the `make`
# call in `srun -A ... --gres=gpu:... ...`.  `scripts/klone-run` does
# this automatically.  See README for the exact srun recipe.
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
# Every driver links libmpi + the C shim, because every driver calls
# mpi.init() / mpi.finalize() (np=1 is just an MPI communicator of
# size 1).
LINK_BASE    = -Xlinker $(BUILD_DIR)/mpi_shim.o \
               -Xlinker -L$(MPI_LIBDIR) -Xlinker -lmpi \
               -Xlinker -lm -Xlinker -lpthread

# Any src/ change rebuilds every driver.  Too coarse for granular
# incremental builds, but Mojo compiles a whole import graph per
# invocation anyway, so finer-grained deps wouldn't help.
SRC_MOJO    := $(wildcard src/*.mojo)

# Driver groupings.
#   CPU_DRIVERS -- MPI drivers that don't touch the GPU at all.  Safe
#                  to build on login nodes.  No simulations here: we
#                  solve exclusively on the GPU.
#   GPU_DRIVERS -- every simulation driver.  All require a GPU at
#                  build time (Mojo elaborates the kernels at compile
#                  time) and at run time.
CPU_DRIVERS  = mpi_hello mpi_partition
GPU_DRIVERS  = advection_gaussian euler_vortex euler_taylor_green euler_sod \
               euler_rising_bubble maxwell_cavity shallow_water_drop \
               mhd_alfven two_fluid_langmuir \
               advection_gaussian_2d_gpu euler_vortex_2d_gpu \
               shallow_water_drop_2d_gpu mhd_alfven_2d_gpu \
               euler_channel_2d_gpu shallow_water_dam_break_2d_gpu \
               advection_outflow_2d_gpu euler_sod_2d_gpu \
               mpi_patch_mesh mpi_halo_pingpong
ALL_DRIVERS  = $(CPU_DRIVERS) $(GPU_DRIVERS)

# Test drivers live under test/; they validate GPU kernels against
# self-consistent invariants (constant-state preservation, upload
# round-trips) or against known analytic solutions.  All require a
# GPU.
TEST_DRIVERS = mpi_advection_test mpi_bc_test diagnostics_test p3_smoke_test \
               local_mesh_2d_gpu_test euler_2d_gpu_test sw_2d_gpu_test \
               mhd_2d_gpu_test limiter_2d_gpu_test

.PHONY: all cpu gpu clean help test test-bc test-reference test-reference-2d test-local-mesh-2d test-local-mesh-2d-gpu test-euler-2d-gpu test-sw-2d-gpu test-mhd-2d-gpu test-limiter-2d-gpu test-diagnostics test-p3 test-all test-klone

help:
	@echo 'mojoxm build targets'
	@echo '  make                     build every driver (needs GPU for most)'
	@echo '  make cpu                 CPU-only smoke tests (login-node safe): $(CPU_DRIVERS)'
	@echo '  make gpu                 GPU-using drivers: $(GPU_DRIVERS)'
	@echo '  make <driver>            build a single driver by name'
	@echo '  make shim                build just build/mpi_shim.o'
	@echo '  make test                run MPI correctness test (np=1 vs np=4, periodic)'
	@echo '  make test-bc             run BC correctness test (non-periodic, np=1 vs np=4)'
	@echo '  make test-reference      run reference-element unit test (P=1..4, host-side)'
	@echo '  make test-diagnostics    run DiagnosticsWriter unit test (GPU, np=1)'
	@echo '  make test-all            run every test above'
	@echo '  make test-klone          run MPI correctness test on Klone'
	@echo '  make clean               remove driver binaries + $(BUILD_DIR)/'
	@echo ''
	@echo 'Overrides (use for Klone + Apptainer):'
	@echo "  MOJO='apptainer exec --nv --bind /sw --bind /gscratch mojo.sif mojo'"
	@echo '  MPICC=/sw/ompi/4.1.6-2/bin/mpicc'
	@echo '  MPI_LIBDIR=/sw/ompi/4.1.6-2/lib'

all: $(ALL_DRIVERS)
cpu: $(CPU_DRIVERS)
gpu: $(GPU_DRIVERS)
shim: $(BUILD_DIR)/mpi_shim.o

# Static pattern rule: every driver is built from examples/<name>.mojo,
# the MPI shim, and whatever's in src/.
$(ALL_DRIVERS): %: examples/%.mojo $(BUILD_DIR)/mpi_shim.o $(SRC_MOJO)
	$(MOJO) build $(MOJO_FLAGS) $< -o $@ $(LINK_BASE)

# Test drivers use the same link line but source from test/.
$(TEST_DRIVERS): %: test/%.mojo $(BUILD_DIR)/mpi_shim.o $(SRC_MOJO)
	$(MOJO) build $(MOJO_FLAGS) $< -o $@ $(LINK_BASE)

# `make test` builds the test driver locally and runs the np=1 vs np=4
# correctness script.  `make test-klone` reroutes through scripts/klone-run
# so the build happens inside the GPU srun allocation that will run it
# (no up-front build dependency: klone-run rebuilds each invocation).
test: $(TEST_DRIVERS)
	test/test_mpi_correctness.sh

# Same structure as `make test`, but routes through mpi_bc_test which
# uses BC_OUTFLOW on all 6 domain faces -- exercises the non-periodic
# BC code path through the full single-rank-vs-multi-rank diff.
test-bc: $(TEST_DRIVERS)
	test/test_mpi_bc_correctness.sh

# Host-side reference-element unit test: validates the 3D Lagrange
# basis construction (Vandermonde + analytic integration) at orders
# P=1..4 via SPD mass matrix + node-position + face-to-element map
# checks.  Host math only, but the same tables are uploaded to the
# GPU by ReferenceElement[P].
test-reference:
	.venv/bin/mojo run -I . test/reference_element_test.mojo

# Same for the 2D triangular reference element (ReferenceElement2D[P]).
# Covers node positions, SPD 2D mass matrix, edge-to-element map.
# ReferenceElement2DGpu uploads from this; validating the host tables
# catches the bulk of basis-construction bugs before a GPU run.
test-reference-2d:
	.venv/bin/mojo run -I . test/reference_element_2d_test.mojo

# 2D triangulated Cartesian mesh topology: element / face counts,
# elem_faces <-> face_elem round-trip, side-0 / side-1 node coordinate
# agreement across shared edges, Jacobian positivity.  The mesh
# LocalMesh2D uploads is built here; catching topology errors on the
# host side avoids expensive GPU debugging.
test-local-mesh-2d:
	.venv/bin/mojo run -I . test/local_mesh_2d_test.mojo

# GPU diagnostics writer test: uniform-field integrals recover
# analytic values; max_abs reports the peak on a checkerboard field;
# empty configuration doesn't crash.  Runs at np=1.
test-diagnostics: diagnostics_test
	./diagnostics_test

# P=3 plumbing smoke test.  Builds Mesh[3] + Solver[Advection, 3],
# fills q with a constant, downloads it back, verifies round-trip at
# the higher-order buffer layout (NP=20).  Does NOT check physical
# accuracy; it only validates that every buffer allocation and GPU
# kernel launch is correctly parameterized for P != 2.  Runs at np=1.
test-p3: p3_smoke_test
	./p3_smoke_test

# 2D GPU mesh upload smoke test (task #19 foundation).  Builds a host
# LocalMesh2D[P], wraps it in LocalMesh2DGpu[P], and downloads a few
# buffers to verify the Float64 -> Float32 conversion + Int32 transfer
# round-trip cleanly.  No GPU kernels yet -- just the buffer plumbing.
test-local-mesh-2d-gpu: local_mesh_2d_gpu_test
	./local_mesh_2d_gpu_test

# Per-physics GPU tests (split out of the monolithic local_mesh_2d_gpu_test
# so the Mojo compiler's comptime-specialization working set stays
# bounded).  Each validates one physics's GPU path vs the CPU Float64
# reference in a one-SSPRK3-step diff at P=2.
test-euler-2d-gpu: euler_2d_gpu_test
	./euler_2d_gpu_test
test-sw-2d-gpu: sw_2d_gpu_test
	./sw_2d_gpu_test
test-mhd-2d-gpu: mhd_2d_gpu_test
	./mhd_2d_gpu_test
test-limiter-2d-gpu: limiter_2d_gpu_test
	./limiter_2d_gpu_test

# Convenience target: run every test in the suite.  Stops on the first
# failure.  Doesn't include test-klone (that's for cluster submission).
test-all: test-reference test-reference-2d test-local-mesh-2d test-local-mesh-2d-gpu test-euler-2d-gpu test-sw-2d-gpu test-mhd-2d-gpu test-limiter-2d-gpu test-diagnostics test-p3 test test-bc
	@echo '=== ALL TESTS PASSED ==='

test-klone:
	test/test_mpi_correctness.sh --klone

$(BUILD_DIR)/mpi_shim.o: src/mpi_shim.c | $(BUILD_DIR)
	$(MPICC) -O2 -fPIC -c $< -o $@

$(BUILD_DIR):
	mkdir -p $@

clean:
	rm -f $(ALL_DRIVERS) $(TEST_DRIVERS)
	rm -rf $(BUILD_DIR) test/dumps_np1 test/dumps_np4 output
