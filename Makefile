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
#                  to build on login nodes.
#   GPU_DRIVERS -- drivers that instantiate Mesh / Solver and therefore
#                  require a GPU at build time (Mojo elaborates the RK
#                  kernel at compile time).
CPU_DRIVERS  = mpi_hello mpi_partition
GPU_DRIVERS  = advection_gaussian euler_vortex euler_taylor_green euler_sod \
               euler_rising_bubble maxwell_cavity shallow_water_drop \
               mhd_alfven two_fluid_langmuir \
               advection_gaussian_2d_gpu \
               mpi_patch_mesh mpi_halo_pingpong
ALL_DRIVERS  = $(CPU_DRIVERS) $(GPU_DRIVERS)

# Test drivers live under test/; they use the same Physics / Solver
# machinery as the examples/ drivers but emit per-rank binary dumps
# that the test harness diffs across rank counts.
TEST_DRIVERS = mpi_advection_test mpi_bc_test diagnostics_test p3_smoke_test local_mesh_2d_gpu_test

.PHONY: all cpu gpu clean help test test-bc test-reference test-reference-2d test-local-mesh-2d test-dg-rhs-2d test-advection-step-2d test-euler2d test-shallow-water-2d test-mesh-2d-bc test-sw-bc-dynamics test-time-integrators-2d test-convergence-2d test-bc-inflow-2d test-ideal-mhd-2d test-local-mesh-2d-gpu test-diagnostics test-p3 test-all test-klone

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

# Host-side reference-element unit test: validates Lagrange basis
# construction (Vandermonde + analytic integration) at orders P=1..4
# via SPD mass matrix + node-position + face-to-element map checks.
# No GPU, no MPI -- runs as a plain `mojo run`.
test-reference:
	.venv/bin/mojo run -I . test/reference_element_test.mojo

# Same for the 2D triangular reference element (ReferenceElement2D[P]).
# Covers node positions, SPD 2D mass matrix, edge-to-element map.
# Host-only; future 2D mesh/solver/VTU work will build on this.
test-reference-2d:
	.venv/bin/mojo run -I . test/reference_element_2d_test.mojo

# 2D triangulated Cartesian mesh topology: element / face counts,
# elem_faces <-> face_elem round-trip, side-0 / side-1 node coordinate
# agreement across shared edges, Jacobian positivity.  Host-only.
test-local-mesh-2d:
	.venv/bin/mojo run -I . test/local_mesh_2d_test.mojo

# Host-side 2D DG advection rhs: constant-state preservation.  On a
# periodic domain a constant q has zero divergence of v.q and the face
# fluxes cancel pair-wise around each cell (divergence theorem), so
# `advection_rhs_2d` must produce rhs = 0 to within roundoff.  Tests
# that the mesh Jacobian, D_ref / Lift_ref operators, face-normal
# convention, and elem_canon_to_ref mapping are internally consistent.
test-dg-rhs-2d:
	.venv/bin/mojo run -I . test/dg_rhs_2d_test.mojo

# End-to-end 2D DG advection: run SSPRK3 for one full period on a
# periodic Gaussian IC and verify L2 error against the initial state is
# small (< 10%).  Proves the mesh + rhs + time stepper compose without
# sign/scale mistakes.  Host-only (Float64 CPU reference path).
test-advection-step-2d:
	.venv/bin/mojo run -I . test/advection_step_2d_test.mojo

# 2D Euler physics sanity: constant-state preservation + Gaussian
# density bump translation under a uniform flow.  Validates the 2D
# Rusanov numerical flux and the Physics2D trait dispatch.
test-euler2d:
	.venv/bin/mojo run -I . test/euler2d_test.mojo

# 2D shallow-water physics sanity: constant-state preservation on
# lake-at-rest + uniform-flow IC.  Verifies the flux for p = g h^2 / 2
# and the 3-component state vector are plumbed correctly.
test-shallow-water-2d:
	.venv/bin/mojo run -I . test/shallow_water_2d_test.mojo

# Non-periodic 2D BC overlay sanity: wall BCs on all four sides produce
# the expected face count (3 Nx Ny + Nx + Ny) and at-rest states for
# ShallowWater2D / Euler2D give rhs = 0 to roundoff (wall reflection
# cancels symmetrically).
test-mesh-2d-bc:
	.venv/bin/mojo run -I . test/mesh_2d_bc_test.mojo

# 2D shallow-water BCs under *evolution*: Gaussian bump in a walled
# box, integrated for T=0.2.  Checks h stays positive + bounded + mass
# stays within 1% over the full simulation -- catches wall-flux bugs
# that don't show up at-rest.
test-sw-bc-dynamics:
	.venv/bin/mojo run -I . test/sw_bc_dynamics_test.mojo

# 2D time integrators: SSPRK2 vs SSPRK3 vs RK4 on a smooth periodic
# Gaussian translation.  Verifies each runs without NaN, conserves
# mass to ~ 1e-10, and higher-order methods produce lower L2 error on
# this smooth problem.
test-time-integrators-2d:
	.venv/bin/mojo run -I . test/time_integrators_2d_test.mojo

# Spatial convergence rate of the 2D DG scheme under successive mesh
# refinement.  Smooth sinusoidal IC, short integration so temporal
# error is negligible; observed rates log2(e_N / e_{2N}) should be
# >= (P + 1) asymptotically.  Checks against loose floors (1.5, 2.5,
# 3.0 for P=1/2/3) so pre-asymptotic / Rusanov dissipation wiggle
# doesn't cause false negatives.
test-convergence-2d:
	.venv/bin/mojo run -I . test/convergence_2d_test.mojo

# 2D BC_INFLOW: at-rest preservation when inflow state equals interior
# (SW + Euler) + Advection inflow fill (domain populates from empty
# via an inflow edge).
test-bc-inflow-2d:
	.venv/bin/mojo run -I . test/bc_inflow_2d_test.mojo

# 2D ideal MHD sanity: uniform (rho, u, v, Bx, By, p) preservation
# on a periodic mesh, with a perfectly-conducting wall variant that
# reflects both velocity and B along the normal.
test-ideal-mhd-2d:
	.venv/bin/mojo run -I . test/ideal_mhd_2d_test.mojo

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

# Convenience target: run every test in the suite.  Stops on the first
# failure.  Doesn't include test-klone (that's for cluster submission).
test-all: test-reference test-reference-2d test-local-mesh-2d test-dg-rhs-2d test-advection-step-2d test-euler2d test-shallow-water-2d test-mesh-2d-bc test-sw-bc-dynamics test-time-integrators-2d test-convergence-2d test-bc-inflow-2d test-ideal-mhd-2d test-local-mesh-2d-gpu test-diagnostics test-p3 test test-bc
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
