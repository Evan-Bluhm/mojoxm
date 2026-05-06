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
# Default: invoke Mojo through `pixi run` so MODULAR_HOME / CONDA_PREFIX
# are set up.  The bare `.pixi/envs/default/bin/mojo` won't find its
# stdlib without those vars.  Override with `make MOJO=mojo` if you've
# already activated the env (`pixi shell`).
MOJO        ?= pixi run mojo
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
               maxwell_cavity_2d_gpu mhd_alfven_glm_2d_gpu \
               mpi_patch_mesh mpi_halo_pingpong
ALL_DRIVERS  = $(CPU_DRIVERS) $(GPU_DRIVERS)

# Test drivers live under test/; they validate GPU kernels against
# self-consistent invariants (constant-state preservation, upload
# round-trips) or against known analytic solutions.  All require a
# GPU.
TEST_DRIVERS = mpi_advection_test mpi_bc_test diagnostics_test p3_smoke_test \
               local_mesh_2d_gpu_test euler_2d_gpu_test sw_2d_gpu_test \
               mhd_2d_gpu_test mhd_glm_2d_gpu_test maxwell_2d_gpu_test \
               limiter_2d_gpu_test limiter_2d_gpu_test_p3 \
               limiter_2d_gpu_test_p4 limiter_2d_gpu_test_p5 \
               limiter_3d_test limiter_3d_test_p3 \
               limiter_3d_test_p4 limiter_3d_test_p5 \
               mhd_3d_test euler_3d_test maxwell_3d_test \
               sw_3d_test two_fluid_3d_test \
               vtu_2d_multi_test vtu_3d_multi_test \
               memory_report_test ssprk3_test partition_test \
               sod_exact_riemann_test frame_writer_multi_test

# Benchmark drivers live under benchmarks/.  Each runs a single
# known-solution problem and asserts a measured metric against a
# threshold; failure = regression.  Complement the tests (which cover
# self-consistency invariants) with end-to-end analytic correctness.
BENCH_DRIVERS = bench_advection_translation_2d \
                bench_advection_translation_2d_p3 \
                bench_advection_translation_2d_p4 \
                bench_advection_translation_2d_p5 \
                bench_advection_outflow_2d bench_advection_inflow_2d \
                bench_euler_vortex_2d bench_euler_vortex_2d_p3 \
                bench_mhd_alfven_2d bench_mhd_alfven_glm_2d \
                bench_mhd_inflow_2d bench_mhd_inflow_2d_glm \
                bench_mhd_wall_2d bench_mhd_wall_2d_glm bench_mhd_wall_3d \
                bench_mhd_alfven_glm_2d_p3 \
                bench_mhd_alfven_glm_2d_p4 \
                bench_mhd_alfven_glm_2d_p5 \
                bench_mhd_glm_psi_transport_2d_p3 \
                bench_mhd_glm_psi_transport_2d \
                bench_mhd_glm_psi_transport_2d_p4 \
                bench_mhd_glm_psi_transport_2d_p5 bench_mhd_glm_psi_damp_2d \
                bench_mhd_glm_psi_damp_2d_p3 \
                bench_mhd_glm_psi_damp_2d_p4 \
                bench_mhd_glm_psi_damp_2d_p5 \
                bench_euler_sod_2d \
                bench_euler_sod_limited_2d bench_euler_sod_limited_2d_p3 \
                bench_euler_sod_limited_2d_p4 \
                bench_euler_sod_limited_2d_p5 \
                bench_euler_inflow_2d \
                bench_euler_smooth_wave_2d bench_euler_hydrostatic_2d \
                bench_euler_hydrostatic_2d_p3 \
                bench_euler_hydrostatic_2d_p4 \
                bench_euler_hydrostatic_2d_p5 \
                bench_euler_smooth_wave_2d_p3 \
                bench_euler_smooth_wave_2d_p4 \
                bench_euler_smooth_wave_2d_p5 \
                bench_euler_channel_steady_2d bench_shallow_water_wave_2d \
                bench_shallow_water_wave_2d_p3 \
                bench_shallow_water_wave_2d_p4 \
                bench_shallow_water_wave_2d_p5 \
                bench_shallow_water_wave_2d_rusanov \
                bench_shallow_water_inflow_2d \
                bench_shallow_water_inflow_2d_rusanov \
                bench_shallow_water_dam_break_2d \
                bench_maxwell_cavity_2d bench_maxwell_plane_wave_2d \
                bench_maxwell_te_plane_wave_2d \
                bench_maxwell_plane_wave_2d_p3 \
                bench_maxwell_plane_wave_2d_p4 \
                bench_maxwell_plane_wave_2d_p5 \
                bench_maxwell_outflow_2d bench_maxwell_inflow_2d \
                bench_maxwell_uniform_j_2d bench_maxwell_uniform_j_2d_p3 \
                bench_maxwell_uniform_m_2d bench_maxwell_uniform_m_2d_p3 \
                bench_advection_3d bench_advection_3d_p3 bench_advection_3d_p4 \
                bench_advection_3d_p5 \
                bench_advection_outflow_3d bench_advection_inflow_3d \
                bench_mhd_alfven_3d bench_mhd_alfven_3d_p3 \
                bench_mhd_inflow_3d \
                bench_mhd_alfven_3d_p4 bench_mhd_alfven_3d_p5 \
                bench_mhd_glm_psi_damp_3d bench_mhd_glm_psi_damp_3d_p3 \
                bench_mhd_glm_psi_damp_3d_p4 bench_mhd_glm_psi_damp_3d_p5 \
                bench_mhd_glm_psi_transport_3d \
                bench_mhd_glm_psi_transport_3d_p3 \
                bench_mhd_glm_psi_transport_3d_p4 \
                bench_mhd_glm_psi_transport_3d_p5 \
                bench_maxwell_cavity_3d bench_maxwell_plane_wave_3d \
                bench_maxwell_plane_wave_3d_p3 \
                bench_maxwell_plane_wave_3d_p4 \
                bench_maxwell_plane_wave_3d_p5 \
                bench_maxwell_uniform_j_3d bench_maxwell_uniform_j_3d_p3 \
                bench_maxwell_uniform_m_3d bench_maxwell_uniform_m_3d_p3 \
                bench_maxwell_outflow_3d bench_maxwell_inflow_3d \
                bench_two_fluid_langmuir_3d bench_two_fluid_outflow_3d \
                bench_two_fluid_walls_3d bench_two_fluid_walls_3d_p3 \
                bench_two_fluid_walls_3d_p4 bench_two_fluid_walls_3d_p5 \
                bench_euler_smooth_wave_3d bench_euler_smooth_wave_3d_p3 \
                bench_euler_smooth_wave_3d_p4 \
                bench_euler_smooth_wave_3d_p5 \
                bench_euler_flux_coverage_3d bench_euler_hydrostatic_3d \
                bench_euler_hydrostatic_3d_p3 \
                bench_euler_hydrostatic_3d_p4 \
                bench_euler_hydrostatic_3d_p5 \
                bench_euler_inflow_3d bench_euler_vortex_3d \
                bench_euler_vortex_3d_p3 \
                bench_euler_sod_3d bench_euler_sod_3d_p3 \
                bench_euler_sod_3d_p4 bench_euler_sod_3d_p5 \
                bench_shallow_water_wave_3d bench_shallow_water_wave_3d_p3 \
                bench_shallow_water_wave_3d_p4 \
                bench_shallow_water_wave_3d_p5 \
                bench_shallow_water_inflow_3d \
                bench_shallow_water_dam_break_3d \
                bench_mhd_brio_wu_3d \
                bench_mhd_brio_wu_3d_p3

.PHONY: all cpu gpu clean help format format-check install-hooks profile-summary profile-summary-test pre-push test test-bc test-vtu-meshio test-reference test-reference-2d test-local-mesh-2d test-local-mesh-2d-gpu test-euler-2d-gpu test-sw-2d-gpu test-mhd-2d-gpu test-mhd-glm-2d-gpu test-maxwell-2d-gpu test-limiter-2d-gpu test-limiter-2d-gpu-p3 test-limiter-2d-gpu-p4 test-limiter-2d-gpu-p5 test-limiter-3d test-limiter-3d-p3 test-limiter-3d-p4 test-limiter-3d-p5 test-mhd-3d test-euler-3d test-maxwell-3d test-sw-3d test-two-fluid-3d test-vtu-2d-multi test-vtu-3d-multi test-memory-report test-ssprk3 test-partition test-sod-exact-riemann test-frame-writer-multi test-diagnostics test-p3 test-quick smoke test-utils test-limiter test-all test-klone bench-quick bench-p5 bench-rates bench-shocks bench-bcs bench-mhd bench-euler bench-maxwell bench-sw bench-advection bench-two-fluid bench-all bench-advection-translation-2d bench-euler-vortex-2d bench-mhd-alfven-2d bench-euler-sod-2d

help:
	@echo 'mojoxm build targets'
	@echo ''
	@echo 'Quick start:'
	@echo '  make gpu                 build all GPU drivers (~30s incremental on a hot cache)'
	@echo '  make bench-euler-sod-2d  build + run one bench (~3s build + <1s run)'
	@echo '  make test-quick          smoke 8 representative tests (~30s w/ cached binaries)'
	@echo '  make bench-quick         smoke 13 representative benches (~65s)'
	@echo '  make smoke               test-utils + test-quick + bench-quick combined (~95s; one-command sanity check)'
	@echo '  make pre-push            format-check + profile-summary-test + smoke (~100s)'
	@echo '  make bench-p5            run 19 P=5 P-parity gates at NP=21/56 (~95s)'
	@echo '  make bench-shocks        run 13 shocked-flow gates -- Sod / dam-break / Brio-Wu (~70s)'
	@echo '  make bench-rates         run 10 convergence-rate gates -- catches order regressions (~50s)'
	@echo '  make bench-bcs           run 26 BC + source-term gates -- inflow / outflow / wall / gravity (~95s cached)'
	@echo '  make bench-{mhd,euler,maxwell,sw,advection,two-fluid}'
	@echo '                           run all gates for one physics module'
	@echo '                           (33/33/23/14/12/6 gates resp., ~25-110s each)'
	@echo '  make bench-all           build + run every analytic-solution gate (121 benches)'
	@echo ''
	@echo 'Note: do NOT use make -j.  Mojo already runs multi-threaded per'
	@echo '      compile; -j contention makes parallel builds 1.5-2x slower'
	@echo '      than serial in practice (measured: -j4 = 18s vs serial = 12s'
	@echo '      for 4 independent benches).'
	@echo ''
	@echo 'Build targets:'
	@echo '  make                     build every driver (needs GPU for most)'
	@echo '  make cpu                 CPU-only smoke tests (login-node safe)'
	@echo '                           drivers: $(CPU_DRIVERS)'
	@echo '  make gpu                 GPU-using drivers (3D + 2D + MPI helpers)'
	@echo '  make <driver>            build a single driver by name'
	@echo '  make shim                build just build/mpi_shim.o'
	@echo '  make clean               remove driver binaries + $(BUILD_DIR)/'
	@echo ''
	@echo 'Test targets (validate kernels against analytic invariants):'
	@echo '  make test                MPI correctness (np=1 vs np=4, periodic)'
	@echo '  make test-bc             MPI correctness (non-periodic, np=1 vs np=4)'
	@echo '  make test-reference      reference-element unit test (P=1..5, host-side)'
	@echo '  make test-reference-2d   2D reference-triangle unit test (P=1..5)'
	@echo '  make test-local-mesh-2d  2D mesh + advection foundation test (P=1..5)'
	@echo '  make test-{euler,sw,mhd,mhd-glm,maxwell}-2d-gpu'
	@echo '                           per-physics 2D constant-state preservation (P=2..5)'
	@echo '  make test-limiter-2d-gpu[-p3,-p4,-p5]'
	@echo '                           2D BJ limiter smooth-passthrough + monotonicity'
	@echo '  make test-{euler,sw,mhd,maxwell,two-fluid}-3d'
	@echo '                           per-physics 3D constant-state preservation (P=2..5)'
	@echo '  make test-limiter-3d[-p3,-p4,-p5]'
	@echo '                           3D BJ limiter cell-mean conservation'
	@echo '  make test-diagnostics    DiagnosticsWriter unit test (GPU, np=1)'
	@echo '  make test-p3             P=3 round-trip smoke test'
	@echo '  make test-memory-report  perf-introspection (memory + DOF/s + bandwidth)'
	@echo '  make test-frame-writer-multi'
	@echo '                           sync multi-field 3D VTU output via FrameWriter'
	@echo '  make test-ssprk3         SSPRK3 stage-plan helper (host-only, <1s)'
	@echo '  make test-partition      build_partition factorisation + neighbours (host)'
	@echo '  make test-sod-exact-riemann'
	@echo '                           Toro 2009 reference values for the analytic Sod solver'
	@echo '  make test-quick          run 8 representative tests (~30s with cached binaries)'
	@echo '  make test-utils          run 5 helper-module tests (~5s; 3 host-only + 2 small GPU)'
	@echo '  make test-limiter        run 8 BJ-limiter unit tests (2D + 3D, P=2/3/4/5; ~20s cached)'
	@echo '  make test-all            run every test above (~3 min wall)'
	@echo '  make test-klone          run MPI test on Klone (requires klone-run)'
	@echo ''
	@echo 'Bench targets (analytic-solution gates):'
	@echo '  make bench-quick         smoke-test one bench per physics per dim + limiter (13 gates, ~65s)'
	@echo '  make bench-p5            P=5 P-parity sweep -- 19 gates at NP=21 (2D) / NP=56 (3D)'
	@echo '                           across all 5 smooth physics + limited shocked Sod + Euler'
	@echo '                           hydrostatic + 3D Two-Fluid (NC=17) + GLM exp-decay rate gates'
	@echo '                           (~90s); the largest comptime configs the suite covers, where'
	@echo '                           high-P regressions show first.'
	@echo '  make bench-shocks        13 shocked-flow gates exercising HLLC / HLLEC + BJ limiter'
	@echo '                           (Euler Sod 2D + 3D, P=2-5), HLL SW dam-break (2D + 3D), and'
	@echo '                           Brio-Wu MHD shock (3D P=2/3); use when iterating on Riemann'
	@echo '                           solvers or the limiter pipeline (~70s).'
	@echo '  make bench-rates         10 convergence-rate gates that explicitly assert log2(e_N /'
	@echo '                           e_2N) >= P-dependent floor.  Advection 2D + 3D P=2-5 (8) +'
	@echo '                           Euler 3D P=2/3 (2).  Catches scheme-order regressions an'
	@echo '                           absolute-L2 sentinel would miss (~50s).'
	@echo '  make bench-bcs           26 boundary-condition + source-term gates exercising every'
	@echo '                           BC dispatch arm (interior / wall / outflow / inflow) across'
	@echo '                           all physics, plus Euler gravity (hydrostatic) and Euler'
	@echo '                           channel steady-state.  Use when iterating on BC routing or'
	@echo '                           the source-term hook in rk_stage_kernel (~95s w/ cached binaries).'
	@echo '  make bench-all           build + run every gate (121 benches, ~10 min)'
	@echo '  make bench-<name>        build + run a single bench (see benchmarks/*.mojo)'
	@echo '                           e.g. bench-euler-sod-2d, bench-mhd-alfven-3d-p4'
	@echo ''
	@echo 'Profiling:'
	@echo '  make profile-bench-<name>'
	@echo '                           profile one bench under nsys, save kernel summary'
	@echo '                           to benchmarks/profile_reports/<name>.kern.txt'
	@echo '  make profile-bench-quick refresh profiles for the 13 bench-quick gates (~10-15 min)'
	@echo '  make profile-bench-all   profile every bench (slow; for baselining)'
	@echo '  make profile-summary     rank every cached profile by us/launch'
	@echo '                           (also: scripts/profile_summary.py --top N / --filter <s>'
	@echo '                            / --by-physics / --show-cv / --csv / --self-test)'
	@echo '  make profile-summary-test'
	@echo '                           run profile_summary.py --self-test (sub-second; CI gate)'
	@echo ''
	@echo 'Code style:'
	@echo '  make format              run `mojo format` on every .mojo file (in-place)'
	@echo '  make format-check        non-mutating CI gate (~3s on 213 files): fail if any'
	@echo '                           .mojo file is out of conformance.  Mirrors the pre-'
	@echo '                           commit hook but covers the full working tree.'
	@echo '  make install-hooks       enable the staged-file pre-commit format gate'
	@echo '                           (sets git core.hooksPath; once per clone)'
	@echo ''
	@echo 'Visualisation (after running an example driver):'
	@echo '  scripts/animate_2d.py output/solution_X.pvd -o movie.mp4 -f rho'
	@echo '                           render a 2D-GPU run to MP4 via ffmpeg.  Suffix .gif'
	@echo '                           switches to a Pillow GIF; -f rho,p,'"'"'|v|'"'"' renders'
	@echo '                           multi-panel side-by-side; --list-fields prints scalars'
	@echo '  scripts/animate_dashboard.py'
	@echo '                           render a 3D run to dashboard.gif (density slice +'
	@echo '                           diagnostics CSV time-series in a 2x2 layout)'
	@echo '  ParaView                 open output/solution_X.pvd directly'
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

# Benchmark drivers -- same link line, sourced from benchmarks/.
$(BENCH_DRIVERS): %: benchmarks/%.mojo $(BUILD_DIR)/mpi_shim.o $(SRC_MOJO)
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
# P=1..5 via SPD mass matrix + node-position + face-to-element map
# + node_weights partition-of-unity (with P=2 closed-form spot
# check) checks.  Host math only, but the same tables are uploaded
# to the GPU by ReferenceElement[P].
test-reference:
	$(MOJO) run -I . test/reference_element_test.mojo

# Same for the 2D triangular reference element (ReferenceElement2D[P]).
# Covers node positions, SPD 2D mass matrix, edge-to-element map.
# ReferenceElement2DGpu uploads from this; validating the host tables
# catches the bulk of basis-construction bugs before a GPU run.
test-reference-2d:
	$(MOJO) run -I . test/reference_element_2d_test.mojo

# 2D triangulated Cartesian mesh topology: element / face counts,
# elem_faces <-> face_elem round-trip, side-0 / side-1 node coordinate
# agreement across shared edges, Jacobian positivity.  The mesh
# LocalMesh2D uploads is built here; catching topology errors on the
# host side avoids expensive GPU debugging.
test-local-mesh-2d:
	$(MOJO) run -I . test/local_mesh_2d_test.mojo

# GPU diagnostics writer test: uniform-field integrals recover
# analytic values; max_abs reports the peak on a checkerboard field;
# empty configuration doesn't crash.  Runs at np=1.
test-diagnostics: diagnostics_test
	./diagnostics_test

# Multi-field 2D VTU writer smoke test: dumps three constant scalar
# fields to a temporary VTU and verifies the XML headers carry all
# three DataArray entries with the first set as the default Scalars.
# Catches gross regressions in `dump_vtu_2d_frame_multi` offset
# arithmetic.  Host-only, no GPU kernels.
test-vtu-2d-multi: vtu_2d_multi_test
	./vtu_2d_multi_test
test-vtu-3d-multi: vtu_3d_multi_test
	./vtu_3d_multi_test

# meshio-roundtrip sanity check on the P=2..5 VTU fixtures produced
# by vtu_3d_multi_test (which writes to /tmp/vtu_3d_multi_test_pN.vtu
# at each P).  Catches regressions where the binary VTU format breaks
# meshio's parser -- the in-Mojo XML-string check in vtu_3d_multi_test
# would not see this.  Depends on test-vtu-3d-multi to produce the
# fixtures first.
test-vtu-meshio: test-vtu-3d-multi
	@scripts/validate_vtu.py \
		/tmp/vtu_3d_multi_test_p2.vtu \
		/tmp/vtu_3d_multi_test_p3.vtu \
		/tmp/vtu_3d_multi_test_p4.vtu \
		/tmp/vtu_3d_multi_test_p5.vtu
test-memory-report: memory_report_test
	./memory_report_test
test-ssprk3: ssprk3_test
	./ssprk3_test
test-partition: partition_test
	./partition_test
test-sod-exact-riemann: sod_exact_riemann_test
	./sod_exact_riemann_test
test-frame-writer-multi: frame_writer_multi_test
	./frame_writer_multi_test

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
test-mhd-glm-2d-gpu: mhd_glm_2d_gpu_test
	./mhd_glm_2d_gpu_test
test-maxwell-2d-gpu: maxwell_2d_gpu_test
	./maxwell_2d_gpu_test
test-limiter-2d-gpu: limiter_2d_gpu_test
	./limiter_2d_gpu_test
test-limiter-2d-gpu-p3: limiter_2d_gpu_test_p3
	./limiter_2d_gpu_test_p3
test-limiter-2d-gpu-p4: limiter_2d_gpu_test_p4
	./limiter_2d_gpu_test_p4
test-limiter-2d-gpu-p5: limiter_2d_gpu_test_p5
	./limiter_2d_gpu_test_p5
test-limiter-3d: limiter_3d_test
	./limiter_3d_test
test-limiter-3d-p3: limiter_3d_test_p3
	./limiter_3d_test_p3
test-limiter-3d-p4: limiter_3d_test_p4
	./limiter_3d_test_p4
test-limiter-3d-p5: limiter_3d_test_p5
	./limiter_3d_test_p5
test-mhd-3d: mhd_3d_test
	./mhd_3d_test
test-euler-3d: euler_3d_test
	./euler_3d_test
test-maxwell-3d: maxwell_3d_test
	./maxwell_3d_test
test-sw-3d: sw_3d_test
	./sw_3d_test
test-two-fluid-3d: two_fluid_3d_test
	./two_fluid_3d_test

# Convenience target: run every test in the suite.  Stops on the first
# failure.  Doesn't include test-klone (that's for cluster submission).
test-all: test-reference test-reference-2d test-local-mesh-2d test-local-mesh-2d-gpu test-euler-2d-gpu test-sw-2d-gpu test-mhd-2d-gpu test-mhd-glm-2d-gpu test-maxwell-2d-gpu test-limiter-2d-gpu test-limiter-2d-gpu-p3 test-limiter-2d-gpu-p4 test-limiter-2d-gpu-p5 test-limiter-3d test-limiter-3d-p3 test-limiter-3d-p4 test-limiter-3d-p5 test-mhd-3d test-euler-3d test-maxwell-3d test-sw-3d test-two-fluid-3d test-vtu-2d-multi test-vtu-3d-multi test-memory-report test-ssprk3 test-partition test-sod-exact-riemann test-frame-writer-multi test-diagnostics test-p3 test test-bc
	@echo '=== ALL TESTS PASSED ==='


# Quick test smoke -- 8 representative tests covering host operator
# construction (reference, reference-2d), one 2D physics + one 3D
# physics constant-state preservation (euler-2d-gpu, euler-3d), the
# 2D + 3D limiter pipelines, the P=3 round-trip parametrization, and
# the 3D multi-field VTU writer (the visualization-output path).
# Catches the broadest class of regressions in ~30s wall with cached
# binaries (~1-2 min cold), vs ~3 min for full test-all.
test-quick: test-reference test-reference-2d test-euler-2d-gpu test-euler-3d \
            test-limiter-2d-gpu test-limiter-3d test-p3 test-vtu-3d-multi
	@echo '=== test-quick: 8 representative tests PASSED ==='


# One-command smoke aggregate: test-quick + bench-quick.  Use this
# right after pulling, after a non-trivial refactor, or before
# committing to catch the broadest class of regressions in ~90s
# wall (cached binaries).  Runs sequentially: a test failure stops
# the bench sweep early so you don't wait through 12 benches when
# the foundation is already broken.
smoke: test-utils test-quick bench-quick
	@echo '=== smoke: test-utils + test-quick + bench-quick PASSED ==='


# Pre-push check: format-check (~3s) + profile_summary self-test
# (~0.2s) + smoke (~95s).  Catches the common pre-push regressions:
# format drift (which the pre-commit hook only sees on staged files,
# not the working tree at large), plus a parser-self-test on the
# profile_summary script, plus the broad smoke aggregator.  Use as
# a routine "is this safe to push?" gate before `git push`.
# Run-time ~100s wall on the cached path.
pre-push: format-check profile-summary-test smoke
	@echo '=== pre-push: format-check + profile-summary-test + smoke PASSED ==='


# Sub-second parser self-test on profile_summary.py.  Catches
# regressions in the regex / classifiers without needing the cached
# .kern.txt baselines (uses an in-script fixture).
profile-summary-test:
	@scripts/profile_summary.py --self-test


# Utility test aggregator -- 5 sub-second tests covering the non-
# physics infrastructure (perf-introspection accounting, multi-field
# VTU writer, SSPRK3 stage-plan helper, MPI partition factorisation,
# analytic Sod Riemann solver).  Three are host-only (test-ssprk3,
# test-partition, test-sod-exact-riemann); two are GPU-using but
# very small (test-memory-report builds a 4^3 mesh and runs ~10
# SSPRK3 steps; test-frame-writer-multi writes 2 frames at NP=10).
# Use when touching any of the helper modules in `src/` to catch
# regressions without paying for the 2D/3D physics-test compile
# times (~5s wall total cached, vs ~30s for test-quick).
test-utils: test-memory-report test-frame-writer-multi test-ssprk3 \
            test-partition test-sod-exact-riemann
	@echo '=== test-utils: 5 utility tests PASSED ==='


# Limiter unit-test sweep -- 8 tests covering the BJ slope limiter
# at every supported P in both dimensions (2D NP=6/10/15/21,
# 3D NP=10/20/35/56).  Each variant has different
# `ReferenceElement[P].node_weights` distributions, so a regression
# in either the kernel weights or the BJ pipeline's downstream use
# of them is caught at the P that introduced it.  Mirrors bench-p5
# coverage on the test side.  ~20s w/ cached binaries.
test-limiter: test-limiter-2d-gpu test-limiter-2d-gpu-p3 \
              test-limiter-2d-gpu-p4 test-limiter-2d-gpu-p5 \
              test-limiter-3d test-limiter-3d-p3 \
              test-limiter-3d-p4 test-limiter-3d-p5
	@echo '=== test-limiter: 8 BJ limiter unit tests PASSED ==='

# Benchmarks: each runs a single known-solution problem and asserts
# the measured metric (L2 error vs analytic, plateau deviation,
# convergence rate) against a published threshold.
bench-advection-translation-2d: bench_advection_translation_2d
	./bench_advection_translation_2d
bench-advection-translation-2d-p3: bench_advection_translation_2d_p3
	./bench_advection_translation_2d_p3
bench-advection-translation-2d-p4: bench_advection_translation_2d_p4
	./bench_advection_translation_2d_p4
bench-advection-translation-2d-p5: bench_advection_translation_2d_p5
	./bench_advection_translation_2d_p5
bench-advection-outflow-2d: bench_advection_outflow_2d
	./bench_advection_outflow_2d
bench-advection-inflow-2d: bench_advection_inflow_2d
	./bench_advection_inflow_2d
bench-euler-vortex-2d: bench_euler_vortex_2d
	./bench_euler_vortex_2d
bench-euler-vortex-2d-p3: bench_euler_vortex_2d_p3
	./bench_euler_vortex_2d_p3
bench-mhd-alfven-2d: bench_mhd_alfven_2d
	./bench_mhd_alfven_2d
bench-mhd-inflow-2d: bench_mhd_inflow_2d
	./bench_mhd_inflow_2d
bench-mhd-alfven-glm-2d: bench_mhd_alfven_glm_2d
	./bench_mhd_alfven_glm_2d
bench-mhd-inflow-2d-glm: bench_mhd_inflow_2d_glm
	./bench_mhd_inflow_2d_glm
bench-mhd-wall-2d-glm: bench_mhd_wall_2d_glm
	./bench_mhd_wall_2d_glm
bench-mhd-wall-2d: bench_mhd_wall_2d
	./bench_mhd_wall_2d
bench-mhd-wall-3d: bench_mhd_wall_3d
	./bench_mhd_wall_3d
bench-mhd-alfven-glm-2d-p3: bench_mhd_alfven_glm_2d_p3
	./bench_mhd_alfven_glm_2d_p3
bench-mhd-alfven-glm-2d-p4: bench_mhd_alfven_glm_2d_p4
	./bench_mhd_alfven_glm_2d_p4
bench-mhd-alfven-glm-2d-p5: bench_mhd_alfven_glm_2d_p5
	./bench_mhd_alfven_glm_2d_p5
bench-mhd-glm-psi-transport-2d: bench_mhd_glm_psi_transport_2d
	./bench_mhd_glm_psi_transport_2d
bench-mhd-glm-psi-transport-2d-p3: bench_mhd_glm_psi_transport_2d_p3
	./bench_mhd_glm_psi_transport_2d_p3
bench-mhd-glm-psi-transport-2d-p4: bench_mhd_glm_psi_transport_2d_p4
	./bench_mhd_glm_psi_transport_2d_p4
bench-mhd-glm-psi-transport-2d-p5: bench_mhd_glm_psi_transport_2d_p5
	./bench_mhd_glm_psi_transport_2d_p5
bench-mhd-glm-psi-damp-2d: bench_mhd_glm_psi_damp_2d
	./bench_mhd_glm_psi_damp_2d
bench-mhd-glm-psi-damp-2d-p3: bench_mhd_glm_psi_damp_2d_p3
	./bench_mhd_glm_psi_damp_2d_p3
bench-mhd-glm-psi-damp-2d-p4: bench_mhd_glm_psi_damp_2d_p4
	./bench_mhd_glm_psi_damp_2d_p4
bench-mhd-glm-psi-damp-2d-p5: bench_mhd_glm_psi_damp_2d_p5
	./bench_mhd_glm_psi_damp_2d_p5
bench-euler-sod-2d: bench_euler_sod_2d
	./bench_euler_sod_2d
bench-euler-sod-limited-2d: bench_euler_sod_limited_2d
	./bench_euler_sod_limited_2d
bench-euler-sod-limited-2d-p3: bench_euler_sod_limited_2d_p3
	./bench_euler_sod_limited_2d_p3
bench-euler-sod-limited-2d-p4: bench_euler_sod_limited_2d_p4
	./bench_euler_sod_limited_2d_p4
bench-euler-sod-limited-2d-p5: bench_euler_sod_limited_2d_p5
	./bench_euler_sod_limited_2d_p5
bench-euler-smooth-wave-2d: bench_euler_smooth_wave_2d
	./bench_euler_smooth_wave_2d
bench-euler-smooth-wave-2d-p3: bench_euler_smooth_wave_2d_p3
	./bench_euler_smooth_wave_2d_p3
bench-euler-smooth-wave-2d-p4: bench_euler_smooth_wave_2d_p4
	./bench_euler_smooth_wave_2d_p4
bench-euler-smooth-wave-2d-p5: bench_euler_smooth_wave_2d_p5
	./bench_euler_smooth_wave_2d_p5
bench-euler-channel-steady-2d: bench_euler_channel_steady_2d
	./bench_euler_channel_steady_2d
bench-euler-inflow-2d: bench_euler_inflow_2d
	./bench_euler_inflow_2d
bench-euler-hydrostatic-2d: bench_euler_hydrostatic_2d
	./bench_euler_hydrostatic_2d
bench-euler-hydrostatic-2d-p3: bench_euler_hydrostatic_2d_p3
	./bench_euler_hydrostatic_2d_p3
bench-euler-hydrostatic-2d-p4: bench_euler_hydrostatic_2d_p4
	./bench_euler_hydrostatic_2d_p4
bench-euler-hydrostatic-2d-p5: bench_euler_hydrostatic_2d_p5
	./bench_euler_hydrostatic_2d_p5
bench-shallow-water-wave-2d: bench_shallow_water_wave_2d
	./bench_shallow_water_wave_2d
bench-shallow-water-wave-2d-p3: bench_shallow_water_wave_2d_p3
	./bench_shallow_water_wave_2d_p3
bench-shallow-water-wave-2d-p4: bench_shallow_water_wave_2d_p4
	./bench_shallow_water_wave_2d_p4
bench-shallow-water-wave-2d-p5: bench_shallow_water_wave_2d_p5
	./bench_shallow_water_wave_2d_p5
bench-shallow-water-wave-2d-rusanov: bench_shallow_water_wave_2d_rusanov
	./bench_shallow_water_wave_2d_rusanov
bench-shallow-water-inflow-2d: bench_shallow_water_inflow_2d
	./bench_shallow_water_inflow_2d
bench-shallow-water-inflow-2d-rusanov: bench_shallow_water_inflow_2d_rusanov
	./bench_shallow_water_inflow_2d_rusanov
bench-shallow-water-dam-break-2d: bench_shallow_water_dam_break_2d
	./bench_shallow_water_dam_break_2d
bench-maxwell-cavity-2d: bench_maxwell_cavity_2d
	./bench_maxwell_cavity_2d
bench-maxwell-plane-wave-2d: bench_maxwell_plane_wave_2d
	./bench_maxwell_plane_wave_2d
bench-maxwell-te-plane-wave-2d: bench_maxwell_te_plane_wave_2d
	./bench_maxwell_te_plane_wave_2d
bench-maxwell-plane-wave-2d-p3: bench_maxwell_plane_wave_2d_p3
	./bench_maxwell_plane_wave_2d_p3
bench-maxwell-plane-wave-2d-p4: bench_maxwell_plane_wave_2d_p4
	./bench_maxwell_plane_wave_2d_p4
bench-maxwell-plane-wave-2d-p5: bench_maxwell_plane_wave_2d_p5
	./bench_maxwell_plane_wave_2d_p5
bench-maxwell-outflow-2d: bench_maxwell_outflow_2d
	./bench_maxwell_outflow_2d
bench-maxwell-inflow-2d: bench_maxwell_inflow_2d
	./bench_maxwell_inflow_2d
bench-maxwell-uniform-j-2d: bench_maxwell_uniform_j_2d
	./bench_maxwell_uniform_j_2d
bench-maxwell-uniform-j-2d-p3: bench_maxwell_uniform_j_2d_p3
	./bench_maxwell_uniform_j_2d_p3
bench-maxwell-uniform-m-2d: bench_maxwell_uniform_m_2d
	./bench_maxwell_uniform_m_2d
bench-maxwell-uniform-m-2d-p3: bench_maxwell_uniform_m_2d_p3
	./bench_maxwell_uniform_m_2d_p3
bench-advection-3d: bench_advection_3d
	./bench_advection_3d
bench-advection-3d-p3: bench_advection_3d_p3
	./bench_advection_3d_p3
bench-advection-3d-p4: bench_advection_3d_p4
	./bench_advection_3d_p4
bench-advection-3d-p5: bench_advection_3d_p5
	./bench_advection_3d_p5
bench-advection-outflow-3d: bench_advection_outflow_3d
	./bench_advection_outflow_3d
bench-advection-inflow-3d: bench_advection_inflow_3d
	./bench_advection_inflow_3d
bench-mhd-alfven-3d: bench_mhd_alfven_3d
	./bench_mhd_alfven_3d
bench-mhd-inflow-3d: bench_mhd_inflow_3d
	./bench_mhd_inflow_3d
bench-mhd-alfven-3d-p3: bench_mhd_alfven_3d_p3
	./bench_mhd_alfven_3d_p3
bench-mhd-alfven-3d-p4: bench_mhd_alfven_3d_p4
	./bench_mhd_alfven_3d_p4
bench-mhd-alfven-3d-p5: bench_mhd_alfven_3d_p5
	./bench_mhd_alfven_3d_p5
bench-mhd-glm-psi-damp-3d: bench_mhd_glm_psi_damp_3d
	./bench_mhd_glm_psi_damp_3d
bench-mhd-glm-psi-damp-3d-p3: bench_mhd_glm_psi_damp_3d_p3
	./bench_mhd_glm_psi_damp_3d_p3
bench-mhd-glm-psi-damp-3d-p4: bench_mhd_glm_psi_damp_3d_p4
	./bench_mhd_glm_psi_damp_3d_p4
bench-mhd-glm-psi-damp-3d-p5: bench_mhd_glm_psi_damp_3d_p5
	./bench_mhd_glm_psi_damp_3d_p5
bench-mhd-glm-psi-transport-3d: bench_mhd_glm_psi_transport_3d
	./bench_mhd_glm_psi_transport_3d
bench-mhd-glm-psi-transport-3d-p3: bench_mhd_glm_psi_transport_3d_p3
	./bench_mhd_glm_psi_transport_3d_p3
bench-mhd-glm-psi-transport-3d-p4: bench_mhd_glm_psi_transport_3d_p4
	./bench_mhd_glm_psi_transport_3d_p4
bench-mhd-glm-psi-transport-3d-p5: bench_mhd_glm_psi_transport_3d_p5
	./bench_mhd_glm_psi_transport_3d_p5
bench-maxwell-cavity-3d: bench_maxwell_cavity_3d
	./bench_maxwell_cavity_3d
bench-maxwell-plane-wave-3d: bench_maxwell_plane_wave_3d
	./bench_maxwell_plane_wave_3d
bench-maxwell-plane-wave-3d-p3: bench_maxwell_plane_wave_3d_p3
	./bench_maxwell_plane_wave_3d_p3
bench-maxwell-plane-wave-3d-p4: bench_maxwell_plane_wave_3d_p4
	./bench_maxwell_plane_wave_3d_p4
bench-maxwell-plane-wave-3d-p5: bench_maxwell_plane_wave_3d_p5
	./bench_maxwell_plane_wave_3d_p5
bench-maxwell-uniform-j-3d: bench_maxwell_uniform_j_3d
	./bench_maxwell_uniform_j_3d
bench-maxwell-uniform-j-3d-p3: bench_maxwell_uniform_j_3d_p3
	./bench_maxwell_uniform_j_3d_p3
bench-maxwell-uniform-m-3d: bench_maxwell_uniform_m_3d
	./bench_maxwell_uniform_m_3d
bench-maxwell-uniform-m-3d-p3: bench_maxwell_uniform_m_3d_p3
	./bench_maxwell_uniform_m_3d_p3
bench-maxwell-outflow-3d: bench_maxwell_outflow_3d
	./bench_maxwell_outflow_3d
bench-maxwell-inflow-3d: bench_maxwell_inflow_3d
	./bench_maxwell_inflow_3d
bench-two-fluid-langmuir-3d: bench_two_fluid_langmuir_3d
	./bench_two_fluid_langmuir_3d
bench-two-fluid-outflow-3d: bench_two_fluid_outflow_3d
	./bench_two_fluid_outflow_3d
bench-two-fluid-walls-3d: bench_two_fluid_walls_3d
	./bench_two_fluid_walls_3d
bench-two-fluid-walls-3d-p3: bench_two_fluid_walls_3d_p3
	./bench_two_fluid_walls_3d_p3
bench-two-fluid-walls-3d-p4: bench_two_fluid_walls_3d_p4
	./bench_two_fluid_walls_3d_p4
bench-two-fluid-walls-3d-p5: bench_two_fluid_walls_3d_p5
	./bench_two_fluid_walls_3d_p5
bench-euler-smooth-wave-3d: bench_euler_smooth_wave_3d
	./bench_euler_smooth_wave_3d
bench-euler-smooth-wave-3d-p3: bench_euler_smooth_wave_3d_p3
	./bench_euler_smooth_wave_3d_p3
bench-euler-smooth-wave-3d-p4: bench_euler_smooth_wave_3d_p4
	./bench_euler_smooth_wave_3d_p4
bench-euler-smooth-wave-3d-p5: bench_euler_smooth_wave_3d_p5
	./bench_euler_smooth_wave_3d_p5
bench-euler-flux-coverage-3d: bench_euler_flux_coverage_3d
	./bench_euler_flux_coverage_3d
bench-euler-hydrostatic-3d: bench_euler_hydrostatic_3d
	./bench_euler_hydrostatic_3d
bench-euler-hydrostatic-3d-p3: bench_euler_hydrostatic_3d_p3
	./bench_euler_hydrostatic_3d_p3
bench-euler-hydrostatic-3d-p4: bench_euler_hydrostatic_3d_p4
	./bench_euler_hydrostatic_3d_p4
bench-euler-hydrostatic-3d-p5: bench_euler_hydrostatic_3d_p5
	./bench_euler_hydrostatic_3d_p5
bench-euler-inflow-3d: bench_euler_inflow_3d
	./bench_euler_inflow_3d
bench-euler-vortex-3d: bench_euler_vortex_3d
	./bench_euler_vortex_3d
bench-euler-vortex-3d-p3: bench_euler_vortex_3d_p3
	./bench_euler_vortex_3d_p3
bench-euler-sod-3d: bench_euler_sod_3d
	./bench_euler_sod_3d
bench-euler-sod-3d-p3: bench_euler_sod_3d_p3
	./bench_euler_sod_3d_p3
bench-euler-sod-3d-p4: bench_euler_sod_3d_p4
	./bench_euler_sod_3d_p4
bench-euler-sod-3d-p5: bench_euler_sod_3d_p5
	./bench_euler_sod_3d_p5
bench-shallow-water-wave-3d: bench_shallow_water_wave_3d
	./bench_shallow_water_wave_3d
bench-shallow-water-wave-3d-p3: bench_shallow_water_wave_3d_p3
	./bench_shallow_water_wave_3d_p3
bench-shallow-water-wave-3d-p4: bench_shallow_water_wave_3d_p4
	./bench_shallow_water_wave_3d_p4
bench-shallow-water-wave-3d-p5: bench_shallow_water_wave_3d_p5
	./bench_shallow_water_wave_3d_p5
bench-shallow-water-inflow-3d: bench_shallow_water_inflow_3d
	./bench_shallow_water_inflow_3d
bench-shallow-water-dam-break-3d: bench_shallow_water_dam_break_3d
	./bench_shallow_water_dam_break_3d
bench-mhd-brio-wu-3d: bench_mhd_brio_wu_3d
	./bench_mhd_brio_wu_3d
bench-mhd-brio-wu-3d-p3: bench_mhd_brio_wu_3d_p3
	./bench_mhd_brio_wu_3d_p3

# Quick smoke-test subset -- one representative bench per physics
# per dimension, plus the 2D limited Sod gate so the BJ limiter +
# shocked-flow path is also covered.  Catches gross regressions in
# ~60s wall-time vs ~10 min for `bench-all`.  Use for fast
# iteration; reach for `bench-all` for the full P-parity matrix +
# all shocked-flow + EM source-term gates.
bench-quick: \
		bench-advection-translation-2d \
		bench-euler-vortex-2d \
		bench-euler-sod-2d \
		bench-euler-sod-limited-2d \
		bench-shallow-water-wave-2d \
		bench-mhd-alfven-2d \
		bench-maxwell-cavity-2d \
		bench-advection-3d \
		bench-euler-sod-3d \
		bench-shallow-water-wave-3d \
		bench-mhd-alfven-3d \
		bench-maxwell-cavity-3d \
		bench-two-fluid-walls-3d
	@echo '=== bench-quick: 13 representative gates PASSED ==='


# Boundary-condition + source-term sweep -- 18 gates exercising
# every BC dispatch arm (BC_INTERIOR / BC_WALL / BC_OUTFLOW /
# BC_INFLOW) across all physics, plus the Euler gravity source
# (hydrostatic) and Euler channel steady-state (3 BC types in one
# gate).  Run when iterating on BC routing, ghost-state assembly,
# inflow_q wiring, or the source-term hook in rk_stage_kernel.
# Run-time ~80s w/ cached binaries (first cold run is ~3-4 min
# since most BC benches aren't in any other aggregator's prebuild
# path).
bench-bcs: \
		bench-advection-outflow-2d bench-advection-inflow-2d \
		bench-advection-outflow-3d bench-advection-inflow-3d \
		bench-euler-channel-steady-2d bench-euler-inflow-2d bench-euler-inflow-3d \
		bench-euler-hydrostatic-2d bench-euler-hydrostatic-2d-p3 \
		bench-euler-hydrostatic-3d bench-euler-hydrostatic-3d-p3 \
		bench-shallow-water-inflow-2d bench-shallow-water-inflow-2d-rusanov bench-shallow-water-inflow-3d \
		bench-mhd-inflow-2d bench-mhd-inflow-2d-glm bench-mhd-inflow-3d \
		bench-mhd-wall-2d bench-mhd-wall-2d-glm bench-mhd-wall-3d \
		bench-maxwell-outflow-2d bench-maxwell-inflow-2d \
		bench-maxwell-outflow-3d bench-maxwell-inflow-3d \
		bench-two-fluid-outflow-3d bench-two-fluid-walls-3d
	@echo '=== bench-bcs: 26 BC + source-term gates PASSED ==='


# Convergence-rate sweep -- 10 gates that explicitly assert
# observed log2(e_N / e_2N) >= a P-dependent floor (rather than
# absolute-L2 sentinels).  These catch order regressions that an
# absolute-L2 gate would miss -- a bug producing a finite but too-
# large error at one resolution can still leave the rate intact;
# a bug breaking the spatial discretization order gets caught here.
# Includes advection 2D + 3D at P=2/3/4/5 (8 gates) and Euler 3D
# at P=2/3 (2 gates -- 2D Euler / Maxwell / SW / MHD all sit at the
# Float32 floor or A^2 nonlinear floor at our default resolutions,
# so a rate gate doesn't fit cleanly there).  Run-time ~50s.
bench-rates: \
		bench-advection-translation-2d bench-advection-translation-2d-p3 \
		bench-advection-translation-2d-p4 bench-advection-translation-2d-p5 \
		bench-advection-3d bench-advection-3d-p3 \
		bench-advection-3d-p4 bench-advection-3d-p5 \
		bench-euler-smooth-wave-3d bench-euler-smooth-wave-3d-p3
	@echo '=== bench-rates: 10 convergence-rate gates PASSED ==='


# Shocked-flow sweep -- 13 gates exercising every Riemann solver +
# limiter path the suite has.  Targets HLLC / HLLEC + BJ limiter
# (Euler Sod 2D / 3D, P=2-5 in each dim), Brio-Wu MHD shock
# (3D P=2 + P=3), and HLL SW dam-break (2D + 3D).  Run when
# iterating on Riemann solvers, the BJ limiter pipeline, or
# anything in the shock-capture path.  Run-time ~70s wall.
bench-shocks: \
		bench-euler-sod-2d bench-euler-sod-3d bench-euler-sod-3d-p3 \
		bench-euler-sod-3d-p4 bench-euler-sod-3d-p5 \
		bench-euler-sod-limited-2d bench-euler-sod-limited-2d-p3 \
		bench-euler-sod-limited-2d-p4 bench-euler-sod-limited-2d-p5 \
		bench-shallow-water-dam-break-2d bench-shallow-water-dam-break-3d \
		bench-mhd-brio-wu-3d bench-mhd-brio-wu-3d-p3
	@echo '=== bench-shocks: 13 shocked-flow gates PASSED ==='


# MHD-focused regression suite -- 33 gates covering every MHD bench
# in the suite: smooth Alfven (2D plain + 2D GLM + 3D, full P-parity),
# GLM rate gates (transport + damp, 2D + 3D, full P-parity), Brio-Wu
# shocks (3D P=2/3), BCs (inflow + wall, 2D plain + 2D GLM + 3D).
# Run when iterating on any MHD code path: physics module, GLM stack,
# BJ limiter on MHD, Riemann solver.  Run-time ~110s w/ cached
# binaries.
bench-mhd: \
		bench-mhd-alfven-2d bench-mhd-alfven-glm-2d \
		bench-mhd-alfven-glm-2d-p3 bench-mhd-alfven-glm-2d-p4 \
		bench-mhd-alfven-glm-2d-p5 \
		bench-mhd-alfven-3d bench-mhd-alfven-3d-p3 \
		bench-mhd-alfven-3d-p4 bench-mhd-alfven-3d-p5 \
		bench-mhd-glm-psi-transport-2d bench-mhd-glm-psi-transport-2d-p3 \
		bench-mhd-glm-psi-transport-2d-p4 bench-mhd-glm-psi-transport-2d-p5 \
		bench-mhd-glm-psi-transport-3d bench-mhd-glm-psi-transport-3d-p3 \
		bench-mhd-glm-psi-transport-3d-p4 bench-mhd-glm-psi-transport-3d-p5 \
		bench-mhd-glm-psi-damp-2d bench-mhd-glm-psi-damp-2d-p3 \
		bench-mhd-glm-psi-damp-2d-p4 bench-mhd-glm-psi-damp-2d-p5 \
		bench-mhd-glm-psi-damp-3d bench-mhd-glm-psi-damp-3d-p3 \
		bench-mhd-glm-psi-damp-3d-p4 bench-mhd-glm-psi-damp-3d-p5 \
		bench-mhd-brio-wu-3d bench-mhd-brio-wu-3d-p3 \
		bench-mhd-inflow-2d bench-mhd-inflow-2d-glm bench-mhd-inflow-3d \
		bench-mhd-wall-2d bench-mhd-wall-2d-glm bench-mhd-wall-3d
	@echo '=== bench-mhd: 33 MHD-focused gates PASSED ==='


# Euler-focused regression suite -- 33 gates covering every Euler
# bench in the suite: smooth (vortex, smooth_wave, hydrostatic, full
# P-parity), shocks (Sod 2D + 3D, sod_limited 2D, full P-parity for
# limited path), BCs (channel-steady, inflow), and the flux-coverage
# 3D gate.  Run when iterating on Euler internals: physics module,
# HLLC / HLLEC Riemann solvers, the BJ limiter on Euler, gravity
# source.  Run-time ~110s w/ cached binaries.
bench-euler: \
		bench-euler-vortex-2d bench-euler-vortex-2d-p3 \
		bench-euler-vortex-3d bench-euler-vortex-3d-p3 \
		bench-euler-smooth-wave-2d bench-euler-smooth-wave-2d-p3 \
		bench-euler-smooth-wave-2d-p4 bench-euler-smooth-wave-2d-p5 \
		bench-euler-smooth-wave-3d bench-euler-smooth-wave-3d-p3 \
		bench-euler-smooth-wave-3d-p4 bench-euler-smooth-wave-3d-p5 \
		bench-euler-sod-2d bench-euler-sod-3d \
		bench-euler-sod-3d-p3 bench-euler-sod-3d-p4 bench-euler-sod-3d-p5 \
		bench-euler-sod-limited-2d bench-euler-sod-limited-2d-p3 \
		bench-euler-sod-limited-2d-p4 bench-euler-sod-limited-2d-p5 \
		bench-euler-hydrostatic-2d bench-euler-hydrostatic-2d-p3 \
		bench-euler-hydrostatic-2d-p4 bench-euler-hydrostatic-2d-p5 \
		bench-euler-hydrostatic-3d bench-euler-hydrostatic-3d-p3 \
		bench-euler-hydrostatic-3d-p4 bench-euler-hydrostatic-3d-p5 \
		bench-euler-channel-steady-2d \
		bench-euler-inflow-2d bench-euler-inflow-3d \
		bench-euler-flux-coverage-3d
	@echo '=== bench-euler: 33 Euler-focused gates PASSED ==='


# Maxwell-focused regression suite -- 23 gates covering every Maxwell
# bench in the suite: cavity (2D + 3D), plane wave (2D + 3D, P=2-5),
# TE plane wave (2D), uniform-J / uniform-M sources (2D + 3D, P=2/3),
# inflow + outflow (2D + 3D).  Run when iterating on Maxwell internals:
# linear-flux face kernel, J/M source coupling, BC dispatch.  Run-time
# ~80s w/ cached binaries.
bench-maxwell: \
		bench-maxwell-cavity-2d bench-maxwell-cavity-3d \
		bench-maxwell-plane-wave-2d bench-maxwell-plane-wave-2d-p3 \
		bench-maxwell-plane-wave-2d-p4 bench-maxwell-plane-wave-2d-p5 \
		bench-maxwell-plane-wave-3d bench-maxwell-plane-wave-3d-p3 \
		bench-maxwell-plane-wave-3d-p4 bench-maxwell-plane-wave-3d-p5 \
		bench-maxwell-te-plane-wave-2d \
		bench-maxwell-uniform-j-2d bench-maxwell-uniform-j-2d-p3 \
		bench-maxwell-uniform-j-3d bench-maxwell-uniform-j-3d-p3 \
		bench-maxwell-uniform-m-2d bench-maxwell-uniform-m-2d-p3 \
		bench-maxwell-uniform-m-3d bench-maxwell-uniform-m-3d-p3 \
		bench-maxwell-inflow-2d bench-maxwell-inflow-3d \
		bench-maxwell-outflow-2d bench-maxwell-outflow-3d
	@echo '=== bench-maxwell: 23 Maxwell-focused gates PASSED ==='


# ShallowWater-focused regression suite -- 14 gates covering every
# SW bench: linear wave (2D + 3D, P=2-5 + Rusanov flux variant for
# 2D), dam-break shocks (2D + 3D), inflow BCs (2D HLL + 2D Rusanov +
# 3D).  Run when iterating on SW internals: HLL / Rusanov face flux,
# wet/dry handling, inflow ghost.  Run-time ~50s w/ cached binaries.
bench-sw: \
		bench-shallow-water-wave-2d bench-shallow-water-wave-2d-p3 \
		bench-shallow-water-wave-2d-p4 bench-shallow-water-wave-2d-p5 \
		bench-shallow-water-wave-2d-rusanov \
		bench-shallow-water-wave-3d bench-shallow-water-wave-3d-p3 \
		bench-shallow-water-wave-3d-p4 bench-shallow-water-wave-3d-p5 \
		bench-shallow-water-dam-break-2d bench-shallow-water-dam-break-3d \
		bench-shallow-water-inflow-2d bench-shallow-water-inflow-2d-rusanov \
		bench-shallow-water-inflow-3d
	@echo '=== bench-sw: 14 ShallowWater-focused gates PASSED ==='


# Advection-focused regression suite -- 12 gates covering scalar
# advection at full P-parity (2D + 3D, P=2/3/4/5) plus inflow +
# outflow BCs in both dimensions.  Run when iterating on the linear
# upwind kernel or BC ghost handling on the simplest physics path
# (NC=1).  Run-time ~25s w/ cached binaries.
bench-advection: \
		bench-advection-translation-2d bench-advection-translation-2d-p3 \
		bench-advection-translation-2d-p4 bench-advection-translation-2d-p5 \
		bench-advection-3d bench-advection-3d-p3 \
		bench-advection-3d-p4 bench-advection-3d-p5 \
		bench-advection-outflow-2d bench-advection-inflow-2d \
		bench-advection-outflow-3d bench-advection-inflow-3d
	@echo '=== bench-advection: 12 Advection-focused gates PASSED ==='


# Two-Fluid focused regression suite -- 6 gates covering the
# FiveMomentTwoFluid (NC=17) physics: Langmuir oscillation (the
# primary plasma test, validates the Maxwell-coupling J source),
# wall + outflow BC preservation (P=2/3/4/5 for walls; P=2 outflow).
# Run when iterating on the Two-Fluid module, the J/M-coupled
# Maxwell stack, or BC dispatch on NC=17.  Run-time ~25s w/ cached
# binaries.
bench-two-fluid: \
		bench-two-fluid-langmuir-3d \
		bench-two-fluid-outflow-3d \
		bench-two-fluid-walls-3d bench-two-fluid-walls-3d-p3 \
		bench-two-fluid-walls-3d-p4 bench-two-fluid-walls-3d-p5
	@echo '=== bench-two-fluid: 6 Two-Fluid-focused gates PASSED ==='


# P=5 P-parity sweep -- 19 gates at the largest comptime config
# the suite covers (2D NP=21 + 3D NP=56 across all 5 smooth physics
# + 2D and 3D limited shocked Sod + 2D and 3D Euler hydrostatic
# (gravity source) + 3D Two-Fluid (NC=17, the largest NC in the
# suite) + 2D and 3D GLM psi exp-decay (operator-split + source-term
# rate gates) + 2D and 3D GLM psi transport (psi advection at c_h)).
# High-P regressions tend to surface here first since these stress
# the comptime template + shared-memory rk_stage_kernel + the BJ
# limiter pipeline hardest.  Run-time ~95s wall.
bench-p5: \
		bench-advection-translation-2d-p5 \
		bench-advection-3d-p5 \
		bench-euler-smooth-wave-2d-p5 \
		bench-euler-smooth-wave-3d-p5 \
		bench-euler-sod-limited-2d-p5 \
		bench-euler-sod-3d-p5 \
		bench-euler-hydrostatic-2d-p5 \
		bench-euler-hydrostatic-3d-p5 \
		bench-maxwell-plane-wave-2d-p5 \
		bench-maxwell-plane-wave-3d-p5 \
		bench-shallow-water-wave-2d-p5 \
		bench-shallow-water-wave-3d-p5 \
		bench-mhd-glm-psi-damp-2d-p5 \
		bench-mhd-glm-psi-damp-3d-p5 \
		bench-mhd-glm-psi-transport-2d-p5 \
		bench-mhd-glm-psi-transport-3d-p5 \
		bench-mhd-alfven-glm-2d-p5 \
		bench-mhd-alfven-3d-p5 \
		bench-two-fluid-walls-3d-p5
	@echo '=== bench-p5: 19 P=5 P-parity gates PASSED ==='


# Aggregate target -- runs every analytic benchmark.  Grouped by
# (dim, physics) for readability; the order doesn't affect correctness
# (each gate is self-contained).  Add a new bench's target name to the
# matching group when adding a new bench.
bench-all: \
		bench-advection-translation-2d bench-advection-translation-2d-p3 \
		bench-advection-translation-2d-p4 bench-advection-translation-2d-p5 \
		bench-advection-outflow-2d bench-advection-inflow-2d \
		bench-euler-vortex-2d bench-euler-vortex-2d-p3 \
		bench-euler-sod-2d \
		bench-euler-sod-limited-2d bench-euler-sod-limited-2d-p3 \
		bench-euler-sod-limited-2d-p4 bench-euler-sod-limited-2d-p5 \
		bench-euler-smooth-wave-2d bench-euler-smooth-wave-2d-p3 \
		bench-euler-smooth-wave-2d-p4 bench-euler-smooth-wave-2d-p5 \
		bench-euler-channel-steady-2d \
		bench-euler-hydrostatic-2d bench-euler-hydrostatic-2d-p3 \
		bench-shallow-water-wave-2d bench-shallow-water-wave-2d-p3 \
		bench-shallow-water-wave-2d-p4 bench-shallow-water-wave-2d-p5 \
		bench-shallow-water-wave-2d-rusanov \
		bench-shallow-water-inflow-2d bench-shallow-water-dam-break-2d \
		bench-mhd-alfven-2d \
		bench-mhd-alfven-glm-2d bench-mhd-alfven-glm-2d-p3 \
		bench-mhd-alfven-glm-2d-p4 bench-mhd-alfven-glm-2d-p5 \
		bench-mhd-glm-psi-transport-2d bench-mhd-glm-psi-transport-2d-p3 \
		bench-mhd-glm-psi-damp-2d bench-mhd-glm-psi-damp-2d-p3 \
		bench-maxwell-cavity-2d \
		bench-maxwell-plane-wave-2d bench-maxwell-te-plane-wave-2d \
		bench-maxwell-plane-wave-2d-p3 bench-maxwell-plane-wave-2d-p4 \
		bench-maxwell-plane-wave-2d-p5 \
		bench-maxwell-outflow-2d bench-maxwell-inflow-2d \
		bench-maxwell-uniform-j-2d bench-maxwell-uniform-j-2d-p3 \
		bench-maxwell-uniform-m-2d bench-maxwell-uniform-m-2d-p3 \
		bench-advection-3d bench-advection-3d-p3 \
		bench-advection-3d-p4 bench-advection-3d-p5 \
		bench-advection-outflow-3d bench-advection-inflow-3d \
		bench-euler-vortex-3d bench-euler-vortex-3d-p3 \
		bench-euler-sod-3d bench-euler-sod-3d-p3 bench-euler-sod-3d-p4 \
		bench-euler-sod-3d-p5 \
		bench-euler-smooth-wave-3d bench-euler-smooth-wave-3d-p3 \
		bench-euler-smooth-wave-3d-p4 bench-euler-smooth-wave-3d-p5 \
		bench-euler-flux-coverage-3d \
		bench-euler-hydrostatic-3d bench-euler-hydrostatic-3d-p3 \
		bench-euler-inflow-3d \
		bench-shallow-water-wave-3d bench-shallow-water-wave-3d-p3 \
		bench-shallow-water-wave-3d-p4 bench-shallow-water-wave-3d-p5 \
		bench-shallow-water-inflow-3d \
		bench-shallow-water-dam-break-3d \
		bench-mhd-alfven-3d bench-mhd-alfven-3d-p3 bench-mhd-alfven-3d-p4 \
		bench-mhd-alfven-3d-p5 \
		bench-mhd-glm-psi-damp-3d bench-mhd-glm-psi-damp-3d-p3 \
		bench-mhd-glm-psi-transport-3d bench-mhd-glm-psi-transport-3d-p3 \
		bench-mhd-brio-wu-3d bench-mhd-brio-wu-3d-p3 \
		bench-maxwell-cavity-3d \
		bench-maxwell-plane-wave-3d bench-maxwell-plane-wave-3d-p3 \
		bench-maxwell-plane-wave-3d-p4 bench-maxwell-plane-wave-3d-p5 \
		bench-maxwell-uniform-j-3d bench-maxwell-uniform-j-3d-p3 \
		bench-maxwell-uniform-m-3d bench-maxwell-uniform-m-3d-p3 \
		bench-maxwell-outflow-3d bench-maxwell-inflow-3d \
		bench-two-fluid-langmuir-3d bench-two-fluid-outflow-3d \
		bench-two-fluid-walls-3d
	@echo '=== ALL BENCHMARKS PASSED ==='

# Profiling: run a benchmark under nsys with --stats=true and capture
# the kernel-time summary to benchmarks/profile_reports/<name>.kern.txt.
# Requires nsys on PATH (ships with CUDA).  The awk filter keeps only
# the kernel-time table from nsys's multi-section report for
# diff-ability across runs; the full .nsys-rep trace is discarded.
PROFILE_BIN = mkdir -p benchmarks/profile_reports; \
              nsys profile --stats=true --force-overwrite=true \
                  -o benchmarks/profile_reports/$$bin ./$$bin 2>&1 \
              | awk '/Executing .cuda_gpu_kern_sum. stats report/,/\[7\/8\]/' \
              > benchmarks/profile_reports/$$bin.kern.txt; \
              echo "Kernel summary: benchmarks/profile_reports/$$bin.kern.txt"; \
              cat benchmarks/profile_reports/$$bin.kern.txt; \
              rm -f benchmarks/profile_reports/$$bin.nsys-rep benchmarks/profile_reports/$$bin.sqlite

profile-bench-advection-translation-2d: bench_advection_translation_2d
	@bin=bench_advection_translation_2d; $(PROFILE_BIN)
profile-bench-advection-translation-2d-p3: bench_advection_translation_2d_p3
	@bin=bench_advection_translation_2d_p3; $(PROFILE_BIN)
profile-bench-advection-translation-2d-p4: bench_advection_translation_2d_p4
	@bin=bench_advection_translation_2d_p4; $(PROFILE_BIN)
profile-bench-advection-translation-2d-p5: bench_advection_translation_2d_p5
	@bin=bench_advection_translation_2d_p5; $(PROFILE_BIN)
profile-bench-advection-outflow-2d: bench_advection_outflow_2d
	@bin=bench_advection_outflow_2d; $(PROFILE_BIN)
profile-bench-advection-inflow-2d: bench_advection_inflow_2d
	@bin=bench_advection_inflow_2d; $(PROFILE_BIN)
profile-bench-euler-vortex-2d: bench_euler_vortex_2d
	@bin=bench_euler_vortex_2d; $(PROFILE_BIN)
profile-bench-euler-vortex-2d-p3: bench_euler_vortex_2d_p3
	@bin=bench_euler_vortex_2d_p3; $(PROFILE_BIN)
profile-bench-mhd-alfven-2d: bench_mhd_alfven_2d
	@bin=bench_mhd_alfven_2d; $(PROFILE_BIN)
profile-bench-mhd-alfven-glm-2d: bench_mhd_alfven_glm_2d
	@bin=bench_mhd_alfven_glm_2d; $(PROFILE_BIN)
profile-bench-mhd-alfven-glm-2d-p3: bench_mhd_alfven_glm_2d_p3
	@bin=bench_mhd_alfven_glm_2d_p3; $(PROFILE_BIN)
profile-bench-mhd-alfven-glm-2d-p4: bench_mhd_alfven_glm_2d_p4
	@bin=bench_mhd_alfven_glm_2d_p4; $(PROFILE_BIN)
profile-bench-mhd-alfven-glm-2d-p5: bench_mhd_alfven_glm_2d_p5
	@bin=bench_mhd_alfven_glm_2d_p5; $(PROFILE_BIN)
profile-bench-mhd-glm-psi-transport-2d: bench_mhd_glm_psi_transport_2d
	@bin=bench_mhd_glm_psi_transport_2d; $(PROFILE_BIN)
profile-bench-mhd-glm-psi-transport-2d-p3: bench_mhd_glm_psi_transport_2d_p3
	@bin=bench_mhd_glm_psi_transport_2d_p3; $(PROFILE_BIN)
profile-bench-mhd-glm-psi-damp-2d: bench_mhd_glm_psi_damp_2d
	@bin=bench_mhd_glm_psi_damp_2d; $(PROFILE_BIN)
profile-bench-mhd-glm-psi-damp-2d-p3: bench_mhd_glm_psi_damp_2d_p3
	@bin=bench_mhd_glm_psi_damp_2d_p3; $(PROFILE_BIN)
profile-bench-euler-sod-2d: bench_euler_sod_2d
	@bin=bench_euler_sod_2d; $(PROFILE_BIN)
profile-bench-euler-sod-limited-2d: bench_euler_sod_limited_2d
	@bin=bench_euler_sod_limited_2d; $(PROFILE_BIN)
profile-bench-euler-sod-limited-2d-p3: bench_euler_sod_limited_2d_p3
	@bin=bench_euler_sod_limited_2d_p3; $(PROFILE_BIN)
profile-bench-euler-sod-limited-2d-p4: bench_euler_sod_limited_2d_p4
	@bin=bench_euler_sod_limited_2d_p4; $(PROFILE_BIN)
profile-bench-euler-sod-limited-2d-p5: bench_euler_sod_limited_2d_p5
	@bin=bench_euler_sod_limited_2d_p5; $(PROFILE_BIN)
profile-bench-euler-smooth-wave-2d: bench_euler_smooth_wave_2d
	@bin=bench_euler_smooth_wave_2d; $(PROFILE_BIN)
profile-bench-euler-smooth-wave-2d-p3: bench_euler_smooth_wave_2d_p3
	@bin=bench_euler_smooth_wave_2d_p3; $(PROFILE_BIN)
profile-bench-euler-smooth-wave-2d-p4: bench_euler_smooth_wave_2d_p4
	@bin=bench_euler_smooth_wave_2d_p4; $(PROFILE_BIN)
profile-bench-euler-smooth-wave-2d-p5: bench_euler_smooth_wave_2d_p5
	@bin=bench_euler_smooth_wave_2d_p5; $(PROFILE_BIN)
profile-bench-euler-hydrostatic-2d: bench_euler_hydrostatic_2d
	@bin=bench_euler_hydrostatic_2d; $(PROFILE_BIN)
profile-bench-euler-hydrostatic-2d-p3: bench_euler_hydrostatic_2d_p3
	@bin=bench_euler_hydrostatic_2d_p3; $(PROFILE_BIN)
profile-bench-euler-channel-steady-2d: bench_euler_channel_steady_2d
	@bin=bench_euler_channel_steady_2d; $(PROFILE_BIN)
profile-bench-shallow-water-wave-2d: bench_shallow_water_wave_2d
	@bin=bench_shallow_water_wave_2d; $(PROFILE_BIN)
profile-bench-shallow-water-wave-2d-p3: bench_shallow_water_wave_2d_p3
	@bin=bench_shallow_water_wave_2d_p3; $(PROFILE_BIN)
profile-bench-shallow-water-wave-2d-p4: bench_shallow_water_wave_2d_p4
	@bin=bench_shallow_water_wave_2d_p4; $(PROFILE_BIN)
profile-bench-shallow-water-wave-2d-p5: bench_shallow_water_wave_2d_p5
	@bin=bench_shallow_water_wave_2d_p5; $(PROFILE_BIN)
profile-bench-shallow-water-wave-2d-rusanov: bench_shallow_water_wave_2d_rusanov
	@bin=bench_shallow_water_wave_2d_rusanov; $(PROFILE_BIN)
profile-bench-shallow-water-inflow-2d: bench_shallow_water_inflow_2d
	@bin=bench_shallow_water_inflow_2d; $(PROFILE_BIN)
profile-bench-shallow-water-dam-break-2d: bench_shallow_water_dam_break_2d
	@bin=bench_shallow_water_dam_break_2d; $(PROFILE_BIN)
profile-bench-maxwell-cavity-2d: bench_maxwell_cavity_2d
	@bin=bench_maxwell_cavity_2d; $(PROFILE_BIN)
profile-bench-maxwell-plane-wave-2d: bench_maxwell_plane_wave_2d
	@bin=bench_maxwell_plane_wave_2d; $(PROFILE_BIN)
profile-bench-maxwell-te-plane-wave-2d: bench_maxwell_te_plane_wave_2d
	@bin=bench_maxwell_te_plane_wave_2d; $(PROFILE_BIN)
profile-bench-maxwell-plane-wave-2d-p3: bench_maxwell_plane_wave_2d_p3
	@bin=bench_maxwell_plane_wave_2d_p3; $(PROFILE_BIN)
profile-bench-maxwell-plane-wave-2d-p4: bench_maxwell_plane_wave_2d_p4
	@bin=bench_maxwell_plane_wave_2d_p4; $(PROFILE_BIN)
profile-bench-maxwell-plane-wave-2d-p5: bench_maxwell_plane_wave_2d_p5
	@bin=bench_maxwell_plane_wave_2d_p5; $(PROFILE_BIN)
profile-bench-maxwell-outflow-2d: bench_maxwell_outflow_2d
	@bin=bench_maxwell_outflow_2d; $(PROFILE_BIN)
profile-bench-maxwell-inflow-2d: bench_maxwell_inflow_2d
	@bin=bench_maxwell_inflow_2d; $(PROFILE_BIN)
profile-bench-maxwell-uniform-j-2d: bench_maxwell_uniform_j_2d
	@bin=bench_maxwell_uniform_j_2d; $(PROFILE_BIN)
profile-bench-maxwell-uniform-j-2d-p3: bench_maxwell_uniform_j_2d_p3
	@bin=bench_maxwell_uniform_j_2d_p3; $(PROFILE_BIN)
profile-bench-maxwell-uniform-m-2d: bench_maxwell_uniform_m_2d
	@bin=bench_maxwell_uniform_m_2d; $(PROFILE_BIN)
profile-bench-maxwell-uniform-m-2d-p3: bench_maxwell_uniform_m_2d_p3
	@bin=bench_maxwell_uniform_m_2d_p3; $(PROFILE_BIN)
profile-bench-advection-3d: bench_advection_3d
	@bin=bench_advection_3d; $(PROFILE_BIN)
profile-bench-advection-3d-p3: bench_advection_3d_p3
	@bin=bench_advection_3d_p3; $(PROFILE_BIN)
profile-bench-advection-3d-p4: bench_advection_3d_p4
	@bin=bench_advection_3d_p4; $(PROFILE_BIN)
profile-bench-advection-3d-p5: bench_advection_3d_p5
	@bin=bench_advection_3d_p5; $(PROFILE_BIN)
profile-bench-advection-outflow-3d: bench_advection_outflow_3d
	@bin=bench_advection_outflow_3d; $(PROFILE_BIN)
profile-bench-advection-inflow-3d: bench_advection_inflow_3d
	@bin=bench_advection_inflow_3d; $(PROFILE_BIN)
profile-bench-mhd-alfven-3d: bench_mhd_alfven_3d
	@bin=bench_mhd_alfven_3d; $(PROFILE_BIN)
profile-bench-mhd-alfven-3d-p3: bench_mhd_alfven_3d_p3
	@bin=bench_mhd_alfven_3d_p3; $(PROFILE_BIN)
profile-bench-mhd-alfven-3d-p4: bench_mhd_alfven_3d_p4
	@bin=bench_mhd_alfven_3d_p4; $(PROFILE_BIN)
profile-bench-mhd-alfven-3d-p5: bench_mhd_alfven_3d_p5
	@bin=bench_mhd_alfven_3d_p5; $(PROFILE_BIN)
profile-bench-mhd-glm-psi-damp-3d: bench_mhd_glm_psi_damp_3d
	@bin=bench_mhd_glm_psi_damp_3d; $(PROFILE_BIN)
profile-bench-mhd-glm-psi-damp-3d-p3: bench_mhd_glm_psi_damp_3d_p3
	@bin=bench_mhd_glm_psi_damp_3d_p3; $(PROFILE_BIN)
profile-bench-mhd-glm-psi-transport-3d: bench_mhd_glm_psi_transport_3d
	@bin=bench_mhd_glm_psi_transport_3d; $(PROFILE_BIN)
profile-bench-mhd-glm-psi-transport-3d-p3: bench_mhd_glm_psi_transport_3d_p3
	@bin=bench_mhd_glm_psi_transport_3d_p3; $(PROFILE_BIN)
profile-bench-maxwell-cavity-3d: bench_maxwell_cavity_3d
	@bin=bench_maxwell_cavity_3d; $(PROFILE_BIN)
profile-bench-maxwell-plane-wave-3d: bench_maxwell_plane_wave_3d
	@bin=bench_maxwell_plane_wave_3d; $(PROFILE_BIN)
profile-bench-maxwell-plane-wave-3d-p3: bench_maxwell_plane_wave_3d_p3
	@bin=bench_maxwell_plane_wave_3d_p3; $(PROFILE_BIN)
profile-bench-maxwell-plane-wave-3d-p4: bench_maxwell_plane_wave_3d_p4
	@bin=bench_maxwell_plane_wave_3d_p4; $(PROFILE_BIN)
profile-bench-maxwell-plane-wave-3d-p5: bench_maxwell_plane_wave_3d_p5
	@bin=bench_maxwell_plane_wave_3d_p5; $(PROFILE_BIN)
profile-bench-maxwell-uniform-j-3d: bench_maxwell_uniform_j_3d
	@bin=bench_maxwell_uniform_j_3d; $(PROFILE_BIN)
profile-bench-maxwell-uniform-j-3d-p3: bench_maxwell_uniform_j_3d_p3
	@bin=bench_maxwell_uniform_j_3d_p3; $(PROFILE_BIN)
profile-bench-maxwell-uniform-m-3d: bench_maxwell_uniform_m_3d
	@bin=bench_maxwell_uniform_m_3d; $(PROFILE_BIN)
profile-bench-maxwell-uniform-m-3d-p3: bench_maxwell_uniform_m_3d_p3
	@bin=bench_maxwell_uniform_m_3d_p3; $(PROFILE_BIN)
profile-bench-maxwell-outflow-3d: bench_maxwell_outflow_3d
	@bin=bench_maxwell_outflow_3d; $(PROFILE_BIN)
profile-bench-maxwell-inflow-3d: bench_maxwell_inflow_3d
	@bin=bench_maxwell_inflow_3d; $(PROFILE_BIN)
profile-bench-two-fluid-langmuir-3d: bench_two_fluid_langmuir_3d
	@bin=bench_two_fluid_langmuir_3d; $(PROFILE_BIN)
profile-bench-two-fluid-outflow-3d: bench_two_fluid_outflow_3d
	@bin=bench_two_fluid_outflow_3d; $(PROFILE_BIN)
profile-bench-two-fluid-walls-3d: bench_two_fluid_walls_3d
	@bin=bench_two_fluid_walls_3d; $(PROFILE_BIN)
profile-bench-euler-smooth-wave-3d: bench_euler_smooth_wave_3d
	@bin=bench_euler_smooth_wave_3d; $(PROFILE_BIN)
profile-bench-euler-smooth-wave-3d-p3: bench_euler_smooth_wave_3d_p3
	@bin=bench_euler_smooth_wave_3d_p3; $(PROFILE_BIN)
profile-bench-euler-smooth-wave-3d-p4: bench_euler_smooth_wave_3d_p4
	@bin=bench_euler_smooth_wave_3d_p4; $(PROFILE_BIN)
profile-bench-euler-smooth-wave-3d-p5: bench_euler_smooth_wave_3d_p5
	@bin=bench_euler_smooth_wave_3d_p5; $(PROFILE_BIN)
profile-bench-euler-flux-coverage-3d: bench_euler_flux_coverage_3d
	@bin=bench_euler_flux_coverage_3d; $(PROFILE_BIN)
profile-bench-euler-hydrostatic-3d: bench_euler_hydrostatic_3d
	@bin=bench_euler_hydrostatic_3d; $(PROFILE_BIN)
profile-bench-euler-hydrostatic-3d-p3: bench_euler_hydrostatic_3d_p3
	@bin=bench_euler_hydrostatic_3d_p3; $(PROFILE_BIN)
profile-bench-euler-inflow-3d: bench_euler_inflow_3d
	@bin=bench_euler_inflow_3d; $(PROFILE_BIN)
profile-bench-euler-vortex-3d: bench_euler_vortex_3d
	@bin=bench_euler_vortex_3d; $(PROFILE_BIN)
profile-bench-euler-vortex-3d-p3: bench_euler_vortex_3d_p3
	@bin=bench_euler_vortex_3d_p3; $(PROFILE_BIN)
profile-bench-euler-sod-3d: bench_euler_sod_3d
	@bin=bench_euler_sod_3d; $(PROFILE_BIN)
profile-bench-euler-sod-3d-p3: bench_euler_sod_3d_p3
	@bin=bench_euler_sod_3d_p3; $(PROFILE_BIN)
profile-bench-euler-sod-3d-p4: bench_euler_sod_3d_p4
	@bin=bench_euler_sod_3d_p4; $(PROFILE_BIN)
profile-bench-euler-sod-3d-p5: bench_euler_sod_3d_p5
	@bin=bench_euler_sod_3d_p5; $(PROFILE_BIN)
profile-bench-shallow-water-wave-3d: bench_shallow_water_wave_3d
	@bin=bench_shallow_water_wave_3d; $(PROFILE_BIN)
profile-bench-shallow-water-wave-3d-p3: bench_shallow_water_wave_3d_p3
	@bin=bench_shallow_water_wave_3d_p3; $(PROFILE_BIN)
profile-bench-shallow-water-wave-3d-p4: bench_shallow_water_wave_3d_p4
	@bin=bench_shallow_water_wave_3d_p4; $(PROFILE_BIN)
profile-bench-shallow-water-wave-3d-p5: bench_shallow_water_wave_3d_p5
	@bin=bench_shallow_water_wave_3d_p5; $(PROFILE_BIN)
profile-bench-shallow-water-inflow-3d: bench_shallow_water_inflow_3d
	@bin=bench_shallow_water_inflow_3d; $(PROFILE_BIN)
profile-bench-shallow-water-dam-break-3d: bench_shallow_water_dam_break_3d
	@bin=bench_shallow_water_dam_break_3d; $(PROFILE_BIN)
profile-bench-mhd-brio-wu-3d: bench_mhd_brio_wu_3d
	@bin=bench_mhd_brio_wu_3d; $(PROFILE_BIN)
profile-bench-mhd-brio-wu-3d-p3: bench_mhd_brio_wu_3d_p3
	@bin=bench_mhd_brio_wu_3d_p3; $(PROFILE_BIN)
profile-bench-euler-hydrostatic-2d-p4: bench_euler_hydrostatic_2d_p4
	@bin=bench_euler_hydrostatic_2d_p4; $(PROFILE_BIN)
profile-bench-euler-hydrostatic-2d-p5: bench_euler_hydrostatic_2d_p5
	@bin=bench_euler_hydrostatic_2d_p5; $(PROFILE_BIN)
profile-bench-euler-hydrostatic-3d-p4: bench_euler_hydrostatic_3d_p4
	@bin=bench_euler_hydrostatic_3d_p4; $(PROFILE_BIN)
profile-bench-euler-hydrostatic-3d-p5: bench_euler_hydrostatic_3d_p5
	@bin=bench_euler_hydrostatic_3d_p5; $(PROFILE_BIN)
profile-bench-euler-inflow-2d: bench_euler_inflow_2d
	@bin=bench_euler_inflow_2d; $(PROFILE_BIN)
profile-bench-mhd-glm-psi-damp-2d-p4: bench_mhd_glm_psi_damp_2d_p4
	@bin=bench_mhd_glm_psi_damp_2d_p4; $(PROFILE_BIN)
profile-bench-mhd-glm-psi-damp-2d-p5: bench_mhd_glm_psi_damp_2d_p5
	@bin=bench_mhd_glm_psi_damp_2d_p5; $(PROFILE_BIN)
profile-bench-mhd-glm-psi-damp-3d-p4: bench_mhd_glm_psi_damp_3d_p4
	@bin=bench_mhd_glm_psi_damp_3d_p4; $(PROFILE_BIN)
profile-bench-mhd-glm-psi-damp-3d-p5: bench_mhd_glm_psi_damp_3d_p5
	@bin=bench_mhd_glm_psi_damp_3d_p5; $(PROFILE_BIN)
profile-bench-mhd-glm-psi-transport-2d-p4: bench_mhd_glm_psi_transport_2d_p4
	@bin=bench_mhd_glm_psi_transport_2d_p4; $(PROFILE_BIN)
profile-bench-mhd-glm-psi-transport-2d-p5: bench_mhd_glm_psi_transport_2d_p5
	@bin=bench_mhd_glm_psi_transport_2d_p5; $(PROFILE_BIN)
profile-bench-mhd-glm-psi-transport-3d-p4: bench_mhd_glm_psi_transport_3d_p4
	@bin=bench_mhd_glm_psi_transport_3d_p4; $(PROFILE_BIN)
profile-bench-mhd-glm-psi-transport-3d-p5: bench_mhd_glm_psi_transport_3d_p5
	@bin=bench_mhd_glm_psi_transport_3d_p5; $(PROFILE_BIN)
profile-bench-mhd-inflow-2d: bench_mhd_inflow_2d
	@bin=bench_mhd_inflow_2d; $(PROFILE_BIN)
profile-bench-mhd-inflow-2d-glm: bench_mhd_inflow_2d_glm
	@bin=bench_mhd_inflow_2d_glm; $(PROFILE_BIN)
profile-bench-mhd-inflow-3d: bench_mhd_inflow_3d
	@bin=bench_mhd_inflow_3d; $(PROFILE_BIN)
profile-bench-mhd-wall-2d: bench_mhd_wall_2d
	@bin=bench_mhd_wall_2d; $(PROFILE_BIN)
profile-bench-mhd-wall-2d-glm: bench_mhd_wall_2d_glm
	@bin=bench_mhd_wall_2d_glm; $(PROFILE_BIN)
profile-bench-mhd-wall-3d: bench_mhd_wall_3d
	@bin=bench_mhd_wall_3d; $(PROFILE_BIN)
profile-bench-shallow-water-inflow-2d-rusanov: bench_shallow_water_inflow_2d_rusanov
	@bin=bench_shallow_water_inflow_2d_rusanov; $(PROFILE_BIN)
profile-bench-two-fluid-walls-3d-p3: bench_two_fluid_walls_3d_p3
	@bin=bench_two_fluid_walls_3d_p3; $(PROFILE_BIN)
profile-bench-two-fluid-walls-3d-p4: bench_two_fluid_walls_3d_p4
	@bin=bench_two_fluid_walls_3d_p4; $(PROFILE_BIN)
profile-bench-two-fluid-walls-3d-p5: bench_two_fluid_walls_3d_p5
	@bin=bench_two_fluid_walls_3d_p5; $(PROFILE_BIN)

# Profile-bench-quick: refresh the kern.txt baselines for just the
# 13 bench-quick gates -- one representative bench per physics per
# dim + the 2D limited Sod gate.  Use after a kernel-level change
# to refresh the most-used profile baselines without paying for the
# full 121-bench `profile-bench-all` sweep.  Run-time scales as
# 13 * (~30-60 s nsys-profile overhead per bench), so ~10-15 min
# wall vs ~90 min for the full all-bench sweep.
profile-bench-quick: \
		profile-bench-advection-translation-2d \
		profile-bench-euler-vortex-2d \
		profile-bench-euler-sod-2d \
		profile-bench-euler-sod-limited-2d \
		profile-bench-shallow-water-wave-2d \
		profile-bench-mhd-alfven-2d \
		profile-bench-maxwell-cavity-2d \
		profile-bench-advection-3d \
		profile-bench-euler-sod-3d \
		profile-bench-shallow-water-wave-3d \
		profile-bench-mhd-alfven-3d \
		profile-bench-maxwell-cavity-3d \
		profile-bench-two-fluid-walls-3d
	@echo '=== profile-bench-quick: 13 representative profiles refreshed ==='


# Aggregate profile target -- captures kernel summaries for every
# bench under nsys.  Same per-physics grouping as `bench-all`.
profile-bench-all: \
		profile-bench-advection-translation-2d profile-bench-advection-translation-2d-p3 \
		profile-bench-advection-translation-2d-p4 profile-bench-advection-translation-2d-p5 \
		profile-bench-advection-outflow-2d profile-bench-advection-inflow-2d \
		profile-bench-euler-vortex-2d profile-bench-euler-vortex-2d-p3 \
		profile-bench-euler-sod-2d \
		profile-bench-euler-sod-limited-2d profile-bench-euler-sod-limited-2d-p3 \
		profile-bench-euler-sod-limited-2d-p4 profile-bench-euler-sod-limited-2d-p5 \
		profile-bench-euler-smooth-wave-2d profile-bench-euler-smooth-wave-2d-p3 \
		profile-bench-euler-smooth-wave-2d-p4 profile-bench-euler-smooth-wave-2d-p5 \
		profile-bench-euler-channel-steady-2d \
		profile-bench-euler-hydrostatic-2d profile-bench-euler-hydrostatic-2d-p3 \
		profile-bench-shallow-water-wave-2d profile-bench-shallow-water-wave-2d-p3 \
		profile-bench-shallow-water-wave-2d-p4 profile-bench-shallow-water-wave-2d-p5 \
		profile-bench-shallow-water-wave-2d-rusanov \
		profile-bench-shallow-water-inflow-2d profile-bench-shallow-water-dam-break-2d \
		profile-bench-mhd-alfven-2d \
		profile-bench-mhd-alfven-glm-2d profile-bench-mhd-alfven-glm-2d-p3 \
		profile-bench-mhd-alfven-glm-2d-p4 profile-bench-mhd-alfven-glm-2d-p5 \
		profile-bench-mhd-glm-psi-transport-2d profile-bench-mhd-glm-psi-transport-2d-p3 \
		profile-bench-mhd-glm-psi-damp-2d profile-bench-mhd-glm-psi-damp-2d-p3 \
		profile-bench-maxwell-cavity-2d \
		profile-bench-maxwell-plane-wave-2d profile-bench-maxwell-te-plane-wave-2d \
		profile-bench-maxwell-plane-wave-2d-p3 profile-bench-maxwell-plane-wave-2d-p4 \
		profile-bench-maxwell-plane-wave-2d-p5 \
		profile-bench-maxwell-outflow-2d profile-bench-maxwell-inflow-2d \
		profile-bench-maxwell-uniform-j-2d profile-bench-maxwell-uniform-j-2d-p3 \
		profile-bench-maxwell-uniform-m-2d profile-bench-maxwell-uniform-m-2d-p3 \
		profile-bench-advection-3d profile-bench-advection-3d-p3 \
		profile-bench-advection-3d-p4 profile-bench-advection-3d-p5 \
		profile-bench-advection-outflow-3d profile-bench-advection-inflow-3d \
		profile-bench-euler-vortex-3d profile-bench-euler-vortex-3d-p3 \
		profile-bench-euler-sod-3d profile-bench-euler-sod-3d-p3 profile-bench-euler-sod-3d-p4 \
		profile-bench-euler-sod-3d-p5 \
		profile-bench-euler-smooth-wave-3d profile-bench-euler-smooth-wave-3d-p3 \
		profile-bench-euler-smooth-wave-3d-p4 profile-bench-euler-smooth-wave-3d-p5 \
		profile-bench-euler-flux-coverage-3d \
		profile-bench-euler-hydrostatic-3d profile-bench-euler-hydrostatic-3d-p3 \
		profile-bench-euler-inflow-3d \
		profile-bench-shallow-water-wave-3d profile-bench-shallow-water-wave-3d-p3 \
		profile-bench-shallow-water-wave-3d-p4 profile-bench-shallow-water-wave-3d-p5 \
		profile-bench-shallow-water-inflow-3d \
		profile-bench-shallow-water-dam-break-3d \
		profile-bench-mhd-alfven-3d profile-bench-mhd-alfven-3d-p3 \
		profile-bench-mhd-alfven-3d-p4 profile-bench-mhd-alfven-3d-p5 \
		profile-bench-mhd-glm-psi-damp-3d profile-bench-mhd-glm-psi-damp-3d-p3 \
		profile-bench-mhd-glm-psi-transport-3d profile-bench-mhd-glm-psi-transport-3d-p3 \
		profile-bench-mhd-brio-wu-3d profile-bench-mhd-brio-wu-3d-p3 \
		profile-bench-maxwell-cavity-3d \
		profile-bench-maxwell-plane-wave-3d profile-bench-maxwell-plane-wave-3d-p3 \
		profile-bench-maxwell-plane-wave-3d-p4 profile-bench-maxwell-plane-wave-3d-p5 \
		profile-bench-maxwell-uniform-j-3d profile-bench-maxwell-uniform-j-3d-p3 \
		profile-bench-maxwell-uniform-m-3d profile-bench-maxwell-uniform-m-3d-p3 \
		profile-bench-maxwell-outflow-3d profile-bench-maxwell-inflow-3d \
		profile-bench-two-fluid-langmuir-3d profile-bench-two-fluid-outflow-3d \
		profile-bench-two-fluid-walls-3d profile-bench-two-fluid-walls-3d-p3 \
		profile-bench-two-fluid-walls-3d-p4 profile-bench-two-fluid-walls-3d-p5 \
		profile-bench-euler-hydrostatic-2d-p4 profile-bench-euler-hydrostatic-2d-p5 \
		profile-bench-euler-hydrostatic-3d-p4 profile-bench-euler-hydrostatic-3d-p5 \
		profile-bench-euler-inflow-2d \
		profile-bench-shallow-water-inflow-2d-rusanov \
		profile-bench-mhd-glm-psi-damp-2d-p4 profile-bench-mhd-glm-psi-damp-2d-p5 \
		profile-bench-mhd-glm-psi-damp-3d-p4 profile-bench-mhd-glm-psi-damp-3d-p5 \
		profile-bench-mhd-glm-psi-transport-2d-p4 profile-bench-mhd-glm-psi-transport-2d-p5 \
		profile-bench-mhd-glm-psi-transport-3d-p4 profile-bench-mhd-glm-psi-transport-3d-p5 \
		profile-bench-mhd-inflow-2d profile-bench-mhd-inflow-2d-glm \
		profile-bench-mhd-inflow-3d \
		profile-bench-mhd-wall-2d profile-bench-mhd-wall-2d-glm \
		profile-bench-mhd-wall-3d
	@echo '=== All profile reports written to benchmarks/profile_reports/ ==='

test-klone:
	test/test_mpi_correctness.sh --klone

$(BUILD_DIR)/mpi_shim.o: src/mpi_shim.c | $(BUILD_DIR)
	$(MPICC) -O2 -fPIC -c $< -o $@

$(BUILD_DIR):
	mkdir -p $@

# `make format` rewrites every .mojo file in place via `mojo format`.
# `make install-hooks` points git's hooksPath at scripts/git-hooks/,
# enabling the pre-commit formatter check.  Each contributor must run
# `make install-hooks` once per clone (git intentionally won't auto-
# enable hooks from a freshly-cloned repo).
MOJO_SOURCES := $(shell find src benchmarks examples test -name '*.mojo')

format:
	$(MOJO) format $(MOJO_SOURCES)

# Non-mutating CI-style format gate.  Mirrors the pre-commit hook but
# covers the *whole working tree*, not just staged files.  Exits
# non-zero if any .mojo file would be rewritten by `mojo format`.
format-check:
	@scripts/format_check.sh

install-hooks:
	git config core.hooksPath scripts/git-hooks
	@echo 'pre-commit hook enabled (scripts/git-hooks/pre-commit).'

# Cross-bench profile summary -- ranks every bench in
# benchmarks/profile_reports/ by dominant-kernel avg us/launch.
# Quick visual scan of where the per-step cost lives across the suite.
# Re-run `make profile-bench-all` first to refresh the underlying
# .kern.txt baselines.
profile-summary:
	@scripts/profile_summary.py

clean:
	rm -f $(ALL_DRIVERS) $(TEST_DRIVERS)
	rm -rf $(BUILD_DIR) test/dumps_np1 test/dumps_np4 output
