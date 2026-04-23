#!/usr/bin/env bash
# ======================================================================
# test_mpi_bc_correctness.sh -- np=1 vs np=4 for non-periodic BCs
#
# Runs test/mpi_bc_test.mojo (BC_OUTFLOW on all 6 sides) at np=1 and
# np=4 for NUM_TEST_STEPS steps and diffs the per-rank dumps.
#
# Exits 0 on PASS, 1 on FAIL.
# ======================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

run_np() {
    local np="$1"
    local subdir="$2"
    rm -rf "$subdir"
    mkdir -p "$subdir"
    rm -rf output && mkdir output
    mpirun -np "$np" ./mpi_bc_test > "$subdir/run.log" 2>&1
    mv output/final_q_rank_*.bin "$subdir/"
}

echo "=== build mpi_bc_test ==="
make -s mpi_bc_test

echo "=== run np=1 (BC_OUTFLOW x 6) ==="
run_np 1 "test/dumps_bc_np1"

echo "=== run np=4 (BC_OUTFLOW x 6) ==="
run_np 4 "test/dumps_bc_np4"

echo "=== diff ==="
if python3 "$SCRIPT_DIR/diff_mpi_dumps.py" \
    --a test/dumps_bc_np1 \
    --b test/dumps_bc_np4 \
    --tol-abs 1e-5 \
    --tol-rel 1e-4
then
    echo "=== MPI BC correctness test PASSED ==="
    exit 0
else
    echo "=== MPI BC correctness test FAILED ==="
    echo "    inspect test/dumps_bc_np1/run.log and test/dumps_bc_np4/run.log"
    exit 1
fi
