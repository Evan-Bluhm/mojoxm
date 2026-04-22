#!/usr/bin/env bash
# ======================================================================
# test_mpi_correctness.sh -- verify np=1 and np=4 produce the same
# final q field (within FP tolerance).
#
# Runs test/mpi_advection_test.mojo for 50 steps at np=1 and np=4,
# moves each rank's owned q dump into test/dumps_np<N>/, and invokes
# test/diff_mpi_dumps.py to compare.
#
# Usage:
#     test/test_mpi_correctness.sh                    # local WSL2 path
#     test/test_mpi_correctness.sh --klone            # cluster path
#
# Exits 0 on PASS, 1 on FAIL.
# ======================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

MODE="${1:-local}"

run_np() {
    # $1 = np (1 or 4)
    # $2 = output-subdir  (under test/)
    local np="$1"
    local subdir="$2"
    rm -rf "$subdir"
    mkdir -p "$subdir"
    rm -rf output && mkdir output

    if [[ "$MODE" == "--klone" ]]; then
        NP="$np" "$PROJECT_DIR/scripts/klone-run" mpi_advection_test \
            > "$subdir/run.log" 2>&1
    else
        mpirun -np "$np" ./mpi_advection_test > "$subdir/run.log" 2>&1
    fi
    mv output/final_q_rank_*.bin "$subdir/"
}

echo "=== build mpi_advection_test ==="
if [[ "$MODE" == "--klone" ]]; then
    # klone-run takes care of Apptainer + host MPI linkage itself and
    # rebuilds inside each srun allocation so PTX matches the GPU.
    :
else
    make -s mpi_advection_test
fi

echo "=== run np=1 ==="
run_np 1 "test/dumps_np1"

echo "=== run np=4 ==="
run_np 4 "test/dumps_np4"

echo "=== diff ==="
if python3 "$SCRIPT_DIR/diff_mpi_dumps.py" \
    --a test/dumps_np1 \
    --b test/dumps_np4 \
    --tol-abs 1e-5 \
    --tol-rel 1e-4
then
    echo "=== MPI correctness test PASSED ==="
    exit 0
else
    echo "=== MPI correctness test FAILED ==="
    echo "    inspect test/dumps_np1/run.log and test/dumps_np4/run.log"
    exit 1
fi
