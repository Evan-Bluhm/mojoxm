#!/usr/bin/env bash
# ======================================================================
# test_mpi_correctness.sh -- verify np=1 and np=4 produce the same
# final q field (within FP tolerance).
#
# Runs examples/mpi_advection_test.mojo for 50 steps at np=1 and np=4,
# dumping each rank's owned q to output/final_q_rank_<rank>.bin.  Moves
# the dumps into per-config directories and invokes
# scripts/diff_mpi_dumps.py to compare.
#
# Usage:
#     scripts/test_mpi_correctness.sh                    # local WSL2 path
#     scripts/test_mpi_correctness.sh --klone            # cluster path
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
    # $2 = output-subdir
    local np="$1"
    local subdir="$2"
    rm -rf "$subdir"
    mkdir -p "$subdir"
    rm -rf output && mkdir output

    if [[ "$MODE" == "--klone" ]]; then
        NP="$np" "$SCRIPT_DIR/klone-run" mpi_advection_test > "$subdir/run.log" 2>&1
    else
        if [[ "$np" == "1" ]]; then
            mpirun -np 1 ./mpi_advection_test > "$subdir/run.log" 2>&1
        else
            mpirun -np "$np" ./mpi_advection_test > "$subdir/run.log" 2>&1
        fi
    fi
    mv output/final_q_rank_*.bin "$subdir/"
}

echo "=== build mpi_advection_test ==="
if [[ "$MODE" == "--klone" ]]; then
    # klone-run takes care of Apptainer + host MPI linkage itself.
    # We still need the binary present before launching, but klone-run
    # builds it as part of the job.  Nothing to do here.
    :
else
    make mpi_advection_test -s
fi

echo "=== run np=1 ==="
run_np 1 "test_dumps_np1"

echo "=== run np=4 ==="
run_np 4 "test_dumps_np4"

echo "=== diff ==="
if python3 "$SCRIPT_DIR/diff_mpi_dumps.py" \
    --a test_dumps_np1 \
    --b test_dumps_np4 \
    --tol-abs 1e-5 \
    --tol-rel 1e-4
then
    echo "=== MPI correctness test PASSED ==="
    exit 0
else
    echo "=== MPI correctness test FAILED ==="
    echo "    inspect test_dumps_np1/run.log and test_dumps_np4/run.log"
    exit 1
fi
