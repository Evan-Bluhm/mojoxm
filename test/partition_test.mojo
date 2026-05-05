# ======================================================================
# partition_test -- unit tests for `src.partition.build_partition`
# ======================================================================
#
# `build_partition` is the only entry point in src/partition.mojo and
# is invoked at the start of every multi-rank 3D driver to compute
# the (px, py, pz) factorisation, the per-rank cube range, and the 6
# face-neighbour ranks under triply-periodic BCs.  A regression in
# the factorisation cost-minimiser, the rank<->coord conversion, or
# the periodic-wrap math would silently break MPI halo exchange in
# all 3D drivers; the failure mode would be subtle correctness drift
# (data going to / coming from the wrong neighbour), not a crash.
#
# Three test cases:
#   (1) nprocs=1 single-rank: every neighbour must equal rank itself,
#       the entire global grid must be owned.
#   (2) nprocs=2 cubic mesh (4x4x4): expect a (1,1,2) split (smallest
#       surface among three tied 2-cell partitions, picked first by
#       the loop ordering); rank 0's +z / -z neighbours are rank 1.
#   (3) nprocs=8 cubic mesh (4x4x4): expect a balanced (2,2,2) split
#       with each rank owning 2x2x2 cubes; rank 0 (rx=0, ry=0, rz=0)
#       has 6 distinct face-neighbours under periodic wrap.
#
# No GPU / no MPI -- pure host integer math.
# ======================================================================

from src.partition import build_partition


def main() raises:
    print("partition_test: unit-test build_partition")

    # ---------- (1) single-rank ----------
    var p1 = build_partition(rank=0, nprocs=1, nx=4, ny=4, nz=4)
    if p1.px != 1 or p1.py != 1 or p1.pz != 1:
        raise Error("partition_test FAILED: nprocs=1 expected (1,1,1)")
    if p1.nx != 4 or p1.ny != 4 or p1.nz != 4:
        raise Error("partition_test FAILED: nprocs=1 owned counts wrong")
    if p1.cx0 != 0 or p1.cx1 != 4 or p1.cy0 != 0 or p1.cy1 != 4 \
            or p1.cz0 != 0 or p1.cz1 != 4:
        raise Error("partition_test FAILED: nprocs=1 owned range wrong")
    if (p1.neighbour_minus_x != 0 or p1.neighbour_plus_x != 0
            or p1.neighbour_minus_y != 0 or p1.neighbour_plus_y != 0
            or p1.neighbour_minus_z != 0 or p1.neighbour_plus_z != 0):
        raise Error("partition_test FAILED: nprocs=1 neighbours not self")
    if p1.num_owned_cubes() != 64:
        raise Error("partition_test FAILED: nprocs=1 num_owned_cubes != 64")

    # ---------- (2) nprocs=2, 4x4x4 mesh ----------
    # The first surface-equal split the loop visits is (px=1, py=1, pz=2)
    # so the cost-minimiser settles there.  Rank 0 is at (0,0,0), rank 1
    # at (0,0,1); each owns half along z.
    var p2_r0 = build_partition(rank=0, nprocs=2, nx=4, ny=4, nz=4)
    var p2_r1 = build_partition(rank=1, nprocs=2, nx=4, ny=4, nz=4)
    if p2_r0.px != 1 or p2_r0.py != 1 or p2_r0.pz != 2:
        raise Error(
            "partition_test FAILED: nprocs=2 4x4x4 expected (1,1,2), got ("
            + String(p2_r0.px) + "," + String(p2_r0.py) + ","
            + String(p2_r0.pz) + ")"
        )
    if p2_r0.cz0 != 0 or p2_r0.cz1 != 2:
        raise Error("partition_test FAILED: nprocs=2 r0 z-range wrong")
    if p2_r1.cz0 != 2 or p2_r1.cz1 != 4:
        raise Error("partition_test FAILED: nprocs=2 r1 z-range wrong")
    # Combined owned counts cover the global grid exactly once.
    if p2_r0.num_owned_cubes() + p2_r1.num_owned_cubes() != 64:
        raise Error("partition_test FAILED: nprocs=2 owned union != global")
    # +z and -z neighbours of rank 0 are rank 1 (periodic wrap with pz=2).
    if p2_r0.neighbour_plus_z != 1 or p2_r0.neighbour_minus_z != 1:
        raise Error(
            "partition_test FAILED: nprocs=2 r0 z-neighbours expected 1, got "
            + String(p2_r0.neighbour_plus_z) + "/"
            + String(p2_r0.neighbour_minus_z)
        )
    # x and y neighbours of rank 0 are itself (px=py=1).
    if (p2_r0.neighbour_minus_x != 0 or p2_r0.neighbour_plus_x != 0
            or p2_r0.neighbour_minus_y != 0 or p2_r0.neighbour_plus_y != 0):
        raise Error("partition_test FAILED: nprocs=2 r0 x/y neighbours not self")

    # ---------- (3) nprocs=8, 4x4x4 mesh: balanced (2,2,2) ----------
    var p3_r0 = build_partition(rank=0, nprocs=8, nx=4, ny=4, nz=4)
    if p3_r0.px != 2 or p3_r0.py != 2 or p3_r0.pz != 2:
        raise Error(
            "partition_test FAILED: nprocs=8 4x4x4 expected (2,2,2), got ("
            + String(p3_r0.px) + "," + String(p3_r0.py) + ","
            + String(p3_r0.pz) + ")"
        )
    if p3_r0.nx != 2 or p3_r0.ny != 2 or p3_r0.nz != 2:
        raise Error("partition_test FAILED: nprocs=8 r0 owned counts != 2,2,2")
    # Owned union over all 8 ranks covers the global grid exactly once.
    var total: Int = 0
    for r in range(8):
        var pp = build_partition(rank=r, nprocs=8, nx=4, ny=4, nz=4)
        total += pp.num_owned_cubes()
    if total != 64:
        raise Error(
            "partition_test FAILED: nprocs=8 union != 64, got " + String(total)
        )
    # Rank 0 (rx=0, ry=0, rz=0) periodic neighbours.  With (px,py,pz)=(2,2,2)
    # and rank-major formula `((rx*py)+ry)*pz+rz`:
    #   minus_x: (rx-1) wraps to 1 -> ((1*2)+0)*2+0 = 4
    #   plus_x:  (rx+1) wraps to 1 -> 4
    #   minus_y: (ry-1) wraps to 1 -> ((0*2)+1)*2+0 = 2
    #   plus_y:  (ry+1) wraps to 1 -> 2
    #   minus_z: (rz-1) wraps to 1 -> ((0*2)+0)*2+1 = 1
    #   plus_z:  (rz+1) wraps to 1 -> 1
    if p3_r0.neighbour_minus_x != 4 or p3_r0.neighbour_plus_x != 4:
        raise Error(
            "partition_test FAILED: nprocs=8 r0 x-neighbour expected 4, got "
            + String(p3_r0.neighbour_minus_x) + "/"
            + String(p3_r0.neighbour_plus_x)
        )
    if p3_r0.neighbour_minus_y != 2 or p3_r0.neighbour_plus_y != 2:
        raise Error(
            "partition_test FAILED: nprocs=8 r0 y-neighbour expected 2, got "
            + String(p3_r0.neighbour_minus_y) + "/"
            + String(p3_r0.neighbour_plus_y)
        )
    if p3_r0.neighbour_minus_z != 1 or p3_r0.neighbour_plus_z != 1:
        raise Error(
            "partition_test FAILED: nprocs=8 r0 z-neighbour expected 1, got "
            + String(p3_r0.neighbour_minus_z) + "/"
            + String(p3_r0.neighbour_plus_z)
        )

    # ---------- (4) error path: indivisible factorisation ----------
    var raised: Bool = False
    try:
        var _bad = build_partition(rank=0, nprocs=3, nx=4, ny=4, nz=4)
    except:
        raised = True
    if not raised:
        raise Error(
            "partition_test FAILED: nprocs=3 / 4x4x4 should raise "
            "(no integer factorisation); did not raise"
        )

    print("=== partition_test PASSED ===")
