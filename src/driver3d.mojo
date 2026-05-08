# ======================================================================
# Driver3D -- shared setup for 3D Solver-pipeline example drivers
# ======================================================================
#
# Absorbs the boilerplate every Solver-pipeline 3D driver duplicated
# verbatim:
#
#   mpi.init()
#   var rank = mpi.world_rank()
#   var size = mpi.world_size()
#   if rank == 0: print("<problem>:", size, "rank(s)")
#   var nvtx = NvtxContext()
#   var refs = build_reference_operators(nvtx)
#   var ctx = DeviceContext()
#   var mesh = Mesh(ctx, build_partition(rank, size, NX, NY, NZ), LX, LY, LZ, bcs)
#   var halo = HaloExchange(ctx, mesh.part, PhysT.NUM_COMPONENTS, mesh.d_perm.unsafe_ptr(), bcs)
#   var solver = Solver[PhysT](ctx^, mesh^, halo^, physics^, refs.D_ref^, refs.Lift_ref^, refs.node_weights^)
#
# Replaces it with:
#
#   var d = Driver3D[Euler](
#       problem_name="euler_sod: GPU DG Euler, P2 tet, HLLEC flux",
#       nx=NX, ny=NY, nz=NZ, lx=LX, ly=LY, lz=LZ,
#       bcs=bcs, physics=physics^,
#   )
#
# After construction the driver accesses `d.rank`, `d.size`, `d.nvtx`,
# and `d.solver` -- matches the field names users were already using
# locally.  Per-driver banner extras (BC summary, physics-specific
# header lines) print after construction; the harness only emits the
# universal "<problem>: ranks <size>" + mesh-shape line so problem-
# specific framing stays user-controlled.
#
# `mpi.finalize()` is left to the driver: harness shutdown is one line
# and tying it to a destructor would interact badly with how Solver's
# device buffers expect to free in main()'s scope.
# ======================================================================

from std.gpu.host import DeviceContext

from src import mpi
from src.partition import build_partition
from src.reference import build_reference_operators
from src.mesh import Mesh
from src.boundary import BoundaryConditions
from src.halo_exchange import HaloExchange
from src.solver import Solver, Physics
from src.nvtx import NvtxContext


struct Driver3D[PhysT: Physics, P: Int = 2](Movable):
    var rank: Int
    var size: Int
    var nvtx: NvtxContext
    var solver: Solver[Self.PhysT, Self.P]

    def __init__(
        out self,
        problem_name: String,
        nx: Int,
        ny: Int,
        nz: Int,
        lx: Float64,
        ly: Float64,
        lz: Float64,
        bcs: BoundaryConditions,
        var physics: Self.PhysT,
    ) raises:
        mpi.init()
        self.rank = mpi.world_rank()
        self.size = mpi.world_size()
        if self.rank == 0:
            print(problem_name, " ranks:", self.size)
            print("  global mesh:", nx, "x", ny, "x", nz, " cells ->", nx * ny * nz * 6, "tets")
        self.nvtx = NvtxContext()
        var refs = build_reference_operators(self.nvtx)
        var ctx = DeviceContext()
        var mesh = Mesh[Self.P](ctx, build_partition(self.rank, self.size, nx, ny, nz), lx, ly, lz, bcs)
        var halo = HaloExchange(ctx, mesh.part, Self.PhysT.NUM_COMPONENTS, mesh.d_perm.unsafe_ptr(), bcs)
        self.solver = Solver[Self.PhysT, Self.P](ctx^, mesh^, halo^, physics^, refs.D_ref^, refs.Lift_ref^, refs.node_weights^)

    def is_single_rank(self) -> Bool:
        """True if running with one MPI rank.  Convenience for gating
        end-of-run snapshots and post-checks that only make sense
        without cross-rank halo data."""
        return self.solver.mesh.part.px * self.solver.mesh.part.py * self.solver.mesh.part.pz == 1
