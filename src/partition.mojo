# ======================================================================
# Cartesian patch decomposition for the Kuhn-tet cube grid
# ======================================================================
#
# Splits the global (Nx, Ny, Nz) cube grid into PX x PY x PZ patches
# where PX * PY * PZ == NPROCS, choosing the factorisation that
# minimises total inter-patch surface area (i.e. ghost-exchange volume
# per RK stage).
#
# Each MPI rank owns a contiguous box of cubes
#   [cx0, cx1) x [cy0, cy1) x [cz0, cz1)
# and a 1-cube-thick ghost ring around it (only on faces that touch
# another patch -- with triply periodic BCs, every face touches a
# neighbour, so the ghost ring is full).
#
# Patch-coordinates wrap periodically: the rank at (px, py, pz) has
# +x neighbour ((px+1) mod PX, py, pz), etc.  The 6 face-neighbours
# are the only ones we exchange data with -- DG halo is a single
# element layer, and Kuhn cubes only share triangular faces with their
# +/-x, +/-y, +/-z cube neighbours (the diagonal Kuhn faces are inside
# a cube, never crossing a cube boundary).
# ======================================================================

@fieldwise_init
struct Partition(Copyable, Movable):
    """Per-rank view of the global cube grid decomposition."""

    # Global grid dimensions (cubes).
    var global_nx: Int
    var global_ny: Int
    var global_nz: Int

    # Process-grid dimensions.
    var px: Int
    var py: Int
    var pz: Int

    # This rank's coordinates within the process grid.
    var rx: Int
    var ry: Int
    var rz: Int

    # Owned cube range in the global grid (half-open intervals).
    var cx0: Int
    var cx1: Int
    var cy0: Int
    var cy1: Int
    var cz0: Int
    var cz1: Int

    # Owned cube counts (= cx1 - cx0, etc.).
    var nx: Int
    var ny: Int
    var nz: Int

    # MPI ranks of the 6 face neighbours, ordered as
    # [-x, +x, -y, +y, -z, +z].  Always defined under triply periodic
    # BCs.  In single-rank mode all six entries equal `rank` (so the
    # rank is its own neighbour) and the ghost layer is unused.
    var neighbour_minus_x: Int
    var neighbour_plus_x:  Int
    var neighbour_minus_y: Int
    var neighbour_plus_y:  Int
    var neighbour_minus_z: Int
    var neighbour_plus_z:  Int

    def num_owned_cubes(self) -> Int:
        return self.nx * self.ny * self.nz

    def neighbour(self, axis: Int, sign: Int) -> Int:
        """Return rank of neighbour along (axis, sign) where axis is
        0/1/2 for x/y/z and sign is -1 or +1."""
        if axis == 0 and sign < 0: return self.neighbour_minus_x
        if axis == 0 and sign > 0: return self.neighbour_plus_x
        if axis == 1 and sign < 0: return self.neighbour_minus_y
        if axis == 1 and sign > 0: return self.neighbour_plus_y
        if axis == 2 and sign < 0: return self.neighbour_minus_z
        return self.neighbour_plus_z


# ----------------------------------------------------------------------
# Process-grid factorisation
# ----------------------------------------------------------------------
# Pick (PX, PY, PZ) such that:
#   PX * PY * PZ == nprocs
#   each Pd evenly divides the corresponding Nd
#   minimised "surface" cost: 2 (Lx Ly + Ly Lz + Lz Lx) where
#       Lx = Nx / PX,  Ly = Ny / PY,  Lz = Nz / PZ
# Tiebreaker: shape closest to (Nx, Ny, Nz) ratio.
# ----------------------------------------------------------------------

@fieldwise_init
struct ProcGrid(Copyable, Movable):
    var px: Int
    var py: Int
    var pz: Int

def choose_proc_grid(
    nprocs: Int, nx: Int, ny: Int, nz: Int,
) raises -> ProcGrid:
    var best_px: Int = 1
    var best_py: Int = 1
    var best_pz: Int = nprocs
    var best_cost: Float64 = 1.0e18
    var found = False

    for px in range(1, nprocs + 1):
        if nprocs % px != 0:
            continue
        if nx % px != 0:
            continue
        var rest = nprocs // px
        for py in range(1, rest + 1):
            if rest % py != 0:
                continue
            if ny % py != 0:
                continue
            var pz = rest // py
            if nz % pz != 0:
                continue
            var lx = Float64(nx) / Float64(px)
            var ly = Float64(ny) / Float64(py)
            var lz = Float64(nz) / Float64(pz)
            # Surface area of one patch (one factor of 2 absorbed; we're
            # minimising the relative cost so absolute scale doesn't
            # matter).
            var surf = lx * ly + ly * lz + lz * lx
            if surf < best_cost:
                best_cost = surf
                best_px = px
                best_py = py
                best_pz = pz
                found = True

    if not found:
        raise Error(
            "no axis-aligned cube-grid factorisation of nprocs="
            + String(nprocs) + " evenly divides ("
            + String(nx) + ", " + String(ny) + ", " + String(nz)
            + ").  Adjust the mesh size to be divisible by your factor "
            + "structure (e.g. mesh sides multiples of small primes).")
    return ProcGrid(best_px, best_py, best_pz)


# ----------------------------------------------------------------------
# Build a Partition for a given rank
# ----------------------------------------------------------------------

def build_partition(
    rank: Int, nprocs: Int,
    nx: Int, ny: Int, nz: Int,
) raises -> Partition:
    var grid = choose_proc_grid(nprocs, nx, ny, nz)

    # rank -> (rx, ry, rz) using row-major order (z fastest, x slowest).
    # Doesn't matter for correctness, but keep it consistent.
    var rx = rank // (grid.py * grid.pz)
    var ry = (rank // grid.pz) % grid.py
    var rz = rank % grid.pz

    var lx = nx // grid.px
    var ly = ny // grid.py
    var lz = nz // grid.pz

    var cx0 = rx * lx
    var cy0 = ry * ly
    var cz0 = rz * lz

    fn _rank_of(rrx: Int, rry: Int, rrz: Int) capturing -> Int:
        return ((rrx * grid.py) + rry) * grid.pz + rrz

    fn _wrap(v: Int, mod: Int) capturing -> Int:
        # Periodic wrap (for triply periodic BCs).  v is in [-1, mod].
        if v < 0: return v + mod
        if v >= mod: return v - mod
        return v

    return Partition(
        global_nx=nx, global_ny=ny, global_nz=nz,
        px=grid.px, py=grid.py, pz=grid.pz,
        rx=rx, ry=ry, rz=rz,
        cx0=cx0, cx1=cx0 + lx,
        cy0=cy0, cy1=cy0 + ly,
        cz0=cz0, cz1=cz0 + lz,
        nx=lx, ny=ly, nz=lz,
        neighbour_minus_x=_rank_of(_wrap(rx - 1, grid.px), ry, rz),
        neighbour_plus_x =_rank_of(_wrap(rx + 1, grid.px), ry, rz),
        neighbour_minus_y=_rank_of(rx, _wrap(ry - 1, grid.py), rz),
        neighbour_plus_y =_rank_of(rx, _wrap(ry + 1, grid.py), rz),
        neighbour_minus_z=_rank_of(rx, ry, _wrap(rz - 1, grid.pz)),
        neighbour_plus_z =_rank_of(rx, ry, _wrap(rz + 1, grid.pz)),
    )
