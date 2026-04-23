# ======================================================================
# Boundary conditions for non-periodic faces
# ======================================================================
#
# Each face in the mesh has an associated `bc_type` (stored in
# `LocalMesh.d_face_bc_type`, one Int32 per face).  A value of
# `BC_INTERIOR` means this face is a regular two-sided interior face
# and the solver calls `physics.numerical_flux` as usual.  Any other
# value identifies a boundary-condition kind that the solver dispatches
# to `physics.boundary_flux(q_interior, bc_type, nx, ny, nz, flux)`
# instead, with the outward unit normal pointing from the interior
# element into the (non-existent) ghost element.
#
# The BC catalogue is deliberately tiny and physics-agnostic.  Each
# physics module decides what each BC kind means -- e.g. `BC_WALL` is
# a slip wall for Euler (reflect normal momentum) and a zero-Dirichlet
# wall for Advection.  Adding a new BC is a matter of bumping this
# enum and handling it in the physics modules that care.
# ======================================================================


# BC kinds.  Int32 so they round-trip through DeviceBuffer[int32]
# without widening.
comptime BC_INTERIOR:  Int32 = 0
comptime BC_WALL:      Int32 = 1   # reflecting / slip wall
comptime BC_OUTFLOW:   Int32 = 2   # zero-gradient transmissive outflow


# ----------------------------------------------------------------------
# Per-domain BC configuration.
#
# Canonical 6-face ordering matches the halo-ring convention used by
# `classify_owned_kernel` in src/mesh.mojo: [-x, +x, -y, +y, -z, +z].
# `BC_INTERIOR` on a given face means "periodic" -- that's how the
# mesh was built before this module existed, and it stays the default.
# ----------------------------------------------------------------------

@fieldwise_init
struct BoundaryConditions(ImplicitlyCopyable, Movable):
    var bc_x_lo: Int32
    var bc_x_hi: Int32
    var bc_y_lo: Int32
    var bc_y_hi: Int32
    var bc_z_lo: Int32
    var bc_z_hi: Int32

    @staticmethod
    def periodic() -> Self:
        return Self(
            BC_INTERIOR, BC_INTERIOR,
            BC_INTERIOR, BC_INTERIOR,
            BC_INTERIOR, BC_INTERIOR,
        )

    def all_periodic(self) -> Bool:
        return (
            self.bc_x_lo == BC_INTERIOR and self.bc_x_hi == BC_INTERIOR
            and self.bc_y_lo == BC_INTERIOR and self.bc_y_hi == BC_INTERIOR
            and self.bc_z_lo == BC_INTERIOR and self.bc_z_hi == BC_INTERIOR
        )
