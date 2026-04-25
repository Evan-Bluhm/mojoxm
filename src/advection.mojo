# ======================================================================
# Advection physics  --  dq/dt + v.grad(q) = 0
# ======================================================================
#
# Conserved variables: q = [rho] (single scalar).
# Flux: F_d = v_d * q.
# Numerical flux: pure upwind on the normal component of the velocity.
#
# This module provides the `Advection` struct that the generic
# `Solver[PhysT]` picks up at compile time.  The per-element kernel
# calls `physics.internal_flux(...)` and `physics.numerical_flux(...)`
# directly; both get inlined at PhysT specialization time, so there's
# no runtime dispatch overhead.
# ======================================================================

from src.solver import Physics
from src.boundary import BC_WALL, BC_OUTFLOW, BC_INFLOW


struct Advection(Physics, ImplicitlyCopyable):
    # Number of conserved components.
    comptime NUM_COMPONENTS = 1

    # Constant advection velocity (world frame).
    var vx: Float32
    var vy: Float32
    var vz: Float32

    # Dirichlet inflow state used when a boundary face has
    # bc_type == BC_INFLOW.  Defaults to 0; existing drivers that only
    # use BC_OUTFLOW / BC_WALL are unaffected.
    var inflow_q: Float32

    def __init__(
        out self,
        vx: Float32, vy: Float32, vz: Float32,
        inflow_q: Float32 = Float32(0.0),
    ):
        self.vx = vx
        self.vy = vy
        self.vz = vz
        self.inflow_q = inflow_q

    # --- DevicePassable plumbing (see std.gpu.host.device_context) ---
    comptime device_type = Self

    def _to_device_type[origin: MutOrigin](
        self, target: UnsafePointer[NoneType, origin]
    ):
        target.bitcast[Self]()[] = self

    @staticmethod
    def get_type_name() -> String:
        return "Advection"

    # Internal (volume) flux at a single node.
    #
    #   q    : pointer to NUM_COMPONENTS values (the nodal state).
    #   flux : pointer to NUM_COMPONENTS * 3 values, output flux tensor.
    #          Layout is column-major in (dimension, component):
    #              flux[d * NC + c] = F^d_c(q)
    #
    # Returns a CFL-based dt estimate from this cell; the caller may
    # disregard it for advection where the face-flux constraint is
    # already tighter.
    def internal_flux(
        self,
        q:    UnsafePointer[Float32, MutAnyOrigin],
        flux: UnsafePointer[Float32, MutAnyOrigin],
    ) -> Float32:
        var q0 = q[0]
        flux[0 * 1 + 0] = self.vx * q0   # F^x_0
        flux[1 * 1 + 0] = self.vy * q0   # F^y_0
        flux[2 * 1 + 0] = self.vz * q0   # F^z_0
        # No volume-side CFL constraint for linear advection.
        return Float32(1.0e30)

    # Numerical (face) flux.  The face normal (nx, ny, nz) points from
    # side 0 to side 1.  For a scalar upwind flux we don't need the
    # tangent / binormal directions; Euler overrides that to do a full
    # rotation-to-local-frame.
    #
    # Returns |v.n|, the signal speed used by the caller for CFL.
    def numerical_flux(
        self,
        q_l:  UnsafePointer[Float32, MutAnyOrigin],
        q_r:  UnsafePointer[Float32, MutAnyOrigin],
        nx: Float32, ny: Float32, nz: Float32,
        flux: UnsafePointer[Float32, MutAnyOrigin],
    ) -> Float32:
        var nc = self.vx * nx + self.vy * ny + self.vz * nz
        var absnc = nc if nc >= Float32(0.0) else -nc
        flux[0] = Float32(0.5) * (
            (nc + absnc) * q_l[0] + (nc - absnc) * q_r[0]
        )
        return absnc

    # Boundary flux.  Per-kind ghost state for a scalar advection BC:
    #   BC_WALL    -> q_ghost = 0 (perfect absorber; wall can't inject).
    #   BC_INFLOW  -> q_ghost = self.inflow_q (user-set Dirichlet state).
    #   BC_OUTFLOW -> use the upwind scalar:
    #     v.n >= 0  (wave leaving the domain): flux = vn * q_int
    #     v.n <  0  (would-be inflow direction): flux = 0
    #   The earlier "q_ghost = q_int unconditionally" handling was
    #   numerically unstable when v.n < 0 along part of the boundary
    #   (it pulled spurious flux into the domain proportional to
    #   q_int).  See README Limitations for the bug discovery.
    def boundary_flux(
        self,
        q_int: UnsafePointer[Float32, MutAnyOrigin],
        bc_type: Int32,
        nx: Float32, ny: Float32, nz: Float32,
        flux: UnsafePointer[Float32, MutAnyOrigin],
    ) -> Float32:
        var nc = self.vx * nx + self.vy * ny + self.vz * nz
        var absnc = nc if nc >= Float32(0.0) else -nc
        var q_ghost: Float32
        if bc_type == BC_WALL:
            q_ghost = Float32(0.0)
        elif bc_type == BC_INFLOW:
            q_ghost = self.inflow_q
        else:
            # BC_OUTFLOW (and default): zero-Dirichlet on the inflowing
            # half so the flux upwinds purely from the interior when
            # vn >= 0 and is exactly zero when vn < 0.
            q_ghost = Float32(0.0)
        flux[0] = Float32(0.5) * (
            (nc + absnc) * q_int[0] + (nc - absnc) * q_ghost
        )
        return absnc

    # Pure hyperbolic conservation law -- no source.
    def source_term(
        self,
        q: UnsafePointer[Float32, MutAnyOrigin],
        x: Float32, y: Float32, z: Float32,
        source_out: UnsafePointer[Float32, MutAnyOrigin],
    ):
        source_out[0] = Float32(0.0)

    def limit_state(
        self,
        q: UnsafePointer[Float32, MutAnyOrigin],
    ):
        pass
