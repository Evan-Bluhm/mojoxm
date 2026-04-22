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


@fieldwise_init
struct Advection(Physics, ImplicitlyCopyable):
    # Number of conserved components.
    comptime NUM_COMPONENTS = 1

    # Constant advection velocity (world frame).
    var vx: Float32
    var vy: Float32
    var vz: Float32

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
