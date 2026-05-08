# ======================================================================
# Maxwell physics -- vacuum electrodynamics
# ======================================================================
#
# Solves the 6-component hyperbolic system
#
#   dE/dt - c^2 curl(B) = 0
#   dB/dt + curl(E)     = 0
#
# in conservation form dU/dt + div(F) = 0 with U = [E; B], where
#
#   (F . n)_E = c^2 * (B x n)
#   (F . n)_B =        (n x E)
#
# Eigenvalues of the Jacobian in any direction: 0, 0, +/- c (each
# doubled), so Rusanov's maximum-speed dissipation parameter is
# `alpha = c`.  The two zero eigenvalues correspond to the divergence
# constraints div(E) = 0 (charge-free) and div(B) = 0 -- they're not
# actively cleaned in this module; see the `maxwell-glm` follow-up on
# the roadmap for a 9-component GLM extension.
#
# Boundary conditions:
#   BC_WALL    = perfect electric conductor.  Ghost state flips
#                tangential E (so the E_tangential average is zero at
#                the wall) and flips normal B (so B.n is zero).
#   BC_OUTFLOW = zero-gradient.  With Rusanov's symmetric dissipation
#                this is a clean one-way boundary for normally-incident
#                waves; obliquely-incident waves leak a fraction of
#                their energy back, as always with a simple
#                characteristic BC.
#
# Source term: uniform current J and optional magnetic current M,
# stored as six Float32 fields (Jx, Jy, Jz, Mx, My, Mz).  Default
# zero; a driver that wants a driven antenna can set nonzero values.
# The source is applied as
#
#   dE/dt += -J / eps0 = -c^2 * J   (SI eps0 = 1 / c^2 in these units)
#   dB/dt += -M                   (if M is the magnetic current)
#
# For our ``eps0 = mu0 = 1, c free'' unit system the coefficient on J
# collapses to c^2.
# ======================================================================

from src.solver import Physics
from src.boundary import BC_WALL, BC_OUTFLOW, BC_INFLOW


struct Maxwell(ImplicitlyCopyable, Physics):
    comptime NUM_COMPONENTS = 6

    # Speed of light.  1 in natural units; drivers that want dimensional
    # SI-style scaling set c = 2.998e8 (and scale charges / fields
    # accordingly, but that loses Float32 precision -- keeping c near 1
    # is a good idea).
    var c: Float32

    # Uniform current source, static in space.  Spatially-varying J
    # would require passing a callback through DevicePassable, which
    # the trait doesn't support; for patterned antennas, subclass this
    # struct and override source_term.  Default zero.
    var Jx: Float32
    var Jy: Float32
    var Jz: Float32
    var Mx: Float32
    var My: Float32
    var Mz: Float32

    # BC_INFLOW ghost state (Ex, Ey, Ez, Bx, By, Bz).  Defaults zero.
    var inflow_Ex: Float32
    var inflow_Ey: Float32
    var inflow_Ez: Float32
    var inflow_Bx: Float32
    var inflow_By: Float32
    var inflow_Bz: Float32

    def __init__(
        out self,
        c: Float32,
        Jx: Float32,
        Jy: Float32,
        Jz: Float32,
        Mx: Float32,
        My: Float32,
        Mz: Float32,
        inflow_Ex: Float32 = Float32(0.0),
        inflow_Ey: Float32 = Float32(0.0),
        inflow_Ez: Float32 = Float32(0.0),
        inflow_Bx: Float32 = Float32(0.0),
        inflow_By: Float32 = Float32(0.0),
        inflow_Bz: Float32 = Float32(0.0),
    ):
        self.c = c
        self.Jx = Jx
        self.Jy = Jy
        self.Jz = Jz
        self.Mx = Mx
        self.My = My
        self.Mz = Mz
        self.inflow_Ex = inflow_Ex
        self.inflow_Ey = inflow_Ey
        self.inflow_Ez = inflow_Ez
        self.inflow_Bx = inflow_Bx
        self.inflow_By = inflow_By
        self.inflow_Bz = inflow_Bz

    # --- DevicePassable plumbing (see std.gpu.host.device_context) ---
    comptime device_type = Self

    def _to_device_type[origin: MutOrigin](self, target: UnsafePointer[NoneType, origin]):
        target.bitcast[Self]()[] = self

    @staticmethod
    def get_type_name() -> String:
        return "Maxwell"

    # Internal (volume) flux at a single node.  Layout
    # flux[d * NC + c] = F_d_c(q), matching the same `flux[d * NC + c]`
    # addressing Advection / Euler use.
    #
    # Component order: q[0..2] = E, q[3..5] = B.
    # F^x_E = c^2 * (B x x^) = c^2 * (0, Bz, -By)
    # F^x_B = x^ x E          = (0, -Ez, Ey)
    # F^y_E = c^2 * (B x y^) = c^2 * (-Bz, 0, Bx)
    # F^y_B = y^ x E          = (Ez, 0, -Ex)
    # F^z_E = c^2 * (B x z^) = c^2 * (By, -Bx, 0)
    # F^z_B = z^ x E          = (-Ey, Ex, 0)
    def internal_flux(
        self,
        q: UnsafePointer[Float32, MutAnyOrigin],
        flux: UnsafePointer[Float32, MutAnyOrigin],
    ) -> Float32:
        var Ex = q[0]
        var Ey = q[1]
        var Ez = q[2]
        var Bx = q[3]
        var By = q[4]
        var Bz = q[5]
        var c2 = self.c * self.c
        # F^x
        flux[0 * 6 + 0] = Float32(0.0)
        flux[0 * 6 + 1] = c2 * Bz
        flux[0 * 6 + 2] = -c2 * By
        flux[0 * 6 + 3] = Float32(0.0)
        flux[0 * 6 + 4] = -Ez
        flux[0 * 6 + 5] = Ey
        # F^y
        flux[1 * 6 + 0] = -c2 * Bz
        flux[1 * 6 + 1] = Float32(0.0)
        flux[1 * 6 + 2] = c2 * Bx
        flux[1 * 6 + 3] = Ez
        flux[1 * 6 + 4] = Float32(0.0)
        flux[1 * 6 + 5] = -Ex
        # F^z
        flux[2 * 6 + 0] = c2 * By
        flux[2 * 6 + 1] = -c2 * Bx
        flux[2 * 6 + 2] = Float32(0.0)
        flux[2 * 6 + 3] = -Ey
        flux[2 * 6 + 4] = Ex
        flux[2 * 6 + 5] = Float32(0.0)
        return self.c

    # Rusanov flux:  F* = 0.5 * (F_n(q_l) + F_n(q_r)) - 0.5 * c * (q_r - q_l).
    # F_n(q) . E components = c^2 * (B x n)
    # F_n(q) . B components = n x E
    def numerical_flux(
        self,
        q_l: UnsafePointer[Float32, MutAnyOrigin],
        q_r: UnsafePointer[Float32, MutAnyOrigin],
        nx: Float32,
        ny: Float32,
        nz: Float32,
        flux: UnsafePointer[Float32, MutAnyOrigin],
    ) -> Float32:
        var Elx = q_l[0]
        var Ely = q_l[1]
        var Elz = q_l[2]
        var Blx = q_l[3]
        var Bly = q_l[4]
        var Blz = q_l[5]
        var Erx = q_r[0]
        var Ery = q_r[1]
        var Erz = q_r[2]
        var Brx = q_r[3]
        var Bry = q_r[4]
        var Brz = q_r[5]
        var c2 = self.c * self.c

        # (F . n)_E = c^2 * (B x n);  (B x n)_x = By nz - Bz ny
        var FEl_x = c2 * (Bly * nz - Blz * ny)
        var FEl_y = c2 * (Blz * nx - Blx * nz)
        var FEl_z = c2 * (Blx * ny - Bly * nx)
        var FEr_x = c2 * (Bry * nz - Brz * ny)
        var FEr_y = c2 * (Brz * nx - Brx * nz)
        var FEr_z = c2 * (Brx * ny - Bry * nx)
        # (F . n)_B = n x E;  (n x E)_x = ny Ez - nz Ey
        var FBl_x = ny * Elz - nz * Ely
        var FBl_y = nz * Elx - nx * Elz
        var FBl_z = nx * Ely - ny * Elx
        var FBr_x = ny * Erz - nz * Ery
        var FBr_y = nz * Erx - nx * Erz
        var FBr_z = nx * Ery - ny * Erx

        var half = Float32(0.5)
        var alpha = self.c
        flux[0] = half * (FEl_x + FEr_x) - half * alpha * (Erx - Elx)
        flux[1] = half * (FEl_y + FEr_y) - half * alpha * (Ery - Ely)
        flux[2] = half * (FEl_z + FEr_z) - half * alpha * (Erz - Elz)
        flux[3] = half * (FBl_x + FBr_x) - half * alpha * (Brx - Blx)
        flux[4] = half * (FBl_y + FBr_y) - half * alpha * (Bry - Bly)
        flux[5] = half * (FBl_z + FBr_z) - half * alpha * (Brz - Blz)
        return self.c

    # Boundary-face flux via a synthetic ghost state.  BC_WALL is the
    # PEC reflection; BC_OUTFLOW is zero-gradient, which Rusanov reads
    # as a perfectly transmitting normal-incidence boundary (and a
    # lossy oblique-incidence one, same as the Euler outflow BC).
    def boundary_flux(
        self,
        q_int: UnsafePointer[Float32, MutAnyOrigin],
        bc_type: Int32,
        nx: Float32,
        ny: Float32,
        nz: Float32,
        flux: UnsafePointer[Float32, MutAnyOrigin],
    ) -> Float32:
        var q_ghost = InlineArray[Float32, 6](fill=0.0)
        if bc_type == BC_WALL:
            # Flip tangential E, flip normal B.
            # E_tangential -> E - (E.n) n   is the tangential part.
            # E_ghost = 2 (E.n) n - E
            # B_ghost = B - 2 (B.n) n
            var En = q_int[0] * nx + q_int[1] * ny + q_int[2] * nz
            var Bn = q_int[3] * nx + q_int[4] * ny + q_int[5] * nz
            q_ghost[0] = Float32(2.0) * En * nx - q_int[0]
            q_ghost[1] = Float32(2.0) * En * ny - q_int[1]
            q_ghost[2] = Float32(2.0) * En * nz - q_int[2]
            q_ghost[3] = q_int[3] - Float32(2.0) * Bn * nx
            q_ghost[4] = q_int[4] - Float32(2.0) * Bn * ny
            q_ghost[5] = q_int[5] - Float32(2.0) * Bn * nz
        elif bc_type == BC_INFLOW:
            q_ghost[0] = self.inflow_Ex
            q_ghost[1] = self.inflow_Ey
            q_ghost[2] = self.inflow_Ez
            q_ghost[3] = self.inflow_Bx
            q_ghost[4] = self.inflow_By
            q_ghost[5] = self.inflow_Bz
        else:
            # BC_OUTFLOW / default: zero-gradient extrapolation.
            for k in range(6):
                q_ghost[k] = q_int[k]
        var q_ghost_p = rebind[UnsafePointer[Float32, MutAnyOrigin]](q_ghost.unsafe_ptr())
        return self.numerical_flux(q_int, q_ghost_p, nx, ny, nz, flux)

    # Uniform current source.  In SI units with eps0 = 1/c^2, mu0 = 1:
    #   dE/dt += -J / eps0 = -c^2 * J
    #   dB/dt += -M                   (magnetic current -- usually zero)
    def source_term(
        self,
        q: UnsafePointer[Float32, MutAnyOrigin],
        x: Float32,
        y: Float32,
        z: Float32,
        source_out: UnsafePointer[Float32, MutAnyOrigin],
    ):
        var c2 = self.c * self.c
        source_out[0] = -c2 * self.Jx
        source_out[1] = -c2 * self.Jy
        source_out[2] = -c2 * self.Jz
        source_out[3] = -self.Mx
        source_out[4] = -self.My
        source_out[5] = -self.Mz

    def limit_state(
        self,
        q: UnsafePointer[Float32, MutAnyOrigin],
    ):
        pass
