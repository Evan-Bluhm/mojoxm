# ======================================================================
# Ideal MHD physics with Dedner GLM divergence cleaning
# ======================================================================
#
# 9-component conservative form:
#
#   q = [rho, rho u, rho v, rho w, E, Bx, By, Bz, psi]
#
# The governing equations are single-fluid ideal MHD
#
#   drho/dt + div(rho u)                                   = 0
#   d(rho u)/dt + div(rho u u + p_tot I - B B)             = 0
#   dE/dt       + div((E + p_tot) u - B (u . B))           = 0
#   dB/dt       + div(u B - B u + psi I)                   = 0
#   dpsi/dt     + c_h^2 div(B)                             = -alpha_d * psi
#
# with
#
#   p_gas = (gamma - 1) * (E - 0.5 rho |u|^2 - 0.5 |B|^2)
#   p_tot = p_gas + 0.5 |B|^2
#
# The psi equation is the Dedner-Kemm-Kroner-Munz-Schnitzer-Wesenberg
# generalised-Lagrange-multiplier divergence cleaner: magnetic
# monopoles get transported out of the domain at hyperbolic speed c_h
# and damped at rate alpha_d.  With (c_h, alpha_d) = (0, 0) GLM is off
# and the scheme degenerates to ideal MHD without cleaning.
#
# Numerical flux: Rusanov with alpha = max(|u_n| + c_f, c_h) where c_f
# is the direction-specific fast magnetosonic speed at the face.
#
# Boundary conditions:
#   BC_WALL    = perfectly conducting wall (reflect u_n AND B_n)
#   BC_OUTFLOW = zero-gradient extrapolation
# ======================================================================

from src.solver import Physics
from src.boundary import BC_WALL, BC_OUTFLOW, BC_INFLOW
from std.math import sqrt


# ----------------------------------------------------------------------
# Floored primitive helpers (density and thermal pressure)
# ----------------------------------------------------------------------

def mhd_rho_floored(
    q: UnsafePointer[Float32, MutAnyOrigin], rho_min: Float32,
) -> Float32:
    var r = q[0]
    return r if r > rho_min else rho_min


def mhd_gas_pressure(
    q: UnsafePointer[Float32, MutAnyOrigin],
    gamma: Float32, rho_min: Float32, press_min: Float32,
) -> Float32:
    var rho = mhd_rho_floored(q, rho_min)
    var mx = q[1]; var my = q[2]; var mz = q[3]
    var ke = Float32(0.5) * (mx * mx + my * my + mz * mz) / rho
    var bx = q[5]; var by = q[6]; var bz = q[7]
    var me = Float32(0.5) * (bx * bx + by * by + bz * bz)
    var p  = (gamma - Float32(1.0)) * (q[4] - ke - me)
    return p if p > press_min else press_min


# Fast magnetosonic speed in a given direction.
#   c_s^2       = gamma p / rho
#   c_a^2       = |B|^2 / rho          (full Alfven, isotropic)
#   c_an^2      = (B . n)^2 / rho      (normal Alfven)
#   c_f^2       = 0.5 (c_s^2 + c_a^2) +
#                 0.5 sqrt((c_s^2 + c_a^2)^2 - 4 c_s^2 c_an^2)
def mhd_fast_speed(
    q: UnsafePointer[Float32, MutAnyOrigin],
    nx: Float32, ny: Float32, nz: Float32,
    gamma: Float32, rho_min: Float32, press_min: Float32,
) -> Float32:
    var rho = mhd_rho_floored(q, rho_min)
    var p   = mhd_gas_pressure(q, gamma, rho_min, press_min)
    var bx = q[5]; var by = q[6]; var bz = q[7]
    var bn = bx * nx + by * ny + bz * nz
    var cs2 = gamma * p / rho
    var ca2 = (bx * bx + by * by + bz * bz) / rho
    var can2 = (bn * bn) / rho
    var disc = (cs2 + ca2) * (cs2 + ca2) - Float32(4.0) * cs2 * can2
    if disc < Float32(0.0):
        disc = Float32(0.0)
    return sqrt(Float32(0.5) * (cs2 + ca2 + sqrt(disc)))


# ----------------------------------------------------------------------
# Ideal-MHD flux in a single direction (packs F^x / F^y / F^z into the
# solver's flux[d*9 + c] layout).  p_tot = p_gas + |B|^2/2.
# ----------------------------------------------------------------------
def mhd_flux_dir(
    gamma: Float32, rho_min: Float32, press_min: Float32, c_h: Float32,
    q: UnsafePointer[Float32, MutAnyOrigin],
    flux: UnsafePointer[Float32, MutAnyOrigin],
):
    var rho = mhd_rho_floored(q, rho_min)
    var mx = q[1]; var my = q[2]; var mz = q[3]
    var u = mx / rho; var v = my / rho; var w = mz / rho
    var E = q[4]
    var bx = q[5]; var by = q[6]; var bz = q[7]
    var psi = q[8]
    var p_gas = mhd_gas_pressure(q, gamma, rho_min, press_min)
    var pB = Float32(0.5) * (bx * bx + by * by + bz * bz)
    var p_tot = p_gas + pB
    var ub = u * bx + v * by + w * bz
    var ch2 = c_h * c_h

    # F^x = [rho u,
    #        rho u^2 + p_tot - Bx^2,
    #        rho u v - Bx By,
    #        rho u w - Bx Bz,
    #        u (E + p_tot) - Bx (u.B),
    #        psi, u By - v Bx, u Bz - w Bx,
    #        c_h^2 Bx]
    flux[0 * 9 + 0] = mx
    flux[0 * 9 + 1] = mx * u + p_tot - bx * bx
    flux[0 * 9 + 2] = mx * v - bx * by
    flux[0 * 9 + 3] = mx * w - bx * bz
    flux[0 * 9 + 4] = u * (E + p_tot) - bx * ub
    flux[0 * 9 + 5] = psi
    flux[0 * 9 + 6] = u * by - v * bx
    flux[0 * 9 + 7] = u * bz - w * bx
    flux[0 * 9 + 8] = ch2 * bx

    # F^y
    flux[1 * 9 + 0] = my
    flux[1 * 9 + 1] = my * u - by * bx
    flux[1 * 9 + 2] = my * v + p_tot - by * by
    flux[1 * 9 + 3] = my * w - by * bz
    flux[1 * 9 + 4] = v * (E + p_tot) - by * ub
    flux[1 * 9 + 5] = v * bx - u * by
    flux[1 * 9 + 6] = psi
    flux[1 * 9 + 7] = v * bz - w * by
    flux[1 * 9 + 8] = ch2 * by

    # F^z
    flux[2 * 9 + 0] = mz
    flux[2 * 9 + 1] = mz * u - bz * bx
    flux[2 * 9 + 2] = mz * v - bz * by
    flux[2 * 9 + 3] = mz * w + p_tot - bz * bz
    flux[2 * 9 + 4] = w * (E + p_tot) - bz * ub
    flux[2 * 9 + 5] = w * bx - u * bz
    flux[2 * 9 + 6] = w * by - v * bz
    flux[2 * 9 + 7] = psi
    flux[2 * 9 + 8] = ch2 * bz


# Normal-direction flux F . n for use inside the Rusanov sum (avoids
# allocating a full 3x9 flux tensor at the face).
def mhd_normal_flux(
    gamma: Float32, rho_min: Float32, press_min: Float32, c_h: Float32,
    q: UnsafePointer[Float32, MutAnyOrigin],
    nx: Float32, ny: Float32, nz: Float32,
    F_out: UnsafePointer[Float32, MutAnyOrigin],
):
    var rho = mhd_rho_floored(q, rho_min)
    var mx = q[1]; var my = q[2]; var mz = q[3]
    var u = mx / rho; var v = my / rho; var w = mz / rho
    var E = q[4]
    var bx = q[5]; var by = q[6]; var bz = q[7]
    var psi = q[8]
    var p_gas = mhd_gas_pressure(q, gamma, rho_min, press_min)
    var pB = Float32(0.5) * (bx * bx + by * by + bz * bz)
    var p_tot = p_gas + pB
    var ub = u * bx + v * by + w * bz
    var un = u * nx + v * ny + w * nz
    var bn = bx * nx + by * ny + bz * nz
    var ch2 = c_h * c_h

    F_out[0] = rho * un
    F_out[1] = rho * u * un + p_tot * nx - bn * bx
    F_out[2] = rho * v * un + p_tot * ny - bn * by
    F_out[3] = rho * w * un + p_tot * nz - bn * bz
    F_out[4] = un * (E + p_tot) - bn * ub
    F_out[5] = un * bx - u * bn + psi * nx
    F_out[6] = un * by - v * bn + psi * ny
    F_out[7] = un * bz - w * bn + psi * nz
    F_out[8] = ch2 * bn


# ======================================================================
# Ideal MHD struct
# ======================================================================

struct IdealMHD(Physics, ImplicitlyCopyable):
    comptime NUM_COMPONENTS = 9

    var gamma:        Float32
    var min_density:  Float32
    var min_pressure: Float32
    # GLM divergence-cleaning parameters.  c_h is the hyperbolic
    # transport speed for div(B) errors (usually set to a conservative
    # upper bound on the mesh-wide fast magnetosonic speed); alpha_d is
    # the Dedner damping rate on psi.  Setting both to zero disables
    # GLM entirely and turns this into "plain" (uncleaned) ideal MHD.
    var c_h:     Float32
    var alpha_d: Float32

    # BC_INFLOW ghost state (rho, rho u, rho v, rho w, E, Bx, By, Bz,
    # psi).  Defaults zero; existing drivers unaffected.
    var inflow_rho:  Float32
    var inflow_rhou: Float32
    var inflow_rhov: Float32
    var inflow_rhow: Float32
    var inflow_E:    Float32
    var inflow_Bx:   Float32
    var inflow_By:   Float32
    var inflow_Bz:   Float32
    var inflow_psi:  Float32

    def __init__(
        out self,
        gamma: Float32,
        min_density: Float32,
        min_pressure: Float32,
        c_h: Float32,
        alpha_d: Float32,
        inflow_rho: Float32  = Float32(0.0),
        inflow_rhou: Float32 = Float32(0.0),
        inflow_rhov: Float32 = Float32(0.0),
        inflow_rhow: Float32 = Float32(0.0),
        inflow_E: Float32    = Float32(0.0),
        inflow_Bx: Float32   = Float32(0.0),
        inflow_By: Float32   = Float32(0.0),
        inflow_Bz: Float32   = Float32(0.0),
        inflow_psi: Float32  = Float32(0.0),
    ):
        self.gamma = gamma
        self.min_density = min_density
        self.min_pressure = min_pressure
        self.c_h = c_h
        self.alpha_d = alpha_d
        self.inflow_rho = inflow_rho
        self.inflow_rhou = inflow_rhou
        self.inflow_rhov = inflow_rhov
        self.inflow_rhow = inflow_rhow
        self.inflow_E = inflow_E
        self.inflow_Bx = inflow_Bx
        self.inflow_By = inflow_By
        self.inflow_Bz = inflow_Bz
        self.inflow_psi = inflow_psi

    # --- DevicePassable plumbing (see std.gpu.host.device_context) ---
    comptime device_type = Self

    def _to_device_type[origin: MutOrigin](
        self, target: UnsafePointer[NoneType, origin]
    ):
        target.bitcast[Self]()[] = self

    @staticmethod
    def get_type_name() -> String:
        return "IdealMHD"

    def internal_flux(
        self,
        q:    UnsafePointer[Float32, MutAnyOrigin],
        flux: UnsafePointer[Float32, MutAnyOrigin],
    ) -> Float32:
        mhd_flux_dir(
            self.gamma, self.min_density, self.min_pressure, self.c_h,
            q, flux,
        )
        # Loose CFL-upper-bound: |u| + sqrt(c_s^2 + c_a^2).  Per-face
        # Rusanov dissipation uses a tighter anisotropic c_f.
        var rho = mhd_rho_floored(q, self.min_density)
        var u = q[1] / rho; var v = q[2] / rho; var w = q[3] / rho
        var p = mhd_gas_pressure(
            q, self.gamma, self.min_density, self.min_pressure
        )
        var bx = q[5]; var by = q[6]; var bz = q[7]
        var cs2 = self.gamma * p / rho
        var ca2 = (bx * bx + by * by + bz * bz) / rho
        var cf = sqrt(cs2 + ca2)
        var speed = sqrt(u * u + v * v + w * w) + cf
        return max(speed, self.c_h)

    def numerical_flux(
        self,
        q_l: UnsafePointer[Float32, MutAnyOrigin],
        q_r: UnsafePointer[Float32, MutAnyOrigin],
        nx: Float32, ny: Float32, nz: Float32,
        flux: UnsafePointer[Float32, MutAnyOrigin],
    ) -> Float32:
        var Fl = InlineArray[Float32, 9](fill=0.0)
        var Fr = InlineArray[Float32, 9](fill=0.0)
        var Fl_p = rebind[UnsafePointer[Float32, MutAnyOrigin]](Fl.unsafe_ptr())
        var Fr_p = rebind[UnsafePointer[Float32, MutAnyOrigin]](Fr.unsafe_ptr())
        mhd_normal_flux(
            self.gamma, self.min_density, self.min_pressure, self.c_h,
            q_l, nx, ny, nz, Fl_p,
        )
        mhd_normal_flux(
            self.gamma, self.min_density, self.min_pressure, self.c_h,
            q_r, nx, ny, nz, Fr_p,
        )
        # Rusanov alpha = max(|u_n| + c_f, c_h).
        var cf_l = mhd_fast_speed(
            q_l, nx, ny, nz,
            self.gamma, self.min_density, self.min_pressure,
        )
        var cf_r = mhd_fast_speed(
            q_r, nx, ny, nz,
            self.gamma, self.min_density, self.min_pressure,
        )
        var rhol = mhd_rho_floored(q_l, self.min_density)
        var rhor = mhd_rho_floored(q_r, self.min_density)
        var unl = (q_l[1] * nx + q_l[2] * ny + q_l[3] * nz) / rhol
        var unr = (q_r[1] * nx + q_r[2] * ny + q_r[3] * nz) / rhor
        var aunl = unl if unl >= Float32(0.0) else -unl
        var aunr = unr if unr >= Float32(0.0) else -unr
        var alpha_l = aunl + cf_l
        var alpha_r = aunr + cf_r
        var alpha = max(max(alpha_l, alpha_r), self.c_h)

        var half = Float32(0.5)
        for c in range(9):
            flux[c] = half * (Fl[c] + Fr[c]) - half * alpha * (q_r[c] - q_l[c])
        return alpha

    # Conducting wall: flip normal u AND normal B; preserve tangential
    # components + density + energy.  psi handled as zero-gradient (the
    # GLM cleaner's own transport takes care of it at walls).
    def boundary_flux(
        self,
        q_int: UnsafePointer[Float32, MutAnyOrigin],
        bc_type: Int32,
        nx: Float32, ny: Float32, nz: Float32,
        flux: UnsafePointer[Float32, MutAnyOrigin],
    ) -> Float32:
        var q_ghost = InlineArray[Float32, 9](fill=0.0)
        if bc_type == BC_WALL:
            q_ghost[0] = q_int[0]
            q_ghost[4] = q_int[4]
            q_ghost[8] = q_int[8]
            var mn = q_int[1] * nx + q_int[2] * ny + q_int[3] * nz
            q_ghost[1] = q_int[1] - Float32(2.0) * mn * nx
            q_ghost[2] = q_int[2] - Float32(2.0) * mn * ny
            q_ghost[3] = q_int[3] - Float32(2.0) * mn * nz
            var bn = q_int[5] * nx + q_int[6] * ny + q_int[7] * nz
            q_ghost[5] = q_int[5] - Float32(2.0) * bn * nx
            q_ghost[6] = q_int[6] - Float32(2.0) * bn * ny
            q_ghost[7] = q_int[7] - Float32(2.0) * bn * nz
        elif bc_type == BC_INFLOW:
            q_ghost[0] = self.inflow_rho
            q_ghost[1] = self.inflow_rhou
            q_ghost[2] = self.inflow_rhov
            q_ghost[3] = self.inflow_rhow
            q_ghost[4] = self.inflow_E
            q_ghost[5] = self.inflow_Bx
            q_ghost[6] = self.inflow_By
            q_ghost[7] = self.inflow_Bz
            q_ghost[8] = self.inflow_psi
        else:
            # BC_OUTFLOW / default: zero-gradient
            q_ghost[0] = q_int[0]
            q_ghost[4] = q_int[4]
            q_ghost[8] = q_int[8]
            q_ghost[1] = q_int[1]
            q_ghost[2] = q_int[2]
            q_ghost[3] = q_int[3]
            q_ghost[5] = q_int[5]
            q_ghost[6] = q_int[6]
            q_ghost[7] = q_int[7]
        var q_ghost_p = rebind[UnsafePointer[Float32, MutAnyOrigin]](
            q_ghost.unsafe_ptr()
        )
        return self.numerical_flux(q_int, q_ghost_p, nx, ny, nz, flux)

    # GLM damping: dpsi/dt += -alpha_d * psi.  All other components have
    # no source in clean ideal MHD.
    def source_term(
        self,
        q: UnsafePointer[Float32, MutAnyOrigin],
        x: Float32, y: Float32, z: Float32,
        source_out: UnsafePointer[Float32, MutAnyOrigin],
    ):
        for c in range(8):
            source_out[c] = Float32(0.0)
        source_out[8] = -self.alpha_d * q[8]

    def limit_state(
        self,
        q: UnsafePointer[Float32, MutAnyOrigin],
    ):
        pass
