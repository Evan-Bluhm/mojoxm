# ======================================================================
# Five-moment two-fluid plasma physics
# ======================================================================
#
# Electron fluid + ion fluid + full Maxwell field + GLM div(B)-cleaner,
# 17 conservative components laid out as
#
#   q[0..4]   = (rho_e, rho_e u_e, rho_e v_e, rho_e w_e, E_e)
#   q[5..9]   = (rho_i, rho_i u_i, rho_i v_i, rho_i w_i, E_i)
#   q[10..15] = (Ex, Ey, Ez, Bx, By, Bz)
#   q[16]     = psi              (GLM scalar for div(B) cleaning)
#
# The fluid fluxes are two independent 5-moment Euler fluxes.  The
# Maxwell+GLM flux is the same as in src/mhd.mojo but without the
# fluid advection term (vacuum-form curl equations).  All inter-fluid
# and fluid-field coupling is through the pointwise source term:
#
#   d(rho_s u_s)/dt += (q_s/m_s) rho_s (E + u_s x B)    (Lorentz force)
#   dE_s/dt          += (q_s/m_s) rho_s (u_s . E)       (work by E)
#   dE/dt            += -J / eps0                        (Ampere source)
#   dB/dt            += 0
#   dpsi/dt          += -alpha_d * psi                   (GLM damping)
#
# with J = sum_s (n_s q_s u_s) = (rho_e q_e/m_e) u_e + (rho_i q_i/m_i) u_i.
#
# Stiffness: the source-term matrix has eigenvalues including the
# plasma frequency omega_p and the cyclotron frequencies Omega_s;
# integrating explicitly requires dt < 1/max(omega_p, Omega_s).  For
# modest fields that's fine.  A proper IMEX split would loosen it.
#
# Numerical flux: Rusanov with alpha taken as the largest of the two
# fluid signal speeds (|u_s . n| + sqrt(gamma_s p_s/rho_s)), the EM
# speed c, and the GLM cleaner speed c_h.
#
# Boundary conditions:
#   BC_WALL    = slip wall for each fluid (reflect normal momentum) +
#                PEC wall for EM (flip tangential E, flip normal B).
#   BC_OUTFLOW = zero-gradient on everything.
# ======================================================================

from src.solver import Physics
from src.boundary import BC_WALL, BC_OUTFLOW, BC_INFLOW
from std.math import sqrt


# ----------------------------------------------------------------------
# Floored primitives for a single 5-moment species (q points at the
# first of that species' 5 components).
# ----------------------------------------------------------------------


def _species_rho_floored(q: UnsafePointer[Float32, MutAnyOrigin], rho_min: Float32) -> Float32:
    var r = q[0]
    return r if r > rho_min else rho_min


def _species_pressure(q: UnsafePointer[Float32, MutAnyOrigin], gamma: Float32, rho_min: Float32, press_min: Float32) -> Float32:
    var rho = _species_rho_floored(q, rho_min)
    var mx = q[1]
    var my = q[2]
    var mz = q[3]
    var ke = Float32(0.5) * (mx * mx + my * my + mz * mz) / rho
    var p = (gamma - Float32(1.0)) * (q[4] - ke)
    return p if p > press_min else press_min


def _species_sound(q: UnsafePointer[Float32, MutAnyOrigin], gamma: Float32, rho_min: Float32, press_min: Float32) -> Float32:
    var rho = _species_rho_floored(q, rho_min)
    var p = _species_pressure(q, gamma, rho_min, press_min)
    return sqrt(gamma * p / rho)


# Write the 5-component Euler flux for this species in direction d into
# flux_out[0..4].  Stride handled by the caller.
def _species_flux_dir(q: UnsafePointer[Float32, MutAnyOrigin], gamma: Float32, rho_min: Float32, press_min: Float32, d: Int, flux_out: UnsafePointer[Float32, MutAnyOrigin]):
    var rho = _species_rho_floored(q, rho_min)
    var mx = q[1]
    var my = q[2]
    var mz = q[3]
    var u = mx / rho
    var v = my / rho
    var w = mz / rho
    var E = q[4]
    var p = _species_pressure(q, gamma, rho_min, press_min)
    if d == 0:
        flux_out[0] = mx
        flux_out[1] = mx * u + p
        flux_out[2] = mx * v
        flux_out[3] = mx * w
        flux_out[4] = u * (E + p)
    elif d == 1:
        flux_out[0] = my
        flux_out[1] = my * u
        flux_out[2] = my * v + p
        flux_out[3] = my * w
        flux_out[4] = v * (E + p)
    else:
        flux_out[0] = mz
        flux_out[1] = mz * u
        flux_out[2] = mz * v
        flux_out[3] = mz * w + p
        flux_out[4] = w * (E + p)


# Normal-direction flux for a single species.  Same result as summing
# _species_flux_dir(d) * n_d but hand-rolled to avoid an extra loop.
def _species_normal_flux(
    q: UnsafePointer[Float32, MutAnyOrigin],
    gamma: Float32,
    rho_min: Float32,
    press_min: Float32,
    nx: Float32,
    ny: Float32,
    nz: Float32,
    flux_out: UnsafePointer[Float32, MutAnyOrigin],
):
    var rho = _species_rho_floored(q, rho_min)
    var mx = q[1]
    var my = q[2]
    var mz = q[3]
    var u = mx / rho
    var v = my / rho
    var w = mz / rho
    var E = q[4]
    var p = _species_pressure(q, gamma, rho_min, press_min)
    var un = u * nx + v * ny + w * nz
    flux_out[0] = rho * un
    flux_out[1] = rho * u * un + p * nx
    flux_out[2] = rho * v * un + p * ny
    flux_out[3] = rho * w * un + p * nz
    flux_out[4] = un * (E + p)


# ======================================================================
# FiveMomentTwoFluid struct
# ======================================================================


struct FiveMomentTwoFluid(ImplicitlyCopyable, Physics):
    comptime NUM_COMPONENTS = 17

    # Per-species adiabatic index.  For most plasma problems both are 5/3.
    var gamma_e: Float32
    var gamma_i: Float32

    # Species charge-to-mass ratios, in whatever units the driver picks.
    # Canonical normalisation: q_e = -1, q_i = +1, m_e = 1, m_i = mass
    # ratio (25 for a reduced-ratio demo; 1836 for real hydrogen).
    var q_e: Float32
    var m_e: Float32
    var q_i: Float32
    var m_i: Float32

    # Permittivity.  Usually 1 in the natural unit system above; set
    # smaller to make the plasma "stiffer" (larger omega_p).
    var eps0: Float32
    # Speed of light in the EM sector.  Usually 1.
    var c_light: Float32

    # GLM cleaner: hyperbolic transport speed for div(B) errors and
    # Dedner damping rate.  Setting both to zero disables GLM.
    var c_h: Float32
    var alpha_d: Float32

    # Floor values for each species' density and pressure.
    var min_density: Float32
    var min_pressure: Float32

    # BC_INFLOW ghost state: one conservative value per component
    # (electrons 0..4, ions 5..9, EM 10..15, GLM psi 16).  Defaults
    # zero so existing drivers keep working with just the physics
    # constants.
    var inflow_rho_e: Float32
    var inflow_mxe: Float32
    var inflow_mye: Float32
    var inflow_mze: Float32
    var inflow_E_e: Float32
    var inflow_rho_i: Float32
    var inflow_mxi: Float32
    var inflow_myi: Float32
    var inflow_mzi: Float32
    var inflow_E_i: Float32
    var inflow_Ex: Float32
    var inflow_Ey: Float32
    var inflow_Ez: Float32
    var inflow_Bx: Float32
    var inflow_By: Float32
    var inflow_Bz: Float32
    var inflow_psi: Float32

    def __init__(
        out self,
        gamma_e: Float32,
        gamma_i: Float32,
        q_e: Float32,
        m_e: Float32,
        q_i: Float32,
        m_i: Float32,
        eps0: Float32,
        c_light: Float32,
        c_h: Float32,
        alpha_d: Float32,
        min_density: Float32,
        min_pressure: Float32,
        inflow_rho_e: Float32 = Float32(0.0),
        inflow_mxe: Float32 = Float32(0.0),
        inflow_mye: Float32 = Float32(0.0),
        inflow_mze: Float32 = Float32(0.0),
        inflow_E_e: Float32 = Float32(0.0),
        inflow_rho_i: Float32 = Float32(0.0),
        inflow_mxi: Float32 = Float32(0.0),
        inflow_myi: Float32 = Float32(0.0),
        inflow_mzi: Float32 = Float32(0.0),
        inflow_E_i: Float32 = Float32(0.0),
        inflow_Ex: Float32 = Float32(0.0),
        inflow_Ey: Float32 = Float32(0.0),
        inflow_Ez: Float32 = Float32(0.0),
        inflow_Bx: Float32 = Float32(0.0),
        inflow_By: Float32 = Float32(0.0),
        inflow_Bz: Float32 = Float32(0.0),
        inflow_psi: Float32 = Float32(0.0),
    ):
        self.gamma_e = gamma_e
        self.gamma_i = gamma_i
        self.q_e = q_e
        self.m_e = m_e
        self.q_i = q_i
        self.m_i = m_i
        self.eps0 = eps0
        self.c_light = c_light
        self.c_h = c_h
        self.alpha_d = alpha_d
        self.min_density = min_density
        self.min_pressure = min_pressure
        self.inflow_rho_e = inflow_rho_e
        self.inflow_mxe = inflow_mxe
        self.inflow_mye = inflow_mye
        self.inflow_mze = inflow_mze
        self.inflow_E_e = inflow_E_e
        self.inflow_rho_i = inflow_rho_i
        self.inflow_mxi = inflow_mxi
        self.inflow_myi = inflow_myi
        self.inflow_mzi = inflow_mzi
        self.inflow_E_i = inflow_E_i
        self.inflow_Ex = inflow_Ex
        self.inflow_Ey = inflow_Ey
        self.inflow_Ez = inflow_Ez
        self.inflow_Bx = inflow_Bx
        self.inflow_By = inflow_By
        self.inflow_Bz = inflow_Bz
        self.inflow_psi = inflow_psi

    # --- DevicePassable plumbing (see std.gpu.host.device_context) ---
    comptime device_type = Self

    def _to_device_type[origin: MutOrigin](self, target: UnsafePointer[NoneType, origin]):
        target.bitcast[Self]()[] = self

    @staticmethod
    def get_type_name() -> String:
        return "FiveMomentTwoFluid"

    # Internal flux for all 17 components, written in the solver's
    # flux[d * NC + c] layout.  Fluid-fluid and fluid-field decoupled
    # in the fluxes; coupling happens entirely in source_term.
    def internal_flux(self, q: UnsafePointer[Float32, MutAnyOrigin], flux: UnsafePointer[Float32, MutAnyOrigin]) -> Float32:
        var q_e_ptr = q + 0  # species e: q[0..4]
        var q_i_ptr = q + 5  # species i: q[5..9]
        var Ex = q[10]
        var Ey = q[11]
        var Ez = q[12]
        var Bx = q[13]
        var By = q[14]
        var Bz = q[15]
        var psi = q[16]
        var c2 = self.c_light * self.c_light
        var ch2 = self.c_h * self.c_h

        var NC = 17
        for d in range(3):
            # Species e: slots 0..4.
            _species_flux_dir(q_e_ptr, self.gamma_e, self.min_density, self.min_pressure, d, flux + d * NC + 0)
            # Species i: slots 5..9.
            _species_flux_dir(q_i_ptr, self.gamma_i, self.min_density, self.min_pressure, d, flux + d * NC + 5)
            # EM + GLM: slots 10..16.
            # F^x = (0, c^2 Bz, -c^2 By, psi, -Ez, Ey, c_h^2 Bx)
            # F^y = (-c^2 Bz, 0, c^2 Bx, Ez, psi, -Ex, c_h^2 By)
            # F^z = (c^2 By, -c^2 Bx, 0, -Ey, Ex, psi, c_h^2 Bz)
            if d == 0:
                flux[d * NC + 10] = Float32(0.0)
                flux[d * NC + 11] = c2 * Bz
                flux[d * NC + 12] = -c2 * By
                flux[d * NC + 13] = psi
                flux[d * NC + 14] = -Ez
                flux[d * NC + 15] = Ey
                flux[d * NC + 16] = ch2 * Bx
            elif d == 1:
                flux[d * NC + 10] = -c2 * Bz
                flux[d * NC + 11] = Float32(0.0)
                flux[d * NC + 12] = c2 * Bx
                flux[d * NC + 13] = Ez
                flux[d * NC + 14] = psi
                flux[d * NC + 15] = -Ex
                flux[d * NC + 16] = ch2 * By
            else:
                flux[d * NC + 10] = c2 * By
                flux[d * NC + 11] = -c2 * Bx
                flux[d * NC + 12] = Float32(0.0)
                flux[d * NC + 13] = -Ey
                flux[d * NC + 14] = Ex
                flux[d * NC + 15] = psi
                flux[d * NC + 16] = ch2 * Bz

        # Loose CFL bound for the caller.  Per-face numerical_flux is
        # tighter; we just want something defensively large.
        var c_e = _species_sound(q_e_ptr, self.gamma_e, self.min_density, self.min_pressure)
        var c_i = _species_sound(q_i_ptr, self.gamma_i, self.min_density, self.min_pressure)
        return max(max(c_e, c_i), max(self.c_light, self.c_h))

    # Rusanov (Lax-Friedrichs) numerical flux with alpha = worst-case
    # signal speed across the two species and the EM + GLM waves.
    def numerical_flux(
        self,
        q_l: UnsafePointer[Float32, MutAnyOrigin],
        q_r: UnsafePointer[Float32, MutAnyOrigin],
        nx: Float32,
        ny: Float32,
        nz: Float32,
        flux: UnsafePointer[Float32, MutAnyOrigin],
    ) -> Float32:
        var Fl = InlineArray[Float32, 17](fill=0.0)
        var Fr = InlineArray[Float32, 17](fill=0.0)
        var Fl_p = rebind[UnsafePointer[Float32, MutAnyOrigin]](Fl.unsafe_ptr())
        var Fr_p = rebind[UnsafePointer[Float32, MutAnyOrigin]](Fr.unsafe_ptr())

        # Species e.
        _species_normal_flux(q_l + 0, self.gamma_e, self.min_density, self.min_pressure, nx, ny, nz, Fl_p + 0)
        _species_normal_flux(q_r + 0, self.gamma_e, self.min_density, self.min_pressure, nx, ny, nz, Fr_p + 0)
        # Species i.
        _species_normal_flux(q_l + 5, self.gamma_i, self.min_density, self.min_pressure, nx, ny, nz, Fl_p + 5)
        _species_normal_flux(q_r + 5, self.gamma_i, self.min_density, self.min_pressure, nx, ny, nz, Fr_p + 5)
        # EM + GLM normal fluxes:
        #   F.n_E = c^2 (B x n)
        #   F.n_B = (n x E) + psi n
        #   F.n_psi = c_h^2 (B . n)
        var c2 = self.c_light * self.c_light
        var ch2 = self.c_h * self.c_h
        for side in range(2):
            var q = q_l if side == 0 else q_r
            var F = Fl_p if side == 0 else Fr_p
            var Ex = q[10]
            var Ey = q[11]
            var Ez = q[12]
            var Bx = q[13]
            var By = q[14]
            var Bz = q[15]
            var psi = q[16]
            var bn = Bx * nx + By * ny + Bz * nz
            F[10] = c2 * (By * nz - Bz * ny)
            F[11] = c2 * (Bz * nx - Bx * nz)
            F[12] = c2 * (Bx * ny - By * nx)
            F[13] = (ny * Ez - nz * Ey) + psi * nx
            F[14] = (nz * Ex - nx * Ez) + psi * ny
            F[15] = (nx * Ey - ny * Ex) + psi * nz
            F[16] = ch2 * bn

        # Alpha: max of fluid and EM signal speeds on both sides.
        var rho_el = _species_rho_floored(q_l + 0, self.min_density)
        var rho_er = _species_rho_floored(q_r + 0, self.min_density)
        var rho_il = _species_rho_floored(q_l + 5, self.min_density)
        var rho_ir = _species_rho_floored(q_r + 5, self.min_density)
        var un_el = (q_l[1] * nx + q_l[2] * ny + q_l[3] * nz) / rho_el
        var un_er = (q_r[1] * nx + q_r[2] * ny + q_r[3] * nz) / rho_er
        var un_il = (q_l[6] * nx + q_l[7] * ny + q_l[8] * nz) / rho_il
        var un_ir = (q_r[6] * nx + q_r[7] * ny + q_r[8] * nz) / rho_ir
        var ce_l = _species_sound(q_l + 0, self.gamma_e, self.min_density, self.min_pressure)
        var ce_r = _species_sound(q_r + 0, self.gamma_e, self.min_density, self.min_pressure)
        var ci_l = _species_sound(q_l + 5, self.gamma_i, self.min_density, self.min_pressure)
        var ci_r = _species_sound(q_r + 5, self.gamma_i, self.min_density, self.min_pressure)
        var aun_el = un_el if un_el >= Float32(0.0) else -un_el
        var aun_er = un_er if un_er >= Float32(0.0) else -un_er
        var aun_il = un_il if un_il >= Float32(0.0) else -un_il
        var aun_ir = un_ir if un_ir >= Float32(0.0) else -un_ir
        var alpha_e = max(aun_el + ce_l, aun_er + ce_r)
        var alpha_i = max(aun_il + ci_l, aun_ir + ci_r)
        var alpha = max(max(alpha_e, alpha_i), max(self.c_light, self.c_h))

        var half = Float32(0.5)
        for c in range(17):
            flux[c] = half * (Fl[c] + Fr[c]) - half * alpha * (q_r[c] - q_l[c])
        return alpha

    def boundary_flux(self, q_int: UnsafePointer[Float32, MutAnyOrigin], bc_type: Int32, nx: Float32, ny: Float32, nz: Float32, flux: UnsafePointer[Float32, MutAnyOrigin]) -> Float32:
        # Ghost state: flip normal momentum on each fluid (slip wall) +
        # PEC reflection on EM, or zero-gradient for outflow.  density
        # and energy for each fluid, plus psi, are copied either way.
        var q_ghost = InlineArray[Float32, 17](fill=0.0)
        for k in range(17):
            q_ghost[k] = q_int[k]
        if bc_type == BC_WALL:
            # Species e.
            var mn_e = q_int[1] * nx + q_int[2] * ny + q_int[3] * nz
            q_ghost[1] = q_int[1] - Float32(2.0) * mn_e * nx
            q_ghost[2] = q_int[2] - Float32(2.0) * mn_e * ny
            q_ghost[3] = q_int[3] - Float32(2.0) * mn_e * nz
            # Species i.
            var mn_i = q_int[6] * nx + q_int[7] * ny + q_int[8] * nz
            q_ghost[6] = q_int[6] - Float32(2.0) * mn_i * nx
            q_ghost[7] = q_int[7] - Float32(2.0) * mn_i * ny
            q_ghost[8] = q_int[8] - Float32(2.0) * mn_i * nz
            # EM: PEC wall (flip tangential E, flip normal B).
            var En = q_int[10] * nx + q_int[11] * ny + q_int[12] * nz
            var Bn = q_int[13] * nx + q_int[14] * ny + q_int[15] * nz
            q_ghost[10] = Float32(2.0) * En * nx - q_int[10]
            q_ghost[11] = Float32(2.0) * En * ny - q_int[11]
            q_ghost[12] = Float32(2.0) * En * nz - q_int[12]
            q_ghost[13] = q_int[13] - Float32(2.0) * Bn * nx
            q_ghost[14] = q_int[14] - Float32(2.0) * Bn * ny
            q_ghost[15] = q_int[15] - Float32(2.0) * Bn * nz
        elif bc_type == BC_INFLOW:
            # Dirichlet inflow: user-set ghost state for all 17
            # conservative components.  Rusanov still arbitrates which
            # side's information propagates into the domain via the
            # wave-speed dissipation, so upstream characteristics pull
            # from the inflow state and downstream-moving ones do not.
            q_ghost[0] = self.inflow_rho_e
            q_ghost[1] = self.inflow_mxe
            q_ghost[2] = self.inflow_mye
            q_ghost[3] = self.inflow_mze
            q_ghost[4] = self.inflow_E_e
            q_ghost[5] = self.inflow_rho_i
            q_ghost[6] = self.inflow_mxi
            q_ghost[7] = self.inflow_myi
            q_ghost[8] = self.inflow_mzi
            q_ghost[9] = self.inflow_E_i
            q_ghost[10] = self.inflow_Ex
            q_ghost[11] = self.inflow_Ey
            q_ghost[12] = self.inflow_Ez
            q_ghost[13] = self.inflow_Bx
            q_ghost[14] = self.inflow_By
            q_ghost[15] = self.inflow_Bz
            q_ghost[16] = self.inflow_psi
        # BC_OUTFLOW / default: ghost == interior (already copied).
        var q_ghost_p = rebind[UnsafePointer[Float32, MutAnyOrigin]](q_ghost.unsafe_ptr())
        return self.numerical_flux(q_int, q_ghost_p, nx, ny, nz, flux)

    # Pointwise source: Lorentz force on each fluid, J on Ampere, GLM
    # damping on psi.  All components read from q and written into
    # source_out in the 17-component layout.
    def source_term(self, q: UnsafePointer[Float32, MutAnyOrigin], x: Float32, y: Float32, z: Float32, source_out: UnsafePointer[Float32, MutAnyOrigin]):
        # --- Electron fluid primitives ---
        var rho_e = _species_rho_floored(q + 0, self.min_density)
        var ue = q[1] / rho_e
        var ve = q[2] / rho_e
        var we = q[3] / rho_e
        # --- Ion fluid primitives ---
        var rho_i = _species_rho_floored(q + 5, self.min_density)
        var ui = q[6] / rho_i
        var vi = q[7] / rho_i
        var wi = q[8] / rho_i
        # --- EM field ---
        var Ex = q[10]
        var Ey = q[11]
        var Ez = q[12]
        var Bx = q[13]
        var By = q[14]
        var Bz = q[15]
        var psi = q[16]

        # qm_s = q_s / m_s; for an MHD-like normalisation with q_e = -1,
        # m_e = 1, q_i = +1, m_i = mi_over_me these collapse to +/-
        # charge-to-mass ratios the driver hands us.
        var qme = self.q_e / self.m_e
        var qmi = self.q_i / self.m_i

        # Lorentz force per unit volume on each species:
        #   n_s q_s (E + u_s x B) = (q_s/m_s) rho_s (E + u_s x B)
        var vxB_e_x = ve * Bz - we * By
        var vxB_e_y = we * Bx - ue * Bz
        var vxB_e_z = ue * By - ve * Bx
        var vxB_i_x = vi * Bz - wi * By
        var vxB_i_y = wi * Bx - ui * Bz
        var vxB_i_z = ui * By - vi * Bx

        source_out[0] = Float32(0.0)  # no mass source
        source_out[1] = qme * rho_e * (Ex + vxB_e_x)
        source_out[2] = qme * rho_e * (Ey + vxB_e_y)
        source_out[3] = qme * rho_e * (Ez + vxB_e_z)
        # Work done by E on the electron fluid; v . (v x B) = 0 so no
        # B contribution to the energy source.
        source_out[4] = qme * rho_e * (ue * Ex + ve * Ey + we * Ez)

        source_out[5] = Float32(0.0)
        source_out[6] = qmi * rho_i * (Ex + vxB_i_x)
        source_out[7] = qmi * rho_i * (Ey + vxB_i_y)
        source_out[8] = qmi * rho_i * (Ez + vxB_i_z)
        source_out[9] = qmi * rho_i * (ui * Ex + vi * Ey + wi * Ez)

        # Current density J = sum_s n_s q_s u_s = (q_s/m_s) rho_s u_s.
        var Jx = qme * rho_e * ue + qmi * rho_i * ui
        var Jy = qme * rho_e * ve + qmi * rho_i * vi
        var Jz = qme * rho_e * we + qmi * rho_i * wi
        var inv_eps0 = Float32(1.0) / self.eps0
        source_out[10] = -Jx * inv_eps0
        source_out[11] = -Jy * inv_eps0
        source_out[12] = -Jz * inv_eps0
        source_out[13] = Float32(0.0)
        source_out[14] = Float32(0.0)
        source_out[15] = Float32(0.0)
        source_out[16] = -self.alpha_d * psi

    def limit_state(self, q: UnsafePointer[Float32, MutAnyOrigin]):
        pass
