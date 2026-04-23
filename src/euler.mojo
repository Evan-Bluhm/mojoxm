# ======================================================================
# Euler physics -- 5-moment compressible gas dynamics
# ======================================================================
#
# Conserved variables (per point, Float32):
#   q[0] = rho           (density)
#   q[1] = rho * u       (x momentum)
#   q[2] = rho * v       (y momentum)
#   q[3] = rho * w       (z momentum)
#   q[4] = E             (total energy = rho*e_internal + 0.5*rho*|u|^2)
#
# Ideal-gas closure:
#   p = (gamma - 1) * (E - 0.5 * rho * |u|^2)
#
# Ported from WARPXM's ~/GitHub/warpxm/src/apps/5-moment/euler.h.  All
# four numerical-flux schemes (Rusanov, Roe, HLLE, HLLEC) live here,
# selected at construction time by `Euler.flux_type` and branched in
# `numerical_flux`.  The Harten-Hyman entropy fix is available for the
# wave-based solvers via `entropy_fix`.
#
# Face rotation: the wave-based solvers work in a coordinate frame
# aligned with the face normal.  We rotate q_l / q_r into that frame,
# solve in 1D, then rotate the numerical flux back into the world
# frame.  The tangent/binormal directions are constructed on the fly
# from the normal so the caller only has to pass (nx, ny, nz) -- same
# API shape as the scalar Advection physics.
# ======================================================================

from std.math import sqrt, exp


# Flux-type enum (Mojo has no native enum; use comptime Ints).
comptime FLUX_RUSANOV: Int = 0
comptime FLUX_ROE:      Int = 1
comptime FLUX_HLLE:     Int = 2
comptime FLUX_HLLEC:    Int = 3


# ----------------------------------------------------------------------
# Floored primitive helpers
# ----------------------------------------------------------------------

def euler_rho_floored(
    q: UnsafePointer[Float32, MutAnyOrigin], rho_min: Float32
) -> Float32:
    var r = q[0]
    return r if r > rho_min else rho_min

def euler_pressure_floored(
    q: UnsafePointer[Float32, MutAnyOrigin],
    gamma: Float32, rho_min: Float32, press_min: Float32,
) -> Float32:
    var rho = euler_rho_floored(q, rho_min)
    var mx = q[1]; var my = q[2]; var mz = q[3]
    var ke = Float32(0.5) * (mx*mx + my*my + mz*mz) / rho
    var p  = (gamma - Float32(1.0)) * (q[4] - ke)
    return p if p > press_min else press_min


# ----------------------------------------------------------------------
# 1D physical flux (x-direction) for a single state q.
# ----------------------------------------------------------------------

def euler_flux_1d(
    gamma: Float32, q: UnsafePointer[Float32, MutAnyOrigin],
    rho_min: Float32, press_min: Float32,
    out_flux: UnsafePointer[Float32, MutAnyOrigin],
):
    var rho = euler_rho_floored(q, rho_min)
    var p   = euler_pressure_floored(q, gamma, rho_min, press_min)
    var u   = q[1] / rho
    out_flux[0] = q[1]
    out_flux[1] = q[1] * u + p
    out_flux[2] = q[2] * u
    out_flux[3] = q[3] * u
    out_flux[4] = u * (q[4] + p)


# ----------------------------------------------------------------------
# Tangent/binormal construction from a unit normal (Gram-Schmidt
# trick used by WARPXM's `gpu_construct_tangent_and_binormal`).
# ----------------------------------------------------------------------
#
# Returns (tx, ty, tz, bx, by, bz) -- tangent and binormal, packed
# into a small struct to keep the call site readable.

@fieldwise_init
struct _TB(ImplicitlyCopyable, Movable):
    var tx: Float32
    var ty: Float32
    var tz: Float32
    var bx: Float32
    var by: Float32
    var bz: Float32

def euler_construct_tb(nx: Float32, ny: Float32, nz: Float32) -> _TB:
    # Start with the normal, then perturb away from the singular axis
    # (hairy-ball theorem: we need the conditional).
    var ax = nx
    var ay = ny
    var az = nz
    var anx = ax if ax >= Float32(0.0) else -ax
    if anx >= Float32(0.9):
        ay += Float32(1.0)
    else:
        ax += Float32(1.0)
    # Remove the normal component (Gram-Schmidt).
    var pro = nx * ax + ny * ay + nz * az
    ax -= nx * pro
    ay -= ny * pro
    az -= nz * pro
    var mag = sqrt(ax*ax + ay*ay + az*az)
    var tx = ax / mag
    var ty = ay / mag
    var tz = az / mag
    # Binormal = n x t (already unit length).
    var bx = ny * tz - nz * ty
    var by = nz * tx - nx * tz
    var bz = nx * ty - ny * tx
    return _TB(tx, ty, tz, bx, by, bz)


# ----------------------------------------------------------------------
# Rotate / anti-rotate a 5-moment vector through the face-normal frame.
#   [0] density   (scalar, unchanged)
#   [1..3] momentum components rotated like a vector
#   [4] total energy (scalar, unchanged)
# ----------------------------------------------------------------------

def euler_rotate(
    q: UnsafePointer[Float32, MutAnyOrigin],
    nx: Float32, ny: Float32, nz: Float32, tb: _TB,
    rq: UnsafePointer[Float32, MutAnyOrigin],
):
    rq[0] = q[0]
    rq[4] = q[4]
    rq[1] = nx * q[1] + ny * q[2] + nz * q[3]
    rq[2] = tb.tx * q[1] + tb.ty * q[2] + tb.tz * q[3]
    rq[3] = tb.bx * q[1] + tb.by * q[2] + tb.bz * q[3]

def euler_antirotate(
    rq: UnsafePointer[Float32, MutAnyOrigin],
    nx: Float32, ny: Float32, nz: Float32, tb: _TB,
    q: UnsafePointer[Float32, MutAnyOrigin],
):
    q[0] = rq[0]
    q[4] = rq[4]
    # Inverse of the orthogonal rotation = transpose.
    q[1] = nx * rq[1] + tb.tx * rq[2] + tb.bx * rq[3]
    q[2] = ny * rq[1] + tb.ty * rq[2] + tb.by * rq[3]
    q[3] = nz * rq[1] + tb.tz * rq[2] + tb.bz * rq[3]


# ----------------------------------------------------------------------
# Roe averages (used by Roe, HLLE, HLLEC).
# ----------------------------------------------------------------------

@fieldwise_init
struct _RoeAvg(ImplicitlyCopyable, Movable):
    var rho:  Float32
    var u:    Float32
    var v:    Float32
    var w:    Float32
    var enth: Float32
    var a:    Float32     # Roe-averaged sound speed

def euler_roe_averages(
    gamma: Float32,
    ql: UnsafePointer[Float32, MutAnyOrigin],
    qr: UnsafePointer[Float32, MutAnyOrigin],
    pl: Float32, pr: Float32, rho_min: Float32,
) -> _RoeAvg:
    var g1 = gamma - Float32(1.0)
    var rhol = euler_rho_floored(ql, rho_min)
    var rhor = euler_rho_floored(qr, rho_min)
    var rhoroe = sqrt(rhol * rhor)
    var sl = sqrt(rhol); var sr = sqrt(rhor)
    var s  = sl + sr
    var u = (ql[1] / sl + qr[1] / sr) / s
    var v = (ql[2] / sl + qr[2] / sr) / s
    var w = (ql[3] / sl + qr[3] / sr) / s
    var enth = ((ql[4] + pl) / sl + (qr[4] + pr) / sr) / s
    var u2v2w2 = u*u + v*v + w*w
    var aa2 = max(g1 * (enth - Float32(0.5) * u2v2w2), Float32(0.0))
    var a = sqrt(aa2)
    return _RoeAvg(rhoroe, u, v, w, enth, a)


# ----------------------------------------------------------------------
# Fluctuations from waves (plain Rankine-Hugoniot, no entropy fix).
# ----------------------------------------------------------------------

def euler_compute_fluctuations(
    s: UnsafePointer[Float32, MutAnyOrigin],         # [3]
    wave: UnsafePointer[Float32, MutAnyOrigin],      # [5 * 3] laid out wave[m*3 + mw]
    amdq: UnsafePointer[Float32, MutAnyOrigin],      # [5]
    apdq: UnsafePointer[Float32, MutAnyOrigin],      # [5]
):
    for m in range(5):
        amdq[m] = Float32(0.0)
        apdq[m] = Float32(0.0)
        for mw in range(3):
            var sm = s[mw]
            var wm = wave[m*3 + mw]
            if sm < Float32(0.0):
                amdq[m] += sm * wm
            else:
                apdq[m] += sm * wm


# ----------------------------------------------------------------------
# Harten-Hyman entropy fix (wave-based solvers).
# ----------------------------------------------------------------------

def euler_harten_hyman_fix(
    gamma: Float32,
    ql: UnsafePointer[Float32, MutAnyOrigin],
    qr: UnsafePointer[Float32, MutAnyOrigin],
    s: UnsafePointer[Float32, MutAnyOrigin],
    wave: UnsafePointer[Float32, MutAnyOrigin],
    rho_min: Float32, press_min: Float32,
    amdq: UnsafePointer[Float32, MutAnyOrigin],
    apdq: UnsafePointer[Float32, MutAnyOrigin],
):
    # Left state characteristic speed of the slow (1-) wave.
    var rho_l = euler_rho_floored(ql, rho_min)
    var p_l   = euler_pressure_floored(ql, gamma, rho_min, press_min)
    var c_l   = sqrt(gamma * p_l / rho_l)
    var s0    = ql[1] / rho_l - c_l

    for m in range(5):
        amdq[m] = Float32(0.0)

    var compute_apdq_only = (s0 > Float32(0.0) and s[0] > Float32(0.0))

    if not compute_apdq_only:
        # *L state
        var q1_0 = max(ql[0] + wave[0*3 + 0], rho_min)
        var q1_1 = ql[1] + wave[1*3 + 0]
        var q1_2 = ql[2] + wave[2*3 + 0]
        var q1_3 = ql[3] + wave[3*3 + 0]
        var q1_4 = ql[4] + wave[4*3 + 0]
        var ke1  = Float32(0.5) * (q1_1*q1_1 + q1_2*q1_2 + q1_3*q1_3) / q1_0
        var p1   = max((gamma - Float32(1.0)) * (q1_4 - ke1), press_min)
        var c1   = sqrt(gamma * p1 / q1_0)
        var s1   = q1_1 / q1_0 - c1

        var sfract: Float32
        if s0 < Float32(0.0) and s1 > Float32(0.0):
            sfract = s0 * (s1 - s[0]) / (s1 - s0)
        elif s[0] < Float32(0.0):
            sfract = s[0]
        else:
            sfract = Float32(0.0)

        for m in range(5):
            amdq[m] = sfract * wave[m*3 + 0]

        # Contact (2-wave) contribution if upwind.
        if s[1] < Float32(0.0):
            for m in range(5):
                amdq[m] += s[1] * wave[m*3 + 1]

            # Now check the 3-wave.
            var rho_r = euler_rho_floored(qr, rho_min)
            var p_r   = euler_pressure_floored(qr, gamma, rho_min, press_min)
            var c_r   = sqrt(gamma * p_r / rho_r)
            var s3    = qr[1] / rho_r + c_r

            var q2_0 = max(qr[0] - wave[0*3 + 2], rho_min)
            var q2_1 = qr[1] - wave[1*3 + 2]
            var q2_2 = qr[2] - wave[2*3 + 2]
            var q2_3 = qr[3] - wave[3*3 + 2]
            var q2_4 = qr[4] - wave[4*3 + 2]
            var ke2  = Float32(0.5) * (q2_1*q2_1 + q2_2*q2_2 + q2_3*q2_3) / q2_0
            var p2   = max((gamma - Float32(1.0)) * (q2_4 - ke2), press_min)
            var c2   = sqrt(gamma * p2 / q2_0)
            var s2   = q2_1 / q2_0 + c2

            var sfract2: Float32
            if s2 < Float32(0.0) and s3 > Float32(0.0):
                sfract2 = s2 * (s3 - s[2]) / (s3 - s2)
            elif s[2] < Float32(0.0):
                sfract2 = s[2]
            else:
                sfract2 = Float32(0.0)
            for m in range(5):
                amdq[m] += sfract2 * wave[m*3 + 2]

    # apdq = sum(s * wave) - amdq
    for m in range(5):
        var ddf = Float32(0.0)
        for mw in range(3):
            ddf += s[mw] * wave[m*3 + mw]
        apdq[m] = ddf - amdq[m]


# ----------------------------------------------------------------------
# Wave-based flux solvers (return wave struct + fluctuations amdq, apdq)
# ----------------------------------------------------------------------

def euler_roe_solver(
    gamma: Float32,
    ql: UnsafePointer[Float32, MutAnyOrigin],
    qr: UnsafePointer[Float32, MutAnyOrigin],
    rho_min: Float32, press_min: Float32, entropy_fix: Bool,
    s: UnsafePointer[Float32, MutAnyOrigin],      # [3]
    wave: UnsafePointer[Float32, MutAnyOrigin],   # [5 * 3]
    amdq: UnsafePointer[Float32, MutAnyOrigin],   # [5]
    apdq: UnsafePointer[Float32, MutAnyOrigin],   # [5]
):
    var g1 = gamma - Float32(1.0)
    var d0 = qr[0] - ql[0]
    var d1 = qr[1] - ql[1]
    var d2 = qr[2] - ql[2]
    var d3 = qr[3] - ql[3]
    var d4 = qr[4] - ql[4]

    var pl = euler_pressure_floored(ql, gamma, rho_min, press_min)
    var pr = euler_pressure_floored(qr, gamma, rho_min, press_min)
    var ra = euler_roe_averages(gamma, ql, qr, pl, pr, rho_min)
    var u = ra.u; var v = ra.v; var w = ra.w; var enth = ra.enth; var a = ra.a

    var u2v2w2 = u*u + v*v + w*w
    var g1a2 = g1 / (a * a)
    var euv = enth - u2v2w2

    var a4 = g1a2 * (euv*d0 + u*d1 + v*d2 + w*d3 - d4)
    var a2 = d2 - v*d0
    var a3 = d3 - w*d0
    var a5 = (d1 + (a - u)*d0 - a*a4) / (Float32(2.0) * a)
    var a1 = d0 - a4 - a5

    # Wave 1: u - c
    wave[0*3 + 0] = a1
    wave[1*3 + 0] = a1 * (u - a)
    wave[2*3 + 0] = a1 * v
    wave[3*3 + 0] = a1 * w
    wave[4*3 + 0] = a1 * (enth - u * a)
    s[0] = u - a

    # Wave 2: u (lumped)
    wave[0*3 + 1] = a4
    wave[1*3 + 1] = a4 * u
    wave[2*3 + 1] = a4 * v + a2
    wave[3*3 + 1] = a4 * w + a3
    wave[4*3 + 1] = a4 * Float32(0.5) * u2v2w2 + a2 * v + a3 * w
    s[1] = u

    # Wave 3: u + c
    wave[0*3 + 2] = a5
    wave[1*3 + 2] = a5 * (u + a)
    wave[2*3 + 2] = a5 * v
    wave[3*3 + 2] = a5 * w
    wave[4*3 + 2] = a5 * (enth + u * a)
    s[2] = u + a

    if entropy_fix:
        euler_harten_hyman_fix(gamma, ql, qr, s, wave, rho_min, press_min,
                               amdq, apdq)
    else:
        euler_compute_fluctuations(s, wave, amdq, apdq)


def euler_hlle_solver(
    gamma: Float32,
    ql: UnsafePointer[Float32, MutAnyOrigin],
    qr: UnsafePointer[Float32, MutAnyOrigin],
    rho_min: Float32, press_min: Float32, entropy_fix: Bool,
    s: UnsafePointer[Float32, MutAnyOrigin],
    wave: UnsafePointer[Float32, MutAnyOrigin],
    amdq: UnsafePointer[Float32, MutAnyOrigin],
    apdq: UnsafePointer[Float32, MutAnyOrigin],
):
    var pl = euler_pressure_floored(ql, gamma, rho_min, press_min)
    var pr = euler_pressure_floored(qr, gamma, rho_min, press_min)
    var ra = euler_roe_averages(gamma, ql, qr, pl, pr, rho_min)
    var s_roe_min = ra.u - ra.a
    var s_roe_max = ra.u + ra.a

    var rho_l = euler_rho_floored(ql, rho_min)
    var a_l   = sqrt(gamma * pl / rho_l)
    var u_l   = ql[1] / rho_l
    var s_l_min = u_l - a_l

    var rho_r = euler_rho_floored(qr, rho_min)
    var a_r   = sqrt(gamma * pr / rho_r)
    var u_r   = qr[1] / rho_r
    var s_r_max = u_r + a_r

    s[0] = min(s_l_min, s_roe_min)
    s[2] = max(s_r_max, s_roe_max)
    s[1] = Float32(0.0)

    var fl = InlineArray[Float32, 5](fill=0.0)
    var fr = InlineArray[Float32, 5](fill=0.0)
    euler_flux_1d(gamma, ql, rho_min, press_min,
                  rebind[UnsafePointer[Float32, MutAnyOrigin]](fl.unsafe_ptr()))
    euler_flux_1d(gamma, qr, rho_min, press_min,
                  rebind[UnsafePointer[Float32, MutAnyOrigin]](fr.unsafe_ptr()))

    # Middle state
    var denom = s[0] - s[2]
    for m in range(5):
        var qhat = (fr[m] - fl[m] - s[2] * qr[m] + s[0] * ql[m]) / denom
        wave[m*3 + 0] = qhat - ql[m]
        wave[m*3 + 1] = Float32(0.0)
        wave[m*3 + 2] = qr[m] - qhat

    if entropy_fix:
        euler_harten_hyman_fix(gamma, ql, qr, s, wave, rho_min, press_min,
                               amdq, apdq)
    else:
        euler_compute_fluctuations(s, wave, amdq, apdq)


def euler_hllec_solver(
    gamma: Float32,
    ql: UnsafePointer[Float32, MutAnyOrigin],
    qr: UnsafePointer[Float32, MutAnyOrigin],
    rho_min: Float32, press_min: Float32, entropy_fix: Bool,
    s: UnsafePointer[Float32, MutAnyOrigin],
    wave: UnsafePointer[Float32, MutAnyOrigin],
    amdq: UnsafePointer[Float32, MutAnyOrigin],
    apdq: UnsafePointer[Float32, MutAnyOrigin],
):
    var pl = euler_pressure_floored(ql, gamma, rho_min, press_min)
    var pr = euler_pressure_floored(qr, gamma, rho_min, press_min)
    var ra = euler_roe_averages(gamma, ql, qr, pl, pr, rho_min)
    var s_roe_min = ra.u - ra.a
    var s_roe_max = ra.u + ra.a

    var rho_l = euler_rho_floored(ql, rho_min)
    var a_l   = sqrt(gamma * pl / rho_l)
    var u_l   = ql[1] / rho_l
    var s_l_min = u_l - a_l

    var rho_r = euler_rho_floored(qr, rho_min)
    var a_r   = sqrt(gamma * pr / rho_r)
    var u_r   = qr[1] / rho_r
    var s_r_max = u_r + a_r

    s[0] = min(s_l_min, s_roe_min)
    s[2] = max(s_r_max, s_roe_max)
    s[1] = (pr - pl + rho_l * u_l * (s[0] - u_l) - rho_r * u_r * (s[2] - u_r)) / (
        rho_l * (s[0] - u_l) - rho_r * (s[2] - u_r)
    )

    # Left middle state
    var lm = rho_l * (s[0] - u_l) / (s[0] - s[1])
    var qhL_0 = lm
    var qhL_1 = lm * s[1]
    var qhL_2 = lm * ql[2] / rho_l
    var qhL_3 = lm * ql[3] / rho_l
    var qhL_4 = lm * (ql[4] / rho_l + (s[1] - u_l)
                      * (s[1] + pl / (rho_l * (s[0] - u_l))))

    # Right middle state
    var rm = rho_r * (s[2] - u_r) / (s[2] - s[1])
    var qhR_0 = rm
    var qhR_1 = rm * s[1]
    var qhR_2 = rm * qr[2] / rho_r
    var qhR_3 = rm * qr[3] / rho_r
    var qhR_4 = rm * (qr[4] / rho_r + (s[1] - u_r)
                      * (s[1] + pr / (rho_r * (s[2] - u_r))))

    # Wave 1 = qhL - ql, wave 2 = qhR - qhL, wave 3 = qr - qhR
    wave[0*3 + 0] = qhL_0 - ql[0]
    wave[1*3 + 0] = qhL_1 - ql[1]
    wave[2*3 + 0] = qhL_2 - ql[2]
    wave[3*3 + 0] = qhL_3 - ql[3]
    wave[4*3 + 0] = qhL_4 - ql[4]

    wave[0*3 + 1] = qhR_0 - qhL_0
    wave[1*3 + 1] = qhR_1 - qhL_1
    wave[2*3 + 1] = qhR_2 - qhL_2
    wave[3*3 + 1] = qhR_3 - qhL_3
    wave[4*3 + 1] = qhR_4 - qhL_4

    wave[0*3 + 2] = qr[0] - qhR_0
    wave[1*3 + 2] = qr[1] - qhR_1
    wave[2*3 + 2] = qr[2] - qhR_2
    wave[3*3 + 2] = qr[3] - qhR_3
    wave[4*3 + 2] = qr[4] - qhR_4

    if entropy_fix:
        euler_harten_hyman_fix(gamma, ql, qr, s, wave, rho_min, press_min,
                               amdq, apdq)
    else:
        euler_compute_fluctuations(s, wave, amdq, apdq)


# ----------------------------------------------------------------------
# Given fluctuations, assemble the final numerical flux:
#   F = 0.5 (F_L + F_R) - 0.5 (APDQ - AMDQ)
# and compute the max wave speed for CFL.
# ----------------------------------------------------------------------

def euler_flux_from_fluctuations(
    gamma: Float32,
    ql: UnsafePointer[Float32, MutAnyOrigin],
    qr: UnsafePointer[Float32, MutAnyOrigin],
    rho_min: Float32, press_min: Float32,
    s:    UnsafePointer[Float32, MutAnyOrigin],
    amdq: UnsafePointer[Float32, MutAnyOrigin],
    apdq: UnsafePointer[Float32, MutAnyOrigin],
    flux: UnsafePointer[Float32, MutAnyOrigin],
) -> Float32:
    var fl = InlineArray[Float32, 5](fill=0.0)
    var fr = InlineArray[Float32, 5](fill=0.0)
    euler_flux_1d(gamma, ql, rho_min, press_min,
                  rebind[UnsafePointer[Float32, MutAnyOrigin]](fl.unsafe_ptr()))
    euler_flux_1d(gamma, qr, rho_min, press_min,
                  rebind[UnsafePointer[Float32, MutAnyOrigin]](fr.unsafe_ptr()))
    for m in range(5):
        flux[m] = Float32(0.5) * (fl[m] + fr[m]) - Float32(0.5) * (apdq[m] - amdq[m])
    # Max |wave speed|
    var s0 = s[0]; var s1 = s[1]; var s2 = s[2]
    var as0 = s0 if s0 >= Float32(0.0) else -s0
    var as1 = s1 if s1 >= Float32(0.0) else -s1
    var as2 = s2 if s2 >= Float32(0.0) else -s2
    var m01 = max(as0, as1)
    return max(m01, as2)


# ======================================================================
# Euler struct
# ======================================================================

from src.solver import Physics
from src.boundary import BC_WALL, BC_OUTFLOW


@fieldwise_init
struct Euler(Physics, ImplicitlyCopyable):
    comptime NUM_COMPONENTS = 5

    var gamma: Float32
    var min_density: Float32
    var min_pressure: Float32
    var flux_type: Int        # FLUX_RUSANOV / FLUX_ROE / FLUX_HLLE / FLUX_HLLEC
    var entropy_fix: Bool

    # Gravity vector (world frame).  Zero by default -- drivers set
    # nonzero components to enable the rho*g momentum + rho*(u.g)
    # energy source terms applied in `source_term`.
    var gx: Float32
    var gy: Float32
    var gz: Float32

    # --- DevicePassable plumbing (see std.gpu.host.device_context) ---
    comptime device_type = Self

    def _to_device_type[origin: MutOrigin](
        self, target: UnsafePointer[NoneType, origin]
    ):
        target.bitcast[Self]()[] = self

    @staticmethod
    def get_type_name() -> String:
        return "Euler"

    # Internal flux at a single node.  flux[d * NC + c] layout.
    def internal_flux(
        self,
        q:    UnsafePointer[Float32, MutAnyOrigin],
        flux: UnsafePointer[Float32, MutAnyOrigin],
    ) -> Float32:
        var rho = euler_rho_floored(q, self.min_density)
        var p   = euler_pressure_floored(q, self.gamma, self.min_density, self.min_pressure)
        var u = q[1] / rho
        var v = q[2] / rho
        var w = q[3] / rho
        # x-direction
        flux[0*5 + 0] = q[1]
        flux[0*5 + 1] = q[1] * u + p
        flux[0*5 + 2] = q[1] * v
        flux[0*5 + 3] = q[1] * w
        flux[0*5 + 4] = u * (q[4] + p)
        # y-direction
        flux[1*5 + 0] = q[2]
        flux[1*5 + 1] = q[2] * u
        flux[1*5 + 2] = q[2] * v + p
        flux[1*5 + 3] = q[2] * w
        flux[1*5 + 4] = v * (q[4] + p)
        # z-direction
        flux[2*5 + 0] = q[3]
        flux[2*5 + 1] = q[3] * u
        flux[2*5 + 2] = q[3] * v
        flux[2*5 + 3] = q[3] * w + p
        flux[2*5 + 4] = w * (q[4] + p)
        return Float32(1.0e30)

    # Numerical flux at a face.  Rotates into the face-normal frame,
    # solves the chosen 1D Riemann problem, and rotates back.  Returns
    # the max wave speed seen at the interface (for CFL).
    def numerical_flux(
        self,
        q_l: UnsafePointer[Float32, MutAnyOrigin],
        q_r: UnsafePointer[Float32, MutAnyOrigin],
        nx: Float32, ny: Float32, nz: Float32,
        flux: UnsafePointer[Float32, MutAnyOrigin],
    ) -> Float32:
        var tb = euler_construct_tb(nx, ny, nz)

        # Rotate both states into the local (normal-aligned) frame.
        var r_ql = InlineArray[Float32, 5](fill=0.0)
        var r_qr = InlineArray[Float32, 5](fill=0.0)
        euler_rotate(
            q_l, nx, ny, nz, tb,
            rebind[UnsafePointer[Float32, MutAnyOrigin]](r_ql.unsafe_ptr()),
        )
        euler_rotate(
            q_r, nx, ny, nz, tb,
            rebind[UnsafePointer[Float32, MutAnyOrigin]](r_qr.unsafe_ptr()),
        )
        var r_ql_p = rebind[UnsafePointer[Float32, MutAnyOrigin]](r_ql.unsafe_ptr())
        var r_qr_p = rebind[UnsafePointer[Float32, MutAnyOrigin]](r_qr.unsafe_ptr())

        var r_flux = InlineArray[Float32, 5](fill=0.0)
        var r_flux_p = rebind[UnsafePointer[Float32, MutAnyOrigin]](r_flux.unsafe_ptr())
        var vmax: Float32

        if self.flux_type == FLUX_RUSANOV:
            # Rusanov (Lax-Friedrichs) -- simplest, robust.
            var rhol = euler_rho_floored(r_ql_p, self.min_density)
            var rhor = euler_rho_floored(r_qr_p, self.min_density)
            var pl   = euler_pressure_floored(r_ql_p, self.gamma, self.min_density, self.min_pressure)
            var pr   = euler_pressure_floored(r_qr_p, self.gamma, self.min_density, self.min_pressure)
            var cl   = sqrt(self.gamma * pl / rhol)
            var cr   = sqrt(self.gamma * pr / rhor)
            var uxl  = r_ql[1] / rhol
            var uxr  = r_qr[1] / rhor
            var auxl = uxl if uxl >= Float32(0.0) else -uxl
            var auxr = uxr if uxr >= Float32(0.0) else -uxr
            var cmax = max(auxl + cl, auxr + cr)

            r_flux[0] = Float32(0.5) * (r_ql[1] + r_qr[1] + cmax * (rhol - rhor))
            r_flux[1] = Float32(0.5) * (r_ql[1] * uxl + pl + r_qr[1] * uxr + pr
                                         + cmax * (r_ql[1] - r_qr[1]))
            r_flux[2] = Float32(0.5) * (r_ql[2] * uxl + r_qr[2] * uxr
                                         + cmax * (r_ql[2] - r_qr[2]))
            r_flux[3] = Float32(0.5) * (r_ql[3] * uxl + r_qr[3] * uxr
                                         + cmax * (r_ql[3] - r_qr[3]))
            r_flux[4] = Float32(0.5) * ((r_ql[4] + pl) * uxl + (r_qr[4] + pr) * uxr
                                         + cmax * (r_ql[4] - r_qr[4]))
            vmax = cmax
        else:
            # Wave-based solvers: run the chosen Riemann solver, get
            # fluctuations, then F = 0.5(F_L + F_R) - 0.5(APDQ - AMDQ).
            var s    = InlineArray[Float32, 3](fill=0.0)
            var wave = InlineArray[Float32, 15](fill=0.0)   # 5 * 3
            var amdq = InlineArray[Float32, 5](fill=0.0)
            var apdq = InlineArray[Float32, 5](fill=0.0)
            var s_p    = rebind[UnsafePointer[Float32, MutAnyOrigin]](s.unsafe_ptr())
            var w_p    = rebind[UnsafePointer[Float32, MutAnyOrigin]](wave.unsafe_ptr())
            var amdq_p = rebind[UnsafePointer[Float32, MutAnyOrigin]](amdq.unsafe_ptr())
            var apdq_p = rebind[UnsafePointer[Float32, MutAnyOrigin]](apdq.unsafe_ptr())

            if self.flux_type == FLUX_ROE:
                euler_roe_solver(
                    self.gamma, r_ql_p, r_qr_p,
                    self.min_density, self.min_pressure, self.entropy_fix,
                    s_p, w_p, amdq_p, apdq_p,
                )
            elif self.flux_type == FLUX_HLLE:
                euler_hlle_solver(
                    self.gamma, r_ql_p, r_qr_p,
                    self.min_density, self.min_pressure, self.entropy_fix,
                    s_p, w_p, amdq_p, apdq_p,
                )
            else:   # FLUX_HLLEC
                euler_hllec_solver(
                    self.gamma, r_ql_p, r_qr_p,
                    self.min_density, self.min_pressure, self.entropy_fix,
                    s_p, w_p, amdq_p, apdq_p,
                )
            vmax = euler_flux_from_fluctuations(
                self.gamma, r_ql_p, r_qr_p,
                self.min_density, self.min_pressure,
                s_p, amdq_p, apdq_p, r_flux_p,
            )

        # Rotate numerical flux back to world frame.
        euler_antirotate(r_flux_p, nx, ny, nz, tb, flux)
        return vmax

    # Boundary flux via the ghost-state approach: build a synthetic
    # `q_ghost` on the other side of the face that encodes the BC, then
    # call `numerical_flux(q_int, q_ghost, ...)`.  This keeps every
    # Riemann solver (Rusanov/Roe/HLLE/HLLEC) applicable at the wall
    # with no new code paths.
    #
    # BC_WALL (slip wall): flip the normal momentum component so
    #   (q_int + q_ghost) has zero normal momentum at the interface;
    #   density, energy, and tangential momentum are copied.
    # BC_OUTFLOW (transmissive): q_ghost = q_int, the zero-gradient
    #   upwind-through extrapolation.  Supersonic outflow is exact;
    #   subsonic is marginal but standard for a first implementation.
    def boundary_flux(
        self,
        q_int: UnsafePointer[Float32, MutAnyOrigin],
        bc_type: Int32,
        nx: Float32, ny: Float32, nz: Float32,
        flux: UnsafePointer[Float32, MutAnyOrigin],
    ) -> Float32:
        var q_ghost = InlineArray[Float32, 5](fill=0.0)
        q_ghost[0] = q_int[0]
        q_ghost[4] = q_int[4]
        if bc_type == BC_WALL:
            # Reflect the normal momentum; preserve tangential.
            var mn = q_int[1] * nx + q_int[2] * ny + q_int[3] * nz
            q_ghost[1] = q_int[1] - Float32(2.0) * mn * nx
            q_ghost[2] = q_int[2] - Float32(2.0) * mn * ny
            q_ghost[3] = q_int[3] - Float32(2.0) * mn * nz
        else:
            # BC_OUTFLOW (default): pure zero-gradient extrapolation.
            q_ghost[1] = q_int[1]
            q_ghost[2] = q_int[2]
            q_ghost[3] = q_int[3]
        var q_ghost_p = rebind[UnsafePointer[Float32, MutAnyOrigin]](
            q_ghost.unsafe_ptr()
        )
        return self.numerical_flux(q_int, q_ghost_p, nx, ny, nz, flux)

    # Gravitational source term:
    #   d(rho u_i)/dt += rho * g_i
    #   dE/dt       += rho * (u . g)
    # With the default (gx, gy, gz) = (0, 0, 0) this is a no-op that
    # the compiler elides; a stratified / gravity-driven problem
    # constructs Euler with a nonzero gravity vector.
    def source_term(
        self,
        q: UnsafePointer[Float32, MutAnyOrigin],
        x: Float32, y: Float32, z: Float32,
        source_out: UnsafePointer[Float32, MutAnyOrigin],
    ):
        var rho = q[0]
        var mx = q[1]
        var my = q[2]
        var mz = q[3]
        source_out[0] = Float32(0.0)
        source_out[1] = rho * self.gx
        source_out[2] = rho * self.gy
        source_out[3] = rho * self.gz
        source_out[4] = mx * self.gx + my * self.gy + mz * self.gz

    # Positivity-preserving floor limiter.  Called on every owned nodal
    # DOF at the end of each RK stage, before the next stage reads back.
    # If density drops below `min_density`, clamp it in place; if
    # pressure drops below `min_pressure`, raise the total energy so the
    # derived pressure hits the floor exactly.  This is a minimal
    # stabilization: it does NOT enforce monotonicity or TVD -- shocks
    # still oscillate -- but it keeps density/pressure positive so the
    # flux routines don't propagate NaN.  For classical Sod this is
    # enough to reach T >= 0.2 (unlimited DG NaNs around t ~ 0.15).
    def limit_state(
        self,
        q: UnsafePointer[Float32, MutAnyOrigin],
    ):
        # NaN-safe: `not (rho > floor)` catches both NaN and rho <= floor
        # (NaN comparisons always return False, so a naive `rho < floor`
        # would let NaN sail right past us).  On density or pressure
        # collapse we zero the momentum and reset energy to the ambient
        # pressure -- it's a crude rescue, but it keeps the integration
        # finite so downstream cells don't propagate NaN.
        var rho = q[0]
        var collapsed = not (rho > self.min_density)
        if collapsed:
            q[0] = self.min_density
            q[1] = Float32(0.0)
            q[2] = Float32(0.0)
            q[3] = Float32(0.0)
            q[4] = self.min_pressure / (self.gamma - Float32(1.0))
            return

        var mx = q[1]
        var my = q[2]
        var mz = q[3]
        # NaN momentum -> treat as collapsed, same rescue above.
        var m2 = mx * mx + my * my + mz * mz
        if not (m2 >= Float32(0.0)):
            q[1] = Float32(0.0)
            q[2] = Float32(0.0)
            q[3] = Float32(0.0)
            q[4] = self.min_pressure / (self.gamma - Float32(1.0))
            return

        var ke = Float32(0.5) * m2 / rho
        var E = q[4]
        var p = (self.gamma - Float32(1.0)) * (E - ke)
        # `not (p > floor)` again catches NaN energy.
        if not (p > self.min_pressure):
            q[4] = ke + self.min_pressure / (self.gamma - Float32(1.0))
