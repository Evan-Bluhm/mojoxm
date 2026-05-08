# ======================================================================
# sod_exact_riemann.mojo -- analytic Riemann solver for the Sod problem
# ======================================================================
#
# Toro-style two-rarefaction-then-Newton exact Riemann solver, used by
# the Sod-shock-tube benchmarks to compare GPU output against the
# closed-form analytic profile at a given (x, t).
#
# Lives in `src/` (rather than `benchmarks/`) so it's importable by
# any number of bench files via `from src.sod_exact_riemann import ...`
# and not duplicated -- the four `bench_euler_sod*_2d*` files all
# need the same machinery.
#
# Pure host-side Float64 math.  Called once per bench at the end to
# build the analytic reference profile; perf is irrelevant.
#
# Typical use:
#
#   from src.sod_exact_riemann import sod_exact_rho
#   var rho_ref = sod_exact_rho(
#       0.85, T_FINAL, GAMMA, RHO_L, P_L, RHO_R, P_R,
#   )
# ======================================================================

from std.math import sqrt, pow


def f_K(p: Float64, rho_K: Float64, p_K: Float64, a_K: Float64, gamma: Float64) -> Float64:
    """Toro's f_K: pressure response for the wave on side K.
    Shock branch when p > p_K, rarefaction otherwise."""
    if p > p_K:
        var A = 2.0 / ((gamma + 1.0) * rho_K)
        var B = (gamma - 1.0) / (gamma + 1.0) * p_K
        return (p - p_K) * sqrt(A / (p + B))
    else:
        var e = (gamma - 1.0) / (2.0 * gamma)
        return (2.0 * a_K / (gamma - 1.0)) * (pow(p / p_K, e) - 1.0)


def solve_star(rho_L: Float64, u_L: Float64, p_L: Float64, a_L: Float64, rho_R: Float64, u_R: Float64, p_R: Float64, a_R: Float64, gamma: Float64) raises -> Float64:
    """Newton solve for p_star on f_L(p) + f_R(p) + (u_R - u_L) = 0.
    Initial guess is the two-rarefaction approximation; falls back to
    (p_L + p_R) / 2 if that goes negative.  Tolerance 1e-10 (relative)
    in 50 iterations -- always converges for the canonical Sod state.
    Raises if non-convergent (shouldn't on physical inputs)."""
    var e = (gamma - 1.0) / (2.0 * gamma)
    var p_tr = pow((a_L + a_R - 0.5 * (gamma - 1.0) * (u_R - u_L)) / (a_L / pow(p_L, e) + a_R / pow(p_R, e)), 1.0 / e)
    var p = p_tr if p_tr > 0.0 else 0.5 * (p_L + p_R)
    var tol: Float64 = 1.0e-10
    for _ in range(50):
        var fL = f_K(p, rho_L, p_L, a_L, gamma)
        var fR = f_K(p, rho_R, p_R, a_R, gamma)
        var dfL: Float64
        var dfR: Float64
        if p > p_L:
            var A_L = 2.0 / ((gamma + 1.0) * rho_L)
            var B_L = (gamma - 1.0) / (gamma + 1.0) * p_L
            var sL = sqrt(A_L / (p + B_L))
            dfL = sL * (1.0 - 0.5 * (p - p_L) / (p + B_L))
        else:
            dfL = (1.0 / (rho_L * a_L)) * pow(p / p_L, -(gamma + 1.0) / (2.0 * gamma))
        if p > p_R:
            var A_R = 2.0 / ((gamma + 1.0) * rho_R)
            var B_R = (gamma - 1.0) / (gamma + 1.0) * p_R
            var sR = sqrt(A_R / (p + B_R))
            dfR = sR * (1.0 - 0.5 * (p - p_R) / (p + B_R))
        else:
            dfR = (1.0 / (rho_R * a_R)) * pow(p / p_R, -(gamma + 1.0) / (2.0 * gamma))
        var resid = fL + fR + (u_R - u_L)
        var dp = -resid / (dfL + dfR)
        var p_new = p + dp
        if p_new <= 0.0:
            p_new = 0.5 * p
        var rel = 2.0 * (p_new - p) / (p_new + p)
        var arel = rel if rel >= 0.0 else -rel
        p = p_new
        if arel < tol:
            return p
    raise Error("Sod Newton did not converge")


def sod_exact_rho(x: Float64, t: Float64, gamma: Float64, rho_L: Float64, p_L: Float64, rho_R: Float64, p_R: Float64) raises -> Float64:
    """Returns rho at (x, t) for the canonical Sod shock tube
    (u_L = u_R = 0, discontinuity at x = 0.5).  Composes the
    rarefaction fan + contact + shock from the standard Riemann
    structure.  Generalised over (gamma, rho_L, p_L, rho_R, p_R)
    so the same helper works for any L/R state with u_L = u_R = 0."""
    var xi = (x - 0.5) / t  # self-similar variable
    var a_L = sqrt(gamma * p_L / rho_L)
    var a_R = sqrt(gamma * p_R / rho_R)
    var p_star = solve_star(rho_L, 0.0, p_L, a_L, rho_R, 0.0, p_R, a_R, gamma)
    var u_star = 0.5 * (f_K(p_star, rho_R, p_R, a_R, gamma) - f_K(p_star, rho_L, p_L, a_L, gamma))

    # Left wave: rarefaction (since p_star < p_L for the canonical IC).
    var rho_star_L = rho_L * pow(p_star / p_L, 1.0 / gamma)
    var a_star_L = a_L * pow(p_star / p_L, (gamma - 1.0) / (2.0 * gamma))
    var xi_head_L = -a_L  # u_L - a_L, u_L = 0
    var xi_tail_L = u_star - a_star_L

    # Right wave: shock (since p_star > p_R for the canonical IC).
    var rho_star_R = rho_R * ((p_star / p_R + (gamma - 1.0) / (gamma + 1.0)) / ((gamma - 1.0) / (gamma + 1.0) * p_star / p_R + 1.0))
    var S_R = a_R * sqrt((gamma + 1.0) / (2.0 * gamma) * p_star / p_R + (gamma - 1.0) / (2.0 * gamma))  # u_R = 0

    if xi < xi_head_L:
        return rho_L
    elif xi < xi_tail_L:
        # Inside left rarefaction fan: isentropic relation rho ~ a^(2/(g-1)).
        var v = 2.0 / (gamma + 1.0) * (a_L + xi)
        var a = a_L - 0.5 * (gamma - 1.0) * v
        return rho_L * pow(a / a_L, 2.0 / (gamma - 1.0))
    elif xi < u_star:
        return rho_star_L
    elif xi < S_R:
        return rho_star_R
    return rho_R


def shock_speed_S_R(rho_L: Float64, p_L: Float64, rho_R: Float64, p_R: Float64, gamma: Float64) raises -> Float64:
    """Returns the right-going shock speed S_R for the canonical Sod
    state (u_L = u_R = 0).  Matches the wave used in `sod_exact_rho`,
    extracted as a standalone helper since the bench files compare
    measured shock-front cell index against
    `x_shock = 0.5 + t * S_R`."""
    var a_L = sqrt(gamma * p_L / rho_L)
    var a_R = sqrt(gamma * p_R / rho_R)
    var p_star = solve_star(rho_L, 0.0, p_L, a_L, rho_R, 0.0, p_R, a_R, gamma)
    return a_R * sqrt((gamma + 1.0) / (2.0 * gamma) * p_star / p_R + (gamma - 1.0) / (2.0 * gamma))
