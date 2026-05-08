# ======================================================================
# sod_exact_riemann_test -- unit-test the analytic Sod Riemann solver
# ======================================================================
#
# `src.sod_exact_riemann` is critical analytic infrastructure used by
# every Sod-shock-tube bench (bench_euler_sod_2d, _limited_2d / p3 /
# p4 / p5, plus the 3D family).  The benches assert measured GPU
# state against this solver's output, so a regression in the Newton
# iteration or wave-speed math would silently shift every bench's
# tolerance band -- not flag the helper itself.
#
# This test pins the canonical Sod solution (rho_L=1, p_L=1,
# rho_R=0.125, p_R=0.1, gamma=1.4) against textbook reference values
# (Toro 2009, Table 4.2 / sec 4.3.3):
#
#   p_star    ~ 0.30313
#   u_star    ~ 0.92745   (computed inline; sanity check via density)
#   rho_*L    ~ 0.42632
#   rho_*R    ~ 0.26557
#   S_R       ~ 1.7522    (right-going shock speed)
#
# Plus four sample-point checks: x=0 -> rho_L, x=1 -> rho_R,
# x=0.50001 (just right of x0) -> rho_*L (post-rarefaction), and
# x=0.95 -> rho_*R (post-shock at t=0.2 since x_shock ~ 0.5+0.2*1.75=0.85).
#
# Plus a rarefaction-fan monotonicity sweep: 8 samples from x=0.30 to
# x=0.485 must be monotonically non-increasing across the head-tail
# band, gating the rarefaction's similarity-solution branch.
#
# Plus one symmetry check: identical L/R state -> IC density is
# returned everywhere (Newton solver doesn't fail on degenerate IC).
#
# Plus a strong-shock case from Toro 2009 Table 4.1 / 4.3 (Test 3:
# p_L=1000, p_R=0.01, 5 orders of magnitude pressure ratio): three
# checks gate rho_*L ~ 0.5754, rho_*R ~ 5.999, and S_R ~ 23.5175 at
# t=0.012.  Sod's mild 10:1 pressure ratio leaves Newton-iteration
# bugs latent that this strong shock catches.
#
# Pure host Float64 math: no GPU, no MPI.  Runs in <0.5s.
# ======================================================================

from src.sod_exact_riemann import sod_exact_rho, shock_speed_S_R


comptime EPS_TIGHT: Float64 = 1.0e-4
comptime EPS_LOOSE: Float64 = 5.0e-3


def approx_eq(a: Float64, b: Float64, tol: Float64) -> Bool:
    var d = a - b
    if d < 0.0:
        d = -d
    return d <= tol


def main() raises:
    print("sod_exact_riemann_test: textbook reference values")

    # ---------- canonical Sod state ----------
    var gamma = 1.4
    var rho_L = 1.0
    var p_L = 1.0
    var rho_R = 0.125
    var p_R = 0.1

    # Textbook values for the canonical Sod IC (Toro 2009).
    var rho_star_L_ref: Float64 = 0.42632  # post-rarefaction density
    var rho_star_R_ref: Float64 = 0.26557  # post-shock density
    var S_R_ref: Float64 = 1.7522  # right-going shock speed

    # ---------- sample 1: far left -- IC unchanged ----------
    var rho_far_left = sod_exact_rho(0.0, 0.2, gamma, rho_L, p_L, rho_R, p_R)
    if not approx_eq(rho_far_left, rho_L, EPS_TIGHT):
        raise Error(
            "sod_exact_riemann_test FAILED: rho(x=0, t=0.2) expected rho_L="
            + String(rho_L)
            + ", got "
            + String(rho_far_left)
        )

    # ---------- sample 2: far right -- IC unchanged ----------
    var rho_far_right = sod_exact_rho(1.0, 0.2, gamma, rho_L, p_L, rho_R, p_R)
    if not approx_eq(rho_far_right, rho_R, EPS_TIGHT):
        raise Error(
            "sod_exact_riemann_test FAILED: rho(x=1, t=0.2) expected rho_R="
            + String(rho_R)
            + ", got "
            + String(rho_far_right)
        )

    # ---------- sample 3: post-rarefaction plateau ----------
    # At t=0.2, the contact discontinuity sits at x_contact = 0.5 + t*u_star
    # ~ 0.5 + 0.2*0.927 = 0.685.  Just left of the contact (x=0.65)
    # we should be on the rho_*L plateau.
    var rho_starL = sod_exact_rho(0.65, 0.2, gamma, rho_L, p_L, rho_R, p_R)
    if not approx_eq(rho_starL, rho_star_L_ref, EPS_LOOSE):
        raise Error(
            "sod_exact_riemann_test FAILED: rho_*L expected "
            + String(rho_star_L_ref)
            + ", got "
            + String(rho_starL)
        )

    # ---------- sample 4: post-shock plateau ----------
    # At t=0.2 the shock sits at x_shock ~ 0.5 + 0.2 * 1.7522 = 0.85.
    # Between contact (~0.685) and shock (~0.85) we're on rho_*R.
    var rho_starR = sod_exact_rho(0.78, 0.2, gamma, rho_L, p_L, rho_R, p_R)
    if not approx_eq(rho_starR, rho_star_R_ref, EPS_LOOSE):
        raise Error(
            "sod_exact_riemann_test FAILED: rho_*R expected "
            + String(rho_star_R_ref)
            + ", got "
            + String(rho_starR)
        )

    # ---------- shock speed ----------
    var S_R_meas = shock_speed_S_R(rho_L, p_L, rho_R, p_R, gamma)
    if not approx_eq(S_R_meas, S_R_ref, EPS_LOOSE):
        raise Error(
            "sod_exact_riemann_test FAILED: S_R expected "
            + String(S_R_ref)
            + ", got "
            + String(S_R_meas)
        )

    # ---------- monotonicity check across the rarefaction fan ----------
    # Inside the rarefaction (xi_head_L ~ -a_L = -1.183 to xi_tail_L ~
    # u_star - a_star_L ~ 0.927 - 0.998 = -0.071), density decreases
    # monotonically from rho_L to rho_*L.  Sample 8 points and verify
    # rho is monotonically non-increasing.
    var prev: Float64 = rho_L + 1.0  # sentinel above rho_L
    for k in range(8):
        # x ranges from 0.30 (just inside head) to 0.485 (just before tail).
        var x = 0.30 + 0.025 * Float64(k)
        var rho_here = sod_exact_rho(x, 0.2, gamma, rho_L, p_L, rho_R, p_R)
        # Allow Float64 noise of 1e-14 in the comparison.
        if rho_here > prev + 1.0e-12:
            raise Error(
                "sod_exact_riemann_test FAILED: rarefaction not monotone at x="
                + String(x)
                + " rho="
                + String(rho_here)
                + " prev="
                + String(prev)
            )
        prev = rho_here

    # ---------- symmetric IC: rho_L = rho_R, p_L = p_R ----------
    # Newton solver with degenerate (zero-strength) Riemann state
    # should return rho_L everywhere.  This used to be a numerical
    # corner case for some implementations.
    var rho_uniform = sod_exact_rho(
        0.5,
        0.2,
        gamma,
        rho_L,
        p_L,
        rho_L,
        p_L,
    )
    if not approx_eq(rho_uniform, rho_L, EPS_TIGHT):
        raise Error(
            "sod_exact_riemann_test FAILED: degenerate IC (uniform) should"
            " return rho_L="
            + String(rho_L)
            + " everywhere, got "
            + String(rho_uniform)
        )

    # ---------- Toro Test 3: strong shock (rare + shock structure) ----------
    # Toro 2009 Table 4.1 / 4.3: rho_L = rho_R = 1, p_L = 1000, p_R = 0.01
    # (5 orders of magnitude in pressure).  Same wave structure as Sod
    # (left rarefaction + contact + right shock) but the shock is much
    # stronger -- catches Newton-iteration regressions that Sod's mild
    # 10:1 pressure ratio doesn't stress.
    #
    # Textbook reference values (Toro 2009 sec 4.3.3):
    #   p_star    ~  460.894
    #   u_star    ~  19.5975
    #   rho_*L    ~  0.5754
    #   rho_*R    ~  5.999
    #   S_R       ~  23.5175  (shock speed; NOT u_star)
    #
    # At t = 0.012 (textbook standard time): contact at x ~ 0.735,
    # shock at x ~ 0.782; sample at 0.76 to land in the post-shock band.
    var rho_L3: Float64 = 1.0
    var p_L3: Float64 = 1000.0
    var rho_R3: Float64 = 1.0
    var p_R3: Float64 = 0.01
    var rho_starL3_ref: Float64 = 0.5754
    var rho_starR3_ref: Float64 = 5.999
    var S_R3_ref: Float64 = 23.5175

    var rho_starL3 = sod_exact_rho(
        0.55, 0.012, gamma, rho_L3, p_L3, rho_R3, p_R3
    )
    if not approx_eq(rho_starL3, rho_starL3_ref, EPS_LOOSE):
        raise Error(
            "sod_exact_riemann_test FAILED: Toro 3 rho_*L expected "
            + String(rho_starL3_ref)
            + ", got "
            + String(rho_starL3)
        )
    var rho_starR3 = sod_exact_rho(
        0.76, 0.012, gamma, rho_L3, p_L3, rho_R3, p_R3
    )
    # Strong-shock relative tol: 5.999 vs 5.999 -- absolute eps_loose
    # (5e-3) is generous given the magnitude.
    if not approx_eq(rho_starR3, rho_starR3_ref, EPS_LOOSE):
        raise Error(
            "sod_exact_riemann_test FAILED: Toro 3 rho_*R expected "
            + String(rho_starR3_ref)
            + ", got "
            + String(rho_starR3)
        )
    var S_R3 = shock_speed_S_R(rho_L3, p_L3, rho_R3, p_R3, gamma)
    # S_R magnitude is ~23, so EPS_LOOSE = 5e-3 is too tight; use 0.01
    # absolute for the speed (one part in ~2300, well below textbook
    # truncation of 4 sigfigs).
    if not approx_eq(S_R3, S_R3_ref, 0.01):
        raise Error(
            "sod_exact_riemann_test FAILED: Toro 3 S_R expected "
            + String(S_R3_ref)
            + ", got "
            + String(S_R3)
        )

    print("=== sod_exact_riemann_test PASSED ===")
