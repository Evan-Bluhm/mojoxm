# ======================================================================
# ssprk3_test -- unit-tests `src.ssprk3.ssprk3_stage_plans`
# ======================================================================
#
# The SSPRK3 stage-plan helper is the single non-templated 3-stage
# routing dance used by all 10 2D-GPU example drivers and ~47 2D
# benches.  It is currently validated only implicitly -- if a stage
# coefficient or buffer routing got swapped, the smooth-flow benches
# would catch it via tolerance violations, but the failure would
# point at the physics path, not the time integrator.  This file
# unit-tests the helper directly.
#
# Five checks:
#   (1) The helper returns exactly three stages.
#   (2) Buffer routing per stage matches the Shu-Osher recipe:
#         stage 0: q_in=q,  q_a=q,  q_b=q,  q_out=q1
#         stage 1: q_in=q1, q_a=q,  q_b=q1, q_out=q2
#         stage 2: q_in=q2, q_a=q,  q_b=q2, q_out=q
#   (3) Coefficients per stage match Gottlieb-Shu 1998:
#         stage 0: a=1.0,   b=0.0,   c=1.0
#         stage 1: a=0.75,  b=0.25,  c=0.25
#         stage 2: a=1/3,   b=2/3,   c=2/3
#   (4) Conservation invariant a + b == 1 in each stage -- the
#       strong-stability-preserving property (each stage state is a
#       convex combination of two SSP states).
#   (5) SSP-coefficient invariant c_k == b_k for stages where b_k > 0.
#       This is what makes Gottlieb-Shu SSPRK3 the OPTIMAL third-order
#       SSP scheme (CFL-equivalent to forward Euler).  Stage 0 has
#       b=0 so the relation is vacuous there.
#
# No GPU device is touched -- the helper is pure data plumbing.
# ======================================================================

from src.ssprk3 import ssprk3_stage_plans


comptime EPS: Float32 = Float32(1.0e-7)


def approx_eq(a: Float32, b: Float32) -> Bool:
    var d = a - b
    if d < Float32(0.0):
        d = -d
    return d <= EPS


def main() raises:
    print("ssprk3_test: unit-test ssprk3_stage_plans")

    # Three distinguishable fake pointers.  The helper never
    # dereferences these -- it just routes them between the
    # stage-plan struct fields.
    var p_q = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=0x1000)
    var p_q1 = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=0x2000)
    var p_q2 = UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=0x3000)

    var plans = ssprk3_stage_plans(d_q=p_q, d_q1=p_q1, d_q2=p_q2)

    # --- (1) length ---
    if len(plans) != 3:
        raise Error("ssprk3_test FAILED: expected 3 stages, got " + String(len(plans)))

    # --- (2) buffer routing per stage ---
    var s0 = plans[0].copy()
    if s0.q_in != p_q or s0.q_a != p_q or s0.q_b != p_q or s0.q_out != p_q1:
        raise Error("ssprk3_test FAILED: stage 0 routing wrong")

    var s1 = plans[1].copy()
    if s1.q_in != p_q1 or s1.q_a != p_q or s1.q_b != p_q1 or s1.q_out != p_q2:
        raise Error("ssprk3_test FAILED: stage 1 routing wrong")

    var s2 = plans[2].copy()
    if s2.q_in != p_q2 or s2.q_a != p_q or s2.q_b != p_q2 or s2.q_out != p_q:
        raise Error("ssprk3_test FAILED: stage 2 routing wrong")

    # --- (3) coefficients per stage ---
    if not (approx_eq(s0.a, Float32(1.0)) and approx_eq(s0.b, Float32(0.0)) and approx_eq(s0.c, Float32(1.0))):
        raise Error(
            "ssprk3_test FAILED: stage 0 coefficients wrong: a="
            + String(s0.a)
            + " b="
            + String(s0.b)
            + " c="
            + String(s0.c)
        )
    if not (approx_eq(s1.a, Float32(0.75)) and approx_eq(s1.b, Float32(0.25)) and approx_eq(s1.c, Float32(0.25))):
        raise Error(
            "ssprk3_test FAILED: stage 1 coefficients wrong: a="
            + String(s1.a)
            + " b="
            + String(s1.b)
            + " c="
            + String(s1.c)
        )
    var third = Float32(1.0 / 3.0)
    var two_thirds = Float32(2.0 / 3.0)
    if not (approx_eq(s2.a, third) and approx_eq(s2.b, two_thirds) and approx_eq(s2.c, two_thirds)):
        raise Error(
            "ssprk3_test FAILED: stage 2 coefficients wrong: a="
            + String(s2.a)
            + " b="
            + String(s2.b)
            + " c="
            + String(s2.c)
        )

    # --- Conservation invariant: a + b == 1 in each stage.  This is
    # what makes the scheme strong-stability-preserving (the next-
    # stage state is a convex combination of two SSP states).
    if not approx_eq(s0.a + s0.b, Float32(1.0)):
        raise Error("ssprk3_test FAILED: stage 0 a+b != 1")
    if not approx_eq(s1.a + s1.b, Float32(1.0)):
        raise Error("ssprk3_test FAILED: stage 1 a+b != 1")
    if not approx_eq(s2.a + s2.b, Float32(1.0)):
        raise Error("ssprk3_test FAILED: stage 2 a+b != 1")

    # --- SSP-coefficient invariant: c_k == b_k for stages where
    # b_k > 0.  This is the specific property that makes Gottlieb-Shu
    # SSPRK3 the OPTIMAL third-order SSP scheme (CFL-equivalent to
    # forward Euler).  Stage 0 has b=0 so the relation is vacuous
    # there (the dt*L(q) term is the only contribution).
    if s1.b > Float32(0.0) and not approx_eq(s1.c, s1.b):
        raise Error(
            "ssprk3_test FAILED: stage 1 c != b (broke SSPRK3 SSP-coefficient property): c="
            + String(s1.c)
            + " b="
            + String(s1.b)
        )
    if s2.b > Float32(0.0) and not approx_eq(s2.c, s2.b):
        raise Error(
            "ssprk3_test FAILED: stage 2 c != b (broke SSPRK3 SSP-coefficient property): c="
            + String(s2.c)
            + " b="
            + String(s2.b)
        )

    print("=== ssprk3_test PASSED ===")
