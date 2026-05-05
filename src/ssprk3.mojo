# ======================================================================
# SSPRK3 stage plan -- non-templated time-stepping coefficient helper
# ======================================================================
#
# The 2D example drivers each embed a 30-line SSPRK3 inner-step loop
# that hand-writes the three Shu-Osher stages with their (q_in, q_a,
# q_b, q_out, a, b, c) routings.  Every driver does the exact same
# coefficient and buffer dance; only the per-physics RK kernel
# differs.  This file factors out just that data into a uniform
# shape so a driver step loop becomes:
#
#     for stage in ssprk3_stage_plans(
#         d_q.unsafe_ptr(), d_q1.unsafe_ptr(), d_q2.unsafe_ptr(),
#     ):
#         my_physics_rk_stage_2d[P](
#             ctx, gpu_mesh,
#             gpu_re.d_Lift_ref.unsafe_ptr(),
#             gpu_re.d_D_ref.unsafe_ptr(),
#             stage.q_in, stage.q_a, stage.q_b, stage.q_out,
#             d_fstar.unsafe_ptr(),
#             # ...physics-specific args...
#             stage.a, stage.b, stage.c, dt,
#         )
#
# The helper itself is NOT templated over kernel type and is NOT a
# closure-taking wrapper -- it returns a `List[SSPRK3StagePlan]` of
# raw pointers + Float32 coefficients.  That keeps the helper a
# single compiled definition with zero per-physics instantiation
# overhead, regardless of how many drivers use it.
#
# Three Shu-Osher stages of the standard SSPRK3 (Gottlieb &
# Shu 1998) -- the same scheme used by `Solver.step_ssprk3` on the
# 3D side, transcribed verbatim:
#
#   q1 = q                       + dt * L(q)
#   q2 = (3/4) q + (1/4) q1      + (1/4) dt * L(q1)
#   q  = (1/3) q + (2/3) q2      + (2/3) dt * L(q2)
# ======================================================================


@fieldwise_init
struct SSPRK3StagePlan(Copyable, Movable):
    """One Shu-Osher stage's buffer routing + accumulation triple.
    Drivers receive these from `ssprk3_stage_plans()` and pass them
    straight into the physics-specific RK-stage kernel."""
    var q_in:  UnsafePointer[Float32, MutAnyOrigin]
    var q_a:   UnsafePointer[Float32, MutAnyOrigin]
    var q_b:   UnsafePointer[Float32, MutAnyOrigin]
    var q_out: UnsafePointer[Float32, MutAnyOrigin]
    var a: Float32
    var b: Float32
    var c: Float32


def ssprk3_stage_plans(
    d_q:  UnsafePointer[Float32, MutAnyOrigin],
    d_q1: UnsafePointer[Float32, MutAnyOrigin],
    d_q2: UnsafePointer[Float32, MutAnyOrigin],
) raises -> List[SSPRK3StagePlan]:
    """Return the three SSPRK3 stages' (q_in, q_a, q_b, q_out, a, b,
    c) routings for the standard d_q / d_q1 / d_q2 buffer trio.  Use
    in a step-inner loop:

        for stage in ssprk3_stage_plans(d_q_p, d_q1_p, d_q2_p):
            <physics>_rk_stage_2d[P](
                ..., stage.q_in, stage.q_a, stage.q_b, stage.q_out,
                ..., stage.a, stage.b, stage.c, dt,
            )

    The function is not templated over kernel type -- the returned
    plans hold raw `UnsafePointer[Float32, MutAnyOrigin]`, so one
    compiled body serves every 2D physics (Advection / Euler / SW /
    Maxwell / IdealMHD / GLM-MHD) and every P."""
    var plans = List[SSPRK3StagePlan]()
    # Stage 1: q1 = q + dt * L(q)
    plans.append(SSPRK3StagePlan(
        q_in=d_q,  q_a=d_q,  q_b=d_q,  q_out=d_q1,
        a=Float32(1.0), b=Float32(0.0), c=Float32(1.0),
    ))
    # Stage 2: q2 = (3/4)*q + (1/4)*(q1 + dt*L(q1))
    plans.append(SSPRK3StagePlan(
        q_in=d_q1, q_a=d_q,  q_b=d_q1, q_out=d_q2,
        a=Float32(0.75), b=Float32(0.25), c=Float32(0.25),
    ))
    # Stage 3: q  = (1/3)*q + (2/3)*(q2 + dt*L(q2))
    plans.append(SSPRK3StagePlan(
        q_in=d_q2, q_a=d_q,  q_b=d_q2, q_out=d_q,
        a=Float32(1.0 / 3.0),
        b=Float32(2.0 / 3.0),
        c=Float32(2.0 / 3.0),
    ))
    return plans^
