# ======================================================================
# local_mesh_2d_gpu_euler.mojo -- 2D Euler (NC=4) GPU kernels
# ======================================================================
# State (rho, mx=rho*u, my=rho*v, E).  Two driver entry points, one
# per Riemann solver:
#   * `euler_rk_stage_2d[P]`        -- Rusanov flux
#   * `euler_rk_stage_hllc_2d[P]`   -- HLLC flux (Toro 1994)
# Each stage is 2 launches: face-flux kernel writes fstar to global,
# then a per-(elem, node) fused kernel computes the volume RHS
# locally, applies the face-lift, and does the RK update.  The
# vol+lift+RK kernel is flux-independent (the only difference
# between the Rusanov and HLLC paths is which face-flux kernel runs).
# ======================================================================

from src.local_mesh_2d_gpu import LocalMesh2DGpu
from src.reference_2d import num_tri_nodes_2d, num_edge_nodes
from src.boundary import BC_INTERIOR, BC_WALL, BC_INFLOW
from std.gpu import global_idx
from std.gpu.host import DeviceContext
from std.math import ceildiv, sqrt


# ----------------------------------------------------------------------
# Fused Euler volume + lift + RK update (2D, NC=4).
# ----------------------------------------------------------------------
# Per-(elem, node) thread.  Same fusion pattern as
# advection_vol_lift_combine_rk_kernel_2d but for the 4-component
# Euler state.  Eliminates the global-memory round-trip through vol_c
# and one of the three kernel launches per RK stage.
# ----------------------------------------------------------------------

def euler_vol_lift_combine_rk_kernel_2d[NP: Int, NFP: Int](
    q_in:              UnsafePointer[Float32, MutAnyOrigin],
    elem_invJ:         UnsafePointer[Float32, MutAnyOrigin],
    D_ref:             UnsafePointer[Float32, MutAnyOrigin],
    fstar:             UnsafePointer[Float32, MutAnyOrigin],
    elem_inv_2A:       UnsafePointer[Float32, MutAnyOrigin],
    elem_faces:        UnsafePointer[Int32,   MutAnyOrigin],
    elem_face_side:    UnsafePointer[Int32,   MutAnyOrigin],
    elem_canon_to_ref: UnsafePointer[Int32,   MutAnyOrigin],
    face_length:       UnsafePointer[Float32, MutAnyOrigin],
    Lift_ref:          UnsafePointer[Float32, MutAnyOrigin],
    q_a:               UnsafePointer[Float32, MutAnyOrigin],
    q_b:               UnsafePointer[Float32, MutAnyOrigin],
    num_elements:      Int,
    gamma:             Float32,
    min_density:       Float32,
    min_pressure:      Float32,
    gx:                Float32,
    gy:                Float32,
    a: Float32, b: Float32, cc: Float32, dt: Float32,
    q_out:             UnsafePointer[Float32, MutAnyOrigin],
):
    var tid = Int(global_idx.x)
    var total = num_elements * NP
    if tid >= total:
        return
    var elem = tid // NP
    var i    = tid %  NP

    # ---- Volume RHS contribution (4 components, computed locally).
    var iJ00 = elem_invJ[elem * 4 + 0]
    var iJ01 = elem_invJ[elem * 4 + 1]
    var iJ10 = elem_invJ[elem * 4 + 2]
    var iJ11 = elem_invJ[elem * 4 + 3]
    var acc0: Float32 = 0.0
    var acc1: Float32 = 0.0
    var acc2: Float32 = 0.0
    var acc3: Float32 = 0.0
    for j in range(NP):
        var base = (elem * NP + j) * 4
        var rho = q_in[base + 0]
        if rho < min_density:
            rho = min_density
        var mx = q_in[base + 1]
        var my = q_in[base + 2]
        var E  = q_in[base + 3]
        var u = mx / rho
        var v = my / rho
        var ke = Float32(0.5) * rho * (u * u + v * v)
        var p = (gamma - Float32(1.0)) * (E - ke)
        if p < min_pressure:
            p = min_pressure
        var Fx0 = mx
        var Fx1 = mx * u + p
        var Fx2 = mx * v
        var Fx3 = u * (E + p)
        var Fy0 = my
        var Fy1 = my * u
        var Fy2 = my * v + p
        var Fy3 = v * (E + p)
        var D_r = D_ref[0 * NP * NP + i * NP + j]
        var D_s = D_ref[1 * NP * NP + i * NP + j]
        acc0 += (iJ00 * Fx0 + iJ01 * Fy0) * D_r + (iJ10 * Fx0 + iJ11 * Fy0) * D_s
        acc1 += (iJ00 * Fx1 + iJ01 * Fy1) * D_r + (iJ10 * Fx1 + iJ11 * Fy1) * D_s
        acc2 += (iJ00 * Fx2 + iJ01 * Fy2) * D_r + (iJ10 * Fx2 + iJ11 * Fy2) * D_s
        acc3 += (iJ00 * Fx3 + iJ01 * Fy3) * D_r + (iJ10 * Fx3 + iJ11 * Fy3) * D_s

    # ---- Lift contribution (NC=4, expanded inline).
    var inv_2A = elem_inv_2A[elem]
    var face0: Float32 = 0.0
    var face1: Float32 = 0.0
    var face2: Float32 = 0.0
    var face3: Float32 = 0.0
    for lf in range(3):
        var fid = Int(elem_faces[elem * 3 + lf])
        var side = Int(elem_face_side[elem * 3 + lf])
        var sign: Float32 = Float32(1.0) if side == 0 else Float32(-1.0)
        var flen = face_length[fid]
        for m in range(NFP):
            var r = Int(
                elem_canon_to_ref[(elem * 3 + lf) * NFP + m]
            )
            var Lim = Lift_ref[lf * NP * NFP + i * NFP + r]
            var sLf = sign * flen * Lim
            var fbase = (fid * NFP + m) * 4
            face0 += sLf * fstar[fbase + 0]
            face1 += sLf * fstar[fbase + 1]
            face2 += sLf * fstar[fbase + 2]
            face3 += sLf * fstar[fbase + 3]

    # ---- Combine + RK update for all 4 components.  Gravity source
    # (matching 3D Euler.source_term: d(rho u_i)/dt += rho * g_i,
    # dE/dt += rho * (u . g)) is read at node i and added to the
    # momentum / energy RHSes.  With default gx = gy = 0 this is a
    # no-op the compiler elides.
    var idx = (elem * NP + i) * 4
    var i_base = (elem * NP + i) * 4
    var i_rho = q_in[i_base + 0]
    if i_rho < min_density:
        i_rho = min_density
    var i_mx = q_in[i_base + 1]
    var i_my = q_in[i_base + 2]
    var rhs0 = acc0 - inv_2A * face0
    var rhs1 = acc1 - inv_2A * face1 + i_rho * gx
    var rhs2 = acc2 - inv_2A * face2 + i_rho * gy
    var rhs3 = acc3 - inv_2A * face3 + i_mx * gx + i_my * gy
    q_out[idx + 0] = a * q_a[idx + 0] + b * q_b[idx + 0] + cc * dt * rhs0
    q_out[idx + 1] = a * q_a[idx + 1] + b * q_b[idx + 1] + cc * dt * rhs1
    q_out[idx + 2] = a * q_a[idx + 2] + b * q_b[idx + 2] + cc * dt * rhs2
    q_out[idx + 3] = a * q_a[idx + 3] + b * q_b[idx + 3] + cc * dt * rhs3


def launch_euler_vol_lift_2d[NP: Int, NFP: Int](
    mut ctx: DeviceContext,
    q_in:              UnsafePointer[Float32, MutAnyOrigin],
    elem_invJ:         UnsafePointer[Float32, MutAnyOrigin],
    D_ref:             UnsafePointer[Float32, MutAnyOrigin],
    fstar:             UnsafePointer[Float32, MutAnyOrigin],
    elem_inv_2A:       UnsafePointer[Float32, MutAnyOrigin],
    elem_faces:        UnsafePointer[Int32,   MutAnyOrigin],
    elem_face_side:    UnsafePointer[Int32,   MutAnyOrigin],
    elem_canon_to_ref: UnsafePointer[Int32,   MutAnyOrigin],
    face_length:       UnsafePointer[Float32, MutAnyOrigin],
    Lift_ref:          UnsafePointer[Float32, MutAnyOrigin],
    q_a:               UnsafePointer[Float32, MutAnyOrigin],
    q_b:               UnsafePointer[Float32, MutAnyOrigin],
    num_elements:      Int,
    gamma:             Float32,
    min_density:       Float32,
    min_pressure:      Float32,
    gx:                Float32,
    gy:                Float32,
    a: Float32, b: Float32, cc: Float32, dt: Float32,
    q_out:             UnsafePointer[Float32, MutAnyOrigin],
) raises:
    var total = num_elements * NP
    comptime _kernel = euler_vol_lift_combine_rk_kernel_2d[NP, NFP]
    ctx.enqueue_function[_kernel, _kernel](
        q_in, elem_invJ, D_ref, fstar,
        elem_inv_2A, elem_faces, elem_face_side, elem_canon_to_ref,
        face_length, Lift_ref, q_a, q_b,
        num_elements, gamma, min_density, min_pressure,
        gx, gy,
        a, b, cc, dt, q_out,
        grid_dim=ceildiv(total, 256),
        block_dim=256,
    )


# ----------------------------------------------------------------------
# Euler2D face flux (Rusanov / Lax-Friedrichs, 4 components).
# ----------------------------------------------------------------------
# Computes fstar[fid, m, 0..3] at every face-local slot using the
# Rusanov flux F* = 0.5(F_L.n + F_R.n) - 0.5 alpha (q_R - q_L) with
# alpha = max(|v.n|+c) over the two sides.  Ghost state for non-interior
# faces mirrors the CPU Euler2D.boundary_flux:
#   BC_WALL    -> reflect normal momentum
#   BC_INFLOW  -> user-set (inflow_rho, inflow_rhou, inflow_rhov, inflow_E)
#   BC_OUTFLOW -> zero-gradient ghost (q_g = q_int)
# One thread per (face, slot) = num_faces * NFP threads.
# ----------------------------------------------------------------------

def euler_face_flux_kernel_2d[NP: Int, NFP: Int](
    q:              UnsafePointer[Float32, MutAnyOrigin],
    face_elem:      UnsafePointer[Int32,   MutAnyOrigin],
    face_elem_node: UnsafePointer[Int32,   MutAnyOrigin],
    face_normal:    UnsafePointer[Float32, MutAnyOrigin],
    face_bc_type:   UnsafePointer[Int32,   MutAnyOrigin],
    num_faces:      Int,
    gamma:          Float32,
    min_density:    Float32,
    min_pressure:   Float32,
    inflow_rho:     Float32,
    inflow_rhou:    Float32,
    inflow_rhov:    Float32,
    inflow_E:       Float32,
    fstar_out:      UnsafePointer[Float32, MutAnyOrigin],
):
    var tid = Int(global_idx.x)
    var total = num_faces * NFP
    if tid >= total:
        return
    var fid = tid // NFP
    var m   = tid %  NFP

    var nx = face_normal[fid * 2 + 0]
    var ny = face_normal[fid * 2 + 1]
    var bc_type = face_bc_type[fid]

    var e_l = Int(face_elem[fid * 2 + 0])
    var n_l = Int(face_elem_node[(fid * 2 + 0) * NFP + m])
    var l_off = (e_l * NP + n_l) * 4
    var qL0 = q[l_off + 0]
    var qL1 = q[l_off + 1]
    var qL2 = q[l_off + 2]
    var qL3 = q[l_off + 3]

    var qR0: Float32
    var qR1: Float32
    var qR2: Float32
    var qR3: Float32
    if bc_type == BC_INTERIOR:
        var e_r = Int(face_elem[fid * 2 + 1])
        var n_r = Int(face_elem_node[(fid * 2 + 1) * NFP + m])
        var r_off = (e_r * NP + n_r) * 4
        qR0 = q[r_off + 0]
        qR1 = q[r_off + 1]
        qR2 = q[r_off + 2]
        qR3 = q[r_off + 3]
    elif bc_type == BC_WALL:
        var m_n = qL1 * nx + qL2 * ny
        qR0 = qL0
        qR1 = qL1 - Float32(2.0) * m_n * nx
        qR2 = qL2 - Float32(2.0) * m_n * ny
        qR3 = qL3
    elif bc_type == BC_INFLOW:
        qR0 = inflow_rho
        qR1 = inflow_rhou
        qR2 = inflow_rhov
        qR3 = inflow_E
    else:
        # BC_OUTFLOW default: zero-gradient
        qR0 = qL0
        qR1 = qL1
        qR2 = qL2
        qR3 = qL3

    # Left internal flux + wave speed.
    var rhoL = qL0
    if rhoL < min_density:
        rhoL = min_density
    var uL = qL1 / rhoL
    var vL = qL2 / rhoL
    var keL = Float32(0.5) * rhoL * (uL * uL + vL * vL)
    var pL = (gamma - Float32(1.0)) * (qL3 - keL)
    if pL < min_pressure:
        pL = min_pressure
    var cL = sqrt(gamma * pL / rhoL)
    var speedL = sqrt(uL * uL + vL * vL) + cL

    var FxL0 = qL1
    var FxL1 = qL1 * uL + pL
    var FxL2 = qL1 * vL
    var FxL3 = uL * (qL3 + pL)
    var FyL0 = qL2
    var FyL1 = qL2 * uL
    var FyL2 = qL2 * vL + pL
    var FyL3 = vL * (qL3 + pL)

    # Right internal flux + wave speed.
    var rhoR = qR0
    if rhoR < min_density:
        rhoR = min_density
    var uR = qR1 / rhoR
    var vR = qR2 / rhoR
    var keR = Float32(0.5) * rhoR * (uR * uR + vR * vR)
    var pR = (gamma - Float32(1.0)) * (qR3 - keR)
    if pR < min_pressure:
        pR = min_pressure
    var cR = sqrt(gamma * pR / rhoR)
    var speedR = sqrt(uR * uR + vR * vR) + cR

    var FxR0 = qR1
    var FxR1 = qR1 * uR + pR
    var FxR2 = qR1 * vR
    var FxR3 = uR * (qR3 + pR)
    var FyR0 = qR2
    var FyR1 = qR2 * uR
    var FyR2 = qR2 * vR + pR
    var FyR3 = vR * (qR3 + pR)

    var alpha: Float32 = speedL if speedL > speedR else speedR
    var half = Float32(0.5)
    var out = (fid * NFP + m) * 4
    fstar_out[out + 0] = half * ((FxL0 + FxR0) * nx + (FyL0 + FyR0) * ny) \
                         - half * alpha * (qR0 - qL0)
    fstar_out[out + 1] = half * ((FxL1 + FxR1) * nx + (FyL1 + FyR1) * ny) \
                         - half * alpha * (qR1 - qL1)
    fstar_out[out + 2] = half * ((FxL2 + FxR2) * nx + (FyL2 + FyR2) * ny) \
                         - half * alpha * (qR2 - qL2)
    fstar_out[out + 3] = half * ((FxL3 + FxR3) * nx + (FyL3 + FyR3) * ny) \
                         - half * alpha * (qR3 - qL3)


def launch_euler_face_flux_2d[NP: Int, NFP: Int](
    mut ctx: DeviceContext,
    q:              UnsafePointer[Float32, MutAnyOrigin],
    face_elem:      UnsafePointer[Int32,   MutAnyOrigin],
    face_elem_node: UnsafePointer[Int32,   MutAnyOrigin],
    face_normal:    UnsafePointer[Float32, MutAnyOrigin],
    face_bc_type:   UnsafePointer[Int32,   MutAnyOrigin],
    num_faces:      Int,
    gamma:          Float32,
    min_density:    Float32,
    min_pressure:   Float32,
    inflow_rho:     Float32,
    inflow_rhou:    Float32,
    inflow_rhov:    Float32,
    inflow_E:       Float32,
    fstar_out:      UnsafePointer[Float32, MutAnyOrigin],
) raises:
    var total = num_faces * NFP
    comptime _kernel = euler_face_flux_kernel_2d[NP, NFP]
    ctx.enqueue_function[_kernel, _kernel](
        q, face_elem, face_elem_node, face_normal, face_bc_type,
        num_faces,
        gamma, min_density, min_pressure,
        inflow_rho, inflow_rhou, inflow_rhov, inflow_E,
        fstar_out,
        grid_dim=ceildiv(total, 256),
        block_dim=256,
    )


# ----------------------------------------------------------------------
# Euler2D HLLC face flux (Toro 1994, 4 components).
# ----------------------------------------------------------------------
# Same signature as the Rusanov kernel; resolves contact discontinuities
# exactly via the three-wave Riemann structure with the middle contact
# speed S_star derived from pressure continuity.  Ghost-state plumbing
# (BC_WALL / BC_INFLOW / BC_OUTFLOW) is identical to the Rusanov kernel
# -- only the interior flux differs.
# ----------------------------------------------------------------------

def euler_face_flux_hllc_kernel_2d[NP: Int, NFP: Int](
    q:              UnsafePointer[Float32, MutAnyOrigin],
    face_elem:      UnsafePointer[Int32,   MutAnyOrigin],
    face_elem_node: UnsafePointer[Int32,   MutAnyOrigin],
    face_normal:    UnsafePointer[Float32, MutAnyOrigin],
    face_bc_type:   UnsafePointer[Int32,   MutAnyOrigin],
    num_faces:      Int,
    gamma:          Float32,
    min_density:    Float32,
    min_pressure:   Float32,
    inflow_rho:     Float32,
    inflow_rhou:    Float32,
    inflow_rhov:    Float32,
    inflow_E:       Float32,
    fstar_out:      UnsafePointer[Float32, MutAnyOrigin],
):
    var tid = Int(global_idx.x)
    var total = num_faces * NFP
    if tid >= total:
        return
    var fid = tid // NFP
    var m   = tid %  NFP

    var nx = face_normal[fid * 2 + 0]
    var ny = face_normal[fid * 2 + 1]
    var bc_type = face_bc_type[fid]

    var e_l = Int(face_elem[fid * 2 + 0])
    var n_l = Int(face_elem_node[(fid * 2 + 0) * NFP + m])
    var l_off = (e_l * NP + n_l) * 4
    var qL0 = q[l_off + 0]
    var qL1 = q[l_off + 1]
    var qL2 = q[l_off + 2]
    var qL3 = q[l_off + 3]

    var qR0: Float32
    var qR1: Float32
    var qR2: Float32
    var qR3: Float32
    if bc_type == BC_INTERIOR:
        var e_r = Int(face_elem[fid * 2 + 1])
        var n_r = Int(face_elem_node[(fid * 2 + 1) * NFP + m])
        var r_off = (e_r * NP + n_r) * 4
        qR0 = q[r_off + 0]
        qR1 = q[r_off + 1]
        qR2 = q[r_off + 2]
        qR3 = q[r_off + 3]
    elif bc_type == BC_WALL:
        var m_n = qL1 * nx + qL2 * ny
        qR0 = qL0
        qR1 = qL1 - Float32(2.0) * m_n * nx
        qR2 = qL2 - Float32(2.0) * m_n * ny
        qR3 = qL3
    elif bc_type == BC_INFLOW:
        qR0 = inflow_rho
        qR1 = inflow_rhou
        qR2 = inflow_rhov
        qR3 = inflow_E
    else:
        qR0 = qL0
        qR1 = qL1
        qR2 = qL2
        qR3 = qL3

    # Primitives on each side (with floors).
    var rho_L = qL0
    if rho_L < min_density: rho_L = min_density
    var rho_R = qR0
    if rho_R < min_density: rho_R = min_density
    var uL = qL1 / rho_L
    var vL = qL2 / rho_L
    var uR = qR1 / rho_R
    var vR = qR2 / rho_R
    var keL = Float32(0.5) * rho_L * (uL * uL + vL * vL)
    var keR = Float32(0.5) * rho_R * (uR * uR + vR * vR)
    var pL = (gamma - Float32(1.0)) * (qL3 - keL)
    if pL < min_pressure: pL = min_pressure
    var pR = (gamma - Float32(1.0)) * (qR3 - keR)
    if pR < min_pressure: pR = min_pressure
    var aL = sqrt(gamma * pL / rho_L)
    var aR = sqrt(gamma * pR / rho_R)
    var unL = uL * nx + vL * ny
    var unR = uR * nx + vR * ny

    # Normal-direction fluxes F_K.n for each side (4 components).
    var FnL0 = qL1 * nx + qL2 * ny
    var FnL1 = (qL1 * uL + pL) * nx + (qL1 * vL) * ny
    var FnL2 = (qL1 * vL) * nx + (qL2 * vL + pL) * ny
    var FnL3 = (uL * (qL3 + pL)) * nx + (vL * (qL3 + pL)) * ny
    var FnR0 = qR1 * nx + qR2 * ny
    var FnR1 = (qR1 * uR + pR) * nx + (qR1 * vR) * ny
    var FnR2 = (qR1 * vR) * nx + (qR2 * vR + pR) * ny
    var FnR3 = (uR * (qR3 + pR)) * nx + (vR * (qR3 + pR)) * ny

    # Davis wave-speed estimates.
    var S_L = unL - aL
    var tmp = unR - aR
    if tmp < S_L: S_L = tmp
    var S_R = unL + aL
    tmp = unR + aR
    if tmp > S_R: S_R = tmp

    # Contact wave speed.
    var num = pR - pL + rho_L * unL * (S_L - unL) - rho_R * unR * (S_R - unR)
    var den = rho_L * (S_L - unL) - rho_R * (S_R - unR)
    var S_star = num / den

    var out = (fid * NFP + m) * 4
    if S_L >= Float32(0.0):
        fstar_out[out + 0] = FnL0
        fstar_out[out + 1] = FnL1
        fstar_out[out + 2] = FnL2
        fstar_out[out + 3] = FnL3
    elif S_R <= Float32(0.0):
        fstar_out[out + 0] = FnR0
        fstar_out[out + 1] = FnR1
        fstar_out[out + 2] = FnR2
        fstar_out[out + 3] = FnR3
    elif S_star >= Float32(0.0):
        var coef = (S_L - unL) / (S_L - S_star)
        var rho_s = rho_L * coef
        var u_s = uL + (S_star - unL) * nx
        var v_s = vL + (S_star - unL) * ny
        var E_over_rho_s = (
            qL3 / rho_L
            + (S_star - unL) * (S_star + pL / (rho_L * (S_L - unL)))
        )
        var qs0 = rho_s
        var qs1 = rho_s * u_s
        var qs2 = rho_s * v_s
        var qs3 = rho_s * E_over_rho_s
        fstar_out[out + 0] = FnL0 + S_L * (qs0 - qL0)
        fstar_out[out + 1] = FnL1 + S_L * (qs1 - qL1)
        fstar_out[out + 2] = FnL2 + S_L * (qs2 - qL2)
        fstar_out[out + 3] = FnL3 + S_L * (qs3 - qL3)
    else:
        var coef = (S_R - unR) / (S_R - S_star)
        var rho_s = rho_R * coef
        var u_s = uR + (S_star - unR) * nx
        var v_s = vR + (S_star - unR) * ny
        var E_over_rho_s = (
            qR3 / rho_R
            + (S_star - unR) * (S_star + pR / (rho_R * (S_R - unR)))
        )
        var qs0 = rho_s
        var qs1 = rho_s * u_s
        var qs2 = rho_s * v_s
        var qs3 = rho_s * E_over_rho_s
        fstar_out[out + 0] = FnR0 + S_R * (qs0 - qR0)
        fstar_out[out + 1] = FnR1 + S_R * (qs1 - qR1)
        fstar_out[out + 2] = FnR2 + S_R * (qs2 - qR2)
        fstar_out[out + 3] = FnR3 + S_R * (qs3 - qR3)


def launch_euler_face_flux_hllc_2d[NP: Int, NFP: Int](
    mut ctx: DeviceContext,
    q:              UnsafePointer[Float32, MutAnyOrigin],
    face_elem:      UnsafePointer[Int32,   MutAnyOrigin],
    face_elem_node: UnsafePointer[Int32,   MutAnyOrigin],
    face_normal:    UnsafePointer[Float32, MutAnyOrigin],
    face_bc_type:   UnsafePointer[Int32,   MutAnyOrigin],
    num_faces:      Int,
    gamma:          Float32,
    min_density:    Float32,
    min_pressure:   Float32,
    inflow_rho:     Float32,
    inflow_rhou:    Float32,
    inflow_rhov:    Float32,
    inflow_E:       Float32,
    fstar_out:      UnsafePointer[Float32, MutAnyOrigin],
) raises:
    var total = num_faces * NFP
    comptime _kernel = euler_face_flux_hllc_kernel_2d[NP, NFP]
    ctx.enqueue_function[_kernel, _kernel](
        q, face_elem, face_elem_node, face_normal, face_bc_type,
        num_faces,
        gamma, min_density, min_pressure,
        inflow_rho, inflow_rhou, inflow_rhov, inflow_E,
        fstar_out,
        grid_dim=ceildiv(total, 256),
        block_dim=256,
    )


# ----------------------------------------------------------------------
# Full Euler RK-stage orchestration.
# ----------------------------------------------------------------------
# Mirrors `advection_rk_stage_2d` but with 4 components per node and
# the Euler volume / face-flux kernels.  Callers provide q/q_a/q_b/q_out
# (each sized num_elements*NP*4) plus scratch buffers for vol, fstar,
# and rhs.
# ----------------------------------------------------------------------

def euler_rk_stage_2d[P: Int](
    mut ctx: DeviceContext,
    mesh: LocalMesh2DGpu[P],
    Lift_ref: UnsafePointer[Float32, MutAnyOrigin],
    D_ref:    UnsafePointer[Float32, MutAnyOrigin],
    q_in:     UnsafePointer[Float32, MutAnyOrigin],
    q_a:      UnsafePointer[Float32, MutAnyOrigin],
    q_b:      UnsafePointer[Float32, MutAnyOrigin],
    q_out:    UnsafePointer[Float32, MutAnyOrigin],
    vol_scratch:   UnsafePointer[Float32, MutAnyOrigin],
    fstar_scratch: UnsafePointer[Float32, MutAnyOrigin],
    rhs_scratch:   UnsafePointer[Float32, MutAnyOrigin],
    gamma: Float32, min_density: Float32, min_pressure: Float32,
    inflow_rho: Float32, inflow_rhou: Float32,
    inflow_rhov: Float32, inflow_E: Float32,
    a: Float32, b: Float32, cc: Float32, dt: Float32,
    gx: Float32 = Float32(0.0),
    gy: Float32 = Float32(0.0),
) raises:
    # Two launches per stage (down from three): face flux, then a fused
    # vol+lift+RK kernel.  vol_scratch / rhs_scratch are unused on the
    # fused path; kept in the signature for backward compatibility.
    # Gravity defaults to zero so existing callers (vortex / smooth-wave /
    # channel-steady / sod / sod-limited / shocked benches and the
    # examples) are source-compatible without edits.
    _ = vol_scratch
    _ = rhs_scratch
    comptime NP = num_tri_nodes_2d(P)
    comptime NFP = num_edge_nodes(P)
    launch_euler_face_flux_2d[NP, NFP](
        ctx, q_in,
        mesh.d_face_elem.unsafe_ptr(),
        mesh.d_face_elem_node.unsafe_ptr(),
        mesh.d_face_normal.unsafe_ptr(),
        mesh.d_face_bc_type.unsafe_ptr(),
        mesh.num_faces,
        gamma, min_density, min_pressure,
        inflow_rho, inflow_rhou, inflow_rhov, inflow_E,
        fstar_scratch,
    )
    launch_euler_vol_lift_2d[NP, NFP](
        ctx, q_in, mesh.d_elem_invJ.unsafe_ptr(), D_ref,
        fstar_scratch,
        mesh.d_elem_inv_2A.unsafe_ptr(),
        mesh.d_elem_faces.unsafe_ptr(),
        mesh.d_elem_face_side.unsafe_ptr(),
        mesh.d_elem_canon_to_ref.unsafe_ptr(),
        mesh.d_face_length.unsafe_ptr(),
        Lift_ref, q_a, q_b,
        mesh.num_elements, gamma, min_density, min_pressure,
        gx, gy,
        a, b, cc, dt, q_out,
    )


# ----------------------------------------------------------------------
# Same as `euler_rk_stage_2d` but uses HLLC instead of Rusanov for the
# interior numerical flux.  Volume RHS and lift-combine kernels are
# unchanged -- HLLC affects only `fstar`.
# ----------------------------------------------------------------------

def euler_rk_stage_hllc_2d[P: Int](
    mut ctx: DeviceContext,
    mesh: LocalMesh2DGpu[P],
    Lift_ref: UnsafePointer[Float32, MutAnyOrigin],
    D_ref:    UnsafePointer[Float32, MutAnyOrigin],
    q_in:     UnsafePointer[Float32, MutAnyOrigin],
    q_a:      UnsafePointer[Float32, MutAnyOrigin],
    q_b:      UnsafePointer[Float32, MutAnyOrigin],
    q_out:    UnsafePointer[Float32, MutAnyOrigin],
    vol_scratch:   UnsafePointer[Float32, MutAnyOrigin],
    fstar_scratch: UnsafePointer[Float32, MutAnyOrigin],
    rhs_scratch:   UnsafePointer[Float32, MutAnyOrigin],
    gamma: Float32, min_density: Float32, min_pressure: Float32,
    inflow_rho: Float32, inflow_rhou: Float32,
    inflow_rhov: Float32, inflow_E: Float32,
    a: Float32, b: Float32, cc: Float32, dt: Float32,
    gx: Float32 = Float32(0.0),
    gy: Float32 = Float32(0.0),
) raises:
    # Two launches per stage (down from three).  HLLC variant of the
    # face flux + the same fused vol+lift+RK kernel as Rusanov path.
    # See `euler_rk_stage_2d` for the gravity convention; gx, gy are
    # forwarded unchanged.
    _ = vol_scratch
    _ = rhs_scratch
    comptime NP = num_tri_nodes_2d(P)
    comptime NFP = num_edge_nodes(P)
    launch_euler_face_flux_hllc_2d[NP, NFP](
        ctx, q_in,
        mesh.d_face_elem.unsafe_ptr(),
        mesh.d_face_elem_node.unsafe_ptr(),
        mesh.d_face_normal.unsafe_ptr(),
        mesh.d_face_bc_type.unsafe_ptr(),
        mesh.num_faces,
        gamma, min_density, min_pressure,
        inflow_rho, inflow_rhou, inflow_rhov, inflow_E,
        fstar_scratch,
    )
    launch_euler_vol_lift_2d[NP, NFP](
        ctx, q_in, mesh.d_elem_invJ.unsafe_ptr(), D_ref,
        fstar_scratch,
        mesh.d_elem_inv_2A.unsafe_ptr(),
        mesh.d_elem_faces.unsafe_ptr(),
        mesh.d_elem_face_side.unsafe_ptr(),
        mesh.d_elem_canon_to_ref.unsafe_ptr(),
        mesh.d_face_length.unsafe_ptr(),
        Lift_ref, q_a, q_b,
        mesh.num_elements, gamma, min_density, min_pressure,
        gx, gy,
        a, b, cc, dt, q_out,
    )
