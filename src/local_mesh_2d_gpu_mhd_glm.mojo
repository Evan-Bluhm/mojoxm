# ======================================================================
# local_mesh_2d_gpu_mhd_glm.mojo -- GLM-enabled 2D MHD (NC=7)
# ======================================================================
# State (rho, mx, my, Bx, By, E, psi).  Driver entry point:
# `mhd_glm_rk_stage_2d[P]` for the 2-launch flux update;
# `launch_mhd_glm_psi_damp_2d[NP]` is invoked separately by drivers
# once per timestep for the Strang-split psi damping (see below).
#
# Adds the Dedner-Kemm-Kroner-Munz-Schnitzer-Wesenberg generalised
# Lagrange multiplier divergence-cleaning equation:
#
#   dB/dt + div(u B - B u + psi I) = 0           (modified)
#   dpsi/dt + c_h^2 div(B)         = -alpha_d * psi
#
# In 2D the modifications relative to plain ideal MHD are:
#   * Bx-flux x-component += psi
#   * By-flux y-component += psi
#   * psi-flux = c_h^2 * (Bx, By)
#   * Rusanov wave-speed bound: alpha = max(|u_n| + c_f, c_h)
#   * BC_WALL: psi reflects (- psi on the ghost) so the slip wall
#     looks fully insulating to the GLM transport.
#
# psi damping (-alpha_d * psi) is applied via Strang/operator splitting
# AFTER the flux update: q_out[psi] *= exp(-alpha_d * dt).  This is
# semi-discretely exact and stable for any alpha_d, dt.
# ======================================================================

from src.local_mesh_2d_gpu import LocalMesh2DGpu
from src.reference_2d import num_tri_nodes_2d, num_edge_nodes
from src.boundary import BC_INTERIOR, BC_WALL
from std.gpu import global_idx
from std.gpu.host import DeviceContext
from std.math import ceildiv, sqrt, exp


# ----------------------------------------------------------------------
# Fused IdealMHD-GLM volume + lift + RK update (2D, NC=7).
# ----------------------------------------------------------------------
# Same fusion pattern with the GLM divergence-cleaning flux additions:
# Fx_Bx += psi, Fy_By += psi, F_psi = c_h^2 * B.
# ----------------------------------------------------------------------

def mhd_glm_vol_lift_combine_rk_kernel_2d[NP: Int, NFP: Int](
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
    c_h:               Float32,
    a: Float32, b: Float32, cc: Float32, dt: Float32,
    q_out:             UnsafePointer[Float32, MutAnyOrigin],
):
    var tid = Int(global_idx.x)
    var total = num_elements * NP
    if tid >= total:
        return
    var elem = tid // NP
    var i    = tid %  NP

    # ---- Volume RHS contribution (NC=7, GLM flux additions).
    var iJ00 = elem_invJ[elem * 4 + 0]
    var iJ01 = elem_invJ[elem * 4 + 1]
    var iJ10 = elem_invJ[elem * 4 + 2]
    var iJ11 = elem_invJ[elem * 4 + 3]
    var acc0: Float32 = 0.0
    var acc1: Float32 = 0.0
    var acc2: Float32 = 0.0
    var acc3: Float32 = 0.0
    var acc4: Float32 = 0.0
    var acc5: Float32 = 0.0
    var acc6: Float32 = 0.0
    var ch2 = c_h * c_h
    for j in range(NP):
        var base = (elem * NP + j) * 7
        var rho = q_in[base + 0]
        if rho < min_density:
            rho = min_density
        var mx = q_in[base + 1]
        var my = q_in[base + 2]
        var Bx = q_in[base + 3]
        var By = q_in[base + 4]
        var E  = q_in[base + 5]
        var psi = q_in[base + 6]
        var u = mx / rho
        var v = my / rho
        var BB = Bx * Bx + By * By
        var ke = Float32(0.5) * (mx * mx + my * my) / rho
        var mp = Float32(0.5) * BB
        var p = (gamma - Float32(1.0)) * (E - ke - mp)
        if p < min_pressure:
            p = min_pressure
        var pstar = p + Float32(0.5) * BB
        var Fx0 = mx
        var Fx1 = mx * u + pstar - Bx * Bx
        var Fx2 = mx * v         - Bx * By
        var Fx3 = psi
        var Fx4 = u * By - v * Bx
        var Fx5 = (E + pstar) * u - Bx * (u * Bx + v * By)
        var Fx6 = ch2 * Bx
        var Fy0 = my
        var Fy1 = my * u         - By * Bx
        var Fy2 = my * v + pstar - By * By
        var Fy3 = v * Bx - u * By
        var Fy4 = psi
        var Fy5 = (E + pstar) * v - By * (u * Bx + v * By)
        var Fy6 = ch2 * By
        var D_r = D_ref[0 * NP * NP + i * NP + j]
        var D_s = D_ref[1 * NP * NP + i * NP + j]
        acc0 += (iJ00 * Fx0 + iJ01 * Fy0) * D_r + (iJ10 * Fx0 + iJ11 * Fy0) * D_s
        acc1 += (iJ00 * Fx1 + iJ01 * Fy1) * D_r + (iJ10 * Fx1 + iJ11 * Fy1) * D_s
        acc2 += (iJ00 * Fx2 + iJ01 * Fy2) * D_r + (iJ10 * Fx2 + iJ11 * Fy2) * D_s
        acc3 += (iJ00 * Fx3 + iJ01 * Fy3) * D_r + (iJ10 * Fx3 + iJ11 * Fy3) * D_s
        acc4 += (iJ00 * Fx4 + iJ01 * Fy4) * D_r + (iJ10 * Fx4 + iJ11 * Fy4) * D_s
        acc5 += (iJ00 * Fx5 + iJ01 * Fy5) * D_r + (iJ10 * Fx5 + iJ11 * Fy5) * D_s
        acc6 += (iJ00 * Fx6 + iJ01 * Fy6) * D_r + (iJ10 * Fx6 + iJ11 * Fy6) * D_s

    # ---- Lift contribution (NC=7).
    var inv_2A = elem_inv_2A[elem]
    var face0: Float32 = 0.0
    var face1: Float32 = 0.0
    var face2: Float32 = 0.0
    var face3: Float32 = 0.0
    var face4: Float32 = 0.0
    var face5: Float32 = 0.0
    var face6: Float32 = 0.0
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
            var fbase = (fid * NFP + m) * 7
            face0 += sLf * fstar[fbase + 0]
            face1 += sLf * fstar[fbase + 1]
            face2 += sLf * fstar[fbase + 2]
            face3 += sLf * fstar[fbase + 3]
            face4 += sLf * fstar[fbase + 4]
            face5 += sLf * fstar[fbase + 5]
            face6 += sLf * fstar[fbase + 6]

    # ---- Combine + RK update.
    var idx = (elem * NP + i) * 7
    var rhs0 = acc0 - inv_2A * face0
    var rhs1 = acc1 - inv_2A * face1
    var rhs2 = acc2 - inv_2A * face2
    var rhs3 = acc3 - inv_2A * face3
    var rhs4 = acc4 - inv_2A * face4
    var rhs5 = acc5 - inv_2A * face5
    var rhs6 = acc6 - inv_2A * face6
    q_out[idx + 0] = a * q_a[idx + 0] + b * q_b[idx + 0] + cc * dt * rhs0
    q_out[idx + 1] = a * q_a[idx + 1] + b * q_b[idx + 1] + cc * dt * rhs1
    q_out[idx + 2] = a * q_a[idx + 2] + b * q_b[idx + 2] + cc * dt * rhs2
    q_out[idx + 3] = a * q_a[idx + 3] + b * q_b[idx + 3] + cc * dt * rhs3
    q_out[idx + 4] = a * q_a[idx + 4] + b * q_b[idx + 4] + cc * dt * rhs4
    q_out[idx + 5] = a * q_a[idx + 5] + b * q_b[idx + 5] + cc * dt * rhs5
    q_out[idx + 6] = a * q_a[idx + 6] + b * q_b[idx + 6] + cc * dt * rhs6


def launch_mhd_glm_vol_lift_2d[NP: Int, NFP: Int](
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
    c_h:               Float32,
    a: Float32, b: Float32, cc: Float32, dt: Float32,
    q_out:             UnsafePointer[Float32, MutAnyOrigin],
) raises:
    var total = num_elements * NP
    comptime _kernel = mhd_glm_vol_lift_combine_rk_kernel_2d[NP, NFP]
    ctx.enqueue_function[_kernel, _kernel](
        q_in, elem_invJ, D_ref, fstar,
        elem_inv_2A, elem_faces, elem_face_side, elem_canon_to_ref,
        face_length, Lift_ref, q_a, q_b,
        num_elements, gamma, min_density, min_pressure, c_h,
        a, b, cc, dt, q_out,
        grid_dim=ceildiv(total, 256),
        block_dim=256,
    )

def mhd_glm_face_flux_kernel_2d[NP: Int, NFP: Int](
    q:              UnsafePointer[Float32, MutAnyOrigin],
    face_elem:      UnsafePointer[Int32,   MutAnyOrigin],
    face_elem_node: UnsafePointer[Int32,   MutAnyOrigin],
    face_normal:    UnsafePointer[Float32, MutAnyOrigin],
    face_bc_type:   UnsafePointer[Int32,   MutAnyOrigin],
    num_faces:      Int,
    gamma:          Float32,
    min_density:    Float32,
    min_pressure:   Float32,
    c_h:            Float32,
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
    var l_off = (e_l * NP + n_l) * 7
    var qL0 = q[l_off + 0]
    var qL1 = q[l_off + 1]
    var qL2 = q[l_off + 2]
    var qL3 = q[l_off + 3]
    var qL4 = q[l_off + 4]
    var qL5 = q[l_off + 5]
    var qL6 = q[l_off + 6]

    var qR0: Float32
    var qR1: Float32
    var qR2: Float32
    var qR3: Float32
    var qR4: Float32
    var qR5: Float32
    var qR6: Float32
    if bc_type == BC_INTERIOR:
        var e_r = Int(face_elem[fid * 2 + 1])
        var n_r = Int(face_elem_node[(fid * 2 + 1) * NFP + m])
        var r_off = (e_r * NP + n_r) * 7
        qR0 = q[r_off + 0]
        qR1 = q[r_off + 1]
        qR2 = q[r_off + 2]
        qR3 = q[r_off + 3]
        qR4 = q[r_off + 4]
        qR5 = q[r_off + 5]
        qR6 = q[r_off + 6]
    elif bc_type == BC_WALL:
        var m_n = qL1 * nx + qL2 * ny
        var B_n = qL3 * nx + qL4 * ny
        qR0 = qL0
        qR1 = qL1 - Float32(2.0) * m_n * nx
        qR2 = qL2 - Float32(2.0) * m_n * ny
        qR3 = qL3 - Float32(2.0) * B_n * nx
        qR4 = qL4 - Float32(2.0) * B_n * ny
        qR5 = qL5
        qR6 = -qL6
    else:
        qR0 = qL0
        qR1 = qL1
        qR2 = qL2
        qR3 = qL3
        qR4 = qL4
        qR5 = qL5
        qR6 = qL6

    var ch2 = c_h * c_h

    # Left internal flux + fast-magnetosonic speed.
    var rhoL = qL0
    if rhoL < min_density:
        rhoL = min_density
    var uL = qL1 / rhoL
    var vL = qL2 / rhoL
    var BBL = qL3 * qL3 + qL4 * qL4
    var keL = Float32(0.5) * (qL1 * qL1 + qL2 * qL2) / rhoL
    var pL = (gamma - Float32(1.0)) * (qL5 - keL - Float32(0.5) * BBL)
    if pL < min_pressure:
        pL = min_pressure
    var pstarL = pL + Float32(0.5) * BBL
    var FxL0 = qL1
    var FxL1 = qL1 * uL + pstarL - qL3 * qL3
    var FxL2 = qL1 * vL         - qL3 * qL4
    var FxL3 = qL6
    var FxL4 = uL * qL4 - vL * qL3
    var FxL5 = (qL5 + pstarL) * uL - qL3 * (uL * qL3 + vL * qL4)
    var FxL6 = ch2 * qL3
    var FyL0 = qL2
    var FyL1 = qL2 * uL         - qL4 * qL3
    var FyL2 = qL2 * vL + pstarL - qL4 * qL4
    var FyL3 = vL * qL3 - uL * qL4
    var FyL4 = qL6
    var FyL5 = (qL5 + pstarL) * vL - qL4 * (uL * qL3 + vL * qL4)
    var FyL6 = ch2 * qL4
    var cs2L = gamma * pL / rhoL
    var ca2L = BBL / rhoL
    var sL = cs2L + ca2L
    var discL = sL * sL - Float32(4.0) * cs2L * (qL3 * qL3) / rhoL
    if discL < Float32(0.0):
        discL = Float32(0.0)
    var cf2L = Float32(0.5) * (sL + sqrt(discL))
    var cfL = sqrt(cf2L)
    var speedL = sqrt(uL * uL + vL * vL) + cfL
    if c_h > speedL:
        speedL = c_h

    # Right internal flux + fast-magnetosonic speed.
    var rhoR = qR0
    if rhoR < min_density:
        rhoR = min_density
    var uR = qR1 / rhoR
    var vR = qR2 / rhoR
    var BBR = qR3 * qR3 + qR4 * qR4
    var keR = Float32(0.5) * (qR1 * qR1 + qR2 * qR2) / rhoR
    var pR = (gamma - Float32(1.0)) * (qR5 - keR - Float32(0.5) * BBR)
    if pR < min_pressure:
        pR = min_pressure
    var pstarR = pR + Float32(0.5) * BBR
    var FxR0 = qR1
    var FxR1 = qR1 * uR + pstarR - qR3 * qR3
    var FxR2 = qR1 * vR         - qR3 * qR4
    var FxR3 = qR6
    var FxR4 = uR * qR4 - vR * qR3
    var FxR5 = (qR5 + pstarR) * uR - qR3 * (uR * qR3 + vR * qR4)
    var FxR6 = ch2 * qR3
    var FyR0 = qR2
    var FyR1 = qR2 * uR         - qR4 * qR3
    var FyR2 = qR2 * vR + pstarR - qR4 * qR4
    var FyR3 = vR * qR3 - uR * qR4
    var FyR4 = qR6
    var FyR5 = (qR5 + pstarR) * vR - qR4 * (uR * qR3 + vR * qR4)
    var FyR6 = ch2 * qR4
    var cs2R = gamma * pR / rhoR
    var ca2R = BBR / rhoR
    var sR = cs2R + ca2R
    var discR = sR * sR - Float32(4.0) * cs2R * (qR3 * qR3) / rhoR
    if discR < Float32(0.0):
        discR = Float32(0.0)
    var cf2R = Float32(0.5) * (sR + sqrt(discR))
    var cfR = sqrt(cf2R)
    var speedR = sqrt(uR * uR + vR * vR) + cfR
    if c_h > speedR:
        speedR = c_h

    var alpha: Float32 = speedL if speedL > speedR else speedR
    var half = Float32(0.5)
    var out = (fid * NFP + m) * 7
    fstar_out[out + 0] = half * ((FxL0 + FxR0) * nx + (FyL0 + FyR0) * ny) \
                         - half * alpha * (qR0 - qL0)
    fstar_out[out + 1] = half * ((FxL1 + FxR1) * nx + (FyL1 + FyR1) * ny) \
                         - half * alpha * (qR1 - qL1)
    fstar_out[out + 2] = half * ((FxL2 + FxR2) * nx + (FyL2 + FyR2) * ny) \
                         - half * alpha * (qR2 - qL2)
    fstar_out[out + 3] = half * ((FxL3 + FxR3) * nx + (FyL3 + FyR3) * ny) \
                         - half * alpha * (qR3 - qL3)
    fstar_out[out + 4] = half * ((FxL4 + FxR4) * nx + (FyL4 + FyR4) * ny) \
                         - half * alpha * (qR4 - qL4)
    fstar_out[out + 5] = half * ((FxL5 + FxR5) * nx + (FyL5 + FyR5) * ny) \
                         - half * alpha * (qR5 - qL5)
    fstar_out[out + 6] = half * ((FxL6 + FxR6) * nx + (FyL6 + FyR6) * ny) \
                         - half * alpha * (qR6 - qL6)


def launch_mhd_glm_face_flux_2d[NP: Int, NFP: Int](
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
    c_h:            Float32,
    fstar_out:      UnsafePointer[Float32, MutAnyOrigin],
) raises:
    var total = num_faces * NFP
    comptime _kernel = mhd_glm_face_flux_kernel_2d[NP, NFP]
    ctx.enqueue_function[_kernel, _kernel](
        q, face_elem, face_elem_node, face_normal, face_bc_type,
        num_faces,
        gamma, min_density, min_pressure, c_h,
        fstar_out,
        grid_dim=ceildiv(total, 256),
        block_dim=256,
    )


# Operator-splitting psi damping: q[psi] *= exp(-alpha_d * dt).
def mhd_glm_psi_damp_kernel_2d[NP: Int](
    q:            UnsafePointer[Float32, MutAnyOrigin],
    num_elements: Int,
    decay:        Float32,
):
    var tid = Int(global_idx.x)
    var total = num_elements * NP
    if tid >= total:
        return
    var idx = tid * 7 + 6
    q[idx] = q[idx] * decay


def launch_mhd_glm_psi_damp_2d[NP: Int](
    mut ctx: DeviceContext,
    q:            UnsafePointer[Float32, MutAnyOrigin],
    num_elements: Int,
    alpha_d:      Float32,
    dt:           Float32,
) raises:
    if alpha_d <= Float32(0.0):
        return
    # exp(-alpha_d * dt) computed on host since the kernel is per-node
    # and the decay factor is the same for every node.
    from std.math import exp
    var decay = Float32(exp(-Float64(alpha_d) * Float64(dt)))
    var total = num_elements * NP
    comptime _kernel = mhd_glm_psi_damp_kernel_2d[NP]
    ctx.enqueue_function[_kernel, _kernel](
        q, num_elements, decay,
        grid_dim=ceildiv(total, 256),
        block_dim=256,
    )


def mhd_glm_rk_stage_2d[P: Int](
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
    c_h: Float32, alpha_d: Float32,
    a: Float32, b: Float32, cc: Float32, dt: Float32,
) raises:
    # Two flux launches per stage (down from three) plus psi damping.
    _ = vol_scratch
    _ = rhs_scratch
    comptime NP = num_tri_nodes_2d(P)
    comptime NFP = num_edge_nodes(P)
    launch_mhd_glm_face_flux_2d[NP, NFP](
        ctx, q_in,
        mesh.d_face_elem.unsafe_ptr(),
        mesh.d_face_elem_node.unsafe_ptr(),
        mesh.d_face_normal.unsafe_ptr(),
        mesh.d_face_bc_type.unsafe_ptr(),
        mesh.num_faces,
        gamma, min_density, min_pressure, c_h,
        fstar_scratch,
    )
    launch_mhd_glm_vol_lift_2d[NP, NFP](
        ctx, q_in, mesh.d_elem_invJ.unsafe_ptr(), D_ref,
        fstar_scratch,
        mesh.d_elem_inv_2A.unsafe_ptr(),
        mesh.d_elem_faces.unsafe_ptr(),
        mesh.d_elem_face_side.unsafe_ptr(),
        mesh.d_elem_canon_to_ref.unsafe_ptr(),
        mesh.d_face_length.unsafe_ptr(),
        Lift_ref, q_a, q_b,
        mesh.num_elements, gamma, min_density, min_pressure, c_h,
        a, b, cc, dt, q_out,
    )
    # NOTE: the alpha_d argument is no longer applied here.  Operator-
    # splitting psi damping must be invoked ONCE PER SSPRK3 step by the
    # caller via `launch_mhd_glm_psi_damp_2d`, not once per stage --
    # otherwise the decay gets applied three times per timestep.  The
    # parameter is retained for backward compatibility with existing
    # drivers that pass alpha_d (the GLM Alfven and psi-transport
    # benches use alpha_d = 0 so the change is observably a no-op for
    # them).
    _ = alpha_d
