# ======================================================================
# local_mesh_2d_gpu_sw.mojo -- 2D Shallow Water (NC=3) GPU kernels
# ======================================================================
# State (h, mx=h*u, my=h*v).  Two driver entry points, one per
# Riemann solver:
#   * `sw_rk_stage_2d[P]`        -- Rusanov flux
#   * `sw_rk_stage_hll_2d[P]`    -- Einfeldt HLL (1988)
# Each stage is 2 launches (face-flux + fused per-(elem, node)
# vol+lift+RK).  The vol+lift+RK kernel is flux-independent.
# ======================================================================

from src.local_mesh_2d_gpu import LocalMesh2DGpu
from src.reference_2d import num_tri_nodes_2d, num_edge_nodes
from src.boundary import BC_INTERIOR, BC_WALL, BC_INFLOW
from std.gpu import global_idx
from std.gpu.host import DeviceContext
from std.math import ceildiv, sqrt


# ----------------------------------------------------------------------
# Fused ShallowWater volume + lift + RK update (2D, NC=3).
# ----------------------------------------------------------------------
# Same fusion pattern as advection / Euler 2D variants.
# ----------------------------------------------------------------------


def sw_vol_lift_combine_rk_kernel_2d[
    NP: Int, NFP: Int
](
    q_in: UnsafePointer[Float32, MutAnyOrigin],
    elem_invJ: UnsafePointer[Float32, MutAnyOrigin],
    D_ref: UnsafePointer[Float32, MutAnyOrigin],
    fstar: UnsafePointer[Float32, MutAnyOrigin],
    elem_inv_2A: UnsafePointer[Float32, MutAnyOrigin],
    elem_faces: UnsafePointer[Int32, MutAnyOrigin],
    elem_face_side: UnsafePointer[Int32, MutAnyOrigin],
    elem_canon_to_ref: UnsafePointer[Int32, MutAnyOrigin],
    face_length: UnsafePointer[Float32, MutAnyOrigin],
    Lift_ref: UnsafePointer[Float32, MutAnyOrigin],
    q_a: UnsafePointer[Float32, MutAnyOrigin],
    q_b: UnsafePointer[Float32, MutAnyOrigin],
    num_elements: Int,
    g: Float32,
    min_h: Float32,
    a: Float32,
    b: Float32,
    cc: Float32,
    dt: Float32,
    q_out: UnsafePointer[Float32, MutAnyOrigin],
):
    var tid = Int(global_idx.x)
    var total = num_elements * NP
    if tid >= total:
        return
    var elem = tid // NP
    var i = tid % NP

    # ---- Volume RHS contribution.
    var iJ00 = elem_invJ[elem * 4 + 0]
    var iJ01 = elem_invJ[elem * 4 + 1]
    var iJ10 = elem_invJ[elem * 4 + 2]
    var iJ11 = elem_invJ[elem * 4 + 3]
    var acc0: Float32 = 0.0
    var acc1: Float32 = 0.0
    var acc2: Float32 = 0.0
    for j in range(NP):
        var base = (elem * NP + j) * 3
        var h = q_in[base + 0]
        if h < min_h:
            h = min_h
        var mx = q_in[base + 1]
        var my = q_in[base + 2]
        var u = mx / h
        var v = my / h
        var p = Float32(0.5) * g * h * h
        var Fx0 = mx
        var Fx1 = mx * u + p
        var Fx2 = mx * v
        var Fy0 = my
        var Fy1 = my * u
        var Fy2 = my * v + p
        var D_r = D_ref[0 * NP * NP + i * NP + j]
        var D_s = D_ref[1 * NP * NP + i * NP + j]
        acc0 += (iJ00 * Fx0 + iJ01 * Fy0) * D_r + (iJ10 * Fx0 + iJ11 * Fy0) * D_s
        acc1 += (iJ00 * Fx1 + iJ01 * Fy1) * D_r + (iJ10 * Fx1 + iJ11 * Fy1) * D_s
        acc2 += (iJ00 * Fx2 + iJ01 * Fy2) * D_r + (iJ10 * Fx2 + iJ11 * Fy2) * D_s

    # ---- Lift contribution (NC=3).
    var inv_2A = elem_inv_2A[elem]
    var face0: Float32 = 0.0
    var face1: Float32 = 0.0
    var face2: Float32 = 0.0
    for lf in range(3):
        var fid = Int(elem_faces[elem * 3 + lf])
        var side = Int(elem_face_side[elem * 3 + lf])
        var sign: Float32 = Float32(1.0) if side == 0 else Float32(-1.0)
        var flen = face_length[fid]
        for m in range(NFP):
            var r = Int(elem_canon_to_ref[(elem * 3 + lf) * NFP + m])
            var Lim = Lift_ref[lf * NP * NFP + i * NFP + r]
            var sLf = sign * flen * Lim
            var fbase = (fid * NFP + m) * 3
            face0 += sLf * fstar[fbase + 0]
            face1 += sLf * fstar[fbase + 1]
            face2 += sLf * fstar[fbase + 2]

    # ---- Combine + RK update.
    var idx = (elem * NP + i) * 3
    var rhs0 = acc0 - inv_2A * face0
    var rhs1 = acc1 - inv_2A * face1
    var rhs2 = acc2 - inv_2A * face2
    q_out[idx + 0] = a * q_a[idx + 0] + b * q_b[idx + 0] + cc * dt * rhs0
    q_out[idx + 1] = a * q_a[idx + 1] + b * q_b[idx + 1] + cc * dt * rhs1
    q_out[idx + 2] = a * q_a[idx + 2] + b * q_b[idx + 2] + cc * dt * rhs2


def launch_sw_vol_lift_2d[
    NP: Int, NFP: Int
](
    mut ctx: DeviceContext,
    q_in: UnsafePointer[Float32, MutAnyOrigin],
    elem_invJ: UnsafePointer[Float32, MutAnyOrigin],
    D_ref: UnsafePointer[Float32, MutAnyOrigin],
    fstar: UnsafePointer[Float32, MutAnyOrigin],
    elem_inv_2A: UnsafePointer[Float32, MutAnyOrigin],
    elem_faces: UnsafePointer[Int32, MutAnyOrigin],
    elem_face_side: UnsafePointer[Int32, MutAnyOrigin],
    elem_canon_to_ref: UnsafePointer[Int32, MutAnyOrigin],
    face_length: UnsafePointer[Float32, MutAnyOrigin],
    Lift_ref: UnsafePointer[Float32, MutAnyOrigin],
    q_a: UnsafePointer[Float32, MutAnyOrigin],
    q_b: UnsafePointer[Float32, MutAnyOrigin],
    num_elements: Int,
    g: Float32,
    min_h: Float32,
    a: Float32,
    b: Float32,
    cc: Float32,
    dt: Float32,
    q_out: UnsafePointer[Float32, MutAnyOrigin],
) raises:
    var total = num_elements * NP
    comptime _kernel = sw_vol_lift_combine_rk_kernel_2d[NP, NFP]
    ctx.enqueue_function[_kernel](
        q_in,
        elem_invJ,
        D_ref,
        fstar,
        elem_inv_2A,
        elem_faces,
        elem_face_side,
        elem_canon_to_ref,
        face_length,
        Lift_ref,
        q_a,
        q_b,
        num_elements,
        g,
        min_h,
        a,
        b,
        cc,
        dt,
        q_out,
        grid_dim=ceildiv(total, 256),
        block_dim=256,
    )


# ----------------------------------------------------------------------
# ShallowWater2D face flux (Rusanov, 3 components).
# ----------------------------------------------------------------------
# BC ghost: WALL reflects normal momentum, INFLOW is user-set, OUTFLOW
# is zero-gradient -- matches ShallowWater2D.boundary_flux on CPU.
# ----------------------------------------------------------------------


def sw_face_flux_kernel_2d[
    NP: Int, NFP: Int
](
    q: UnsafePointer[Float32, MutAnyOrigin],
    face_elem: UnsafePointer[Int32, MutAnyOrigin],
    face_elem_node: UnsafePointer[Int32, MutAnyOrigin],
    face_normal: UnsafePointer[Float32, MutAnyOrigin],
    face_bc_type: UnsafePointer[Int32, MutAnyOrigin],
    num_faces: Int,
    g: Float32,
    min_h: Float32,
    inflow_h: Float32,
    inflow_hu: Float32,
    inflow_hv: Float32,
    fstar_out: UnsafePointer[Float32, MutAnyOrigin],
):
    var tid = Int(global_idx.x)
    var total = num_faces * NFP
    if tid >= total:
        return
    var fid = tid // NFP
    var m = tid % NFP

    var nx = face_normal[fid * 2 + 0]
    var ny = face_normal[fid * 2 + 1]
    var bc_type = face_bc_type[fid]

    var e_l = Int(face_elem[fid * 2 + 0])
    var n_l = Int(face_elem_node[(fid * 2 + 0) * NFP + m])
    var l_off = (e_l * NP + n_l) * 3
    var qL0 = q[l_off + 0]
    var qL1 = q[l_off + 1]
    var qL2 = q[l_off + 2]

    var qR0: Float32
    var qR1: Float32
    var qR2: Float32
    if bc_type == BC_INTERIOR:
        var e_r = Int(face_elem[fid * 2 + 1])
        var n_r = Int(face_elem_node[(fid * 2 + 1) * NFP + m])
        var r_off = (e_r * NP + n_r) * 3
        qR0 = q[r_off + 0]
        qR1 = q[r_off + 1]
        qR2 = q[r_off + 2]
    elif bc_type == BC_WALL:
        var m_n = qL1 * nx + qL2 * ny
        qR0 = qL0
        qR1 = qL1 - Float32(2.0) * m_n * nx
        qR2 = qL2 - Float32(2.0) * m_n * ny
    elif bc_type == BC_INFLOW:
        qR0 = inflow_h
        qR1 = inflow_hu
        qR2 = inflow_hv
    else:
        qR0 = qL0
        qR1 = qL1
        qR2 = qL2

    var hL = qL0
    if hL < min_h:
        hL = min_h
    var uL = qL1 / hL
    var vL = qL2 / hL
    var pL = Float32(0.5) * g * hL * hL
    var cL = sqrt(g * hL)
    var speedL = sqrt(uL * uL + vL * vL) + cL
    var FxL0 = qL1
    var FxL1 = qL1 * uL + pL
    var FxL2 = qL1 * vL
    var FyL0 = qL2
    var FyL1 = qL2 * uL
    var FyL2 = qL2 * vL + pL

    var hR = qR0
    if hR < min_h:
        hR = min_h
    var uR = qR1 / hR
    var vR = qR2 / hR
    var pR = Float32(0.5) * g * hR * hR
    var cR = sqrt(g * hR)
    var speedR = sqrt(uR * uR + vR * vR) + cR
    var FxR0 = qR1
    var FxR1 = qR1 * uR + pR
    var FxR2 = qR1 * vR
    var FyR0 = qR2
    var FyR1 = qR2 * uR
    var FyR2 = qR2 * vR + pR

    var alpha: Float32 = speedL if speedL > speedR else speedR
    var half = Float32(0.5)
    var out = (fid * NFP + m) * 3
    fstar_out[out + 0] = half * ((FxL0 + FxR0) * nx + (FyL0 + FyR0) * ny) - half * alpha * (qR0 - qL0)
    fstar_out[out + 1] = half * ((FxL1 + FxR1) * nx + (FyL1 + FyR1) * ny) - half * alpha * (qR1 - qL1)
    fstar_out[out + 2] = half * ((FxL2 + FxR2) * nx + (FyL2 + FyR2) * ny) - half * alpha * (qR2 - qL2)


def launch_sw_face_flux_2d[
    NP: Int, NFP: Int
](
    mut ctx: DeviceContext,
    q: UnsafePointer[Float32, MutAnyOrigin],
    face_elem: UnsafePointer[Int32, MutAnyOrigin],
    face_elem_node: UnsafePointer[Int32, MutAnyOrigin],
    face_normal: UnsafePointer[Float32, MutAnyOrigin],
    face_bc_type: UnsafePointer[Int32, MutAnyOrigin],
    num_faces: Int,
    g: Float32,
    min_h: Float32,
    inflow_h: Float32,
    inflow_hu: Float32,
    inflow_hv: Float32,
    fstar_out: UnsafePointer[Float32, MutAnyOrigin],
) raises:
    var total = num_faces * NFP
    comptime _kernel = sw_face_flux_kernel_2d[NP, NFP]
    ctx.enqueue_function[_kernel](q, face_elem, face_elem_node, face_normal, face_bc_type, num_faces, g, min_h, inflow_h, inflow_hu, inflow_hv, fstar_out, grid_dim=ceildiv(total, 256), block_dim=256)


# ----------------------------------------------------------------------
# ShallowWater2D HLL face flux (Einfeldt 1988, 3 components).
# ----------------------------------------------------------------------
# Same signature as the Rusanov kernel; uses Davis wave-speed estimates
# S_L = min(unL - cL, unR - cR), S_R = max(unL + cL, unR + cR), then
# the three-region HLL formula for the middle state.  Less dissipative
# than Rusanov's single-alpha fan, particularly around contacts and
# expansion fans.  Ghost-state plumbing (WALL / INFLOW / OUTFLOW) is
# the same as the Rusanov kernel -- only the interior flux differs.
# ----------------------------------------------------------------------


def sw_face_flux_hll_kernel_2d[
    NP: Int, NFP: Int
](
    q: UnsafePointer[Float32, MutAnyOrigin],
    face_elem: UnsafePointer[Int32, MutAnyOrigin],
    face_elem_node: UnsafePointer[Int32, MutAnyOrigin],
    face_normal: UnsafePointer[Float32, MutAnyOrigin],
    face_bc_type: UnsafePointer[Int32, MutAnyOrigin],
    num_faces: Int,
    g: Float32,
    min_h: Float32,
    inflow_h: Float32,
    inflow_hu: Float32,
    inflow_hv: Float32,
    fstar_out: UnsafePointer[Float32, MutAnyOrigin],
):
    var tid = Int(global_idx.x)
    var total = num_faces * NFP
    if tid >= total:
        return
    var fid = tid // NFP
    var m = tid % NFP

    var nx = face_normal[fid * 2 + 0]
    var ny = face_normal[fid * 2 + 1]
    var bc_type = face_bc_type[fid]

    var e_l = Int(face_elem[fid * 2 + 0])
    var n_l = Int(face_elem_node[(fid * 2 + 0) * NFP + m])
    var l_off = (e_l * NP + n_l) * 3
    var qL0 = q[l_off + 0]
    var qL1 = q[l_off + 1]
    var qL2 = q[l_off + 2]

    var qR0: Float32
    var qR1: Float32
    var qR2: Float32
    if bc_type == BC_INTERIOR:
        var e_r = Int(face_elem[fid * 2 + 1])
        var n_r = Int(face_elem_node[(fid * 2 + 1) * NFP + m])
        var r_off = (e_r * NP + n_r) * 3
        qR0 = q[r_off + 0]
        qR1 = q[r_off + 1]
        qR2 = q[r_off + 2]
    elif bc_type == BC_WALL:
        var m_n = qL1 * nx + qL2 * ny
        qR0 = qL0
        qR1 = qL1 - Float32(2.0) * m_n * nx
        qR2 = qL2 - Float32(2.0) * m_n * ny
    elif bc_type == BC_INFLOW:
        qR0 = inflow_h
        qR1 = inflow_hu
        qR2 = inflow_hv
    else:
        qR0 = qL0
        qR1 = qL1
        qR2 = qL2

    var hL = qL0
    if hL < min_h:
        hL = min_h
    var hR = qR0
    if hR < min_h:
        hR = min_h
    var uL = qL1 / hL
    var vL = qL2 / hL
    var uR = qR1 / hR
    var vR = qR2 / hR
    var pL = Float32(0.5) * g * hL * hL
    var pR = Float32(0.5) * g * hR * hR
    var cL = sqrt(g * hL)
    var cR = sqrt(g * hR)
    var unL = uL * nx + vL * ny
    var unR = uR * nx + vR * ny

    # Davis wave speeds.
    var S_L = unL - cL
    var tmp = unR - cR
    if tmp < S_L:
        S_L = tmp
    var S_R = unL + cL
    tmp = unR + cR
    if tmp > S_R:
        S_R = tmp

    # Normal fluxes on each side.
    var FxL0 = qL1
    var FxL1 = qL1 * uL + pL
    var FxL2 = qL1 * vL
    var FyL0 = qL2
    var FyL1 = qL2 * uL
    var FyL2 = qL2 * vL + pL
    var FxR0 = qR1
    var FxR1 = qR1 * uR + pR
    var FxR2 = qR1 * vR
    var FyR0 = qR2
    var FyR1 = qR2 * uR
    var FyR2 = qR2 * vR + pR
    var FnL0 = FxL0 * nx + FyL0 * ny
    var FnL1 = FxL1 * nx + FyL1 * ny
    var FnL2 = FxL2 * nx + FyL2 * ny
    var FnR0 = FxR0 * nx + FyR0 * ny
    var FnR1 = FxR1 * nx + FyR1 * ny
    var FnR2 = FxR2 * nx + FyR2 * ny

    var out = (fid * NFP + m) * 3
    if S_L >= Float32(0.0):
        fstar_out[out + 0] = FnL0
        fstar_out[out + 1] = FnL1
        fstar_out[out + 2] = FnL2
    elif S_R <= Float32(0.0):
        fstar_out[out + 0] = FnR0
        fstar_out[out + 1] = FnR1
        fstar_out[out + 2] = FnR2
    else:
        var inv = Float32(1.0) / (S_R - S_L)
        fstar_out[out + 0] = (S_R * FnL0 - S_L * FnR0 + S_L * S_R * (qR0 - qL0)) * inv
        fstar_out[out + 1] = (S_R * FnL1 - S_L * FnR1 + S_L * S_R * (qR1 - qL1)) * inv
        fstar_out[out + 2] = (S_R * FnL2 - S_L * FnR2 + S_L * S_R * (qR2 - qL2)) * inv


def launch_sw_face_flux_hll_2d[
    NP: Int, NFP: Int
](
    mut ctx: DeviceContext,
    q: UnsafePointer[Float32, MutAnyOrigin],
    face_elem: UnsafePointer[Int32, MutAnyOrigin],
    face_elem_node: UnsafePointer[Int32, MutAnyOrigin],
    face_normal: UnsafePointer[Float32, MutAnyOrigin],
    face_bc_type: UnsafePointer[Int32, MutAnyOrigin],
    num_faces: Int,
    g: Float32,
    min_h: Float32,
    inflow_h: Float32,
    inflow_hu: Float32,
    inflow_hv: Float32,
    fstar_out: UnsafePointer[Float32, MutAnyOrigin],
) raises:
    var total = num_faces * NFP
    comptime _kernel = sw_face_flux_hll_kernel_2d[NP, NFP]
    ctx.enqueue_function[_kernel](q, face_elem, face_elem_node, face_normal, face_bc_type, num_faces, g, min_h, inflow_h, inflow_hu, inflow_hv, fstar_out, grid_dim=ceildiv(total, 256), block_dim=256)


def sw_rk_stage_2d[
    P: Int
](
    mut ctx: DeviceContext,
    mesh: LocalMesh2DGpu[P],
    Lift_ref: UnsafePointer[Float32, MutAnyOrigin],
    D_ref: UnsafePointer[Float32, MutAnyOrigin],
    q_in: UnsafePointer[Float32, MutAnyOrigin],
    q_a: UnsafePointer[Float32, MutAnyOrigin],
    q_b: UnsafePointer[Float32, MutAnyOrigin],
    q_out: UnsafePointer[Float32, MutAnyOrigin],
    fstar_scratch: UnsafePointer[Float32, MutAnyOrigin],
    g: Float32,
    min_h: Float32,
    a: Float32,
    b: Float32,
    cc: Float32,
    dt: Float32,
    inflow_h: Float32 = Float32(0.0),
    inflow_hu: Float32 = Float32(0.0),
    inflow_hv: Float32 = Float32(0.0),
) raises:
    # Two launches per stage: face flux + fused vol+lift+RK.  inflow_*
    # default to zero; callers using BC_INFLOW pass via kwarg
    # (`inflow_h=...`).
    comptime NP = num_tri_nodes_2d(P)
    comptime NFP = num_edge_nodes(P)
    launch_sw_face_flux_2d[NP, NFP](
        ctx,
        q_in,
        mesh.d_face_elem.unsafe_ptr(),
        mesh.d_face_elem_node.unsafe_ptr(),
        mesh.d_face_normal.unsafe_ptr(),
        mesh.d_face_bc_type.unsafe_ptr(),
        mesh.num_faces,
        g,
        min_h,
        inflow_h,
        inflow_hu,
        inflow_hv,
        fstar_scratch,
    )
    launch_sw_vol_lift_2d[NP, NFP](
        ctx,
        q_in,
        mesh.d_elem_invJ.unsafe_ptr(),
        D_ref,
        fstar_scratch,
        mesh.d_elem_inv_2A.unsafe_ptr(),
        mesh.d_elem_faces.unsafe_ptr(),
        mesh.d_elem_face_side.unsafe_ptr(),
        mesh.d_elem_canon_to_ref.unsafe_ptr(),
        mesh.d_face_length.unsafe_ptr(),
        Lift_ref,
        q_a,
        q_b,
        mesh.num_elements,
        g,
        min_h,
        a,
        b,
        cc,
        dt,
        q_out,
    )


# ----------------------------------------------------------------------
# ShallowWater RK stage using HLL instead of Rusanov for the interior
# numerical flux.  Drop-in replacement for sw_rk_stage_2d.
# ----------------------------------------------------------------------


def sw_rk_stage_hll_2d[
    P: Int
](
    mut ctx: DeviceContext,
    mesh: LocalMesh2DGpu[P],
    Lift_ref: UnsafePointer[Float32, MutAnyOrigin],
    D_ref: UnsafePointer[Float32, MutAnyOrigin],
    q_in: UnsafePointer[Float32, MutAnyOrigin],
    q_a: UnsafePointer[Float32, MutAnyOrigin],
    q_b: UnsafePointer[Float32, MutAnyOrigin],
    q_out: UnsafePointer[Float32, MutAnyOrigin],
    fstar_scratch: UnsafePointer[Float32, MutAnyOrigin],
    g: Float32,
    min_h: Float32,
    a: Float32,
    b: Float32,
    cc: Float32,
    dt: Float32,
    inflow_h: Float32 = Float32(0.0),
    inflow_hu: Float32 = Float32(0.0),
    inflow_hv: Float32 = Float32(0.0),
) raises:
    # HLL flux variant of the face flux + the same fused vol+lift+RK
    # kernel as the Rusanov path.  See sw_rk_stage_2d for the
    # inflow_* default convention.
    comptime NP = num_tri_nodes_2d(P)
    comptime NFP = num_edge_nodes(P)
    launch_sw_face_flux_hll_2d[NP, NFP](
        ctx,
        q_in,
        mesh.d_face_elem.unsafe_ptr(),
        mesh.d_face_elem_node.unsafe_ptr(),
        mesh.d_face_normal.unsafe_ptr(),
        mesh.d_face_bc_type.unsafe_ptr(),
        mesh.num_faces,
        g,
        min_h,
        inflow_h,
        inflow_hu,
        inflow_hv,
        fstar_scratch,
    )
    launch_sw_vol_lift_2d[NP, NFP](
        ctx,
        q_in,
        mesh.d_elem_invJ.unsafe_ptr(),
        D_ref,
        fstar_scratch,
        mesh.d_elem_inv_2A.unsafe_ptr(),
        mesh.d_elem_faces.unsafe_ptr(),
        mesh.d_elem_face_side.unsafe_ptr(),
        mesh.d_elem_canon_to_ref.unsafe_ptr(),
        mesh.d_face_length.unsafe_ptr(),
        Lift_ref,
        q_a,
        q_b,
        mesh.num_elements,
        g,
        min_h,
        a,
        b,
        cc,
        dt,
        q_out,
    )
