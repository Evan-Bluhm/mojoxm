# ======================================================================
# local_mesh_2d_gpu_maxwell.mojo -- 2D Maxwell (NC=6 EM) GPU kernels
# ======================================================================
# State [Ex, Ey, Ez, Bx, By, Bz].  2D embedding of the full 3D
# Maxwell equations: kernels iterate over triangular elements with
# 2D normals (nx, ny, nz=0) so only F^x and F^y enter the volume
# RHS.  All six EM components are still tracked, matching the 3D
# Maxwell layout for consistency.  TM mode (Ex=Ey=Bz=0) and TE mode
# (Ez=Bx=By=0) are both representable.
#
# Flux structure (3D version with z-derivatives dropped):
#   F^x = [0, c^2 Bz, -c^2 By, 0, -Ez, Ey]
#   F^y = [-c^2 Bz, 0, c^2 Bx, Ez, 0, -Ex]
# F^z is identically zero in the 2D pipeline because no kernel computes
# its derivative.
#
# Numerical flux: Rusanov with alpha = c (the unique characteristic
# speed of EM waves).  BC ghost: PEC reflection on BC_WALL,
# prescribed (Ex,Ey,Ez,Bx,By,Bz) on BC_INFLOW, zero-gradient on
# BC_OUTFLOW (the default else branch).
# ======================================================================

from src.local_mesh_2d_gpu import LocalMesh2DGpu
from src.reference_2d import num_tri_nodes_2d, num_edge_nodes
from src.boundary import BC_INTERIOR, BC_WALL, BC_INFLOW
from std.gpu import global_idx
from std.gpu.host import DeviceContext
from std.math import ceildiv


def maxwell_vol_lift_combine_rk_kernel_2d[
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
    c: Float32,
    Jx: Float32,
    Jy: Float32,
    Jz: Float32,
    Mx: Float32,
    My: Float32,
    Mz: Float32,
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

    # ---- Volume RHS contribution (6 components, NC=6).
    var iJ00 = elem_invJ[elem * 4 + 0]
    var iJ01 = elem_invJ[elem * 4 + 1]
    var iJ10 = elem_invJ[elem * 4 + 2]
    var iJ11 = elem_invJ[elem * 4 + 3]
    var c2 = c * c
    var acc0: Float32 = 0.0
    var acc1: Float32 = 0.0
    var acc2: Float32 = 0.0
    var acc3: Float32 = 0.0
    var acc4: Float32 = 0.0
    var acc5: Float32 = 0.0
    for j in range(NP):
        var base = (elem * NP + j) * 6
        var Ex = q_in[base + 0]
        var Ey = q_in[base + 1]
        var Ez = q_in[base + 2]
        var Bx = q_in[base + 3]
        var By = q_in[base + 4]
        var Bz = q_in[base + 5]
        var Fx0 = Float32(0.0)
        var Fx1 = c2 * Bz
        var Fx2 = -c2 * By
        var Fx3 = Float32(0.0)
        var Fx4 = -Ez
        var Fx5 = Ey
        var Fy0 = -c2 * Bz
        var Fy1 = Float32(0.0)
        var Fy2 = c2 * Bx
        var Fy3 = Ez
        var Fy4 = Float32(0.0)
        var Fy5 = -Ex
        var D_r = D_ref[0 * NP * NP + i * NP + j]
        var D_s = D_ref[1 * NP * NP + i * NP + j]
        acc0 += (iJ00 * Fx0 + iJ01 * Fy0) * D_r + (
            iJ10 * Fx0 + iJ11 * Fy0
        ) * D_s
        acc1 += (iJ00 * Fx1 + iJ01 * Fy1) * D_r + (
            iJ10 * Fx1 + iJ11 * Fy1
        ) * D_s
        acc2 += (iJ00 * Fx2 + iJ01 * Fy2) * D_r + (
            iJ10 * Fx2 + iJ11 * Fy2
        ) * D_s
        acc3 += (iJ00 * Fx3 + iJ01 * Fy3) * D_r + (
            iJ10 * Fx3 + iJ11 * Fy3
        ) * D_s
        acc4 += (iJ00 * Fx4 + iJ01 * Fy4) * D_r + (
            iJ10 * Fx4 + iJ11 * Fy4
        ) * D_s
        acc5 += (iJ00 * Fx5 + iJ01 * Fy5) * D_r + (
            iJ10 * Fx5 + iJ11 * Fy5
        ) * D_s

    # ---- Lift contribution.
    var inv_2A = elem_inv_2A[elem]
    var face0: Float32 = 0.0
    var face1: Float32 = 0.0
    var face2: Float32 = 0.0
    var face3: Float32 = 0.0
    var face4: Float32 = 0.0
    var face5: Float32 = 0.0
    for lf in range(3):
        var fid = Int(elem_faces[elem * 3 + lf])
        var side = Int(elem_face_side[elem * 3 + lf])
        var sign: Float32 = Float32(1.0) if side == 0 else Float32(-1.0)
        var flen = face_length[fid]
        for m in range(NFP):
            var r = Int(elem_canon_to_ref[(elem * 3 + lf) * NFP + m])
            var Lim = Lift_ref[lf * NP * NFP + i * NFP + r]
            var sLf = sign * flen * Lim
            var fbase = (fid * NFP + m) * 6
            face0 += sLf * fstar[fbase + 0]
            face1 += sLf * fstar[fbase + 1]
            face2 += sLf * fstar[fbase + 2]
            face3 += sLf * fstar[fbase + 3]
            face4 += sLf * fstar[fbase + 4]
            face5 += sLf * fstar[fbase + 5]

    # Volume + lift, with uniform J / M source per src/maxwell.mojo:
    #   dE/dt += -c^2 * J,  dB/dt += -M
    # (matches the 3D Maxwell.source_term path.)
    var idx = (elem * NP + i) * 6
    var rhs0 = acc0 - inv_2A * face0 - c2 * Jx
    var rhs1 = acc1 - inv_2A * face1 - c2 * Jy
    var rhs2 = acc2 - inv_2A * face2 - c2 * Jz
    var rhs3 = acc3 - inv_2A * face3 - Mx
    var rhs4 = acc4 - inv_2A * face4 - My
    var rhs5 = acc5 - inv_2A * face5 - Mz
    q_out[idx + 0] = a * q_a[idx + 0] + b * q_b[idx + 0] + cc * dt * rhs0
    q_out[idx + 1] = a * q_a[idx + 1] + b * q_b[idx + 1] + cc * dt * rhs1
    q_out[idx + 2] = a * q_a[idx + 2] + b * q_b[idx + 2] + cc * dt * rhs2
    q_out[idx + 3] = a * q_a[idx + 3] + b * q_b[idx + 3] + cc * dt * rhs3
    q_out[idx + 4] = a * q_a[idx + 4] + b * q_b[idx + 4] + cc * dt * rhs4
    q_out[idx + 5] = a * q_a[idx + 5] + b * q_b[idx + 5] + cc * dt * rhs5


def launch_maxwell_vol_lift_2d[
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
    c: Float32,
    Jx: Float32,
    Jy: Float32,
    Jz: Float32,
    Mx: Float32,
    My: Float32,
    Mz: Float32,
    a: Float32,
    b: Float32,
    cc: Float32,
    dt: Float32,
    q_out: UnsafePointer[Float32, MutAnyOrigin],
) raises:
    var total = num_elements * NP
    comptime _kernel = maxwell_vol_lift_combine_rk_kernel_2d[NP, NFP]
    ctx.enqueue_function[_kernel, _kernel](
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
        c,
        Jx,
        Jy,
        Jz,
        Mx,
        My,
        Mz,
        a,
        b,
        cc,
        dt,
        q_out,
        grid_dim=ceildiv(total, 256),
        block_dim=256,
    )


# Maxwell face flux (Rusanov, 6 components).  PEC wall reflects E
# tangentially (En unchanged, Et flipped) and B normally (Bn unchanged,
# Bt flipped); BC_INFLOW pins the ghost to the user-supplied
# (inflow_Ex, inflow_Ey, inflow_Ez, inflow_Bx, inflow_By, inflow_Bz);
# BC_OUTFLOW is zero-gradient transmissive (the default else branch).
def maxwell_face_flux_kernel_2d[
    NP: Int, NFP: Int
](
    q: UnsafePointer[Float32, MutAnyOrigin],
    face_elem: UnsafePointer[Int32, MutAnyOrigin],
    face_elem_node: UnsafePointer[Int32, MutAnyOrigin],
    face_normal: UnsafePointer[Float32, MutAnyOrigin],
    face_bc_type: UnsafePointer[Int32, MutAnyOrigin],
    num_faces: Int,
    c: Float32,
    inflow_Ex: Float32,
    inflow_Ey: Float32,
    inflow_Ez: Float32,
    inflow_Bx: Float32,
    inflow_By: Float32,
    inflow_Bz: Float32,
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
    var l_off = (e_l * NP + n_l) * 6
    var qL0 = q[l_off + 0]
    var qL1 = q[l_off + 1]
    var qL2 = q[l_off + 2]
    var qL3 = q[l_off + 3]
    var qL4 = q[l_off + 4]
    var qL5 = q[l_off + 5]

    var qR0: Float32
    var qR1: Float32
    var qR2: Float32
    var qR3: Float32
    var qR4: Float32
    var qR5: Float32
    if bc_type == BC_INTERIOR:
        var e_r = Int(face_elem[fid * 2 + 1])
        var n_r = Int(face_elem_node[(fid * 2 + 1) * NFP + m])
        var r_off = (e_r * NP + n_r) * 6
        qR0 = q[r_off + 0]
        qR1 = q[r_off + 1]
        qR2 = q[r_off + 2]
        qR3 = q[r_off + 3]
        qR4 = q[r_off + 4]
        qR5 = q[r_off + 5]
    elif bc_type == BC_WALL:
        # PEC reflection.  In 2D the wall normal n = (nx, ny, 0):
        #   * Ex, Ey: split into normal (En*n_hat) and tangential parts;
        #     tangential flips, normal preserved -> q_ghost = 2*En*n - q.
        #   * Ez is PURELY TANGENTIAL (nz=0), so it flips: q_ghost = -q.
        #   * Bx, By: normal component flips, tangential preserved
        #     -> q_ghost = q - 2*Bn*n.
        #   * Bz is purely tangential, preserved unchanged.
        var En = qL0 * nx + qL1 * ny
        var Bn = qL3 * nx + qL4 * ny
        qR0 = Float32(2.0) * En * nx - qL0
        qR1 = Float32(2.0) * En * ny - qL1
        qR2 = -qL2
        qR3 = qL3 - Float32(2.0) * Bn * nx
        qR4 = qL4 - Float32(2.0) * Bn * ny
        qR5 = qL5
    elif bc_type == BC_INFLOW:
        # Prescribed inflow: ghost = user-supplied (E, B).
        qR0 = inflow_Ex
        qR1 = inflow_Ey
        qR2 = inflow_Ez
        qR3 = inflow_Bx
        qR4 = inflow_By
        qR5 = inflow_Bz
    else:
        # BC_OUTFLOW: zero-gradient.
        qR0 = qL0
        qR1 = qL1
        qR2 = qL2
        qR3 = qL3
        qR4 = qL4
        qR5 = qL5

    # F . n components.  Normal vector is (nx, ny, 0).
    # (F . n)_E = c^2 * (B x n);  (F . n)_B = n x E
    # B x n with nz=0:   (-Bz*ny, Bz*nx, Bx*ny - By*nx)
    # n x E with nz=0:   (ny*Ez, -nx*Ez, nx*Ey - ny*Ex)
    var c2 = c * c
    var FEl_x = c2 * (-qL5 * ny)
    var FEl_y = c2 * (qL5 * nx)
    var FEl_z = c2 * (qL3 * ny - qL4 * nx)
    var FBl_x = ny * qL2
    var FBl_y = -nx * qL2
    var FBl_z = nx * qL1 - ny * qL0

    var FEr_x = c2 * (-qR5 * ny)
    var FEr_y = c2 * (qR5 * nx)
    var FEr_z = c2 * (qR3 * ny - qR4 * nx)
    var FBr_x = ny * qR2
    var FBr_y = -nx * qR2
    var FBr_z = nx * qR1 - ny * qR0

    var half = Float32(0.5)
    var alpha = c
    var out = (fid * NFP + m) * 6
    fstar_out[out + 0] = half * (FEl_x + FEr_x) - half * alpha * (qR0 - qL0)
    fstar_out[out + 1] = half * (FEl_y + FEr_y) - half * alpha * (qR1 - qL1)
    fstar_out[out + 2] = half * (FEl_z + FEr_z) - half * alpha * (qR2 - qL2)
    fstar_out[out + 3] = half * (FBl_x + FBr_x) - half * alpha * (qR3 - qL3)
    fstar_out[out + 4] = half * (FBl_y + FBr_y) - half * alpha * (qR4 - qL4)
    fstar_out[out + 5] = half * (FBl_z + FBr_z) - half * alpha * (qR5 - qL5)


def launch_maxwell_face_flux_2d[
    NP: Int, NFP: Int
](
    mut ctx: DeviceContext,
    q: UnsafePointer[Float32, MutAnyOrigin],
    face_elem: UnsafePointer[Int32, MutAnyOrigin],
    face_elem_node: UnsafePointer[Int32, MutAnyOrigin],
    face_normal: UnsafePointer[Float32, MutAnyOrigin],
    face_bc_type: UnsafePointer[Int32, MutAnyOrigin],
    num_faces: Int,
    c: Float32,
    inflow_Ex: Float32,
    inflow_Ey: Float32,
    inflow_Ez: Float32,
    inflow_Bx: Float32,
    inflow_By: Float32,
    inflow_Bz: Float32,
    fstar_out: UnsafePointer[Float32, MutAnyOrigin],
) raises:
    var total = num_faces * NFP
    comptime _kernel = maxwell_face_flux_kernel_2d[NP, NFP]
    ctx.enqueue_function[_kernel, _kernel](
        q,
        face_elem,
        face_elem_node,
        face_normal,
        face_bc_type,
        num_faces,
        c,
        inflow_Ex,
        inflow_Ey,
        inflow_Ez,
        inflow_Bx,
        inflow_By,
        inflow_Bz,
        fstar_out,
        grid_dim=ceildiv(total, 256),
        block_dim=256,
    )


def maxwell_rk_stage_2d[
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
    c: Float32,
    a: Float32,
    b: Float32,
    cc: Float32,
    dt: Float32,
    inflow_Ex: Float32 = Float32(0.0),
    inflow_Ey: Float32 = Float32(0.0),
    inflow_Ez: Float32 = Float32(0.0),
    inflow_Bx: Float32 = Float32(0.0),
    inflow_By: Float32 = Float32(0.0),
    inflow_Bz: Float32 = Float32(0.0),
    Jx: Float32 = Float32(0.0),
    Jy: Float32 = Float32(0.0),
    Jz: Float32 = Float32(0.0),
    Mx: Float32 = Float32(0.0),
    My: Float32 = Float32(0.0),
    Mz: Float32 = Float32(0.0),
) raises:
    # Two launches per stage (face flux + fused vol+lift+RK).  Following
    # the same fusion pattern as the other 2D physics paths.  J / M
    # default to zero so existing callers (vacuum / cavity / plane-wave
    # benches) are source-compatible without edits.
    comptime NP = num_tri_nodes_2d(P)
    comptime NFP = num_edge_nodes(P)
    launch_maxwell_face_flux_2d[NP, NFP](
        ctx,
        q_in,
        mesh.d_face_elem.unsafe_ptr(),
        mesh.d_face_elem_node.unsafe_ptr(),
        mesh.d_face_normal.unsafe_ptr(),
        mesh.d_face_bc_type.unsafe_ptr(),
        mesh.num_faces,
        c,
        inflow_Ex,
        inflow_Ey,
        inflow_Ez,
        inflow_Bx,
        inflow_By,
        inflow_Bz,
        fstar_scratch,
    )
    launch_maxwell_vol_lift_2d[NP, NFP](
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
        c,
        Jx,
        Jy,
        Jz,
        Mx,
        My,
        Mz,
        a,
        b,
        cc,
        dt,
        q_out,
    )
