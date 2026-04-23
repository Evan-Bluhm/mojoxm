# ======================================================================
# Host-side 2D DG right-hand side evaluator (scalar advection)
# ======================================================================
#
# Evaluates one RK stage's rhs for linear advection q_t + v . grad q = 0
# on a LocalMesh2D[P] using a ReferenceElement2D[P].  Runs entirely on
# CPU in Float64 -- intended for correctness prototyping before the GPU
# port lands (see project_2d_triangles_scope.md).
#
# Weak-form DG with nodal Lagrange collocation; at each owned node the
# output is dq/dt = -v . grad q (the advection equation's time
# derivative, NOT the conservative flux divergence):
#
#   rhs[e, i] = vol_c - inv_2A * face_c
#
#   vol_c        = sum_j D_ref[k, i, j] * (invJ[k, :] . F(q_j))_k
#   face_c       = sum_faces sign * face_len * Lift_ref[lf, i, r] * fstar_m
#   fstar_m      = v.n * q_upwind   (upwind at the face-local slot m)
#   sign         = +1 on side 0, -1 on side 1
#   F(q)         = (vx * q, vy * q)
#
# Directly analogous to the 3D `rk_stage_kernel` in src/solver.mojo;
# an SSPRK3 driver would do q_new = a * q_a + b * q_b + cc * dt * rhs.
# ======================================================================

from src.local_mesh_2d import LocalMesh2D
from src.reference_2d import (
    ReferenceElement2D, num_tri_nodes_2d, num_edge_nodes,
)


# ----------------------------------------------------------------------
# SSPRK3 time step wrapper (host-side, 2D advection)
# ----------------------------------------------------------------------
# Standard strong-stability-preserving RK3 (Gottlieb-Shu):
#   q1   = q       + dt * L(q)
#   q2   = 3/4 * q + 1/4 * (q1 + dt * L(q1))
#   qnew = 1/3 * q + 2/3 * (q2 + dt * L(q2))
# where L(q) = advection_rhs_2d(q).  Mutates `q` in place on the final
# stage so the caller can chain steps without buffer management.
# ----------------------------------------------------------------------

def ssprk3_step_2d[P: Int](
    mesh: LocalMesh2D[P],
    re: ReferenceElement2D[P],
    vx: Float64, vy: Float64,
    dt: Float64,
    mut q: List[Float64],
    mut scratch_q1: List[Float64],
    mut scratch_q2: List[Float64],
    mut scratch_rhs: List[Float64],
) raises:
    var n = len(q)
    if len(scratch_q1) != n or len(scratch_q2) != n or len(scratch_rhs) != n:
        raise Error("ssprk3_step_2d: scratch buffer size mismatch")

    # Stage 1: q1 = q + dt * L(q)
    advection_rhs_2d[P](mesh, re, vx, vy, q, scratch_rhs)
    for k in range(n):
        scratch_q1[k] = q[k] + dt * scratch_rhs[k]

    # Stage 2: q2 = 3/4 q + 1/4 (q1 + dt * L(q1))
    advection_rhs_2d[P](mesh, re, vx, vy, scratch_q1, scratch_rhs)
    for k in range(n):
        scratch_q2[k] = (
            0.75 * q[k]
            + 0.25 * (scratch_q1[k] + dt * scratch_rhs[k])
        )

    # Stage 3: q <- 1/3 q + 2/3 (q2 + dt * L(q2))
    advection_rhs_2d[P](mesh, re, vx, vy, scratch_q2, scratch_rhs)
    for k in range(n):
        q[k] = (
            (1.0 / 3.0) * q[k]
            + (2.0 / 3.0) * (scratch_q2[k] + dt * scratch_rhs[k])
        )


def advection_rhs_2d[P: Int](
    mesh: LocalMesh2D[P],
    re: ReferenceElement2D[P],
    vx: Float64, vy: Float64,
    q_in: List[Float64],
    mut rhs: List[Float64],
) raises:
    comptime NP = num_tri_nodes_2d(P)
    comptime NFP = num_edge_nodes(P)

    if len(q_in) != mesh.num_elements * NP:
        raise Error("advection_rhs_2d: q_in size mismatch")
    if len(rhs) != mesh.num_elements * NP:
        raise Error("advection_rhs_2d: rhs size mismatch")
    for k in range(len(rhs)):
        rhs[k] = 0.0

    # 1) Upwind fstar on every face.  `face_normal` is side-0's outward
    # normal; on side 1 it points inward.  Upwind for advection:
    #   fstar = vn * q_upwind
    # where vn = v . n (side-0's normal).
    var fstar_buf = List[Float64]()
    for _ in range(mesh.num_faces * NFP):
        fstar_buf.append(0.0)
    for fid in range(mesh.num_faces):
        var e_l = Int(mesh.face_elem[fid * 2 + 0])
        var e_r = Int(mesh.face_elem[fid * 2 + 1])
        var nx = mesh.face_normal[fid * 2 + 0]
        var ny = mesh.face_normal[fid * 2 + 1]
        var vn = vx * nx + vy * ny
        for m in range(NFP):
            var n_l = Int(mesh.face_elem_node[(fid * 2 + 0) * NFP + m])
            var n_r = Int(mesh.face_elem_node[(fid * 2 + 1) * NFP + m])
            var q_l = q_in[e_l * NP + n_l]
            var q_r = q_in[e_r * NP + n_r]
            var f_star: Float64
            if vn >= 0.0:
                f_star = vn * q_l
            else:
                f_star = vn * q_r
            fstar_buf[fid * NFP + m] = f_star

    # 2) Per-element volume + face summations.
    for elem in range(mesh.num_elements):
        var iJ00 = mesh.elem_invJ[elem * 4 + 0]
        var iJ01 = mesh.elem_invJ[elem * 4 + 1]
        var iJ10 = mesh.elem_invJ[elem * 4 + 2]
        var iJ11 = mesh.elem_invJ[elem * 4 + 3]
        var inv_2A = mesh.elem_inv_2A[elem]

        for i in range(NP):
            # Volume: sum_j D_r[i, j] * (invJ @ v q|_j)
            var vol_c: Float64 = 0.0
            for j in range(NP):
                var qj = q_in[elem * NP + j]
                var fx = vx * qj
                var fy = vy * qj
                var fr0 = iJ00 * fx + iJ01 * fy
                var fr1 = iJ10 * fx + iJ11 * fy
                var D_r = re.D_ref[0 * NP * NP + i * NP + j]
                var D_s = re.D_ref[1 * NP * NP + i * NP + j]
                vol_c += fr0 * D_r + fr1 * D_s

            # Face: sum over 3 edges of sign * face_len * Lift[lf, i, m] * fstar
            var face_c: Float64 = 0.0
            for lf in range(3):
                var side = Int(mesh.elem_face_side[elem * 3 + lf])
                var sign: Float64
                if side == 0:
                    sign = 1.0
                else:
                    sign = -1.0
                var fid = Int(mesh.elem_faces[elem * 3 + lf])
                var face_len = mesh.face_length[fid]
                for m in range(NFP):
                    var r = Int(
                        mesh.elem_canon_to_ref[(elem * 3 + lf) * NFP + m]
                    )
                    var Lim = re.Lift_ref[lf * NP * NFP + i * NFP + r]
                    face_c += sign * face_len * Lim * fstar_buf[fid * NFP + m]

            rhs[elem * NP + i] = vol_c - inv_2A * face_c
