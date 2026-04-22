# ======================================================================
# GPU DG solver for the 3D advection equation
# ======================================================================
#
# Kernels
# -------
#   rk_stage_kernel    : fused face flux + volume flux + RHS assembly +
#                        RK linear combination.  One kernel launch per
#                        RK stage.
#
# Data layout (all flat, row-major; dtype = Float32 on device)
# -------------------------------------------------------------
#   q, q1, q2 [num_elements * N_P]   (three RK buffers; no scratch
#                                     rhs/face_flux buffers — fused
#                                     into the stage kernel)
#   elem_invJ [num_elements * 9]     (flattened 3x3 inverse Jacobian)
#   elem_inv_6V [num_elements]       ( 1 / (6 * V_e) )
#   elem_faces [num_elements * N_F]  (int32)  global face index per local face
#   elem_face_side [num_elements * N_F] (int32)  0 or 1
#   face_elem [num_faces * 2]        (int32)  element index per side
#   face_elem_node [num_faces * 2 * N_FP] (int32)  per-side element-node idx
#   face_normal [num_faces * 3]
#   face_area   [num_faces]
#   D_ref      [N_D * N_P * N_P]     reference volume operator
#   Lift_ref   [N_F * N_P * N_FP]    reference face-lift operator
# ======================================================================

from reference import N_P, N_F, N_FP, N_D
from nvtx import NvtxContext
from mesh import Mesh
from std.gpu import global_idx
from std.gpu.host import DeviceContext, DeviceBuffer
from std.math import ceildiv, exp
from std.memory import memcpy

comptime dtype = DType.float32
comptime BLOCK = 256

# ----------------------------------------------------------------------
# Fused RK-stage kernel
# ----------------------------------------------------------------------
# One thread per (element, element-node).  Each thread:
#   1. Computes the upwind numerical flux at its 4 incident faces'
#      6 face-nodes on the fly (reads q from both sides of each face).
#      This duplicates the face-flux work between the two elements
#      sharing a face, but removes a full kernel launch per RK stage
#      and eliminates the intermediate `face_flux` buffer.
#   2. Computes the volume DG term.
#   3. Combines q_out = a * q_a + b * q_b + c * dt * rhs (RK stage update).
#
# For an SSPRK3 step, this kernel is launched exactly 3 times (once per
# stage).
# ----------------------------------------------------------------------

def rk_stage_kernel(
    q_in:  UnsafePointer[Float32, MutAnyOrigin],    # q evaluated for rhs
    q_a:   UnsafePointer[Float32, MutAnyOrigin],    # blend arg 1
    q_b:   UnsafePointer[Float32, MutAnyOrigin],    # blend arg 2
    q_out: UnsafePointer[Float32, MutAnyOrigin],    # result
    elem_invJ:         UnsafePointer[Float32, MutAnyOrigin],
    elem_inv_6V:       UnsafePointer[Float32, MutAnyOrigin],
    elem_faces:        UnsafePointer[Int32,   MutAnyOrigin],
    elem_face_side:    UnsafePointer[Int32,   MutAnyOrigin],
    elem_canon_to_ref: UnsafePointer[Int32,   MutAnyOrigin],
    face_elem:         UnsafePointer[Int32,   MutAnyOrigin],
    face_elem_node:    UnsafePointer[Int32,   MutAnyOrigin],
    face_normal:       UnsafePointer[Float32, MutAnyOrigin],
    face_area:         UnsafePointer[Float32, MutAnyOrigin],
    D_ref:             UnsafePointer[Float32, MutAnyOrigin],
    Lift_ref:          UnsafePointer[Float32, MutAnyOrigin],
    num_elements: Int,
    vx: Float32, vy: Float32, vz: Float32,
    a: Float32, b: Float32, c: Float32, dt: Float32,
):
    var work = Int(global_idx.x)
    var total = num_elements * N_P
    if work >= total:
        return
    var e = work // N_P
    var i = work % N_P
    var q_base = e * N_P

    # Reference-direction velocity: v_ref[k] = sum_d (dr_k/dx_d) v_d.
    var invJ_base = e * 9
    var vr0 = elem_invJ[invJ_base + 0] * vx + elem_invJ[invJ_base + 1] * vy + elem_invJ[invJ_base + 2] * vz
    var vr1 = elem_invJ[invJ_base + 3] * vx + elem_invJ[invJ_base + 4] * vy + elem_invJ[invJ_base + 5] * vz
    var vr2 = elem_invJ[invJ_base + 6] * vx + elem_invJ[invJ_base + 7] * vy + elem_invJ[invJ_base + 8] * vz

    # ---- Volume term -----------------------------------------------
    var vol: Float32 = 0.0
    for j in range(N_P):
        var qj = q_in[q_base + j]
        var d0 = D_ref[0 * N_P * N_P + i * N_P + j]
        var d1 = D_ref[1 * N_P * N_P + i * N_P + j]
        var d2 = D_ref[2 * N_P * N_P + i * N_P + j]
        vol += qj * (vr0 * d0 + vr1 * d1 + vr2 * d2)

    # ---- Face term (fused face-flux + lift) ------------------------
    var face_contrib: Float32 = 0.0
    var inv_6V = elem_inv_6V[e]
    for lf in range(N_F):
        var fid  = Int(elem_faces[e * N_F + lf])
        var side = Int(elem_face_side[e * N_F + lf])
        var sign = Float32(1.0) if side == 0 else Float32(-1.0)
        var area = face_area[fid]

        # Face normal (oriented side 0 -> side 1)
        var nx = face_normal[fid * 3 + 0]
        var ny = face_normal[fid * 3 + 1]
        var nz = face_normal[fid * 3 + 2]
        var nc = vx * nx + vy * ny + vz * nz
        var absnc = nc if nc >= 0.0 else -nc
        var nc_plus  = Float32(0.5) * (nc + absnc)   # coefficient on q_l
        var nc_minus = Float32(0.5) * (nc - absnc)   # coefficient on q_r

        var e_l = Int(face_elem[fid * 2 + 0])
        var e_r = Int(face_elem[fid * 2 + 1])

        var accum: Float32 = 0.0
        for m_canon in range(N_FP):
            # Upwind numerical flux at canonical face-node m.
            var n_l = Int(face_elem_node[fid * 2 * N_FP + 0 * N_FP + m_canon])
            var n_r = Int(face_elem_node[fid * 2 * N_FP + 1 * N_FP + m_canon])
            var q_l = q_in[e_l * N_P + n_l]
            var q_r = q_in[e_r * N_P + n_r]
            var fstar = nc_plus * q_l + nc_minus * q_r

            # Map canonical face-node idx -> reference face-local idx
            # for this element, then apply Lift_ref.
            var r = Int(elem_canon_to_ref[(e * N_F + lf) * N_FP + m_canon])
            var Lim = Lift_ref[lf * N_P * N_FP + i * N_FP + r]
            accum += Lim * fstar
        face_contrib += sign * area * accum

    var rhs_val = vol - inv_6V * face_contrib

    # ---- RK linear combination -------------------------------------
    q_out[q_base + i] = a * q_a[q_base + i] + b * q_b[q_base + i] + c * dt * rhs_val


# ----------------------------------------------------------------------
# Initial-condition kernel (periodic Gaussian)
# ----------------------------------------------------------------------
# Evaluates
#   q(x, y, z) = exp( - (dx^2 + dy^2 + dz^2) / (2 sigma^2) )
# where (dx, dy, dz) is the nearest-image displacement from the pulse
# center (cx, cy, cz) under periodic identification with box size
# (Lx, Ly, Lz).  One thread per solution DOF (num_elements * N_P).
# ----------------------------------------------------------------------

def gaussian_ic_kernel(
    q: UnsafePointer[Float32, MutAnyOrigin],
    elem_node_xyz: UnsafePointer[Float32, MutAnyOrigin],
    total: Int,
    cx: Float32, cy: Float32, cz: Float32,
    Lx: Float32, Ly: Float32, Lz: Float32,
    inv_two_sigma2: Float32,
):
    var idx = Int(global_idx.x)
    if idx >= total:
        return
    var px = elem_node_xyz[idx * 3 + 0]
    var py = elem_node_xyz[idx * 3 + 1]
    var pz = elem_node_xyz[idx * 3 + 2]
    var dx = px - cx
    if dx >  Lx * Float32(0.5): dx -= Lx
    if dx < -Lx * Float32(0.5): dx += Lx
    var dy = py - cy
    if dy >  Ly * Float32(0.5): dy -= Ly
    if dy < -Ly * Float32(0.5): dy += Ly
    var dz = pz - cz
    if dz >  Lz * Float32(0.5): dz -= Lz
    if dz < -Lz * Float32(0.5): dz += Lz
    q[idx] = exp(-(dx*dx + dy*dy + dz*dz) * inv_two_sigma2)


# ----------------------------------------------------------------------
# Host-side solver harness
# ----------------------------------------------------------------------

struct Solver:
    var ctx: DeviceContext
    var num_elements: Int
    var num_faces: Int
    var total_dof: Int
    var total_face_dof: Int

    # Device buffers
    var d_q: DeviceBuffer[dtype]
    var d_q1: DeviceBuffer[dtype]
    var d_q2: DeviceBuffer[dtype]

    # Mesh arrays are owned by `mesh`; we borrow pointers from it when
    # launching kernels.  D_ref / Lift_ref are the only buffers the
    # solver constructs itself.
    var mesh: Mesh
    var d_D_ref: DeviceBuffer[dtype]
    var d_Lift_ref: DeviceBuffer[dtype]

    fn __init__(
        out self,
        var ctx: DeviceContext,
        var mesh: Mesh,
        D_ref: List[Float32],
        Lift_ref: List[Float32],
    ) raises:
        self.ctx = ctx^
        self.mesh = mesh^
        self.num_elements = self.mesh.num_elements
        self.num_faces = self.mesh.num_faces
        self.total_dof = self.num_elements * N_P
        self.total_face_dof = self.num_faces * N_FP

        self.d_q = self.ctx.enqueue_create_buffer[dtype](self.total_dof)
        self.d_q1 = self.ctx.enqueue_create_buffer[dtype](self.total_dof)
        self.d_q2 = self.ctx.enqueue_create_buffer[dtype](self.total_dof)

        # Only the reference operators need uploading now -- all the
        # per-element/per-face mesh data was produced directly on the
        # device by Mesh's GPU build kernels.
        self.d_D_ref = _upload_f32(self.ctx, D_ref)
        self.d_Lift_ref = _upload_f32(self.ctx, Lift_ref)
        self.ctx.synchronize()

    fn upload_q(mut self, q_host: List[Float32]) raises:
        var hbuf = self.ctx.enqueue_create_host_buffer[dtype](self.total_dof)
        memcpy(
            dest=hbuf.unsafe_ptr(),
            src=q_host.unsafe_ptr(),
            count=self.total_dof,
        )
        self.ctx.enqueue_copy(self.d_q, hbuf)
        self.ctx.synchronize()

    fn set_initial_gaussian(
        mut self,
        cx: Float32, cy: Float32, cz: Float32,
        Lx: Float32, Ly: Float32, Lz: Float32,
        sigma: Float32,
    ) raises:
        """Fill d_q directly on the device with a periodic Gaussian pulse
        centered at (cx, cy, cz) with standard deviation `sigma`.

        Entirely host-to-device-transfer free: one GPU kernel reads the
        already-resident mesh node coordinates and writes d_q in place.
        """
        var total = self.total_dof
        var inv_two_sigma2 = Float32(1.0) / (Float32(2.0) * sigma * sigma)
        self.ctx.enqueue_function[gaussian_ic_kernel, gaussian_ic_kernel](
            self.d_q.unsafe_ptr(),
            self.mesh.d_elem_node_xyz.unsafe_ptr(),
            total,
            cx, cy, cz,
            Lx, Ly, Lz,
            inv_two_sigma2,
            grid_dim=ceildiv(total, BLOCK),
            block_dim=BLOCK,
        )

    fn download_q(mut self, mut q_host: List[Float32],
                  mut nvtx: NvtxContext) raises:
        nvtx.push_range("download_q")
        var hbuf = self.ctx.enqueue_create_host_buffer[dtype](self.total_dof)
        self.ctx.enqueue_copy(hbuf, self.d_q)
        self.ctx.synchronize()
        var p = hbuf.unsafe_ptr()
        for i in range(self.total_dof):
            q_host[i] = p[i]
        nvtx.pop_range()


    fn _launch_rk_stage(
        mut self,
        q_in_ptr:  UnsafePointer[Float32, MutAnyOrigin],
        q_a_ptr:   UnsafePointer[Float32, MutAnyOrigin],
        q_b_ptr:   UnsafePointer[Float32, MutAnyOrigin],
        q_out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        vx: Float32, vy: Float32, vz: Float32,
        a: Float32, b: Float32, c: Float32, dt: Float32,
    ) raises:
        var work = self.total_dof
        self.ctx.enqueue_function[rk_stage_kernel, rk_stage_kernel](
            q_in_ptr, q_a_ptr, q_b_ptr, q_out_ptr,
            self.mesh.d_elem_invJ.unsafe_ptr(),
            self.mesh.d_elem_inv_6V.unsafe_ptr(),
            self.mesh.d_elem_faces.unsafe_ptr(),
            self.mesh.d_elem_face_side.unsafe_ptr(),
            self.mesh.d_elem_canon_to_ref.unsafe_ptr(),
            self.mesh.d_face_elem.unsafe_ptr(),
            self.mesh.d_face_elem_node.unsafe_ptr(),
            self.mesh.d_face_normal.unsafe_ptr(),
            self.mesh.d_face_area.unsafe_ptr(),
            self.d_D_ref.unsafe_ptr(),
            self.d_Lift_ref.unsafe_ptr(),
            self.num_elements,
            vx, vy, vz,
            a, b, c, dt,
            grid_dim=ceildiv(work, BLOCK),
            block_dim=BLOCK,
        )

    fn step_ssprk3(
        mut self, dt: Float32,
        vx: Float32, vy: Float32, vz: Float32,
        mut nvtx: NvtxContext,
    ) raises:
        # Three kernel launches per timestep (one per SSPRK3 stage).
        nvtx.push_range("ssprk3_step")
        var p_q  = self.d_q.unsafe_ptr()
        var p_q1 = self.d_q1.unsafe_ptr()
        var p_q2 = self.d_q2.unsafe_ptr()
        # Stage 1: q1 = q + dt * rhs(q)
        nvtx.push_range("rk_stage_1")
        self._launch_rk_stage(
            p_q, p_q, p_q, p_q1, vx, vy, vz,
            Float32(1.0), Float32(0.0), Float32(1.0), dt,
        )
        nvtx.pop_range()
        # Stage 2: q2 = 3/4 q + 1/4 q1 + 1/4 dt * rhs(q1)
        nvtx.push_range("rk_stage_2")
        self._launch_rk_stage(
            p_q1, p_q, p_q1, p_q2, vx, vy, vz,
            Float32(0.75), Float32(0.25), Float32(0.25), dt,
        )
        nvtx.pop_range()
        # Stage 3: q = 1/3 q + 2/3 q2 + 2/3 dt * rhs(q2)
        nvtx.push_range("rk_stage_3")
        self._launch_rk_stage(
            p_q2, p_q, p_q2, p_q, vx, vy, vz,
            Float32(1.0 / 3.0), Float32(2.0 / 3.0),
            Float32(2.0 / 3.0), dt,
        )
        nvtx.pop_range()
        nvtx.pop_range()


def _upload_f32(
    mut ctx: DeviceContext, src: List[Float32]
) raises -> DeviceBuffer[dtype]:
    # Queue the host->device transfer without synchronizing; the caller
    # is responsible for issuing a single ctx.synchronize() after all
    # uploads have been enqueued.
    var n = len(src)
    var hbuf = ctx.enqueue_create_host_buffer[dtype](n)
    memcpy(
        dest=hbuf.unsafe_ptr(),
        src=src.unsafe_ptr(),
        count=n,
    )
    var dbuf = ctx.enqueue_create_buffer[dtype](n)
    ctx.enqueue_copy(dbuf, hbuf)
    return dbuf^

def _upload_i32(
    mut ctx: DeviceContext, src: List[Int32]
) raises -> DeviceBuffer[DType.int32]:
    var n = len(src)
    var hbuf = ctx.enqueue_create_host_buffer[DType.int32](n)
    memcpy(
        dest=hbuf.unsafe_ptr(),
        src=src.unsafe_ptr(),
        count=n,
    )
    var dbuf = ctx.enqueue_create_buffer[DType.int32](n)
    ctx.enqueue_copy(dbuf, hbuf)
    return dbuf^
