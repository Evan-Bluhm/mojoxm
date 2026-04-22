# ======================================================================
# Generic GPU DG solver
# ======================================================================
#
# Parameterized by `PhysT`, a physics type that must provide:
#   * comptime NUM_COMPONENTS: Int     -- conserved variables count
#   * def internal_flux(self, q, flux) -> Float32
#   * def numerical_flux(self, q_l, q_r, nx, ny, nz, flux) -> Float32
#
# See `src/advection.mojo` and `src/euler.mojo` for concrete examples.
# Both `internal_flux` and `numerical_flux` are invoked per-node inside
# the RK-stage kernel, so they get inlined away at PhysT specialization
# time and compile into a monolithic kernel with no runtime dispatch.
#
# Data layout (all flat, row-major; dtype = Float32 on device)
# -------------------------------------------------------------
#   q, q1, q2 [num_elements * N_P * NC]  (three RK buffers;
#                                          component-major-inside-node,
#                                          q[(e*N_P + n)*NC + c])
#   elem_invJ [num_elements * 9]         (flattened 3x3 inverse Jacobian)
#   elem_inv_6V [num_elements]           ( 1 / (6 * V_e) )
#   elem_faces [num_elements * N_F]      (int32)
#   elem_face_side [num_elements * N_F]  (int32)  0 or 1
#   face_elem [num_faces * 2]            (int32)
#   face_elem_node [num_faces * 2 * N_FP](int32)
#   face_normal [num_faces * 3]
#   face_area   [num_faces]
#   D_ref      [N_D * N_P * N_P]         reference volume operator
#   Lift_ref   [N_F * N_P * N_FP]        reference face-lift operator
# ======================================================================

from reference import N_P, N_F, N_FP, N_D
from nvtx import NvtxContext
from mesh import Mesh
from std.gpu import thread_idx, block_idx, barrier
from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu.host.device_context import DevicePassable
from std.gpu.memory import AddressSpace
from std.math import ceildiv
from std.memory import memcpy, stack_allocation

comptime dtype = DType.float32

# One block handles ELEMS_PER_BLOCK elements cooperatively.  Each
# element gets N_P threads (one per nodal DOF).  At N_P = 10 and 16
# elements the block is 160 threads = 5 warps.  The block size is a
# multiple of N_P so the (element, node-in-element) mapping is
# contiguous and there's no cross-block element split.
comptime ELEMS_PER_BLOCK = 16
comptime THREADS_PER_BLOCK = ELEMS_PER_BLOCK * N_P


# ----------------------------------------------------------------------
# Physics trait
# ----------------------------------------------------------------------
# A hyperbolic-system physics type that the generic Solver specializes
# on.  Concrete implementations (Advection, Euler) define the three
# entries below.  `internal_flux` writes the [d * NC + c] physical flux
# tensor; `numerical_flux` writes the NC-vector upwind flux at a face
# given a (nx, ny, nz) outward unit normal.  Both return a signal speed
# (used by CFL-controlling callers); `numerical_flux`'s return is the
# max |wave speed| at the interface.
# ----------------------------------------------------------------------

trait Physics(Copyable, Movable, ImplicitlyDestructible, DevicePassable):
    comptime NUM_COMPONENTS: Int

    def internal_flux(
        self,
        q:    UnsafePointer[Float32, MutAnyOrigin],
        flux: UnsafePointer[Float32, MutAnyOrigin],
    ) -> Float32:
        ...

    def numerical_flux(
        self,
        q_l:  UnsafePointer[Float32, MutAnyOrigin],
        q_r:  UnsafePointer[Float32, MutAnyOrigin],
        nx: Float32, ny: Float32, nz: Float32,
        flux: UnsafePointer[Float32, MutAnyOrigin],
    ) -> Float32:
        ...


# ----------------------------------------------------------------------
# Fused RK-stage kernel (generic over NC and PhysT)
# ----------------------------------------------------------------------
# Block layout: each block owns ELEMS_PER_BLOCK elements, each element
# owns N_P threads (one per nodal DOF).  Within one element the 10
# threads cooperate through shared memory so that per-element work
# that doesn't depend on (i) is done exactly once instead of 10 times:
#
#   Phase 1 (per (element, node_i)):
#       - Each thread computes the physical flux tensor F_d_c for its
#         own node's q and writes it to shared memory.
#   Phase 2 (strided across faces within one element):
#       - The 10 threads of the element cooperatively compute the 24
#         numerical fluxes at the element's (4 faces) x (6 face-nodes),
#         2-3 solves per thread, and write them to shared memory.
#   Phase 3 (per (element, node_i)):
#       - Each thread sums its own volume term (D_ref * shared vol flux)
#         and face term (Lift_ref * shared face flux).
#   Phase 4 (per (element, node_i)):
#       - Each thread writes the RK linear combination into q_out.
#
# This removes the 10x redundancy in `internal_flux` / `numerical_flux`
# calls that the previous "one thread does everything" layout had, and
# replaces 240 scattered q_l / q_r global loads per element (duplicated
# across the 10 threads) with 24 shared-memory broadcasts.
# ----------------------------------------------------------------------

def rk_stage_kernel[
    NC: Int, PhysT: Physics,
](
    physics: PhysT,
    q_in:  UnsafePointer[Float32, MutAnyOrigin],
    q_a:   UnsafePointer[Float32, MutAnyOrigin],
    q_b:   UnsafePointer[Float32, MutAnyOrigin],
    q_out: UnsafePointer[Float32, MutAnyOrigin],
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
    a: Float32, b: Float32, cc: Float32, dt: Float32,
):
    var tid = Int(thread_idx.x)
    var bid = Int(block_idx.x)
    var elem_in_block = tid // N_P
    var i = tid % N_P
    var e = bid * ELEMS_PER_BLOCK + elem_in_block
    var valid = e < num_elements

    # Shared memory for cooperative storage of per-element flux values.
    # vol_flux layout: [elem_in_block][node_j][d * NC + c]  size NC * 3 per node
    # face_flux layout: [elem_in_block][lf][m_canon][c]     size NC per face-node
    var shared_vol_flux = stack_allocation[
        ELEMS_PER_BLOCK * N_P * N_D * NC,
        Scalar[DType.float32],
        address_space=AddressSpace.SHARED,
    ]()
    var shared_face_flux = stack_allocation[
        ELEMS_PER_BLOCK * N_F * N_FP * NC,
        Scalar[DType.float32],
        address_space=AddressSpace.SHARED,
    ]()

    # ---- Phase 1: one internal_flux per (element, node) ------------
    if valid:
        var q_my_ptr = q_in + (e * N_P + i) * NC
        var my_flux_dc = InlineArray[Float32, NC * 3](fill=0.0)
        var my_flux_dc_p = rebind[UnsafePointer[Float32, MutAnyOrigin]](
            my_flux_dc.unsafe_ptr()
        )
        _ = physics.internal_flux(
            rebind[UnsafePointer[Float32, MutAnyOrigin]](q_my_ptr),
            my_flux_dc_p,
        )
        var vol_base = (elem_in_block * N_P + i) * N_D * NC
        for k in range(N_D * NC):
            shared_vol_flux[vol_base + k] = my_flux_dc[k]

    # ---- Phase 2: cooperative numerical_flux across element faces --
    # 24 face-nodes per element, 10 threads per element. Strided
    # assignment: thread i handles face-nodes {i, i+10, i+20}.  Since
    # 24 = 2 * 10 + 4, threads 0..3 each do 3 solves, threads 4..9
    # each do 2.  Skip when fn_idx >= N_F * N_FP.
    if valid:
        comptime FN_TOTAL = N_F * N_FP
        for k in range(3):
            var fn_idx = i + k * N_P
            if fn_idx < FN_TOTAL:
                var lf = fn_idx // N_FP
                var m_canon = fn_idx % N_FP

                var fid = Int(elem_faces[e * N_F + lf])
                var nx = face_normal[fid * 3 + 0]
                var ny = face_normal[fid * 3 + 1]
                var nz = face_normal[fid * 3 + 2]
                var e_l = Int(face_elem[fid * 2 + 0])
                var e_r = Int(face_elem[fid * 2 + 1])
                var n_l = Int(face_elem_node[fid * 2 * N_FP + 0 * N_FP + m_canon])
                var n_r = Int(face_elem_node[fid * 2 * N_FP + 1 * N_FP + m_canon])

                var q_l_ptr = q_in + (e_l * N_P + n_l) * NC
                var q_r_ptr = q_in + (e_r * N_P + n_r) * NC
                var fstar = InlineArray[Float32, NC](fill=0.0)
                var fstar_p = rebind[UnsafePointer[Float32, MutAnyOrigin]](
                    fstar.unsafe_ptr()
                )
                _ = physics.numerical_flux(
                    rebind[UnsafePointer[Float32, MutAnyOrigin]](q_l_ptr),
                    rebind[UnsafePointer[Float32, MutAnyOrigin]](q_r_ptr),
                    nx, ny, nz, fstar_p,
                )
                var face_base = (
                    (elem_in_block * N_F + lf) * N_FP + m_canon
                ) * NC
                for c in range(NC):
                    shared_face_flux[face_base + c] = fstar[c]

    barrier()

    if not valid:
        return

    # ---- Phase 3 + 4: per-component, accumulate and write ----------
    # Loop order flipped so each component is produced and flushed
    # before the next -- the register allocator keeps only one scalar
    # accumulator live at a time.  On Euler (NC=5) this also slightly
    # lowers per-load redundancy visible to L1.
    var invJ_base = e * 9
    var iJ00 = elem_invJ[invJ_base + 0]
    var iJ01 = elem_invJ[invJ_base + 1]
    var iJ02 = elem_invJ[invJ_base + 2]
    var iJ10 = elem_invJ[invJ_base + 3]
    var iJ11 = elem_invJ[invJ_base + 4]
    var iJ12 = elem_invJ[invJ_base + 5]
    var iJ20 = elem_invJ[invJ_base + 6]
    var iJ21 = elem_invJ[invJ_base + 7]
    var iJ22 = elem_invJ[invJ_base + 8]
    var inv_6V = elem_inv_6V[e]
    var out_base = (e * N_P + i) * NC

    for c in range(NC):
        var vol_c: Float32 = 0.0
        for j in range(N_P):
            var shared_base = (elem_in_block * N_P + j) * N_D * NC
            var fx = shared_vol_flux[shared_base + 0 * NC + c]
            var fy = shared_vol_flux[shared_base + 1 * NC + c]
            var fz = shared_vol_flux[shared_base + 2 * NC + c]
            var fr0 = iJ00 * fx + iJ01 * fy + iJ02 * fz
            var fr1 = iJ10 * fx + iJ11 * fy + iJ12 * fz
            var fr2 = iJ20 * fx + iJ21 * fy + iJ22 * fz
            var d0 = D_ref[0 * N_P * N_P + i * N_P + j]
            var d1 = D_ref[1 * N_P * N_P + i * N_P + j]
            var d2 = D_ref[2 * N_P * N_P + i * N_P + j]
            vol_c += fr0 * d0 + fr1 * d1 + fr2 * d2

        var face_c: Float32 = 0.0
        for lf in range(N_F):
            var side = Int(elem_face_side[e * N_F + lf])
            var sign = Float32(1.0) if side == 0 else Float32(-1.0)
            var fid = Int(elem_faces[e * N_F + lf])
            var area = face_area[fid]
            for m_canon in range(N_FP):
                var r = Int(
                    elem_canon_to_ref[(e * N_F + lf) * N_FP + m_canon]
                )
                var Lim = Lift_ref[lf * N_P * N_FP + i * N_FP + r]
                var face_base = (
                    (elem_in_block * N_F + lf) * N_FP + m_canon
                ) * NC
                face_c += sign * area * Lim * shared_face_flux[
                    face_base + c
                ]

        var rhs_val = vol_c - inv_6V * face_c
        q_out[out_base + c] = (
            a * q_a[out_base + c]
            + b * q_b[out_base + c]
            + cc * dt * rhs_val
        )


# ----------------------------------------------------------------------
# Host-side solver harness
# ----------------------------------------------------------------------

struct Solver[PhysT: Physics](Movable):
    comptime NC = Self.PhysT.NUM_COMPONENTS

    var ctx: DeviceContext
    var physics: Self.PhysT

    var num_elements: Int
    var num_faces: Int
    var total_dof: Int             # num_elements * N_P (nodes, not incl. components)
    var total_q_len: Int           # total_dof * NC (element count of each q buffer)
    var total_face_dof: Int

    # Device buffers (q buffers carry all NC components).
    var d_q: DeviceBuffer[dtype]
    var d_q1: DeviceBuffer[dtype]
    var d_q2: DeviceBuffer[dtype]

    # Mesh arrays are owned by `mesh`; we borrow pointers from it when
    # launching kernels.  D_ref / Lift_ref are the only buffers the
    # solver constructs itself.
    var mesh: Mesh
    var d_D_ref: DeviceBuffer[dtype]
    var d_Lift_ref: DeviceBuffer[dtype]

    def __init__(
        out self,
        var ctx: DeviceContext,
        var mesh: Mesh,
        var physics: Self.PhysT,
        D_ref: List[Float32],
        Lift_ref: List[Float32],
    ) raises:
        self.ctx = ctx^
        self.mesh = mesh^
        self.physics = physics^
        self.num_elements = self.mesh.num_elements
        self.num_faces = self.mesh.num_faces
        self.total_dof = self.num_elements * N_P
        self.total_q_len = self.total_dof * Self.NC
        self.total_face_dof = self.num_faces * N_FP

        self.d_q  = self.ctx.enqueue_create_buffer[dtype](self.total_q_len)
        self.d_q1 = self.ctx.enqueue_create_buffer[dtype](self.total_q_len)
        self.d_q2 = self.ctx.enqueue_create_buffer[dtype](self.total_q_len)

        # Only the reference operators need uploading now -- all the
        # per-element / per-face mesh data was produced directly on the
        # device by Mesh's GPU build kernels.
        self.d_D_ref = _upload_f32(self.ctx, D_ref)
        self.d_Lift_ref = _upload_f32(self.ctx, Lift_ref)
        self.ctx.synchronize()

    def upload_q(mut self, q_host: List[Float32]) raises:
        var hbuf = self.ctx.enqueue_create_host_buffer[dtype](self.total_q_len)
        memcpy(
            dest=hbuf.unsafe_ptr(),
            src=q_host.unsafe_ptr(),
            count=self.total_q_len,
        )
        self.ctx.enqueue_copy(self.d_q, hbuf)
        self.ctx.synchronize()

    def download_q(mut self, mut q_host: List[Float32],
                  mut nvtx: NvtxContext) raises:
        """Download the full multi-component q into `q_host`.

        `q_host` must have at least `total_q_len` entries allocated.
        """
        nvtx.push_range("download_q")
        var hbuf = self.ctx.enqueue_create_host_buffer[dtype](self.total_q_len)
        self.ctx.enqueue_copy(hbuf, self.d_q)
        self.ctx.synchronize()
        var p = hbuf.unsafe_ptr()
        for i in range(self.total_q_len):
            q_host[i] = p[i]
        nvtx.pop_range()

    def download_component(mut self, c: Int, mut scalar_host: List[Float32],
                          mut nvtx: NvtxContext) raises:
        """Download just component `c` of q into `scalar_host` (size
        total_dof).  Does one full device->host copy and then strides
        through the result.  For NC=1 this is an equivalent full copy.
        """
        nvtx.push_range("download_component")
        var hbuf = self.ctx.enqueue_create_host_buffer[dtype](self.total_q_len)
        self.ctx.enqueue_copy(hbuf, self.d_q)
        self.ctx.synchronize()
        var p = hbuf.unsafe_ptr()
        var stride = Self.NC
        for i in range(self.total_dof):
            scalar_host[i] = p[i * stride + c]
        nvtx.pop_range()

    def _launch_rk_stage(
        mut self,
        q_in_ptr:  UnsafePointer[Float32, MutAnyOrigin],
        q_a_ptr:   UnsafePointer[Float32, MutAnyOrigin],
        q_b_ptr:   UnsafePointer[Float32, MutAnyOrigin],
        q_out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        a: Float32, b: Float32, cc: Float32, dt: Float32,
    ) raises:
        comptime kernel = rk_stage_kernel[Self.NC, Self.PhysT]
        self.ctx.enqueue_function[kernel, kernel](
            self.physics,
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
            a, b, cc, dt,
            grid_dim=ceildiv(self.num_elements, ELEMS_PER_BLOCK),
            block_dim=THREADS_PER_BLOCK,
        )

    def step_ssprk3(
        mut self, dt: Float32, mut nvtx: NvtxContext,
    ) raises:
        # Three kernel launches per timestep (one per SSPRK3 stage).
        nvtx.push_range("ssprk3_step")
        var p_q  = self.d_q.unsafe_ptr()
        var p_q1 = self.d_q1.unsafe_ptr()
        var p_q2 = self.d_q2.unsafe_ptr()
        # Stage 1: q1 = q + dt * rhs(q)
        nvtx.push_range("rk_stage_1")
        self._launch_rk_stage(
            p_q, p_q, p_q, p_q1,
            Float32(1.0), Float32(0.0), Float32(1.0), dt,
        )
        nvtx.pop_range()
        # Stage 2: q2 = 3/4 q + 1/4 q1 + 1/4 dt * rhs(q1)
        nvtx.push_range("rk_stage_2")
        self._launch_rk_stage(
            p_q1, p_q, p_q1, p_q2,
            Float32(0.75), Float32(0.25), Float32(0.25), dt,
        )
        nvtx.pop_range()
        # Stage 3: q = 1/3 q + 2/3 q2 + 2/3 dt * rhs(q2)
        nvtx.push_range("rk_stage_3")
        self._launch_rk_stage(
            p_q2, p_q, p_q2, p_q,
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
