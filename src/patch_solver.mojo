# ======================================================================
# Patch-aware DG solver (multi-rank, one patch per MPI rank)
# ======================================================================
#
# Built on top of `PatchMesh` + `HaloExchange`.  Each rank owns a
# contiguous sub-box of the global cube grid plus a 1-cube-thick
# ghost ring; q is stored for the full local mesh (owned + ghost) but
# only the owned portion is evolved by the RK-stage kernel.  Ghost
# entries are refreshed from the 6 face-neighbour patches via MPI at
# the top of each RK stage so the numerical-flux reads see the right
# data on both sides of every patch-boundary face.
#
# Data layout
# -----------
#   * q / q1 / q2 : [num_local_elements * N_P * NC] Float32 on device
#   * d_owned_elem_ids : scattered indices of owned elements within the
#     flat local-mesh element space (built by PatchMesh).
#
# Kernel dispatch
# ---------------
#   * The RK-stage kernel iterates over `owned_idx` in
#     [0, num_owned_elements), maps `owned_idx` -> local element id via
#     `owned_elem_ids`, and then does the same cooperative-shared-mem
#     flux computation as the single-rank kernel.  Per block:
#         elem_in_block = thread // N_P
#         i             = thread %  N_P
#         owned_idx     = block_idx * ELEMS_PER_BLOCK + elem_in_block
#         e             = owned_elem_ids[owned_idx]   (local elem id)
#
# Phase 3 is sequential: `halo.exchange(q_in)` blocks before each RK
# stage launches.  Phase 4 will split into `rk_stage_interior` /
# `rk_stage_halo` launches on separate streams.
# ======================================================================

from src.reference import N_P, N_F, N_FP, N_D
from src.patch_mesh import PatchMesh
from src.halo_exchange import HaloExchange
from src.solver import Physics, ELEMS_PER_BLOCK, THREADS_PER_BLOCK
from src.nvtx import NvtxContext
from std.gpu import thread_idx, block_idx, barrier
from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu.memory import AddressSpace
from std.math import ceildiv
from std.memory import memcpy, stack_allocation

comptime dtype = DType.float32


# ----------------------------------------------------------------------
# Patch-aware RK-stage kernel.
#
# Identical to `src.solver.rk_stage_kernel` except:
#   * iterates over a `owned_elem_ids` list instead of [0, num_elements);
#   * the block/grid sizing therefore comes from num_owned_elements.
#
# The cooperative shared-memory scheme (per-element internal_flux +
# per-element numerical_flux + Lift/volume-sum + RK update) is
# preserved verbatim, just using the redirected element id `e`.
# ----------------------------------------------------------------------

def rk_stage_kernel_patch[
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
    owned_elem_ids:    UnsafePointer[Int32,   MutAnyOrigin],
    num_owned: Int,
    a: Float32, b: Float32, cc: Float32, dt: Float32,
):
    var tid = Int(thread_idx.x)
    var bid = Int(block_idx.x)
    var elem_in_block = tid // N_P
    var i = tid % N_P
    var owned_idx = bid * ELEMS_PER_BLOCK + elem_in_block
    var valid = owned_idx < num_owned
    var e: Int = 0
    if valid:
        e = Int(owned_elem_ids[owned_idx])

    # Shared memory for cooperative flux computation.  Layout mirrors
    # src.solver.rk_stage_kernel exactly -- see that file for the
    # invariant and block-size rationale.
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

    # ---- Phase 3 + 4: per-component sum and RK update ---------------
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
# PatchSolver
# ----------------------------------------------------------------------

struct PatchSolver[PhysT: Physics](Movable):
    comptime NC = Self.PhysT.NUM_COMPONENTS

    var ctx: DeviceContext
    var physics: Self.PhysT

    var patch: PatchMesh
    var halo:  HaloExchange

    var num_local_elements: Int
    var num_owned_elements: Int
    # All DOF counts below are in element-nodes (not times NC).
    var total_local_dof: Int
    var total_owned_dof: Int
    var total_q_len: Int      # total_local_dof * NC

    # RK-stage buffers sized for the full local mesh (owned + ghost).
    var d_q:  DeviceBuffer[dtype]
    var d_q1: DeviceBuffer[dtype]
    var d_q2: DeviceBuffer[dtype]

    # Reference DG operators (uploaded once).
    var d_D_ref:    DeviceBuffer[dtype]
    var d_Lift_ref: DeviceBuffer[dtype]

    def __init__(
        out self,
        var ctx: DeviceContext,
        var patch: PatchMesh,
        var halo:  HaloExchange,
        var physics: Self.PhysT,
        D_ref: List[Float32],
        Lift_ref: List[Float32],
    ) raises:
        self.ctx = ctx^
        self.patch = patch^
        self.halo  = halo^
        self.physics = physics^
        self.num_local_elements = self.patch.mesh.num_elements
        self.num_owned_elements = self.patch.num_owned_elements
        self.total_local_dof = self.num_local_elements * N_P
        self.total_owned_dof = self.num_owned_elements * N_P
        self.total_q_len = self.total_local_dof * Self.NC

        self.d_q  = self.ctx.enqueue_create_buffer[dtype](self.total_q_len)
        self.d_q1 = self.ctx.enqueue_create_buffer[dtype](self.total_q_len)
        self.d_q2 = self.ctx.enqueue_create_buffer[dtype](self.total_q_len)
        # Zero-initialise so ghost slots are sane until the first halo
        # exchange populates them.
        self.d_q.enqueue_fill(Float32(0.0))
        self.d_q1.enqueue_fill(Float32(0.0))
        self.d_q2.enqueue_fill(Float32(0.0))

        self.d_D_ref = _upload_f32(self.ctx, D_ref)
        self.d_Lift_ref = _upload_f32(self.ctx, Lift_ref)
        self.ctx.synchronize()

    # --- Download helpers --------------------------------------------
    def download_owned_component_with_ids(
        mut self, c: Int,
        mut scalar_host: List[Float32],
        mut global_elem_ids_host: List[Int32],
        nx_global: Int, ny_global: Int, nz_global: Int,
        mut nvtx: NvtxContext,
    ) raises:
        """Download component `c` for owned elements AND compute each
        element's global-mesh ID so a downstream test harness can
        reassemble a whole-domain field from multiple ranks' dumps.

        `scalar_host`          length >= num_owned_elements * N_P
        `global_elem_ids_host`  length >= num_owned_elements
        """
        nvtx.push_range("download_owned_component_with_ids")
        var hbuf = self.ctx.enqueue_create_host_buffer[dtype](
            self.total_q_len
        )
        var h_ids = self.ctx.enqueue_create_host_buffer[DType.int32](
            self.num_owned_elements
        )
        self.ctx.enqueue_copy(hbuf,  self.d_q)
        self.ctx.enqueue_copy(h_ids, self.patch.d_owned_elem_ids)
        self.ctx.synchronize()
        var q_p   = hbuf.unsafe_ptr()
        var ids_p = h_ids.unsafe_ptr()
        var stride = Self.NC

        # Local grid dimensions used to decode cube coords from element id.
        var loc_nx = self.patch.part.nx + 2
        var loc_ny = self.patch.part.ny + 2
        var cx0 = self.patch.part.cx0
        var cy0 = self.patch.part.cy0
        var cz0 = self.patch.part.cz0

        for i in range(self.num_owned_elements):
            var e = Int(ids_p[i])
            # Kuhn tets per cube is 6; N_F is faces-per-tet (4).
            var cube = e // 6
            var tet = e - cube * 6
            var lcz = cube // (loc_nx * loc_ny)
            var rem = cube - lcz * loc_nx * loc_ny
            var lcy = rem // loc_nx
            var lcx = rem - lcy * loc_nx
            # Local -> global: owned cubes are at (lcx in [1, nx+1)).
            var gcx = cx0 + (lcx - 1)
            var gcy = cy0 + (lcy - 1)
            var gcz = cz0 + (lcz - 1)
            var gcube = gcx + nx_global * (gcy + ny_global * gcz)
            global_elem_ids_host[i] = Int32(gcube * 6 + tet)
            for nn in range(N_P):
                scalar_host[i * N_P + nn] = q_p[
                    (e * N_P + nn) * stride + c
                ]
        nvtx.pop_range()

    def download_owned_component(
        mut self, c: Int, mut scalar_host: List[Float32],
        mut nvtx: NvtxContext,
    ) raises:
        """Download component `c` of q for *owned* elements only into
        `scalar_host` (length >= num_owned_elements * N_P)."""
        nvtx.push_range("download_owned_component")
        # Full-buffer download, then host-side gather through
        # owned_elem_ids.  Ghost-slot values are simply skipped.
        var hbuf = self.ctx.enqueue_create_host_buffer[dtype](
            self.total_q_len
        )
        var h_ids = self.ctx.enqueue_create_host_buffer[DType.int32](
            self.num_owned_elements
        )
        self.ctx.enqueue_copy(hbuf,  self.d_q)
        self.ctx.enqueue_copy(h_ids, self.patch.d_owned_elem_ids)
        self.ctx.synchronize()
        var q_p   = hbuf.unsafe_ptr()
        var ids_p = h_ids.unsafe_ptr()
        var stride = Self.NC
        for i in range(self.num_owned_elements):
            var e = Int(ids_p[i])
            for nn in range(N_P):
                scalar_host[i * N_P + nn] = q_p[
                    (e * N_P + nn) * stride + c
                ]
        nvtx.pop_range()

    # --- Internal: one RK stage kernel launch -----------------------
    # `elem_ids_ptr` + `num_elems` select which subset of owned
    # elements the kernel iterates over (interior, halo, or all owned).
    def _launch_rk_stage(
        mut self,
        elem_ids_ptr: UnsafePointer[Int32,   MutAnyOrigin],
        num_elems:    Int,
        q_in_ptr:  UnsafePointer[Float32, MutAnyOrigin],
        q_a_ptr:   UnsafePointer[Float32, MutAnyOrigin],
        q_b_ptr:   UnsafePointer[Float32, MutAnyOrigin],
        q_out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        a: Float32, b: Float32, cc: Float32, dt: Float32,
    ) raises:
        if num_elems == 0:
            return
        comptime kernel = rk_stage_kernel_patch[Self.NC, Self.PhysT]
        self.ctx.enqueue_function[kernel, kernel](
            self.physics,
            q_in_ptr, q_a_ptr, q_b_ptr, q_out_ptr,
            self.patch.mesh.d_elem_invJ.unsafe_ptr(),
            self.patch.mesh.d_elem_inv_6V.unsafe_ptr(),
            self.patch.mesh.d_elem_faces.unsafe_ptr(),
            self.patch.mesh.d_elem_face_side.unsafe_ptr(),
            self.patch.mesh.d_elem_canon_to_ref.unsafe_ptr(),
            self.patch.mesh.d_face_elem.unsafe_ptr(),
            self.patch.mesh.d_face_elem_node.unsafe_ptr(),
            self.patch.mesh.d_face_normal.unsafe_ptr(),
            self.patch.mesh.d_face_area.unsafe_ptr(),
            self.d_D_ref.unsafe_ptr(),
            self.d_Lift_ref.unsafe_ptr(),
            elem_ids_ptr, num_elems,
            a, b, cc, dt,
            grid_dim=ceildiv(num_elems, ELEMS_PER_BLOCK),
            block_dim=THREADS_PER_BLOCK,
        )

    # Run one RK stage with split interior / halo kernels bracketing
    # a non-blocking halo exchange.  Interior compute overlaps with
    # MPI progress on the host; halo compute runs after the ghost q
    # values have been unpacked.
    def _step_stage_overlapped(
        mut self,
        q_in_ptr:  UnsafePointer[Float32, MutAnyOrigin],
        q_a_ptr:   UnsafePointer[Float32, MutAnyOrigin],
        q_b_ptr:   UnsafePointer[Float32, MutAnyOrigin],
        q_out_ptr: UnsafePointer[Float32, MutAnyOrigin],
        a: Float32, b: Float32, cc: Float32, dt: Float32,
        mut nvtx: NvtxContext,
    ) raises:
        # Kick off pack + D->H + MPI_Isend/Irecv; returns while MPI
        # progresses on the host.
        nvtx.push_range("submit_pack")
        self.halo.submit_pack(self.ctx, q_in_ptr)
        nvtx.pop_range()

        # Interior compute: needs only owned q, no ghost data.  Runs
        # on the default stream concurrently with MPI.
        nvtx.push_range("rk_stage_interior")
        self._launch_rk_stage(
            self.patch.d_interior_elem_ids.unsafe_ptr(),
            self.patch.num_interior_elements,
            q_in_ptr, q_a_ptr, q_b_ptr, q_out_ptr,
            a, b, cc, dt,
        )
        nvtx.pop_range()

        # Waitall + H->D + unpack.  After this returns, ghost q is up
        # to date on device.
        nvtx.push_range("complete_exchange")
        self.halo.complete_exchange(self.ctx, q_in_ptr)
        nvtx.pop_range()

        # Halo compute: needs ghost q.
        nvtx.push_range("rk_stage_halo")
        self._launch_rk_stage(
            self.patch.d_halo_elem_ids.unsafe_ptr(),
            self.patch.num_halo_elements,
            q_in_ptr, q_a_ptr, q_b_ptr, q_out_ptr,
            a, b, cc, dt,
        )
        nvtx.pop_range()

    # --- SSPRK3 time step with comm-compute overlap ------------------
    def step_ssprk3(
        mut self, dt: Float32, mut nvtx: NvtxContext,
    ) raises:
        nvtx.push_range("ssprk3_step")
        var p_q  = self.d_q.unsafe_ptr()
        var p_q1 = self.d_q1.unsafe_ptr()
        var p_q2 = self.d_q2.unsafe_ptr()

        # Stage 1: rhs(q) -> q1
        nvtx.push_range("rk_stage_1")
        self._step_stage_overlapped(
            p_q, p_q, p_q, p_q1,
            Float32(1.0), Float32(0.0), Float32(1.0), dt,
            nvtx,
        )
        nvtx.pop_range()

        # Stage 2: rhs(q1) -> q2
        nvtx.push_range("rk_stage_2")
        self._step_stage_overlapped(
            p_q1, p_q, p_q1, p_q2,
            Float32(0.75), Float32(0.25), Float32(0.25), dt,
            nvtx,
        )
        nvtx.pop_range()

        # Stage 3: rhs(q2) -> q
        nvtx.push_range("rk_stage_3")
        self._step_stage_overlapped(
            p_q2, p_q, p_q2, p_q,
            Float32(1.0 / 3.0), Float32(2.0 / 3.0),
            Float32(2.0 / 3.0), dt,
            nvtx,
        )
        nvtx.pop_range()

        nvtx.pop_range()


def _upload_f32(
    mut ctx: DeviceContext, src: List[Float32]
) raises -> DeviceBuffer[dtype]:
    var n = len(src)
    var hbuf = ctx.enqueue_create_host_buffer[dtype](n)
    memcpy(dest=hbuf.unsafe_ptr(), src=src.unsafe_ptr(), count=n)
    var dbuf = ctx.enqueue_create_buffer[dtype](n)
    ctx.enqueue_copy(dbuf, hbuf)
    return dbuf^
