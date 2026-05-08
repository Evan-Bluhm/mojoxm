# ======================================================================
# MPI halo exchange for per-rank q data
# ======================================================================
#
# The patch mesh stores q for both owned and ghost elements in one
# flat array of length num_local_elements * N_P * NC.  Owned entries
# are kept up-to-date by the solver; ghost entries are populated once
# per RK stage by MPI exchange with the 6 face-neighbour patches.
#
# Per neighbour direction d in [-x, +x, -y, +y, -z, +z] we keep:
#
#   * a "pack list" -- `send_count[d]` owned element IDs that lie on
#     our d-boundary.  A pack kernel copies their q values into
#     `d_send_buf[d]` in deterministic (lcy, lcz) order.
#
#   * an "unpack list" -- the same count of ghost element IDs on our
#     d-ghost ring.  After MPI_Isend / MPI_Irecv complete, an unpack
#     kernel copies `d_recv_buf[d]` back into q at those IDs.
#
# Pack and unpack lists are built in matching order on both sides of
# each neighbour pair so no serialised indexing metadata crosses the
# wire -- only the raw Float32 payload.
#
# This module currently does a *sequential* exchange (pack -> MPI ->
# unpack), single-stream.  Phase 4 adds the comm-stream / compute
# overlap once this is proven correct.
# ======================================================================

from src import mpi
from src.reference import N_P
from src.partition import Partition
from src.boundary import BoundaryConditions, BC_INTERIOR
from std.gpu import global_idx
from std.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from std.math import ceildiv
from std.memory import memcpy


# ----------------------------------------------------------------------
# Kernel: remap an array of element-id VALUES through a permutation.
# ----------------------------------------------------------------------


def remap_ids_kernel(
    ids: UnsafePointer[Int32, MutAnyOrigin],
    perm: UnsafePointer[Int32, MutAnyOrigin],  # perm[old] = new
    n: Int,
):
    var idx = Int(global_idx.x)
    if idx >= n:
        return
    var old_v = Int(ids[idx])
    if old_v >= 0:
        ids[idx] = perm[old_v]


# Kuhn tets per Cartesian cube (keep in sync with src/mesh.mojo).
comptime KUHN_TETS = 6
comptime halo_f = DType.float32
comptime halo_i = DType.int32
comptime HALO_BLOCK = 256

# MPI tag encoding: (axis << 1) | sign (0 for minus, 1 for plus), from
# the sender's viewpoint.  A recv from the d-neighbour expects the
# opposite-direction tag (XOR 1 on the low bit).
comptime TAG_MINUS_X = 0
comptime TAG_PLUS_X = 1
comptime TAG_MINUS_Y = 2
comptime TAG_PLUS_Y = 3
comptime TAG_MINUS_Z = 4
comptime TAG_PLUS_Z = 5


# Direction ordering for all 6-entry arrays below:
#   [0] -x   [1] +x   [2] -y   [3] +y   [4] -z   [5] +z


# ----------------------------------------------------------------------
# Pack/unpack kernels.  Both operate over `count * N_P` threads: one
# thread per (element-in-list, node-in-element) pair, and inner-loop
# over the NC components.
# ----------------------------------------------------------------------


def pack_kernel(
    send_buf: UnsafePointer[Float32, MutAnyOrigin],
    q: UnsafePointer[Float32, MutAnyOrigin],
    indices: UnsafePointer[Int32, MutAnyOrigin],
    count: Int,
    NC: Int,
):
    var idx = Int(global_idx.x)
    var total = count * N_P
    if idx >= total:
        return
    var i = idx // N_P
    var nn = idx % N_P
    var elem = Int(indices[i])
    for c in range(NC):
        send_buf[(i * N_P + nn) * NC + c] = q[(elem * N_P + nn) * NC + c]


def unpack_kernel(
    q: UnsafePointer[Float32, MutAnyOrigin],
    recv_buf: UnsafePointer[Float32, MutAnyOrigin],
    indices: UnsafePointer[Int32, MutAnyOrigin],
    count: Int,
    NC: Int,
):
    var idx = Int(global_idx.x)
    var total = count * N_P
    if idx >= total:
        return
    var i = idx // N_P
    var nn = idx % N_P
    var elem = Int(indices[i])
    for c in range(NC):
        q[(elem * N_P + nn) * NC + c] = recv_buf[(i * N_P + nn) * NC + c]


# ----------------------------------------------------------------------
# Per-face-ring index-list computation on the host.
#
# `fill` must be True to actually write indices; False only counts.
# Returns the total number of element IDs written.  The list iterates
# the face ring in a canonical order so that the sender's and the
# receiver's lists match exactly when laid out on opposite sides of a
# neighbour pair.
# ----------------------------------------------------------------------


def _pack_list_for_dir(
    mut out: List[Int32],
    nx: Int,
    ny: Int,
    nz: Int,
    loc_nx: Int,
    loc_ny: Int,
    axis: Int,
    sign: Int,  # axis in [0,1,2]; sign in {-1, +1}
    is_ghost: Bool,  # False = owned boundary, True = ghost ring
) raises -> Int:
    # Determine which local cube coordinate is fixed, and which range.
    # Owned -x face has fixed lcx = 1; ghost -x ring has fixed lcx = 0.
    # +x side: owned fixed lcx = nx; ghost fixed lcx = nx+1.
    var fixed_lcx = -1
    var fixed_lcy = -1
    var fixed_lcz = -1
    if axis == 0:
        if sign < 0:
            fixed_lcx = 0 if is_ghost else 1
        else:
            fixed_lcx = (nx + 1) if is_ghost else nx
    elif axis == 1:
        if sign < 0:
            fixed_lcy = 0 if is_ghost else 1
        else:
            fixed_lcy = (ny + 1) if is_ghost else ny
    else:  # axis == 2
        if sign < 0:
            fixed_lcz = 0 if is_ghost else 1
        else:
            fixed_lcz = (nz + 1) if is_ghost else nz

    out.clear()
    var count = 0
    if axis == 0:
        var fx = fixed_lcx
        for lcy in range(1, ny + 1):
            for lcz in range(1, nz + 1):
                var cell = fx + loc_nx * (lcy + loc_ny * lcz)
                for t in range(KUHN_TETS):
                    out.append(Int32(cell * KUHN_TETS + t))
                    count += 1
    elif axis == 1:
        var fy = fixed_lcy
        for lcx in range(1, nx + 1):
            for lcz in range(1, nz + 1):
                var cell = lcx + loc_nx * (fy + loc_ny * lcz)
                for t in range(KUHN_TETS):
                    out.append(Int32(cell * KUHN_TETS + t))
                    count += 1
    else:  # axis == 2
        var fz = fixed_lcz
        for lcx in range(1, nx + 1):
            for lcy in range(1, ny + 1):
                var cell = lcx + loc_nx * (lcy + loc_ny * fz)
                for t in range(KUHN_TETS):
                    out.append(Int32(cell * KUHN_TETS + t))
                    count += 1
    return count


# ----------------------------------------------------------------------
# HaloExchange
# ----------------------------------------------------------------------


struct HaloExchange(Movable):
    var nc: Int
    # Runtime flag: if True we hand device pointers directly to MPI
    # (saves 12 D/H copies per exchange).  Otherwise we stage through
    # pinned HostBuffers.
    var cuda_aware: Bool
    # True if any of the 6 rings has a non-zero count.  At np=1 with a
    # single-patch mesh (no ghost ring), every ring is empty and the
    # whole exchange -- pack, MPI, unpack -- becomes a no-op.  We cache
    # this flag so submit_pack / complete_exchange can short-circuit
    # without iterating the ring list.
    var has_halo: Bool

    # 6 neighbour ranks in the order [-x, +x, -y, +y, -z, +z].
    var neighbour: List[Int]
    # Element count in each of the 6 face rings (owned == ghost).
    var ring_count: List[Int]
    # Per-direction "skip MPI" flag.  True for directions where this
    # rank sits on a non-periodic global boundary -- there's no peer
    # to exchange with, so we skip pack / Isend / Irecv / unpack (the
    # boundary mesh's BC faces handle q for those owned elements, and
    # the ghost ring on that side is never read by the solver).
    var skip_mpi: List[Bool]

    # Per-direction device buffers.  Index [d] in each list refers to
    # the same (axis, sign) direction encoded above.
    var d_pack_idx: List[DeviceBuffer[halo_i]]
    var d_unpack_idx: List[DeviceBuffer[halo_i]]
    var d_send_buf: List[DeviceBuffer[halo_f]]
    var d_recv_buf: List[DeviceBuffer[halo_f]]
    # Pinned host staging buffers.  Populated only when cuda_aware is
    # False (skipping the allocation entirely when we don't need it).
    var h_send_buf: List[HostBuffer[halo_f]]
    var h_recv_buf: List[HostBuffer[halo_f]]

    # 12 MPI_Request handles stored contiguously as Int64 (OpenMPI
    # MPI_Request is an opaque 8-byte pointer).  Layout:
    #   [0..6):  send requests
    #   [6..12): recv requests
    var req_storage: UnsafePointer[Int64, MutExternalOrigin]

    def __init__(
        out self,
        mut ctx: DeviceContext,
        part: Partition,
        nc: Int,
        d_perm: UnsafePointer[Int32, MutAnyOrigin],
        bcs: BoundaryConditions = BoundaryConditions.periodic(),
    ) raises:
        self.nc = nc
        self.cuda_aware = mpi.is_cuda_aware()

        self.neighbour = List[Int]()
        self.neighbour.append(part.neighbour_minus_x)
        self.neighbour.append(part.neighbour_plus_x)
        self.neighbour.append(part.neighbour_minus_y)
        self.neighbour.append(part.neighbour_plus_y)
        self.neighbour.append(part.neighbour_minus_z)
        self.neighbour.append(part.neighbour_plus_z)

        # Classify each of the 6 directions as "real peer" or "global
        # non-periodic boundary".  A direction is BC-skipped only when
        # (a) the user marked that axis/side non-periodic AND (b) this
        # rank actually sits on that global face.  Interior-of-the-
        # partition edges still need to exchange with their neighbour.
        self.skip_mpi = List[Bool]()
        self.skip_mpi.append(part.rx == 0 and bcs.bc_x_lo != BC_INTERIOR)
        self.skip_mpi.append(
            part.rx == part.px - 1 and bcs.bc_x_hi != BC_INTERIOR
        )
        self.skip_mpi.append(part.ry == 0 and bcs.bc_y_lo != BC_INTERIOR)
        self.skip_mpi.append(
            part.ry == part.py - 1 and bcs.bc_y_hi != BC_INTERIOR
        )
        self.skip_mpi.append(part.rz == 0 and bcs.bc_z_lo != BC_INTERIOR)
        self.skip_mpi.append(
            part.rz == part.pz - 1 and bcs.bc_z_hi != BC_INTERIOR
        )

        self.ring_count = List[Int]()
        self.d_pack_idx = List[DeviceBuffer[halo_i]]()
        self.d_unpack_idx = List[DeviceBuffer[halo_i]]()
        self.d_send_buf = List[DeviceBuffer[halo_f]]()
        self.d_recv_buf = List[DeviceBuffer[halo_f]]()
        self.h_send_buf = List[HostBuffer[halo_f]]()
        self.h_recv_buf = List[HostBuffer[halo_f]]()
        self.req_storage = alloc[Int64](12)

        # Single-patch fast path: the Mesh at np=1 has no ghost ring,
        # so no halo exchange is ever needed.  Skip all the per-direction
        # pack/unpack setup and record has_halo=False so submit_pack /
        # complete_exchange short-circuit at runtime.
        if part.px * part.py * part.pz == 1:
            for _ in range(6):
                self.ring_count.append(0)
            self.has_halo = False
            return

        var nx = part.nx
        var ny = part.ny
        var nz = part.nz
        var loc_nx = nx + 2
        var loc_ny = ny + 2

        var axes = [0, 0, 1, 1, 2, 2]
        var signs = [-1, 1, -1, 1, -1, 1]

        for d in range(6):
            var owned_list = List[Int32]()
            var ghost_list = List[Int32]()
            var n_owned = _pack_list_for_dir(
                owned_list,
                nx,
                ny,
                nz,
                loc_nx,
                loc_ny,
                axes[d],
                signs[d],
                is_ghost=False,
            )
            var n_ghost = _pack_list_for_dir(
                ghost_list,
                nx,
                ny,
                nz,
                loc_nx,
                loc_ny,
                axes[d],
                signs[d],
                is_ghost=True,
            )
            if n_owned != n_ghost:
                raise Error("HaloExchange: owned/ghost ring size mismatch")
            self.ring_count.append(n_owned)
            # Upload the build-time (pre-permutation) IDs, then remap
            # through the Mesh element permutation so references
            # match the reordered mesh arrays.
            var d_pack = _upload_i32(ctx, owned_list)
            var d_unpack = _upload_i32(ctx, ghost_list)
            ctx.enqueue_function[remap_ids_kernel](
                d_pack.unsafe_ptr(),
                d_perm,
                n_owned,
                grid_dim=ceildiv(n_owned, HALO_BLOCK),
                block_dim=HALO_BLOCK,
            )
            ctx.enqueue_function[remap_ids_kernel](
                d_unpack.unsafe_ptr(),
                d_perm,
                n_ghost,
                grid_dim=ceildiv(n_ghost, HALO_BLOCK),
                block_dim=HALO_BLOCK,
            )
            self.d_pack_idx.append(d_pack^)
            self.d_unpack_idx.append(d_unpack^)
            var buf_floats = n_owned * N_P * nc
            self.d_send_buf.append(
                ctx.enqueue_create_buffer[halo_f](buf_floats)
            )
            self.d_recv_buf.append(
                ctx.enqueue_create_buffer[halo_f](buf_floats)
            )
            # Host staging only needed for non-CUDA-aware MPI.
            if not self.cuda_aware:
                self.h_send_buf.append(
                    ctx.enqueue_create_host_buffer[halo_f](buf_floats)
                )
                self.h_recv_buf.append(
                    ctx.enqueue_create_host_buffer[halo_f](buf_floats)
                )

        ctx.synchronize()

        var total_ring: Int = 0
        for d in range(6):
            total_ring += self.ring_count[d]
        self.has_halo = total_ring > 0

    def submit_pack(
        mut self,
        mut ctx: DeviceContext,
        q: UnsafePointer[Float32, MutAnyOrigin],
    ) raises:
        """Pack owned-boundary q values, stage to pinned host memory,
        sync, and post non-blocking MPI_Isend/Irecv.  Returns
        immediately; MPI runs on the host side while the caller is
        free to launch interior compute on the default stream.

        Must be paired with `complete_exchange()` before any compute
        that reads ghost q values.
        """
        if not self.has_halo:
            # Nothing to pack / send.  At np=1 with a single-patch mesh
            # there are no ghost elements to update, and mpi.waitall
            # over zero requests would read uninitialised req_storage
            # slots -- skip the whole exchange.
            return
        # Mark all 12 request slots as MPI_REQUEST_NULL up front.  Any
        # direction we skip (ring empty or global-BC boundary) will
        # leave its slots null, and MPI_Waitall treats those as no-ops.
        mpi.fill_request_null(self.req_storage, 12)
        # ---- Phase 1: pack on GPU --------------------------------
        # If MPI is CUDA-aware the Isend/Irecv below use device
        # pointers directly -- no D<->H copy needed.  Otherwise we
        # stage through pinned host buffers.
        for d in range(6):
            var count = self.ring_count[d]
            if count == 0 or self.skip_mpi[d]:
                continue
            var total = count * N_P
            ctx.enqueue_function[pack_kernel](
                self.d_send_buf[d].unsafe_ptr(),
                q,
                self.d_pack_idx[d].unsafe_ptr(),
                count,
                self.nc,
                grid_dim=ceildiv(total, HALO_BLOCK),
                block_dim=HALO_BLOCK,
            )
            if not self.cuda_aware:
                ctx.enqueue_copy(self.h_send_buf[d], self.d_send_buf[d])
        ctx.synchronize()  # pack (+ D->H if staged) done before MPI

        # ---- Phase 2: post non-blocking Irecvs + Isends -------------
        # Tag encoding: directions 0..5 for (-x, +x, -y, +y, -z, +z).
        # Send toward d => tag = d.  Recv from d => tag = d XOR 1.
        for d in range(6):
            var count_fl = self.ring_count[d] * N_P * self.nc
            if count_fl == 0 or self.skip_mpi[d]:
                continue
            var neigh = self.neighbour[d]
            var send_tag = d
            var recv_tag = d ^ 1
            var recv_ptr = (
                self.d_recv_buf[d]
                .unsafe_ptr() if self.cuda_aware else self.h_recv_buf[d]
                .unsafe_ptr()
            )
            var send_ptr = (
                self.d_send_buf[d]
                .unsafe_ptr() if self.cuda_aware else self.h_send_buf[d]
                .unsafe_ptr()
            )
            mpi.irecv_float(
                recv_ptr,
                count_fl,
                neigh,
                recv_tag,
                self.req_storage + (6 + d),
            )
            mpi.isend_float(
                send_ptr,
                count_fl,
                neigh,
                send_tag,
                self.req_storage + d,
            )

    def complete_exchange(
        mut self,
        mut ctx: DeviceContext,
        q: UnsafePointer[Float32, MutAnyOrigin],
    ) raises:
        """Wait for MPI to finish, copy received payload back to
        device, and unpack into the ghost ring.  Blocking: returns
        after ghost q values are visible to subsequent kernel
        launches on the default stream.
        """
        if not self.has_halo:
            return
        # Wait for the 12 non-blocking MPI ops posted in submit_pack.
        mpi.waitall(12, self.req_storage)

        # Copy host->device (if staged) and unpack.
        for d in range(6):
            var count = self.ring_count[d]
            if count == 0 or self.skip_mpi[d]:
                continue
            if not self.cuda_aware:
                ctx.enqueue_copy(self.d_recv_buf[d], self.h_recv_buf[d])
            var total = count * N_P
            ctx.enqueue_function[unpack_kernel](
                q,
                self.d_recv_buf[d].unsafe_ptr(),
                self.d_unpack_idx[d].unsafe_ptr(),
                count,
                self.nc,
                grid_dim=ceildiv(total, HALO_BLOCK),
                block_dim=HALO_BLOCK,
            )
        ctx.synchronize()

    def exchange(
        mut self,
        mut ctx: DeviceContext,
        q: UnsafePointer[Float32, MutAnyOrigin],
    ) raises:
        """Blocking exchange.  Thin convenience wrapper over
        `submit_pack` + `complete_exchange` for callers that don't
        want to overlap compute with comm."""
        self.submit_pack(ctx, q)
        self.complete_exchange(ctx, q)


# ----------------------------------------------------------------------
# Small helper: upload a List[Int32] to the device, returns the buffer.
# ----------------------------------------------------------------------


def _upload_i32(
    mut ctx: DeviceContext, src: List[Int32]
) raises -> DeviceBuffer[halo_i]:
    var n = len(src)
    var hbuf = ctx.enqueue_create_host_buffer[halo_i](n)
    memcpy(dest=hbuf.unsafe_ptr(), src=src.unsafe_ptr(), count=n)
    var dbuf = ctx.enqueue_create_buffer[halo_i](n)
    ctx.enqueue_copy(dbuf, hbuf)
    return dbuf^
