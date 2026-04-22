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
from std.gpu import global_idx
from std.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from std.math import ceildiv
from std.memory import memcpy

# Kuhn tets per Cartesian cube (keep in sync with src/mesh.mojo).
comptime KUHN_TETS = 6
comptime halo_f = DType.float32
comptime halo_i = DType.int32
comptime HALO_BLOCK = 256

# MPI tag encoding: (axis << 1) | sign (0 for minus, 1 for plus), from
# the sender's viewpoint.  A recv from the d-neighbour expects the
# opposite-direction tag (XOR 1 on the low bit).
comptime TAG_MINUS_X = 0
comptime TAG_PLUS_X  = 1
comptime TAG_MINUS_Y = 2
comptime TAG_PLUS_Y  = 3
comptime TAG_MINUS_Z = 4
comptime TAG_PLUS_Z  = 5


# Direction ordering for all 6-entry arrays below:
#   [0] -x   [1] +x   [2] -y   [3] +y   [4] -z   [5] +z


# ----------------------------------------------------------------------
# Pack/unpack kernels.  Both operate over `count * N_P` threads: one
# thread per (element-in-list, node-in-element) pair, and inner-loop
# over the NC components.
# ----------------------------------------------------------------------

def pack_kernel(
    send_buf: UnsafePointer[Float32, MutAnyOrigin],
    q:        UnsafePointer[Float32, MutAnyOrigin],
    indices:  UnsafePointer[Int32,   MutAnyOrigin],
    count: Int, NC: Int,
):
    var idx = Int(global_idx.x)
    var total = count * N_P
    if idx >= total:
        return
    var i = idx // N_P
    var nn = idx % N_P
    var elem = Int(indices[i])
    for c in range(NC):
        send_buf[(i * N_P + nn) * NC + c] = (
            q[(elem * N_P + nn) * NC + c]
        )

def unpack_kernel(
    q:        UnsafePointer[Float32, MutAnyOrigin],
    recv_buf: UnsafePointer[Float32, MutAnyOrigin],
    indices:  UnsafePointer[Int32,   MutAnyOrigin],
    count: Int, NC: Int,
):
    var idx = Int(global_idx.x)
    var total = count * N_P
    if idx >= total:
        return
    var i = idx // N_P
    var nn = idx % N_P
    var elem = Int(indices[i])
    for c in range(NC):
        q[(elem * N_P + nn) * NC + c] = (
            recv_buf[(i * N_P + nn) * NC + c]
        )


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
    nx: Int, ny: Int, nz: Int,
    loc_nx: Int, loc_ny: Int,
    axis: Int, sign: Int,   # axis in [0,1,2]; sign in {-1, +1}
    is_ghost: Bool,          # False = owned boundary, True = ghost ring
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

    # 6 neighbour ranks in the order [-x, +x, -y, +y, -z, +z].
    var neighbour: List[Int]
    # Element count in each of the 6 face rings (owned == ghost).
    var ring_count: List[Int]

    # Per-direction device buffers.  Index [d] in each list refers to
    # the same (axis, sign) direction encoded above.
    var d_pack_idx:   List[DeviceBuffer[halo_i]]
    var d_unpack_idx: List[DeviceBuffer[halo_i]]
    var d_send_buf:   List[DeviceBuffer[halo_f]]
    var d_recv_buf:   List[DeviceBuffer[halo_f]]
    # Pinned host staging buffers (used when MPI isn't CUDA-aware).
    # Indexed the same way as d_send_buf / d_recv_buf.
    var h_send_buf:   List[HostBuffer[halo_f]]
    var h_recv_buf:   List[HostBuffer[halo_f]]

    # 12 MPI_Request handles stored contiguously as Int64 (OpenMPI
    # MPI_Request is an opaque 8-byte pointer).  Layout:
    #   [0..6):  send requests
    #   [6..12): recv requests
    var req_storage:  UnsafePointer[Int64, MutExternalOrigin]

    def __init__(
        out self,
        mut ctx: DeviceContext,
        part: Partition,
        nc: Int,
    ) raises:
        self.nc = nc
        var nx = part.nx
        var ny = part.ny
        var nz = part.nz
        var loc_nx = nx + 2
        var loc_ny = ny + 2

        self.neighbour = List[Int]()
        self.neighbour.append(part.neighbour_minus_x)
        self.neighbour.append(part.neighbour_plus_x)
        self.neighbour.append(part.neighbour_minus_y)
        self.neighbour.append(part.neighbour_plus_y)
        self.neighbour.append(part.neighbour_minus_z)
        self.neighbour.append(part.neighbour_plus_z)

        var axes = [0, 0, 1, 1, 2, 2]
        var signs = [-1, 1, -1, 1, -1, 1]

        self.ring_count = List[Int]()
        self.d_pack_idx   = List[DeviceBuffer[halo_i]]()
        self.d_unpack_idx = List[DeviceBuffer[halo_i]]()
        self.d_send_buf   = List[DeviceBuffer[halo_f]]()
        self.d_recv_buf   = List[DeviceBuffer[halo_f]]()
        self.h_send_buf   = List[HostBuffer[halo_f]]()
        self.h_recv_buf   = List[HostBuffer[halo_f]]()

        for d in range(6):
            var owned_list = List[Int32]()
            var ghost_list = List[Int32]()
            var n_owned = _pack_list_for_dir(
                owned_list, nx, ny, nz, loc_nx, loc_ny,
                axes[d], signs[d], is_ghost=False,
            )
            var n_ghost = _pack_list_for_dir(
                ghost_list, nx, ny, nz, loc_nx, loc_ny,
                axes[d], signs[d], is_ghost=True,
            )
            if n_owned != n_ghost:
                raise Error(
                    "HaloExchange: owned/ghost ring size mismatch"
                )
            self.ring_count.append(n_owned)
            self.d_pack_idx.append(_upload_i32(ctx, owned_list))
            self.d_unpack_idx.append(_upload_i32(ctx, ghost_list))
            var buf_floats = n_owned * N_P * nc
            self.d_send_buf.append(
                ctx.enqueue_create_buffer[halo_f](buf_floats)
            )
            self.d_recv_buf.append(
                ctx.enqueue_create_buffer[halo_f](buf_floats)
            )
            self.h_send_buf.append(
                ctx.enqueue_create_host_buffer[halo_f](buf_floats)
            )
            self.h_recv_buf.append(
                ctx.enqueue_create_host_buffer[halo_f](buf_floats)
            )

        ctx.synchronize()
        self.req_storage = alloc[Int64](12)

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
        # ---- Phase 1: pack on GPU, then copy device -> host ---------
        for d in range(6):
            var count = self.ring_count[d]
            if count == 0:
                continue
            var total = count * N_P
            ctx.enqueue_function[pack_kernel, pack_kernel](
                self.d_send_buf[d].unsafe_ptr(),
                q, self.d_pack_idx[d].unsafe_ptr(),
                count, self.nc,
                grid_dim=ceildiv(total, HALO_BLOCK),
                block_dim=HALO_BLOCK,
            )
            ctx.enqueue_copy(self.h_send_buf[d], self.d_send_buf[d])
        ctx.synchronize()   # pack + D->H done before MPI reads bufs

        # ---- Phase 2: post non-blocking Irecvs + Isends -------------
        # Tag encoding: directions 0..5 for (-x, +x, -y, +y, -z, +z).
        # Send toward d => tag = d.  Recv from d => tag = d XOR 1.
        for d in range(6):
            var count_fl = self.ring_count[d] * N_P * self.nc
            if count_fl == 0:
                continue
            var neigh = self.neighbour[d]
            var send_tag = d
            var recv_tag = d ^ 1
            mpi.irecv_float(
                self.h_recv_buf[d].unsafe_ptr(),
                count_fl, neigh, recv_tag,
                self.req_storage + (6 + d),
            )
            mpi.isend_float(
                self.h_send_buf[d].unsafe_ptr(),
                count_fl, neigh, send_tag,
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
        # Wait for the 12 non-blocking MPI ops posted in submit_pack.
        mpi.waitall(12, self.req_storage)

        # Copy host recv buffers back to device and unpack.
        for d in range(6):
            var count = self.ring_count[d]
            if count == 0:
                continue
            ctx.enqueue_copy(self.d_recv_buf[d], self.h_recv_buf[d])
            var total = count * N_P
            ctx.enqueue_function[unpack_kernel, unpack_kernel](
                q, self.d_recv_buf[d].unsafe_ptr(),
                self.d_unpack_idx[d].unsafe_ptr(),
                count, self.nc,
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
