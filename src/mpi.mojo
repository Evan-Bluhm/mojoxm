# ======================================================================
# mojoxm MPI bindings (FFI to libmpi via the C shim)
# ======================================================================
#
# All MPI handles (communicators, requests, datatypes) are opaque
# pointers in OpenMPI 4.x.  Mojo cannot reference C extern variables
# directly, so the constants (MPI_COMM_WORLD, MPI_FLOAT, ...) and the
# few struct-flavored APIs we need are wrapped in a tiny C shim
# (`src/mpi_shim.c`) that exposes plain int/void* functions.  This file
# is a thin Mojo veneer over those wrappers.
#
# To build the shim:
#   mpicc -O2 -fPIC -c src/mpi_shim.c -o build/mpi_shim.o
#
# To link the shim into a Mojo binary:
#   mojo build ... -Xlinker build/mpi_shim.o \
#                  -Xlinker -L/usr/lib/x86_64-linux-gnu/openmpi/lib \
#                  -Xlinker -lmpi
#
# Single-rank programs that don't call init() will still work; this
# module never auto-initialises MPI.  Drivers that opt into MPI must
# call `init()` early in main() and `finalize()` before exit.
# ======================================================================

from std.ffi import external_call, c_int


# ----------------------------------------------------------------------
# Lifecycle
# ----------------------------------------------------------------------


def init() raises:
    """Initialize MPI with funneled threading.  Safe to call once."""
    var rc = Int(external_call["mxm_mpi_init", c_int]())
    if rc != 0:
        raise Error("MPI_Init_thread failed, rc=" + String(rc))


def finalize() raises:
    var rc = Int(external_call["mxm_mpi_finalize", c_int]())
    if rc != 0:
        raise Error("MPI_Finalize failed, rc=" + String(rc))


def initialized() -> Bool:
    return Int(external_call["mxm_mpi_initialized", c_int]()) != 0


def is_cuda_aware() -> Bool:
    """True iff the linked MPI library was built with CUDA support
    (i.e. accepts device pointers in Isend/Irecv).  Detected at shim
    compile time via MPIX_CUDA_AWARE_SUPPORT / MPIX_Query_cuda_support."""
    return Int(external_call["mxm_mpi_is_cuda_aware", c_int]()) != 0


# ----------------------------------------------------------------------
# MPI_COMM_WORLD identity
# ----------------------------------------------------------------------


def world_rank() -> Int:
    return Int(external_call["mxm_mpi_world_rank", c_int]())


def world_size() -> Int:
    return Int(external_call["mxm_mpi_world_size", c_int]())


def barrier_world():
    external_call["mxm_mpi_barrier_world", NoneType]()


# ----------------------------------------------------------------------
# Sentinels
# ----------------------------------------------------------------------


def proc_null() -> Int:
    return Int(external_call["mxm_mpi_proc_null", c_int]())


def any_source() -> Int:
    return Int(external_call["mxm_mpi_any_source", c_int]())


def any_tag() -> Int:
    return Int(external_call["mxm_mpi_any_tag", c_int]())


# ----------------------------------------------------------------------
# Point-to-point on MPI_COMM_WORLD with MPI_FLOAT
#
# MPI_Request is opaque; we store it as Int64 inside the caller's
# array.  `requests` is an UnsafePointer[Int64] of capacity n.
# ----------------------------------------------------------------------


def isend_float(
    buf: UnsafePointer[Float32, MutAnyOrigin],
    count: Int,
    dest: Int,
    tag: Int,
    request_out: UnsafePointer[Int64, MutAnyOrigin],
) raises:
    var rc = Int(external_call["mxm_mpi_isend_float", c_int](buf, c_int(count), c_int(dest), c_int(tag), request_out))
    if rc != 0:
        raise Error("MPI_Isend failed, rc=" + String(rc))


def irecv_float(
    buf: UnsafePointer[Float32, MutAnyOrigin],
    count: Int,
    src: Int,
    tag: Int,
    request_out: UnsafePointer[Int64, MutAnyOrigin],
) raises:
    var rc = Int(external_call["mxm_mpi_irecv_float", c_int](buf, c_int(count), c_int(src), c_int(tag), request_out))
    if rc != 0:
        raise Error("MPI_Irecv failed, rc=" + String(rc))


def waitall(n: Int, requests: UnsafePointer[Int64, MutAnyOrigin]) raises:
    var rc = Int(external_call["mxm_mpi_waitall", c_int](c_int(n), requests))
    if rc != 0:
        raise Error("MPI_Waitall failed, rc=" + String(rc))


def fill_request_null(
    requests: UnsafePointer[Int64, MutAnyOrigin],
    n: Int,
):
    """Set `n` request slots to MPI_REQUEST_NULL so a subsequent
    `waitall` treats them as no-ops.  Can't zero-init from Mojo:
    MPI_REQUEST_NULL is implementation-defined (a sentinel pointer,
    not 0) so we route through the C shim."""
    _ = external_call["mxm_mpi_fill_request_null", NoneType](
        requests,
        c_int(n),
    )


def sendrecv_float(
    sendbuf: UnsafePointer[Float32, MutAnyOrigin],
    sendcount: Int,
    dest: Int,
    sendtag: Int,
    recvbuf: UnsafePointer[Float32, MutAnyOrigin],
    recvcount: Int,
    src: Int,
    recvtag: Int,
) raises:
    var rc = Int(
        external_call["mxm_mpi_sendrecv_float", c_int](
            sendbuf,
            c_int(sendcount),
            c_int(dest),
            c_int(sendtag),
            recvbuf,
            c_int(recvcount),
            c_int(src),
            c_int(recvtag),
        )
    )
    if rc != 0:
        raise Error("MPI_Sendrecv failed, rc=" + String(rc))


# ----------------------------------------------------------------------
# Collectives
# ----------------------------------------------------------------------


def allreduce_float_min(
    sendbuf: UnsafePointer[Float32, MutAnyOrigin],
    recvbuf: UnsafePointer[Float32, MutAnyOrigin],
    count: Int,
) raises:
    var rc = Int(external_call["mxm_mpi_allreduce_float_min", c_int](sendbuf, recvbuf, c_int(count)))
    if rc != 0:
        raise Error("MPI_Allreduce(MIN) failed, rc=" + String(rc))


def allreduce_float_max(
    sendbuf: UnsafePointer[Float32, MutAnyOrigin],
    recvbuf: UnsafePointer[Float32, MutAnyOrigin],
    count: Int,
) raises:
    var rc = Int(external_call["mxm_mpi_allreduce_float_max", c_int](sendbuf, recvbuf, c_int(count)))
    if rc != 0:
        raise Error("MPI_Allreduce(MAX) failed, rc=" + String(rc))


def allreduce_float_sum(
    sendbuf: UnsafePointer[Float32, MutAnyOrigin],
    recvbuf: UnsafePointer[Float32, MutAnyOrigin],
    count: Int,
) raises:
    var rc = Int(external_call["mxm_mpi_allreduce_float_sum", c_int](sendbuf, recvbuf, c_int(count)))
    if rc != 0:
        raise Error("MPI_Allreduce(SUM) failed, rc=" + String(rc))


def allreduce_double_sum(
    sendbuf: UnsafePointer[Float64, MutAnyOrigin],
    recvbuf: UnsafePointer[Float64, MutAnyOrigin],
    count: Int,
) raises:
    """Float64 (double-precision) sum allreduce.  Use this for
    domain-integrated diagnostics where the Float32 SUM would lose
    precision from summing O(1M) nodal values."""
    var rc = Int(external_call["mxm_mpi_allreduce_double_sum", c_int](sendbuf, recvbuf, c_int(count)))
    if rc != 0:
        raise Error("MPI_Allreduce(DOUBLE SUM) failed, rc=" + String(rc))


def allreduce_int_sum(
    sendbuf: UnsafePointer[Int32, MutAnyOrigin],
    recvbuf: UnsafePointer[Int32, MutAnyOrigin],
    count: Int,
) raises:
    var rc = Int(external_call["mxm_mpi_allreduce_int_sum", c_int](sendbuf, recvbuf, c_int(count)))
    if rc != 0:
        raise Error("MPI_Allreduce(int SUM) failed, rc=" + String(rc))
