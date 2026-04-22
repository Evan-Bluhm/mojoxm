# ======================================================================
# AsyncWriter -- pthread-based fire-and-forget file writer
# ======================================================================
#
# A lightweight helper that issues a POSIX write() on a background
# thread so the main simulation loop can continue computing and
# enqueueing GPU work while disk I/O is in flight.
#
# Design
# ------
#   * Each submit() spawns a new pthread that writes the caller's
#     buffer to disk, closes the file, and frees the buffer.  The
#     pthread owns its buffer for the duration of the write.
#   * A small list of outstanding pthread_t values is kept in the
#     `AsyncWriter` struct.  When `max_concurrent` in-flight writes
#     is reached, submit() joins the oldest before spawning a new
#     one.  This caps peak memory without stalling normal operation.
#   * wait_all() joins everything; the destructor calls wait_all().
#   * Writes go via `open/write/close` syscalls directly (no Path
#     wrapper, no fsync, no stdio buffering) -- the OS page cache
#     absorbs the data immediately and flushes it in the background.
#
# The caller is responsible for allocating the buffer with `alloc[UInt8]`
# and transferring ownership via submit().  The writer thread always
# frees the buffer.
# ======================================================================

from std.ffi import external_call, c_int, c_size_t, c_ssize_t
from std.memory import alloc, memcpy

# Linux fcntl.h constants: O_WRONLY=1, O_CREAT=0o100, O_TRUNC=0o1000.
comptime _OPEN_FLAGS = c_int(0o1101)
comptime _OPEN_MODE  = c_int(0o644)


@fieldwise_init
struct WriteSegment(ImplicitlyCopyable, Movable):
    """One segment in a scatter-gather write.  `ptr` is NOT freed by the
    writer when `owned` is False (e.g. a long-lived buffer owned by the
    caller).  When True, the writer frees the segment after the
    write completes."""
    var ptr:   UnsafePointer[UInt8, MutAnyOrigin]
    var nbytes: Int
    var owned: Bool


struct _WriteJob(ImplicitlyCopyable, Movable):
    # Heap-allocated array of WriteSegment values.  Owned (freed after
    # the writev() syscall).
    var segs:   UnsafePointer[WriteSegment, MutAnyOrigin]
    var nsegs:  Int
    var path_c: UnsafePointer[UInt8, MutAnyOrigin]   # owned, null-terminated

# Layout of struct iovec on Linux x86-64:
#   void*  iov_base  (8 bytes)
#   size_t iov_len   (8 bytes)
# We pack an array of these and hand the pointer to writev.
@fieldwise_init
struct _Iovec(ImplicitlyCopyable, Movable):
    var base: UnsafePointer[UInt8, MutAnyOrigin]
    var length: Int


# -- Worker entry (module scope so its address can be passed to pthread_create).
#
# Signature matches `void *(*)(void *)`: argument and return are both
# opaque pointers in the MutAnyOrigin universe.
def _writer_entry(
    arg: UnsafePointer[Int8, MutAnyOrigin]
) -> UnsafePointer[Int8, MutAnyOrigin]:
    var job_ptr = arg.bitcast[_WriteJob]()
    var segs   = job_ptr[].segs
    var nsegs  = job_ptr[].nsegs
    var path_c = job_ptr[].path_c

    var fd = Int(external_call["open", c_int](path_c, _OPEN_FLAGS, _OPEN_MODE))
    if fd >= 0:
        for s in range(nsegs):
            var remaining = segs[s].nbytes
            var p = segs[s].ptr
            while remaining > 0:
                var n = Int(external_call["write", c_ssize_t](
                    fd, p, c_size_t(remaining)
                ))
                if n <= 0:
                    break
                remaining -= n
                p = p + n
        _ = external_call["close", c_int](c_int(fd))
    # Free owned segments + metadata.
    for i in range(nsegs):
        if segs[i].owned:
            segs[i].ptr.free()
    segs.free()
    path_c.free()
    job_ptr.free()
    return UnsafePointer[Int8, MutAnyOrigin]()


struct AsyncWriter(Movable):
    var _thread_ids: List[UInt64]
    var _max_concurrent: Int

    def __init__(out self, max_concurrent: Int = 4):
        self._thread_ids = List[UInt64]()
        self._max_concurrent = max_concurrent

    def submit(
        mut self,
        path: String,
        segments: List[WriteSegment],
    ) raises:
        """Submit a scatter-gather write.  Segments with owned=True are
        freed by the writer thread after writev() returns."""
        # Cap concurrency: join the oldest if we're at the limit.
        while len(self._thread_ids) >= self._max_concurrent:
            var oldest = self._thread_ids[0]
            _ = self._thread_ids.pop(0)
            var retval = UnsafePointer[Int8, MutAnyOrigin]()
            _ = external_call["pthread_join", Int32](oldest, retval)

        # Copy path into a heap-allocated, null-terminated C string.
        var pn = len(path)
        var path_c = alloc[UInt8](pn + 1)
        memcpy(dest=path_c, src=path.unsafe_ptr(), count=pn)
        path_c[pn] = 0

        # Copy segments into a heap-allocated array (takes ownership).
        var nsegs = len(segments)
        var segs_ptr = alloc[WriteSegment](nsegs)
        for i in range(nsegs):
            segs_ptr[i] = segments[i]

        # Allocate the job descriptor on the heap and fill it.
        var job = alloc[_WriteJob](1)
        job[].segs   = segs_ptr
        job[].nsegs  = nsegs
        job[].path_c = path_c

        # Spawn the writer thread.
        var tid_storage = alloc[UInt64](1)
        var attr_null = UnsafePointer[Int8, MutAnyOrigin]()
        var rc = external_call["pthread_create", Int32](
            tid_storage,
            attr_null,
            _writer_entry,
            job.bitcast[Int8](),
        )
        var tid = tid_storage[0]
        tid_storage.free()

        if rc != 0:
            # Thread creation failed; do it synchronously on this thread.
            _ = _writer_entry(job.bitcast[Int8]())
            return
        self._thread_ids.append(tid)

    def wait_all(mut self):
        for i in range(len(self._thread_ids)):
            var retval = UnsafePointer[Int8, MutAnyOrigin]()
            _ = external_call["pthread_join", Int32](
                self._thread_ids[i], retval
            )
        self._thread_ids.clear()

    def __del__(deinit self):
        self.wait_all()
