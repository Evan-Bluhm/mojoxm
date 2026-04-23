/* ======================================================================
 * mojoxm MPI shim
 * ======================================================================
 *
 * Mojo cannot reference C extern variables directly via `external_call`
 * (it only knows how to call functions), and OpenMPI's MPI_COMM_WORLD,
 * MPI_FLOAT, MPI_INT, etc., are addresses of static library variables.
 * MPI_Comm / MPI_Request / MPI_Datatype are also opaque types whose
 * concrete representation differs between MPI implementations (in
 * OpenMPI they are 8-byte pointers; in MPICH they are int handles).
 *
 * This shim:
 *   - exposes MPI handles as `void*` pointers Mojo can pass through
 *     `UnsafePointer[Int8, MutAnyOrigin]`,
 *   - hides all MPI_* constants behind plain C accessor functions,
 *   - keeps MPI_Status off the Mojo side entirely (we never inspect
 *     statuses; everything uses MPI_STATUS_IGNORE / STATUSES_IGNORE).
 *
 * Build:
 *   mpicc -O2 -fPIC -c src/mpi_shim.c -o build/mpi_shim.o
 *
 * Link from Mojo:
 *   mojo build ... -Xlinker build/mpi_shim.o \
 *                  -Xlinker -L/usr/lib/x86_64-linux-gnu/openmpi/lib \
 *                  -Xlinker -lmpi
 * ====================================================================== */

#include <mpi.h>
#include <stddef.h>

/* mpi-ext.h defines MPIX_Query_cuda_support (runtime) and
 * MPIX_CUDA_AWARE_SUPPORT (compile-time) when the MPI implementation
 * provides them.  Guarded because generic build hosts (e.g. a WSL2
 * apt-get'd OpenMPI) don't ship this header. */
#if __has_include(<mpi-ext.h>)
#  include <mpi-ext.h>
#endif

/* --- Lifecycle ---------------------------------------------------- */

int mxm_mpi_init(void) {
    /* MPI-2 allows NULL/NULL; we don't need to forward Mojo's argv. */
    int provided;
    return MPI_Init_thread(NULL, NULL, MPI_THREAD_FUNNELED, &provided);
}

int mxm_mpi_finalize(void) {
    return MPI_Finalize();
}

int mxm_mpi_initialized(void) {
    int flag = 0;
    MPI_Initialized(&flag);
    return flag;
}

/* --- Communicator helpers (always MPI_COMM_WORLD for now) --------- */

int mxm_mpi_world_rank(void) {
    int r = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &r);
    return r;
}

int mxm_mpi_world_size(void) {
    int s = -1;
    MPI_Comm_size(MPI_COMM_WORLD, &s);
    return s;
}

void mxm_mpi_barrier_world(void) {
    MPI_Barrier(MPI_COMM_WORLD);
}

/* --- Point-to-point on MPI_COMM_WORLD with MPI_FLOAT --------------
 *
 * Returns 0 on success.  `request_out` is filled with the opaque
 * MPI_Request value re-cast to void*.  We allocate the request itself
 * outside (Mojo passes a UnsafePointer[Int64] of count `n`).
 * ----------------------------------------------------------------- */

int mxm_mpi_isend_float(const void* buf, int count, int dest, int tag,
                        void** request_out) {
    MPI_Request req = MPI_REQUEST_NULL;
    int rc = MPI_Isend(buf, count, MPI_FLOAT, dest, tag,
                       MPI_COMM_WORLD, &req);
    *request_out = (void*) req;
    return rc;
}

int mxm_mpi_irecv_float(void* buf, int count, int src, int tag,
                        void** request_out) {
    MPI_Request req = MPI_REQUEST_NULL;
    int rc = MPI_Irecv(buf, count, MPI_FLOAT, src, tag,
                       MPI_COMM_WORLD, &req);
    *request_out = (void*) req;
    return rc;
}

/* `requests` is an array of `n` opaque MPI_Request values stored as
 * void*.  We waitall and ignore statuses. */
int mxm_mpi_waitall(int n, void* requests) {
    return MPI_Waitall(n, (MPI_Request*) requests, MPI_STATUSES_IGNORE);
}

/* Fill `n` slots with MPI_REQUEST_NULL so subsequent MPI_Waitall
 * treats unused slots as no-ops.  Used by the halo-exchange code when
 * some of the 6 face rings are on a global non-periodic boundary and
 * thus have no paired Isend/Irecv to post.  Can't zero-init from Mojo:
 * MPI_REQUEST_NULL is an opaque implementation-defined value
 * (e.g. `&ompi_request_null.request` in OpenMPI, not NULL). */
void mxm_mpi_fill_request_null(void* requests, int n) {
    MPI_Request* r = (MPI_Request*) requests;
    for (int i = 0; i < n; i++) {
        r[i] = MPI_REQUEST_NULL;
    }
}

int mxm_mpi_sendrecv_float(const void* sendbuf, int sendcount, int dest,   int sendtag,
                                 void* recvbuf, int recvcount, int source, int recvtag) {
    return MPI_Sendrecv(sendbuf, sendcount, MPI_FLOAT, dest,   sendtag,
                        recvbuf, recvcount, MPI_FLOAT, source, recvtag,
                        MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

/* --- Collectives -------------------------------------------------- */

int mxm_mpi_allreduce_float_min(const void* sendbuf, void* recvbuf, int count) {
    return MPI_Allreduce(sendbuf, recvbuf, count, MPI_FLOAT, MPI_MIN, MPI_COMM_WORLD);
}

int mxm_mpi_allreduce_float_max(const void* sendbuf, void* recvbuf, int count) {
    return MPI_Allreduce(sendbuf, recvbuf, count, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
}

int mxm_mpi_allreduce_double_sum(const void* sendbuf, void* recvbuf, int count) {
    return MPI_Allreduce(sendbuf, recvbuf, count, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
}

int mxm_mpi_allreduce_float_sum(const void* sendbuf, void* recvbuf, int count) {
    return MPI_Allreduce(sendbuf, recvbuf, count, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
}

int mxm_mpi_allreduce_int_sum(const void* sendbuf, void* recvbuf, int count) {
    return MPI_Allreduce(sendbuf, recvbuf, count, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
}

/* --- Sentinel value getters --------------------------------------- */

int mxm_mpi_proc_null(void) {
    return MPI_PROC_NULL;
}

int mxm_mpi_any_source(void) {
    return MPI_ANY_SOURCE;
}

int mxm_mpi_any_tag(void) {
    return MPI_ANY_TAG;
}

/* --- CUDA-aware MPI capability check ----------------------------- */

int mxm_mpi_is_cuda_aware(void) {
#if defined(MPIX_Query_cuda_support)
    /* OpenMPI >=5 / newer OMPI builds expose a runtime query. */
    return MPIX_Query_cuda_support();
#elif defined(MPIX_CUDA_AWARE_SUPPORT)
    /* OpenMPI 4.x pattern: compile-time macro defined to 1 when the
     * library was built with --with-cuda.  Klone's ompi/4.1.6-2 has
     * this set.  Trust the compile-time header. */
    return MPIX_CUDA_AWARE_SUPPORT;
#else
    /* No detection hook at all -- assume host staging is required. */
    return 0;
#endif
}
