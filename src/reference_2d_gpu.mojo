# ======================================================================
# GPU-resident 2D reference-element operators
# ======================================================================
#
# Wraps a host `ReferenceElement2D[P]` (Float64) and uploads the two
# static operators the DG stencil needs -- differentiation matrices in
# (r, s) and the edge-lift operators -- as Float32 DeviceBuffers.  These
# are small tables (at P=2, NP=6 and NFP_edge=3 so D_ref is 2*6*6 = 72
# floats and Lift_ref is 3*6*3 = 54 floats) so uploading each as a
# single pinned-staging + enqueue_copy is cheap.
#
# Together with `LocalMesh2DGpu[P]` this gives the GPU everything it
# needs to evaluate the DG rhs for a 2D problem.  A future
# `rk_stage_kernel_2d[NC, EPB, P, PhysT]` launched over element blocks
# will consume both.
# ======================================================================

from src.reference_2d import (
    ReferenceElement2D,
    num_tri_nodes_2d,
    num_edge_nodes,
)
from std.gpu.host import DeviceContext, DeviceBuffer


comptime ref2d_f = DType.float32


def _upload_f64_as_f32(
    mut ctx: DeviceContext, src: List[Float64]
) raises -> DeviceBuffer[ref2d_f]:
    var n = len(src)
    var hbuf = ctx.enqueue_create_host_buffer[ref2d_f](n)
    var hptr = hbuf.unsafe_ptr()
    for k in range(n):
        hptr[k] = Float32(src[k])
    var dbuf = ctx.enqueue_create_buffer[ref2d_f](n)
    ctx.enqueue_copy(dbuf, hbuf)
    return dbuf^


struct ReferenceElement2DGpu[P: Int = 2](Movable):
    """Device mirror of `ReferenceElement2D[P]` static tables.

    Shapes (for the GPU kernel's indexing convenience):
      d_D_ref:        [2 * NP * NP]          (2 directions, row-major)
      d_Lift_ref:     [3 * NP * (P + 1)]     (3 edges, row-major)
      d_node_weights: [NP]                   (cell-mean quadrature
                                              weights, sum = 1)
    """

    comptime NP = num_tri_nodes_2d(Self.P)
    comptime NFP_edge = num_edge_nodes(Self.P)

    var d_D_ref: DeviceBuffer[ref2d_f]
    var d_Lift_ref: DeviceBuffer[ref2d_f]
    var d_node_weights: DeviceBuffer[ref2d_f]

    def __init__(
        out self,
        mut ctx: DeviceContext,
        host: ReferenceElement2D[Self.P],
    ) raises:
        self.d_D_ref = _upload_f64_as_f32(ctx, host.D_ref)
        self.d_Lift_ref = _upload_f64_as_f32(ctx, host.Lift_ref)
        self.d_node_weights = _upload_f64_as_f32(ctx, host.node_weights)
        ctx.synchronize()

    def device_bytes(self) -> Int:
        """Total device footprint of this reference element's
        operator tables.  Used by `MemoryReport` 2D-side construction
        in drivers."""
        comptime SZ = 4  # Float32 == 4 bytes
        comptime NP = Self.NP
        comptime NFP = Self.NFP_edge
        return SZ * (
            2 * NP * NP + 3 * NP * NFP + NP
        )  # D_ref (r + s directions)  # Lift_ref (3 edges)  # node_weights
