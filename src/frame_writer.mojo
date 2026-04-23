# ======================================================================
# FrameWriter -- single-field VTU frame output for single-rank drivers
# ======================================================================
#
# Pulls together the three objects every non-MPI driver used to build
# by hand (VtuWriter + AsyncWriter + a persistent scalar snapshot
# buffer), plus the per-frame and shutdown logic, into one type that
# a driver instantiates once and hands to the time integrator.
#
# Usage:
#   var writer = FrameWriter[Euler](solver, nvtx, component=0)
#   writer.write_frame(solver, t=0.0, nvtx)
#   ...
#   writer.finalize("output/solution.pvd", nvtx)
#
# NVTX ranges owned by this module (pushed internally; drivers don't
# need to):
#   init_frame_writer, write_frame, vtu_build_segments, vtu_submit,
#   wait_async_writes
# ======================================================================

from src.solver import Solver, Physics
from src.vtu import VtuWriter, write_pvd
from src.async_writer import AsyncWriter
from src.nvtx import NvtxContext


struct FrameWriter[PhysT: Physics](Movable):
    var _vtu: VtuWriter
    var _aw: AsyncWriter
    var _snapshot: List[Float32]
    var _paths: List[String]
    var _times: List[Float64]
    var _component: Int
    var _output_dir: String

    def __init__(
        out self,
        mut solver: Solver[Self.PhysT],
        mut nvtx: NvtxContext,
        output_dir: String = String("output"),
        component: Int = 0,
        max_concurrent: Int = 8,
    ) raises:
        nvtx.push_range("init_frame_writer")
        self._vtu = VtuWriter(
            solver.mesh.num_elements, solver.mesh.elem_node_xyz_f32_ptr
        )
        self._aw = AsyncWriter(max_concurrent=max_concurrent)
        self._snapshot = List[Float32]()
        for _ in range(solver.total_dof):
            self._snapshot.append(Float32(0.0))
        self._paths = List[String]()
        self._times = List[Float64]()
        self._component = component
        self._output_dir = output_dir
        nvtx.pop_range()

    def write_frame(
        mut self,
        mut solver: Solver[Self.PhysT],
        t: Float64,
        mut nvtx: NvtxContext,
    ) raises:
        """Download one component, enqueue an async VTU write, record
        (path, time) for the eventual PVD collection file.  The frame
        index is derived from the number of frames written so far."""
        nvtx.push_range("write_frame")
        solver.download_component(self._component, self._snapshot, nvtx)
        var frame_id = len(self._paths)
        var fname = String("frame_")
        var sid = String(frame_id)
        for _ in range(5 - sid.byte_length()):
            fname += "0"
        fname += sid
        fname += ".vtu"
        nvtx.push_range("vtu_build_segments")
        var segs = self._vtu.build_segments(self._snapshot)
        nvtx.pop_range()
        nvtx.push_range("vtu_submit")
        self._aw.submit(self._output_dir + "/" + fname, segs)
        nvtx.pop_range()
        self._paths.append(fname)
        self._times.append(t)
        nvtx.pop_range()

    def finalize(
        mut self,
        pvd_path: String,
        mut nvtx: NvtxContext,
    ) raises:
        """Block until every outstanding async write has flushed, then
        write the ParaView collection file.  Returns nothing; callers
        that want the wait time can bracket this with perf_counter_ns
        themselves or use TimeLoopResult from the time integrator."""
        nvtx.push_range("wait_async_writes")
        self._aw.wait_all()
        nvtx.pop_range()
        write_pvd(pvd_path, self._paths, self._times)

    def num_frames_written(self) -> Int:
        return len(self._paths)
