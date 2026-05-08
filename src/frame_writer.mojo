# ======================================================================
# FrameWriter -- per-rank VTU frame output (single-field async + multi-field sync)
# ======================================================================
#
# Pulls together the three objects every driver used to build by hand
# (VtuWriter + AsyncWriter + a persistent scalar snapshot buffer),
# plus the per-frame and shutdown logic, into one type that a driver
# instantiates once and hands to the time integrator.
#
# Two write paths share the same per-rank output-dir + PVD machinery:
#
#   * `write_frame(solver, t, nvtx)`        -- single-field async path,
#     one scalar per frame (named "density" in the VTU regardless of
#     what `component` was passed at construction); uses VtuWriter +
#     AsyncWriter for `writev` overlap with compute.
#   * `write_frame_multi(solver, t, names, fields, nvtx)`  -- sync
#     multi-field path, N named scalars per frame.  No async overlap
#     (one blocking `dump_vtu_3d_frame_multi` per call), but lets
#     drivers compute derived fields (rho/p/|v| for Euler, By/|B|/psi
#     for MHD, etc.) and ship them in a single VTU.
#
# Both share `finalize(pvd_path)` for the PVD collection emit.
#
# Usage:
#   var writer = FrameWriter[Euler](solver, nvtx, component=0)
#   writer.write_frame(solver, t=0.0, nvtx)         # single-field async
#   ...
#   writer.write_frame_multi(solver, t, names, fields, nvtx)  # multi-field sync
#   writer.finalize("output/solution.pvd", nvtx)
#
# Multi-rank runs
# ---------------
# FrameWriter reads `solver.mesh.part.rx/ry/rz` and `px/py/pz` to detect
# whether it's running inside an MPI allocation.  At np=1 it writes
# into `output/` with filenames `frame_NNNNN.vtu` and `solution.pvd`
# (unchanged from the pre-MPI-unification layout).  At np>1 each rank
# writes into `output/rank_NNN/` so file names don't collide, and the
# `finalize()` path is still a per-rank `solution.pvd` that references
# only that rank's VTU files.  ParaView can open any rank's pvd to
# inspect just that patch.
#
# NVTX ranges owned by this module (pushed internally; drivers don't
# need to):
#   init_frame_writer, write_frame, vtu_build_segments, vtu_submit,
#   wait_async_writes
# ======================================================================

from src.solver import Solver, Physics
from src.vtu import VtuWriter, write_pvd, dump_vtu_3d_frame_multi
from src.async_writer import AsyncWriter
from src.nvtx import NvtxContext
from src.reference import num_tet_nodes
from std.pathlib import Path


# --- mkdir -p equivalent ---------------------------------------------
# The AsyncWriter's `creat()` call (src/async_writer.mojo) silently
# fails if a path component doesn't exist yet, so before submitting any
# async frame writes we make sure every directory in `path` exists.
# `Path.write_text` (used for the PVD file at finalize) creates parent
# directories on its own, which previously masked the problem:
# finalize() produced a solution.pvd but every frame_NNNNN.vtu
# silently vanished.
#
# We piggyback on the same write_text path here by writing a zero-byte
# sentinel file into the target directory.  That forces Mojo's pathlib
# to mkdir -p the parents, after which the sentinel serves no purpose
# -- we leave it in place because the cost is one 0-byte inode per
# ParaView run and the alternative (a POSIX `mkdir` external_call) has
# been fighting Mojo's type inference (see git history of this file).
def _ensure_dir(dir_path: String) raises:
    if dir_path.byte_length() == 0:
        return
    Path(dir_path + "/.mojoxm_keep").write_text("")


struct FrameWriter[PhysT: Physics, P: Int = 2](Movable):
    var _vtu: VtuWriter
    var _aw: AsyncWriter
    var _snapshot: List[Float32]
    var _paths: List[String]
    var _times: List[Float64]
    var _component: Int
    var _output_dir: String

    def __init__(
        out self,
        mut solver: Solver[Self.PhysT, Self.P],
        mut nvtx: NvtxContext,
        output_dir: String = String("output"),
        component: Int = 0,
        max_concurrent: Int = 8,
    ) raises:
        nvtx.push_range("init_frame_writer")
        # Per-rank VTU: show only this rank's owned elements.  At np=1
        # that is the entire mesh; at np>1 each rank writes its own
        # file with no ghost geometry.  Pass the per-P nodes-per-element
        # count so VtuWriter picks the right VTK cell type (24 for P=2's
        # quadratic tet, 71 for higher-order Lagrange).
        self._vtu = VtuWriter(
            solver.mesh.num_owned_elements,
            solver.mesh.owned_node_xyz_f32_ptr,
            num_tet_nodes(Self.P),
        )
        self._aw = AsyncWriter(max_concurrent=max_concurrent)
        self._snapshot = List[Float32]()
        for _ in range(solver.total_owned_dof):
            self._snapshot.append(Float32(0.0))
        self._paths = List[String]()
        self._times = List[Float64]()
        self._component = component

        # At np>1 direct output to a per-rank subdirectory so the frame
        # filenames don't collide across ranks.  The rank-zero run
        # keeps the legacy flat `output/frame_NNNNN.vtu` layout.
        var part = solver.mesh.part.copy()
        var nprocs = part.px * part.py * part.pz
        if nprocs > 1:
            var rank = ((part.rx * part.py) + part.ry) * part.pz + part.rz
            var sr = String(rank)
            var r_pad = String()
            for _ in range(3 - sr.byte_length()):
                r_pad += "0"
            r_pad += sr
            self._output_dir = output_dir + "/rank_" + r_pad
        else:
            self._output_dir = output_dir
        # Make sure the output directory exists before any frame-write
        # thread issues creat() into it.
        _ensure_dir(self._output_dir)
        nvtx.pop_range()

    def write_frame(
        mut self,
        mut solver: Solver[Self.PhysT, Self.P],
        t: Float64,
        mut nvtx: NvtxContext,
    ) raises:
        """Download one component, enqueue an async VTU write, record
        (path, time) for the eventual PVD collection file.  The frame
        index is derived from the number of frames written so far."""
        nvtx.push_range("write_frame")
        solver.download_owned_component(
            self._component,
            self._snapshot,
            nvtx,
        )
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

    def write_frame_multi(
        mut self,
        mut solver: Solver[Self.PhysT, Self.P],
        t: Float64,
        field_names: List[String],
        field_data: List[List[Float64]],
        mut nvtx: NvtxContext,
    ) raises:
        """Write one multi-field 3D VTU frame *synchronously* using
        pre-computed host-side `field_data` (one List[Float64] of length
        `num_owned_elements * num_tet_nodes(P)` per field, in the order
        named by `field_names`).  Drivers compute derived fields (rho /
        p / |v| / |B| / ...) themselves on host and pass them in.

        This is the multi-field counterpart to `write_frame`.  Unlike
        `write_frame`, this path is synchronous -- it calls the
        in-memory `dump_vtu_3d_frame_multi` directly without going
        through `AsyncWriter`, so the driver pays the VTU pack + write
        cost on the calling thread.  Use `write_frame` for the
        single-field hot path; reach for this when richer ParaView
        output (e.g. rho + p + |v|) outweighs the async overlap.

        Records (path, time) so the eventual PVD collection at
        `finalize()` references this frame correctly."""
        nvtx.push_range("write_frame_multi")
        var frame_id = len(self._paths)
        var fname = String("frame_")
        var sid = String(frame_id)
        for _ in range(5 - sid.byte_length()):
            fname += "0"
        fname += sid
        fname += ".vtu"
        dump_vtu_3d_frame_multi(
            num_elements=solver.num_owned_elements,
            nodes_per_elem=num_tet_nodes(Self.P),
            elem_node_xyz=solver.mesh.owned_node_xyz_f32_ptr,
            field_names=field_names,
            field_data=field_data,
            path=self._output_dir + "/" + fname,
        )
        self._paths.append(fname)
        self._times.append(t)
        nvtx.pop_range()

    def finalize(
        mut self,
        pvd_path: String,
        mut nvtx: NvtxContext,
    ) raises:
        """Block until every outstanding async write has flushed, then
        write the ParaView collection file.  At np>1 each rank writes
        its own per-rank pvd under its rank_NNN subdirectory so the
        user can open any rank's file independently in ParaView."""
        nvtx.push_range("wait_async_writes")
        self._aw.wait_all()
        nvtx.pop_range()

        # The caller passes an np=1-style "output/solution.pvd" path.
        # If we redirected frames into a per-rank subdir above, write
        # the pvd there too (with just the basename preserved) so its
        # relative frame_NNNNN.vtu references still resolve.
        var base_idx = pvd_path.byte_length()
        for i in range(pvd_path.byte_length() - 1, -1, -1):
            if String(pvd_path[byte=i]) == "/":
                base_idx = i + 1
                break
        var basename = String()
        for i in range(base_idx, pvd_path.byte_length()):
            basename += String(pvd_path[byte=i])
        var dst = self._output_dir + "/" + basename
        write_pvd(dst, self._paths, self._times)

    def num_frames_written(self) -> Int:
        return len(self._paths)


# ----------------------------------------------------------------------
# Standalone snapshot helper for end-of-run multi-field dumps.
# ----------------------------------------------------------------------
# Companion to FrameWriter.write_frame_multi but without the per-frame
# PVD bookkeeping.  Drivers use this for one-shot multi-field VTUs at
# end-of-run (rho + p + |v| etc.) that ParaView opens directly without
# a time-series collection -- side-effect file alongside the per-frame
# async pipeline.  Wraps the rebind + nvtx range so each driver's
# snapshot block stays roughly 30 lines (download N components, derive,
# call this) instead of ~50 lines.
#
# Argument convention matches dump_vtu_3d_frame_multi (the underlying
# helper this dispatches to): one List[Float64] of length
# `num_owned_elements * num_tet_nodes(P)` per field in `field_data`,
# names in the same order as fields, first field becomes the
# PointData `Scalars` default.
#
# Example (pulled from examples/euler_vortex.mojo):
#
#   var f_rho  = List[Float64]()
#   var f_p    = List[Float64]()
#   var f_vmag = List[Float64]()
#   # ... download components, compute derived fields, fill the lists ...
#   var fields = List[List[Float64]]()
#   fields.append(f_rho^)
#   fields.append(f_p^)
#   fields.append(f_vmag^)
#   var names = List[String]()
#   names.append(String("rho"))
#   names.append(String("p"))
#   names.append(String("|v|"))
#   write_snapshot_3d_multi(
#       solver=solver, field_names=names, field_data=fields,
#       path=String("output/snapshot_t_final.vtu"), nvtx=nvtx,
#   )
#
# Gate on np=1 in the calling driver (this helper does NOT) -- at
# np>1 each rank would dump only its owned slab to the same path
# and stomp on the other ranks' output.
def write_snapshot_3d_multi[
    PhysT: Physics,
    P: Int = 2,
](
    mut solver: Solver[PhysT, P],
    field_names: List[String],
    field_data: List[List[Float64]],
    path: String,
    mut nvtx: NvtxContext,
) raises:
    nvtx.push_range("snapshot_3d_multi")
    dump_vtu_3d_frame_multi(
        num_elements=solver.num_owned_elements,
        nodes_per_elem=num_tet_nodes(P),
        elem_node_xyz=rebind[UnsafePointer[Float32, MutAnyOrigin]](solver.mesh.owned_node_xyz_f32_ptr),
        field_names=field_names,
        field_data=field_data,
        path=path,
    )
    nvtx.pop_range()
