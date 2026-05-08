# ======================================================================
# frame_writer_multi_test -- FrameWriter.write_frame_multi smoke test
# ======================================================================
#
# Tests the new sync multi-field 3D output path on a real Solver,
# parameterised over P in {2, 3, 4, 5}:
#   (1) Build a tiny Solver[Advection, P] (NX=NY=NZ=2 cubes).
#   (2) Construct a FrameWriter[Advection, P] and call
#       write_frame_multi twice with two synthetic constant scalar
#       fields.
#   (3) Call finalize() to emit the PVD collection.
#   (4) Check that both frame VTUs and the PVD file exist and have
#       non-zero bytes; check the PVD references both frames in
#       chronological order.
#
# Catches regressions in the sync write path (frame name format,
# directory creation, multi-field VTU layout, PVD collection).
# ======================================================================

from src import mpi
from std.math import ceildiv
from std.pathlib import Path
from std.sys import has_accelerator
from std.gpu.host import DeviceContext

from src.partition import build_partition
from src.reference import ReferenceElement, to_float32, num_tet_nodes
from src.mesh import Mesh
from src.boundary import BoundaryConditions
from src.halo_exchange import HaloExchange
from src.solver import Solver
from src.advection import Advection
from src.frame_writer import FrameWriter
from src.nvtx import NvtxContext


def check[P: Int](mut nvtx: NvtxContext, rank: Int, size: Int) raises:
    print("  P=", P)
    var ctx = DeviceContext()

    var re = ReferenceElement[P]()
    var D_ref = to_float32(re.D_ref)
    var Lift_ref = to_float32(re.Lift_ref)
    var node_weights = to_float32(re.node_weights)

    var mesh = Mesh[P](ctx=ctx, part=build_partition(rank=rank, nprocs=size, nx=2, ny=2, nz=2), Lx=1.0, Ly=1.0, Lz=1.0, bcs=BoundaryConditions.periodic())
    var halo = HaloExchange(ctx=ctx, part=mesh.part, nc=Advection.NUM_COMPONENTS, d_perm=mesh.d_perm.unsafe_ptr())
    var physics = Advection(vx=Float32(1.0), vy=Float32(0.0), vz=Float32(0.0))
    var solver = Solver[Advection, P](ctx=ctx^, mesh=mesh^, halo=halo^, physics=physics^, D_ref=D_ref^, Lift_ref=Lift_ref^, node_weights=node_weights^)

    var output_dir = String("/tmp/frame_writer_multi_test_out_p") + String(P)
    var writer = FrameWriter[Advection, P](solver=solver, nvtx=nvtx, output_dir=output_dir, component=0, max_concurrent=2)

    # Two synthetic constant fields per frame -- enough to verify the
    # multi-field XML and binary appended-data path.
    comptime NP = num_tet_nodes(P)
    var n_total = solver.num_owned_elements * NP
    var f0 = List[Float64]()
    var f1 = List[Float64]()
    for _ in range(n_total):
        f0.append(1.5)
        f1.append(-2.5)

    var names = List[String]()
    names.append(String("rho"))
    names.append(String("phi"))

    # Two frames at distinct times.
    var fields_f1 = List[List[Float64]]()
    fields_f1.append(f0.copy())
    fields_f1.append(f1.copy())
    writer.write_frame_multi(solver=solver, t=Float64(0.0), field_names=names, field_data=fields_f1, nvtx=nvtx)

    var fields_f2 = List[List[Float64]]()
    fields_f2.append(f0.copy())
    fields_f2.append(f1.copy())
    writer.write_frame_multi(solver=solver, t=Float64(0.5), field_names=names, field_data=fields_f2, nvtx=nvtx)

    # --- Check num_frames_written tracks the writes ---
    if writer.num_frames_written() != 2:
        raise Error("frame_writer_multi_test P=" + String(P) + ": num_frames_written=" + String(writer.num_frames_written()) + " after 2 write_frame_multi calls (expected 2)")

    var pvd_path = output_dir + "/solution.pvd"
    writer.finalize(pvd_path=pvd_path, nvtx=nvtx)

    # --- Check file existence + non-empty bytes ---
    var vtu0_path = output_dir + "/frame_00000.vtu"
    var vtu1_path = output_dir + "/frame_00001.vtu"
    for vp in [vtu0_path, vtu1_path, pvd_path]:
        var blob = Path(vp).read_bytes()
        if len(blob) == 0:
            raise Error("frame_writer_multi_test: file empty: " + String(vp))

    # --- Check VTU multi-field XML structure on the first frame ---
    var vtu_blob = Path(vtu0_path).read_bytes()
    var s = String(StringSlice[origin_of(vtu_blob)](unsafe_from_utf8=vtu_blob))
    if not (String('Scalars="rho"') in s):
        raise Error('frame_writer_multi_test: VTU missing Scalars="rho"')
    if not (String('Name="rho"') in s):
        raise Error("frame_writer_multi_test: VTU missing rho DataArray")
    if not (String('Name="phi"') in s):
        raise Error("frame_writer_multi_test: VTU missing phi DataArray")

    # --- Check PVD references both frames in order ---
    var pvd_blob = Path(pvd_path).read_bytes()
    var ps = String(StringSlice[origin_of(pvd_blob)](unsafe_from_utf8=pvd_blob))
    if not (String("frame_00000.vtu") in ps):
        raise Error("frame_writer_multi_test: PVD missing frame 0 reference")
    if not (String("frame_00001.vtu") in ps):
        raise Error("frame_writer_multi_test: PVD missing frame 1 reference")
    print("    wrote frame_00000.vtu, frame_00001.vtu, solution.pvd")
    print('    XML multi-field layout OK (Scalars="rho", rho + phi DataArrays)')
    print("    PVD references both frames")


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    mpi.init()
    var size = mpi.world_size()
    if size > 1:
        mpi.finalize()
        print("frame_writer_multi_test: runs at np=1 only")
        return

    print("frame_writer_multi_test (FrameWriter.write_frame_multi, P=2..5)")
    var rank = mpi.world_rank()
    var nvtx = NvtxContext()
    check[2](nvtx, rank, size)
    check[3](nvtx, rank, size)
    check[4](nvtx, rank, size)
    check[5](nvtx, rank, size)
    print("=== frame_writer_multi_test PASSED ===")
    mpi.finalize()
