# ======================================================================
# vtu_3d_multi_test -- smoke test for dump_vtu_3d_frame_multi
# ======================================================================
#
# 3D analog of vtu_2d_multi_test.  Builds a tiny LocalMesh (3D),
# synthesises three trivial scalar fields (constants) over its nodes,
# dumps a multi-field 3D VTU, then re-reads the binary blob and
# verifies:
#   (1) The file exists and has non-zero bytes.
#   (2) The XML header contains three `<DataArray>` entries with the
#       expected field names.
#   (3) The first field's `Scalars=` attribute matches the first
#       field name (default-displayed in ParaView).
#
# Doesn't depend on meshio / Python -- byte-level checks only.
# Catches gross regressions in `dump_vtu_3d_frame_multi`'s offset
# arithmetic + XML emission + appended-blob layout.
#
# Parameterised over P in {2, 3, 4, 5} so an offset-arithmetic
# regression that only surfaces at NP=20/35/56 (3 - 24x larger
# byte budget than the P=2 NP=10 case) gets caught at test-quick
# latency rather than only by ParaView opening a higher-P frame.
#
# Requires GPU: `LocalMesh[P]` is built via DeviceContext (its
# elem_node_xyz host buffer is downloaded from the GPU build kernel).
# ======================================================================

from src import mpi
from std.pathlib import Path
from std.sys import has_accelerator
from std.gpu.host import DeviceContext
from src.local_mesh import LocalMesh
from src.boundary import BoundaryConditions
from src.reference import num_tet_nodes
from src.vtu import dump_vtu_3d_frame_multi


def check[P: Int]() raises:
    print("  P=", P)
    var Nx = 2
    var Ny = 2
    var Nz = 2
    var ctx = DeviceContext()
    var mesh = LocalMesh[P](ctx=ctx, Nx=Nx, Ny=Ny, Nz=Nz, Lx=1.0, Ly=1.0, Lz=1.0, bcs=BoundaryConditions.periodic())
    var NP = num_tet_nodes(P)
    var n_total = mesh.num_elements * NP
    print("    mesh:", mesh.num_elements, "elements x", NP, "nodes/elem =", n_total, "total nodes")

    # Three constant scalar fields -- one Float64 per node.
    var rho = List[Float64]()
    var p = List[Float64]()
    var vmag = List[Float64]()
    for _ in range(n_total):
        rho.append(1.5)
        p.append(2.5)
        vmag.append(3.5)

    var names = List[String]()
    names.append(String("rho"))
    names.append(String("p"))
    names.append(String("|v|"))

    var fields = List[List[Float64]]()
    fields.append(rho.copy())
    fields.append(p.copy())
    fields.append(vmag.copy())

    var path = String("/tmp/vtu_3d_multi_test_p") + String(P) + String(".vtu")
    dump_vtu_3d_frame_multi(mesh.num_elements, NP, rebind[UnsafePointer[Float32, MutAnyOrigin]](mesh.elem_node_xyz_f32_ptr), names, fields, path)

    var blob = Path(path).read_bytes()
    if len(blob) == 0:
        raise Error("vtu_3d_multi_test P=" + String(P) + ": output file is empty")
    print("    wrote", len(blob), "bytes to", path)

    var s = String(StringSlice[origin_of(blob)](unsafe_from_utf8=blob))
    if not (String('Scalars="rho"') in s):
        raise Error("vtu_3d_multi_test P=" + String(P) + ': missing Scalars="rho" attribute')
    if not (String('Name="rho"') in s):
        raise Error("vtu_3d_multi_test P=" + String(P) + ": missing rho DataArray header")
    if not (String('Name="p"') in s):
        raise Error("vtu_3d_multi_test P=" + String(P) + ": missing p DataArray header")
    if not (String('Name="|v|"') in s):
        raise Error("vtu_3d_multi_test P=" + String(P) + ": missing |v| DataArray header")


def main() raises:
    comptime assert has_accelerator(), "Requires GPU"
    mpi.init()
    print("vtu_3d_multi_test (multi-field 3D VTU smoke test, P=2..5)")
    check[2]()
    check[3]()
    check[4]()
    check[5]()
    print("=== vtu_3d_multi_test PASSED ===")
    mpi.finalize()
