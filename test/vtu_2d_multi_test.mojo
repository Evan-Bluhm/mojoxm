# ======================================================================
# vtu_2d_multi_test -- smoke test for dump_vtu_2d_frame_multi
# ======================================================================
#
# Builds a tiny LocalMesh2D[P], synthesises three trivial scalar fields
# (constants) over its nodes, dumps a multi-field VTU, then re-reads
# the binary blob and verifies:
#   (1) The file exists and has non-zero bytes.
#   (2) The XML header contains three `<DataArray>` entries with the
#       expected field names.
#   (3) The first field's `Scalars=` attribute matches the first
#       field name (default-displayed in ParaView).
#
# Doesn't depend on meshio / Python -- just byte-level checks against
# the expected XML strings.  Catches gross regressions in
# `dump_vtu_2d_frame_multi`'s offset arithmetic + XML emission.
#
# Parameterised over P in {2, 3, 4, 5} so an offset-arithmetic
# regression that only surfaces at NP_p=10/15/21 (vs the P=2 NP_p=6
# case) gets caught at test-quick latency rather than only by ParaView
# opening a higher-P frame.
# ======================================================================

from std.pathlib import Path
from std.sys import has_accelerator
from src.local_mesh_2d import LocalMesh2D
from src.reference_2d import num_tri_nodes_2d
from src.vtu_2d import dump_vtu_2d_frame_multi


def check[P: Int]() raises:
    print("  P=", P)
    var NX = 4
    var NY = 4
    var mesh = LocalMesh2D[P](NX, NY, 1.0, 1.0)
    var NP_p = num_tri_nodes_2d(P)
    var n_total = mesh.num_elements * NP_p
    print(
        "    mesh:",
        mesh.num_elements,
        "elements x",
        NP_p,
        "nodes/elem =",
        n_total,
        "total nodes",
    )

    # Three constant scalar fields.
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

    var path = String("/tmp/vtu_2d_multi_test_p") + String(P) + String(".vtu")
    dump_vtu_2d_frame_multi[P](mesh, names, fields, path)

    var blob = Path(path).read_bytes()
    if len(blob) == 0:
        raise Error("vtu_2d_multi_test P=" + String(P) + ": output file is empty")
    print("    wrote", len(blob), "bytes to", path)

    # Re-decode the XML prologue (everything before the appended raw
    # marker `_`).  Every DataArray header is plain ASCII.
    var s = String(StringSlice[origin_of(blob)](unsafe_from_utf8=blob))
    # ParaView's default-displayed scalar must be field 0.
    if not (String('Scalars="rho"') in s):
        raise Error("vtu_2d_multi_test P=" + String(P) + ': missing Scalars="rho" attribute')
    if not (String('Name="rho"') in s):
        raise Error("vtu_2d_multi_test P=" + String(P) + ": missing rho DataArray header")
    if not (String('Name="p"') in s):
        raise Error("vtu_2d_multi_test P=" + String(P) + ": missing p DataArray header")
    if not (String('Name="|v|"') in s):
        raise Error("vtu_2d_multi_test P=" + String(P) + ": missing |v| DataArray header")


def main() raises:
    print("vtu_2d_multi_test (multi-field VTU smoke test, P=2..5)")
    check[2]()
    check[3]()
    check[4]()
    check[5]()
    print("=== vtu_2d_multi_test PASSED ===")
