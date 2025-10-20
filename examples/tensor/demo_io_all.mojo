# MIT License
# SPDX-License-Identifier: MIT
# Project:      momijo.tensor
# Module:       tests.tensor
# File:         examples/Tensor/demo_io_all.mojo
#
# Description:
#   End-to-end demos for CSV / JSON / XML I/O across Float64 / Float32 / Int.
 

from collections.list import List
from momijo.tensor import tensor

# -----------------------------------------------------------------------------
# Helpers: small sample tensors for each dtype/shape
# -----------------------------------------------------------------------------
fn _mk_tensor_f64_3x3() -> tensor.Tensor[Float64]:
    var data = tensor.Tensor([
        [1.1, 2.5, 0.3],
        [4.7, 5.6, 6.1],
        [10.0, 1.1, 2.5]
    ])
    return data.copy()

fn _mk_tensor_f32_2x4() -> tensor.Tensor[Float32]:
    # Build in Float64 then cast to Float32 for a 3x3 example; slice to 2x4 if needed in your API.
    var data = tensor.to_float32(
        tensor.Tensor([
            [1.1, 2.5, 0.3],
            [4.7, 5.6, 6.1],
            [10.0, 1.1, 2.5]
        ])
    )
    return data.copy()

fn _mk_tensor_int_1d() -> tensor.Tensor[Int]:
    var data = tensor.Tensor([1, 2, 3, 4, 5, 6, 10])
    return data.copy()

# -----------------------------------------------------------------------------
# CSV demos
# -----------------------------------------------------------------------------
fn demo_csv_f64() -> None:
    var tbl = _mk_tensor_f64_3x3()

    # Write without header (default delimiter ",")
    var ok = tensor.write_csv_f64(tbl, "demo_f64.csv")
    if not ok:
        print("write_csv_f64 failed")
        return

    # Write with header (optional): provide column names
    var header = List[String]()
    header.append("c1"); header.append("c2"); header.append("c3")
    var ok2 = tensor.write_csv_f64(tbl, "demo_f64_with_header.csv", ",", header.copy())
    if not ok2:
        print("write_csv_f64 with header failed")

    # Read plain CSV
    var t_plain = tensor.read_csv_f64("demo_f64.csv")
    print("CSV f64 plain shape: " + t_plain.shape().__str__())

    # Read CSV with header: skip the first line via CsvOptions(header=True)
    var t_with_hdr = tensor.read_csv_f64("demo_f64_with_header.csv", tensor.CsvOptions(",", True))
    print("CSV f64 header shape: " + t_with_hdr.shape().__str__())

fn demo_csv_f32() -> None:
    var tbl = _mk_tensor_f32_2x4()
    var ok = tensor.write_csv_f32(tbl, "demo_f32.csv")
    if not ok:
        print("write_csv_f32 failed")
        return
    var t2 = tensor.read_csv_f32("demo_f32.csv")
    print("CSV f32 shape: " + t2.shape().__str__())

fn demo_csv_int() -> None:
    var tbl = _mk_tensor_int_1d()
    var ok = tensor.write_csv_int(tbl, "demo_int.csv")
    if not ok:
        print("write_csv_int failed")
        return
    var t2 = tensor.read_csv_int("demo_int.csv")
    print("CSV int shape: " + t2.shape().__str__())

# -----------------------------------------------------------------------------
# JSON demos
# -----------------------------------------------------------------------------
fn demo_json_f64() -> None:
    var tbl = _mk_tensor_f64_3x3()
    # pretty=True for human-readable JSON
    var ok = tensor.write_json_f64(tbl, "demo_f64.json", True)
    if not ok:
        print("write_json_f64 failed")
        return
    var t2 = tensor.read_json_f64("demo_f64.json")
    print("JSON f64 shape: " + t2.shape().__str__())

fn demo_json_f32() -> None:
    var tbl = _mk_tensor_f32_2x4()
    var ok = tensor.write_json_f32(tbl, "demo_f32.json", True)
    if not ok:
        print("write_json_f32 failed")
        return
    var t2 = tensor.read_json_f32("demo_f32.json")
    print("JSON f32 shape: " + t2.shape().__str__())

fn demo_json_int() -> None:
    var tbl = _mk_tensor_int_1d()
    var ok = tensor.write_json_int(tbl, "demo_int.json", True)
    if not ok:
        print("write_json_int failed")
        return
    var t2 = tensor.read_json_int("demo_int.json")
    print("JSON int shape: " + t2.shape().__str__())

# -----------------------------------------------------------------------------
# XML demos
# -----------------------------------------------------------------------------
fn demo_xml_f64() -> None:
    var tbl = _mk_tensor_f64_3x3()
    # Tag names: root="tensor", row="row", value="v"
    var ok = tensor.write_xml_f64(tbl, "demo_f64.xml", "tensor", "row", "v")
    if not ok:
        print("write_xml_f64 failed")
        return
    var t2 = tensor.read_xml_f64("demo_f64.xml", "row", "v")
    print("XML f64 shape: " + t2.shape().__str__())

fn demo_xml_f32() -> None:
    var tbl = _mk_tensor_f32_2x4()
    var ok = tensor.write_xml_f32(tbl, "demo_f32.xml", "tensor", "row", "v")
    if not ok:
        print("write_xml_f32 failed")
        return
    var t2 = tensor.read_xml_f32("demo_f32.xml", "row", "v")
    print("XML f32 shape: " + t2.shape().__str__())

fn demo_xml_int() -> None:
    var tbl = _mk_tensor_int_1d()
    var ok = tensor.write_xml_int(tbl, "demo_int.xml", "tensor", "row", "v")
    if not ok:
        print("write_xml_int failed")
        return
    var t2 = tensor.read_xml_int("demo_int.xml", "row", "v")
    print("XML int shape: " + t2.shape().__str__())

# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------
fn main() -> None:
    print("--- CSV demos ---")
    demo_csv_f64()
    demo_csv_f32()
    demo_csv_int()

    print("--- JSON demos ---")
    demo_json_f64()
    demo_json_f32()
    demo_json_int()

    print("--- XML demos ---")
    demo_xml_f64()
    demo_xml_f32()
    demo_xml_int()
