# Project:      Momijo
# Module:       src.momijo.io.serialization.safetensors_like
# File:         safetensors_like.mojo
# Path:         src/momijo/io/serialization/safetensors_like.mojo
#
# Description:  Filesystem/IO helpers with Path-centric APIs and safe resource
#               management (binary/text modes and encoding clarity).
#
# Author(s):    Morteza Taleblou & Mitra Daneshmand
# Website:      https://taleblou.ir/
# Repository:   https://github.com/taleblou/momijo
#
# License:      MIT License
# SPDX-License-Identifier: MIT
# Copyright:    (c) 2025 Morteza Taleblou & Mitra Daneshmand
#
# Notes:
#   - Key functions: save_safetensors, load_safetensors, _self_test, main
#   - Performs file/Path IO; prefer context-managed patterns.


from momijo.tensor.tensor import Tensor
import json
import os
import struct

fn save_safetensors(path: String, tensors: Dict[String, Tensor]) -> None:
    var header: Dict[String, Any] = {}
    var offset: Int = 0
    var data_blobs = List[Bytes]()

    for (name, t) in tensors.items():
        var raw = t.to_bytes()
        var entry = {
            "dtype": t.dtype_name(),
            "shape": t.shape_as_list(),
            "offset": [offset, offset + len(raw)]
        }
        header[name] = entry
        data_blobs.append(raw)
        offset += len(raw)

    var header_json = json.dumps(header)
    var header_bytes = header_json.encode("utf-8")
    var header_len = len(header_bytes)

    var f = open(path, "wb")
    f.write(struct.pack("<Q", header_len))
    f.write(header_bytes)
    for blob in data_blobs:
        f.write(blob)
    f.close()

# -----------------------------------------------------------------------------
# Load tensors from safetensors-like format
# -----------------------------------------------------------------------------
fn load_safetensors(path: String) -> Dict[String, Tensor]:
    if not os.path.exists(path):
        raise FileNotFoundError("Safetensors file not found: " + path)

    var f = open(path, "rb")
    var (header_len,) = struct.unpack("<Q", f.read(8))
    var header_json = f.read(header_len).decode("utf-8")
    var header = json.loads(header_json)

    var result = Dict[String, Tensor]()
    for (name, entry) in header.items():
        var off_start = entry["offset"][0]
        var off_end = entry["offset"][1]
        var length = off_end - off_start

        f.seek(8 + header_len + off_start)
        var buf = f.read(length)
        var dtype = entry["dtype"]
        var shape = entry["shape"]
        var t = Tensor.from_bytes(buf, dtype, shape)
        result[name] = t

    f.close()
    return result

# -----------------------------------------------------------------------------
# Minimal self-test
# -----------------------------------------------------------------------------
fn _self_test() -> Bool:
    var ok = True
    var tensors = Dict[String, Tensor]()
    tensors["a"] = Tensor.ones([2,2], dtype="f32")
    tensors["b"] = Tensor.zeros([3], dtype="f32")

    var tmp_file = "tmp_test_safe.safetensors"
    save_safetensors(tmp_file, tensors)
    ok = ok and os.path.exists(tmp_file)

    var loaded = load_safetensors(tmp_file)
    ok = ok and loaded.contains("a")
    ok = ok and loaded.contains("b")
    ok = ok and loaded["a"].shape_as_list() == [2,2]
    ok = ok and loaded["b"].shape_as_list() == [3]

    os.remove(tmp_file)
    return ok
fn main() -> None:
    if _self_test():
        print("Safetensors-like module self-test: OK")
    else:
        print("Safetensors-like module self-test: FAIL")