# MIT License
# Copyright (c) 2025 Morteza Talebou and Mitra Daneshmand
# Project: momijo  |  Source: https://github.com/taleblou/momijo
# This file is part of the Momijo project. See the LICENSE file at the repository root.
# Momijo 
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Morteza Talebou and Mitra Daneshmand
# Website: https://taleblou.ir/
# Repository: https://github.com/taleblou/momijo
#
# Project: momijo.io.checkpoints
# File: src/momijo/io/checkpoints/mnp.mojo

from momijo.io.checkpoints.weights import StateDict
from momijo.tensor.tensor import Tensor
import json
import struct

fn save_mnp(filename: String, state: StateDict) -> None:
    var header: Dict[String, Any] = {"params": []}
    var offset: Int = 0
    var blobs = List[Bytes]()

    for (name, tensor) in state.items():
        var data: Bytes = tensor.to_bytes()
        var entry = {
            "name": name,
            "dtype": tensor.dtype_name(),
            "shape": tensor.shape_as_list(),
            "offset": offset,
            "nbytes": len(data)
        }
        header["params"].append(entry)
        blobs.append(data)
        offset += len(data)

    var header_json = json.dumps(header)
    var header_bytes = header_json.encode("utf-8")
    var length_bytes = struct.pack("<Q", len(header_bytes))

    var f = open(filename, "wb")
    f.write(length_bytes)
    f.write(header_bytes)
    for b in blobs:
        f.write(b)
    f.close()

# Load state_dict from MNP file
fn load_mnp(filename: String) -> StateDict:
    var f = open(filename, "rb")
    var length_bytes = f.read(8)
    var (header_len,) = struct.unpack("<Q", length_bytes)

    var header_bytes = f.read(header_len)
    var header_json = header_bytes.decode("utf-8")
    var header = json.loads(header_json)

    var state = StateDict()

    for entry in header["params"]:
        var name: String = entry["name"]
        var dtype: String = entry["dtype"]
        var shape: List[Int] = entry["shape"]
        var offset: Int = entry["offset"]
        var nbytes: Int = entry["nbytes"]

        f.seek(8 + header_len + offset)
        var data = f.read(nbytes)
        var tensor = Tensor.from_bytes(data, dtype, shape)
        state[name] = tensor

    f.close()
    return state

# -----------------------------------------------------------------------------
# Minimal self-test
# -----------------------------------------------------------------------------
fn _self_test() -> Bool:
    var state = StateDict()
    state["w1"] = Tensor.ones([2,2], dtype="f32")
    state["b1"] = Tensor.zeros([2], dtype="f32")

    var filename = "tmp_test.mnp"
    save_mnp(filename, state)
    var loaded = load_mnp(filename)

    var ok = True
    ok = ok and loaded.contains("w1")
    ok = ok and loaded.contains("b1")
    ok = ok and loaded["w1"].shape_as_list() == [2,2]
    ok = ok and loaded["b1"].shape_as_list() == [2]

    return ok
fn main() -> None:
    if _self_test():
        print("MNP module self-test: OK")
    else:
        print("MNP module self-test: FAIL")