# Project:      Momijo
# Module:       src.momijo.io.serialization.tensor_reader
# File:         tensor_reader.mojo
# Path:         src/momijo/io/serialization/tensor_reader.mojo
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
#   - Key functions: read_tensor, read_tensors, _self_test, main


from momijo.io.checkpoints.mnp import load_mnp
from momijo.io.formats.npy import load_npy
from momijo.io.formats.npy import save_npy
from momijo.io.formats.npz import load_npz
from momijo.io.formats.npz import save_npz
from momijo.io.serialization.safetensors_like import load_safetensors
from momijo.io.serialization.safetensors_like import save_safetensors
from momijo.tensor.tensor import Tensor
import os

fn read_tensor(path: String) -> Tensor:
    if path.endswith(".npy"):
        return load_npy(path)
    elif path.endswith(".safetensors"):
        var d = load_safetensors(path)
        # Return first tensor
        for (k,v) in d.items():
            return v
        raise ValueError("No tensors found in safetensors file: " + path)
    elif path.endswith(".mnp"):
        var d = load_mnp(path)
        for (k,v) in d.items():
            return v
        raise ValueError("No tensors found in mnp file: " + path)
    else:
        raise NotImplementedError("Unsupported tensor file format: " + path)

# -----------------------------------------------------------------------------
# Read multiple tensors from file (npz, safetensors, mnp)
# -----------------------------------------------------------------------------
fn read_tensors(path: String) -> Dict[String, Tensor]:
    if path.endswith(".npz"):
        return load_npz(path)
    elif path.endswith(".safetensors"):
        return load_safetensors(path)
    elif path.endswith(".mnp"):
        return load_mnp(path)
    else:
        raise NotImplementedError("Unsupported multi-tensor file format: " + path)

# -----------------------------------------------------------------------------
# Minimal self-test
# -----------------------------------------------------------------------------
fn _self_test() -> Bool:
    var ok = True

    # Test .npy
    var t = Tensor.ones([2,2], dtype="f32")
    var tmp_npy = "tmp_reader.npy"
    save_npy(tmp_npy, t)
    var t2 = read_tensor(tmp_npy)
    ok = ok and t2.shape_as_list() == [2,2]
    os.remove(tmp_npy)

    # Test .npz
    var tmp_npz = "tmp_reader.npz"
    var d = Dict[String, Tensor]()
    d["a"] = Tensor.ones([3], dtype="f32")
    save_npz(tmp_npz, d)
    var d2 = read_tensors(tmp_npz)
    ok = ok and d2.contains("a")
    os.remove(tmp_npz)

    # Test .safetensors
    var tmp_safe = "tmp_reader.safetensors"
    var d3 = Dict[String, Tensor]()
    d3["b"] = Tensor.zeros([4], dtype="f32")
    save_safetensors(tmp_safe, d3)
    var t3 = read_tensor(tmp_safe)
    ok = ok and t3.shape_as_list() == [4]
    os.remove(tmp_safe)

    return ok
fn main() -> None:
    if _self_test():
        print("Tensor reader module self-test: OK")
    else:
        print("Tensor reader module self-test: FAIL")