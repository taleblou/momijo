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
# Project: momijo.kernels.common
# File: src/momijo/kernels/common/tensor_iterators.mojo

from momijo.io.checkpoints.mnp import save_mnp
from momijo.io.formats.npy import save_npy
from momijo.io.formats.npz import save_npz
from momijo.io.serialization.safetensors_like import save_safetensors
from momijo.tensor.tensor import Tensor
import os

fn write_tensor(path: String, tensor: Tensor) -> None:
    if path.endswith(".npy"):
        save_npy(path, tensor)
    elif path.endswith(".safetensors"):
        var d = Dict[String, Tensor]()
        d["tensor"] = tensor
        save_safetensors(path, d)
    elif path.endswith(".mnp"):
        var d2 = Dict[String, Tensor]()
        d2["tensor"] = tensor
        save_mnp(path, d2)
    else:
        raise NotImplementedError("Unsupported tensor file format: " + path)

# -----------------------------------------------------------------------------
# Write multiple tensors to file (npz, safetensors, mnp)
# -----------------------------------------------------------------------------
fn write_tensors(path: String, tensors: Dict[String, Tensor]) -> None:
    if path.endswith(".npz"):
        save_npz(path, tensors)
    elif path.endswith(".safetensors"):
        save_safetensors(path, tensors)
    elif path.endswith(".mnp"):
        save_mnp(path, tensors)
    else:
        raise NotImplementedError("Unsupported multi-tensor file format: " + path)

# -----------------------------------------------------------------------------
# Minimal self-test
# -----------------------------------------------------------------------------
fn _self_test() -> Bool:
    var ok = True

    # Test .npy
    var t = Tensor.ones([2,2], dtype="f32")
    var tmp_npy = "tmp_writer.npy"
    write_tensor(tmp_npy, t)
    ok = ok and os.path.exists(tmp_npy)
    os.remove(tmp_npy)

    # Test .npz
    var tmp_npz = "tmp_writer.npz"
    var d = Dict[String, Tensor]()
    d["a"] = Tensor.ones([3], dtype="f32")
    write_tensors(tmp_npz, d)
    ok = ok and os.path.exists(tmp_npz)
    os.remove(tmp_npz)

    # Test .safetensors
    var tmp_safe = "tmp_writer.safetensors"
    var d2 = Dict[String, Tensor]()
    d2["b"] = Tensor.zeros([4], dtype="f32")
    write_tensors(tmp_safe, d2)
    ok = ok and os.path.exists(tmp_safe)
    os.remove(tmp_safe)

    return ok