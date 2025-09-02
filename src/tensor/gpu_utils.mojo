# MIT License
# Copyright (c) 2025 Morteza Talebou and Mitra Daneshmand
# Project: momijo  |  Source: https://github.com/taleblou/momijo
# This file is part of the Momijo project. See the LICENSE file at the repository root.
# SPDX-License-Identifier: MIT
#
# Project: momijo.tensor
# File: momijo/tensor/gpu_utils.mojo

from momijo.tensor.tensor import Tensor

# ---------- small helpers ----------

fn argmax_index(xs: List[Float64]) -> Int:
    if len(xs) == 0:
        return -1
    var best = xs[0]
    var idx = 0
    var i = 1
    while i < len(xs):
        if xs[i] > best:
            best = xs[i]
            idx = i
        i += 1
    return idx

fn argmin_index(xs: List[Float64]) -> Int:
    if len(xs) == 0:
        return -1
    var best = xs[0]
    var idx = 0
    var i = 1
    while i < len(xs):
        if xs[i] < best:
            best = xs[i]
            idx = i
        i += 1
    return idx

fn ensure_not_empty[T: Copyable & Movable](xs: List[T]) -> Bool:
    return len(xs) > 0

fn __module_name__() -> String:
    return String("momijo/tensor/gpu_utils.mojo")

fn __self_test__() -> Bool:
    return True

# ---------- GPU stubs (Float32 entry points) ----------
alias F32Tensor = Tensor[Float32]

fn add_f32_gpu(a: F32Tensor, b: F32Tensor, dst: F32Tensor) -> Bool:
    # TODO: compile & launch a GPU kernel. Return True on success.
    return False

fn matmul_f32_gpu(a: F32Tensor, b: F32Tensor, dst: F32Tensor) -> Bool:
    # TODO: block/thread tiling + shared memory. Return True on success.
    return False
