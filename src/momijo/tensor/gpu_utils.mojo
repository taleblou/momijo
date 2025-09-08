# Project:      Momijo
# Module:       src.momijo.tensor.gpu_utils
# File:         gpu_utils.mojo
# Path:         src/momijo/tensor/gpu_utils.mojo
#
# Description:  Core tensor/ndarray components: shapes/strides, broadcasting rules,
#               element-wise ops, and foundational kernels.
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
#   - Key functions: argmax_index, argmin_index, __module_name__, __self_test__, add_f32_gpu, matmul_f32_gpu
#   - Uses generic functions/types with explicit trait bounds.
#   - GPU/device utilities present; validate backend assumptions.


from momijo.tensor.tensor import Tensor

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

fn add_f32_gpu(a: F32Tensor, b: F32Tensor, dst: F32Tensor) -> Bool:
    # TODO: compile & launch a GPU kernel. Return True on success.
    return False
fn matmul_f32_gpu(a: F32Tensor, b: F32Tensor, dst: F32Tensor) -> Bool:
    # TODO: block/thread tiling + shared memory. Return True on success.
    return False