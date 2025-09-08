# Project:      Momijo
# Module:       src.momijo.tensor.ops.reduce
# File:         reduce.mojo
# Path:         src/momijo/tensor/ops/reduce.mojo
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
#   - Key functions: argmax_index, argmin_index, __module_name__, __self_test__, sum_list_f32, mean_list_f32, sum_f32, mean_f32
#   - Uses generic functions/types with explicit trait bounds.


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
    return String("momijo/tensor/ops/reduce.mojo")
fn __self_test__() -> Bool:
    # smoke test for list-based reducers
    var xs = [Float32(1.0), Float32(2.0), Float32(3.0), Float32(4.0)]
    var s = sum_list_f32(xs)
    var m = mean_list_f32(xs)
    if (s != Float32(10.0)): return False
    if (m != Float32(2.5)): return False
    return True

# ---------- list reducers (work now, no Tensor dependency) ----------
fn sum_list_f32(xs: List[Float32]) -> Float32:
    var acc: Float32 = Float32(0.0)
    var i = 0
    var n = len(xs)
    while i < n:
        acc = acc + xs[i]
        i += 1
    return acc
fn mean_list_f32(xs: List[Float32]) -> Float32:
    var n = len(xs)
    if n == 0:
        return Float32(0.0)
    return sum_list_f32(xs) / Float32(n)

# ---------- Tensor entry points (Float32) ----------
# Keep these concrete to avoid generic parsing issues.
# They are safe stubs until your Tensor APIs (shape/strides/element access) are finalized.

fn sum_f32(x: F32Tensor) -> F32Tensor:
    # TODO: replace with real reduction over Tensor once APIs are stable
    return x
fn mean_f32(x: F32Tensor) -> F32Tensor:
    # TODO: replace with real reduction over Tensor once APIs are stable
    return x