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
# Project: momijo.tensor
# File: src/momijo/tensor/reduce.mojo

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
    return String("momijo/tensor/reduce.mojo")
fn __self_test__() -> Bool:
    # This is a cheap smoke-test hook; extend with real checks as needed.
    return True

fn sum1d(a: Tensor[Float64]) -> Float64:
    assert(len(a.shape) == 1, "sum1d expects rank-1 tensor")
    var s: Float64 = 0.0
    var n = a.size()
    for i in range(0, n):
        s += a.data[a.offset + i]
    return s
fn mean1d(a: Tensor[Float64]) -> Float64:
    return sum1d(a) / Float64(a.size())