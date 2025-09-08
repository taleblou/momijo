# Project:      Momijo
# Module:       src.momijo.tensor.strides
# File:         strides.mojo
# Path:         src/momijo/tensor/strides.mojo
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
#   - Key functions: argmax_index, argmin_index, __module_name__, __self_test__, max_fn, default_strides
#   - Uses generic functions/types with explicit trait bounds.
#   - GPU/device utilities present; validate backend assumptions.


from momijo.tensor.shape import Shape

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
    return String("momijo/tensor/strides.mojo")
fn __self_test__() -> Bool:
    return True

# simple replacement for Python's max
fn max_fn(a: Int, b: Int) -> Int:
    if a > b:
        return a
    return b

# Row-major default strides; treats zero-sized dims with max(1, d) in the accumulator
fn default_strides(shape: Shape) -> List[Int]:
    var r = shape.rank()
    var s = List[Int]()
    # Pre-size with zeros
    var k = 0
    while k < r:
        s.append(0)
        k += 1

    var acc: Int = 1
    var i = r - 1
    while i >= 0:
        var d = shape.dim(i)
        s[i] = acc
        acc = acc * max_fn(1, d)
        i -= 1
    return s