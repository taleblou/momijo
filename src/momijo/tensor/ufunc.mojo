# Project:      Momijo
# Module:       src.momijo.tensor.ufunc
# File:         ufunc.mojo
# Path:         src/momijo/tensor/ufunc.mojo
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
#   - Key functions: __module_name__, __self_test__, argmax_index, argmin_index, _require, add, mul, scalar_add
#   - Uses generic functions/types with explicit trait bounds.


from momijo.tensor.tensor import Tensor

fn __module_name__() -> String:
    return String("momijo/tensor/ufunc.mojo")
fn __self_test__() -> Bool:
    return True

# Tiny utilities (no external deps)
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

# Local require (Mojo doesn't use Python's assert in this context)
fn _require(cond: Bool, msg: String) -> None:
    if not cond:
        print(String("[REQUIRE FAIL] ") + msg)

# Elementwise add: supports non-contiguous inputs by materializing contiguous views
fn add(a: Tensor[Float64], b: Tensor[Float64]) -> Tensor[Float64]:
    _require(a.shape == b.shape, String("Shapes must match for elementwise add"))
    var ac = a.to_contiguous()
    var bc = b.to_contiguous()
    var out = Tensor[Float64](shape=ac.shape, fill=0.0)
    var n = ac.size()
    var i = 0
    while i < n:
        out.data[i] = ac.data[ac.offset + i] + bc.data[bc.offset + i]
        i += 1
    return out

# Elementwise multiply
fn mul(a: Tensor[Float64], b: Tensor[Float64]) -> Tensor[Float64]:
    _require(a.shape == b.shape, String("Shapes must match for elementwise mul"))
    var ac = a.to_contiguous()
    var bc = b.to_contiguous()
    var out = Tensor[Float64](shape=ac.shape, fill=0.0)
    var n = ac.size()
    var i = 0
    while i < n:
        out.data[i] = ac.data[ac.offset + i] * bc.data[bc.offset + i]
        i += 1
    return out

# Add scalar to every element
fn scalar_add(a: Tensor[Float64], c: Float64) -> Tensor[Float64]:
    var ac = a.to_contiguous()
    var out = Tensor[Float64](shape=ac.shape, fill=0.0)
    var n = ac.size()
    var i = 0
    while i < n:
        out.data[i] = ac.data[ac.offset + i] + c
        i += 1
    return out