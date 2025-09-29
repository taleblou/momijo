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
# File: src/momijo/tensor/strides.mojo

 
 
from momijo.tensor.shape import rank  # chosen by proximity
 
from momijo.tensor.shape import dim
from momijo.tensor.shape import Shape

 
 
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

fn shape_product(shape: List[Int]) -> Int:
    var prod = 1
    var i = 0
    while i < len(shape):
        prod *= shape[i]
        i += 1
    return prod

fn compute_strides_rowmajor(shape: List[Int]) -> List[Int]:
    var n = len(shape)
    var strides = List[Int]()
    while len(strides) < n:
        strides.append(0)
    var acc = 1
    var i = n - 1
    while i >= 0:
        strides[i] = acc
        acc *= shape[i]
        i -= 1
    return strides

fn compute_strides_colmajor(shape: List[Int]) -> List[Int]:
    var n = len(shape)
    var strides = List[Int]()
    while len(strides) < n:
        strides.append(0)
    var acc = 1
    var i = 0
    while i < n:
        strides[i] = acc
        acc *= shape[i]
        i += 1
    return strides

fn is_contiguous_rowmajor(shape: List[Int], strides: List[Int]) -> Bool:
    assert len(shape) == len(strides)
    var expected = compute_strides_rowmajor(shape)
    var i = 0
    while i < len(shape):
        if strides[i] != expected[i]:
            return False
        i += 1
    return True

fn is_contiguous_colmajor(shape: List[Int], strides: List[Int]) -> Bool:
    assert len(shape) == len(strides)
    var expected = compute_strides_colmajor(shape)
    var i = 0
    while i < len(shape):
        if strides[i] != expected[i]:
            return False
        i += 1
    return True

fn index_to_offset(idx: List[Int], shape: List[Int], strides: List[Int], offset: Int) -> Int:
    assert len(idx) == len(shape)
    var off = offset
    var i = 0
    while i < len(shape):
        var ii = idx[i]
        assert ii >= 0 and ii < shape[i]
        off += ii * strides[i]
        i += 1
    return off

fn normalize_index(idx: Int, dim: Int) -> Int:
    assert dim > 0
    if idx < 0:
        return dim + idx
    return idx
