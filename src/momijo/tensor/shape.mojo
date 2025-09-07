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
# File: src/momijo/tensor/shape.mojo

fn __module_name__() -> String:
    return String("momijo/tensor/shape.mojo")
fn __self_test__() -> Bool:
    var s = Shape([2, 3, 4])
    if s.rank() != 3:
        return False
    if s.numel() != 24:
        return False
    Shape_set_dim(s, 1, 5)
    if s.dim(1) != 5:
        return False
    return True
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

struct Shape(Copyable, Movable):
    var dims: List[Int]
fn __init__(out self) -> None:
        self.dims = List[Int]()
fn __init__(out self, dims: List[Int]) -> None:
        self.dims = dims
fn __copyinit__(out self, other: Self) -> None:
        var tmp = List[Int]()
        var i = 0
        while i < len(other.dims):
            tmp.append(other.dims[i])
            i += 1
        self.dims = tmp
fn rank(self) -> Int:
        return len(self.dims)
fn dim(self, i: Int) -> Int:
        return self.dims[i]
fn set_dim(mut self, i: Int, v: Int) -> None:
        self.dims[i] = v
fn numel(self) -> Int:
        var total: Int = 1
        var i = 0
        var n = len(self.dims)
        while i < n:
            var d = self.dims[i]
            if d < 0:
                return -1
            total = total * d
            i += 1
        return total
fn as_list(self) -> List[Int]:
        var out = List[Int]()
        var i = 0
        while i < len(self.dims):
            out.append(self.dims[i])
            i += 1
        return out
fn with_appended(self, d: Int) -> Shape:
        var nd = List[Int]()
        var i = 0
        var n = len(self.dims)
        while i < n:
            nd.append(self.dims[i])
            i += 1
        nd.append(d)
        return Shape(nd)
fn Shape_set_dim(mut s: Shape, i: Int, v: Int) -> None:
    s.set_dim(i, v)