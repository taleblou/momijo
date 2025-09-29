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
from momijo.tensor.tensor_base import Tensor
from momijo.tensor.strides import shape_product
 
  

struct Shape(Copyable, Movable):
    var dims: List[Int]

    fn __init__(out self):
        self.dims = List[Int]()

    fn __init__(out self, dims: List[Int]):
        self.dims = dims

    fn __copyinit__(out self, other: Self):
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

    fn set_dim(mut self, i: Int, v: Int):
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


fn reshape_infer[T: Copyable & Movable](t: Tensor[T], new_shape: List[Int]) -> Tensor[T]:
    # Infer -1 dimension
    var known = 1
    var infer_idx = -1
    var i = 0
    while i < len(new_shape):
        if new_shape[i] == -1:
            assert infer_idx == -1, "only one -1 allowed in reshape"
            infer_idx = i
        else:
            known *= new_shape[i]
        i += 1
    var total = t.size()
    if infer_idx != -1:
        assert total % known == 0, "reshape_infer not divisible"
        new_shape[infer_idx] = total // known
    assert shape_product(new_shape) == total, "reshape product mismatch"
    var out = t.copy()
    out._shape = new_shape
    return out

fn squeeze[T: Copyable & Movable](t: Tensor[T]) -> Tensor[T]:
    var new_shape = List[Int]()
    var i = 0
    while i < len(t.shape()):
        if t.shape()[i] != 1:
            new_shape.append(t.shape()[i])
        i += 1
    if len(new_shape) == 0:
        new_shape.append(1)
    var out = t.copy()
    out._shape = new_shape
    return out

fn expand_dims[T: Copyable & Movable](t: Tensor[T], axis: Int) -> Tensor[T]:
    var new_shape = t.shape()
    new_shape.insert(axis, 1)
    var out = t.copy()
    out._shape = new_shape
    return out

fn flatten[T: Copyable & Movable](t: Tensor[T]) -> Tensor[T]:
    var out = t.copy()
    out._shape = [t.size()]
    return out

fn ravel[T: Copyable & Movable](t: Tensor[T]) -> Tensor[T]:
    # Equivalent to flatten for materialized storage
    return flatten(t)
