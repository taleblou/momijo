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
# File: src/momijo/tensor/conversion.mojo

from momijo.tensor.tensor_base import Tensor
from momijo.tensor.dtype import DType, int32, float64 
 
from momijo.tensor.tensor import Tensor  # adjust if your Tensor lives elsewhere
 

# Public API (stub): keep signature simple so it compiles today.
# Later you can generalize to support multiple dtypes and real casting.
fn astype(
    x: Tensor[Float32],
    dtype_code: Int  # placeholder (e.g., 0=f32,1=f64,2=i32,...) — evolve to your DType later
) -> Tensor[Float32]:
    # TODO: when Tensor exposes data+dtype, return a new Tensor with casted data.
    # For now, pass-through so tests that only import this symbol compile and run.
    return x


fn astype[T: Copyable & Movable, U: Copyable & Movable](t: Tensor[T], dtype: DType) -> Tensor[U]:
    var out = Tensor[U](t.shape(), U(0), dtype)
    var n = t.size()
    var i = 0
    while i < n:
        out.set_flat(i, U(t.get_flat(i)))
        i += 1
    return out

fn to_list_1d[T: Copyable & Movable](t: Tensor[T]) -> List[T]:
    assert t.ndim() == 1, "to_list_1d: requires 1D tensor"
    var out = List[T]()
    var n = t.shape()[0]
    var i = 0
    while i < n:
        out.append(t.get([i]))
        i += 1
    return out

fn to_list_2d[T: Copyable & Movable](t: Tensor[T]) -> List[List[T]]:
    assert t.ndim() == 2, "to_list_2d: requires 2D tensor"
    var out = List[List[T]]()
    var R = t.shape()[0]
    var C = t.shape()[1]
    var r = 0
    while r < R:
        var row = List[T]()
        var c = 0
        while c < C:
            row.append(t.get([r, c]))
            c += 1
        out.append(row)
        r += 1
    return out

fn from_list_1d[T: Copyable & Movable](xs: List[T]) -> Tensor[T]:
    var out = Tensor[T]([len(xs)], xs[0], DType(0,0,"generic"))
    var i = 0
    while i < len(xs):
        out.set([i], xs[i])
        i += 1
    return out

fn from_list_2d[T: Copyable & Movable](xs: List[List[T]]) -> Tensor[T]:
    var R = len(xs)
    assert R > 0, "from_list_2d: empty outer list"
    var C = len(xs[0])
    var out = Tensor[T]([R, C], xs[0][0], DType(0,0,"generic"))
    var r = 0
    while r < R:
        assert len(xs[r]) == C, "from_list_2d: inconsistent inner list length"
        var c = 0
        while c < C:
            out.set([r, c], xs[r][c])
            c += 1
        r += 1
    return out
